"""detect_laser: find laser spots and visible beam lines with plain OpenCV (spec 5.1).

Method: smooth the intensity map, find local maxima above an adaptive threshold,
grow each maximum to its half-maximum core, and keep compact cores as spots.
Long thin cores are beams, not spots. Thin faint beams are also found with a
Hough line search. Tuned later on real images (spec section 17).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from core.camera import ImageStats, image_stats, intensity_map
from core.tools.geometry import Box, point_segment_distance

SEED_RADIUS = 5  # px: a spot must be the brightest within this distance
WINDOW = 60  # px: half-size of the window used to grow a core
MAX_ELONGATION = 6.0  # cores more elongated than this are beams, not spots
CORE_FRACTIONS = (0.5, 0.65, 0.8, 0.9)  # share of (peak - background) that bounds the core


@dataclass
class LaserCandidate:
    id: str
    center: tuple[float, float]
    box: Box
    area: int
    peak: float  # 0..1
    saturated_fraction: float


@dataclass
class BeamLine:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


@dataclass
class Detection:
    candidates: list[LaserCandidate] = field(default_factory=list)
    lines: list[BeamLine] = field(default_factory=list)
    threshold: float = 0.0
    stats: ImageStats | None = None


def _threshold(v: np.ndarray, stats: ImageStats, k_noise: float, rel: float) -> float | None:
    peak = float(np.percentile(v, 99.99))
    span = peak - stats.background_level
    if span < k_noise * stats.noise_level:
        return None  # nothing stands out from the noise
    return stats.background_level + max(k_noise * stats.noise_level, rel * span)


def _seeds(smooth: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """Local maxima above thr. Flat (saturated) tops are collapsed to one seed each."""
    size = 2 * SEED_RADIUS + 1
    maxima = ((smooth >= cv2.dilate(smooth, np.ones((size, size), np.uint8))) & (smooth > thr))
    count, labels = cv2.connectedComponents(maxima.astype(np.uint8), connectivity=8)
    seeds = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        mid = len(ys) // 2
        seeds.append((int(xs[mid]), int(ys[mid])))
    return seeds


def _grow_core(v: np.ndarray, seed: tuple[int, int], bg: float, max_area: int):
    """Core around a seed: (ys, xs, peak) or None.

    Starts at half maximum. If the core is too big or too stretched (the spot sits on a
    bright object or beam), the level is raised until only the spot is left.
    """
    sx, sy = seed
    x0, y0 = max(sx - WINDOW, 0), max(sy - WINDOW, 0)
    win = v[y0:sy + WINDOW + 1, x0:sx + WINDOW + 1]
    px, py = sx - x0, sy - y0
    near = win[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3]
    peak = float(near.max())
    iy, ix = np.unravel_index(int(near.argmax()), near.shape)
    at = (max(py - 2, 0) + iy, max(px - 2, 0) + ix)  # brightest pixel next to the seed
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    best = None
    for fraction in CORE_FRACTIONS:
        mask = (win >= bg + fraction * (peak - bg)).astype(np.uint8)
        # Opening strips a thin beam attached to the spot; keep it only if the spot survives.
        for candidate in (cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse), mask):
            _, labels = cv2.connectedComponents(candidate, connectivity=8)
            if labels[at] == 0:
                continue
            ys, xs = np.nonzero(labels == labels[at])
            best = (ys + y0, xs + x0, peak)
            if len(xs) <= max_area and _elongation(xs, ys) <= MAX_ELONGATION:
                return best
            break
    return best


def _elongation(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 3:
        return 1.0
    cov = np.cov(np.vstack([xs, ys])) + np.eye(2) / 12.0
    ev = np.linalg.eigvalsh(cov)
    return float(math.sqrt(ev[1] / ev[0]))


def _find_spots(v, smooth, thr, bg, min_area, max_area, max_candidates):
    spots, claimed = [], np.zeros(v.shape, bool)
    for sx, sy in sorted(_seeds(smooth, thr), key=lambda s: -smooth[s[1], s[0]]):
        if claimed[sy, sx]:
            continue
        core = _grow_core(v, (sx, sy), bg, max_area)
        if core is None:
            continue
        ys, xs, peak = core
        claimed[ys, xs] = True
        claimed[max(sy - 2, 0):sy + 3, max(sx - 2, 0):sx + 3] = True
        if not (min_area <= len(xs) <= max_area) or _elongation(xs, ys) > MAX_ELONGATION:
            continue
        spots.append((xs, ys, peak))
        if len(spots) >= max_candidates:
            break
    return spots


def _candidate(index: int, xs, ys, peak, v) -> LaserCandidate:
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    return LaserCandidate(
        id=f"c{index}", center=(float(xs.mean()), float(ys.mean())),
        box=(x0, y0, x1 - x0, y1 - y0), area=int(len(xs)), peak=peak,
        saturated_fraction=float((v[ys, xs] >= 0.996).mean()),
    )


def _find_lines(v, stats, k_noise, max_lines=8) -> list[BeamLine]:
    span = float(np.percentile(v, 99.99)) - stats.background_level
    low = stats.background_level + max(k_noise * stats.noise_level, 0.1 * span)
    mask = cv2.morphologyEx((v > low).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    raw = cv2.HoughLinesP(mask, 1, math.pi / 180, threshold=40, minLineLength=40, maxLineGap=8)
    lines: list[BeamLine] = []
    for x1, y1, x2, y2 in (raw.reshape(-1, 4) if raw is not None else []):
        line = BeamLine(float(x1), float(y1), float(x2), float(y2))
        if not _duplicate(line, lines):
            lines.append(line)
    return sorted(lines, key=lambda ln: -ln.length)[:max_lines]


def _duplicate(line: BeamLine, kept: list[BeamLine], tol: float = 4.0) -> bool:
    return any(
        point_segment_distance((line.x1, line.y1), (k.x1, k.y1), (k.x2, k.y2)) < tol
        and point_segment_distance((line.x2, line.y2), (k.x1, k.y1), (k.x2, k.y2)) < tol
        for k in kept
    )


def detect_laser(
    rgb: np.ndarray,
    stats: ImageStats | None = None,
    *,
    min_area: int = 3,
    max_area_fraction: float = 0.02,
    max_candidates: int = 12,
    k_noise: float = 6.0,
    relative_threshold: float = 0.2,
    find_lines: bool = True,
) -> Detection:
    """Return laser spot candidates (brightest first) and beam line segments."""
    stats = stats or image_stats(rgb)
    v = intensity_map(rgb)
    thr = _threshold(v, stats, k_noise, relative_threshold)
    if thr is None:
        return Detection(stats=stats)
    smooth = cv2.GaussianBlur(v, (0, 0), 1.0)
    spots = _find_spots(v, smooth, thr, stats.background_level, min_area,
                        int(max_area_fraction * v.size), max_candidates)
    candidates = [_candidate(i + 1, xs, ys, peak, v)
                  for i, (xs, ys, peak) in enumerate(sorted(spots, key=lambda s: -s[2]))]
    lines = _find_lines(v, stats, k_noise) if find_lines else []
    return Detection(candidates=candidates, lines=lines, threshold=thr, stats=stats)
