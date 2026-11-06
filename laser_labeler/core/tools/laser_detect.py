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

MIN_NOISES = 6.0  # a spot must rise this many noise levels above the background to be looked at
NEST_PEAK_RATIO = 0.35  # a dimmer region around a brighter one is the same (spread) spot when its peak is at least this share
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
    contrast: float = 0.0  # how far the brightest spot rises above the background, 0..1
    seeds: int = 0  # bright places looked at
    rejected: dict[str, int] = field(default_factory=dict)  # why seeds were not kept: too_small, too_large, too_stretched

    def explain(self) -> str:
        """Plain words on what the search found, and when it found nothing, why."""
        s = self.stats
        noise = s.noise_level if s else 0.0
        found = f"{len(self.candidates)} candidate laser spot(s), {len(self.lines)} beam-like line(s)"
        facts = f"brightest place is {self.contrast * 255:.0f} levels above the background (noise {noise * 255:.1f})"
        if self.candidates:
            return f"Image search found {found}; {facts}."
        if self.threshold == 0.0:
            return (f"Image search found no laser spot: {facts}, but a spot must rise at least "
                    f"{MIN_NOISES:.0f} noise levels ({MIN_NOISES * noise * 255:.0f} levels) above the background. "
                    "The laser may be too faint or too spread out for this image.")
        names = {"too_small": "too small", "too_large": "too large", "too_stretched": "too stretched (beam-like)",
                 "not_standing_out": "not brighter than what surrounds them"}
        why = ", ".join(f"{n} {names[k]}" for k, n in self.rejected.items() if n) or "none stood out as a compact spot"
        return f"Image search found no laser spot: {self.seeds} bright place(s) looked at, rejected: {why}; {facts}."


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


def _grow_core(v: np.ndarray, seed: tuple[int, int], bg: float, max_area: int, window: int = WINDOW):
    """Core around a seed: (ys, xs, peak) or None.

    Starts at half maximum. If the core is too big or too stretched (the spot sits on a
    bright object or beam), the level is raised until only the spot is left.
    """
    sx, sy = seed
    x0, y0 = max(sx - window, 0), max(sy - window, 0)
    win = v[y0:sy + window + 1, x0:sx + window + 1]
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


def _stands_out_locally(v, xs, ys, peak: float, noise: float) -> bool:
    """The spot must rise clearly above the pixels right around it, not just above the image's overall background.
    This rejects noise bumps on a bright object or at the bright end of a lighting gradient."""
    height, width = v.shape
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    pad = max(6, x1 - x0, y1 - y0)
    ex0, ex1, ey0, ey1 = max(x0 - pad, 0), min(x1 + pad, width), max(y0 - pad, 0), min(y1 + pad, height)
    block = v[ey0:ey1, ex0:ex1].copy()
    block[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = np.nan  # leave out the spot's own box
    ring = block[~np.isnan(block)]
    if ring.size == 0:
        return True
    return bool(peak - float(np.median(ring)) >= MIN_NOISES * noise)


def _fills_window(v, xs, ys, seed, window) -> bool:
    """A region that runs to the edge of the search window is bigger than any spot we looked for: a lit object or a
    lighting gradient. The exception is a spot the sensor has clipped to white, which can be very large."""
    sx, sy = seed
    height, width = v.shape
    touches = (xs.min() <= max(sx - window, 0) or xs.max() >= min(sx + window, width - 1)
               or ys.min() <= max(sy - window, 0) or ys.max() >= min(sy + window, height - 1))
    return bool(touches and float((v[ys, xs] >= 0.996).mean()) < 0.5)


def _merge_nested(spots):
    """One spread laser spot can show several bright lobes, each grown into its own region, one inside the other.
    A region whose centre lies inside a larger one (and is not hugely brighter than it) is the same spot: keep the
    larger region and the highest peak."""
    ordered = sorted(spots, key=lambda s: -len(s[0]))
    kept: list[list] = []
    for xs, ys, peak in ordered:
        cx, cy = float(xs.mean()), float(ys.mean())
        host = next((k for k in kept if k[0].min() <= cx <= k[0].max() and k[1].min() <= cy <= k[1].max()
                     and min(k[2], peak) >= NEST_PEAK_RATIO * max(k[2], peak)), None)
        if host is not None:
            host[2] = max(host[2], peak)
        else:
            kept.append([xs, ys, peak])
    return [(xs, ys, peak) for xs, ys, peak in kept]


def _find_spots(v, smooth, thr, bg, min_area, max_area, max_candidates, window=WINDOW, rejected=None, noise=0.0):
    spots, claimed = [], np.zeros(v.shape, bool)
    rejected = rejected if rejected is not None else {}
    for sx, sy in sorted(_seeds(smooth, thr), key=lambda s: -smooth[s[1], s[0]]):
        if claimed[sy, sx]:
            continue
        core = _grow_core(v, (sx, sy), bg, max_area, window)
        if core is None:
            continue
        ys, xs, peak = core
        claimed[ys, xs] = True
        claimed[max(sy - 2, 0):sy + 3, max(sx - 2, 0):sx + 3] = True
        if _fills_window(v, xs, ys, (sx, sy), window):
            rejected["too_large"] = rejected.get("too_large", 0) + 1  # a bright area (object, lighting), not a spot
            continue
        if len(xs) < min_area:
            rejected["too_small"] = rejected.get("too_small", 0) + 1
            continue
        if len(xs) > max_area:
            rejected["too_large"] = rejected.get("too_large", 0) + 1
            continue
        if _elongation(xs, ys) > MAX_ELONGATION:
            rejected["too_stretched"] = rejected.get("too_stretched", 0) + 1
            continue
        if noise and not _stands_out_locally(v, xs, ys, peak, noise):
            rejected["not_standing_out"] = rejected.get("not_standing_out", 0) + 1
            continue
        spots.append((xs, ys, peak))
        if len(spots) >= 4 * max_candidates:  # lobes of one spread spot are merged below, so look at more than we keep
            break
    return _merge_nested(spots)[:max_candidates]


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
    max_area_fraction: float = 0.05,
    max_candidates: int = 12,
    k_noise: float = 6.0,
    relative_threshold: float = 0.2,
    find_lines: bool = True,
) -> Detection:
    """Return laser spot candidates (brightest first) and beam line segments."""
    stats = stats or image_stats(rgb)
    v = intensity_map(rgb)
    contrast = float(np.percentile(v, 99.99)) - stats.background_level
    thr = _threshold(v, stats, k_noise, relative_threshold)
    if thr is None:
        return Detection(stats=stats, contrast=max(contrast, 0.0))
    smooth = cv2.GaussianBlur(v, (0, 0), 1.0)
    window = max(WINDOW, int(0.1 * max(v.shape)))  # a spread spot on a big image is wider than a fixed 120 px window
    rejected: dict[str, int] = {}
    seeds = len(_seeds(smooth, thr))
    spots = _find_spots(v, smooth, thr, stats.background_level, min_area,
                        int(max_area_fraction * v.size), max_candidates, window, rejected, stats.noise_level)
    candidates = [_candidate(i + 1, xs, ys, peak, v)
                  for i, (xs, ys, peak) in enumerate(sorted(spots, key=lambda s: -s[2]))]
    lines = _find_lines(v, stats, k_noise) if find_lines else []
    return Detection(candidates=candidates, lines=lines, threshold=thr, stats=stats, contrast=max(contrast, 0.0),
                     seeds=seeds, rejected=rejected)
