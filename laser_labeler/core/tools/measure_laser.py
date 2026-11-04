"""measure_laser: deterministic numbers for one laser box (spec section 4).

Same image + same box => same numbers. The AI never supplies these values.

Definitions (first draft, refine on real images):
- Intensity is the brightest channel, 0..1. "Raw" values are on that scale;
  "_norm" values are (value - background) / (1 - background), the share of the
  headroom above the image's own background (spec 4.1).
- Core = pixels above half of (peak - background), connected to the peak.
- scatter_ratio = ring / (ring + core) energy, so it stays inside 0..1. The spec
  wording (ring / core) is unbounded; this is the bounded form of the same idea.
- Energies subtract the image noise floor so empty background adds nothing.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from core.camera import ImageStats, intensity_map
from core.tools.geometry import clamp_box, direction_deg, point_segment_distance

SATURATED = 0.996  # intensity counted as sensor maximum
MIN_CONTRAST_NOISES = 6.0  # a spot must rise this many noise levels above background


def angle_into(center, prev_center=None, lines: Sequence = (), snap: float = 40.0) -> float | None:
    """Direction of the beam segment arriving at center, degrees (0 = right, 90 = up).

    Uses a detected beam line ending near the spot; otherwise the line from the
    previous spot. None when neither exists.
    """
    best = None
    for line in lines:
        ends = [(line.x1, line.y1), (line.x2, line.y2)]
        near, far = sorted(ends, key=lambda p: math.hypot(p[0] - center[0], p[1] - center[1]))
        dist = point_segment_distance(center, ends[0], ends[1])
        if dist <= snap and (best is None or dist < best[0]):
            best = (dist, near, far)
    if best:
        _, near, far = best
        return direction_deg(near[0] - far[0], near[1] - far[1])
    if prev_center is not None:
        return direction_deg(center[0] - prev_center[0], center[1] - prev_center[1])
    return None


def _empty(stats: ImageStats) -> dict[str, float]:
    return {"background_level": stats.background_level, "noise_level": stats.noise_level,
            "peak_intensity": 0.0, "peak_intensity_norm": 0.0, "mean_intensity": 0.0,
            "mean_intensity_norm": 0.0, "core_area_px": 0.0, "aspect_ratio": 1.0,
            "scatter_ratio": 0.0, "halo_radius_px": 0.0, "total_light": 0.0,
            "clipped_fraction": 0.0, "confidence": 0.0}


def _norm(value: float, bg: float) -> float:
    return float(max(0.0, (value - bg) / max(1.0 - bg, 1e-6)))


def _core_mask(win: np.ndarray, peak_xy: tuple[int, int], bg: float, peak: float) -> np.ndarray:
    half = bg + 0.5 * (peak - bg)
    _, labels = cv2.connectedComponents((win >= half).astype(np.uint8), connectivity=8)
    return labels == labels[peak_xy[1], peak_xy[0]]


def _aspect(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 3:
        return 1.0
    ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])) + np.eye(2) / 12.0)
    return float(math.sqrt(ev[1] / ev[0]))


def _halo_radius(excess: np.ndarray, centroid, peak_excess: float, noise: float, min_r: float):
    """Radius where the mean radial profile drops to ~background. Returns (radius, truncated)."""
    yy, xx = np.indices(excess.shape)
    r = np.hypot(xx - centroid[0], yy - centroid[1])
    bins = r.astype(int)
    sums = np.bincount(bins.ravel(), weights=excess.ravel())
    counts = np.maximum(np.bincount(bins.ravel()), 1)
    profile = sums / counts
    limit = int(min(centroid[0], centroid[1], excess.shape[1] - 1 - centroid[0],
                    excess.shape[0] - 1 - centroid[1]))
    thr = max(0.5 * noise, 0.02 * peak_excess)
    for k in range(int(math.ceil(min_r)), max(limit, int(math.ceil(min_r)) + 1)):
        if k >= len(profile) or profile[k] <= thr:
            return float(k), False
    return float(max(limit, min_r)), True


def measure_laser(
    rgb: np.ndarray,
    box: Sequence[float],
    stats: ImageStats,
    is_grayscale: bool,
    *,
    kind: str = "other",
    first_total: float | None = None,
    prev_center: tuple[float, float] | None = None,
    lines: Sequence = (),
    v: np.ndarray | None = None,
) -> dict[str, float]:
    """Return every section 4 number for the laser spot inside box (x, y, w, h)."""
    v = intensity_map(rgb) if v is None else v
    height, width = v.shape
    bg, noise = stats.background_level, stats.noise_level
    x, y, w, h = clamp_box(box, width, height)
    pad = max(20, 3 * max(w, h))
    wx, wy, ww, wh = clamp_box((x - pad, y - pad, w + 2 * pad, h + 2 * pad), width, height)
    win = v[wy:wy + wh, wx:wx + ww]
    inner = v[y:y + h, x:x + w]
    peak = float(inner.max())
    if peak - bg <= MIN_CONTRAST_NOISES * noise:
        return _empty(stats)  # nothing stands out of the noise in this box

    iy, ix = np.unravel_index(int(inner.argmax()), inner.shape)
    peak_xy = (x - wx + int(ix), y - wy + int(iy))
    core = _core_mask(win, peak_xy, bg, peak)
    ys, xs = np.nonzero(core)
    excess = np.clip(win - bg - noise, 0.0, None)
    core_energy = float(excess[core].sum())
    weights = np.maximum(win[core] - bg, 1e-6)
    centroid = (float((xs * weights).sum() / weights.sum()), float((ys * weights).sum() / weights.sum()))

    equiv_r = math.sqrt(len(xs) / math.pi)
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)
    ring = (dist > 0) & (dist <= max(4.0, 2.0 * equiv_r))
    ring_energy = float(excess[ring].sum())
    halo, truncated = _halo_radius(excess, centroid, peak - bg, noise, max(equiv_r, 1.0))
    yy, xx = np.indices(win.shape)
    disk = np.hypot(xx - centroid[0], yy - centroid[1]) <= halo
    total = float(excess[disk | core].sum())

    mean = float(win[core].mean())
    clipped = float((win[core] >= SATURATED).mean())
    center = (wx + centroid[0], wy + centroid[1])
    snr = (peak - bg) / noise
    confidence = min(1.0, snr / 20.0) * (1.0 - 0.5 * clipped) * (1.0 if len(xs) >= 3 else 0.5)

    out = _empty(stats)
    out.update({
        "peak_intensity": peak, "peak_intensity_norm": _norm(peak, bg),
        "mean_intensity": mean, "mean_intensity_norm": _norm(mean, bg),
        "core_area_px": float(len(xs)), "aspect_ratio": _aspect(xs, ys),
        "scatter_ratio": ring_energy / (ring_energy + core_energy) if core_energy > 0 else 0.0,
        "halo_radius_px": halo, "halo_truncated": 1.0 if truncated else 0.0,
        "total_light": total, "clipped_fraction": clipped, "confidence": float(confidence),
        "center_x": center[0], "center_y": center[1],
    })
    if kind == "first_incidence":
        out["reflectivity"] = 1.0
    elif first_total and first_total > 0:
        out["reflectivity"] = float(min(1.0, total / first_total))
    angle = angle_into(center, prev_center, lines)
    if angle is not None:
        out["angle_deg"] = angle
    colors = rgb[wy:wy + wh, wx:wx + ww][core].astype(np.float32)
    if is_grayscale:
        out["gray_level"] = float(colors.mean())
    else:
        out.update(zip(("color_r", "color_g", "color_b"), (float(c) for c in colors.mean(axis=0))))
    return out
