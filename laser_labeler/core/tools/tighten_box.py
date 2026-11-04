"""tighten_box: refine a box to the edges of the same-colored region around a laser hit.

Classical method (no SAM): sample the region's color in Lab space, keep the
connected pixels close to it, take their bounding box. Falls back to the input
box when the region is unreliable (textured surface, region runs off the window).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from core.tools.geometry import Box, clamp_box
from core.tools.color import SATURATED_V

MAX_TOLERANCE = 30.0  # Lab distance; a looser match means the surface is textured
MIN_KEEP = 0.25  # region smaller than this share of the input box => distrust it


@dataclass
class TightenResult:
    box: Box
    confidence: float  # 0..1
    hit_window_edge: bool  # region reaches the search window: it may extend further
    changed: bool


def tighten_box(rgb: np.ndarray, box: Sequence[float], *, margin: float = 1.0) -> TightenResult:
    height, width = rgb.shape[:2]
    x, y, w, h = clamp_box(box, width, height)
    pad = int(max(24, margin * max(w, h)))
    wx, wy, ww, wh = clamp_box((x - pad, y - pad, w + 2 * pad, h + 2 * pad), width, height)
    window = rgb[wy:wy + wh, wx:wx + ww]
    lab = cv2.cvtColor(window.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    bright = window.max(axis=2) / 255.0 >= SATURATED_V

    ix, iy = x - wx, y - wy
    seed_lab = lab[iy:iy + h, ix:ix + w][~bright[iy:iy + h, ix:ix + w]]
    if len(seed_lab) < 5:
        return TightenResult((x, y, w, h), 0.0, False, False)
    seed = np.median(seed_lab, axis=0)
    spread = np.median(np.abs(seed_lab - seed), axis=0)
    tol = max(10.0, 3.0 * float(np.linalg.norm(spread)))
    if tol > MAX_TOLERANCE:
        return TightenResult((x, y, w, h), 0.1, False, False)  # too textured to trust edges
    region = ((np.linalg.norm(lab - seed, axis=2) < tol) | bright).astype(np.uint8)
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    _, labels = cv2.connectedComponents(region, connectivity=8)
    inside = labels[iy:iy + h, ix:ix + w]
    counts = np.bincount(inside[inside > 0].ravel(), minlength=1)
    if counts.sum() == 0:
        return TightenResult((x, y, w, h), 0.0, False, False)
    ys, xs = np.nonzero(labels == int(counts.argmax()))
    new = (wx + int(xs.min()), wy + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    edge = bool((xs.min() == 0 and wx > 0) or (ys.min() == 0 and wy > 0)
                or (xs.max() == ww - 1 and wx + ww < width) or (ys.max() == wh - 1 and wy + wh < height))
    if new[2] * new[3] < MIN_KEEP * w * h:
        return TightenResult((x, y, w, h), 0.2, edge, False)
    confidence = 0.4 if edge else 0.8
    return TightenResult(new, confidence, edge, new != (x, y, w, h))
