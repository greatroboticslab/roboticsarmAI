"""crop_zoom and image helpers for what the AI is shown (spec 5.1, section 14)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from core.tools.geometry import Box, clamp_box, expand_box


@dataclass
class Crop:
    image: np.ndarray  # RGB uint8
    region: Box  # where it came from, in source pixels
    scale: float  # output pixels per source pixel


def crop_zoom(rgb: np.ndarray, box: Sequence[float], *, pad_fraction: float = 0.5,
              min_side: int = 48, target: int = 384) -> Crop:
    """Crop around box with context, enlarged so its longest side is about `target` px."""
    height, width = rgb.shape[:2]
    x, y, w, h = clamp_box(box, width, height)
    pad = int(max(pad_fraction * max(w, h), (min_side - max(w, h)) / 2, 4))
    region = expand_box((x, y, w, h), pad, width, height)
    patch = rgb[region[1]:region[1] + region[3], region[0]:region[0] + region[2]]
    scale = target / max(region[2], region[3])
    size = (max(1, round(region[2] * scale)), max(1, round(region[3] * scale)))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return Crop(cv2.resize(patch, size, interpolation=interp), region, scale)


def downscale_to_max(rgb: np.ndarray, max_px: int) -> tuple[np.ndarray, float]:
    """Shrink so the longest side is at most max_px. Returns (image, shown / original)."""
    longest = max(rgb.shape[:2])
    if max_px <= 0 or longest <= max_px:
        return rgb, 1.0
    factor = max_px / longest
    size = (round(rgb.shape[1] * factor), round(rgb.shape[0] * factor))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), factor


def encode_png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("could not encode image")
    return buf.tobytes()
