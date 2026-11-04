"""sample_color: dominant color inside a box, from pixels, not from the AI (spec 5.1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from core.tools.geometry import clamp_box

# Reference swatches (RGB). Names missing here (for example "clear") are never
# chosen by pixels; the AI or the user decides those.
REFERENCE_RGB: dict[str, tuple[int, int, int]] = {
    "black": (25, 25, 25), "white": (240, 240, 240), "gray": (128, 128, 128),
    "red": (200, 35, 35), "orange": (235, 130, 30), "yellow": (235, 215, 50),
    "green": (45, 150, 65), "blue": (45, 75, 200), "purple": (130, 55, 160),
    "pink": (240, 150, 190), "brown": (110, 70, 40), "silver": (195, 195, 200),
    "gold": (212, 175, 55),
}
SATURATED_V = 0.95  # pixels this bright are laser glare, not surface color


@dataclass
class ColorSample:
    rgb: tuple[int, int, int]
    gray_level: float
    name: str | None  # taxonomy color name, or the placeholder on grayscale
    confidence: float  # 0..1, from the gap between best and second-best match


def _lab(rgb_rows: np.ndarray) -> np.ndarray:
    pixels = np.asarray(rgb_rows, np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).reshape(-1, 3)


def nearest_color(rgb: Sequence[float], allowed: Iterable[str]) -> tuple[str | None, float]:
    """Closest allowed reference color and a 0..1 confidence."""
    names = [n for n in allowed if n in REFERENCE_RGB]
    if not names:
        return None, 0.0
    refs = _lab(np.array([REFERENCE_RGB[n] for n in names]))
    dist = np.linalg.norm(refs - _lab(np.array([rgb]))[0], axis=1)
    order = np.argsort(dist)
    best = float(dist[order[0]])
    second = float(dist[order[1]]) if len(names) > 1 else best + 40.0
    confidence = float(np.clip((second - best) / max(second, 1e-6), 0.0, 1.0))
    confidence *= float(np.clip(1.0 - best / 60.0, 0.0, 1.0))  # far from every swatch: unsure
    return names[int(order[0])], confidence


def sample_color(
    rgb: np.ndarray,
    box: Sequence[float],
    *,
    is_grayscale: bool = False,
    allowed_colors: Iterable[str] = REFERENCE_RGB,
    placeholder: str = "na",
) -> ColorSample:
    height, width = rgb.shape[:2]
    x, y, w, h = clamp_box(box, width, height)
    pixels = rgb[y:y + h, x:x + w].reshape(-1, 3)
    keep = pixels.max(axis=1) / 255.0 < SATURATED_V
    if keep.sum() >= 5:
        pixels = pixels[keep]
    median = np.median(pixels, axis=0)
    color = (int(round(median[0])), int(round(median[1])), int(round(median[2])))
    gray = float(median.mean())
    if is_grayscale:
        return ColorSample(color, gray, placeholder, 1.0)
    name, confidence = nearest_color(color, allowed_colors)
    return ColorSample(color, gray, name, confidence)
