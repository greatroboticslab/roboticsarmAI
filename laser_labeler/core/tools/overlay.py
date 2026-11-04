"""Draw labelled boxes on a copy of an image (for the AI, the review UI and the PDF)."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

PALETTE = ((255, 64, 64), (64, 200, 255), (255, 200, 0), (64, 255, 128), (200, 100, 255), (255, 128, 0))


def draw_boxes(rgb: np.ndarray, items: Sequence[tuple[str, Sequence[float]]], scale: float = 1.0) -> np.ndarray:
    """items = (label, (x, y, w, h)) in source pixels; scale maps them onto rgb."""
    out = np.ascontiguousarray(rgb.copy())
    for index, (label, box) in enumerate(items):
        x, y, w, h = (int(round(v * scale)) for v in box)
        color = PALETTE[index % len(PALETTE)]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, label, (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out
