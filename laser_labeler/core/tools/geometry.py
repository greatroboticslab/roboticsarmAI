"""Small box and angle helpers shared by the vision tools. Boxes are (x, y, w, h) pixels."""
from __future__ import annotations

import math
from typing import Sequence

Box = tuple[int, int, int, int]


def clamp_box(box: Sequence[float], width: int, height: int) -> Box:
    """Clip a box to the image; always at least 1 x 1 pixel inside the image."""
    x, y, w, h = (int(round(v)) for v in box)
    x0, y0 = min(max(x, 0), width - 1), min(max(y, 0), height - 1)
    x1, y1 = min(max(x + w, x0 + 1), width), min(max(y + h, y0 + 1), height)
    return (x0, y0, x1 - x0, y1 - y0)


def expand_box(box: Sequence[float], pad: float, width: int, height: int) -> Box:
    x, y, w, h = box
    return clamp_box((x - pad, y - pad, w + 2 * pad, h + 2 * pad), width, height)


def box_center(box: Sequence[float]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def scale_box(box: Sequence[float], factor: float) -> tuple[float, float, float, float]:
    return tuple(v * factor for v in box)  # type: ignore[return-value]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def direction_deg(dx: float, dy: float) -> float:
    """Angle in degrees, 0 = right, 90 = up (image y points down, so it is flipped)."""
    return math.degrees(math.atan2(-dy, dx)) % 360.0


def point_segment_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))
