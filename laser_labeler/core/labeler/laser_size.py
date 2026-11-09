"""Make laser boxes bigger than the bare spot, so they cover the glow and a useful part of what the beam hits.

A box that hugs the bright core is hard to see and to grab in the review tab, and it says little about where on the
object the beam landed. After the object boxes exist, every plain laser box (label scheme "full") is

1. padded by GLOW_MARGIN of its own size on each side, so the glow around the core is inside it, then
2. grown, if an object box lists it, until each side is `fraction` of that object box's side (a third by default),
   but never past half of the object box's side, and never outside the image.

The box only ever grows, and it stays centred on the spot. Measurements are made before this and are not changed.
Parts (component1/2/3) are not touched: they are defined by the pixels.
"""
from __future__ import annotations

from typing import Sequence

from core.models import LaserBox, ObjectBox
from core.tools.geometry import clamp_box

GLOW_MARGIN = 0.25
MAX_OBJECT_FRACTION = 0.5


def grown_box(box: Sequence[float], object_box: Sequence[float] | None, fraction: float,
              width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w, new_h = w * (1 + 2 * GLOW_MARGIN), h * (1 + 2 * GLOW_MARGIN)
    if object_box is not None and fraction > 0:
        want = min(fraction, MAX_OBJECT_FRACTION)
        new_w, new_h = max(new_w, want * object_box[2]), max(new_h, want * object_box[3])
    return clamp_box((cx - new_w / 2.0, cy - new_h / 2.0, new_w, new_h), width, height)


def grow_laser_boxes(lasers: list[LaserBox], objects: list[ObjectBox], fraction: float,
                     width: int, height: int) -> None:
    """Resize plain laser boxes in place. `fraction` 0 turns this off."""
    if fraction <= 0:
        return
    for laser in lasers:
        hit = next((o for o in objects if laser.id in o.laser_ids), None)
        laser.box = grown_box(laser.box, hit.box if hit else None, fraction, width, height)
