"""Keep box positions on the picture a viewer shows.

Roboflow, browsers and phones show a photo turned upright by its EXIF orientation, and Roboflow expects boxes on that
upright picture. Older versions of this app measured on the picture as stored (not turned), so for a photo with an
orientation of 2 to 8 their boxes sit in the wrong place once uploaded (a quarter turn, a mirror, a half turn).
New label sets are made on the upright picture (`pixel_space == "upright"`). `to_upright` moves an older one over,
exactly, from the orientation number alone.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.camera import exif_orientation
from core.models import ImageLabelSet


def point_to_upright(x: float, y: float, orientation: int, raw_w: float, raw_h: float) -> tuple[float, float]:
    """A point on the picture as stored (raw_w x raw_h) -> the same spot on the upright picture."""
    return {
        1: (x, y),
        2: (raw_w - x, y),
        3: (raw_w - x, raw_h - y),
        4: (x, raw_h - y),
        5: (y, x),
        6: (raw_h - y, x),
        7: (raw_h - y, raw_w - x),
        8: (y, raw_w - x),
    }.get(orientation, (x, y))


def box_to_upright(box, orientation: int, raw_w: float, raw_h: float) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1, y1 = point_to_upright(x, y, orientation, raw_w, raw_h)
    x2, y2 = point_to_upright(x + w, y + h, orientation, raw_w, raw_h)
    left, top = min(x1, x2), min(y1, y2)
    return (int(round(left)), int(round(top)), max(1, int(round(abs(x2 - x1)))), max(1, int(round(abs(y2 - y1)))))


def to_upright(ls: ImageLabelSet, image_path: str | Path | None) -> bool:
    """Move an older label set onto the upright picture, in place. Returns True when anything moved."""
    if ls.pixel_space == "upright":
        return False
    ls.pixel_space = "upright"
    orientation = 1
    if image_path and Path(image_path).is_file():
        try:
            with Image.open(image_path) as pil:
                orientation = exif_orientation(pil)
        except Exception:
            return False
    if orientation == 1:
        return False
    raw_w, raw_h = ls.image_width, ls.image_height
    if not (raw_w and raw_h):
        return False
    for laser in ls.laser_boxes:
        laser.box = box_to_upright(laser.box, orientation, raw_w, raw_h)
        if "center_x" in laser.metrics and "center_y" in laser.metrics:
            laser.metrics["center_x"], laser.metrics["center_y"] = point_to_upright(
                laser.metrics["center_x"], laser.metrics["center_y"], orientation, raw_w, raw_h)
    for obj in ls.object_boxes:
        obj.box = box_to_upright(obj.box, orientation, raw_w, raw_h)
    if orientation >= 5:
        ls.image_width, ls.image_height = raw_h, raw_w
    ls.notes = (ls.notes + f" Boxes moved onto the upright picture (the photo's EXIF orientation is {orientation}).").strip()
    return True
