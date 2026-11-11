"""Translate the boxes a model writes into pixels of the picture it was shown, and from there to the original.

The app tells the model "image size shown: WxH px" and expects boxes (x, y, w, h) in those pixels. Models do not
always comply: some answer on a 0 to 1000 grid (Qwen3-VL, Gemini), some as fractions 0 to 1, and a server may shrink
the picture before the model sees it. A box in the wrong space lands in the wrong place however well the model
found the object. `resolve_space` works out which space a reply used, and `to_shown_px` converts it.

Spaces: "pixels" (the picture as shown), "thousand" (0..1000 across the width and height), "fraction" (0..1) and
"side:N" (pixels of a copy whose longest side is N, because the server shrank or enlarged the picture before the model
saw it).
"""
from __future__ import annotations

import math
from typing import Sequence

SPACES = ("auto", "pixels", "thousand", "fraction")
CONCRETE = ("pixels", "thousand", "fraction")
MODEL_SIDES = (336, 448, 512, 560, 672, 768, 896, 1024, 1280, 1344, 1536, 2048)  # longest sides a server may shrink a picture to
GOOD_ENOUGH = 0.02  # pixels already put the lasers within this fraction of the long side of a spot: leave them
SWITCH_RATIO = 0.5  # a space must put the lasers at most this fraction as far from the detected spots as "pixels" does
NEAR_FRACTION = 0.06  # ... and within this fraction of the picture's long side


def to_shown_px(box: Sequence[float], space: str, shown_w: int, shown_h: int) -> tuple[float, float, float, float]:
    """One box from `space` into pixels of the shown picture."""
    x, y, w, h = (float(v) for v in box)
    if space == "thousand":
        return (x * shown_w / 1000.0, y * shown_h / 1000.0, w * shown_w / 1000.0, h * shown_h / 1000.0)
    if space == "fraction":
        return (x * shown_w, y * shown_h, w * shown_w, h * shown_h)
    if space.startswith("side:"):
        k = max(shown_w, shown_h) / float(space[5:])
        return (x * k, y * k, w * k, h * k)
    return (x, y, w, h)


def _fits_one_space(boxes: Sequence[Sequence[float]], limit: float) -> bool:
    return all(v >= -0.01 and v <= limit for b in boxes for v in b[:4]) and all(
        b[0] + b[2] <= limit * 1.02 and b[1] + b[3] <= limit * 1.02 for b in boxes)


def guess_by_range(boxes: Sequence[Sequence[float]], shown_w: int, shown_h: int) -> str:
    """With nothing to compare against, look at the numbers: all within 0..1 is a fraction; numbers past the picture's
    own size but within 1000 are on a 0..1000 grid; anything else is pixels."""
    if not boxes:
        return "pixels"
    if _fits_one_space(boxes, 1.0) and max(shown_w, shown_h) > 8:
        return "fraction"
    biggest = max(max(b[0] + b[2], b[1] + b[3]) for b in boxes)
    if biggest > max(shown_w, shown_h) * 1.02 and _fits_one_space(boxes, 1000.0):
        return "thousand"
    return "pixels"


def _distance_to_spots(boxes, spots, space: str, shown_w: int, shown_h: int) -> float:
    """Mean distance (shown px) from each laser box's centre to the nearest detected spot's centre."""
    total = 0.0
    for box in boxes:
        x, y, w, h = to_shown_px(box, space, shown_w, shown_h)
        cx, cy = x + w / 2.0, y + h / 2.0
        total += min(math.hypot(cx - sx, cy - sy) for sx, sy in spots)
    return total / len(boxes)


def resolve_space(laser_boxes: Sequence[Sequence[float]], other_boxes: Sequence[Sequence[float]],
                  spots: Sequence[tuple[float, float]], shown_w: int, shown_h: int,
                  setting: str = "auto") -> tuple[str, str]:
    """(space, why). A fixed `setting` is used as given. In "auto" the model's laser boxes are compared with the spots
    the app found in the pixels (centres, in shown px): a space that puts them clearly closer wins; with no spots to
    compare, the numbers' range decides."""
    if setting in CONCRETE:
        return setting, f"set in AI settings to {setting}"
    everything = [*laser_boxes, *other_boxes]
    if not everything:
        return "pixels", ""
    if laser_boxes and spots:
        long_side = max(shown_w, shown_h)
        others = ["thousand", "fraction", *(f"side:{n}" for n in MODEL_SIDES if n != long_side)]
        scores = {s: _distance_to_spots(laser_boxes, spots, s, shown_w, shown_h) for s in ["pixels", *others]}
        reach = NEAR_FRACTION * long_side
        if scores["pixels"] <= GOOD_ENOUGH * long_side:
            return "pixels", ""
        best = min(others, key=lambda s: scores[s])
        if scores[best] <= reach and scores[best] <= SWITCH_RATIO * scores["pixels"]:
            return best, (f"the model's laser boxes sit {scores[best]:.0f} px from the detected spots read as {best}, "
                          f"but {scores['pixels']:.0f} px away read as pixels")
        if scores["pixels"] <= reach:
            return "pixels", ""
    guessed = guess_by_range(everything, shown_w, shown_h)
    return guessed, ("the box numbers are all between 0 and 1" if guessed == "fraction" else
                     "the box numbers go past the picture's size but stay within 1000" if guessed == "thousand" else "")
