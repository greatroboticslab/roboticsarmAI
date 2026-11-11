"""A laser box that is nowhere near a laser spot is moved onto the nearest spot the app found.

The app measures spots from the pixels, so it knows where the bright places are. In agent mode the model writes the
laser boxes itself, and one that sits far from every detected spot was placed wrongly (wrong number scale, a resized
picture, a guess). Such a box is moved onto the nearest spot that no other laser claimed, and the label is flagged so
a person checks it. Pipeline mode takes its boxes from the spots, so nothing there is ever moved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Snap:
    laser_id: str
    moved: bool  # False when no free spot was left: the box stays, but is still flagged
    distance: float  # how far the model's box centre was from the nearest spot (original px)
    candidate_id: str = ""


def _centre(box) -> tuple[float, float]:
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def is_near(box, cand_box) -> bool:
    """A laser box is near a spot when it holds the spot's centre, or its centre is within about one box size of it."""
    bx, by = _centre(box)
    cx, cy = _centre(cand_box)
    if box[0] <= cx <= box[0] + box[2] and box[1] <= cy <= box[1] + box[3]:
        return True
    reach = max(box[2], box[3], cand_box[2], cand_box[3]) + 10
    return math.hypot(bx - cx, by - cy) <= reach


def snap_lasers(raw_lasers: list[dict], candidates) -> tuple[list[dict], list[Snap]]:
    """New list of laser proposals (original px) with the misplaced ones moved, and what was done to each."""
    if not candidates:
        return raw_lasers, []
    out, snaps = [dict(item) for item in raw_lasers], []
    claimed: set[str] = set()
    for item in out:  # a box already on a spot keeps it, and that spot is not offered to another laser
        near = [c for c in candidates if is_near(item["box"], c.box)]
        if near:
            claimed.add(min(near, key=lambda c: math.dist(_centre(item["box"]), _centre(c.box))).id)
            item["_placed"] = True
    for item in out:
        if item.pop("_placed", False):
            continue
        bx, by = _centre(item["box"])
        ranked = sorted(candidates, key=lambda c: math.dist((bx, by), _centre(c.box)))
        distance = math.dist((bx, by), _centre(ranked[0].box))
        free = next((c for c in ranked if c.id not in claimed), None)
        if free is None:
            snaps.append(Snap(item["id"], False, distance))
            continue
        claimed.add(free.id)
        item["box"] = tuple(free.box)
        snaps.append(Snap(item["id"], True, distance, free.id))
    return out, snaps
