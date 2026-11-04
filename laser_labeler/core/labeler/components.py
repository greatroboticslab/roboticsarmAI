"""Label scheme "components": split each laser spot into its parts (component1, component2, component3).

    component1 = the bright core of the spot (the box the labeler already finds and measures)
    component2 = the halo or glow around the core
    component3 = the visible beam line leading to the spot

The app builds these from the pixels, so the model's job does not change: it still decides which candidates
are real laser spots and what object each one hits. A part that is not visible is simply not made.
Each halo or beam box points at its core with `from_id` and carries the core's confidence, so the queue's
"weakest box" figure is not dragged down by a part the app could not measure on its own.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from core.models import LaserBox
from core.tools.geometry import Box, clamp_box, direction_deg, point_segment_distance

SCHEMES = ("full", "components")
HALO_EXTRA_PX = 3.0  # the halo must reach this far beyond the core's own half-size to count as a separate part
BEAM_SNAP_PX = 40.0  # a beam line counts as leading to a spot when it ends within this distance of it
MIN_PART_PX = 6  # smallest side of a halo or beam box
RIDGE_SIDE_PX = 5  # a beam must be brighter than the pixels this far to both sides of it
RIDGE_NOISES = 3.0  # ...by this many noise levels (the edge of a bright object is a step, not a ridge)


def _free_id(core_id: str, suffix: str, taken: set[str]) -> str:
    candidate = f"{core_id}{suffix}"
    n = 2
    while candidate in taken:
        candidate = f"{core_id}{suffix}{n}"
        n += 1
    taken.add(candidate)
    return candidate


def halo_box(core: LaserBox, width: int, height: int) -> Box | None:
    """A square around the core as wide as its measured halo, or None when there is no halo beyond the core."""
    m = core.metrics
    radius = m.get("halo_radius_px", 0.0)
    if "center_x" not in m or radius <= 0:
        return None
    if radius < max(core.box[2], core.box[3]) / 2.0 + HALO_EXTRA_PX:
        return None
    side = max(int(round(2 * radius)), MIN_PART_PX)
    return clamp_box((m["center_x"] - side / 2.0, m["center_y"] - side / 2.0, side, side), width, height)


def _line_box(line, width: int, height: int) -> Box:
    x0, x1 = sorted((line.x1, line.x2))
    y0, y1 = sorted((line.y1, line.y2))
    w, h = max(x1 - x0, MIN_PART_PX), max(y1 - y0, MIN_PART_PX)  # a straight line has no width: give it some
    return clamp_box((x0 - 1, y0 - 1, w + 2, h + 2), width, height)


def _nearest_core(line, cores: Sequence[LaserBox]) -> tuple[LaserBox, float] | None:
    best = None
    for core in cores:
        if "center_x" not in core.metrics:
            continue
        center = (core.metrics["center_x"], core.metrics["center_y"])
        d = point_segment_distance(center, (line.x1, line.y1), (line.x2, line.y2))
        if d <= BEAM_SNAP_PX and (best is None or d < best[1]):
            best = (core, d)
    return best


def is_ridge(line, v: np.ndarray, noise: float, samples: int = 40,
             exclude: tuple[float, float, float] | None = None) -> bool:
    """True when the line follows a bright ridge. A real beam is brighter than both of its sides; the edge of a
    bright object (which the line search also picks up) has one bright side, so it is not a beam.

    `exclude` = (x, y, radius) leaves out the part of the line inside a spot's own glow: a line through any spot
    looks like a ridge there, because the spot itself is bright."""
    height, width = v.shape
    dx, dy = line.x2 - line.x1, line.y2 - line.y1
    length = math.hypot(dx, dy)
    if length < 1:
        return False
    nx, ny = -dy / length, dx / length
    t = np.linspace(0.0, 1.0, samples)
    px, py = line.x1 + t * dx, line.y1 + t * dy
    if exclude is not None:
        far = np.hypot(px - exclude[0], py - exclude[1]) > exclude[2]
        px, py = px[far], py[far]
    if len(px) < 5:
        return False  # nothing of the line lies outside the spot's glow

    def mean_at(offset: float) -> float:
        xs = np.clip(np.round(px + nx * offset).astype(int), 0, width - 1)
        ys = np.clip(np.round(py + ny * offset).astype(int), 0, height - 1)
        return float(v[ys, xs].mean())

    on_line = max(mean_at(d) for d in (-1.0, 0.0, 1.0))  # tolerate a line a pixel off the true ridge
    sides = max(mean_at(RIDGE_SIDE_PX), mean_at(-RIDGE_SIDE_PX))
    return on_line - sides > RIDGE_NOISES * noise


def _beam_metrics(line, core: LaserBox) -> dict[str, float]:
    center = (core.metrics["center_x"], core.metrics["center_y"])
    near, far = sorted(((line.x1, line.y1), (line.x2, line.y2)),
                       key=lambda p: math.hypot(p[0] - center[0], p[1] - center[1]))
    return {"length_px": float(line.length), "angle_deg": direction_deg(near[0] - far[0], near[1] - far[1]),
            "confidence": core.metrics.get("confidence", 0.0)}


def apply_components(lasers: list[LaserBox], lines: Sequence, width: int, height: int, *,
                     intensity: np.ndarray | None = None, noise: float = 0.0) -> list[LaserBox]:
    """Return the laser boxes with each core marked component1 and its halo and beam parts added.

    With the image's intensity map, a line that is not a bright ridge (for example an object's edge) is not a beam.
    Boxes that already have a component are left alone, so running this twice changes nothing."""
    taken = {b.id for b in lasers}
    cores = [b for b in lasers if b.component is None and not b.is_part]
    parts: list[LaserBox] = []
    for core in cores:
        core.component = "component1"
        box = halo_box(core, width, height)
        if box is not None:
            m = core.metrics
            parts.append(LaserBox(_free_id(core.id, "h", taken), core.kind, box, core.order, core.id,
                                  {"halo_radius_px": m["halo_radius_px"], "scatter_ratio": m.get("scatter_ratio", 0.0),
                                   "total_light": m.get("total_light", 0.0), "center_x": m["center_x"],
                                   "center_y": m["center_y"], "confidence": m.get("confidence", 0.0)},
                                  "halo around " + core.id, "component2"))
    claimed: set[int] = set()
    for line in lines:  # a beam line belongs to the one spot it ends at
        hit = _nearest_core(line, cores)
        if hit is None or id(hit[0]) in claimed:
            continue
        core = hit[0]
        if intensity is not None:
            glow = max(core.metrics.get("halo_radius_px", 0.0), max(core.box[2:]) / 2.0) + 3.0
            if not is_ridge(line, intensity, noise, exclude=(core.metrics["center_x"], core.metrics["center_y"], glow)):
                continue
        claimed.add(id(hit[0]))
        parts.append(LaserBox(_free_id(core.id, "b", taken), core.kind, _line_box(line, width, height), core.order,
                              core.id, _beam_metrics(line, core), "beam leading to " + core.id, "component3"))
    return lasers + parts
