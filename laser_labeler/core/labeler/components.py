"""Label schemes "components" and "merged": each laser spot is labeled by how the light got there.

    component1 = a direct hit: the beam lands on the surface straight from the laser
                 (two separate direct hits on the same object are both component1)
    component2 = a bounce: light that was reflected, refracted or scattered off something and is still visible,
                 however many bounces it took

The class comes from the event kind the model gave the spot (first_incidence is a direct hit, every other kind is a
bounce), so every laser spot gets exactly one of the two. There are no halo or beam-line boxes any more.
"""
from __future__ import annotations

from typing import Iterable

from core.models import LaserBox

SCHEMES = ("full", "components", "merged")
DIRECT_KIND = "first_incidence"


def component_for_kind(kind: str) -> str:
    """component1 for a direct hit, component2 for anything that bounced."""
    return "component1" if kind.replace("-", "_") == DIRECT_KIND else "component2"


def apply_components(lasers: Iterable[LaserBox]) -> list[LaserBox]:
    """Give every spot its component class. The boxes and measurements are not changed."""
    out = list(lasers)
    for laser in out:
        laser.component = component_for_kind(laser.kind)
    return out
