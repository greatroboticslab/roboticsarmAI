"""Is an image already labeled in the CURRENT labeling system? (no GUI, no network)

Roboflow reports the class names an image already carries. An image counts as labeled in the current system when it has
at least one laser-side label written the way the current scheme writes it:

    full        las_<event>                          e.g. las_first-incidence
    components  component1 | component2
    merged      <obj class>__component1/2, or a bare component1/2 for a spot that lands on no object

Object labels (obj_<object>_<material>_<color>_<incidence>) are the same in every scheme, so they alone do not prove an
image was labeled in the current system. An image that has labels but none of the current kind (an older scheme, or only
objects, or class names this app does not write) is "older": it still needs labeling. A leftover component3 (or higher)
is the sign of the first components scheme, where parts were core/halo/beam, so it marks an image as older even when
it also has component1 or component2.
"""
from __future__ import annotations

import re
from typing import Iterable

from core.schema.class_names import (ClassNameError, ComponentClass, LaserClass, MergedClass, decode_class)

NONE, CURRENT, OLDER, UNKNOWN = "none", "current", "older", "unknown"

_OLD_PART = re.compile(r"(?:^|__)component(\d+)$")


def is_old_component(name: str) -> bool:
    """A component class from the first scheme (component3 and up never exist now)."""
    m = _OLD_PART.search(name or "")
    return bool(m) and int(m.group(1)) > 2


def laser_label_fits(name: str, scheme: str) -> bool:
    """True when `name` is a laser-side label the given scheme would write."""
    try:
        decoded = decode_class(name)
    except ClassNameError:
        return False
    if scheme == "full":
        return isinstance(decoded, LaserClass)
    if scheme == "components":
        return isinstance(decoded, ComponentClass)
    if scheme == "merged":
        return isinstance(decoded, (MergedClass, ComponentClass))
    return False


def labels_state(labels: Iterable[str] | None, scheme: str) -> str:
    """UNKNOWN (Roboflow did not report labels), NONE (no labels), CURRENT, or OLDER (labeled, but not in this system)."""
    if labels is None:
        return UNKNOWN
    names = [str(n) for n in labels if n]
    if not names:
        return NONE
    if any(is_old_component(n) for n in names):
        return OLDER
    return CURRENT if any(laser_label_fits(n, scheme) for n in names) else OLDER


def scheme_example(scheme: str) -> str:
    """A class name in the style of the scheme, for tooltips."""
    return {"full": "las_first-incidence", "components": "component1",
            "merged": "obj_pen_rubber_red_first__component1"}.get(scheme, "")
