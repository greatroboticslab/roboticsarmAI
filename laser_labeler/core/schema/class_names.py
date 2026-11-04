"""Encode and decode Roboflow class names (build spec section 3.2).

    Object: obj_<object>-<sub-object>_<material>_<color>_<incidence>
    Laser:  las_<event>
    Parts of one laser spot (label scheme "components"): component1, component2, component3

"-" and "_" are separators, so taxonomy entries may not contain them. Entries
are restricted to lowercase letters and digits. Laser events are the one place
a "-" is allowed (first-incidence), because the event is a single field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

OBJECT_PREFIX = "obj"
LASER_PREFIX = "las"

COMPONENT_NAMES = ("component1", "component2", "component3")
COMPONENT_MEANING = {
    "component1": "the bright core of the laser spot",
    "component2": "the halo or glow around the core",
    "component3": "the visible beam line leading to the spot",
}

_TOKEN = re.compile(r"^[a-z0-9]+$")
_EVENT = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class ClassNameError(ValueError):
    """Raised for names that cannot be encoded or decoded."""


@dataclass(frozen=True)
class ObjectClass:
    general_object: str
    sub_object: str
    material: str
    color: str
    incidence: str


@dataclass(frozen=True)
class LaserClass:
    event: str


@dataclass(frozen=True)
class ComponentClass:
    number: int


def check_token(value: str, label: str) -> str:
    """Return value if it is a legal taxonomy token, else raise ClassNameError."""
    if not isinstance(value, str) or not _TOKEN.match(value):
        raise ClassNameError(
            f"{label} {value!r} must be lowercase letters/digits only "
            "(no '-', '_', spaces or uppercase; write multi-word names as one word)"
        )
    return value


def check_event(value: str) -> str:
    if not isinstance(value, str) or not _EVENT.match(value):
        raise ClassNameError(
            f"laser event {value!r} must be lowercase words joined by single '-' "
            "(for example first-incidence)"
        )
    return value


def resolve_color(color: str, is_grayscale: bool, placeholder: str = "na") -> str:
    """Grayscale images have no color: return the placeholder instead."""
    return check_token(placeholder, "color placeholder") if is_grayscale else color


def encode_object_class(
    general_object: str, sub_object: str, material: str, color: str, incidence: str
) -> str:
    parts = [
        check_token(general_object, "general object"),
        check_token(sub_object, "sub-object"),
        check_token(material, "material"),
        check_token(color, "color"),
        check_token(incidence, "incidence"),
    ]
    return f"{OBJECT_PREFIX}_{parts[0]}-{parts[1]}_{parts[2]}_{parts[3]}_{parts[4]}"


def encode_laser_class(event: str) -> str:
    return f"{LASER_PREFIX}_{check_event(event)}"


def encode_component_class(name: str) -> str:
    if name not in COMPONENT_NAMES:
        raise ClassNameError(f"component {name!r} must be one of {', '.join(COMPONENT_NAMES)}")
    return name


def decode_class(name: str) -> ObjectClass | LaserClass | ComponentClass:
    """Inverse of the encode functions. Raises ClassNameError on bad input."""
    if not isinstance(name, str) or not name:
        raise ClassNameError("class name must be a non-empty string")
    if name in COMPONENT_NAMES:
        return ComponentClass(int(name[-1]))
    prefix, sep, rest = name.partition("_")
    if not sep or not rest:
        raise ClassNameError(f"class name {name!r} has no fields after the prefix")
    if prefix == LASER_PREFIX:
        return LaserClass(event=check_event(rest))
    if prefix == OBJECT_PREFIX:
        return _decode_object(name, rest)
    raise ClassNameError(f"class name {name!r} must start with 'obj_' or 'las_', or be one of {', '.join(COMPONENT_NAMES)}")


def _decode_object(name: str, rest: str) -> ObjectClass:
    fields = rest.split("_")
    if len(fields) != 4:
        raise ClassNameError(
            f"object class {name!r} needs 4 '_'-separated fields after 'obj_' "
            f"(object-sub, material, color, incidence); found {len(fields)}"
        )
    names = fields[0].split("-")
    if len(names) != 2:
        raise ClassNameError(f"object class {name!r}: first field must be '<object>-<sub-object>'")
    return ObjectClass(
        general_object=check_token(names[0], "general object"),
        sub_object=check_token(names[1], "sub-object"),
        material=check_token(fields[1], "material"),
        color=check_token(fields[2], "color"),
        incidence=check_token(fields[3], "incidence"),
    )


def class_family(name: str) -> str:
    """Return 'object' or 'laser' for a valid class name (component classes are laser parts)."""
    return "object" if isinstance(decode_class(name), ObjectClass) else "laser"
