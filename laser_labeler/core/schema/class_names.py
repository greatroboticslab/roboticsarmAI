"""Encode and decode Roboflow class names (build spec section 3.2).

    Object: obj_<object>_<material>_<color>_<incidence>
    Laser:  las_<event>
    By how the light got there (label schemes "components" and "merged"): component1 = direct hit, component2 = bounce
    Merged (label scheme "merged"): <object class>__<component>, e.g. obj_pen_rubber_red_first__component1

Older exports also had a sub-object, obj_<object>-<sub-object>_<material>_<color>_<incidence>. Those names still decode
(the sub-object is dropped) so ground truth exported earlier can be compared, but they are no longer produced.

"-" and "_" are separators, so taxonomy entries may not contain them. Entries
are restricted to lowercase letters and digits. Laser events are the one place
a "-" is allowed (first-incidence), because the event is a single field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

OBJECT_PREFIX = "obj"
LASER_PREFIX = "las"
MERGE_SEP = "__"

COMPONENT_NAMES = ("component1", "component2")
COMPONENT_MEANING = {
    "component1": "direct hit: the beam lands on the object",
    "component2": "bounce: light reflected, refracted or scattered off something",
}

_TOKEN = re.compile(r"^[a-z0-9]+$")
_EVENT = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class ClassNameError(ValueError):
    """Raised for names that cannot be encoded or decoded."""


@dataclass(frozen=True)
class ObjectClass:
    general_object: str
    material: str
    color: str
    incidence: str


@dataclass(frozen=True)
class LaserClass:
    event: str


@dataclass(frozen=True)
class ComponentClass:
    number: int


@dataclass(frozen=True)
class MergedClass:
    """A laser part box that also names the object it lands on."""
    object: ObjectClass
    component: ComponentClass


def check_token(value: str, label: str) -> str:
    """Return value if it is a legal taxonomy token, else raise ClassNameError."""
    if not isinstance(value, str) or not _TOKEN.match(value):
        raise ClassNameError(
            f"{label} {value!r} must be lowercase letters/digits only "
            "(no '-', '_', spaces or uppercase; write multi-word names as one word)"
        )
    return value


def normalise_token(value, default: str = "other") -> str:
    """Make a free-text model value a legal token: lowercase letters and digits only ("Dark Blue" -> "darkblue")."""
    cleaned = re.sub(r"[^a-z0-9]", "", str(value or "").lower())
    return cleaned or default


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


def encode_object_class(general_object: str, material: str, color: str, incidence: str) -> str:
    parts = [
        check_token(general_object, "object"),
        check_token(material, "material"),
        check_token(color, "color"),
        check_token(incidence, "incidence"),
    ]
    return f"{OBJECT_PREFIX}_{'_'.join(parts)}"


def encode_laser_class(event: str) -> str:
    return f"{LASER_PREFIX}_{check_event(event)}"


def encode_component_class(name: str) -> str:
    if name not in COMPONENT_NAMES:
        raise ClassNameError(f"component {name!r} must be one of {', '.join(COMPONENT_NAMES)}")
    return name


def encode_merged_class(object_class: str, component: str) -> str:
    """Object class and laser part in one name. `object_class` must already be a valid object class name."""
    decode_class(object_class)
    return f"{object_class}{MERGE_SEP}{encode_component_class(component)}"


def decode_class(name: str) -> ObjectClass | LaserClass | ComponentClass | MergedClass:
    """Inverse of the encode functions. Raises ClassNameError on bad input."""
    if not isinstance(name, str) or not name:
        raise ClassNameError("class name must be a non-empty string")
    if MERGE_SEP in name:
        left, _, right = name.partition(MERGE_SEP)
        obj, part = decode_class(left), decode_class(right)
        if not isinstance(obj, ObjectClass) or not isinstance(part, ComponentClass):
            raise ClassNameError(f"merged class {name!r} must be '<object class>{MERGE_SEP}<componentN>'")
        return MergedClass(obj, part)
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
            f"(object, material, color, incidence); found {len(fields)}"
        )
    names = fields[0].split("-")
    if len(names) > 2:  # "<object>-<sub-object>" is the older form: keep the object, drop the sub-object
        raise ClassNameError(f"object class {name!r}: first field must be the object name")
    return ObjectClass(
        general_object=check_token(names[0], "object"),
        material=check_token(fields[1], "material"),
        color=check_token(fields[2], "color"),
        incidence=check_token(fields[3], "incidence"),
    )


def class_family(name: str) -> str:
    """Return 'object' or 'laser' for a valid class name (component classes are laser parts)."""
    return "object" if isinstance(decode_class(name), ObjectClass) else "laser"  # merged names are laser parts
