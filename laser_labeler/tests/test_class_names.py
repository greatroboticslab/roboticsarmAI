import pytest

from core.schema.class_names import (
    ClassNameError, LaserClass, ObjectClass, class_family, decode_class,
    encode_laser_class, encode_object_class, resolve_color,
)


def test_object_round_trip():
    name = encode_object_class("pen", "ballpoint", "rubber", "black", "first")
    assert name == "obj_pen-ballpoint_rubber_black_first"
    assert decode_class(name) == ObjectClass("pen", "ballpoint", "rubber", "black", "first")


def test_laser_round_trip():
    name = encode_laser_class("first-incidence")
    assert name == "las_first-incidence"
    assert decode_class(name) == LaserClass("first-incidence")


def test_grayscale_color_is_na():
    color = resolve_color("black", is_grayscale=True)
    assert encode_object_class("pen", "ballpoint", "rubber", color, "first") == \
        "obj_pen-ballpoint_rubber_na_first"
    assert resolve_color("black", is_grayscale=False) == "black"
    assert resolve_color("black", True, placeholder="gray0") == "gray0"


@pytest.mark.parametrize("bad", ["ball-point", "ball_point", "ball point", "Pen", "", "pen!"])
def test_separators_and_bad_chars_rejected(bad):
    with pytest.raises(ClassNameError):
        encode_object_class(bad, "ballpoint", "rubber", "black", "first")
    with pytest.raises(ClassNameError):
        encode_object_class("pen", bad, "rubber", "black", "first")


@pytest.mark.parametrize("bad", ["first_incidence", "First", "-x", "x-", "a--b", ""])
def test_bad_laser_events_rejected(bad):
    with pytest.raises(ClassNameError):
        encode_laser_class(bad)


@pytest.mark.parametrize("bad", [
    "", "pen", "obj_", "xyz_pen-ballpoint_rubber_black_first",
    "obj_pen_rubber_black_first",                     # missing sub-object hyphen field
    "obj_pen-ballpoint_rubber_black",                 # too few fields
    "obj_pen-ballpoint_rubber_black_first_extra",     # too many fields
    "obj_pen-ball-point_rubber_black_first",          # extra hyphen
    "las_", "las_Bad",
])
def test_decode_bad_input(bad):
    with pytest.raises(ClassNameError):
        decode_class(bad)


def test_decode_non_string():
    with pytest.raises(ClassNameError):
        decode_class(None)  # type: ignore[arg-type]


def test_class_family():
    assert class_family("las_scatter") == "laser"
    assert class_family("obj_table-desk_wood_brown_first") == "object"
