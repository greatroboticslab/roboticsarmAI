import pytest

from core.models import ImageLabelSet, LaserBox, ObjectBox


def test_json_round_trip():
    ls = ImageLabelSet(
        image_id="1", image_name="a.jpg", camera_id="cam", is_grayscale=False,
        background_level=0.2, exposure={"time": 0.01}, status="pending",
        laser_boxes=[LaserBox("l1", "scatter", [1.2, 2, 3, 4], 0, metrics={"peak_intensity": 0.5})],
        object_boxes=[ObjectBox("o1", (1, 2, 3, 4), "pen", "glass", "clear", "first",
                                laser_ids=["l1"], confidence=0.7)],
    )
    again = ImageLabelSet.from_json(ls.to_json())
    assert again == ls
    assert again.laser_boxes[0].box == (1, 2, 3, 4)
    assert again.laser_boxes[0].event_name == "scatter"


def test_validation():
    with pytest.raises(ValueError):
        LaserBox("l", "bogus", (0, 0, 1, 1), 0)
    with pytest.raises(ValueError):
        LaserBox("l", "other", (0, 0, 1), 0)
    with pytest.raises(ValueError):
        ObjectBox("o", (0, 0, 1, 1), "a", "c", "d", "sideways")
    with pytest.raises(ValueError):
        ObjectBox("o", (0, 0, 1, 1), "a", "c", "d", "first", confidence=2)
    with pytest.raises(ValueError):
        ImageLabelSet("1", "a", "c", False, 0.0, status="nope")


def test_first_incidence_event_name():
    assert LaserBox("l", "first_incidence", (0, 0, 1, 1), 0).event_name == "first-incidence"
