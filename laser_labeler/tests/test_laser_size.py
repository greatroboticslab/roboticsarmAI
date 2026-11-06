"""Laser boxes are bigger than the bare spot and reach about a third of the object box they hit."""
from pathlib import Path

import pytest
from PIL import Image

from config.settings import SettingsError, load_settings, validate_settings
from core.labeler.label_image import label_image
from core.labeler.laser_size import GLOW_MARGIN, grow_laser_boxes, grown_box
from core.models import LaserBox, ObjectBox
from core.schema.taxonomy import load_taxonomy
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot

TAX = load_taxonomy(Path(__file__).resolve().parent.parent / "taxonomy")


def laser(box=(100, 100, 20, 20), component=None, id="l1"):
    return LaserBox(id, "first_incidence", box, 0, component=component)


def obj(box, laser_ids=("l1",)):
    return ObjectBox("o1", box, "pen", "rubber", "red", "first", list(laser_ids), None, 0.9, "")


def test_without_an_object_the_box_only_gets_the_glow_margin():
    box = grown_box((100, 100, 20, 20), None, 0.33, 400, 400)
    assert box == (95, 95, 30, 30)  # 25% more on each side, same centre
    assert (box[0] + box[2] / 2, box[1] + box[3] / 2) == (110, 110)


def test_box_grows_to_a_third_of_the_object_and_stays_centred():
    box = grown_box((100, 100, 20, 20), (50, 60, 240, 150), 0.33, 600, 600)
    assert box[2] == pytest.approx(0.33 * 240, abs=1) and box[3] == pytest.approx(0.33 * 150, abs=1)
    assert (box[0] + box[2] / 2, box[1] + box[3] / 2) == pytest.approx((110, 110), abs=1)


def test_box_never_passes_half_the_object_and_never_shrinks():
    big = grown_box((100, 100, 20, 20), (50, 60, 240, 150), 0.5, 600, 600)
    assert big[2] == 120 and big[3] == 75
    capped = grown_box((100, 100, 20, 20), (50, 60, 240, 150), 0.9, 600, 600)
    assert capped[2] == 120  # asking for more than half is held at half
    already_big = grown_box((100, 100, 200, 200), (150, 150, 30, 30), 0.33, 600, 600)
    assert already_big[2] >= 200 * (1 + 2 * GLOW_MARGIN) - 1


def test_box_stays_inside_the_image():
    x, y, w, h = grown_box((2, 2, 20, 20), (0, 0, 300, 300), 0.33, 120, 120)
    assert x >= 0 and y >= 0 and x + w <= 120 and y + h <= 120


def test_zero_fraction_and_component_parts_are_left_alone():
    tight, part = laser(), laser(component="component2", id="l2")
    grow_laser_boxes([tight, part], [obj((50, 60, 240, 150), ("l1", "l2"))], 0, 600, 600)
    assert tight.box == (100, 100, 20, 20)
    grow_laser_boxes([tight, part], [obj((50, 60, 240, 150), ("l1", "l2"))], 0.33, 600, 600)
    assert tight.box != (100, 100, 20, 20) and part.box == (100, 100, 20, 20)


def test_setting_is_validated():
    s = load_settings()
    assert s["labeling"]["laser_box_object_fraction"] == 0.33
    s["labeling"]["laser_box_object_fraction"] = 0.8
    with pytest.raises(SettingsError):
        validate_settings(s)


@pytest.fixture
def image(tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / "cam1" / "red_object.png"
    path.parent.mkdir()
    Image.fromarray(finish(img)).save(path)
    return path


def labeled(image, tmp_path, fraction):
    settings = load_settings()
    settings["labeling"]["laser_box_object_fraction"] = fraction
    reply = text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                 "from_id": None, "object": PEN_RUBBER}]))
    return label_image(image, provider=ScriptedProvider([reply]), taxonomy=TAX, settings=settings,
                       data_dir=tmp_path / "data")


def test_label_image_makes_the_laser_box_bigger_than_the_tight_one(image, tmp_path):
    tight = labeled(image, tmp_path, 0).laser_boxes[0]
    wide = labeled(image, tmp_path, 0.33).laser_boxes[0]
    ox, oy, ow, oh = labeled(image, tmp_path, 0.33).object_boxes[0].box
    assert wide.box[2] > tight.box[2] and wide.box[3] > tight.box[3]
    assert wide.box[2] >= 0.33 * ow - 1 and wide.box[3] >= 0.33 * oh - 1
    assert wide.metrics == tight.metrics  # measured before the box was enlarged
    cx = lambda b: b[0] + b[2] / 2  # noqa: E731
    assert abs(cx(wide.box) - cx(tight.box)) <= 1.5
