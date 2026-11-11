"""Boxes must land on the right place whatever picture size, rotation or number system is involved."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image

from config.settings import SettingsError, load_settings, validate_settings
from core.camera import load_image, oriented_size
from core.labeler.coords import guess_by_range, resolve_space, to_shown_px
from core.labeler.label_image import label_image
from core.models import ImageLabelSet, LaserBox, ObjectBox
from core.schema.orientation import box_to_upright, point_to_upright, to_upright
from tests.fakes import ScriptedProvider, text, tool
from tests.test_label_image import FINAL, TAX, image  # noqa: F401  (the fixture)
from tests.synth import finish, rect, scene, spot


# ---------------------------------------------------------------- EXIF orientation
def _marked_picture(path, orientation, w=200, h=100):
    """A picture stored w x h with a bright 10 x 10 square at (30, 20), tagged with an EXIF orientation."""
    arr = np.zeros((h, w, 3), np.uint8)
    arr[20:30, 30:40] = 255
    pil = Image.fromarray(arr)
    exif = pil.getexif()
    exif[0x0112] = orientation
    pil.save(path, exif=exif)


def _centroid(rgb):
    ys, xs = np.nonzero(rgb[..., 0] > 128)
    return xs.mean() + 0.5, ys.mean() + 0.5


@pytest.mark.parametrize("orientation", range(1, 9))
def test_loader_turns_the_picture_upright_and_the_formula_agrees(tmp_path, orientation):
    path = tmp_path / f"o{orientation}.jpg"
    _marked_picture(path.with_suffix(".png"), orientation)  # PNG keeps EXIF too; lossless, so the square stays exact
    path = path.with_suffix(".png")
    data = load_image(path)
    assert (data.width, data.height) == oriented_size(path) == ((100, 200) if orientation >= 5 else (200, 100))
    cx, cy = _centroid(data.rgb)
    expected = point_to_upright(35, 25, orientation, 200, 100)
    assert (cx, cy) == pytest.approx(expected, abs=0.6)
    box = box_to_upright((30, 20, 10, 10), orientation, 200, 100)
    assert box[2:] == ((10, 10)) and (box[0] + 5, box[1] + 5) == pytest.approx(expected, abs=0.6)


def test_older_label_set_is_moved_onto_the_upright_picture(tmp_path):
    path = tmp_path / "p.png"
    _marked_picture(path, 6)
    old = ImageLabelSet("p", "p.png", "cam", False, 0.1, laser_boxes=[LaserBox("l1", "first_incidence", (30, 20, 10, 10), 0, None, {"center_x": 35.0, "center_y": 25.0})],
                        object_boxes=[ObjectBox("o1", (10, 10, 60, 40), "pen", "rubber", "red", "first", ["l1"], None, 0.9)],
                        image_width=200, image_height=100)
    assert old.pixel_space == "stored"
    assert to_upright(old, path)
    assert (old.image_width, old.image_height, old.pixel_space) == (100, 200, "upright")
    upright = load_image(path)
    cx, cy = _centroid(upright.rgb)
    x, y, w, h = old.laser_boxes[0].box
    assert (x + w / 2, y + h / 2) == pytest.approx((cx, cy), abs=0.6)
    assert old.laser_boxes[0].metrics["center_x"] == pytest.approx(cx, abs=0.6)
    assert not to_upright(old, path)  # a second call changes nothing


def test_a_photo_without_orientation_is_left_alone(tmp_path):
    path = tmp_path / "plain.png"
    _marked_picture(path, 1)
    ls = ImageLabelSet("p", "p.png", "cam", False, 0.1, laser_boxes=[LaserBox("l1", "first_incidence", (30, 20, 10, 10), 0, None, {})],
                       image_width=200, image_height=100)
    assert not to_upright(ls, path) and ls.laser_boxes[0].box == (30, 20, 10, 10) and ls.pixel_space == "upright"


def test_new_label_sets_are_made_on_the_upright_picture(tmp_path):
    arr = finish(scene())
    path = tmp_path / "rot.png"
    pil = Image.fromarray(arr)
    exif = pil.getexif()
    exif[0x0112] = 6
    pil.save(path, exif=exif)
    from config.settings import load_settings as ls_
    result = label_image(path, provider=ScriptedProvider([]), taxonomy=TAX, settings=ls_(), data_dir=tmp_path / "d")
    assert result.pixel_space == "upright" and (result.image_width, result.image_height) == (240, 320)


def test_thumbnail_and_canvas_show_the_upright_picture(tmp_path):
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from gui.widgets.image_canvas import ImageCanvas
    from gui.widgets.thumb_grid import make_thumbnail
    path = tmp_path / "t.png"
    _marked_picture(path, 6)
    thumb = make_thumbnail(str(path), [("laser", (30, 20, 10, 10), 100, 200)], 150)
    assert thumb.height() > thumb.width()  # taller than wide: turned upright
    canvas = ImageCanvas()
    assert canvas.set_image(str(path)) and canvas.image_size() == (100, 200)


# ---------------------------------------------------------------- the model's number system
def test_conversion_between_spaces():
    assert to_shown_px((100, 200, 50, 40), "thousand", 800, 600) == (80.0, 120.0, 40.0, 24.0)
    assert to_shown_px((0.5, 0.25, 0.1, 0.2), "fraction", 800, 600) == (400.0, 150.0, 80.0, 120.0)
    assert to_shown_px((10, 20, 30, 40), "pixels", 800, 600) == (10, 20, 30, 40)


def test_range_guess():
    assert guess_by_range([(0.2, 0.3, 0.1, 0.1)], 800, 600) == "fraction"
    assert guess_by_range([(500, 400, 100, 100)], 400, 300) == "thousand"  # runs past a 400 px wide picture
    assert guess_by_range([(100, 80, 40, 40)], 800, 600) == "pixels"


def test_spots_decide_when_the_numbers_could_be_either():
    spots = [(320.0, 240.0)]  # shown picture 640 x 480
    laser = [(480, 480, 40, 40)]  # on a 0..1000 grid that is (320, 240) in pixels
    space, why = resolve_space(laser, [], spots, 640, 480)
    assert space == "thousand" and "detected spots" in why
    assert resolve_space([(300, 220, 40, 40)], [], spots, 640, 480)[0] == "pixels"  # pixels already fit: left alone
    assert resolve_space(laser, [], spots, 640, 480, "pixels")[0] == "pixels"  # a fixed setting wins


def test_agent_reply_on_a_thousand_grid_lands_in_the_right_place(image, tmp_path):  # noqa: F811
    settings = load_settings()
    settings["labeling"]["laser_box_object_fraction"] = 0
    settings["provider"]["max_image_px"] = 160  # shown 160 x 120; the model answers on a 0..1000 grid
    grid = {**FINAL, "laser_boxes": [{**FINAL["laser_boxes"][0], "box": [406, 417, 62, 83]}],
            "object_boxes": [{**FINAL["object_boxes"][0], "box": [312, 333, 250, 250]}]}
    result = label_image(image, provider=ScriptedProvider([tool("detect_laser"), text(grid)]), taxonomy=TAX,
                         settings=settings, data_dir=tmp_path / "data", mode="agent")
    x, y, w, h = result.laser_boxes[0].box
    assert (x + w / 2, y + h / 2) == pytest.approx((140, 110), abs=8)  # the spot is at (140, 110) in the original
    assert "read as thousand" in result.notes
    assert result.object_boxes[0].box[0] == pytest.approx(100, abs=8)


def test_agent_reply_in_fractions_lands_in_the_right_place(image, tmp_path):  # noqa: F811
    settings = load_settings()
    settings["labeling"]["laser_box_object_fraction"] = 0
    fractions = {**FINAL, "laser_boxes": [{**FINAL["laser_boxes"][0], "box": [0.40, 0.41, 0.06, 0.08]}],
                 "object_boxes": [{**FINAL["object_boxes"][0], "box": [0.31, 0.33, 0.25, 0.25]}]}
    result = label_image(image, provider=ScriptedProvider([tool("detect_laser"), text(fractions)]), taxonomy=TAX,
                         settings=settings, data_dir=tmp_path / "data", mode="agent")
    x, y, w, h = result.laser_boxes[0].box
    assert (x + w / 2, y + h / 2) == pytest.approx((140, 110), abs=8) and "read as fraction" in result.notes


def test_pixel_replies_are_not_touched(image, tmp_path):  # noqa: F811
    settings = load_settings()
    settings["labeling"]["laser_box_object_fraction"] = 0
    result = label_image(image, provider=ScriptedProvider([tool("detect_laser"), text(FINAL)]), taxonomy=TAX,
                         settings=settings, data_dir=tmp_path / "data", mode="agent")
    assert result.laser_boxes[0].box == (130, 100, 20, 20) and "read as" not in result.notes


def test_setting_is_validated():
    s = load_settings()
    assert s["labeling"]["model_coordinates"] == "auto"
    s["labeling"]["model_coordinates"] = "metres"
    with pytest.raises(SettingsError):
        validate_settings(s)


# ---------------------------------------------------------------- a laser box far from every spot
def test_snap_moves_a_far_box_and_keeps_a_good_one():
    from core.labeler.snap import snap_lasers

    class Cand:
        def __init__(self, id, box):
            self.id, self.box = id, box

    cands = [Cand("c1", (100, 100, 10, 10)), Cand("c2", (300, 200, 10, 10))]
    items = [{"id": "l1", "box": (96, 96, 20, 20)}, {"id": "l2", "box": (10, 10, 20, 20)}, {"id": "l3", "box": (5, 150, 20, 20)}]
    out, snaps = snap_lasers(items, cands)
    assert out[0]["box"] == (96, 96, 20, 20)  # already on c1: untouched
    assert out[1]["box"] == (300, 200, 10, 10)  # moved to the only free spot
    assert out[2]["box"] == (5, 150, 20, 20)  # nothing free: left, but reported
    assert [(s.laser_id, s.moved) for s in snaps] == [("l2", True), ("l3", False)]
    assert snap_lasers(items, [])[1] == []  # no spots found: nothing to compare with


def test_agent_laser_box_far_from_the_spot_is_moved_and_flagged(image, tmp_path):  # noqa: F811
    settings = load_settings()
    settings["labeling"]["laser_box_object_fraction"] = 0
    far = {**FINAL, "laser_boxes": [{**FINAL["laser_boxes"][0], "box": [20, 200, 30, 30]}]}
    result = label_image(image, provider=ScriptedProvider([tool("detect_laser"), text(far)]), taxonomy=TAX,
                         settings=settings, data_dir=tmp_path / "data", mode="agent")
    laser = result.laser_boxes[0]
    x, y, w, h = laser.box
    assert abs((x + w / 2) - 140) < 12 and abs((y + h / 2) - 110) < 12  # on the spot the app found
    assert laser.metrics["confidence"] <= 0.3 and "moved onto the nearest detected spot" in laser.notes
    flag = next(q for q in result.questions if q.id == "qm_l1")
    assert flag.target_id == "l1" and "moved it onto that spot" in flag.text and result.status == "needs_input"
    assert "far from any laser spot" in result.notes
    from core.review.reasons import reason_line
    assert reason_line(result) == "box moved onto a spot"


def test_a_box_on_the_spot_is_not_flagged(image, tmp_path):  # noqa: F811
    settings = load_settings()
    result = label_image(image, provider=ScriptedProvider([tool("detect_laser"), text(FINAL)]), taxonomy=TAX,
                         settings=settings, data_dir=tmp_path / "data", mode="agent")
    assert not any(q.id.startswith("qm_") for q in result.questions)


def test_a_model_that_saw_a_resized_copy_is_translated():
    # the app showed 640 x 480; the server shrank it to 448 on the long side, so the model's numbers are on 448 x 336
    spots = [(320.0, 240.0)]
    k = 448 / 640
    laser = [(320 * k - 10, 240 * k - 10, 20, 20)]
    space, why = resolve_space(laser, [], spots, 640, 480)
    assert space == "side:448" and "side:448" in why
    x, y, w, h = to_shown_px(laser[0], space, 640, 480)
    assert (x + w / 2, y + h / 2) == pytest.approx((320, 240), abs=1)
    assert resolve_space([(318, 238, 8, 8)], [], spots, 640, 480)[0] == "pixels"  # already right: left alone
