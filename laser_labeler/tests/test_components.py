"""Label scheme "components": object classes stay, laser spots become component1 (core), component2 (halo), component3 (beam)."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from config.settings import SettingsError, load_settings, validate_settings
from core.labeler.components import BEAM_SNAP_PX, apply_components, halo_box
from core.labeler.label_image import label_image
from core.models import ImageLabelSet, LaserBox, ObjectBox
from core.reports.measurements_csv import rows_for
from core.reports.pdf import class_label
from core.review.diffs import diff_label_sets
from core.review.service import add_laser_box, delete_box, measure_boxes
from core.schema.class_names import (COMPONENT_NAMES, ClassNameError, ComponentClass, LaserClass, class_family,
                                     decode_class, encode_component_class)
from core.schema.coco_export import build_coco, class_names_for
from core.schema.taxonomy import load_taxonomy
from core.tools.laser_detect import BeamLine
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot

ROOT = Path(__file__).resolve().parent.parent
TAX = load_taxonomy(ROOT / "taxonomy")


# ---------------------------------------------------------------- class names and the model
def test_component_class_names():
    assert COMPONENT_NAMES == ("component1", "component2", "component3")
    assert encode_component_class("component2") == "component2"
    assert decode_class("component3") == ComponentClass(3)
    assert class_family("component1") == "laser" and class_family("las_reflection") == "laser"
    assert class_family("obj_pen-ballpoint_rubber_red_first") == "object"
    assert isinstance(decode_class("las_reflection"), LaserClass)
    for bad in ("component4", "component", "Component1", "component1 "):
        with pytest.raises(ClassNameError):
            decode_class(bad)
    with pytest.raises(ClassNameError):
        encode_component_class("halo")


def test_laser_box_class_name_and_validation():
    plain = LaserBox("l1", "first_incidence", (1, 2, 3, 4), 0)
    assert plain.class_name == "las_first-incidence" and plain.component is None and not plain.is_part
    core = LaserBox("l1", "first_incidence", (1, 2, 3, 4), 0, component="component1")
    halo = LaserBox("l1h", "first_incidence", (1, 2, 3, 4), 0, "l1", component="component2")
    assert core.class_name == "component1" and not core.is_part
    assert halo.class_name == "component2" and halo.is_part
    with pytest.raises(ValueError):
        LaserBox("l1", "first_incidence", (1, 2, 3, 4), 0, component="component9")


def test_old_sidecars_without_a_component_still_load():
    data = {"id": "l1", "kind": "other", "box": [1, 2, 3, 4], "order": 0}
    assert LaserBox.from_dict(data).component is None
    ls = ImageLabelSet("a", "a.png", "cam", False, 0.1, laser_boxes=[LaserBox.from_dict(data)])
    back = ImageLabelSet.from_json(ls.to_json())
    assert back.laser_boxes[0].component is None


def test_settings_scheme_is_validated():
    s = load_settings()
    assert s["labeling"]["scheme"] == "full"
    s["labeling"]["scheme"] = "components"
    validate_settings(s)
    s["labeling"]["scheme"] = "everything"
    with pytest.raises(SettingsError, match="scheme"):
        validate_settings(s)


# ---------------------------------------------------------------- building the parts
def core_box(**metrics):
    base = {"center_x": 100.0, "center_y": 80.0, "halo_radius_px": 14.0, "scatter_ratio": 0.4, "total_light": 9.0,
            "confidence": 0.7}
    base.update(metrics)
    return LaserBox("l1", "first_incidence", (96, 76, 8, 8), 0, None, base)


def test_halo_box_needs_a_halo_bigger_than_the_core():
    box = halo_box(core_box(), 320, 240)
    assert box == (86, 66, 28, 28)
    assert halo_box(core_box(halo_radius_px=5.0), 320, 240) is None  # barely wider than the core: no separate part
    assert halo_box(core_box(halo_radius_px=0.0), 320, 240) is None
    assert halo_box(LaserBox("l1", "other", (1, 1, 4, 4), 0), 320, 240) is None  # nothing measured
    near_edge = halo_box(core_box(center_x=5.0, center_y=5.0), 320, 240)
    assert near_edge[0] == 0 and near_edge[1] == 0  # clamped inside the image


def test_apply_components_makes_core_halo_and_beam():
    lasers = [core_box()]
    line = BeamLine(20.0, 140.0, 96.0, 83.0)
    out = apply_components(lasers, [line], 320, 240)
    by = {b.component: b for b in out}
    assert set(by) == {"component1", "component2", "component3"}
    assert by["component1"].id == "l1" and not by["component1"].is_part
    halo, beam = by["component2"], by["component3"]
    assert halo.id == "l1h" and halo.from_id == "l1" and halo.order == 0 and halo.kind == "first_incidence"
    assert beam.id == "l1b" and beam.from_id == "l1"
    assert beam.box[0] <= 20 and beam.box[0] + beam.box[2] >= 96 and beam.box[1] <= 83 and beam.box[1] + beam.box[3] >= 140
    assert beam.metrics["length_px"] == pytest.approx(line.length)
    assert 0 <= beam.metrics["angle_deg"] < 360
    assert halo.metrics["confidence"] == beam.metrics["confidence"] == 0.7  # parts carry the core's confidence


def test_a_beam_line_far_from_every_spot_makes_no_component3():
    far = BeamLine(250.0, 200.0, 300.0, 230.0)
    out = apply_components([core_box(halo_radius_px=0.0)], [far], 320, 240)
    assert [b.component for b in out] == ["component1"]
    assert BEAM_SNAP_PX > 0


def test_each_beam_line_goes_to_one_spot_only():
    a = core_box()
    b = LaserBox("l2", "reflection", (196, 76, 8, 8), 1, "l1", {"center_x": 200.0, "center_y": 80.0, "halo_radius_px": 0.0,
                                                                  "confidence": 0.9})
    between = BeamLine(104.0, 80.0, 196.0, 80.0)  # ends at both spots
    out = apply_components([a, b], [between], 320, 240)
    assert sum(x.component == "component3" for x in out) == 1


def test_running_twice_changes_nothing():
    once = apply_components([core_box()], [], 320, 240)
    twice = apply_components(copy.deepcopy(once), [], 320, 240)
    assert [(b.id, b.component) for b in once] == [(b.id, b.component) for b in twice]


def test_ids_never_collide_with_existing_boxes():
    other = LaserBox("l1h", "other", (1, 1, 5, 5), 3)  # an unrelated box already using the halo id
    out = apply_components([core_box(), other], [], 320, 240)
    assert len({b.id for b in out}) == len(out)


# ---------------------------------------------------------------- export, CSV, PDF, diffs
def label_set(boxes):
    obj = ObjectBox("o1", (90, 70, 40, 30), "pen", "ballpoint", "rubber", "red", "first", ["l1"], None, 0.8)
    return ImageLabelSet("a", "a.png", "cam", False, 0.1, laser_boxes=boxes, object_boxes=[obj], status="approved",
                         image_width=320, image_height=240)


def test_coco_uses_component_classes_and_keeps_object_classes():
    ls = label_set(apply_components([core_box()], [BeamLine(20.0, 140.0, 96.0, 83.0)], 320, 240))
    names = [n for n, _ in class_names_for(ls)]
    assert sorted(names) == ["component1", "component2", "component3", "obj_pen-ballpoint_rubber_red_first"]
    coco = build_coco([ls])
    assert [c["name"] for c in coco["categories"]] == ["component1", "component2", "component3",
                                                       "obj_pen-ballpoint_rubber_red_first"]
    assert len(coco["annotations"]) == 4


def test_mixed_schemes_in_one_export():
    old = label_set([LaserBox("l1", "reflection", (96, 76, 8, 8), 0)])
    new = label_set([core_box()])
    new.image_id, new.image_name = "b", "b.png"
    new.laser_boxes[0].component = "component1"
    names = {c["name"] for c in build_coco([old, new])["categories"]}
    assert {"las_reflection", "component1"} <= names


def test_csv_pdf_and_diff_name_the_component():
    boxes = apply_components([core_box()], [], 320, 240)
    ls = label_set(boxes)
    assert {r["class"] for r in rows_for(ls)} == {"component1", "component2"}
    assert class_label(ls, "laser", boxes[1]) == "component2"
    edited = copy.deepcopy(ls)
    edited.laser_boxes[1].component = "component3"
    changes = diff_label_sets(ls, edited)
    assert any(c.field == "component" and c.before == "component2" and c.after == "component3" for c in changes)
    assert "component component2 -> component3" in " ".join(c.describe() for c in changes)


# ---------------------------------------------------------------- the whole labeler
@pytest.fixture
def halo_image(tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, halo_amp=0.25, halo_sigma=14.0, color=(1, 1, 1))
    path = tmp_path / "halo.png"
    Image.fromarray(finish(img)).save(path)
    return path


def good_reply():
    return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                "from_id": None, "object": PEN_RUBBER}]))


def run(image, tmp_path, scheme):
    settings = load_settings()
    settings["labeling"]["scheme"] = scheme
    return label_image(image, provider=ScriptedProvider([good_reply()]), taxonomy=TAX, settings=settings,
                       data_dir=tmp_path / "data")


def test_full_scheme_is_unchanged(halo_image, tmp_path):
    result = run(halo_image, tmp_path, "full")
    assert [b.component for b in result.laser_boxes] == [None]
    assert result.laser_boxes[0].class_name == "las_first-incidence"


def test_components_scheme_labels_core_and_halo(halo_image, tmp_path):
    result = run(halo_image, tmp_path, "components")
    classes = sorted(b.class_name for b in result.laser_boxes)
    assert classes[0] == "component1" and "component2" in classes
    core = next(b for b in result.laser_boxes if b.component == "component1")
    halo = next(b for b in result.laser_boxes if b.component == "component2")
    assert core.metrics["core_area_px"] > 3 and halo.from_id == core.id
    assert halo.box[2] > core.box[2] and halo.box[3] > core.box[3]  # the halo box is wider than the core box
    assert result.object_boxes[0].laser_ids == [core.id]  # the object is linked to the core, not to the halo
    assert result.status == "pending" and not result.questions
    assert 0 < result.object_boxes[0].confidence <= 1
    names = [n for n, _ in class_names_for(result)]
    assert any(n.startswith("obj_pen-ballpoint") for n in names) and "component1" in names


def test_components_scheme_with_a_plain_spot_has_just_the_core(tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))  # no halo
    path = tmp_path / "plain.png"
    Image.fromarray(finish(img)).save(path)
    result = run(path, tmp_path, "components")
    assert "component1" in [b.component for b in result.laser_boxes]
    assert all(b.component in ("component1", "component2", "component3") for b in result.laser_boxes)


# ---------------------------------------------------------------- review helpers
def test_remeasure_leaves_parts_alone(halo_image, tmp_path):
    result = run(halo_image, tmp_path, "components")
    halo = next(b for b in result.laser_boxes if b.component == "component2")
    before = dict(halo.metrics)
    result.image_width = result.image_width or 320
    measure_boxes(result, halo_image, load_settings(), [b.id for b in result.laser_boxes])
    assert halo.metrics == before


def test_drawn_laser_box_is_a_core_under_components(halo_image, tmp_path):
    settings = load_settings()
    settings["labeling"]["scheme"] = "components"
    result = run(halo_image, tmp_path, "components")
    new = add_laser_box(result, halo_image, settings, (20, 20, 12, 12), "reflection")
    assert new.component == "component1" and not new.is_part
    plain_settings = load_settings()
    other = add_laser_box(result, halo_image, plain_settings, (30, 30, 12, 12), "reflection")
    assert other.component is None  # full scheme: a plain laser box


def test_drawing_a_halo_attaches_it_to_the_nearest_core(halo_image, tmp_path):
    settings = load_settings()
    result = run(halo_image, tmp_path, "components")
    core = next(b for b in result.laser_boxes if b.component == "component1")
    part = add_laser_box(result, halo_image, settings, (core.box[0] - 10, core.box[1] - 10, 40, 40), "first_incidence", "component2")
    assert part.is_part and part.from_id == core.id and part.order == core.order
    empty = ImageLabelSet("x", "x.png", "cam", False, 0.1, image_width=320, image_height=240)
    from core.review.service import ReviewError
    with pytest.raises(ReviewError, match="core"):
        add_laser_box(empty, halo_image, settings, (10, 10, 20, 20), "first_incidence", "component3")
    with pytest.raises(ReviewError, match="Component must"):
        add_laser_box(result, halo_image, settings, (10, 10, 20, 20), "first_incidence", "component7")


def test_deleting_a_core_removes_its_parts():
    ls = label_set(apply_components([core_box()], [BeamLine(20.0, 140.0, 96.0, 83.0)], 320, 240))
    delete_box(ls, "l1")
    assert ls.laser_boxes == [] and ls.object_boxes[0].laser_ids == []
    ls2 = label_set(apply_components([core_box()], [], 320, 240))
    delete_box(ls2, "l1h")  # deleting just the halo keeps the core
    assert [b.component for b in ls2.laser_boxes] == ["component1"]


def test_golden_scoring_understands_component_classes():
    from core.labeler.golden import TruthBox, score_image
    truth = [TruthBox("component1", (96, 76, 8, 8)), TruthBox("component2", (86, 66, 28, 28)),
             TruthBox("obj_pen-ballpoint_rubber_red_first", (90, 70, 40, 30))]
    ls = label_set(apply_components([core_box()], [], 320, 240))
    scores = score_image(truth, ls)
    assert scores["laser"].truth == 2 and scores["laser"].predicted == 2 and scores["laser"].matched == 2
    assert scores["laser"].agree["event"] == 2  # both parts named correctly
    ls.laser_boxes[1].component = "component3"
    assert score_image(truth, ls)["laser"].agree["event"] == 1


# ---------------------------------------------------------------- a beam must be a ridge, not an object's edge
def _noise_and_map(img):
    from core.camera import image_stats, intensity_map
    rgb = finish(img)
    return intensity_map(rgb), image_stats(rgb).noise_level


def test_object_edges_are_not_beams_but_thin_beams_are():
    from core.labeler.components import is_ridge
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))  # a bright object: its edges are steps
    for t in np.linspace(0, 1, 400):  # a thin bright beam on the dark background
        x, y = int(10 + t * 80), int(200 - t * 60)
        img[y - 1:y + 1, x - 1:x + 1] += 120
    v, noise = _noise_and_map(img)
    assert is_ridge(BeamLine(10.0, 200.0, 90.0, 140.0), v, noise)  # the beam
    assert not is_ridge(BeamLine(100.0, 80.0, 180.0, 80.0), v, noise)  # top edge of the object
    assert not is_ridge(BeamLine(100.0, 80.0, 100.0, 140.0), v, noise)  # left edge
    assert not is_ridge(BeamLine(50.0, 20.0, 60.0, 20.0), v, noise)  # empty background
    assert not is_ridge(BeamLine(5.0, 5.0, 5.0, 5.0), v, noise)  # no length


def test_apply_components_uses_the_ridge_check_when_given_an_image():
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    v, noise = _noise_and_map(img)
    edge = BeamLine(100.0, 80.0, 100.0, 140.0)
    core = core_box(center_x=102.0, center_y=100.0, halo_radius_px=0.0)
    assert [b.component for b in apply_components([core], [edge], 320, 240, intensity=v, noise=noise)] == ["component1"]
    core2 = core_box(center_x=102.0, center_y=100.0, halo_radius_px=0.0)
    assert len(apply_components([core2], [edge], 320, 240)) == 2  # without the image the line is trusted


def test_a_line_through_a_glowing_spot_is_not_a_beam_but_a_beam_leading_to_it_is():
    img = scene()
    rect(img, 60, 70, 200, 100, (150, 30, 30))  # a large object with the spot in the middle
    spot(img, 140, 110, sigma=2.5, amp=0.5, halo_amp=0.25, halo_sigma=14.0, color=(1, 1, 1))
    for t in np.linspace(0, 1, 500):  # a thin beam arriving from the lower left
        x, y = int(8 + t * 104), int(205 - t * 93)
        img[y - 1:y + 1, x - 1:x + 1] += 120
    v, noise = _noise_and_map(img)

    def parts(line):
        core = core_box(center_x=139.5, center_y=109.5, halo_radius_px=30.0)
        return [b.component for b in apply_components([core], [line], 320, 240, intensity=v, noise=noise)]

    through = BeamLine(70.0, 80.0, 250.0, 160.0)  # across the object and through the spot's own glow
    assert parts(through).count("component3") == 0
    assert parts(BeamLine(8.0, 205.0, 112.0, 112.0)).count("component3") == 1  # the real beam
    assert parts(BeamLine(130.0, 115.0, 138.0, 110.0)).count("component3") == 0  # entirely inside the glow
