"""Label scheme "components": object classes stay, laser spots become component1 (direct hit) or component2 (bounce)."""
import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from config.settings import SettingsError, load_settings, validate_settings
from core.labeler.components import apply_components, component_for_kind
from core.labeler.label_image import label_image
from core.models import ImageLabelSet, LaserBox, ObjectBox, Question
from core.reports.measurements_csv import rows_for
from core.reports.pdf import class_label
from core.review.diffs import diff_label_sets
from core.review.service import add_laser_box, delete_box, measure_boxes
from core.schema.class_names import (COMPONENT_NAMES, ClassNameError, ComponentClass, LaserClass, class_family,
                                     decode_class, encode_component_class)
from core.schema.coco_export import build_coco, class_names_for
from core.schema.taxonomy import load_taxonomy
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot

ROOT = Path(__file__).resolve().parent.parent
TAX = load_taxonomy(ROOT / "taxonomy")


# ---------------------------------------------------------------- class names and the model
def test_component_class_names():
    assert COMPONENT_NAMES == ("component1", "component2")
    assert encode_component_class("component2") == "component2"
    assert decode_class("component2") == ComponentClass(2)
    assert class_family("component1") == "laser" and class_family("las_reflection") == "laser"
    assert class_family("obj_pen_rubber_red_first") == "object"
    assert isinstance(decode_class("las_reflection"), LaserClass)
    for bad in ("component3", "component4", "component", "Component1", "component1 "):
        with pytest.raises(ClassNameError):
            decode_class(bad)
    with pytest.raises(ClassNameError):
        encode_component_class("halo")


def test_laser_box_class_name_and_validation():
    plain = LaserBox("l1", "first_incidence", (1, 2, 3, 4), 0)
    assert plain.class_name == "las_first-incidence" and plain.component is None
    core = LaserBox("l1", "first_incidence", (1, 2, 3, 4), 0, component="component1")
    bounce = LaserBox("l2", "reflection", (1, 2, 3, 4), 1, "l1", component="component2")
    assert core.class_name == "component1" and bounce.class_name == "component2"
    with pytest.raises(ValueError):
        LaserBox("l1", "first_incidence", (1, 2, 3, 4), 0, component="component3")


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


# ---------------------------------------------------------------- direct hit or bounce
@pytest.mark.parametrize("kind,expected", [("first_incidence", "component1"), ("first-incidence", "component1"),
                                           ("reflection", "component2"), ("refraction", "component2"),
                                           ("scatter", "component2"), ("other", "component2")])
def test_the_event_decides_the_component(kind, expected):
    assert component_for_kind(kind) == expected


def test_every_spot_is_labeled_and_two_direct_hits_are_both_component1():
    lasers = [LaserBox("l1", "first_incidence", (10, 10, 8, 8), 0), LaserBox("l2", "first_incidence", (60, 10, 8, 8), 0),
              LaserBox("l3", "reflection", (90, 40, 8, 8), 1, "l1"), LaserBox("l4", "scatter", (30, 40, 8, 8), 1, "l2"),
              LaserBox("l5", "refraction", (50, 80, 8, 8), 2, "l3"), LaserBox("l6", "other", (70, 80, 8, 8), 3, "l5")]
    out = apply_components(lasers)
    assert [b.component for b in out] == ["component1", "component1", "component2", "component2", "component2", "component2"]
    assert [b.box for b in out] == [b.box for b in lasers]  # nothing is added, dropped or moved


def test_running_twice_changes_nothing():
    once = apply_components([LaserBox("l1", "reflection", (1, 2, 3, 4), 0)])
    assert [b.component for b in apply_components(once)] == ["component2"]


# ---------------------------------------------------------------- export, CSV, PDF, diffs
def label_set(boxes):
    obj = ObjectBox("o1", (90, 70, 40, 30), "pen", "rubber", "red", "first", ["l1"], None, 0.8)
    return ImageLabelSet("a", "a.png", "cam", False, 0.1, laser_boxes=boxes, object_boxes=[obj], status="approved",
                         image_width=320, image_height=240)


def two_spots():
    return apply_components([LaserBox("l1", "first_incidence", (96, 76, 8, 8), 0),
                             LaserBox("l2", "reflection", (150, 76, 8, 8), 1, "l1")])


def test_coco_uses_component_classes_and_keeps_object_classes():
    ls = label_set(two_spots())
    names = [n for n, _ in class_names_for(ls)]
    assert sorted(names) == ["component1", "component2", "obj_pen_rubber_red_first"]
    coco = build_coco([ls])
    assert [c["name"] for c in coco["categories"]] == ["component1", "component2", "obj_pen_rubber_red_first"]
    assert len(coco["annotations"]) == 3


def test_mixed_schemes_in_one_export():
    old = label_set([LaserBox("l1", "reflection", (96, 76, 8, 8), 0)])
    new = label_set([LaserBox("l1", "first_incidence", (96, 76, 8, 8), 0, component="component1")])
    new.image_id, new.image_name = "b", "b.png"
    names = {c["name"] for c in build_coco([old, new])["categories"]}
    assert {"las_reflection", "component1"} <= names


def test_csv_pdf_and_diff_name_the_component():
    boxes = two_spots()
    ls = label_set(boxes)
    assert {r["class"] for r in rows_for(ls)} == {"component1", "component2"}
    assert class_label(ls, "laser", boxes[1]) == "component2"
    edited = copy.deepcopy(ls)
    edited.laser_boxes[1].component = "component1"
    changes = diff_label_sets(ls, edited)
    assert any(c.field == "component" and c.before == "component2" and c.after == "component1" for c in changes)
    assert "component component2 -> component1" in " ".join(c.describe() for c in changes)


# ---------------------------------------------------------------- old saved labels
def test_old_halo_and_beam_boxes_are_migrated_away():
    old = label_set([LaserBox("l1", "first_incidence", (96, 76, 8, 8), 0, component="component1"),
                     LaserBox("l1h", "first_incidence", (86, 66, 28, 28), 0, "l1", component="component2"),
                     LaserBox("l1b", "first_incidence", (20, 70, 70, 10), 0, "l1", component="component2"),
                     LaserBox("l2", "reflection", (150, 76, 8, 8), 1, "l1", component="component1")])
    old.scheme = "components"
    old.object_boxes[0].laser_ids = ["l1", "l1h", "l1b"]
    old.questions = [Question("q1", "about the halo", target_id="l1h")]
    data = json.loads(old.to_json())
    data.pop("scheme_version")
    data["laser_boxes"][2]["component"] = "component3"  # what an old file really holds; no longer a valid class
    back = ImageLabelSet.from_dict(data)
    assert [(b.id, b.component) for b in back.laser_boxes] == [("l1", "component1"), ("l2", "component2")]
    assert back.object_boxes[0].laser_ids == ["l1"]
    assert back.questions[0].target_id == "l1"
    assert back.scheme_version == 2
    again = ImageLabelSet.from_json(back.to_json())  # a migrated file is not touched twice
    assert [b.id for b in again.laser_boxes] == ["l1", "l2"]


def test_old_full_scheme_sidecars_are_left_alone():
    old = label_set([LaserBox("l1", "reflection", (96, 76, 8, 8), 0)])
    data = json.loads(old.to_json())
    data.pop("scheme_version")
    back = ImageLabelSet.from_dict(data)
    assert back.laser_boxes[0].component is None and back.laser_boxes[0].class_name == "las_reflection"


# ---------------------------------------------------------------- the whole labeler
@pytest.fixture
def spot_image(tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, halo_amp=0.25, halo_sigma=14.0, color=(1, 1, 1))
    path = tmp_path / "spot.png"
    Image.fromarray(finish(img)).save(path)
    return path


def good_reply(kind="first_incidence"):
    return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": kind, "order": 0,
                                "from_id": None, "object": PEN_RUBBER}]))


def run(image, tmp_path, scheme, kind="first_incidence"):
    settings = load_settings()
    settings["labeling"]["scheme"] = scheme
    return label_image(image, provider=ScriptedProvider([good_reply(kind)]), taxonomy=TAX, settings=settings,
                       data_dir=tmp_path / "data")


def test_full_scheme_is_unchanged(spot_image, tmp_path):
    result = run(spot_image, tmp_path, "full")
    assert [b.component for b in result.laser_boxes] == [None]
    assert result.laser_boxes[0].class_name == "las_first-incidence"


def test_components_scheme_gives_one_box_per_spot_and_no_halo(spot_image, tmp_path):
    result = run(spot_image, tmp_path, "components")
    assert [b.class_name for b in result.laser_boxes] == ["component1"]  # a glowing spot is still one spot
    spot_box = result.laser_boxes[0]
    assert spot_box.metrics["core_area_px"] > 3
    assert result.object_boxes[0].laser_ids == [spot_box.id]
    assert result.status == "pending" and not result.questions
    names = [n for n, _ in class_names_for(result)]
    assert any(n.startswith("obj_pen_") for n in names) and "component1" in names


def test_a_bounce_is_component2(spot_image, tmp_path):
    result = run(spot_image, tmp_path, "components", kind="reflection")
    assert [b.class_name for b in result.laser_boxes] == ["component2"]


def test_merged_scheme_names_the_spot_after_its_object(spot_image, tmp_path):
    result = run(spot_image, tmp_path, "merged")
    assert result.scheme == "merged"
    assert [n for n, _ in class_names_for(result)][0].endswith("__component1")


# ---------------------------------------------------------------- review helpers
def test_remeasure_updates_every_spot(spot_image, tmp_path):
    result = run(spot_image, tmp_path, "components")
    box = result.laser_boxes[0]
    box.metrics = {}
    measure_boxes(result, spot_image, load_settings())
    assert box.metrics


def test_drawn_laser_box_follows_its_event(spot_image, tmp_path):
    result = run(spot_image, tmp_path, "components")
    settings = load_settings()
    settings["labeling"]["scheme"] = "components"
    direct = add_laser_box(result, spot_image, settings, (150, 90, 12, 12))
    bounce = add_laser_box(result, spot_image, settings, (170, 90, 12, 12), kind="reflection")
    assert direct.component == "component1" and bounce.component == "component2"
    assert direct.metrics and bounce.metrics
    full = load_settings()
    assert add_laser_box(result, spot_image, full, (150, 120, 12, 12)).component is None
    with pytest.raises(Exception, match="Component must be"):
        add_laser_box(result, spot_image, settings, (150, 90, 12, 12), component="component3")


def test_deleting_a_spot_removes_just_that_box():
    ls = label_set(two_spots())
    delete_box(ls, "l1")
    assert [b.id for b in ls.laser_boxes] == ["l2"] and ls.object_boxes[0].laser_ids == []


def test_golden_scoring_understands_component_classes():
    from core.labeler.golden import TruthBox, score_image
    truth = [TruthBox("component1", (96, 76, 8, 8)), TruthBox("component2", (150, 76, 8, 8)),
             TruthBox("obj_pen_rubber_red_first", (90, 70, 40, 30))]
    ls = label_set(two_spots())
    scores = score_image(truth, ls)
    assert scores["laser"].truth == 2 and scores["laser"].predicted == 2 and scores["laser"].matched == 2
    assert scores["laser"].agree["event"] == 2
    ls.laser_boxes[1].component = "component1"
    assert score_image(truth, ls)["laser"].agree["event"] == 1


def test_a_drawn_laser_box_is_linked_to_the_object_it_lands_in(spot_image, tmp_path):
    result = run(spot_image, tmp_path, "merged")
    settings = load_settings()
    settings["labeling"]["scheme"] = "merged"
    obj = result.object_boxes[0]
    x, y, w, h = obj.box
    drawn = add_laser_box(result, spot_image, settings, (x + w / 2 - 6, y + h / 2 - 6, 12, 12), kind="reflection")
    assert drawn.id in obj.laser_ids
    name = result.laser_class_name(drawn)
    assert name.startswith("obj_pen_") and name.endswith("__component2")  # object and component in one label


def test_apply_scheme_converts_without_the_model():
    from core.review.service import apply_scheme
    ls = label_set([LaserBox("l1", "first_incidence", (96, 76, 8, 8), 0), LaserBox("l2", "scatter", (150, 76, 8, 8), 1, "l1")])
    assert apply_scheme(ls, "merged") and ls.scheme == "merged"
    assert [b.component for b in ls.laser_boxes] == ["component1", "component2"]
    assert not apply_scheme(ls, "merged")  # nothing left to change
    assert apply_scheme(ls, "full") and [b.component for b in ls.laser_boxes] == [None, None]
    with pytest.raises(Exception, match="Label scheme must be"):
        apply_scheme(ls, "everything")


def test_an_unlinked_spot_inside_an_object_still_gets_the_merged_name():
    ls = label_set([LaserBox("l1", "first_incidence", (100, 80, 8, 8), 0, component="component1")])
    ls.scheme = "merged"
    ls.object_boxes[0].laser_ids = []  # nobody linked it
    assert ls.laser_class_name(ls.laser_boxes[0]) == "obj_pen_rubber_red_first__component1"
    far = label_set([LaserBox("l1", "first_incidence", (300, 200, 8, 8), 0, component="component1")])
    far.scheme = "merged"
    far.object_boxes[0].laser_ids = []
    assert far.laser_class_name(far.laser_boxes[0]) == "component1"  # on no object, so nothing to name it after


def test_switching_scheme_is_a_saved_change_but_not_a_prompt_correction():
    before = label_set([LaserBox("l1", "first_incidence", (96, 76, 8, 8), 0)])
    after = copy.deepcopy(before)
    from core.review.service import apply_scheme
    apply_scheme(after, "merged")
    changes = diff_label_sets(before, after)
    assert [c.kind for c in changes] == ["scheme"] and "label scheme full -> merged" in changes[0].describe()


def test_crop_size_setting_reaches_the_request(spot_image, tmp_path):
    from PIL import Image as PILImage
    import io
    sizes = {}
    for px in (128, 384):
        settings = load_settings()
        settings["run"]["crop_px"] = px
        prov = ScriptedProvider([good_reply()])
        label_image(spot_image, provider=prov, taxonomy=TAX, settings=settings, data_dir=tmp_path / f"d{px}")
        user = next(m for m in prov.calls[0] if m.role == "user")
        crop = PILImage.open(io.BytesIO(user.images[1].data))  # image 1 is the whole image, image 2 the first crop
        sizes[px] = max(crop.size)
    assert sizes[128] <= 130 and sizes[384] > sizes[128]
