"""Label scheme "merged": each laser part box is named after the object it lands on."""
import json
from pathlib import Path

import pytest
from PIL import Image

from config.settings import SettingsError, load_settings, validate_settings
from core.labeler.golden import TruthBox, score_image
from core.labeler.label_image import label_image
from core.models import ImageLabelSet, LaserBox, ObjectBox
from core.reports.measurements_csv import rows_for
from core.schema.class_names import (ClassNameError, ComponentClass, MergedClass, ObjectClass, class_family,
                                     decode_class, encode_merged_class)
from core.schema.coco_export import build_coco, class_names_for
from core.schema.taxonomy import load_taxonomy
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot

TAX = load_taxonomy(Path(__file__).resolve().parent.parent / "taxonomy")
OBJ = "obj_pen_rubber_red_first"


def label_set(scheme="merged", with_object=True):
    lasers = [LaserBox("l1", "first_incidence", (100, 100, 20, 20), 0, component="component1"),
              LaserBox("l2", "first_incidence", (90, 90, 40, 40), 0, from_id="l1", component="component2")]
    objects = [ObjectBox("o1", (60, 60, 100, 80), "pen", "rubber", "red", "first", ["l1"], None, 0.9, "")]
    return ImageLabelSet("i1", "a.png", "cam", False, 0.1, laser_boxes=lasers,
                         object_boxes=objects if with_object else [], status="approved",
                         image_width=300, image_height=300, scheme=scheme)


# ------------------------------------------------------------------ class names
def test_merged_names_round_trip():
    name = encode_merged_class(OBJ, "component1")
    assert name == "obj_pen_rubber_red_first__component1"
    got = decode_class(name)
    assert got == MergedClass(ObjectClass("pen", "rubber", "red", "first"), ComponentClass(1))
    assert class_family(name) == "laser"


@pytest.mark.parametrize("bad", ["las_first-incidence__component1", "obj_pen_rubber_red_first__component9",
                                 "component1__component2", "obj_pen_rubber_red__component1"])
def test_bad_merged_names_are_refused(bad):
    with pytest.raises(ClassNameError):
        decode_class(bad)


def test_encoding_needs_a_valid_object_class_and_component():
    with pytest.raises(ClassNameError):
        encode_merged_class("nonsense", "component1")
    with pytest.raises(ClassNameError):
        encode_merged_class(OBJ, "halo")


# ------------------------------------------------------------------ label set naming and export
def test_parts_are_named_after_their_object_and_the_object_box_is_not_exported_twice():
    names = [n for n, _ in class_names_for(label_set())]
    assert names == [f"{OBJ}__component1", f"{OBJ}__component2"]  # the halo finds its object through its core


def test_other_schemes_are_unchanged():
    names = [n for n, _ in class_names_for(label_set("components"))]
    assert names == ["component1", "component2", OBJ]


def test_a_part_with_no_object_keeps_its_plain_name_and_an_unclaimed_object_is_kept():
    names = [n for n, _ in class_names_for(label_set(with_object=False))]
    assert names == ["component1", "component2"]
    ls = label_set()
    ls.object_boxes[0].laser_ids = []  # an object nobody hits
    assert [n for n, _ in class_names_for(ls)] == ["component1", "component2", OBJ]


def test_grayscale_objects_use_the_color_placeholder():
    ls = label_set()
    ls.is_grayscale = True
    assert class_names_for(ls, "na")[0][0] == "obj_pen_rubber_na_first__component1"


def test_editing_the_object_changes_the_exported_names():
    ls = label_set()
    ls.object_boxes[0].material = "metal"
    assert class_names_for(ls)[0][0] == "obj_pen_metal_red_first__component1"


def test_coco_has_only_merged_categories():
    coco = build_coco([label_set()])
    assert sorted(c["name"] for c in coco["categories"]) == [f"{OBJ}__component1", f"{OBJ}__component2"]
    assert len(coco["annotations"]) == 2


def test_csv_rows_use_the_merged_class():
    assert [r["class"] for r in rows_for(label_set())] == [f"{OBJ}__component1", f"{OBJ}__component2"]


def test_old_sidecars_without_a_scheme_still_load():
    data = label_set("full").to_dict()
    data.pop("scheme")
    assert ImageLabelSet.from_dict(json.loads(json.dumps(data))).scheme == "full"


# ------------------------------------------------------------------ scoring against Roboflow ground truth
def test_golden_scoring_checks_the_object_part_of_a_merged_name():
    truth = [TruthBox(f"{OBJ}__component1", (100, 100, 20, 20)),
             TruthBox(f"{OBJ}__component2", (90, 90, 40, 40))]
    scores = score_image(truth, label_set())
    assert scores["laser"].matched == 2 and scores["laser"].agree["material"] == 2
    wrong = [TruthBox("obj_pen_metal_red_first__component1", (100, 100, 20, 20))]
    scores = score_image(wrong, label_set())
    assert scores["laser"].agree["event"] == 1 and scores["laser"].agree["material"] == 0


# ------------------------------------------------------------------ settings and end to end
def test_setting_is_accepted_and_junk_refused():
    s = load_settings()
    s["labeling"]["scheme"] = "merged"
    validate_settings(s)
    s["labeling"]["scheme"] = "both"
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


def test_label_image_with_the_merged_scheme(image, tmp_path):
    settings = load_settings()
    settings["labeling"]["scheme"] = "merged"
    reply = text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                 "from_id": None, "object": PEN_RUBBER}]))
    result = label_image(image, provider=ScriptedProvider([reply]), taxonomy=TAX, settings=settings,
                         data_dir=tmp_path / "data")
    assert result.scheme == "merged"
    names = [n for n, _ in class_names_for(result)]
    assert names and all("__component" in n for n in names), names
    assert any(n.endswith("__component1") for n in names)
    assert not any(n.startswith("obj_") and "__" not in n for n in names)
