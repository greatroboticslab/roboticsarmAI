"""One object name per box: no sub-object in class names, prompts, taxonomy or the editor."""
import json
from pathlib import Path

import pytest
from PIL import Image

from config.settings import load_settings
from core.labeler.label_image import label_image
from core.models import ImageLabelSet
from core.prompts.manager import build_system_prompt
from core.schema.coco_export import class_names_for
from core.schema.taxonomy import load_taxonomy
from tests.fakes import ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot

TAX = load_taxonomy(Path(__file__).resolve().parent.parent / "taxonomy")


@pytest.fixture
def image(tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / "cam1" / "red_object.png"
    path.parent.mkdir()
    Image.fromarray(finish(img)).save(path)
    return path


def label(image, tmp_path, obj):
    reply = text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                 "from_id": None, "object": obj}]))
    return label_image(image, provider=ScriptedProvider([reply]), taxonomy=TAX, settings=load_settings(),
                       data_dir=tmp_path / "data")


def test_class_name_is_object_material_color_incidence(image, tmp_path):
    result = label(image, tmp_path, {"general_object": "pen", "material": "rubber", "finish": None,
                                     "incidence": "first", "confidence": 0.9})
    names = [n for n, _ in class_names_for(result)]
    assert any(n.startswith("obj_pen_rubber_") and n.endswith("_first") for n in names), names


def test_a_reply_that_still_has_a_sub_object_is_accepted_and_the_sub_object_ignored(image, tmp_path):
    result = label(image, tmp_path, {"general_object": "pen", "sub_object": "ballpoint", "material": "rubber",
                                     "finish": None, "incidence": "first", "confidence": 0.9})
    assert not result.questions or all("sub" not in q.text.lower() for q in result.questions)
    assert not any("ballpoint" in n for n, _ in class_names_for(result))


def test_old_sidecars_with_a_sub_object_still_load():
    data = {"image_id": "i", "image_name": "a.png", "camera_id": "c", "is_grayscale": False, "background_level": 0.1,
            "object_boxes": [{"id": "o1", "box": [1, 2, 3, 4], "general_object": "pen", "sub_object": "gel",
                              "material": "glass", "color": "red", "incidence": "first", "laser_ids": [],
                              "finish": None, "confidence": 0.5, "notes": ""}]}
    ls = ImageLabelSet.from_dict(json.loads(json.dumps(data)))
    assert ls.object_boxes[0].general_object == "pen" and not hasattr(ls.object_boxes[0], "sub_object")


def test_prompt_lists_objects_without_sub_objects():
    built = build_system_prompt("pipeline", TAX, table_rule="only_if_no_object_hit", color_placeholder="na")
    assert "Objects: pen, table" in built.text
    assert "sub-object" not in built.text.lower() and "sub_object" not in built.text


def test_editor_has_no_sub_object_field(tmp_path):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    from gui.widgets.box_editor import BoxEditor
    app = QApplication.instance() or QApplication([])
    editor = BoxEditor()
    assert not hasattr(editor, "sub")
    app.processEvents()
