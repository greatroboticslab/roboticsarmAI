import json
from pathlib import Path

import yaml
from PIL import Image

from cli import cmd_compare, cmd_label
from cli.__main__ import main
from core.labeler.golden import (format_report, load_truth_coco, normalize_name, score_image, score_run)
from core.models import ImageLabelSet, LaserBox, ObjectBox
from core.providers.check import check_provider, format_check
from core.schema.coco_export import build_coco
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text, tool
from tests.synth import finish, rect, scene, spot


def label_set(name="a.png", material="rubber", box=(100, 80, 80, 60)):
    return ImageLabelSet(
        "a", name, "cam", False, 0.1, status="approved", image_width=320, image_height=240,
        laser_boxes=[LaserBox("l1", "first_incidence", (130, 100, 20, 20), 0)],
        object_boxes=[ObjectBox("o1", box, "pen", material, "red", "first", ["l1"], confidence=0.9)])


def write_truth(tmp_path, ls, file_name="a.rf.abc123def.png"):
    coco = build_coco([ls])
    coco["images"][0]["file_name"] = file_name
    coco["categories"].insert(0, {"id": 0, "name": "group", "supercategory": "none"})
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(coco))
    return path


def test_normalize_roboflow_names():
    assert normalize_name("a.rf.0123abcd.jpg") == "a.jpg"
    assert normalize_name("my_image.rf.Ab9.png") == "my_image.png"
    assert normalize_name("plain.png") == "plain.png"


def test_perfect_prediction_scores_100(tmp_path):
    truth = load_truth_coco(write_truth(tmp_path, label_set()))
    assert list(truth) == ["a.png"] and len(truth["a.png"]) == 2  # group category skipped
    run = score_run("perfect", truth, [label_set()])
    assert run.laser.recall == run.laser.precision == run.object.f1 == 1.0
    assert run.object.mean_iou == 1.0 and run.object.agreement("material") == 1.0
    assert run.valid_reply_rate == 1.0


def test_wrong_material_and_shifted_box_are_scored(tmp_path):
    truth = load_truth_coco(write_truth(tmp_path, label_set()))
    wrong = label_set(material="glass")
    run = score_run("wrong", truth, [wrong])
    assert run.object.matched == 1 and run.object.agreement("material") == 0.0
    assert run.object.agreement("general_object") == 1.0
    far = score_image(truth["a.png"], label_set(box=(250, 150, 30, 30)))
    assert far["object"].matched == 0 and far["object"].recall == 0 and far["object"].precision == 0


def test_failed_images_count_and_report_renders(tmp_path):
    truth = load_truth_coco(write_truth(tmp_path, label_set()))
    failed = ImageLabelSet("a", "a.png", "cam", False, 0.1, status="failed")
    bad = score_run("bad", truth, [failed])
    good = score_run("good", truth, [label_set()])
    assert bad.failed == 1 and bad.valid_reply_rate == 0 and bad.laser.recall == 0
    report = format_report([good, bad])
    assert "| Measure | good | bad |" in report and "Object material agreement" in report


# ---------------- provider check
def test_check_provider_valid_and_tool_ok():
    provider = ScriptedProvider([text({"dot": True}), text('```json\n{"dot": true}\n```'), text("hmm"),
                                 tool("echo", word="ping")])
    result = check_provider(provider)
    assert (result.valid_json, result.attempts) == (2, 3) and result.tool_call_ok is True
    assert "tool calls works" in format_check("fake", result)


def test_check_provider_without_tools_recommends_pipeline():
    result = check_provider(ScriptedProvider([text({"dot": True})] * 3, supports_tools=False))
    assert result.tool_call_ok is None and "pipeline" in format_check("fake", result)


# ---------------- CLI
def make_image(path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    Image.fromarray(finish(img)).save(path)
    return path


def good():
    return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                "object": PEN_RUBBER}]))


def test_cli_detect(tmp_path, capsys):
    image = make_image(tmp_path / "a.png")
    assert main(["detect", str(image), "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "1 spot(s)" in out and "c1:" in out and (tmp_path / "o" / "a_detect.png").is_file()


def test_cli_camera_check(tmp_path, capsys):
    for cam in ("camA", "camB"):
        (tmp_path / cam).mkdir()
        for i in range(3):
            Image.fromarray(finish(scene(seed=i))).save(tmp_path / cam / f"{i}.png")
    assert main(["camera-check", str(tmp_path), "--camera-from", "folder"]) == 0
    out = capsys.readouterr().out
    assert "camA" in out and "camB" in out and "steady" in out
    assert main(["camera-check", str(tmp_path / "camA" / "nothing")]) == 1


def test_cli_label_runs_and_reports(tmp_path, monkeypatch, capsys):
    image = make_image(tmp_path / "a.png")
    provider = ScriptedProvider([good()])
    monkeypatch.setattr(cmd_label, "make_provider", lambda settings, overrides=None: provider)
    code = main(["label", str(image), "--data-dir", str(tmp_path / "d"), "--mode", "pipeline", "--tag", "camera:x"])
    assert code == 0 and "a.png: pending (pipeline)" in capsys.readouterr().out
    side = json.loads((tmp_path / "d" / "sidecars" / "a.json").read_text())
    assert side["camera_id"] == "x"


def test_cli_label_without_key_explains(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    from config import credentials
    monkeypatch.setattr(credentials, "get_key", lambda s: None)
    settings = tmp_path / "s.yaml"
    settings.write_text(yaml.safe_dump({"provider": {"provider": "gemini", "model": "m"}}))
    image = make_image(tmp_path / "a.png")
    assert main(["label", str(image), "--settings", str(settings), "--data-dir", str(tmp_path / "d")]) == 1
    err = capsys.readouterr().err
    assert "No Google Gemini key" in err and "Sign in tab" in err


def test_cli_compare_end_to_end(tmp_path, monkeypatch, capsys):
    images = tmp_path / "golden"
    images.mkdir()
    make_image(images / "a.png")
    ls = label_set()
    ls.object_boxes[0].box = (100, 80, 80, 60)
    truth = write_truth(tmp_path, ls)
    runs = tmp_path / "runs.yaml"
    runs.write_text(yaml.safe_dump({"runs": [{"name": "run-one", "mode": "pipeline"},
                                              {"name": "run-two", "mode": "agent"}]}))
    providers = iter([ScriptedProvider([good()]), ScriptedProvider([text("junk")] * 3 + [good()])])
    monkeypatch.setattr(cmd_compare, "make_provider", lambda settings, overrides=None: next(providers))
    monkeypatch.setattr(cmd_compare, "data_dir_for", lambda settings, override=None: tmp_path / "data")
    report = tmp_path / "report.md"
    code = main(["compare", "--images", str(images), "--truth", str(truth), "--runs", str(runs), "--out", str(report)])
    assert code == 0
    text_out = report.read_text()
    assert "run-one" in text_out and "run-two" in text_out and "Laser boxes: recall" in text_out
    assert "| Images | 1 | 1 |" in text_out
