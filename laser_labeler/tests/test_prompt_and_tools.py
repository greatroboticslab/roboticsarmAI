import json
from pathlib import Path

import pytest
from PIL import Image

from core.camera import load_image
from core.labeler.context import build_context
from core.labeler.tool_dispatch import TOOL_SPECS, run_tool
from core.prompts.manager import build_system_prompt, list_parts
from core.providers.base import ToolCall
from core.schema.taxonomy import load_taxonomy
from tests.synth import finish, rect, scene, spot

TAX = load_taxonomy(Path(__file__).resolve().parent.parent / "taxonomy")


def test_prompt_modes_pick_the_right_parts():
    agent = [p.name for p in list_parts("agent")]
    pipe = [p.name for p in list_parts("pipeline")]
    assert "04_tools.agent.md" in agent and "04_tools.agent.md" not in pipe
    assert "06_output_format.pipeline.md" in pipe and "06_output_format.pipeline.md" not in agent
    assert "01_role_task.md" in agent and "01_role_task.md" in pipe
    with pytest.raises(ValueError):
        list_parts("bogus")


def test_prompt_fills_placeholders_and_versions_by_content(tmp_path):
    built = build_system_prompt("agent", TAX, table_rule="only_if_no_object_hit", color_placeholder="na")
    assert "{{" not in built.text
    assert "pen: ballpoint" in built.text and "only_if_no_object_hit" in built.text
    assert built.version.startswith("draft-")
    assert build_system_prompt("agent", TAX).version == built.version  # stable
    parts = tmp_path / "parts"
    parts.mkdir()
    (parts / "01.md").write_text("hello {{table_rule}}")
    a = build_system_prompt("pipeline", TAX, parts_dir=parts, table_rule="x")
    (parts / "01.md").write_text("hello again {{table_rule}}")
    b = build_system_prompt("pipeline", TAX, parts_dir=parts, table_rule="x")
    assert a.text == "hello x" and a.version != b.version
    with pytest.raises(FileNotFoundError):
        build_system_prompt("pipeline", TAX, parts_dir=tmp_path / "empty")


def make_ctx(tmp_path, max_px=1024):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    Image.fromarray(finish(img)).save(tmp_path / "a.png")
    return build_context(load_image(tmp_path / "a.png"), TAX, max_image_px=max_px)


def call(name, **args):
    return ToolCall("t1", name, args)


def test_every_tool_spec_has_a_handler_and_schema():
    names = [t.name for t in TOOL_SPECS]
    assert len(names) == len(set(names)) == 7
    assert all(t.parameters["type"] == "object" for t in TOOL_SPECS)


def test_detect_laser_reports_shown_coordinates(tmp_path):
    ctx = make_ctx(tmp_path, max_px=160)  # half size
    data = json.loads(run_tool(ctx, call("detect_laser")).text)
    assert data["image_size"] == [160, 120]
    x, y, w, h = data["candidates"][0]["box"]
    assert abs(x + w / 2 - 70) < 4 and abs(y + h / 2 - 55) < 4  # (140,110) halved


def test_box_tools(tmp_path):
    ctx = make_ctx(tmp_path)
    crop = run_tool(ctx, call("crop_zoom", box=[120, 90, 40, 40]))
    assert crop.images and crop.images[0].data[:4] == b"\x89PNG"
    assert json.loads(run_tool(ctx, call("sample_color", box=[105, 85, 20, 20])).text)["color"] == "red"
    tight = json.loads(run_tool(ctx, call("tighten_box", box=[120, 90, 40, 40])).text)
    assert tight["box"][2] >= 70
    metrics = json.loads(run_tool(ctx, call("measure_laser", box=[130, 100, 20, 20])).text)
    assert metrics["core_area_px"] > 3
    assert "materials" in json.loads(run_tool(ctx, call("lookup_taxonomy", what="materials")).text)


def test_tool_errors_are_text_not_exceptions(tmp_path):
    ctx = make_ctx(tmp_path)
    for bad in (call("crop_zoom"), call("crop_zoom", box=[1, 2, 3]), call("crop_zoom", box=[1, 2, 0, 5]),
                call("lookup_taxonomy", what="x"), call("ask_user"), call("nope")):
        result = run_tool(ctx, bad)
        assert result.is_error and result.text.startswith("error:")
    assert run_tool(ctx, ToolCall("t", "detect_laser", {"_raw": "{bad"})).is_error


def test_ask_user_parks_a_question(tmp_path):
    ctx = make_ctx(tmp_path)
    result = run_tool(ctx, call("ask_user", question="Is this glass?", target_id="o1", choices=["yes", "no"]))
    assert not result.is_error and ctx.questions[0].choices == ["yes", "no"]
