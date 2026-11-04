import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from config.settings import load_settings
from core.labeler.guard import TrackerGuard
from core.labeler.label_image import label_image
from core.models import ImageLabelSet
from core.providers.base import ProviderError, ProviderReply, Usage
from core.providers.cost import CostTracker, SpendCapExceeded
from core.reports.sidecar import read_sidecar
from core.schema.taxonomy import load_taxonomy
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text, tool
from tests.synth import finish, rect, scene, spot

TAX = load_taxonomy(Path(__file__).resolve().parent.parent / "taxonomy")


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def image(tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / "cam1" / "red_object.png"
    path.parent.mkdir()
    Image.fromarray(finish(img)).save(path)
    return path


def run(image, settings, tmp_path, replies, mode="pipeline", **kw):
    provider = kw.pop("provider", None) or ScriptedProvider(replies)
    result = label_image(image, provider=provider, taxonomy=TAX, settings=settings, data_dir=tmp_path / "data",
                         mode=mode, **kw)
    return result, provider


def good_reply(**obj_changes):
    obj = {**PEN_RUBBER, **obj_changes}
    return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                "from_id": None, "object": obj}]))


# ---------------- pipeline mode
def test_pipeline_happy_path(image, settings, tmp_path):
    result, provider = run(image, settings, tmp_path, [good_reply()])
    assert result.status == "pending" and result.run_mode == "pipeline"
    assert len(provider.calls) == 1 and provider.tool_specs == [None]
    laser, = result.laser_boxes
    assert laser.kind == "first_incidence" and laser.metrics["core_area_px"] > 3
    assert laser.metrics["reflectivity"] == 1.0 and "color_r" in laser.metrics
    obj, = result.object_boxes
    assert (obj.general_object, obj.material, obj.color) == ("pen", "rubber", "red")  # color from pixels
    assert abs(obj.box[0] - 100) <= 3 and abs(obj.box[2] - 80) <= 6  # tightened to the red region
    assert obj.laser_ids == [laser.id]
    assert result.camera_id == "unknown" and result.provider == "fake" and result.prompt_version.startswith("draft-")
    # the overview and the one crop were both sent as images
    assert len(provider.calls[0][1].images) == 2


def test_outputs_written(image, settings, tmp_path):
    result, _ = run(image, settings, tmp_path, [good_reply()])
    data = tmp_path / "data"
    assert read_sidecar(data / "sidecars" / "red_object.json") == result
    rows = list(csv.DictReader((data / "measurements.csv").open()))
    assert len(rows) == 1 and rows[0]["class"] == "las_first-incidence" and float(rows[0]["peak_intensity"]) > 0.5
    transcript = json.loads(Path(result.transcript_path).read_text())
    assert [e["role"] for e in transcript["entries"] if e["type"] == "message"] == ["system", "user", "assistant"]
    assert (data / "transcripts" / "red_object" / "img_001.png").is_file()
    assert transcript["usage"]["input_tokens"] == 100


def test_relabeling_replaces_csv_rows(image, settings, tmp_path):
    run(image, settings, tmp_path, [good_reply()])
    run(image, settings, tmp_path, [good_reply()])
    assert len(list(csv.DictReader((tmp_path / "data" / "measurements.csv").open()))) == 1


def test_camera_from_tag(image, settings, tmp_path):
    result, _ = run(image, settings, tmp_path, [good_reply()], tags=["split:train", "camera:logitech"])
    assert result.camera_id == "logitech"


def test_invalid_json_is_repaired(image, settings, tmp_path):
    result, provider = run(image, settings, tmp_path, [text("I think it is a pen"), good_reply()])
    assert result.status == "pending" and "1 repair" in result.notes
    assert "could not be used" in provider.calls[1][-1].text


def test_unrepairable_reply_gives_failed_status_not_exception(image, settings, tmp_path):
    result, _ = run(image, settings, tmp_path, [text("nope")] * 3)
    assert result.status == "failed" and "failed" in result.notes
    assert (tmp_path / "data" / "sidecars" / "red_object.json").is_file()


def test_provider_error_gives_failed_status(image, settings, tmp_path):
    def boom(_):
        raise ProviderError("server down")
    result, _ = run(image, settings, tmp_path, [boom])
    assert result.status == "failed" and "server down" in result.notes


def test_value_outside_taxonomy_parks_a_question(image, settings, tmp_path):
    result, _ = run(image, settings, tmp_path, [good_reply(material="unobtainium")])
    assert result.status == "needs_input"
    question = next(q for q in result.questions if "unobtainium" in q.text)
    assert question.allow_taxonomy_add and question.target_id == result.object_boxes[0].id


def test_low_confidence_parks_a_question(image, settings, tmp_path):
    result, _ = run(image, settings, tmp_path, [good_reply(confidence=0.1)])
    assert result.status == "needs_input" and any("Low confidence" in q.text for q in result.questions)


def test_model_question_is_kept(image, settings, tmp_path):
    reply = text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                 "object": PEN_RUBBER}], [{"text": "Matte or glossy?", "target_id": "c1"}]))
    result, _ = run(image, settings, tmp_path, [reply])
    assert result.status == "needs_input" and result.questions[0].text == "Matte or glossy?"


def test_no_laser_skips_the_model_and_asks_user(settings, tmp_path):
    path = tmp_path / "dark.png"
    Image.fromarray(finish(scene())).save(path)
    result, provider = run(path, settings, tmp_path, [])
    assert provider.calls == [] and result.status == "needs_input"
    assert result.questions[0].id == "q_no_laser" and result.laser_boxes == []


def test_candidate_rejected_as_not_laser(image, settings, tmp_path):
    reply = text(pipeline_json([{"id": "c1", "is_laser": False, "kind": None, "order": None, "object": None}]))
    result, _ = run(image, settings, tmp_path, [reply])
    assert result.laser_boxes == [] and result.status == "needs_input" and "c1" in result.notes


def test_grayscale_color_is_na(settings, tmp_path):
    img = scene(gray=True)
    spot(img, 140, 110, sigma=2.5, amp=0.6, color=(1, 1, 1))
    path = tmp_path / "g.png"
    Image.fromarray(finish(img)[..., 0]).save(path)
    result, _ = run(path, settings, tmp_path, [good_reply()])
    assert result.is_grayscale and result.camera_id == "grayscale"
    assert result.object_boxes[0].color == "na" and "gray_level" in result.laser_boxes[0].metrics


def test_table_rule_drops_table_when_object_is_hit(image, settings, tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 120, 100, sigma=2.5, amp=0.5, color=(1, 1, 1))
    spot(img, 250, 200, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / "two.png"
    Image.fromarray(finish(img)).save(path)
    table = {"general_object": "table", "sub_object": "desk", "material": "wood", "finish": None,
             "incidence": "reflected", "confidence": 0.8}
    reply = text(pipeline_json([
        {"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0, "object": PEN_RUBBER},
        {"id": "c2", "is_laser": True, "kind": "reflection", "order": 1, "from_id": "c1", "object": table}]))
    result, _ = run(path, settings, tmp_path, [reply])
    assert len(result.laser_boxes) == 2
    assert [o.general_object for o in result.object_boxes] == ["pen"]
    assert result.laser_boxes[1].from_id == result.laser_boxes[0].id
    assert result.laser_boxes[1].metrics["reflectivity"] <= 1.0 and "angle_deg" in result.laser_boxes[1].metrics


# ---------------- agent mode
FINAL = {"laser_boxes": [{"id": "l1", "kind": "first_incidence", "box": [130, 100, 20, 20], "order": 0}],
         "object_boxes": [{"id": "o1", "box": [100, 80, 80, 60], "general_object": "pen", "sub_object": "ballpoint",
                           "material": "rubber", "color": "red", "incidence": "first", "laser_ids": ["l1"],
                           "confidence": 0.9}], "questions": []}


def test_agent_mode_uses_tools_then_answers(image, settings, tmp_path):
    replies = [tool("detect_laser", "a"), tool("crop_zoom", "b", box=[120, 90, 40, 40]), text(FINAL)]
    result, provider = run(image, settings, tmp_path, replies, mode="agent")
    assert result.status == "pending" and result.run_mode == "agent" and len(provider.calls) == 3
    assert provider.tool_specs[0] and len(provider.tool_specs[0]) == 7
    tool_msg = provider.calls[2][-1]
    assert tool_msg.role == "tool" and tool_msg.tool_call_id == "b" and tool_msg.images  # crop came back
    assert result.laser_boxes[0].metrics["core_area_px"] > 3  # measured by the app


def test_agent_coordinates_are_scaled_back_to_original(image, settings, tmp_path):
    settings["provider"]["max_image_px"] = 160  # the model sees half size
    shown = {**FINAL, "laser_boxes": [{**FINAL["laser_boxes"][0], "box": [65, 50, 10, 10]}],
             "object_boxes": [{**FINAL["object_boxes"][0], "box": [50, 40, 40, 30]}]}
    result, _ = run(image, settings, tmp_path, [tool("detect_laser"), text(shown)], mode="agent")
    assert result.laser_boxes[0].box == (130, 100, 20, 20) and result.object_boxes[0].box == (100, 80, 80, 60)


def test_agent_tool_questions_are_kept(image, settings, tmp_path):
    replies = [tool("ask_user", question="Is the pen glossy?"), text(FINAL)]
    result, _ = run(image, settings, tmp_path, replies, mode="agent")
    assert result.status == "needs_input" and result.questions[0].text == "Is the pen glossy?"


def test_agent_answering_without_tools_falls_back_to_pipeline(image, settings, tmp_path):
    result, provider = run(image, settings, tmp_path, [text(FINAL), good_reply()], mode="agent")
    assert result.run_mode == "pipeline" and len(provider.calls) == 2
    events = [e["text"] for e in json.loads(Path(result.transcript_path).read_text())["entries"]
              if e["type"] == "fallback"]
    assert events and "without calling any tool" in events[0]


def test_agent_with_bad_tool_calls_falls_back(image, settings, tmp_path):
    replies = [tool("crop_zoom", str(i)) for i in range(4)] + [good_reply()]
    result, _ = run(image, settings, tmp_path, replies, mode="agent")
    assert result.run_mode == "pipeline" and result.status == "pending"


def test_agent_that_never_finishes_falls_back(image, settings, tmp_path):
    settings["run"]["max_agent_steps"] = 3
    replies = [tool("detect_laser", str(i)) for i in range(3)] + [good_reply()]
    result, _ = run(image, settings, tmp_path, replies, mode="agent")
    assert result.run_mode == "pipeline"


def test_provider_without_tools_uses_pipeline_even_if_agent_requested(image, settings, tmp_path):
    provider = ScriptedProvider([good_reply()], supports_tools=False)
    result, _ = run(image, settings, tmp_path, [], mode="agent", provider=provider)
    assert result.run_mode == "pipeline" and provider.tool_specs == [None]


# ---------------- spend cap
def test_spend_cap_stops_the_run(image, settings, tmp_path):
    tracker = CostTracker(price_in_per_mtok=1_000_000, price_out_per_mtok=0, daily_cap_usd=50.0)
    guard = TrackerGuard(tracker)
    run(image, settings, tmp_path, [good_reply()], guard=guard)  # 100 tokens = $100 > cap
    with pytest.raises(SpendCapExceeded):
        run(image, settings, tmp_path, [good_reply()], guard=guard)
