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


def test_by_default_only_the_object_is_matched(image, settings, tmp_path):
    result, _ = run(image, settings, tmp_path, [good_reply(material="Unobtainium")])
    assert result.status != "needs_input" and result.object_boxes[0].material == "unobtainium"  # taken as said, made a token
    assert not any("taxonomy" in q.text for q in result.questions)


def test_an_object_off_the_list_still_asks_by_default(image, settings, tmp_path):
    result, _ = run(image, settings, tmp_path, [good_reply(general_object="Spaceship")])
    question = next(q for q in result.questions if "spaceship" in q.text)
    assert question.allow_taxonomy_add and result.object_boxes[0].general_object == "spaceship"


def test_value_outside_taxonomy_parks_a_question(image, settings, tmp_path):
    settings["labeling"]["taxonomy_match"] = "all"
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
    table = {"general_object": "table", "material": "wood", "finish": None,
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
         "object_boxes": [{"id": "o1", "box": [100, 80, 80, 60], "general_object": "pen", "material": "rubber", "color": "red", "incidence": "first", "laser_ids": ["l1"],
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
    settings["labeling"]["laser_box_object_fraction"] = 0  # this test is about coordinates, not box size
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


# ---------------- taxonomy match: off, and reusing a listed word
def test_match_entry_finds_the_listed_word():
    from core.schema.taxonomy import match_entry
    objects = ["pen", "cup", "bottle", "box"]
    assert match_entry("Pen", objects) == "pen" and match_entry("pens", objects) == "pen"
    assert match_entry("Ballpoint pen", objects) == "pen" and match_entry("tall glass bottles", objects) == "bottle"
    assert match_entry("boxes", objects) == "box"
    assert match_entry("pen holder", objects) is None  # the noun is "holder", not pen
    assert match_entry("", objects) is None and match_entry("spaceship", objects) is None


def test_off_adds_a_new_object_without_asking_and_reuses_it_next_time(image, settings, tmp_path):
    import copy
    tax, added = copy.deepcopy(TAX), []
    settings["labeling"]["taxonomy_match"] = "off"
    kwargs = dict(taxonomy=tax, settings=settings, data_dir=tmp_path / "data", mode="pipeline",
                  on_taxonomy_add=lambda kind, value: added.append((kind, value)))
    first = label_image(image, provider=ScriptedProvider([good_reply(general_object="Space Ship", material="Velvet")]),
                        **kwargs)
    assert first.status == "pending" and not first.questions
    assert (first.object_boxes[0].general_object, first.object_boxes[0].material) == ("spaceship", "velvet")
    assert added == [("object", "spaceship"), ("material", "velvet")] and "spaceship" in tax.objects
    again = label_image(image, provider=ScriptedProvider([good_reply(general_object="spaceships", material="velvet")]),
                        **{**kwargs, "image_id": "second"})
    assert again.object_boxes[0].general_object == "spaceship" and len(added) == 2  # reused, nothing new added
    assert "added object 'spaceship'" in Path(first.transcript_path).read_text()  # the conversation says what was added


def test_a_listed_word_is_reused_in_every_mode_but_all(image, settings, tmp_path):
    for mode in ("off", "object"):
        settings["labeling"]["taxonomy_match"] = mode
        result, _ = run(image, settings, tmp_path / mode, [good_reply(general_object="Ballpoint Pens")])
        assert result.object_boxes[0].general_object == "pen" and not any("taxonomy" in q.text for q in result.questions)


# ---------------- an object box must hold the laser spot that hits it
def test_an_object_box_that_misses_its_laser_spot_is_replaced_and_flagged(image, settings, tmp_path):
    far = {**FINAL, "object_boxes": [{**FINAL["object_boxes"][0], "box": [300, 220, 200, 30]}]}  # nowhere near the spot
    result, _ = run(image, settings, tmp_path, [tool("detect_laser"), text(far)], mode="agent")
    obj = result.object_boxes[0]
    laser = result.laser_boxes[0]
    cx, cy = laser.box[0] + laser.box[2] / 2, laser.box[1] + laser.box[3] / 2
    assert obj.box[0] <= cx <= obj.box[0] + obj.box[2] and obj.box[1] <= cy <= obj.box[1] + obj.box[3]
    assert obj.confidence <= 0.3 and "did not contain its laser spot" in obj.notes
    assert result.status == "needs_input" and any(q.target_id == obj.id for q in result.questions)


def test_a_box_holding_its_spot_is_left_alone(image, settings, tmp_path):
    settings["labeling"]["laser_box_object_fraction"] = 0
    result, _ = run(image, settings, tmp_path, [tool("detect_laser"), text(FINAL)], mode="agent")
    assert result.object_boxes[0].box == (100, 80, 80, 60) and "did not contain" not in result.object_boxes[0].notes


# ---------------- the model may name a new object
def test_structured_schema_only_limits_objects_when_everything_is_matched():
    from core.labeler.schemas import pipeline_schema
    def surface(strict):
        schema = pipeline_schema(["c1"], TAX, strict)
        return schema["properties"]["candidates"]["items"]["properties"]["object"]["anyOf"][1]["properties"]
    assert "enum" in surface(True)["general_object"] and "enum" in surface(True)["material"]
    assert surface(False)["general_object"] == {"type": "string"} and surface(False)["material"] == {"type": "string"}


def test_the_prompt_lets_the_model_name_a_new_object_unless_everything_is_matched():
    from core.prompts.manager import build_system_prompt
    strict = build_system_prompt("pipeline", TAX, taxonomy_match="all").text
    assert "Naming objects" not in strict
    for mode, sentence in (("off", "adds a new name to the project's lists by itself"),
                           ("object", "ask the reviewer to confirm a new name")):
        text_ = build_system_prompt("pipeline", TAX, taxonomy_match=mode).text
        assert "Naming objects" in text_ and sentence in text_ and "do not ask a question just because" in text_


def test_structured_run_sends_a_free_text_schema_when_not_matching_everything(image, settings, tmp_path):
    settings["provider"]["structured_output"] = True
    _, provider = run(image, settings, tmp_path, [good_reply(general_object="spaceship")])
    schema = provider.schemas[0]
    assert schema is not None
    assert "enum" not in str(schema["properties"]["candidates"]["items"]["properties"]["object"]["anyOf"][1]["properties"]["general_object"])


def test_off_mode_does_not_add_unknown_as_an_object_but_asks(image, settings, tmp_path):
    import copy
    tax, added = copy.deepcopy(TAX), []
    settings["labeling"]["taxonomy_match"] = "off"
    result = label_image(image, provider=ScriptedProvider([good_reply(general_object="unknown")]), taxonomy=tax,
                         settings=settings, data_dir=tmp_path / "data", mode="pipeline",
                         on_taxonomy_add=lambda k, v: added.append((k, v)))
    assert "unknown" not in tax.objects and ("object", "unknown") not in added
    assert result.status == "needs_input" and any("could not tell what this object is" in q.text for q in result.questions)


# ---------------- a reply cut off by the token limit is reported as that, not repaired
def test_a_truncated_reply_stops_with_a_clear_message_and_is_not_repaired(image, settings, tmp_path):
    from core.providers.base import ProviderReply, Usage
    cut = ProviderReply(text='{"status": "ok", "candidates": [{"id": "c1", "is_la', stop_reason="length",
                        usage=Usage(3500, 596))
    result, provider = run(image, settings, tmp_path, [cut, good_reply()])
    assert result.status == "failed"
    assert "ran out of tokens" in result.notes and "cut off after 596 tokens" in result.notes and "3500" in result.notes
    assert "Together that is 4096 tokens" in result.notes  # the sum is spelled out so the window size is obvious
    assert len(provider.calls) == 1  # no repair request that would only make the context longer


# ---------------- the request shrinks itself to fit a small context window
def _window_fake(limit_images_px, replies_ok, calls):
    """Provider that refuses (like Ollama) any request whose images are too many or too big."""
    from core.providers.base import ContextTooLong
    import io
    from PIL import Image as PILImage

    def make(messages):
        user = next(m for m in messages if m.role == "user")
        sizes = [max(PILImage.open(io.BytesIO(i.data)).size) for i in user.images]
        calls.append(sizes)
        tokens = 2000 + sum((s // 16) ** 2 for s in sizes)
        if tokens > limit_images_px:
            raise ContextTooLong(f"request ({tokens} tokens) exceeds the available context size ({limit_images_px} tokens)",
                                 tokens, limit_images_px)
        return replies_ok
    return make


def test_context_error_is_parsed_with_its_numbers():
    from core.providers.openai_compat import _context_error
    e = _context_error("BadRequestError: Error code: 400 - request (4948 tokens) exceeds the available context size "
                       "(4096 tokens), try increasing it", 400)
    assert (e.request_tokens, e.window) == (4948, 4096)
    e = _context_error("This model's maximum context length is 4096 tokens. However, your messages resulted in 5100 tokens.", 400)
    assert (e.request_tokens, e.window) == (5100, 4096)
    assert _context_error("something else", 400) is None


def test_shrink_steps_stop_at_a_floor():
    from core.labeler.fit import RequestSize, shrink
    size, steps = RequestSize(1024, 6, 384), 0
    while (nxt := shrink(size)) is not None:
        assert nxt != size and nxt.max_image_px <= size.max_image_px and nxt.crop_px <= size.crop_px
        size, steps = nxt, steps + 1
    assert size == RequestSize(384, 1, 128) and 2 <= steps <= 10


def test_an_oversized_request_is_retried_smaller_and_remembered(image, settings, tmp_path):
    from core.labeler.fit import FitMemory
    calls, fit = [], FitMemory()
    settings["labeling"]["laser_box_object_fraction"] = 0
    ok = good_reply()
    provider = ScriptedProvider([_window_fake(2600, ok, calls)] * 8)
    result, _ = label_image_unsaved_for_test(image, settings, tmp_path, provider, fit)
    assert result.status != "failed" and len(calls) >= 2 and calls[-1] != calls[0]
    assert "did not fit the model's context window" in result.notes and "shrunk to" in result.notes
    assert fit.size is not None and fit.size.max_image_px <= 1024
    before = len(calls)
    second, _ = label_image_unsaved_for_test(image, settings, tmp_path / "again", provider, fit)
    assert second.status != "failed" and len(calls) == before + 1  # the second image went straight in at the size that fitted


def test_auto_fit_can_be_turned_off_and_a_floor_still_fails_clearly(image, settings, tmp_path):
    settings["run"]["auto_fit_context"] = False
    calls = []
    result, _ = label_image_unsaved_for_test(image, settings, tmp_path,
                                             ScriptedProvider([_window_fake(100, good_reply(), calls)] * 4), None)
    assert result.status == "failed" and len(calls) == 1 and "context size" in result.notes
    settings["run"]["auto_fit_context"] = True
    calls.clear()
    result, _ = label_image_unsaved_for_test(image, settings, tmp_path / "b",
                                             ScriptedProvider([_window_fake(100, good_reply(), calls)] * 12), None)
    assert result.status == "failed" and 2 <= len(calls) <= 8  # it tried smaller sizes, then gave up


def label_image_unsaved_for_test(image, settings, tmp_path, provider, fit):
    from core.labeler.label_image import label_image_unsaved
    return label_image_unsaved(image, provider=provider, taxonomy=TAX, settings=settings, mode="pipeline", fit=fit)
