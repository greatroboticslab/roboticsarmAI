import json

import pytest

from core.labeler.parse_reply import (ReplyParseError, extract_json, parse_agent_reply, parse_pipeline_reply)

LASER = {"id": "l1", "kind": "first_incidence", "box": [1, 2, 3, 4], "order": 0, "from_id": None}
OBJ = {"id": "o1", "box": [1, 2, 30, 40], "general_object": "pen", "material": "glass", "color": "clear", "incidence": "first", "laser_ids": ["l1"], "confidence": 0.7}


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 1,}\n```\nDone.') == {"a": 1}
    assert extract_json('here: {"a": {"b": [1, 2,]}} bye') == {"a": {"b": [1, 2]}}


@pytest.mark.parametrize("bad", ["", "   ", "no json here", "{broken", '{"a": }', "[1, 2]"])
def test_extract_json_errors(bad):
    with pytest.raises(ReplyParseError):
        extract_json(bad)


def test_agent_reply_ok():
    reply = parse_agent_reply(json.dumps({"laser_boxes": [LASER], "object_boxes": [OBJ],
                                          "questions": [{"text": "which?", "choices": ["a"]}]}))
    assert reply.laser_boxes[0]["kind"] == "first_incidence"
    assert reply.object_boxes[0]["confidence"] == 0.7
    assert reply.questions[0].id == "q1" and reply.questions[0].choices == ["a"]


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d["laser_boxes"][0].update(kind="bounce"), "kind"),
    (lambda d: d["laser_boxes"][0].update(box=[1, 2, 3]), "box"),
    (lambda d: d["laser_boxes"][0].update(box=[1, 2, 0, 4]), "positive"),
    (lambda d: d["object_boxes"][0].pop("material"), "material"),
    (lambda d: d["object_boxes"][0].update(confidence=1.5), "confidence"),
    (lambda d: d.pop("object_boxes"), "object_boxes"),
])
def test_agent_reply_rejections_name_the_problem(mutate, message):
    data = {"laser_boxes": [dict(LASER)], "object_boxes": [dict(OBJ)]}
    mutate(data)
    with pytest.raises(ReplyParseError, match=message):
        parse_agent_reply(json.dumps(data))


def cand(cid, **kw):
    return {"id": cid, "is_laser": True, "kind": "first_incidence", "order": 0, "object": None, **kw}


def test_pipeline_reply_requires_every_id_and_known_ids():
    ok = parse_pipeline_reply(json.dumps({"candidates": [cand("c1"), cand("c2", is_laser=False)]}), ["c1", "c2"])
    assert [c["id"] for c in ok.candidates] == ["c1", "c2"] and not ok.candidates[1]["is_laser"]
    with pytest.raises(ReplyParseError, match="c2"):
        parse_pipeline_reply(json.dumps({"candidates": [cand("c1")]}), ["c1", "c2"])
    with pytest.raises(ReplyParseError, match="unknown id"):
        parse_pipeline_reply(json.dumps({"candidates": [cand("c9")]}), ["c1"])
    with pytest.raises(ReplyParseError, match="kind"):
        parse_pipeline_reply(json.dumps({"candidates": [cand("c1", kind="zap")]}), ["c1"])
    with pytest.raises(ReplyParseError, match="material"):
        parse_pipeline_reply(json.dumps({"candidates": [cand("c1", object={"general_object": "a", "incidence": "first"})]}), ["c1"])
