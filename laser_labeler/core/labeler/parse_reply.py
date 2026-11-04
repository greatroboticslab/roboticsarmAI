"""Strict JSON parsing and light repair of model replies (spec section 7, item 6).

Repair is deliberately small: strip a code fence, cut to the outermost braces,
drop trailing commas. Anything else is reported back to the model so it can
fix its own answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from core.models import LASER_KINDS, Question


class ReplyParseError(ValueError):
    """The reply is not usable. The message is safe to show to the model."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def extract_json(text: str) -> dict[str, Any]:
    """Return the JSON object inside a reply, tolerating fences and chatter around it."""
    if not text or not text.strip():
        raise ReplyParseError("the reply was empty; reply with one JSON object")
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ReplyParseError("no JSON object found; reply with only one JSON object")
    candidate = body[start:end + 1]
    for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            data = json.loads(attempt)
        except json.JSONDecodeError as exc:
            error = exc
            continue
        if not isinstance(data, dict):
            raise ReplyParseError("the JSON must be an object {...}")
        return data
    raise ReplyParseError(f"invalid JSON ({error.msg} at position {error.pos})")


def _need(item: Any, key: str, kind: type | tuple, where: str) -> Any:
    if not isinstance(item, dict) or key not in item:
        raise ReplyParseError(f"{where}: missing '{key}'")
    value = item[key]
    if not isinstance(value, kind) or isinstance(value, bool) and kind is not bool:
        raise ReplyParseError(f"{where}: '{key}' has the wrong type")
    return value


def _box(item: dict, where: str) -> list[float]:
    box = _need(item, "box", list, where)
    if len(box) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
        raise ReplyParseError(f"{where}: 'box' must be [x, y, w, h] numbers")
    if box[2] <= 0 or box[3] <= 0:
        raise ReplyParseError(f"{where}: box width and height must be positive")
    return [float(v) for v in box]


def _questions(data: dict) -> list[Question]:
    out = []
    for i, raw in enumerate(data.get("questions") or []):
        where = f"questions[{i}]"
        text = _need(raw, "text", str, where)
        out.append(Question(
            id=str(raw.get("id") or f"q{i + 1}"), text=text,
            target_id=raw.get("target_id"), field_name=raw.get("field"),
            choices=[str(c) for c in raw.get("choices") or []],
            allow_taxonomy_add=bool(raw.get("allow_taxonomy_add", False)),
        ))
    return out


def _laser(item: dict, i: int) -> dict:
    where = f"laser_boxes[{i}]"
    kind = _need(item, "kind", str, where)
    if kind not in LASER_KINDS:
        raise ReplyParseError(f"{where}: kind '{kind}' must be one of {', '.join(LASER_KINDS)}")
    return {"id": str(_need(item, "id", (str, int), where)), "kind": kind, "box": _box(item, where),
            "order": int(item.get("order", i)), "from_id": item.get("from_id"), "notes": str(item.get("notes", ""))}


def _object(item: dict, i: int) -> dict:
    where = f"object_boxes[{i}]"
    out = {"id": str(_need(item, "id", (str, int), where)), "box": _box(item, where)}
    for key in ("general_object", "sub_object", "material", "incidence"):
        out[key] = str(_need(item, key, str, where))
    out.update(color=item.get("color"), finish=item.get("finish"), notes=str(item.get("notes", "")),
               laser_ids=[str(v) for v in item.get("laser_ids") or []],
               confidence=_confidence(item.get("confidence"), where))
    return out


def _confidence(value: Any, where: str) -> float:
    if value is None:
        return 0.5
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise ReplyParseError(f"{where}: 'confidence' must be a number from 0 to 1")
    return float(value)


@dataclass
class AgentReply:
    laser_boxes: list[dict] = field(default_factory=list)
    object_boxes: list[dict] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)


@dataclass
class PipelineReply:
    candidates: list[dict] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)


def parse_agent_reply(text: str) -> AgentReply:
    data = extract_json(text)
    lasers = _need(data, "laser_boxes", list, "reply")
    objects = _need(data, "object_boxes", list, "reply")
    return AgentReply([_laser(x, i) for i, x in enumerate(lasers)],
                      [_object(x, i) for i, x in enumerate(objects)], _questions(data))


def parse_pipeline_reply(text: str, expected_ids: list[str]) -> PipelineReply:
    data = extract_json(text)
    candidates = _need(data, "candidates", list, "reply")
    seen: dict[str, dict] = {}
    for i, item in enumerate(candidates):
        where = f"candidates[{i}]"
        cid = str(_need(item, "id", (str, int), where))
        if cid not in expected_ids:
            raise ReplyParseError(f"{where}: unknown id '{cid}'; ids are {', '.join(expected_ids)}")
        is_laser = item.get("is_laser", True)
        if not isinstance(is_laser, bool):
            raise ReplyParseError(f"{where}: 'is_laser' must be true or false")
        kind = item.get("kind") or "other"
        if is_laser and kind not in LASER_KINDS:
            raise ReplyParseError(f"{where}: kind '{kind}' must be one of {', '.join(LASER_KINDS)}")
        obj = item.get("object")
        if obj is not None:
            if not isinstance(obj, dict):
                raise ReplyParseError(f"{where}: 'object' must be an object or null")
            for key in ("general_object", "sub_object", "material", "incidence"):
                _need(obj, key, str, f"{where}.object")
            _confidence(obj.get("confidence"), f"{where}.object")
        seen[cid] = {"id": cid, "is_laser": is_laser, "kind": kind, "order": item.get("order"),
                     "from_id": item.get("from_id"), "object": obj, "notes": str(item.get("notes", ""))}
    missing = [c for c in expected_ids if c not in seen]
    if missing:
        raise ReplyParseError(f"candidates missing for: {', '.join(missing)}; include every id")
    return PipelineReply([seen[c] for c in expected_ids], _questions(data))
