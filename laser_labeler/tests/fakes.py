"""Scripted stand-in for a model provider: no network, fully deterministic."""
from __future__ import annotations

import json

from core.providers.base import Message, ProviderReply, ToolCall, ToolSpec, Usage


class ScriptedProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, replies, supports_tools=True):
        self.replies = list(replies)
        self.supports_tools = supports_tools
        self.calls: list[list[Message]] = []
        self.tool_specs: list = []
        self.schemas: list = []

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             json_schema: dict | None = None) -> ProviderReply:
        self.schemas.append(json_schema)
        self.calls.append(list(messages))
        self.tool_specs.append(tools)
        if not self.replies:
            raise AssertionError("ScriptedProvider ran out of replies")
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply


def text(data) -> ProviderReply:
    body = data if isinstance(data, str) else json.dumps(data)
    return ProviderReply(text=body, usage=Usage(100, 50))


def tool(name: str, call_id: str = "t1", **arguments) -> ProviderReply:
    return ProviderReply(tool_calls=[ToolCall(call_id, name, arguments)], usage=Usage(100, 20))


def pipeline_json(candidates, questions=None, status="ok"):
    return {"status": status, "candidates": candidates, "questions": questions or []}


PEN_RUBBER = {"general_object": "pen", "material": "rubber",
              "finish": None, "incidence": "first", "confidence": 0.8}
