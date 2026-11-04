"""Shared model-call loop: send messages, parse the JSON reply, ask for repairs."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

from core.labeler.parse_reply import ReplyParseError
from core.labeler.transcript import Transcript
from core.providers.base import Message, ProviderReply, ToolSpec, VisionProvider

T = TypeVar("T")


class CostGuard:
    """Optional hook: called before each model call, may raise to stop the run."""
    def before_call(self) -> None: ...
    def after_call(self, reply: ProviderReply) -> None: ...


def call_model(provider: VisionProvider, messages: list[Message], transcript: Transcript,
               tools: list[ToolSpec] | None = None, guard: CostGuard | None = None,
               json_schema: dict[str, Any] | None = None) -> ProviderReply:
    """One provider call, recorded in the transcript."""
    if guard:
        guard.before_call()
    reply = provider.chat(messages, tools, json_schema) if json_schema else provider.chat(messages, tools)
    transcript.add_usage(reply.usage)
    if guard:
        guard.after_call(reply)
    assistant = reply.to_message()
    messages.append(assistant)
    transcript.add_message(assistant)
    return reply


def repair_loop(provider: VisionProvider, messages: list[Message], transcript: Transcript,
                first_reply: ProviderReply, parse: Callable[[str], T], max_repairs: int = 2,
                guard: CostGuard | None = None, json_schema: dict[str, Any] | None = None) -> tuple[T, int]:
    """Parse the reply; on failure tell the model what is wrong and retry.

    Returns (parsed, repairs_used). Raises ReplyParseError when repairs run out.
    """
    reply = first_reply
    for repairs in range(max_repairs + 1):
        try:
            return parse(reply.text), repairs
        except ReplyParseError as exc:
            transcript.add_event("invalid_reply", str(exc))
            if repairs == max_repairs:
                raise
            note = Message("user", f"Your reply could not be used: {exc}. Reply again with ONLY the corrected JSON object.")
            messages.append(note)
            transcript.add_message(note)
            reply = call_model(provider, messages, transcript, None, guard, json_schema)
    raise AssertionError("unreachable")
