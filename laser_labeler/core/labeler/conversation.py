"""Shared model-call loop: send messages, parse the JSON reply, ask for repairs."""
from __future__ import annotations

from typing import Any, Callable, TypeVar

from core.labeler.parse_reply import ReplyParseError
from core.labeler.transcript import Transcript
from core.providers.base import Message, ProviderError, ProviderReply, ToolSpec, VisionProvider

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


LENGTH_STOPS = {"length", "max_tokens", "max_output_tokens"}  # OpenAI-style, Anthropic, Gemini (lowercased)


class ReplyTruncated(ProviderError):
    """The reply was cut off because the model ran out of tokens. Asking again would only make the request longer."""


def _truncation_message(reply: ProviderReply) -> str:
    u = reply.usage
    total = u.input_tokens + u.output_tokens
    return (f"The model ran out of tokens: its reply was cut off after {u.output_tokens} tokens, with the request "
            f"at {u.input_tokens} tokens. Together that is {total} tokens, which is where the context window ended "
            f"(if the reply is exactly your 'Longest reply allowed' number, that limit stopped it instead).")


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
            if str(reply.stop_reason or "").lower() in LENGTH_STOPS:
                transcript.add_event("truncated", _truncation_message(reply))
                raise ReplyTruncated(_truncation_message(reply)) from exc
            if repairs == max_repairs:
                raise
            note = Message("user", f"Your reply could not be used: {exc}. Reply again with ONLY the corrected JSON object.")
            messages.append(note)
            transcript.add_message(note)
            reply = call_model(provider, messages, transcript, None, guard, json_schema)
    raise AssertionError("unreachable")
