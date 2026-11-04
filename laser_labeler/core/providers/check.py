"""Test provider (spec section 6): does the model return valid JSON and call tools?"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from core.labeler.parse_reply import ReplyParseError, extract_json
from core.providers.base import ImagePart, Message, ProviderError, ToolSpec, VisionProvider
from core.tools.crop import encode_png

ECHO = ToolSpec("echo", "Repeat a word back.", {"type": "object", "properties": {"word": {"type": "string"}},
                                                  "required": ["word"]})


@dataclass
class ProviderCheck:
    attempts: int = 0
    valid_json: int = 0
    tool_call_ok: bool | None = None  # None = provider says it has no tool support
    seconds_per_call: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def valid_json_rate(self) -> float:
        return self.valid_json / self.attempts if self.attempts else 0.0


def sample_image() -> ImagePart:
    """A dark picture with one bright dot: enough to prove images are accepted."""
    img = np.full((128, 128, 3), 30, np.uint8)
    img[60:68, 60:68] = (255, 80, 80)
    return ImagePart(encode_png(img))


def check_provider(provider: VisionProvider, images: list[ImagePart] | None = None, attempts: int = 3) -> ProviderCheck:
    """Send a few image prompts asking for JSON, then one tool-call prompt."""
    images = images or [sample_image()]
    result, started = ProviderCheck(), time.monotonic()
    calls = 0
    for i in range(attempts):
        result.attempts += 1
        calls += 1
        ask = Message("user", 'Is there a bright dot in this image? Reply with ONLY this JSON: {"dot": true or false}',
                      images=[images[i % len(images)]])
        try:
            extract_json(provider.chat([ask]).text)
            result.valid_json += 1
        except (ReplyParseError, ProviderError) as exc:
            result.errors.append(str(exc))
    if provider.supports_tools:
        calls += 1
        try:
            reply = provider.chat([Message("user", "Call the echo tool with the word 'ping'.")], [ECHO])
            result.tool_call_ok = any(c.name == "echo" and isinstance(c.arguments.get("word"), str)
                                      for c in reply.tool_calls)
        except ProviderError as exc:
            result.tool_call_ok = False
            result.errors.append(str(exc))
    result.seconds_per_call = (time.monotonic() - started) / calls
    return result


def format_check(name: str, c: ProviderCheck) -> str:
    tool = {None: "not supported", True: "works", False: "FAILED"}[c.tool_call_ok]
    lines = [f"{name}: valid JSON {c.valid_json}/{c.attempts} ({c.valid_json_rate:.0%}), tool calls {tool}, "
             f"{c.seconds_per_call:.1f}s per call"]
    lines += [f"  problem: {e}" for e in c.errors[:3]]
    if c.tool_call_ok is not True:
        lines.append("  -> use pipeline mode with this model")
    return "\n".join(lines)
