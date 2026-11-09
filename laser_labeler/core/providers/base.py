"""Provider-neutral message types and the VisionProvider interface (spec section 6).

Adapters translate these to and from each vendor SDK. Nothing here imports an SDK.
Model names are never hardcoded: they come from settings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, TypeVar

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ImagePart:
    data: bytes
    mime: str = "image/png"


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: Role
    text: str = ""
    images: list[ImagePart] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)  # on assistant messages
    tool_call_id: str | None = None  # on tool messages: which call this answers
    tool_name: str | None = None  # on tool messages
    # Vendor-native object for an assistant turn. Adapters put it here and use it
    # to resend the turn exactly (some vendors require it). Not serialised.
    raw: Any = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ProviderReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    raw: Any = None

    def to_message(self) -> Message:
        """The assistant message to append to the conversation."""
        return Message("assistant", self.text, tool_calls=list(self.tool_calls), raw=self.raw)


class VisionProvider(Protocol):
    name: str
    model: str
    supports_tools: bool

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             json_schema: dict[str, Any] | None = None) -> ProviderReply:
        """json_schema (optional): ask the server to constrain the reply to this JSON Schema.
        Adapters that cannot do it ignore it; the reply is validated either way."""
        ...


class ProviderError(RuntimeError):
    """Any failure talking to a provider."""


class AuthError(ProviderError):
    """Bad or missing key. Not retried."""


class ContextTooLong(ProviderError):
    """The request does not fit the model's context window. Retrying the same request cannot work."""
    def __init__(self, message: str, request_tokens: int | None = None, window: int | None = None) -> None:
        super().__init__(message)
        self.request_tokens, self.window = request_tokens, window


class RateLimitError(ProviderError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(ProviderError):
    """Timeouts, 5xx, dropped connections. Retried."""


T = TypeVar("T")


def with_retries(
    fn: Callable[[], T],
    attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn, retrying RateLimitError and TransientError with exponential backoff."""
    for attempt in range(attempts):
        try:
            return fn()
        except (RateLimitError, TransientError) as exc:
            if attempt == attempts - 1:
                raise
            wait = getattr(exc, "retry_after", None) or base_delay * (2 ** attempt)
            sleep(min(float(wait), max_delay))
    raise AssertionError("unreachable")
