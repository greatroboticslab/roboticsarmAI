"""Adapter for the Anthropic Messages API (system prompt, base64 images, tool_use blocks)."""
from __future__ import annotations

import base64
import time
from typing import Any, Callable

from core.providers.base import (
    AuthError, ImagePart, Message, ProviderError, ProviderReply, RateLimitError,
    ToolCall, ToolSpec, TransientError, Usage, with_retries,
)


def _image_block(img: ImagePart) -> dict[str, Any]:
    data = base64.b64encode(img.data).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": img.mime, "data": data}}


def _text_block(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text}] if text else []  # empty text blocks are rejected


def _tool_result_block(msg: Message) -> dict[str, Any]:
    content = _text_block(msg.text) + [_image_block(i) for i in msg.images]
    return {"type": "tool_result", "tool_use_id": msg.tool_call_id or "", "content": content or ""}


def _assistant_blocks(msg: Message) -> list[dict[str, Any]]:
    blocks = _text_block(msg.text)
    blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
               for c in msg.tool_calls]
    return blocks


def to_anthropic(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Return (system_text, messages). Consecutive tool results merge into one user turn."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    merge_into: list[dict[str, Any]] | None = None  # content list of the open tool-result turn
    for msg in messages:
        if msg.role == "tool":
            block = _tool_result_block(msg)
            if merge_into is None:
                merge_into = [block]
                out.append({"role": "user", "content": merge_into})
            else:
                merge_into.append(block)
            continue
        merge_into = None
        if msg.role == "system":
            system_parts.append(msg.text)
        elif msg.role == "assistant":
            out.append({"role": "assistant", "content": _assistant_blocks(msg)})
        else:
            out.append({"role": "user",
                        "content": _text_block(msg.text) + [_image_block(i) for i in msg.images]})
    return "\n\n".join(p for p in system_parts if p), out


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [{"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools]


def _retry_after(exc: Any) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after")) if headers else None
    except (TypeError, ValueError):
        return None


def map_error(exc: Exception, secret: str | None = None) -> ProviderError:
    """Translate an anthropic SDK exception into our error hierarchy."""
    import anthropic

    msg = f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"
    if secret:
        msg = msg.replace(secret, "***")
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return AuthError(msg)
    if status == 429:
        return RateLimitError(msg, _retry_after(exc))
    if isinstance(exc, anthropic.APIConnectionError) or (isinstance(status, int) and status >= 500):
        return TransientError(msg)
    return ProviderError(msg)


def _is_temperature_rejection(exc: Exception) -> bool:
    text = str(getattr(exc, "message", None) or exc).lower()
    return getattr(exc, "status_code", None) == 400 and "temperature" in text


class AnthropicProvider:
    """VisionProvider for Claude models via the Messages API."""

    name = "anthropic"
    supports_tools = True

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 temperature: float | None = 0.0, client: Any = None, max_tokens: int = 4096,
                 sleep: Callable[[float], None] | None = None, attempts: int = 5) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or None
        self._client = client
        self._sleep = sleep or (lambda s: time.sleep(s))
        self._attempts = attempts
        self._send_temperature = temperature is not None

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self.model!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key, base_url=self.base_url,
                                               max_retries=0)  # retries via with_retries
        return self._client

    def _request(self, messages: list[Message], tools: list[ToolSpec] | None) -> dict[str, Any]:
        system, converted = to_anthropic(messages)
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens,
                                  "messages": converted}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)
        return kwargs

    def _create(self, kwargs: dict[str, Any]) -> Any:
        """One API call; on a 400 that rejects `temperature`, retry once without it."""
        if self._send_temperature:
            kwargs = {**kwargs, "temperature": self.temperature}
        try:
            return self._get_client().messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if self._send_temperature and _is_temperature_rejection(exc):
                self._send_temperature = False
                kwargs.pop("temperature", None)
                return self._create(kwargs)
            if isinstance(exc, ProviderError):
                raise
            raise map_error(exc, self._api_key) from None

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             json_schema: dict[str, Any] | None = None) -> ProviderReply:  # schema: not used here
        kwargs = self._request(messages, tools)
        resp = with_retries(lambda: self._create(dict(kwargs)), attempts=self._attempts,
                            sleep=self._sleep)
        return self._parse(resp)

    @staticmethod
    def _parse(resp: Any) -> ProviderReply:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in getattr(resp, "content", None) or []:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input or {})))
        usage = getattr(resp, "usage", None)
        return ProviderReply(
            text="".join(texts), tool_calls=calls,
            usage=Usage(getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0),
            stop_reason=getattr(resp, "stop_reason", None), raw=resp,
        )
