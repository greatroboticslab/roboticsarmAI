"""Adapter for OpenAI-compatible Chat Completions servers.

Works with OpenAI itself and with Ollama, LM Studio, llama.cpp, vLLM and OpenRouter
by changing ``base_url``. Function tools only; no model names are hardcoded.
"""
from __future__ import annotations

import base64
import json
import time
from typing import Any, Callable

from core.providers.base import (
    AuthError, ImagePart, Message, ProviderError, ProviderReply, RateLimitError,
    ToolCall, ToolSpec, TransientError, Usage, with_retries,
)

# Local servers need no key but the SDK refuses None. Deliberately not key-shaped.
PLACEHOLDER_KEY = "no-key-needed"


def _data_url(img: ImagePart) -> str:
    return f"data:{img.mime};base64,{base64.b64encode(img.data).decode('ascii')}"


def _image_block(img: ImagePart) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": _data_url(img)}}


def _user_content(msg: Message) -> Any:
    """Plain string when there are no images, else a list of content parts."""
    if not msg.images:
        return msg.text
    parts: list[dict[str, Any]] = [{"type": "text", "text": msg.text}] if msg.text else []
    return parts + [_image_block(i) for i in msg.images]


def _assistant_message(msg: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": "assistant", "content": msg.text or None}
    if msg.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
            for c in msg.tool_calls
        ]
    return out


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert neutral messages. Tool-result images become a user message that is
    emitted after the whole run of consecutive tool messages (the API requires every
    tool call to be answered before any other role appears)."""
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if pending:
            out.append({"role": "user", "content": list(pending)})
            pending.clear()

    for msg in messages:
        if msg.role != "tool":
            flush()
        if msg.role == "tool":
            out.append({"role": "tool", "tool_call_id": msg.tool_call_id or "",
                        "content": msg.text or "(no text output)"})
            if msg.images:
                label = f"Image(s) returned by tool {msg.tool_name or 'call'}:"
                pending.append({"type": "text", "text": label})
                pending.extend(_image_block(i) for i in msg.images)
        elif msg.role == "assistant":
            out.append(_assistant_message(msg))
        else:
            out.append({"role": msg.role, "content": _user_content(msg)})
    flush()
    return out


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools]


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string. Invalid JSON becomes {"_raw": raw}."""
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return {"_raw": raw}
    return value if isinstance(value, dict) else {"_raw": raw}


def _retry_after(exc: Any) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after")) if headers else None
    except (TypeError, ValueError):
        return None


def map_error(exc: Exception, secret: str | None = None) -> ProviderError:
    """Translate an openai SDK exception into our error hierarchy."""
    import openai

    msg = f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"
    if secret:
        msg = msg.replace(secret, "***")
    status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return AuthError(msg)
    if status == 429:
        return RateLimitError(msg, _retry_after(exc))
    if isinstance(exc, openai.APIConnectionError) or (isinstance(status, int) and status >= 500):
        return TransientError(msg)
    return ProviderError(msg)


class OpenAICompatProvider:
    """VisionProvider for any server that speaks OpenAI Chat Completions."""

    name = "openai_compat"
    supports_tools = True

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 temperature: float | None = 0.0, client: Any = None,
                 sleep: Callable[[float], None] | None = None, attempts: int = 5,
                 structured_output: bool = False) -> None:
        self.structured_output = structured_output  # send response_format=json_schema when asked
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._api_key = api_key or None
        self._client = client
        self._sleep = sleep or (lambda s: time.sleep(s))
        self._attempts = attempts

    def __repr__(self) -> str:
        return f"OpenAICompatProvider(model={self.model!r}, base_url={self.base_url!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=self._api_key or PLACEHOLDER_KEY,
                base_url=self.base_url or None,
                max_retries=0,  # retries are handled by with_retries
            )
        return self._client

    def _request(self, messages: list[Message], tools: list[ToolSpec] | None,
                 json_schema: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": to_openai_messages(messages)}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        elif json_schema and self.structured_output:  # tools and a schema do not mix on most servers
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "reply", "schema": json_schema, "strict": True}}
        return kwargs

    def _call(self, kwargs: dict[str, Any]) -> Any:
        try:
            return self._get_client().chat.completions.create(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 - mapped to our hierarchy
            raise map_error(exc, self._api_key) from None

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             json_schema: dict[str, Any] | None = None) -> ProviderReply:
        kwargs = self._request(messages, tools, json_schema)
        resp = with_retries(lambda: self._call(kwargs), attempts=self._attempts, sleep=self._sleep)
        return self._parse(resp)

    @staticmethod
    def _parse(resp: Any) -> ProviderReply:
        if not getattr(resp, "choices", None):
            raise ProviderError("Provider returned no choices")
        choice = resp.choices[0]
        message = choice.message
        calls = [ToolCall(id=c.id, name=c.function.name, arguments=parse_arguments(c.function.arguments))
                 for c in (getattr(message, "tool_calls", None) or [])]
        usage = getattr(resp, "usage", None)
        return ProviderReply(
            text=message.content or "",
            tool_calls=calls,
            usage=Usage(getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0),
            stop_reason=getattr(choice, "finish_reason", None),
            raw=resp,
        )
