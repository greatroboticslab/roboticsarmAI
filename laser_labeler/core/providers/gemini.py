"""Adapter for Google Gemini via the google-genai SDK.

Newer Gemini models need their own model Content (with thought signatures) sent back
unchanged on later turns, so the vendor Content is stored in ``ProviderReply.raw`` /
``Message.raw`` and resent verbatim when present.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable

from core.providers.base import (
    AuthError, Message, ProviderError, ProviderReply, RateLimitError, ToolCall, ToolSpec,
    TransientError, Usage, with_retries,
)

SYNTH_PREFIX = "gemini_call_"  # ids we invent when the model gives none; never sent back


def _user_parts(msg: Message, types: Any) -> list[Any]:
    parts = [types.Part(text=msg.text)] if msg.text else []
    return parts + [types.Part.from_bytes(data=i.data, mime_type=i.mime) for i in msg.images]


def _model_content(msg: Message, types: Any) -> Any:
    if msg.raw is not None and hasattr(msg.raw, "parts"):
        return msg.raw  # vendor-native turn, resent verbatim (thought signatures)
    parts = [types.Part(text=msg.text)] if msg.text else []
    for c in msg.tool_calls:
        keep_id = None if c.id.startswith(SYNTH_PREFIX) else c.id
        parts.append(types.Part(function_call=types.FunctionCall(name=c.name, args=c.arguments, id=keep_id)))
    return types.Content(role="model", parts=parts)


def _tool_name(msg: Message, history: list[Message]) -> str:
    if msg.tool_name:
        return msg.tool_name
    for prev in reversed(history):
        for c in prev.tool_calls:
            if c.id == msg.tool_call_id:
                return c.name
    return "unknown_tool"


def _function_response(msg: Message, name: str, types: Any) -> Any:
    keep_id = None if (msg.tool_call_id or SYNTH_PREFIX).startswith(SYNTH_PREFIX) else msg.tool_call_id
    return types.Part(function_response=types.FunctionResponse(
        name=name, response={"result": msg.text or ""}, id=keep_id))


def to_gemini(messages: list[Message], types: Any) -> tuple[str, list[Any]]:
    """Return (system_text, contents). Tool results from one round share a user Content,
    function responses first, then any images the tools returned."""
    system: list[str] = []
    contents: list[Any] = []
    run: tuple[list[Any], list[Any]] | None = None  # (function responses, images)
    for i, msg in enumerate(messages):
        if msg.role == "tool":
            if run is None:
                run = ([], [])
                contents.append(run)
            run[0].append(_function_response(msg, _tool_name(msg, messages[:i]), types))
            run[1].extend(types.Part.from_bytes(data=im.data, mime_type=im.mime) for im in msg.images)
            continue
        run = None
        if msg.role == "system":
            system.append(msg.text)
        elif msg.role == "assistant":
            contents.append(_model_content(msg, types))
        else:
            contents.append(types.Content(role="user", parts=_user_parts(msg, types)))
    done = [types.Content(role="user", parts=c[0] + c[1]) if isinstance(c, tuple) else c
            for c in contents]
    return "\n\n".join(s for s in system if s), done


def to_gemini_tools(tools: list[ToolSpec], types: Any) -> list[Any]:
    decls = [types.FunctionDeclaration(name=t.name, description=t.description,
                                       parameters_json_schema=t.parameters) for t in tools]
    return [types.Tool(function_declarations=decls)]


def _retry_after(exc: Any) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        if headers and headers.get("retry-after"):
            return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        pass
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", str(getattr(exc, "details", "")))
    return float(match.group(1)) if match else None


def _is_transport_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    module = type(exc).__module__.split(".")[0]
    name = type(exc).__name__
    return module in ("httpx", "httpx2", "httpcore") and any(
        k in name for k in ("Timeout", "Connect", "Network", "Protocol", "Read", "Write", "Close"))


def map_error(exc: Exception, secret: str | None = None) -> ProviderError:
    """Translate a google-genai (or transport) exception into our error hierarchy."""
    msg = f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"
    if secret:
        msg = msg.replace(secret, "***")
    code = getattr(exc, "code", None)
    if code in (401, 403) or (code == 400 and "api key" in msg.lower()):
        return AuthError(msg)
    if code == 429:
        return RateLimitError(msg, _retry_after(exc))
    if _is_transport_error(exc) or (isinstance(code, int) and code >= 500):
        return TransientError(msg)
    return ProviderError(msg)


class GeminiProvider:
    """VisionProvider for Gemini models."""

    name = "gemini"
    supports_tools = True

    def __init__(self, model: str, api_key: str | None = None, temperature: float | None = 0.0,
                 client: Any = None, sleep: Callable[[float], None] | None = None,
                 attempts: int = 5) -> None:
        self.model = model
        self.temperature = temperature
        self._api_key = api_key or None
        self._client = client
        self._sleep = sleep or (lambda s: time.sleep(s))
        self._attempts = attempts

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self.model!r})"

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _request(self, messages: list[Message], tools: list[ToolSpec] | None) -> dict[str, Any]:
        from google.genai import types

        system, contents = to_gemini(messages, types)
        cfg: dict[str, Any] = {
            "system_instruction": system or None,
            "temperature": self.temperature,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if tools:
            cfg["tools"] = to_gemini_tools(tools, types)
        return {"model": self.model, "contents": contents,
                "config": types.GenerateContentConfig(**cfg)}

    def _call(self, kwargs: dict[str, Any]) -> Any:
        try:
            return self._get_client().models.generate_content(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise map_error(exc, self._api_key) from None

    def chat(self, messages: list[Message], tools: list[ToolSpec] | None = None,
             json_schema: dict[str, Any] | None = None) -> ProviderReply:  # schema: not used here
        kwargs = self._request(messages, tools)
        resp = with_retries(lambda: self._call(kwargs), attempts=self._attempts, sleep=self._sleep)
        return self._parse(resp)

    @staticmethod
    def _parse(resp: Any) -> ProviderReply:
        if not getattr(resp, "candidates", None):
            raise ProviderError(f"Gemini returned no candidates ({getattr(resp, 'prompt_feedback', None)})")
        cand = resp.candidates[0]
        content = getattr(cand, "content", None)
        texts: list[str] = []
        calls: list[ToolCall] = []
        for part in (getattr(content, "parts", None) or []):
            fc = getattr(part, "function_call", None)
            if fc is not None:
                calls.append(ToolCall(fc.id or f"{SYNTH_PREFIX}{len(calls)}", fc.name, dict(fc.args or {})))
            elif part.text and not getattr(part, "thought", None):
                texts.append(part.text)
        meta = getattr(resp, "usage_metadata", None)
        out_tokens = (getattr(meta, "candidates_token_count", 0) or 0) + (getattr(meta, "thoughts_token_count", 0) or 0)
        reason = getattr(cand, "finish_reason", None)
        return ProviderReply(
            text="".join(texts), tool_calls=calls,
            usage=Usage(getattr(meta, "prompt_token_count", 0) or 0, out_tokens),
            stop_reason=getattr(reason, "name", reason), raw=content,
        )
