"""Adapter tests. Fake clients mimic SDK response shapes; no network is used."""
import json
from types import SimpleNamespace as NS

import anthropic as anthropic_sdk
import httpx2
import openai as openai_sdk
import pytest
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as gt

from core.providers.anthropic import AnthropicProvider
from core.providers.base import (
    AuthError, ImagePart, Message, ProviderError, RateLimitError, ToolCall, ToolSpec,
    TransientError,
)
from core.providers.gemini import GeminiProvider
from core.providers.openai_compat import PLACEHOLDER_KEY, OpenAICompatProvider
from core.providers.registry import PROVIDER_CHOICES, build_provider

PNG = ImagePart(b"\x89PNGdata", "image/png")
CROP = ImagePart(b"cropbytes", "image/jpeg")
SPEC = ToolSpec("zoom", "Zoom in", {"type": "object", "properties": {"x": {"type": "integer"}}})
SECRET = "sk-test-SECRET-123"


class Scripted:
    """Returns/raises queued outcomes and records every call's kwargs."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def __call__(self, **kw):
        self.calls.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def http_resp(status, headers=None):
    return httpx2.Response(status, headers=headers or {}, request=httpx2.Request("POST", "http://test/x"))


def convo():
    """system, user+image, assistant with two parallel tool calls, two results (one with image)."""
    return [
        Message("system", "Be precise."),
        Message("user", "Label this", images=[PNG]),
        Message("assistant", "Checking", tool_calls=[ToolCall("c1", "zoom", {"x": 1}),
                                                     ToolCall("c2", "zoom", {"x": 2})]),
        Message("tool", "crop one", images=[CROP], tool_call_id="c1", tool_name="zoom"),
        Message("tool", "crop two", tool_call_id="c2", tool_name="zoom"),
    ]


def no_sleep(log=None):
    return (lambda s: log.append(s)) if log is not None else (lambda s: None)


# ---------------------------------------------------------------- OpenAI-compatible

def oa_client(*outcomes):
    s = Scripted(*outcomes)
    return NS(chat=NS(completions=NS(create=s))), s


def oa_reply(text="hi", calls=(), usage=(11, 7), finish="stop"):
    tcs = [NS(id=i, type="function", function=NS(name=n, arguments=a)) for i, n, a in calls]
    msg = NS(content=text, tool_calls=tcs or None)
    return NS(choices=[NS(message=msg, finish_reason=finish)],
              usage=NS(prompt_tokens=usage[0], completion_tokens=usage[1]))


def test_openai_request_conversion():
    client, s = oa_client(oa_reply())
    OpenAICompatProvider("m1", client=client, temperature=0.2).chat(convo(), [SPEC])
    kw = s.calls[0]
    assert kw["model"] == "m1" and kw["temperature"] == 0.2
    assert kw["tools"] == [{"type": "function", "function": {
        "name": "zoom", "description": "Zoom in", "parameters": SPEC.parameters}}]
    m = kw["messages"]
    assert m[0] == {"role": "system", "content": "Be precise."}
    assert m[1]["content"][0] == {"type": "text", "text": "Label this"}
    assert m[1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    asst = m[2]
    assert [c["id"] for c in asst["tool_calls"]] == ["c1", "c2"]
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
    # both tool messages first, then ONE user message holding the crop
    assert [x["role"] for x in m[3:]] == ["tool", "tool", "user"]
    assert (m[3]["tool_call_id"], m[3]["content"]) == ("c1", "crop one")
    assert m[4]["tool_call_id"] == "c2"
    imgs = [p for p in m[5]["content"] if p["type"] == "image_url"]
    assert len(imgs) == 1 and imgs[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_plain_text_user_and_no_tools_key():
    client, s = oa_client(oa_reply())
    OpenAICompatProvider("m", client=client, temperature=None).chat([Message("user", "hello")])
    kw = s.calls[0]
    assert kw["messages"] == [{"role": "user", "content": "hello"}]
    assert "tools" not in kw and "temperature" not in kw


def test_openai_response_parsing():
    reply = oa_reply("", calls=[("a", "zoom", '{"x": 3}')], usage=(100, 20), finish="tool_calls")
    client, _ = oa_client(reply)
    r = OpenAICompatProvider("m", client=client).chat([Message("user", "q")], [SPEC])
    assert r.text == "" and r.tool_calls == [ToolCall("a", "zoom", {"x": 3})]
    assert (r.usage.input_tokens, r.usage.output_tokens) == (100, 20)
    assert r.stop_reason == "tool_calls" and r.raw is reply
    assert r.to_message().tool_calls[0].id == "a"


def test_openai_invalid_json_arguments_become_raw():
    client, _ = oa_client(oa_reply(calls=[("a", "zoom", "{not json"), ("b", "zoom", "[1]"), ("c", "zoom", "")]))
    r = OpenAICompatProvider("m", client=client).chat([Message("user", "q")])
    assert r.tool_calls[0].arguments == {"_raw": "{not json"}
    assert r.tool_calls[1].arguments == {"_raw": "[1]"}
    assert r.tool_calls[2].arguments == {}


def test_openai_missing_usage_is_zero():
    resp = NS(choices=[NS(message=NS(content="x", tool_calls=None), finish_reason="stop")], usage=None)
    client, _ = oa_client(resp)
    r = OpenAICompatProvider("m", client=client).chat([Message("user", "q")])
    assert (r.usage.input_tokens, r.usage.output_tokens) == (0, 0)


@pytest.mark.parametrize("exc,expected", [
    (openai_sdk.AuthenticationError("bad", response=http_resp(401), body=None), AuthError),
    (openai_sdk.PermissionDeniedError("no", response=http_resp(403), body=None), AuthError),
    (openai_sdk.InternalServerError("oops", response=http_resp(503), body=None), TransientError),
    (openai_sdk.APIConnectionError(request=httpx2.Request("POST", "http://t")), TransientError),
    (openai_sdk.BadRequestError("bad", response=http_resp(400), body=None), ProviderError),
])
def test_openai_error_mapping(exc, expected):
    client, _ = oa_client(exc, exc, exc, exc, exc)
    p = OpenAICompatProvider("m", client=client, sleep=no_sleep(), attempts=2)
    with pytest.raises(expected) as info:
        p.chat([Message("user", "q")])
    assert type(info.value) is expected or expected is ProviderError


def test_openai_rate_limit_retry_after_and_retry():
    rl = openai_sdk.RateLimitError("slow", response=http_resp(429, {"retry-after": "7"}), body=None)
    client, s = oa_client(rl, oa_reply("ok"))
    waits = []
    r = OpenAICompatProvider("m", client=client, sleep=no_sleep(waits)).chat([Message("user", "q")])
    assert r.text == "ok" and len(s.calls) == 2 and waits == [7.0]


def test_openai_rate_limit_exhausted_carries_retry_after():
    rl = openai_sdk.RateLimitError("slow", response=http_resp(429, {"retry-after": "3"}), body=None)
    client, s = oa_client(rl, rl)
    with pytest.raises(RateLimitError) as info:
        OpenAICompatProvider("m", client=client, sleep=no_sleep(), attempts=2).chat([Message("user", "q")])
    assert info.value.retry_after == 3.0 and len(s.calls) == 2


def test_openai_auth_error_not_retried_and_key_not_leaked():
    exc = openai_sdk.AuthenticationError(f"Incorrect key {SECRET}", response=http_resp(401), body=None)
    client, s = oa_client(exc)
    p = OpenAICompatProvider("m", api_key=SECRET, client=client, sleep=no_sleep())
    with pytest.raises(AuthError) as info:
        p.chat([Message("user", "q")])
    assert len(s.calls) == 1 and SECRET not in str(info.value) and SECRET not in repr(p)


def test_openai_retry_uses_time_sleep_by_default(monkeypatch):
    import time
    waits = []
    monkeypatch.setattr(time, "sleep", waits.append)
    err = openai_sdk.InternalServerError("x", response=http_resp(500), body=None)
    client, _ = oa_client(err, oa_reply("ok"))
    assert OpenAICompatProvider("m", client=client).chat([Message("user", "q")]).text == "ok"
    assert waits == [1.0]


def test_openai_real_client_uses_placeholder_key_and_base_url(monkeypatch):
    seen = {}
    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: seen.update(kw) or oa_client(oa_reply())[0])
    OpenAICompatProvider("m", api_key=None, base_url="http://localhost:11434/v1").chat([Message("user", "q")])
    assert seen["api_key"] == PLACEHOLDER_KEY and seen["base_url"] == "http://localhost:11434/v1"
    seen.clear()
    OpenAICompatProvider("m", api_key="", base_url=None).chat([Message("user", "q")])
    assert seen["api_key"] == PLACEHOLDER_KEY


def test_openai_timeout_is_not_retried_and_says_so():
    client, s = oa_client(openai_sdk.APITimeoutError(httpx2.Request("POST", "http://t")))
    p = OpenAICompatProvider("m", client=client, sleep=no_sleep(), attempts=5, timeout=300.0)
    with pytest.raises(ProviderError, match="Request timed out after 300 seconds") as info:
        p.chat([Message("user", "q")])
    assert not isinstance(info.value, TransientError) and len(s.calls) == 1  # one try, not five slow ones


def test_openai_thinking_off_and_timeout_are_sent_only_when_asked(monkeypatch):
    client, s = oa_client(oa_reply(), oa_reply())
    OpenAICompatProvider("m", client=client).chat([Message("user", "q")])
    assert "reasoning_effort" not in s.calls[0]
    OpenAICompatProvider("m", client=client, disable_thinking=True).chat([Message("user", "q")])
    assert s.calls[1]["reasoning_effort"] == "none"
    seen = {}
    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: seen.update(kw) or oa_client(oa_reply())[0])
    OpenAICompatProvider("m", timeout=120.0).chat([Message("user", "q")])
    assert seen["timeout"] == 120.0
    seen.clear()
    OpenAICompatProvider("m").chat([Message("user", "q")])
    assert "timeout" not in seen


def test_a_reply_limit_is_sent_only_when_set():
    client, s = oa_client(oa_reply(), oa_reply(), oa_reply())
    OpenAICompatProvider("m", client=client).chat([Message("user", "q")])
    OpenAICompatProvider("m", client=client, reply_limit=0).chat([Message("user", "q")])
    OpenAICompatProvider("m", client=client, reply_limit=1200).chat([Message("user", "q")])
    assert "max_tokens" not in s.calls[0] and "max_tokens" not in s.calls[1] and s.calls[2]["max_tokens"] == 1200


def test_build_provider_passes_the_reply_limit():
    settings = {"provider": "openai_compat", "model": "m", "base_url": "http://localhost:11434/v1", "reply_limit": 900}
    assert build_provider(settings, None).reply_limit == 900
    assert build_provider({**settings, "reply_limit": 0}, None).reply_limit is None


def test_build_provider_passes_timeout_and_thinking():
    settings = {"provider": "openai_compat", "model": "qwen3-vl:4b", "base_url": "http://localhost:11434/v1",
                "timeout_s": 90, "disable_thinking": True}
    p = build_provider(settings, None)
    assert p.timeout == 90.0 and p.disable_thinking is True
    assert build_provider({**settings, "timeout_s": 0}, None).timeout is None


# ---------------------------------------------------------------- Anthropic

def an_client(*outcomes):
    s = Scripted(*outcomes)
    return NS(messages=NS(create=s)), s


def an_reply(blocks=(("text", "hi"),), usage=(5, 3), stop="end_turn"):
    content = []
    if isinstance(blocks, str):
        blocks = (("text", blocks),)
    for b in blocks:
        content.append(NS(type="text", text=b[1]) if b[0] == "text"
                       else NS(type="tool_use", id=b[1], name=b[2], input=b[3]))
    return NS(content=content, usage=NS(input_tokens=usage[0], output_tokens=usage[1]), stop_reason=stop)


def an_400(msg):
    return anthropic_sdk.BadRequestError(msg, response=http_resp(400), body=None)


def test_anthropic_request_conversion():
    client, s = an_client(an_reply())
    AnthropicProvider("m1", client=client, temperature=0.3, max_tokens=999).chat(convo(), [SPEC])
    kw = s.calls[0]
    assert kw["model"] == "m1" and kw["max_tokens"] == 999 and kw["temperature"] == 0.3
    assert kw["system"] == "Be precise."
    assert kw["tools"] == [{"name": "zoom", "description": "Zoom in", "input_schema": SPEC.parameters}]
    m = kw["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "user"]  # system removed, results merged
    img = m[0]["content"][1]
    assert img["type"] == "image" and img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/png" and img["source"]["data"]
    uses = [b for b in m[1]["content"] if b["type"] == "tool_use"]
    assert [(u["id"], u["input"]) for u in uses] == [("c1", {"x": 1}), ("c2", {"x": 2})]
    results = m[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["c1", "c2"]
    assert [b["type"] for b in results[0]["content"]] == ["text", "image"]
    assert results[0]["content"][1]["source"]["media_type"] == "image/jpeg"
    assert results[1]["content"] == [{"type": "text", "text": "crop two"}]


def test_anthropic_default_max_tokens_and_no_empty_text_blocks():
    client, s = an_client(an_reply())
    AnthropicProvider("m", client=client).chat(
        [Message("user", "q"), Message("assistant", "", tool_calls=[ToolCall("t", "zoom", {})]),
         Message("tool", "", tool_call_id="t")])
    kw = s.calls[0]
    assert kw["max_tokens"] == 4096 and "system" not in kw and "tools" not in kw
    assert [b["type"] for b in kw["messages"][1]["content"]] == ["tool_use"]
    assert kw["messages"][2]["content"][0]["content"] == ""


def test_anthropic_response_parsing():
    reply = an_reply([("text", "a"), ("tool_use", "tu1", "zoom", {"x": 9}), ("text", "b")],
                     usage=(40, 12), stop="tool_use")
    client, _ = an_client(reply)
    r = AnthropicProvider("m", client=client).chat([Message("user", "q")], [SPEC])
    assert r.text == "ab" and r.tool_calls == [ToolCall("tu1", "zoom", {"x": 9})]
    assert (r.usage.input_tokens, r.usage.output_tokens) == (40, 12)
    assert r.stop_reason == "tool_use" and r.raw is reply


def test_anthropic_temperature_none_not_sent():
    client, s = an_client(an_reply())
    AnthropicProvider("m", client=client, temperature=None).chat([Message("user", "q")])
    assert "temperature" not in s.calls[0]


def test_anthropic_temperature_fallback_retries_once_and_remembers():
    client, s = an_client(an_400("`temperature` is deprecated for this model"), an_reply("ok"), an_reply("again"))
    p = AnthropicProvider("m", client=client, temperature=0.0, sleep=no_sleep())
    assert p.chat([Message("user", "q")]).text == "ok"
    assert "temperature" in s.calls[0] and "temperature" not in s.calls[1]
    assert p.chat([Message("user", "q")]).text == "again"
    assert "temperature" not in s.calls[2] and len(s.calls) == 3


def test_anthropic_other_400_not_retried():
    client, s = an_client(an_400("max_tokens too large"))
    with pytest.raises(ProviderError) as info:
        AnthropicProvider("m", client=client, sleep=no_sleep()).chat([Message("user", "q")])
    assert not isinstance(info.value, (AuthError, TransientError)) and len(s.calls) == 1


@pytest.mark.parametrize("exc,expected", [
    (anthropic_sdk.AuthenticationError("bad", response=http_resp(401), body=None), AuthError),
    (anthropic_sdk.PermissionDeniedError("no", response=http_resp(403), body=None), AuthError),
    (anthropic_sdk.InternalServerError("oops", response=http_resp(500), body=None), TransientError),
    (anthropic_sdk.APIStatusError("overloaded", response=http_resp(529), body=None), TransientError),
    (anthropic_sdk.APITimeoutError(httpx2.Request("POST", "http://t")), TransientError),
    (anthropic_sdk.APIConnectionError(request=httpx2.Request("POST", "http://t")), TransientError),
])
def test_anthropic_error_mapping(exc, expected):
    client, _ = an_client(exc, exc)
    with pytest.raises(expected):
        AnthropicProvider("m", client=client, sleep=no_sleep(), attempts=2).chat([Message("user", "q")])


def test_anthropic_rate_limit_retry_after_then_success():
    rl = anthropic_sdk.RateLimitError("slow", response=http_resp(429, {"retry-after": "12"}), body=None)
    client, s = an_client(rl, an_reply("ok"))
    waits = []
    r = AnthropicProvider("m", client=client, sleep=no_sleep(waits)).chat([Message("user", "q")])
    assert r.text == "ok" and waits == [12.0] and len(s.calls) == 2


def test_anthropic_key_not_leaked():
    exc = anthropic_sdk.AuthenticationError(f"bad {SECRET}", response=http_resp(401), body=None)
    client, _ = an_client(exc)
    p = AnthropicProvider("m", api_key=SECRET, client=client)
    with pytest.raises(AuthError) as info:
        p.chat([Message("user", "q")])
    assert SECRET not in str(info.value) and SECRET not in repr(p)


def test_anthropic_real_client_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(anthropic_sdk, "Anthropic", lambda **kw: seen.update(kw) or an_client(an_reply())[0])
    AnthropicProvider("m", api_key="k").chat([Message("user", "q")])
    assert seen["api_key"] == "k" and seen["max_retries"] == 0


# ---------------------------------------------------------------- Gemini

def gm_client(*outcomes):
    s = Scripted(*outcomes)
    return NS(models=NS(generate_content=s)), s


def gm_reply(parts, usage=(8, 4, 0), finish="STOP"):
    content = gt.Content(role="model", parts=parts)
    meta = gt.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage[0], candidates_token_count=usage[1], thoughts_token_count=usage[2])
    return gt.GenerateContentResponse(
        candidates=[gt.Candidate(content=content, finish_reason=finish)], usage_metadata=meta)


def gm_error(code, headers=None):
    return genai_errors.ClientError(code, {"error": {"message": "boom", "status": "X"}},
                                    response=http_resp(code, headers))


def test_gemini_request_conversion():
    client, s = gm_client(gm_reply([gt.Part(text="ok")]))
    GeminiProvider("g1", client=client, temperature=0.4).chat(convo(), [SPEC])
    kw = s.calls[0]
    assert kw["model"] == "g1"
    cfg = kw["config"]
    assert cfg.system_instruction == "Be precise." and cfg.temperature == 0.4
    decl = cfg.tools[0].function_declarations[0]
    assert (decl.name, decl.description, decl.parameters_json_schema) == ("zoom", "Zoom in", SPEC.parameters)
    c = kw["contents"]
    assert [x.role for x in c] == ["user", "model", "user"]
    assert c[0].parts[0].text == "Label this"
    assert c[0].parts[1].inline_data.data == PNG.data and c[0].parts[1].inline_data.mime_type == "image/png"
    calls = [p.function_call for p in c[1].parts if p.function_call]
    assert [(f.id, f.name, f.args) for f in calls] == [("c1", "zoom", {"x": 1}), ("c2", "zoom", {"x": 2})]
    frs = [p.function_response for p in c[2].parts if p.function_response]
    assert [(f.id, f.name, f.response) for f in frs] == [
        ("c1", "zoom", {"result": "crop one"}), ("c2", "zoom", {"result": "crop two"})]
    last = c[2].parts[-1]  # the crop image rides in the same user content, after responses
    assert last.inline_data.data == CROP.data and last.inline_data.mime_type == "image/jpeg"
    assert len(c[2].parts) == 3


def test_gemini_tool_name_recovered_from_history_when_missing():
    client, s = gm_client(gm_reply([gt.Part(text="ok")]))
    msgs = [Message("user", "q"), Message("assistant", "", tool_calls=[ToolCall("c9", "zoom", {})]),
            Message("tool", "res", tool_call_id="c9")]
    GeminiProvider("g", client=client).chat(msgs)
    assert s.calls[0]["contents"][2].parts[0].function_response.name == "zoom"


def test_gemini_response_parsing_and_raw_roundtrip():
    sig = b"thought-signature"
    parts = [gt.Part(text="thinking...", thought=True),
             gt.Part(text="Here"),
             gt.Part(function_call=gt.FunctionCall(name="zoom", args={"x": 5}), thought_signature=sig)]
    reply = gm_reply(parts, usage=(30, 10, 5))
    client, s = gm_client(reply, gm_reply([gt.Part(text="done")]))
    p = GeminiProvider("g", client=client)
    r = p.chat([Message("user", "q")], [SPEC])
    assert r.text == "Here"
    assert len(r.tool_calls) == 1 and r.tool_calls[0].name == "zoom" and r.tool_calls[0].arguments == {"x": 5}
    assert r.tool_calls[0].id  # synthesised when the model gives none
    assert (r.usage.input_tokens, r.usage.output_tokens) == (30, 15)
    assert r.stop_reason == "STOP"
    assert r.raw is reply.candidates[0].content
    # second turn: the vendor Content is resent verbatim, signature intact
    history = [Message("user", "q"), r.to_message(),
               Message("tool", "res", tool_call_id=r.tool_calls[0].id, tool_name="zoom")]
    p.chat(history)
    sent = s.calls[1]["contents"]
    assert sent[1] is reply.candidates[0].content
    assert sent[1].parts[2].thought_signature == sig
    assert sent[2].parts[0].function_response.id is None  # synthesised ids are not echoed


def test_gemini_missing_candidates_is_provider_error():
    client, _ = gm_client(gt.GenerateContentResponse(candidates=[]))
    with pytest.raises(ProviderError):
        GeminiProvider("g", client=client).chat([Message("user", "q")])


@pytest.mark.parametrize("exc,expected", [
    (gm_error(401), AuthError),
    (gm_error(403), AuthError),
    (gm_error(400), ProviderError),
    (genai_errors.ServerError(503, {"error": {"message": "down"}}, response=http_resp(503)), TransientError),
    (httpx2.ReadTimeout("slow"), TransientError),
    (httpx2.ConnectError("refused"), TransientError),
    (TimeoutError("t"), TransientError),
])
def test_gemini_error_mapping(exc, expected):
    client, _ = gm_client(exc, exc)
    with pytest.raises(expected):
        GeminiProvider("g", client=client, sleep=no_sleep(), attempts=2).chat([Message("user", "q")])


def test_gemini_invalid_key_400_is_auth_error():
    exc = genai_errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}})
    client, s = gm_client(exc)
    with pytest.raises(AuthError):
        GeminiProvider("g", client=client, sleep=no_sleep()).chat([Message("user", "q")])
    assert len(s.calls) == 1


def test_gemini_rate_limit_retry_after_then_success():
    client, s = gm_client(gm_error(429, {"retry-after": "9"}), gm_reply([gt.Part(text="ok")]))
    waits = []
    r = GeminiProvider("g", client=client, sleep=no_sleep(waits)).chat([Message("user", "q")])
    assert r.text == "ok" and waits == [9.0] and len(s.calls) == 2


def test_gemini_key_not_leaked():
    exc = genai_errors.ClientError(403, {"error": {"message": f"key {SECRET} denied"}})
    client, _ = gm_client(exc)
    p = GeminiProvider("g", api_key=SECRET, client=client)
    with pytest.raises(AuthError) as info:
        p.chat([Message("user", "q")])
    assert SECRET not in str(info.value) and SECRET not in repr(p)


def test_gemini_real_client_args(monkeypatch):
    seen = {}
    monkeypatch.setattr(genai, "Client", lambda **kw: seen.update(kw) or gm_client(gm_reply([gt.Part(text="x")]))[0])
    GeminiProvider("g", api_key="k").chat([Message("user", "q")])
    assert seen == {"api_key": "k"}


# ---------------------------------------------------------------- registry

def settings(**over):
    base = {"provider": "openai_compat", "model": "some-model", "base_url": "http://localhost:11434/v1",
            "temperature": 0.0, "max_image_px": 1024, "run_mode": "pipeline", "daily_cost_cap_usd": 5.0}
    return {**base, **over}


def test_registry_builds_each_provider():
    assert PROVIDER_CHOICES == ("openai_compat", "gemini", "anthropic")
    expected = {"openai_compat": OpenAICompatProvider, "gemini": GeminiProvider, "anthropic": AnthropicProvider}
    for name, cls in expected.items():
        p = build_provider(settings(provider=name, temperature=0.5), None, client=object())
        assert type(p) is cls and p.name == name and p.model == "some-model"
        assert p.supports_tools and p.temperature == 0.5
    assert build_provider(settings(), None).base_url == "http://localhost:11434/v1"


def test_registry_injects_client():
    client, s = oa_client(oa_reply("via fake"))
    p = build_provider(settings(), None, client=client)
    assert p.chat([Message("user", "q")]).text == "via fake"


@pytest.mark.parametrize("bad", [{"provider": "nope"}, {"provider": ""}, {"model": ""}, {"model": "  "}])
def test_registry_rejects_bad_settings(bad):
    with pytest.raises(ValueError):
        build_provider(settings(**bad), None)


def test_registry_empty_model_message_points_to_settings():
    with pytest.raises(ValueError, match="AI settings"):
        build_provider(settings(model=""), None)
