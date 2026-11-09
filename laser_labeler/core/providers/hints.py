"""Turn a provider error into a plain-words cause and a next step. Pure text, no network."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

_TRUNCATED = re.compile(r"cut off after (\d+) tokens, with the request at (\d+) tokens", re.I)


def failure_hint(error: str | None, provider: str = "", model: str = "") -> str:
    """One or two sentences saying what probably went wrong and what to try. Empty when nothing useful can be said."""
    text = (error or "").lower()
    if not text:
        return ""
    local = provider in ("", "openai_compat")
    if "ran out of tokens" in text:
        numbers = ""
        if m := _TRUNCATED.search(text):
            reply, request = int(m.group(1)), int(m.group(2))
            numbers = f"Request {request} + reply {reply} = {request + reply} tokens, the end of the window unless your 'Longest reply allowed' stopped it. "
            if reply > 1500:
                numbers += ("The reply alone used a large part of it, and the answer the app needs is only a few hundred "
                            "tokens: the model is thinking or rambling. Use an instruct model (not a thinking one), tick "
                            "'Turn off thinking', and set 'Longest reply allowed' to about 1200 so it fails fast. ")
        return (numbers + "The reply was cut off, so the request plus the answer did not fit in the context window. "
                "Making the window bigger is the last thing to try (more video memory, slower, and a 16k window often "
                "makes timeouts worse). Shrink the request first (AI settings: lower 'Largest image side sent', 'Spots "
                "shown per image' and 'Zoom crop size'); a bigger window uses more video memory. If the reply token "
                "count is large (more than about 1,500), the model is thinking or rambling: tick 'Turn off thinking' "
                "or use an instruct model, not a thinking one.")
    if "context size" in text or "context length" in text or "context window" in text or "num_ctx" in text \
            or "maximum context" in text or "too many tokens" in text:
        extra = (" To raise the window on Ollama, set OLLAMA_CONTEXT_LENGTH (for example 6144) and restart it, or make a "
                 "copy of the model with `PARAMETER num_ctx 6144` in a Modelfile (see the README); a bigger window uses "
                 "more video memory, so check `ollama ps` still says 100% GPU." if local else "")
        return ("The request is longer than the model's context window (Ollama defaults to 4096 tokens). The app "
                "retries with smaller requests on its own when 'Fit the request to the context window' is on; if it "
                "still fails, even the smallest request is too long for this window. Lower 'Spots shown per image', "
                "'Zoom crop size' and the image size in AI settings, or raise the window." + extra)
    if any(w in text for w in ("does not support image", "doesn't support image", "image input", "multimodal", "vision",
                               "images are not supported", "image_url")) and "not found" not in text:
        return ("This model cannot read images. Pick a vision model (one that accepts pictures) in AI settings, "
                "then press Test provider.")
    if "404" in text or "not found" in text or "no such model" in text or "model_not_found" in text or "unknown model" in text:
        shown = f" '{model}'" if model else ""
        hint = (f"The server does not have a model named exactly{shown}. Copy the name again from your provider, "
                "with any :tag after it.")
        return hint + (" For Ollama run `ollama list` and copy the NAME column, or `ollama pull <name>` first." if local else "")
    if "401" in text or "403" in text or "authenticationerror" in text or "autherror" in text or "api key" in text \
            or "unauthorized" in text or "permission" in text:
        return "The key was refused. Check it on the Sign in tab (and that it belongs to the provider you picked)."
    if "429" in text or "rate limit" in text or "ratelimit" in text or "quota" in text or "resource_exhausted" in text:
        return "The provider is limiting you (rate limit or quota). Wait a while, or lower how fast you label; check your plan."
    if "request timed out" in text:
        return ("The model did not answer in time. A slow local model is the usual cause: use a smaller model, "
                "lower 'Largest image side sent', show fewer spots per image, tick 'Turn off thinking' (or use an "
                "instruct model), and set 'Longest reply allowed' to about 1200 so a model that rambles fails fast. "
                "A bigger context window (16k, say) makes every request slower and can push the model onto the CPU, so "
                "raising it usually makes timeouts worse; keep it at the smallest size that fits (about 6144). "
                "For Ollama run `ollama ps` and check the model is on the GPU. Raising the request timeout only "
                "helps if the model is slow but steady.")
    if "connection" in text or "connect" in text or "refused" in text or "timed out" in text or "timeout" in text \
            or "name or service not known" in text or "getaddrinfo" in text:
        return ("The app could not reach the server. If it is a local server, make sure it is running and the "
                "server address in AI settings is right (Ollama: http://localhost:11434/v1).")
    if "valid json" in text or "json" in text or "reply was empty" in text or "candidates" in text and "missing" in text:
        return ("The model answered but not in the format the app needs. Try pipeline mode, tick the structured "
                "output box if your server supports it, or use a stronger model.")
    if "400" in text or "badrequest" in text or "invalid" in text:
        return ("The server rejected the request. Check the model name and settings, then press Test provider "
                "for a quick check.")
    if "500" in text or "502" in text or "503" in text or "overloaded" in text or "internalserver" in text:
        return "The server had a problem on its side. Try again in a minute; for a local server look at its own log."
    return ""


def clean_model_name(text: str | None) -> str:
    """A pasted model name, tidied: first line only, no surrounding quotes or backticks, and without a pasted
    `ollama run` / `ollama pull` in front. Spaces inside a name are left alone (the server will say if it is wrong)."""
    lines = (text or "").strip().splitlines()
    name = lines[0].strip() if lines else ""
    name = name.strip("`'\"").strip()
    lowered = name.lower()
    for prefix in ("ollama run ", "ollama pull "):
        if lowered.startswith(prefix):
            name = name[len(prefix):].strip()
            break
    return name.strip("`'\"").strip()


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} seconds"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes"
    return f"{seconds / 3600:.1f} hours"


def limit_advice(message: str, retry_after: float | None = None, now: datetime | None = None) -> str:
    """Plain words for a paused run: what was hit, when it may reopen, and what pressing Resume does."""
    lowered = (message or "").lower()
    if retry_after and retry_after > 0:
        when = (now or datetime.now()) + timedelta(seconds=retry_after)
        wait = f"The provider asked for a wait of about {_duration(retry_after)} (until about {when:%H:%M})."
    elif "per day" in lowered or "daily" in lowered or "perday" in lowered:
        wait = "This looks like a daily quota. It usually reopens when your provider's day rolls over; check its usage page."
    else:
        wait = ("The provider did not say how long. A per-minute limit reopens within a minute or two; a daily quota "
                "reopens when your provider's day rolls over. Check its usage page.")
    first = (message or "").strip().splitlines()[0][:200] if (message or "").strip() else "the provider refused the request"
    return (f"The provider's limit was reached ({first}). {wait} The image was put back in the queue and nothing is lost. "
            "Press Resume when the limit has reopened.")
