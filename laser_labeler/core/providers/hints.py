"""Turn a provider error into a plain-words cause and a next step. Pure text, no network."""
from __future__ import annotations


def failure_hint(error: str | None, provider: str = "", model: str = "") -> str:
    """One or two sentences saying what probably went wrong and what to try. Empty when nothing useful can be said."""
    text = (error or "").lower()
    if not text:
        return ""
    local = provider in ("", "openai_compat")
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
