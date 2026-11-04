"""Build a VisionProvider from the `provider:` settings block. Adapters import lazily."""
from __future__ import annotations

from typing import Any

from core.providers.hints import clean_model_name

from core.providers.base import VisionProvider

PROVIDER_CHOICES = ("openai_compat", "gemini", "anthropic")


def build_provider(provider_settings: dict[str, Any], api_key: str | None,
                   client: Any = None) -> VisionProvider:
    """Create the adapter named by settings["provider"].

    Raises ValueError for an unknown provider or an empty model name. `client` is an
    optional pre-built SDK client (used by tests).
    """
    kind = (provider_settings.get("provider") or "").strip()
    if kind not in PROVIDER_CHOICES:
        raise ValueError(f"Unknown provider {kind!r}; choose one of {', '.join(PROVIDER_CHOICES)}")
    model = clean_model_name(provider_settings.get("model"))
    if not model:
        raise ValueError("No model set: choose a model in AI settings")
    temperature = provider_settings.get("temperature", 0.0)
    if kind == "openai_compat":
        from core.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(model, api_key, provider_settings.get("base_url") or None,
                                    temperature, client,
                                    structured_output=bool(provider_settings.get("structured_output")))
    if kind == "gemini":
        from core.providers.gemini import GeminiProvider
        return GeminiProvider(model, api_key, temperature, client)
    from core.providers.anthropic import AnthropicProvider
    extra = {"max_tokens": provider_settings["max_tokens"]} if provider_settings.get("max_tokens") else {}
    return AnthropicProvider(model, api_key, temperature=temperature, client=client, **extra)
