"""Sign-in storage for API keys: Roboflow and the AI providers.

Rules:
- Nothing is built in. The user must provide every key.
- Keys are saved in the OS keyring only (config/secrets.py), never in files or logs.
- An environment variable is accepted as a fallback, so headless machines work.
  The keyring wins when both exist.
- Local OpenAI-compatible servers (Ollama, LM Studio) usually need no key, so that
  service is not marked required.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from config import secrets
from config.secrets import SecretStoreError


class CredentialError(ValueError):
    """The key the user entered is not acceptable."""


class MissingCredentialError(RuntimeError):
    """A key is needed but none has been provided."""


@dataclass(frozen=True)
class ServiceInfo:
    id: str
    label: str
    secret_name: str
    env_vars: tuple[str, ...]
    required: bool


SERVICES: dict[str, ServiceInfo] = {
    s.id: s for s in (
        ServiceInfo("roboflow", "Roboflow", secrets.ROBOFLOW_KEY, ("ROBOFLOW_API_KEY",), True),
        ServiceInfo("openai_compat", "OpenAI-compatible (local server, OpenAI, OpenRouter)",
                    secrets.provider_key_name("openai_compat"), ("OPENAI_API_KEY",), False),
        ServiceInfo("gemini", "Google Gemini",
                    secrets.provider_key_name("gemini"), ("GEMINI_API_KEY", "GOOGLE_API_KEY"), True),
        ServiceInfo("anthropic", "Claude (Anthropic)",
                    secrets.provider_key_name("anthropic"), ("ANTHROPIC_API_KEY",), True),
    )
}


@dataclass(frozen=True)
class CredentialStatus:
    service: str
    label: str
    source: str  # "keyring", "environment" or "missing"
    masked: str  # safe to show; empty when missing
    required: bool


def _info(service: str) -> ServiceInfo:
    try:
        return SERVICES[service]
    except KeyError:
        raise CredentialError(f"unknown service {service!r}; expected one of {sorted(SERVICES)}") from None


def clean_key(value: str) -> str:
    """Strip surrounding whitespace; reject empty keys and keys with inner whitespace."""
    key = (value or "").strip()
    if not key:
        raise CredentialError("the key is empty")
    if any(ch.isspace() for ch in key):
        raise CredentialError("the key contains spaces or line breaks; paste only the key")
    return key


def mask_key(value: str | None) -> str:
    """Show only the last 4 characters of long keys, nothing for short ones."""
    if not value:
        return ""
    return "•" * 4 + value[-4:] if len(value) >= 12 else "•" * 4


def save_key(service: str, value: str) -> None:
    """Store a key in the keyring. Raises SecretStoreError if no keyring is available."""
    secrets.set_secret(_info(service).secret_name, clean_key(value))


def clear_key(service: str) -> None:
    """Remove a stored key (an environment variable, if any, is not touched)."""
    secrets.delete_secret(_info(service).secret_name)


def _keyring_value(info: ServiceInfo) -> str | None:
    try:
        return secrets.get_secret(info.secret_name)
    except SecretStoreError:
        return None  # no keyring on this machine: fall back to the environment


def _env_value(info: ServiceInfo) -> str | None:
    for name in info.env_vars:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def get_key(service: str) -> str | None:
    info = _info(service)
    return _keyring_value(info) or _env_value(info)


def require_key(service: str) -> str:
    """Return the key or raise MissingCredentialError saying how to provide it."""
    key = get_key(service)
    if key:
        return key
    info = _info(service)
    raise MissingCredentialError(
        f"No {info.label} key has been provided. Add it on the Sign in tab, run "
        f"'python -m cli credentials set {service}', or set {info.env_vars[0]}."
    )


def key_source(service: str) -> str:
    info = _info(service)
    if _keyring_value(info):
        return "keyring"
    return "environment" if _env_value(info) else "missing"


def status() -> list[CredentialStatus]:
    """Where each key comes from, with a masked preview. Never returns a full key."""
    rows = []
    for info in SERVICES.values():
        source = key_source(info.id)
        masked = mask_key(get_key(info.id)) if source != "missing" else ""
        rows.append(CredentialStatus(info.id, info.label, source, masked, info.required))
    return rows
