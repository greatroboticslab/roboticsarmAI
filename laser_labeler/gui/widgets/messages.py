"""Plain-words text for statuses and errors. Never contains a full key."""
from __future__ import annotations

from config.credentials import SERVICES, CredentialError, CredentialStatus
from config.secrets import SecretStoreError
from config.settings import SettingsError


def env_names(service: str) -> str:
    info = SERVICES.get(service)
    return " or ".join(info.env_vars) if info else "the matching environment variable"


def describe_error(exc: Exception, service: str | None = None) -> str:
    if isinstance(exc, SecretStoreError):
        return (f"No system keyring was found on this machine, so the key cannot be saved here. "
                f"Instead, set the environment variable {env_names(service or '')} before "
                f"starting the app. ({exc})")
    if isinstance(exc, CredentialError):
        return f"That key cannot be used: {exc}."
    if isinstance(exc, SettingsError):
        return f"The settings could not be saved: {exc}."
    return f"Something went wrong: {exc}"


def describe_status(st: CredentialStatus) -> str:
    if st.source == "keyring":
        return f"Saved in the system keyring: {st.masked}"
    if st.source == "environment":
        return f"Taken from environment variable {env_names(st.service)}: {st.masked}"
    return "No key saved" + ("" if st.required else " (optional)")
