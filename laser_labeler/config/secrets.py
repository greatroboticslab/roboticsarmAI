"""API keys live in the OS keyring only. Never in settings files, logs, sidecars or PDFs."""
from __future__ import annotations

SERVICE = "laser_labeler"
ROBOFLOW_KEY = "roboflow_api_key"


class SecretStoreError(RuntimeError):
    """Raised when no usable keyring backend exists."""


def _keyring():
    try:
        import keyring
        from keyring.backends.fail import Keyring as FailKeyring
    except ImportError as exc:  # pragma: no cover
        raise SecretStoreError("the 'keyring' package is not installed") from exc
    if isinstance(keyring.get_keyring(), FailKeyring):
        raise SecretStoreError(
            "no system keyring backend is available on this machine; "
            "install one (Windows Credential Manager, macOS Keychain or Secret Service)"
        )
    return keyring


def provider_key_name(provider: str) -> str:
    return f"provider_{provider}_api_key"


def set_secret(name: str, value: str) -> None:
    _keyring().set_password(SERVICE, name, value)


def get_secret(name: str) -> str | None:
    return _keyring().get_password(SERVICE, name)


def delete_secret(name: str) -> None:
    kr = _keyring()
    try:
        kr.delete_password(SERVICE, name)
    except kr.errors.PasswordDeleteError:
        pass  # already gone
