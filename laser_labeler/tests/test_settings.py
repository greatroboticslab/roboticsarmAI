import pytest

from config import secrets
from config.settings import SettingsError, load_settings, save_settings


def test_defaults_load_and_validate():
    s = load_settings()
    assert s["labeling"]["table_rule"] == "only_if_no_object_hit"
    assert s["provider"]["model"] == ""  # never hardcoded


def test_user_override_merges(tmp_path):
    path = tmp_path / "user.yaml"
    path.write_text("provider:\n  model: my-model\n")
    s = load_settings(path)
    assert s["provider"]["model"] == "my-model"
    assert s["provider"]["base_url"].startswith("http")  # default kept


def test_save_round_trip(tmp_path):
    s = load_settings()
    s["provider"]["model"] = "m"
    save_settings(s, tmp_path / "sub" / "user.yaml")
    assert load_settings(tmp_path / "sub" / "user.yaml")["provider"]["model"] == "m"


def test_save_refuses_secrets(tmp_path):
    s = load_settings()
    s["provider"]["api_key"] = "sk-123"
    with pytest.raises(SettingsError):
        save_settings(s, tmp_path / "u.yaml")
    assert not (tmp_path / "u.yaml").exists()


def test_invalid_values(tmp_path):
    path = tmp_path / "u.yaml"
    path.write_text("labeling:\n  table_rule: nope\n")
    with pytest.raises(SettingsError):
        load_settings(path)


def test_secret_store_round_trip(monkeypatch):
    import keyring
    from keyring.backend import KeyringBackend

    class Memory(KeyringBackend):
        priority = 1
        def __init__(self): self.d = {}
        def set_password(self, s, u, p): self.d[(s, u)] = p
        def get_password(self, s, u): return self.d.get((s, u))
        def delete_password(self, s, u):
            if (s, u) not in self.d:
                raise keyring.errors.PasswordDeleteError("missing")
            del self.d[(s, u)]

    backend = Memory()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)

    secrets.set_secret(secrets.ROBOFLOW_KEY, "abc")
    assert secrets.get_secret(secrets.ROBOFLOW_KEY) == "abc"
    secrets.delete_secret(secrets.ROBOFLOW_KEY)
    secrets.delete_secret(secrets.ROBOFLOW_KEY)  # second delete is a no-op
    assert secrets.get_secret(secrets.ROBOFLOW_KEY) is None


def test_no_keyring_backend_gives_clear_error(monkeypatch):
    import keyring
    from keyring.backends.fail import Keyring as Fail
    monkeypatch.setattr(keyring, "get_keyring", lambda: Fail())
    with pytest.raises(secrets.SecretStoreError):
        secrets.get_secret("x")


def test_reply_limit_is_validated(tmp_path):
    import pytest as _pytest
    from config.settings import SettingsError, validate_settings
    settings = load_settings()
    assert settings["provider"]["reply_limit"] == 0
    for bad in (-1, 40000, "many", True):
        settings["provider"]["reply_limit"] = bad
        with _pytest.raises(SettingsError):
            validate_settings(settings)
    settings["provider"]["reply_limit"] = 1200
    validate_settings(settings)
