import argparse
import getpass

import keyring
import pytest
from keyring.backend import KeyringBackend

from cli import cmd_credentials
from config import credentials
from core.roboflow.auth import VerifyResult

KEY = "rf_CLISECRETKEY_424242"


class Memory(KeyringBackend):
    priority = 1

    def __init__(self): self.d = {}
    def set_password(self, s, u, p): self.d[(s, u)] = p
    def get_password(self, s, u): return self.d.get((s, u))

    def delete_password(self, s, u):
        if (s, u) not in self.d:
            raise keyring.errors.PasswordDeleteError("missing")
        del self.d[(s, u)]


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch):
    backend = Memory()
    monkeypatch.setattr(keyring, "get_keyring", lambda: backend)
    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)
    for info in credentials.SERVICES.values():
        for name in info.env_vars:
            monkeypatch.delenv(name, raising=False)
    return backend


def run(*argv, monkeypatch=None):
    parser = argparse.ArgumentParser()
    cmd_credentials.add_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["credentials", *argv])
    return args.func(args)


def type_key(monkeypatch, value):
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": value)


def test_set_stores_key_and_never_prints_it(monkeypatch, capsys):
    type_key(monkeypatch, KEY)
    assert run("set", "roboflow") == 0
    out = capsys.readouterr()
    assert credentials.get_key("roboflow") == KEY
    assert KEY not in out.out + out.err and "424242"[-4:] in out.out


def test_key_is_not_accepted_on_the_command_line(capsys):
    with pytest.raises(SystemExit):
        run("set", "roboflow", KEY)
    assert KEY not in capsys.readouterr().out


def test_set_rejects_empty_and_spaced_keys(monkeypatch, capsys):
    type_key(monkeypatch, "  ")
    assert run("set", "gemini") == 1
    type_key(monkeypatch, "has space inside")
    assert run("set", "gemini") == 1
    out = capsys.readouterr()
    assert "Error" in out.err and "has space inside" not in out.out + out.err
    assert credentials.get_key("gemini") is None


def test_set_openai_compat_empty_is_fine(monkeypatch, capsys, fake_keyring):
    type_key(monkeypatch, "")
    assert run("set", "openai_compat") == 0
    assert "Nothing entered" in capsys.readouterr().out and fake_keyring.d == {}


def test_clear(monkeypatch, capsys):
    type_key(monkeypatch, KEY)
    run("set", "anthropic")
    assert run("clear", "anthropic") == 0
    assert credentials.get_key("anthropic") is None
    assert run("clear", "anthropic") == 0  # clearing twice is fine


def test_status_table_is_masked(monkeypatch, capsys):
    type_key(monkeypatch, KEY)
    run("set", "roboflow")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-ENVIRONMENTKEY-7777")
    capsys.readouterr()
    assert run("status") == 0
    out = capsys.readouterr().out
    assert "keyring" in out and "environment" in out and "missing" in out
    assert "4242" in out and "7777" in out
    assert KEY not in out and "ENVIRONMENTKEY" not in out


@pytest.mark.parametrize("state,code", [("valid", 0), ("invalid", 1), ("unreachable", 2)])
def test_verify_exit_codes(monkeypatch, capsys, state, code):
    type_key(monkeypatch, KEY)
    run("set", "roboflow")
    seen = []
    monkeypatch.setattr(cmd_credentials, "verify_roboflow_key",
                        lambda k: seen.append(k) or VerifyResult(state == "valid", state, ["ws"], f"msg {state}"))
    assert run("verify", "roboflow") == code
    out = capsys.readouterr()
    assert seen == [KEY] and KEY not in out.out + out.err and f"msg {state}" in out.out


def test_verify_without_key(capsys):
    assert run("verify", "roboflow") == 1
    assert "No Roboflow key is saved" in capsys.readouterr().err


def test_no_keyring_message(monkeypatch, capsys):
    from keyring.backends.fail import Keyring as Fail
    monkeypatch.setattr(keyring, "get_keyring", lambda: Fail())
    type_key(monkeypatch, KEY)
    assert run("set", "anthropic") == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and KEY not in err
