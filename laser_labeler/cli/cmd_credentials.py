"""`credentials` subcommand: set, clear, status and verify keys without the GUI.

Keys are read with getpass, never from argv, so they stay out of shell history.
Exit codes: 0 ok, 1 failure, 2 (verify only) the key could not be checked.
"""
from __future__ import annotations

import argparse
import getpass
import sys

from config import credentials
from config.credentials import CredentialError, MissingCredentialError
from config.secrets import SecretStoreError
from core.roboflow.auth import verify_roboflow_key


def add_parser(subparsers) -> argparse.ArgumentParser:
    """Register `credentials` on the main parser's subparsers."""
    p = subparsers.add_parser("credentials", help="manage API keys (stored in the OS keyring)")
    p.set_defaults(func=run)
    actions = p.add_subparsers(dest="credentials_action", required=True, metavar="action")
    services = sorted(credentials.SERVICES)
    for name, text in (("set", "save a key (asked for with hidden input)"),
                       ("clear", "remove a saved key")):
        sp = actions.add_parser(name, help=text)
        sp.add_argument("service", choices=services)
    actions.add_parser("status", help="show where each key comes from (masked)")
    vp = actions.add_parser("verify", help="check a saved key with the service")
    vp.add_argument("service", choices=["roboflow"])
    return p


def _out(text: str = "", stream=None) -> None:
    """Print, replacing the mask dots if the terminal cannot show them."""
    stream = stream or sys.stdout
    try:
        text.encode(stream.encoding or "ascii")
    except UnicodeEncodeError:
        text = text.replace("•", "*")
    print(text, file=stream)


def _set(service: str) -> int:
    label = credentials.SERVICES[service].label
    key = getpass.getpass(f"{label} key (hidden, press Enter when done): ")
    if not key.strip() and not credentials.SERVICES[service].required:
        _out("Nothing entered, nothing saved (this service usually needs no key).")
        return 0
    credentials.save_key(service, key)  # CredentialError if empty or contains spaces
    masked = credentials.mask_key(credentials.clean_key(key))
    _out(f"Saved the {label} key in the system keyring: {masked}")
    return 0


def _clear(service: str) -> int:
    credentials.clear_key(service)
    _out(f"Removed the saved {credentials.SERVICES[service].label} key from the keyring.")
    if credentials.key_source(service) == "environment":
        _out("Note: an environment variable still provides a key.")
    return 0


def _status() -> int:
    rows = credentials.status()
    width = max(len(r.label) for r in rows)
    for r in rows:
        need = "required" if r.required else "optional"
        _out(f"{r.label:<{width}}  {r.source:<11}  {r.masked or '-':<9}  {need}")
    return 0


def _verify(service: str) -> int:
    key = credentials.get_key(service)
    if not key:
        raise MissingCredentialError(
            f"No Roboflow key is saved. Run: python -m cli credentials set {service}")
    result = verify_roboflow_key(key)
    _out(result.message)
    if result.workspace_ids:
        _out("Workspace(s): " + ", ".join(result.workspace_ids))
    return {"valid": 0, "invalid": 1}.get(result.state, 2)


def run(args: argparse.Namespace) -> int:
    action = args.credentials_action
    try:
        if action == "set":
            return _set(args.service)
        if action == "clear":
            return _clear(args.service)
        if action == "verify":
            return _verify(args.service)
        return _status()
    except SecretStoreError as exc:
        env = credentials.SERVICES[getattr(args, "service", "roboflow")].env_vars[0]
        _out(f"No system keyring was found, so keys cannot be saved here: {exc}\n"
             f"Instead set the environment variable {env} before running the app.", sys.stderr)
    except (CredentialError, MissingCredentialError) as exc:
        _out(f"Error: {exc}", sys.stderr)
    return 1
