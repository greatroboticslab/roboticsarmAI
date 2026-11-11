"""Load and save settings. Defaults ship in defaults.yaml; user overrides are merged on top.

Secrets are never stored here. save_settings refuses anything that looks like a key.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")
_SECRET_HINTS = ("api_key", "apikey", "secret", "password", "token")


class SettingsError(ValueError):
    """Raised for invalid settings."""


def load_defaults() -> dict[str, Any]:
    return yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _find_secret_keys(data: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            here = f"{path}.{key}" if path else str(key)
            if any(hint in str(key).lower() for hint in _SECRET_HINTS):
                found.append(here)
            found.extend(_find_secret_keys(value, here))
    return found


def load_settings(user_path: str | Path | None = None) -> dict[str, Any]:
    """Defaults, overridden by the user's settings file if it exists."""
    settings = load_defaults()
    if user_path and Path(user_path).is_file():
        user = yaml.safe_load(Path(user_path).read_text(encoding="utf-8")) or {}
        if not isinstance(user, dict):
            raise SettingsError(f"{user_path}: expected a mapping at the top level")
        settings = _merge(settings, user)
    validate_settings(settings)
    return settings


def validate_settings(settings: dict[str, Any]) -> None:
    from core.labeler.table_rule import RULES

    if settings["labeling"]["table_rule"] not in RULES:
        raise SettingsError(f"labeling.table_rule must be one of {RULES}")
    if settings["labeling"].get("scheme", "full") not in ("full", "components", "merged"):
        raise SettingsError("labeling.scheme must be 'full', 'components' or 'merged'")
    if settings["labeling"].get("taxonomy_match", "object") not in ("off", "object", "all"):
        raise SettingsError("labeling.taxonomy_match must be 'off', 'object' or 'all'")
    if settings["labeling"].get("model_coordinates", "auto") not in ("auto", "pixels", "thousand", "fraction"):
        raise SettingsError("labeling.model_coordinates must be 'auto', 'pixels', 'thousand' or 'fraction'")
    fraction = settings["labeling"].get("laser_box_object_fraction", 0.0)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 0.5:
        raise SettingsError("labeling.laser_box_object_fraction must be a number from 0 to 0.5")
    cap = settings["provider"].get("reply_limit", 0)
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 0 <= cap <= 32768:
        raise SettingsError("provider.reply_limit must be a number of tokens from 0 to 32768 (0 = no limit)")
    timeout = settings["provider"].get("timeout_s", 300)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 <= timeout <= 3600:
        raise SettingsError("provider.timeout_s must be a number of seconds from 0 to 3600")
    for key, low, high in (("max_candidates", 1, 12), ("crop_px", 64, 1024)):
        value = settings["run"].get(key, low)
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise SettingsError(f"run.{key} must be a whole number from {low} to {high}")
    if settings["provider"]["run_mode"] not in ("pipeline", "agent"):
        raise SettingsError("provider.run_mode must be 'pipeline' or 'agent'")
    if settings["provider"]["provider"] not in ("openai_compat", "gemini", "anthropic"):
        raise SettingsError("provider.provider must be openai_compat, gemini or anthropic")
    profiles = settings["provider"].get("profiles", {})
    if not isinstance(profiles, dict):
        raise SettingsError("provider.profiles must be a mapping of provider name to its saved model and prices")
    for name, profile in profiles.items():
        if name not in ("openai_compat", "gemini", "anthropic") or not isinstance(profile, dict) \
                or not isinstance(profile.get("model", ""), str):
            raise SettingsError(f"provider.profiles.{name}: expected a provider name with a 'model' text and optional prices")
    if settings["provider"]["daily_cost_cap_usd"] < 0:
        raise SettingsError("provider.daily_cost_cap_usd cannot be negative")


def save_settings(settings: dict[str, Any], user_path: str | Path) -> None:
    """Write settings to disk. Refuses to write anything that looks like a secret."""
    leaks = _find_secret_keys(settings)
    if leaks:
        raise SettingsError(
            f"refusing to save secrets in the settings file ({', '.join(leaks)}); "
            "API keys belong in the keyring"
        )
    validate_settings(settings)
    path = Path(user_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
