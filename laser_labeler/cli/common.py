"""Helpers shared by the CLI commands: settings, data folder, provider, spend guard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from config import credentials
from config.settings import load_settings
from core.labeler.guard import TrackerGuard
from core.providers.base import VisionProvider
from core.providers.cost import CostTracker
from core.providers.registry import build_provider
from core.schema.taxonomy import load_taxonomy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER_SETTINGS = PROJECT_ROOT / "data" / "user_settings.yaml"


def load_cli_settings(path: str | None = None) -> dict[str, Any]:
    """Defaults, then the user settings file (--settings, else data/user_settings.yaml if present)."""
    chosen = Path(path) if path else DEFAULT_USER_SETTINGS
    return load_settings(chosen if chosen.is_file() else None)


def data_dir_for(settings: dict, override: str | None = None) -> Path:
    path = Path(override or settings["data_dir"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def project_taxonomy_dir() -> Path:
    return PROJECT_ROOT / "taxonomy"


def load_project_taxonomy(directory: str | None = None):
    return load_taxonomy(directory or project_taxonomy_dir())


def make_provider(settings: dict, overrides: dict | None = None) -> VisionProvider:
    """Build the provider, getting its key from the keyring or environment (never from settings)."""
    block = {**settings["provider"], **(overrides or {})}
    kind = block["provider"]
    key = credentials.get_key(kind)
    if not key and kind != "openai_compat":
        credentials.require_key(kind)  # raises with instructions on how to provide it
    return build_provider(block, key)


def make_guard(settings: dict, data_dir: Path, overrides: dict | None = None) -> TrackerGuard:
    block = {**settings["provider"], **(overrides or {})}
    tracker = CostTracker(block.get("price_in_per_mtok", 0), block.get("price_out_per_mtok", 0),
                          block.get("daily_cost_cap_usd"), data_dir / "spend.json")
    return TrackerGuard(tracker)
