"""Where the GUI keeps its files."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_user_settings_path() -> Path:
    """data/user_settings.yaml under the project root (the data folder is git-ignored)."""
    return PROJECT_ROOT / "data" / "user_settings.yaml"
