"""Shared state of the running app: settings, data folder, queue database and review service.

Tabs get one AppContext and talk to each other only through its signals, so no tab imports another.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from config import credentials
from config.settings import load_settings, save_settings
from core.estimate import estimate_run
from core.labeler.factory import make_tracker
from core.prompts.versions import active_version
from core.providers.cost import CostTracker
from core.providers.hints import clean_model_name
from core.providers.registry import build_provider
from core.queue_db import QueueDB
from core.review.service import ReviewService
from gui.paths import PROJECT_ROOT, default_user_settings_path


class AppContext(QObject):
    settings_changed = pyqtSignal()  # AI settings, escalation or project saved
    queue_changed = pyqtSignal()  # images added, labeled, approved, uploaded
    prompt_changed = pyqtSignal()  # a prompt version was saved or activated
    open_image = pyqtSignal(str)  # ask the Review tab to show this image

    def __init__(self, settings_path: str | Path | None = None, project_root: str | Path = PROJECT_ROOT,
                 data_dir: str | Path | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.root = Path(project_root)
        self.settings_path = Path(settings_path or default_user_settings_path())
        self._data_override = Path(data_dir) if data_dir else None
        self.taxonomy_dir = self.root / "taxonomy"
        self.parts_dir = self.root / "prompts" / "parts"
        self.versions_dir = self.root / "prompts" / "versions"
        self._db: QueueDB | None = None

    # ------------------------------------------------------------------ settings
    def settings(self) -> dict:
        return load_settings(self.settings_path)

    def update_settings(self, change: Callable[[dict], None]) -> dict:
        """Apply `change` to the full settings dict and save it. Secrets are refused by save_settings."""
        settings = self.settings()
        change(settings)
        save_settings(settings, self.settings_path)
        self.settings_changed.emit()
        return settings

    @property
    def data_dir(self) -> Path:
        if self._data_override:
            return self._data_override
        path = Path(self.settings()["data_dir"])
        return path if path.is_absolute() else self.root / path

    # ------------------------------------------------------------------ services
    @property
    def db(self) -> QueueDB:
        if self._db is None:
            self._db = QueueDB(self.data_dir / "queue.db")
        return self._db

    def service(self) -> ReviewService:
        return ReviewService(self.db, self.data_dir, self.taxonomy_dir, self.settings())

    def provider(self, overrides: dict | None = None):
        """The configured model. Raises credentials.MissingCredentialError or ValueError with a plain message."""
        block = {**self.settings()["provider"], **(overrides or {})}
        block["model"] = clean_model_name(block.get("model"))
        kind = block["provider"]
        key = credentials.get_key(kind)
        if not key and kind != "openai_compat":
            credentials.require_key(kind)
        return build_provider(block, key)

    def tracker(self) -> CostTracker:
        return make_tracker(self.settings()["provider"], self.data_dir)

    def estimate(self, images: int):
        return estimate_run(images, self.settings(), self.data_dir)

    def prompt_version_text(self) -> str:
        active = active_version(self.versions_dir)
        return active or "draft (no saved version)"

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
