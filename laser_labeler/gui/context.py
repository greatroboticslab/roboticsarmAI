"""Shared state of the running app: settings, data folder, queue database and review service.

Tabs get one AppContext and talk to each other only through its signals, so no tab imports another.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from config import credentials
from config.settings import DEFAULTS_PATH, load_settings, save_settings
from core.estimate import estimate_run, run_stats
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
        self._settings_cache: tuple[tuple, dict] | None = None
        self._stats_cache: tuple[float, tuple] | None = None  # (when read, run_stats)
        self.queue_changed.connect(self._forget_stats)

    # ------------------------------------------------------------------ settings
    @staticmethod
    def _stamp(path: Path) -> tuple:
        try:
            st = path.stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return ()

    def settings(self) -> dict:
        """The settings, parsed again only when a settings file changed (parsing the YAML takes milliseconds, and
        the tabs ask for the settings very often). The caller gets its own copy and may change it freely."""
        stamp = (self._stamp(DEFAULTS_PATH), self._stamp(self.settings_path))
        if self._settings_cache is None or self._settings_cache[0] != stamp:
            self._settings_cache = (stamp, load_settings(self.settings_path))
        return copy.deepcopy(self._settings_cache[1])

    def update_settings(self, change: Callable[[dict], None]) -> dict:
        """Apply `change` to the full settings dict and save it. Secrets are refused by save_settings."""
        settings = self.settings()
        change(settings)
        save_settings(settings, self.settings_path)
        self._settings_cache = None
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

    def _forget_stats(self) -> None:
        self._stats_cache = None

    def estimate(self, images: int, max_age_s: float = 30.0):
        """Time and cost for `images`. What past runs measured (it means reading transcript files) is reused for
        `max_age_s` seconds, since the tick boxes ask for an estimate on every click."""
        now = time.monotonic()
        if self._stats_cache is None or now - self._stats_cache[0] > max_age_s:
            self._stats_cache = (now, run_stats(self.data_dir))
        return estimate_run(images, self.settings(), self.data_dir, stats=self._stats_cache[1])

    def prompt_version_text(self) -> str:
        active = active_version(self.versions_dir)
        return active or "draft (no saved version)"

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
