"""Sign in tab: enter and verify the Roboflow key and the AI provider keys."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from core.roboflow.auth import list_projects, verify_roboflow_key
from gui.paths import default_user_settings_path
from gui.widgets.providers_box import ProvidersBox
from gui.widgets.roboflow_box import RoboflowBox


class SignInTab(QWidget):
    """Only talks to config.credentials and core.roboflow; holds no key itself."""
    credentials_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None, *, user_settings_path: Path | None = None,
                 verify_fn: Callable = verify_roboflow_key, projects_fn: Callable = list_projects) -> None:
        super().__init__(parent)
        self.user_settings_path = Path(user_settings_path or default_user_settings_path())
        self.roboflow = RoboflowBox(verify_fn, projects_fn, self.user_settings_path)
        self.providers = ProvidersBox()
        layout = QVBoxLayout(self)
        layout.addWidget(self.roboflow)
        layout.addWidget(self.providers)
        layout.addStretch(1)
        self.roboflow.changed.connect(self.credentials_changed)
        self.providers.changed.connect(self.credentials_changed)

    def is_busy(self) -> bool:
        """True while a network call is running in the background."""
        return self.roboflow.is_busy()

    def refresh(self) -> None:
        self.roboflow.refresh()
        self.providers.refresh()
