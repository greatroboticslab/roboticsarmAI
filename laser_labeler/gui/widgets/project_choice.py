"""Workspace / project / version pickers, saved to the user settings file."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QFormLayout, QLabel, QLineEdit, QPushButton, QWidget)

from config import credentials
from config.settings import SettingsError, load_defaults, load_settings, save_settings
from gui.widgets.messages import describe_error
from gui.workers.verify_worker import start_call


class ProjectChoice(QWidget):
    """Editable dropdowns, so a name can be typed when Roboflow could not be asked."""
    saved = pyqtSignal()

    def __init__(self, projects_fn: Callable, settings_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._projects_fn, self._path = projects_fn, Path(settings_path)
        self._runner = None
        self.workspace = QComboBox()
        self.project = QComboBox()
        self.version = QLineEdit()
        for combo in (self.workspace, self.project):
            combo.setEditable(True)
        self.version.setPlaceholderText("optional, e.g. 3")
        self.load_btn, self.save_btn = QPushButton("Load projects"), QPushButton("Save choice")
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Workspace", self.workspace)
        form.addRow("Project", self.project)
        form.addRow("Version", self.version)
        form.addRow(self.load_btn, self.save_btn)
        form.addRow(self.message_label)
        self.workspace.activated.connect(lambda _i: self.load_projects())
        self.load_btn.clicked.connect(self.load_projects)
        self.save_btn.clicked.connect(self.save_choice)
        self._prefill()

    def is_busy(self) -> bool:
        return self._runner is not None

    def _settings(self) -> dict:
        try:
            return load_settings(self._path)
        except Exception:  # a broken settings file must not break sign-in
            return load_defaults()

    def _prefill(self) -> None:
        rf = self._settings()["roboflow"]
        self.workspace.setCurrentText(rf.get("workspace") or "")
        self.project.setCurrentText(rf.get("project") or "")
        self.version.setText("" if rf.get("version") is None else str(rf["version"]))

    def set_workspaces(self, ids: list[str]) -> None:
        keep = self.workspace.currentText().strip()
        self.workspace.clear()
        self.workspace.addItems(ids)
        self.workspace.setCurrentText(keep if keep in ids or not ids else ids[0])

    def load_projects(self) -> None:
        key, ws = credentials.get_key("roboflow"), self.workspace.currentText().strip()
        if not key or not ws:
            self.message_label.setText("Sign in and enter a workspace first.")
            return
        self.load_btn.setEnabled(False)
        self.message_label.setText("Loading projects...")
        self._runner = start_call(self._projects_fn, (ws, key), self._on_projects, self._on_failed, self)

    def _done(self) -> None:
        self._runner = None
        self.load_btn.setEnabled(True)

    def _on_projects(self, projects: list) -> None:
        self._done()
        keep = self.project.currentText().strip()
        self.project.clear()
        for p in projects:
            self.project.addItem(f"{p.name} ({p.id})", p.id)
        self.project.setCurrentText(keep)
        self.message_label.setText(f"{len(projects)} project(s) found." if projects else
                                   "No projects listed; type the project name yourself.")

    def _on_failed(self, message: str) -> None:
        self._done()
        self.message_label.setText(f"Could not load projects: {message}")

    def _project_value(self) -> str:
        text, idx = self.project.currentText().strip(), self.project.currentIndex()
        if idx >= 0 and text == self.project.itemText(idx):
            return str(self.project.itemData(idx))
        return text

    def save_choice(self) -> None:
        version = self.version.text().strip()
        try:
            settings = self._settings()
            settings["roboflow"].update(
                workspace=self.workspace.currentText().strip(), project=self._project_value(),
                version=int(version) if version.isdigit() else (version or None))
            save_settings(settings, self._path)
        except (SettingsError, OSError) as exc:
            self.message_label.setText(describe_error(exc))
            return
        self.message_label.setText("Choice saved.")
        self.saved.emit()
