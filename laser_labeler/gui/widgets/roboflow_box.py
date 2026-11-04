"""The "Roboflow" group box: key entry, verification, workspace/project choice."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from config import credentials
from config.credentials import CredentialError
from config.secrets import SecretStoreError
from gui.widgets.key_field import KeyField
from gui.widgets.messages import describe_error, describe_status
from gui.widgets.project_choice import ProjectChoice
from gui.workers.verify_worker import start_call

SERVICE = "roboflow"


class RoboflowBox(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, verify_fn: Callable, projects_fn: Callable, settings_path: Path,
                 parent: QWidget | None = None) -> None:
        super().__init__("Roboflow", parent)
        self._verify_fn, self._runner = verify_fn, None
        self.key_field = KeyField("Paste your Roboflow API key")
        self.verify_btn = QPushButton("Verify and save")
        self.anyway_btn = QPushButton("Save anyway (not checked)")
        self.remove_btn = QPushButton("Remove saved key")
        self.status_label, self.message_label = QLabel(), QLabel()
        for label in (self.status_label, self.message_label):
            label.setWordWrap(True)
        self.choice = ProjectChoice(projects_fn, settings_path)
        self.choice.saved.connect(self.changed)
        buttons = QHBoxLayout()
        for btn in (self.verify_btn, self.anyway_btn, self.remove_btn):
            buttons.addWidget(btn)
        col = QVBoxLayout(self)
        col.addWidget(self.key_field)
        col.addLayout(buttons)
        for widget in (self.status_label, self.message_label, self.choice):
            col.addWidget(widget)
        self.anyway_btn.hide()
        self.verify_btn.clicked.connect(self.verify)
        self.key_field.returnPressed.connect(self.verify)
        self.key_field.textChanged.connect(self.anyway_btn.hide)
        self.anyway_btn.clicked.connect(self.save_unchecked)
        self.remove_btn.clicked.connect(self.remove)
        self.refresh()

    def is_busy(self) -> bool:
        return self._runner is not None or self.choice.is_busy()

    def refresh(self) -> None:
        st = next(s for s in credentials.status() if s.service == SERVICE)
        self.status_label.setText(describe_status(st))
        self.choice.setEnabled(st.source != "missing")

    def verify(self) -> None:
        if self._runner is not None:
            return
        try:
            key = credentials.clean_key(self.key_field.text())
        except CredentialError as exc:
            self.message_label.setText(describe_error(exc))
            return
        self.anyway_btn.hide()
        self.verify_btn.setEnabled(False)
        self.message_label.setText("Checking the key with Roboflow...")
        self._runner = start_call(self._verify_fn, (key,), lambda r: self._on_verified(r, key),
                                  self._on_failed, self)

    def _end_verify(self) -> None:
        self._runner = None
        self.verify_btn.setEnabled(True)

    def _on_failed(self, message: str) -> None:
        self._end_verify()
        self.message_label.setText(f"The key could not be checked: {message}")

    def _on_verified(self, result, key: str) -> None:
        self._end_verify()
        if result.state == "valid":
            if self._store(key, "Key verified and saved."):
                self.choice.set_workspaces(result.workspace_ids)
                self.choice.load_projects()
        elif result.state == "invalid":
            self.message_label.setText(f"{result.message} The key was not saved; check it and try again.")
        else:
            self.message_label.setText(
                f"{result.message} The key was NOT saved. If you are offline you can save it "
                "anyway and check it later.")
            self.anyway_btn.show()

    def _store(self, key: str, done_text: str) -> bool:
        try:
            credentials.save_key(SERVICE, key)
        except (CredentialError, SecretStoreError) as exc:
            self.message_label.setText(describe_error(exc, SERVICE))
            return False
        self.key_field.clear()
        self.anyway_btn.hide()
        self.message_label.setText(done_text)
        self.refresh()
        self.changed.emit()
        return True

    def save_unchecked(self) -> None:
        self._store(self.key_field.text(), "Key saved without checking. Verify it once you are online.")

    def remove(self) -> None:
        try:
            credentials.clear_key(SERVICE)
        except (CredentialError, SecretStoreError) as exc:
            self.message_label.setText(describe_error(exc, SERVICE))
            return
        self.anyway_btn.hide()
        self.message_label.setText("Saved key removed.")
        self.refresh()
        self.changed.emit()
