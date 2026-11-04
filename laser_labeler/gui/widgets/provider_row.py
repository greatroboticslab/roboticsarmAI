"""One AI provider: key field, Save, Remove and a status line."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from config import credentials
from config.credentials import CredentialError
from config.secrets import SecretStoreError
from gui.widgets.key_field import KeyField
from gui.widgets.messages import describe_error, describe_status

LOCAL_NOTE = "Local servers (Ollama, LM Studio) usually need no key; leave empty."


class ProviderRow(QWidget):
    changed = pyqtSignal()

    def __init__(self, service: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        info = credentials.SERVICES[service]
        self.key_field = KeyField()
        self.save_btn, self.remove_btn = QPushButton("Save"), QPushButton("Remove")
        self.status_label, self.message_label = QLabel(), QLabel()
        self.message_label.setWordWrap(True)
        self.status_label.setWordWrap(True)
        buttons = QHBoxLayout()
        buttons.addWidget(self.key_field, 1)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.remove_btn)
        col = QVBoxLayout(self)
        title = QLabel(f"<b>{info.label}</b>")
        col.addWidget(title)
        if not info.required:
            note = QLabel(LOCAL_NOTE)
            note.setWordWrap(True)
            col.addWidget(note)
        col.addLayout(buttons)
        col.addWidget(self.status_label)
        col.addWidget(self.message_label)
        self.save_btn.clicked.connect(self.save)
        self.remove_btn.clicked.connect(self.remove)
        self.key_field.returnPressed.connect(self.save)
        self.refresh()

    def refresh(self) -> None:
        st = next(s for s in credentials.status() if s.service == self.service)
        self.status_label.setText(describe_status(st))

    def save(self) -> None:
        if not self.key_field.text().strip():
            self.message_label.setText("Nothing to save: the key field is empty.")
            return
        try:
            credentials.save_key(self.service, self.key_field.text())
        except (CredentialError, SecretStoreError) as exc:
            self.message_label.setText(describe_error(exc, self.service))
            return
        self.key_field.clear()
        self.message_label.setText("Key saved.")
        self.refresh()
        self.changed.emit()

    def remove(self) -> None:
        try:
            credentials.clear_key(self.service)
        except (CredentialError, SecretStoreError) as exc:
            self.message_label.setText(describe_error(exc, self.service))
            return
        self.message_label.setText("Saved key removed.")
        self.refresh()
        self.changed.emit()
