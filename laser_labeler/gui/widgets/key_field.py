"""Password-style key entry with a Show checkbox."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLineEdit, QWidget


class KeyField(QWidget):
    """Hidden by default; the key is never kept anywhere but the line edit."""
    textChanged = pyqtSignal()
    returnPressed = pyqtSignal()

    def __init__(self, placeholder: str = "Paste key here", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit.setPlaceholderText(placeholder)
        self.show_box = QCheckBox("Show")
        self.show_box.toggled.connect(self._toggle)
        self.edit.textChanged.connect(lambda _t: self.textChanged.emit())
        self.edit.returnPressed.connect(self.returnPressed)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit, 1)
        row.addWidget(self.show_box)

    def _toggle(self, shown: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
        self.edit.setEchoMode(mode)

    def text(self) -> str:
        return self.edit.text()

    def clear(self) -> None:
        """Empty the field and hide the text again."""
        self.edit.clear()
        self.show_box.setChecked(False)
