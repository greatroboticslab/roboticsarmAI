"""The "AI providers" group box: one ProviderRow per non-Roboflow service."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QWidget

from config.credentials import SERVICES
from gui.widgets.provider_row import ProviderRow


class ProvidersBox(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("AI providers", parent)
        layout = QVBoxLayout(self)
        self.rows = {sid: ProviderRow(sid) for sid in SERVICES if sid != "roboflow"}
        for row in self.rows.values():
            row.changed.connect(self.changed)
            layout.addWidget(row)

    def refresh(self) -> None:
        for row in self.rows.values():
            row.refresh()
