"""Answer one parked question: pick a listed choice or type an answer, optionally adding it to the taxonomy."""
from __future__ import annotations

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QLabel, QLineEdit, QVBoxLayout, QWidget)

from core.models import Question


class QuestionDialog(QDialog):
    def __init__(self, question: Question, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Question from the labeler")
        self.question = question
        self.setMinimumWidth(420)
        label = QLabel(question.text)
        label.setWordWrap(True)
        self.choice = QComboBox()
        self.choice.setEditable(True)  # a typed value is always allowed; the list is a convenience
        self.choice.addItems(question.choices)
        if question.choices:
            self.choice.setCurrentIndex(0)
        else:
            self.choice.setCurrentText("")
        self.choice.lineEdit().setPlaceholderText("Choose or type an answer")
        self.add_box = QCheckBox("Add this value to the taxonomy if it is new")
        self.add_box.setVisible(question.allow_taxonomy_add and bool(question.field_name))
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        col = QVBoxLayout(self)
        for widget in (label, self.choice, self.add_box, self.buttons):
            col.addWidget(widget)

    def answer(self) -> str:
        return self.choice.currentText().strip()

    def add_to_taxonomy(self) -> bool:
        return self.add_box.isVisible() and self.add_box.isChecked()
