"""Shows exactly what an upload will send, then sends it (dry run by default; spec section 9.3)."""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton,
                             QVBoxLayout, QWidget)

from core.roboflow.upload import UploadPlan, UploadSummary
from gui.workers.verify_worker import start_call


class UploadDialog(QDialog):
    """`plan_fn(overwrite)` builds the plan; `send_fn(plan, dry_run)` runs it and returns an UploadSummary;
    `done_fn(summary)` is called after a real upload so the caller can update statuses."""
    uploaded = pyqtSignal(list)

    def __init__(self, plan_fn: Callable[[bool], UploadPlan], send_fn: Callable[[UploadPlan, bool], UploadSummary],
                 done_fn: Callable[[UploadSummary], None] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Upload approved labels to Roboflow")
        self.resize(760, 560)
        self._plan_fn, self._send_fn, self._done_fn = plan_fn, send_fn, done_fn
        self._plan: UploadPlan | None = None
        self._runner = None
        self.busy = False
        self.dry_run = QCheckBox("Dry run: show what would be sent, send nothing")
        self.dry_run.setChecked(True)
        self.note = QLabel("Only labels are sent, to images that are already on Roboflow; no image is uploaded again. "
                           "Tick Overwrite to replace labels an image already has (otherwise Roboflow refuses with 409).")
        self.note.setWordWrap(True)
        self.overwrite = QCheckBox("Overwrite annotations that already exist on Roboflow")
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Consolas, Menlo, monospace", 9))
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.go = QPushButton()
        self.close_btn = QPushButton("Close")
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.go)
        buttons.addWidget(self.close_btn)
        col = QVBoxLayout(self)
        for widget in (self.note, self.dry_run, self.overwrite, self.text, self.status):
            col.addWidget(widget)
        col.addLayout(buttons)
        self.dry_run.toggled.connect(self._refresh_button)
        self.overwrite.toggled.connect(lambda _v: self.rebuild())
        self.go.clicked.connect(self.start)
        self.close_btn.clicked.connect(self.reject)
        self.rebuild()

    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        try:
            self._plan = self._plan_fn(self.overwrite.isChecked())
        except Exception as exc:
            self._plan = None
            self.text.setPlainText(f"The upload plan could not be built: {exc}")
        else:
            self.text.setPlainText(self._plan.describe())
        self._refresh_button()

    def _refresh_button(self) -> None:
        count = len(self._plan.items) if self._plan else 0
        self.go.setText("Show dry run" if self.dry_run.isChecked() else f"Upload {count} image(s) now")
        self.go.setEnabled(bool(self._plan) and count > 0 and not self.busy)
        self.overwrite.setEnabled(not self.busy)
        self.dry_run.setEnabled(not self.busy)

    def start(self) -> None:
        if self._plan is None or self.busy:
            return
        dry = self.dry_run.isChecked()
        if not dry:
            ask = QMessageBox.question(
                self, "Upload now?", f"Send {len(self._plan.items)} image(s) to {self._plan.workspace}/{self._plan.project}?"
                + ("\n\nExisting annotations WILL BE REPLACED." if self._plan.overwrite else ""),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if ask != QMessageBox.StandardButton.Yes:
                return
        self.busy = True
        self.status.setText("Working...")
        self._refresh_button()
        plan = self._plan
        self._runner = start_call(self._send_fn, (plan, dry), self._ok, self._fail, self)

    def _ok(self, summary: UploadSummary) -> None:
        self.busy = False
        if summary.dry_run:
            self.status.setText("Dry run finished. Nothing was sent. Untick 'Dry run' to upload for real.")
        else:
            bits = [f"uploaded {len(summary.uploaded)}", f"failed {len(summary.failed)}", f"skipped {len(summary.skipped)}"]
            if summary.tag_failed:
                bits.append(f"tags failed on {len(summary.tag_failed)}")
            self.status.setText("Upload finished: " + ", ".join(bits) + ".")
            lines = [f"failed {i}: {why}" for i, why in summary.failed.items()]
            lines += [f"skipped {i}: {why}" for i, why in summary.skipped.items()]
            lines += [f"tags not added to {i}: {why}" for i, why in summary.tag_failed.items()]
            if lines:
                self.text.appendPlainText("\n" + "\n".join(lines))
            if self._done_fn:
                self._done_fn(summary)
            self.uploaded.emit(list(summary.uploaded))
            self.rebuild()
        self._refresh_button()

    def _fail(self, message: str) -> None:
        self.busy = False
        self.status.setText(f"The upload stopped: {message}")
        self._refresh_button()
