"""Label run: start, pause, stop; live image and conversation; counts, time left and spend (spec section 10)."""
from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QSpinBox,
                             QSplitter, QVBoxLayout, QWidget)
from PyQt6.QtCore import Qt

from config.credentials import CredentialError, MissingCredentialError
from core.providers.hints import failure_hint
from core.labeler.factory import build_escalation, build_label_fn
from core.labeler.guard import TrackerGuard
from core.labeler.runner import STOP_TEXT, RunSummary
from core.models import ImageLabelSet
from gui.context import AppContext
from gui.widgets.conversation_viewer import ConversationViewer
from gui.widgets.image_canvas import BoxSpec, ImageCanvas
from gui.workers.run_worker import RunWorker


def boxes_of(ls: ImageLabelSet) -> list[BoxSpec]:
    return ([BoxSpec(b.id, b.box, f"{b.id}:{b.component or b.kind}", "laser") for b in ls.laser_boxes]
            + [BoxSpec(o.id, o.box, f"{o.id}:{o.general_object}", "object") for o in ls.object_boxes])


class LabelRunTab(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None, *, provider_fn: Callable | None = None,
                 confirm_fn: Callable[[str], bool] | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._provider_fn = provider_fn or ctx.provider
        self._confirm = confirm_fn or self._ask
        self.worker: RunWorker | None = None
        self.mode = QComboBox()
        self.mode.addItem("Use my AI settings", None)
        self.mode.addItem("Pipeline mode", "pipeline")
        self.mode.addItem("Agent mode", "agent")
        self.limit = QSpinBox()
        self.limit.setRange(0, 1_000_000)
        self.limit.setSpecialValueText("All queued images")
        self.limit.setSuffix(" images")
        self.start_btn, self.pause_btn, self.stop_btn = QPushButton("Start"), QPushButton("Pause"), QPushButton("Stop")
        self.retry_btn = QPushButton("Retry failed images")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.counts = QLabel()
        self.progress_label = QLabel("Not running.")
        self.spend_bar = QProgressBar()
        self.spend_bar.setFormat("%v / %m cents of the daily cap")
        self.spend_label = QLabel()
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.failure = QLabel()  # the most recent failure and what to try
        self.failure.setWordWrap(True)
        self.failure.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.failure.setStyleSheet("color: #b3261e")
        self.canvas = ImageCanvas()
        self.viewer = ConversationViewer()
        self.viewer.setPlaceholderText("The conversation with the model appears here while a run is going.")
        top = QHBoxLayout()
        for w in (QLabel("Mode"), self.mode, QLabel("Label"), self.limit, self.start_btn, self.pause_btn, self.stop_btn, self.retry_btn):
            top.addWidget(w)
        top.addStretch(1)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.canvas)
        split.addWidget(self.viewer)
        split.setSizes([500, 400])
        col = QVBoxLayout(self)
        col.addLayout(top)
        col.addWidget(self.counts)
        col.addWidget(self.progress_label)
        col.addWidget(self.spend_label)
        col.addWidget(self.spend_bar)
        col.addWidget(self.failure)
        col.addWidget(split, 1)
        col.addWidget(self.message)
        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn.clicked.connect(self.stop)
        self.retry_btn.clicked.connect(self.retry_failed)
        ctx.queue_changed.connect(self.refresh_counts)
        ctx.settings_changed.connect(self.refresh_spend)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh_spend)
        self.refresh_counts()
        self.refresh_spend()

    # ------------------------------------------------------------------ display
    def refresh_counts(self) -> None:
        c = self.ctx.db.counts()
        self.counts.setText("  ".join(f"{name.replace('_', ' ')}: {c[name]}" for name in
                                      ("queued", "needs_input", "failed", "pending", "approved", "edited", "rejected", "uploaded")))
        self.retry_btn.setEnabled(c["failed"] > 0 and not self.running())
        self.start_btn.setEnabled(not self.running())

    def refresh_spend(self) -> None:
        spent = self.ctx.tracker().spent_today
        cap = self.ctx.settings()["provider"].get("daily_cost_cap_usd")
        paid = bool(self.ctx.settings()["provider"].get("price_in_per_mtok") or self.ctx.settings()["provider"].get("price_out_per_mtok"))
        text = f"Spent today: ${spent:.2f}" + (f" of ${cap:.2f}" if cap else "") + ("" if paid else " (no price set, so this stays at $0)")
        self.spend_label.setText(text)
        if cap:
            self.spend_bar.setRange(0, int(cap * 100))
            self.spend_bar.setValue(min(int(spent * 100), int(cap * 100)))
        self.spend_bar.setVisible(bool(cap) and paid)

    def running(self) -> bool:
        return self.worker is not None

    # ------------------------------------------------------------------ run control
    @staticmethod
    def _ask(text: str) -> bool:
        return QMessageBox.question(None, "Start labeling?", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.Yes) == QMessageBox.StandardButton.Yes

    def start(self) -> bool:
        if self.running():
            return False
        queued = self.ctx.db.counts()["queued"]
        if queued == 0:
            self.message.setText("Nothing is queued. Add images on the Select images tab first.")
            return False
        count = min(queued, self.limit.value()) if self.limit.value() else queued
        settings = self.ctx.settings()
        try:
            provider = self._provider_fn()
            escalation = build_escalation(settings, self.ctx.data_dir)
        except (MissingCredentialError, CredentialError, ValueError) as exc:
            self.message.setText(f"Cannot start: {exc}")
            return False
        estimate = self.ctx.estimate(count)
        text = f"{provider.name}/{provider.model} will label {estimate.describe()}"
        if estimate.over_cap:
            text += "\n\nThis would go over today's spend cap, so the run will stop when the cap is reached."
        if not self._confirm(text + "\n\nStart?"):
            return False
        guard, taxonomy, mode = TrackerGuard(self.ctx.tracker()), self.ctx.service().taxonomy(), self.mode.currentData()
        data_dir = self.ctx.data_dir

        def make(on_entry):
            return build_label_fn(settings, provider, taxonomy, data_dir, guard=guard, mode=mode, escalation=escalation,
                                  on_entry=on_entry)

        worker = RunWorker(self.ctx.db, make, data_dir / "sidecars", limit=self.limit.value() or None, data_dir=data_dir,
                           keep_awake=bool(settings["labeling"].get("keep_awake")), parent=self)
        worker.image_started.connect(self._image_started)
        worker.entry.connect(self._entry)
        worker.progress.connect(self._progress)
        worker.problem.connect(lambda m: self.message.setText(m))
        worker.run_finished.connect(self._finished)
        self.worker = worker
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.retry_btn.setEnabled(False)
        self.message.setText("Running. You can review finished images on the Review queue tab while this goes on.")
        self._timer.start(1500)
        worker.start()
        return True

    def show_failure(self, name: str, error: str, provider: str = "", model: str = "") -> None:
        settings = self.ctx.settings()["provider"]
        hint = failure_hint(error, provider or settings["provider"], model or settings.get("model", ""))
        text = f"Last failure ({name}): {error or 'no details were saved'}"
        self.failure.setText(text + (f"\n{hint}" if hint else ""))

    def toggle_pause(self) -> None:
        if self.worker is None:
            return
        if self.worker.paused:
            self.worker.resume()
            self.pause_btn.setText("Pause")
            self.message.setText("Running.")
        else:
            self.worker.pause()
            self.pause_btn.setText("Resume")
            self.message.setText("Pausing after the image in progress...")

    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.message.setText("Stopping after the image in progress...")
            self.stop_btn.setEnabled(False)

    def retry_failed(self) -> None:
        self.failure.clear()
        ids = [r.image_id for r in self.ctx.db.list("failed")]
        self.ctx.db.requeue(ids)
        self.message.setText(f"Put {len(ids)} failed image(s) back in the queue.")
        self.ctx.queue_changed.emit()

    # ------------------------------------------------------------------ worker signals
    def _image_started(self, image_id: str, path: str) -> None:
        self.canvas.set_image(path)
        self.canvas.set_boxes([])
        self.viewer.clear_view()
        self.progress_label.setText(f"Labeling {image_id}...")

    def _entry(self, entry: dict, image_id: str) -> None:
        self.viewer.append_entry(entry)

    def _progress(self, summary: RunSummary, row, result: ImageLabelSet | None) -> None:
        failed = self.ctx.db.get(row.image_id)
        if failed is not None and failed.status == "failed":
            self.show_failure(failed.name, failed.error or "", failed.provider or "", failed.model or "")
        if result is not None:
            self.canvas.set_boxes(boxes_of(result))
        eta = f"about {summary.eta_seconds / 60:.0f} min left" if summary.queued_left else "queue empty"
        self.progress_label.setText(
            f"Done {summary.processed}: {summary.pending} ready for review, {summary.needs_input} need your input, "
            f"{summary.failed} failed. {summary.seconds_per_image:.1f} s per image, {eta}.")
        self.ctx.queue_changed.emit()

    def _finished(self, summary: RunSummary) -> None:
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self._timer.stop()
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        why = STOP_TEXT.get(summary.stopped, summary.stopped)
        self.message.setText(f"Finished {summary.processed} image(s)." + (f" The run {why}." if summary.stopped else ""))
        self.refresh_counts()
        self.refresh_spend()
        self.ctx.queue_changed.emit()
