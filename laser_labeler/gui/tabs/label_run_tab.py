"""Label run: start, pause, stop; live image and conversation; counts, time left and spend (spec section 10)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QFrame
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMenu, QMessageBox, QProgressBar, QPushButton, QSpinBox,
                             QSplitter, QVBoxLayout, QWidget)
from PyQt6.QtCore import Qt

from config.credentials import CredentialError, MissingCredentialError
from core.providers.hints import failure_hint, limit_advice
from core.review.reasons import describe_summary, reason_line, summarize
from core.reports.sidecar import read_sidecar
from core.labeler.factory import build_escalation, build_label_fn
from core.labeler.guard import TrackerGuard
from core.labeler.runner import STOP_TEXT, RunSummary
from core.models import ImageLabelSet
from gui.context import AppContext
from gui.widgets.conversation_viewer import ConversationViewer
from gui.widgets.image_canvas import BoxSpec, ImageCanvas
from gui.workers.run_worker import RunWorker


def laser_title(ls: ImageLabelSet, box) -> str:
    """What a laser box is called on the label: the merged object + component name under "merged", else its class."""
    if ls.scheme == "merged" and box.component:
        return ls.laser_class_name(box)
    return box.component or box.kind


def boxes_of(ls: ImageLabelSet) -> list[BoxSpec]:
    carried = ls.merged_object_ids()  # under "merged" these objects live inside their laser box's label
    return ([BoxSpec(b.id, b.box, f"{b.id}:{laser_title(ls, b)}", "laser") for b in ls.laser_boxes]
            + [BoxSpec(o.id, o.box, f"{o.id}:{o.general_object}" + (" (in laser label)" if o.id in carried else ""),
                       "object") for o in ls.object_boxes])


class LabelRunTab(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None, *, provider_fn: Callable | None = None,
                 confirm_fn: Callable[[str], bool] | None = None,
                 clear_confirm_fn: Callable[[str], bool] | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._provider_fn = provider_fn or ctx.provider
        self._confirm = confirm_fn or self._ask
        self._confirm_clear = clear_confirm_fn or self._ask_clear
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
        self.clear_btn = QPushButton("Clear queue...")
        self.clear_menu = QMenu(self.clear_btn)
        self.clear_waiting_action = self.clear_menu.addAction("Remove images still waiting to be labeled")
        self.clear_failed_action = self.clear_menu.addAction("Remove failed images")
        self.clear_all_action = self.clear_menu.addAction("Remove everything not uploaded yet...")
        self.clear_btn.setMenu(self.clear_menu)
        self.clear_btn.setToolTip("Take images out of the queue. Image files and saved results stay on disk; "
                                  "images already uploaded are never touched.")
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
        self.notice = QLabel()  # the latest image that needs input, and why
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #9a6700")
        # shown while the run waits for a provider rate limit / quota to reopen
        self.limit_box = QFrame()
        self.limit_box.setStyleSheet("QFrame { background: #fff4d6; border: 1px solid #d4a72c; border-radius: 4px; }"
                                     "QLabel { border: none; }")
        self.limit_text = QLabel()
        self.limit_text.setWordWrap(True)
        self.limit_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.limit_resume_btn = QPushButton("Resume now")
        self.limit_stop_btn = QPushButton("Stop the run")
        limit_row = QHBoxLayout(self.limit_box)
        limit_row.addWidget(self.limit_text, 1)
        limit_row.addWidget(self.limit_resume_btn)
        limit_row.addWidget(self.limit_stop_btn)
        self.limit_box.hide()
        self._limit_deadline: float | None = None
        self._auto_deadline: float | None = None  # when the run will try again by itself (auto-resume), else None
        self._limit_open = False
        self._limit_message = ""
        self._limit_alerted = False
        self._limit_timer = QTimer(self)
        self._limit_timer.setInterval(1000)
        self._limit_timer.timeout.connect(self._tick_limit)
        self.canvas = ImageCanvas()
        self.viewer = ConversationViewer()
        self.viewer.setPlaceholderText("The conversation with the model appears here while a run is going.")
        top = QHBoxLayout()
        for w in (QLabel("Mode"), self.mode, QLabel("Label"), self.limit, self.start_btn, self.pause_btn, self.stop_btn, self.retry_btn,
                  self.clear_btn):
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
        col.addWidget(self.limit_box)
        col.addWidget(self.failure)
        col.addWidget(self.notice)
        col.addWidget(split, 1)
        col.addWidget(self.message)
        self.start_btn.clicked.connect(self.start)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.stop_btn.clicked.connect(self.stop)
        self.retry_btn.clicked.connect(self.retry_failed)
        self.clear_waiting_action.triggered.connect(lambda: self.clear_queue(("queued",)))
        self.clear_failed_action.triggered.connect(lambda: self.clear_queue(("failed",)))
        self.clear_all_action.triggered.connect(lambda: self.clear_queue(None))
        self.limit_resume_btn.clicked.connect(self.resume_after_limit)
        self.limit_stop_btn.clicked.connect(self.stop)
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
        self.clear_btn.setEnabled(not self.running() and sum(c.values()) - c["uploaded"] > 0)
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
                                  on_entry=on_entry, taxonomy_dir=self.ctx.taxonomy_dir)

        worker = RunWorker(self.ctx.db, make, data_dir / "sidecars", limit=self.limit.value() or None, data_dir=data_dir,
                           keep_awake=bool(settings["labeling"].get("keep_awake")),
                           auto_resume=bool(settings["run"].get("auto_resume_on_limit", True)), parent=self)
        worker.image_started.connect(self._image_started)
        worker.entry.connect(self._entry)
        worker.progress.connect(self._progress)
        worker.problem.connect(lambda m: self.message.setText(m))
        worker.rate_limited.connect(self._rate_limited)
        worker.auto_wait.connect(self._auto_waiting)
        worker.run_finished.connect(self._finished)
        self.worker = worker
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.retry_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.message.setText("Running. You can review finished images on the Review queue tab while this goes on.")
        self._timer.start(1500)
        worker.start()
        return True

    def show_failure(self, name: str, error: str, provider: str = "", model: str = "") -> None:
        settings = self.ctx.settings()["provider"]
        hint = failure_hint(error, provider or settings["provider"], model or settings.get("model", ""))
        text = f"Last failure ({name}): {error or 'no details were saved'}"
        passed = ("This image is marked failed and the run moved on to the next one; 'Retry failed images' tries it "
                  "again later.")
        self.failure.setText(text + f"\n{passed}" + (f"\n{hint}" if hint else ""))

    def waiting_summary(self) -> str:
        """How many images wait on each kind of question, for example '9 x no laser found, 3 x low confidence'."""
        sets = []
        for row in self.ctx.db.list("needs_input"):
            try:
                sets.append(read_sidecar(row.sidecar_path))
            except (OSError, ValueError, TypeError):
                continue
        return describe_summary(summarize(sets))

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

    # ------------------------------------------------------------------ provider limit: pause and ask
    def _rate_limited(self, message: str, retry_after) -> None:
        """The run is waiting: say why, count down if the provider gave a time, and ask before going on."""
        self._limit_message = limit_advice(message, retry_after)
        self._limit_deadline = time.monotonic() + float(retry_after) if retry_after else None
        self._auto_deadline = None
        self._limit_open = True
        self._limit_alerted = False
        self.pause_btn.setEnabled(False)
        self.progress_label.setText("Paused: the provider's limit was reached. Nothing is lost.")
        self.message.setText("Waiting for you to resume.")
        self._show_limit_text()
        self.limit_box.show()
        self._limit_timer.start()
        QApplication.alert(self.window())  # flash the taskbar in case you are in another window

    def _auto_waiting(self, seconds: float) -> None:
        """Auto-resume is on: say when the run will try again by itself."""
        self._auto_deadline = time.monotonic() + float(seconds)
        self.progress_label.setText("Paused: the provider's limit was reached. The run carries on by itself; nothing is lost.")
        self.message.setText("Waiting for the limit to reopen. Leave this window open (or press Resume now / Stop).")
        self._show_limit_text()

    def _show_limit_text(self) -> None:
        text = self._limit_message
        if self._auto_deadline is not None:
            left = max(0.0, self._auto_deadline - time.monotonic())
            at = datetime.now() + timedelta(seconds=left)
            text = (text.replace(" Press Resume when the limit has reopened.", "")
                    + f"  Trying again by itself at about {at:%H:%M} (in {max(1, round(left / 60))} min). "
                      "Press Resume now to try sooner, or Stop the run.")
        elif self._limit_deadline is not None:
            left = self._limit_deadline - time.monotonic()
            text += (f"  Time left on the provider's wait: about {max(1, round(left / 60))} min." if left > 0
                     else "  The wait the provider asked for is over.")
        self.limit_text.setText(text)

    def _tick_limit(self) -> None:
        self._show_limit_text()
        if self._auto_deadline is None and self._limit_deadline is not None \
                and time.monotonic() >= self._limit_deadline and not self._limit_alerted:
            self._limit_alerted = True
            self.progress_label.setText("The provider's wait is over. Press Resume now to continue.")
            QApplication.alert(self.window())

    def _hide_limit(self) -> None:
        self._limit_timer.stop()
        self.limit_box.hide()
        self._limit_open = False
        self._auto_deadline = None

    def resume_after_limit(self) -> None:
        if self.worker is None or not self.worker.waiting_for_limit:
            return
        self._hide_limit()
        self.pause_btn.setEnabled(True)
        self.message.setText("Running again.")
        self.progress_label.setText("Resuming...")
        self.worker.resume_after_limit()

    def shutdown(self, wait_ms: int = 30000) -> None:
        """Called when the window closes: end the run (a pending limit wait too) and wait for the thread."""
        worker = self.worker
        if worker is not None:
            worker.stop()
            worker.wait(wait_ms)

    def stop(self) -> None:
        if self.worker is not None:
            self._hide_limit()
            self.worker.stop()
            self.message.setText("Stopping after the image in progress...")
            self.stop_btn.setEnabled(False)

    @staticmethod
    def _ask_clear(text: str) -> bool:
        return QMessageBox.question(None, "Clear the queue?", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def clear_queue(self, statuses: tuple[str, ...] | None) -> int:
        """Remove images from the queue after asking. `statuses=None` means everything not uploaded yet.
        Returns how many were removed. Image files and saved results stay on disk."""
        if self.running():
            self.message.setText("Stop the run before clearing the queue.")
            return 0
        counts = self.ctx.db.counts()
        wanted = tuple(s for s in counts if s != "uploaded") if statuses is None else statuses
        how_many = sum(counts[s] for s in wanted)
        if how_many == 0:
            self.message.setText("Nothing to remove there.")
            return 0
        if statuses is None:
            detail = ("This also removes images you have labeled but not uploaded yet, with their review state "
                      f"({counts['pending'] + counts['approved'] + counts['edited'] + counts['needs_input']} of them are "
                      "waiting for review or upload). ")
        else:
            detail = ""
        text = (f"Remove {how_many} image(s) from the queue? {detail}Image files and saved results stay on disk, "
                "and nothing on Roboflow changes.")
        if not self._confirm_clear(text):
            return 0
        removed = self.ctx.db.clear(wanted)
        self.failure.clear()
        self.notice.clear()
        self.message.setText(f"Removed {removed} image(s) from the queue.")
        self.ctx.queue_changed.emit()
        return removed

    def retry_failed(self) -> None:
        self.failure.clear()
        ids = [r.image_id for r in self.ctx.db.list("failed")]
        self.ctx.db.requeue(ids)
        self.message.setText(f"Put {len(ids)} failed image(s) back in the queue.")
        self.ctx.queue_changed.emit()

    # ------------------------------------------------------------------ worker signals
    def _image_started(self, image_id: str, path: str) -> None:
        if self._limit_open:  # the run carried on by itself after waiting out a provider limit
            self._hide_limit()
            self.pause_btn.setEnabled(True)
            self.message.setText("The provider's limit reopened; running again.")
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
        if result is not None and result.status == "needs_input":
            first = next((q.text for q in result.questions if q.answer is None), "")
            self.notice.setText(f"Needs your input ({row.name}): {reason_line(result)}. {first}")
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
        self._hide_limit()
        self.pause_btn.setText("Pause")
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        why = STOP_TEXT.get(summary.stopped, summary.stopped)
        waiting = self.waiting_summary()
        self.message.setText(f"Finished {summary.processed} image(s)." + (f" The run {why}." if summary.stopped else "")
                             + (f" Waiting for you: {waiting}. Open the Review queue, filter 'Needs your input'." if waiting else ""))
        self.refresh_counts()
        self.refresh_spend()
        self.ctx.queue_changed.emit()
