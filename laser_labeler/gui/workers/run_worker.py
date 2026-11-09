"""The labeling run in a background thread (spec sections 10 and 11): start, pause, resume, stop."""
from __future__ import annotations

import dataclasses
import threading
import time
from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal

from core.estimate import save_seconds_per_image
from core.keep_awake import KeepAwake
from core.labeler.limits import MAX_AUTO_RETRIES, auto_wait_seconds
from core.labeler.runner import LimitInfo, RunSummary, run_queue
from core.queue_db import QueueDB, QueueRow


class RunWorker(QThread):
    """`make_label_fn(on_entry)` builds the labeling function; entries of the live conversation arrive as `entry`."""
    image_started = pyqtSignal(str, str)  # image id, file path
    entry = pyqtSignal(dict, str)  # transcript entry, image id
    progress = pyqtSignal(object, object, object)  # RunSummary (a copy), QueueRow, ImageLabelSet or None
    run_finished = pyqtSignal(object)  # RunSummary
    problem = pyqtSignal(str)
    rate_limited = pyqtSignal(str, object)  # message, the provider's retry-after in seconds (or None): the run is waiting
    auto_wait = pyqtSignal(float)  # right after rate_limited: the run will try again by itself in this many seconds

    def __init__(self, db: QueueDB, make_label_fn: Callable, sidecar_dir, *, limit: int | None = None,
                 max_failures: int = 5, data_dir=None, keep_awake: bool = False, auto_resume: bool = False,
                 parent=None) -> None:
        super().__init__(parent)
        self._db, self._make, self._sidecars = db, make_label_fn, sidecar_dir
        self._limit, self._max_failures, self._data_dir = limit, max_failures, data_dir
        self._keep_awake = keep_awake
        self._auto_resume = auto_resume
        self._stop = threading.Event()
        self._go = threading.Event()
        self._go.set()
        self._limit_answer = threading.Event()  # set when the user answered the "limit reached" question
        self._limit_resume = False
        self._current = ""
        self._waiting_limit = False

    # ---- controls (called from the GUI thread)
    def pause(self) -> None:
        """Finish the image in progress, then wait."""
        self._go.clear()

    def resume(self) -> None:
        self._go.set()

    def stop(self) -> None:
        """Finish the image in progress, then end the run. A model call already sent cannot be taken back."""
        self._stop.set()
        self._go.set()
        self._limit_answer.set()  # also ends a wait for a provider limit to reopen

    @property
    def waiting_for_limit(self) -> bool:
        return self._waiting_limit

    def resume_after_limit(self) -> None:
        """The user says the provider's limit has reopened: try the same image again."""
        self._limit_resume = True
        self._limit_answer.set()

    def _wait_for_limit(self, info: LimitInfo) -> bool:
        """Called on this thread by the run loop. The image is already back in the queue. Wait for the user, or, with
        auto-resume on, until the provider's limit should have reopened, then try the same image again."""
        self._limit_answer.clear()
        self._limit_resume = False
        self._waiting_limit = True
        automatic = self._auto_resume and info.consecutive <= MAX_AUTO_RETRIES
        wait = auto_wait_seconds(info.retry_after, info.consecutive) if automatic else None
        self.rate_limited.emit(info.message, info.retry_after)
        if wait is not None:
            self.auto_wait.emit(wait)
        deadline = None if wait is None else time.monotonic() + wait
        while not self._limit_answer.wait(0.1):
            if deadline is not None and time.monotonic() >= deadline:
                self._limit_resume = True
                break
        self._waiting_limit = False
        return self._limit_resume and not self._stop.is_set()

    @property
    def paused(self) -> bool:
        return not self._go.is_set() and not self._stop.is_set()

    # ---- thread body
    def _should_stop(self) -> bool:
        while not self._go.wait(0.1):
            if self._stop.is_set():
                break
        return self._stop.is_set()

    def run(self) -> None:
        inner = self._make(lambda e: self.entry.emit(e, self._current))

        def label(row: QueueRow):
            self._current = row.image_id
            self.image_started.emit(row.image_id, row.path)
            return inner(row)

        awake = KeepAwake()
        try:
            if self._keep_awake:
                awake.start()
            summary = run_queue(self._db, label, sidecar_dir=self._sidecars, should_stop=self._should_stop,
                                limit=self._limit, max_consecutive_failures=self._max_failures,
                                on_progress=lambda s, row, res: self.progress.emit(dataclasses.replace(s), row, res),
                                clock=time.monotonic, on_limit=self._wait_for_limit)
        except Exception as exc:  # reported to the screen; the queue keeps every finished image
            self.problem.emit(f"The run stopped because of an error: {type(exc).__name__}: {exc}")
            summary = RunSummary(stopped="error")
        finally:
            awake.stop()
        if self._data_dir is not None and summary.processed:
            save_seconds_per_image(self._data_dir, summary.seconds_per_image)
        self.run_finished.emit(summary)
