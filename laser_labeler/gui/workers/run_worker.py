"""The labeling run in a background thread (spec sections 10 and 11): start, pause, resume, stop."""
from __future__ import annotations

import dataclasses
import threading
import time
from typing import Callable

from PyQt6.QtCore import QThread, pyqtSignal

from core.estimate import save_seconds_per_image
from core.keep_awake import KeepAwake
from core.labeler.runner import RunSummary, run_queue
from core.queue_db import QueueDB, QueueRow


class RunWorker(QThread):
    """`make_label_fn(on_entry)` builds the labeling function; entries of the live conversation arrive as `entry`."""
    image_started = pyqtSignal(str, str)  # image id, file path
    entry = pyqtSignal(dict, str)  # transcript entry, image id
    progress = pyqtSignal(object, object, object)  # RunSummary (a copy), QueueRow, ImageLabelSet or None
    run_finished = pyqtSignal(object)  # RunSummary
    problem = pyqtSignal(str)

    def __init__(self, db: QueueDB, make_label_fn: Callable, sidecar_dir, *, limit: int | None = None,
                 max_failures: int = 5, data_dir=None, keep_awake: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._db, self._make, self._sidecars = db, make_label_fn, sidecar_dir
        self._limit, self._max_failures, self._data_dir = limit, max_failures, data_dir
        self._keep_awake = keep_awake
        self._stop = threading.Event()
        self._go = threading.Event()
        self._go.set()
        self._current = ""

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
                                clock=time.monotonic)
        except Exception as exc:  # reported to the screen; the queue keeps every finished image
            self.problem.emit(f"The run stopped because of an error: {type(exc).__name__}: {exc}")
            summary = RunSummary(stopped="error")
        finally:
            awake.stop()
        if self._data_dir is not None and summary.processed:
            save_seconds_per_image(self._data_dir, summary.seconds_per_image)
        self.run_finished.emit(summary)
