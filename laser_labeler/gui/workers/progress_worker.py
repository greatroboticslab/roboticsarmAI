"""A background job that reports progress: fn(progress, should_stop) -> result."""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QThread, pyqtSignal


class ProgressWorker(QThread):
    progress = pyqtSignal(int, int)  # done, total (total 0 = unknown)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[Callable[[int, int], None], Callable[[], bool]], Any], parent=None) -> None:
        super().__init__(parent)
        self._fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            result = self._fn(lambda done, total: self.progress.emit(done, total), lambda: self._cancel)
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.succeeded.emit(result)
