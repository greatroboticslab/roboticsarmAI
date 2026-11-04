"""Run a blocking call (network) off the GUI thread and report back on it."""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


class CallWorker(QThread):
    """Runs fn(*args) in a thread; emits the result or an error text."""
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable, args: tuple, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._fn, self._args = fn, args

    def run(self) -> None:
        try:
            result = self._fn(*self._args)
        except Exception as exc:  # reported to the GUI, never raised in the thread
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.succeeded.emit(result)


class CallRunner(QObject):
    """Lives on the GUI thread: starts a worker and calls on_ok / on_fail there."""

    def __init__(self, fn: Callable, args: tuple, on_ok: Callable[[Any], None],
                 on_fail: Callable[[str], None], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._on_ok, self._on_fail = on_ok, on_fail
        self._worker = CallWorker(fn, args)
        self._worker.succeeded.connect(self._ok)
        self._worker.failed.connect(self._fail)

    def start(self) -> None:
        self._worker.start()

    def _finish(self) -> None:
        self._worker.wait()  # the thread is already past its last statement
        self.deleteLater()

    @pyqtSlot(object)
    def _ok(self, value: Any) -> None:
        self._finish()
        self._on_ok(value)

    @pyqtSlot(str)
    def _fail(self, message: str) -> None:
        self._finish()
        self._on_fail(message)


def start_call(fn: Callable, args: tuple, on_ok: Callable[[Any], None],
               on_fail: Callable[[str], None], parent: QObject | None = None) -> CallRunner:
    """Start fn(*args) in a background thread; keep the returned runner referenced."""
    runner = CallRunner(fn, args, on_ok, on_fail, parent)
    runner.start()
    return runner
