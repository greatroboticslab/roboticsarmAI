"""Keep the computer awake during a long run (settings: labeling.keep_awake). Best effort, never fatal."""
from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


class KeepAwake:
    """Use as a context manager, or call start() and stop()."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self.active = False

    def start(self) -> bool:
        try:
            if sys.platform.startswith("win"):
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)  # type: ignore[attr-defined]
                self.active = True
            elif sys.platform == "darwin" and shutil.which("caffeinate"):
                self._proc = subprocess.Popen(["caffeinate", "-i"])
                self.active = True
            elif shutil.which("systemd-inhibit"):
                self._proc = subprocess.Popen(["systemd-inhibit", "--what=idle:sleep", "--who=Laser Labeler",
                                               "--why=labeling run", "sleep", "infinity"])
                self.active = True
        except Exception:
            self.active = False
        return self.active

    def stop(self) -> None:
        try:
            if sys.platform.startswith("win") and self.active:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)  # type: ignore[attr-defined]
            if self._proc is not None:
                self._proc.terminate()
        except Exception:
            pass
        self._proc, self.active = None, False

    def __enter__(self) -> "KeepAwake":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
