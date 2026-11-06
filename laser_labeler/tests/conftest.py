"""One QApplication for the whole test session.

Each GUI test module asks for `QApplication.instance() or QApplication([])`. If the first module's fixture owned the only
Python reference, the application was destroyed when that module finished, while widgets and worker threads from its tests
could still be alive, and the next module crashed the interpreter at random. Keeping the reference here avoids that.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:  # the core tests run without the GUI libraries
    QApplication = None

_APP = (QApplication.instance() or QApplication([])) if QApplication is not None else None
