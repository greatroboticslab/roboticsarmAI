"""Start the Laser Labeler window: python app.py"""
from __future__ import annotations

import sys
import traceback


def main() -> int:
    from PyQt6.QtWidgets import QApplication

    from gui.main_window import MainWindow

    sys.excepthook = lambda *exc: traceback.print_exception(*exc)  # do not abort on slot errors
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
