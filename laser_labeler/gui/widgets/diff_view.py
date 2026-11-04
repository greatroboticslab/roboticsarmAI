"""Side-by-side text diff (prompt versions and "Generate new prompt")."""
from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QTextCursor, QTextFormat
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QTextEdit, QVBoxLayout, QWidget

from core.prompts.diffview import align_diff, changed_count

REMOVED, ADDED, CHANGED_OLD, CHANGED_NEW = (QColor("#ffd7d5"), QColor("#cdeccf"), QColor("#ffe9b8"), QColor("#d6e8ff"))
BLANK = QColor("#eeeeee")


def _editor() -> QPlainTextEdit:
    edit = QPlainTextEdit()
    edit.setReadOnly(True)
    edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
    edit.setFont(QFont("Consolas, Menlo, monospace", 9))
    return edit


class SideBySideDiff(QWidget):
    """Two read-only panes with changed lines colored and scrolling kept in step."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.left, self.right = _editor(), _editor()
        self.left_title, self.right_title = QLabel(), QLabel()
        self.summary = QLabel()
        titles, panes = QHBoxLayout(), QHBoxLayout()
        titles.addWidget(self.left_title)
        titles.addWidget(self.right_title)
        panes.addWidget(self.left)
        panes.addWidget(self.right)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(self.summary)
        col.addLayout(titles)
        col.addLayout(panes, 1)
        self.left.verticalScrollBar().valueChanged.connect(self.right.verticalScrollBar().setValue)
        self.right.verticalScrollBar().valueChanged.connect(self.left.verticalScrollBar().setValue)
        self.left.horizontalScrollBar().valueChanged.connect(self.right.horizontalScrollBar().setValue)
        self.right.horizontalScrollBar().valueChanged.connect(self.left.horizontalScrollBar().setValue)

    def set_texts(self, old: str, new: str, old_title: str = "Before", new_title: str = "After") -> int:
        """Show both texts aligned line by line. Returns the number of changed lines."""
        rows = align_diff(old, new)
        self.left_title.setText(f"<b>{old_title}</b>")
        self.right_title.setText(f"<b>{new_title}</b>")
        self.left.setPlainText("\n".join(r[0] for r in rows))
        self.right.setPlainText("\n".join(r[1] for r in rows))
        count = changed_count(rows)
        self.summary.setText("No differences." if not count else f"{count} line(s) differ.")
        for edit, side in ((self.left, 0), (self.right, 1)):
            edit.setExtraSelections(self._selections(edit, rows, side))
            edit.moveCursor(QTextCursor.MoveOperation.Start)
        return count

    @staticmethod
    def _selections(edit: QPlainTextEdit, rows, side: int) -> list:
        colors = {("delete", 0): REMOVED, ("insert", 1): ADDED, ("replace", 0): CHANGED_OLD, ("replace", 1): CHANGED_NEW,
                  ("insert", 0): BLANK, ("delete", 1): BLANK}
        out = []
        block = edit.document().firstBlock()
        for left, right, tag in rows:
            color = colors.get((tag, side))
            if color is not None and block.isValid():
                sel = QTextEdit.ExtraSelection()
                sel.format.setBackground(color)
                sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
                sel.cursor = QTextCursor(block)
                out.append(sel)
            block = block.next()
        return out


class PartsDiff(QWidget):
    """A diff for several named parts: pick the part to look at."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.picker = QComboBox()
        self.view = SideBySideDiff()
        self._pairs: list[tuple[str, str, str]] = []
        self._titles = ("Before", "After")
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.addWidget(self.picker)
        col.addWidget(self.view, 1)
        self.picker.currentIndexChanged.connect(self._show)

    def set_pairs(self, pairs: list[tuple[str, str, str]], old_title: str = "Current", new_title: str = "Proposed") -> None:
        self._pairs, self._titles = pairs, (old_title, new_title)
        self.picker.blockSignals(True)
        self.picker.clear()
        self.picker.addItems([name for name, _, _ in pairs])
        self.picker.blockSignals(False)
        if pairs:
            self.picker.setCurrentIndex(0)
            self._show(0)
        else:
            self.view.set_texts("", "", *self._titles)
            self.view.summary.setText("Nothing to compare.")

    def _show(self, index: int) -> None:
        if 0 <= index < len(self._pairs):
            name, old, new = self._pairs[index]
            self.view.set_texts(old, new, f"{self._titles[0]}: {name}", f"{self._titles[1]}: {name}")
