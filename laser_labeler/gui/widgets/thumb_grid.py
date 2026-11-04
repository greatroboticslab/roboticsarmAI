"""Thumbnail grid with optional box overlays and tick boxes. Thumbnails load a few at a time so the window never freezes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from PyQt6.QtCore import QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QListView, QListWidget, QListWidgetItem, QWidget

THUMB = 150
BATCH = 4  # thumbnails decoded per timer tick
LASER, OBJECT = QColor("#ff3b6b"), QColor("#18b8a6")


@dataclass
class GridItem:
    key: str
    title: str
    path: str | None = None  # local image file; None shows a placeholder
    tooltip: str = ""
    boxes: list[tuple[str, tuple, int, int]] = field(default_factory=list)  # (kind, (x, y, w, h), image w, image h)
    checked: bool | None = None  # None = no tick box
    enabled: bool = True
    cache_key: str = ""  # changes when the thumbnail should be redrawn


def placeholder(text: str = "") -> QPixmap:
    pix = QPixmap(THUMB, THUMB)
    pix.fill(QColor("#3a3d44"))
    if text:
        painter = QPainter(pix)
        painter.setPen(QColor("#9aa0a8"))
        painter.drawText(QRect(0, 0, THUMB, THUMB), Qt.AlignmentFlag.AlignCenter, text)
        painter.end()
    return pix


def make_thumbnail(path: str, boxes=(), size: int = THUMB) -> QPixmap:
    """The image scaled to fit `size`, with its boxes drawn on top. A placeholder if it cannot be read."""
    try:
        with Image.open(path) as im:
            full_w, full_h = im.size
            im.draft("RGB", (size * 2, size * 2))
            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            im.thumbnail((size, size))
            rgb = im.convert("RGB")
    except Exception:
        try:
            from core.camera import load_image
            data = load_image(path)
            rgb = Image.fromarray(data.rgb)
            full_w, full_h = rgb.size
            rgb.thumbnail((size, size))
        except Exception:
            return placeholder("cannot read")
    image = QImage(rgb.tobytes(), rgb.width, rgb.height, rgb.width * 3, QImage.Format.Format_RGB888).copy()
    pix = QPixmap.fromImage(image)
    if boxes:
        painter = QPainter(pix)
        for kind, (x, y, w, h), iw, ih in boxes:
            sx, sy = pix.width() / (iw or full_w), pix.height() / (ih or full_h)
            pen = QPen(LASER if kind == "laser" else OBJECT, 2)
            painter.setPen(pen)
            painter.drawRect(int(x * sx), int(y * sy), max(int(w * sx), 2), max(int(h * sy), 2))
        painter.end()
    return pix


class ThumbnailGrid(QListWidget):
    """Items are GridItems. Emits `activated_key` for a double click or Enter, `current_key` when the selection changes."""
    activated_key = pyqtSignal(str)
    current_key = pyqtSignal(str)
    checks_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None, size: int = THUMB) -> None:
        super().__init__(parent)
        self._size = size
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setIconSize(QSize(size, size))
        self.setGridSize(QSize(size + 24, size + 44))
        self.setWordWrap(True)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._items: dict[str, QListWidgetItem] = {}
        self._data: dict[str, GridItem] = {}
        self._pending: list[str] = []
        self._cache: dict[str, tuple[str, QIcon]] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._load_batch)
        self.itemDoubleClicked.connect(lambda it: self.activated_key.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.itemActivated.connect(lambda it: self.activated_key.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.currentItemChanged.connect(lambda cur, _old: self.current_key.emit(cur.data(Qt.ItemDataRole.UserRole) if cur else ""))
        self.itemChanged.connect(lambda _it: self.checks_changed.emit())

    # ------------------------------------------------------------------ content
    def set_items(self, items: list[GridItem], keep_selection: bool = True) -> None:
        keep = self.current_key_value() if keep_selection else ""
        self.blockSignals(True)
        self.clear()
        self._items, self._data, self._pending = {}, {}, []
        for item in items:
            row = QListWidgetItem(item.title)
            row.setData(Qt.ItemDataRole.UserRole, item.key)
            row.setToolTip(item.tooltip)
            row.setSizeHint(QSize(self._size + 20, self._size + 40))
            flags = row.flags()
            if not item.enabled:
                flags &= ~Qt.ItemFlag.ItemIsEnabled
            if item.checked is not None:
                flags |= Qt.ItemFlag.ItemIsUserCheckable
                row.setCheckState(Qt.CheckState.Checked if item.checked else Qt.CheckState.Unchecked)
            row.setFlags(flags)
            cached = self._cache.get(item.key)
            if cached and cached[0] == item.cache_key:
                row.setIcon(cached[1])
            else:
                row.setIcon(QIcon(placeholder("...")))
                self._pending.append(item.key)
            self.addItem(row)
            self._items[item.key], self._data[item.key] = row, item
        self.blockSignals(False)
        if keep in self._items:
            self.setCurrentItem(self._items[keep])
        if self._pending:
            self._timer.start(0)

    def current_key_value(self) -> str:
        item = self.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def select_key(self, key: str) -> None:
        if key in self._items:
            self.setCurrentItem(self._items[key])

    def keys(self) -> list[str]:
        return list(self._items)

    def set_title(self, key: str, title: str) -> None:
        if key in self._items:
            self._items[key].setText(title)

    # ------------------------------------------------------------------ ticks
    def checked_keys(self) -> list[str]:
        return [k for k, it in self._items.items() if it.checkState() == Qt.CheckState.Checked]

    def set_checked(self, keys, checked: bool = True) -> None:
        wanted = set(keys)
        self.blockSignals(True)
        for key, item in self._items.items():
            if key in wanted and item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.flags() & Qt.ItemFlag.ItemIsEnabled:
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.blockSignals(False)
        self.checks_changed.emit()

    # ------------------------------------------------------------------ lazy thumbnails
    def loading(self) -> bool:
        return bool(self._pending)

    def _visible_first(self) -> list[str]:
        view = self.viewport().rect()
        seen = [k for k in self._pending if self.visualItemRect(self._items[k]).intersects(view)]
        return (seen + [k for k in self._pending if k not in seen])[:BATCH]

    def _load_batch(self) -> None:
        for key in self._visible_first():
            self._pending.remove(key)
            item, data = self._items.get(key), self._data.get(key)
            if item is None or data is None:
                continue
            pix = make_thumbnail(data.path, data.boxes, self._size) if data.path else placeholder("not downloaded")
            icon = QIcon(pix)
            self._cache[key] = (data.cache_key, icon)
            item.setIcon(icon)
        if self._pending:
            self._timer.start(0)

    def finish_loading(self) -> None:
        """Load every remaining thumbnail now (tests, or before saving a screenshot)."""
        while self._pending:
            self._load_batch()
