"""Thumbnail grid with optional box overlays and tick boxes. Thumbnails load a few at a time so the window never freezes."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps
from PyQt6.QtCore import QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QListView, QListWidget, QListWidgetItem, QWidget

from core.camera import exif_orientation  # noqa: E402

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


_SHARED: dict[str, QIcon] = {}


def shared_icon(text: str) -> QIcon:
    """One icon per text, shared by every item that shows it (drawing a pixmap per item costs seconds for a big list)."""
    if text not in _SHARED:
        _SHARED[text] = QIcon(placeholder(text))
    return _SHARED[text]


def _with_orientation(im: Image.Image, orientation: int) -> Image.Image:
    """`im` (already shrunk, so its EXIF is gone) tagged with the original's orientation, ready to be turned upright."""
    exif = Image.Exif()
    exif[0x0112] = orientation
    out = im.copy()
    out.info["exif"] = exif.tobytes()
    return out


def make_thumbnail(path: str, boxes=(), size: int = THUMB) -> QPixmap:
    """The image scaled to fit `size`, with its boxes drawn on top. A placeholder if it cannot be read."""
    try:
        with Image.open(path) as im:
            quarter = exif_orientation(im) >= 5
            full_w, full_h = (im.height, im.width) if quarter else im.size  # the size a viewer shows (turned upright)
            orientation = exif_orientation(im)
            im.draft("RGB", (size * 2, size * 2))
            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            im.thumbnail((size, size))
            rgb = im.convert("RGB")
            if orientation != 1:
                rgb = ImageOps.exif_transpose(_with_orientation(rgb, orientation))
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
        self.setLayoutMode(QListView.LayoutMode.Batched)  # lay out a few hundred items at a time, not all before painting
        self.setBatchSize(200)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._items: dict[str, QListWidgetItem] = {}
        self._data: dict[str, GridItem] = {}
        self._pending: dict[str, None] = {}  # keys waiting for a thumbnail, in order (a dict: cheap to remove from)
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
        self.setUpdatesEnabled(False)
        self.blockSignals(True)
        try:
            self.clear()
            self._items, self._data, self._pending = {}, {}, {}
            size_hint = QSize(self._size + 20, self._size + 40)
            for item in items:
                row = QListWidgetItem(item.title)
                row.setData(Qt.ItemDataRole.UserRole, item.key)
                row.setToolTip(item.tooltip)
                row.setSizeHint(size_hint)
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
                elif not item.path:
                    row.setIcon(shared_icon("not downloaded"))  # nothing to decode: no waiting in line
                else:
                    row.setIcon(shared_icon("..."))
                    self._pending[item.key] = None
                self.addItem(row)
                self._items[item.key], self._data[item.key] = row, item
        finally:
            self.blockSignals(False)
            self.setUpdatesEnabled(True)
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

    def _tickable(self, item: QListWidgetItem) -> bool:
        return bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable and item.flags() & Qt.ItemFlag.ItemIsEnabled)

    def _write_checks(self, wanted: dict[str, bool]) -> None:
        """Set many tick boxes in one go: only the ones that change are touched, and the view is told once at the end
        (telling it per item made 'tick all' take seconds on a long list)."""
        model = self.model()
        self.setUpdatesEnabled(False)
        model.blockSignals(True)
        try:
            for key, on in wanted.items():
                item = self._items.get(key)
                state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
                if item is not None and self._tickable(item) and item.checkState() != state:
                    item.setCheckState(state)
        finally:
            model.blockSignals(False)
            self.setUpdatesEnabled(True)
        self.viewport().update()
        self.checks_changed.emit()

    def set_checked(self, keys, checked: bool = True) -> None:
        self._write_checks({k: checked for k in keys})

    def set_checked_exactly(self, keys) -> None:
        """Tick exactly these keys and untick every other item."""
        on = set(keys)
        self._write_checks({k: k in on for k in self._items})

    # ------------------------------------------------------------------ lazy thumbnails
    def loading(self) -> bool:
        return bool(self._pending)

    def _visible_keys(self) -> list[str]:
        """Keys of the items on screen, found by asking which item sits at a grid of points (cheap, however long the list)."""
        view = self.viewport().rect()
        step_x, step_y = max(self.gridSize().width(), 1), max(self.gridSize().height(), 1)
        seen: dict[str, None] = {}
        for y in range(view.top() + 4, view.bottom(), step_y // 2 or 1):
            for x in range(view.left() + 4, view.right(), step_x // 2 or 1):
                item = self.itemAt(x, y)
                if item is not None:
                    seen[item.data(Qt.ItemDataRole.UserRole)] = None
        return list(seen)

    def _next_keys(self) -> list[str]:
        picked = [k for k in self._visible_keys() if k in self._pending][:BATCH]
        for key in self._pending:  # then the rest, in order
            if len(picked) >= BATCH:
                break
            if key not in picked:
                picked.append(key)
        return picked

    def _load_batch(self) -> None:
        for key in self._next_keys():
            self._pending.pop(key, None)
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
