"""Image view with labeled boxes that can be selected, moved, resized and drawn (spec section 10).

All coordinates going in and out are ORIGINAL image pixels (x, y, w, h). The view zooms with the wheel,
pans by dragging empty space, and has four modes:
  view  - click a box to select it
  edit  - the selected box can be dragged to move, or by its edges and corners to resize
  draw  - drag a new box on empty space (emits boxDrawn)
  point - a click emits pointClicked (used to answer "where does the beam first hit?")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsSimpleTextItem,
                             QGraphicsView, QWidget)

LASER_COLOR = QColor("#ff3b6b")
OBJECT_COLOR = QColor("#18b8a6")
SELECTED_COLOR = QColor("#ffd23f")
MIN_SIZE = 2.0
HANDLE_PX = 9  # on-screen size of a resize handle
GRAB_PX = 7  # how close (on screen) the cursor must be to an edge to resize


@dataclass
class BoxSpec:
    id: str
    box: tuple[float, float, float, float]
    label: str = ""
    kind: str = "object"  # "object" or "laser"


class ImageCanvas(QGraphicsView):
    boxChanged = pyqtSignal(str, tuple)  # id, (x, y, w, h) in image pixels, after a move or resize
    boxSelected = pyqtSignal(str)  # id, or "" when nothing is selected
    boxDrawn = pyqtSignal(tuple)  # (x, y, w, h) of a newly drawn rectangle
    pointClicked = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor("#1d1f23")))
        self.setMinimumSize(240, 180)
        self._pix_item = None
        self._size = (0, 0)
        self._boxes: dict[str, BoxSpec] = {}
        self._items: list = []
        self._selected = ""
        self._mode = "view"
        self._drag: tuple | None = None  # (kind, box id, start scene point, start rect, edges)
        self._pan_last = None
        self._rubber: QGraphicsRectItem | None = None
        self.setMouseTracking(True)

    # ------------------------------------------------------------------ content
    def set_image(self, source: str | Path | QImage | None) -> bool:
        """Show an image file (or a QImage). Returns False when it cannot be read."""
        self._scene.clear()
        self._items, self._pix_item, self._rubber = [], None, None
        if source is None:
            self._size = (0, 0)
            return False
        image = source if isinstance(source, QImage) else QImage(str(source))
        if image.isNull():
            self._size = (0, 0)
            return False
        self._pix_item = self._scene.addPixmap(QPixmap.fromImage(image))
        self._size = (image.width(), image.height())
        self._scene.setSceneRect(QRectF(0, 0, *self._size))
        self.fit()
        self._redraw()
        return True

    def image_size(self) -> tuple[int, int]:
        return self._size

    def set_boxes(self, boxes: list[BoxSpec], selected: str = "") -> None:
        self._boxes = {b.id: b for b in boxes}
        self._selected = selected if selected in self._boxes else ""
        self._redraw()

    def boxes(self) -> list[BoxSpec]:
        return list(self._boxes.values())

    def selected(self) -> str:
        return self._selected

    def select(self, box_id: str, emit: bool = False) -> None:
        box_id = box_id if box_id in self._boxes else ""
        if box_id != self._selected:
            self._selected = box_id
            self._redraw()
            if emit:
                self.boxSelected.emit(box_id)

    def set_mode(self, mode: str) -> None:
        if mode not in ("view", "edit", "draw", "point"):
            raise ValueError(f"unknown canvas mode {mode!r}")
        self._mode = mode
        self._drag = None
        cursor = {"view": Qt.CursorShape.ArrowCursor, "edit": Qt.CursorShape.ArrowCursor,
                  "draw": Qt.CursorShape.CrossCursor, "point": Qt.CursorShape.CrossCursor}[mode]
        self.viewport().setCursor(cursor)
        self._redraw()

    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------ drawing
    def _scale(self) -> float:
        return max(self.transform().m11(), 1e-6)

    def _redraw(self) -> None:
        for item in self._items:
            self._scene.removeItem(item)
        self._items = []
        if self._pix_item is None:
            return
        for spec in self._boxes.values():
            selected = spec.id == self._selected
            color = SELECTED_COLOR if selected else (LASER_COLOR if spec.kind == "laser" else OBJECT_COLOR)
            pen = QPen(color, 3 if selected else 2)
            pen.setCosmetic(True)
            if selected and self._mode == "edit":
                pen.setStyle(Qt.PenStyle.DashLine)
            rect = QGraphicsRectItem(QRectF(*spec.box))
            rect.setPen(pen)
            rect.setZValue(2 if selected else 1)
            self._scene.addItem(rect)
            self._items.append(rect)
            if spec.label:
                text = QGraphicsSimpleTextItem(spec.label)
                text.setBrush(QBrush(color))
                text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                text.setPos(spec.box[0], spec.box[1])
                text.setZValue(3)
                self._scene.addItem(text)
                self._items.append(text)
            if selected and self._mode == "edit":
                self._add_handles(spec.box)

    def _add_handles(self, box) -> None:
        x, y, w, h = box
        for hx in (x, x + w / 2, x + w):
            for hy in (y, y + h / 2, y + h):
                if hx == x + w / 2 and hy == y + h / 2:
                    continue
                handle = QGraphicsRectItem(-HANDLE_PX / 2, -HANDLE_PX / 2, HANDLE_PX, HANDLE_PX)
                handle.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                handle.setBrush(QBrush(SELECTED_COLOR))
                handle.setPen(QPen(QColor("#000000"), 1))
                handle.setPos(hx, hy)
                handle.setZValue(4)
                self._scene.addItem(handle)
                self._items.append(handle)

    # ------------------------------------------------------------------ view control
    def fit(self) -> None:
        if self._size[0]:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor: float) -> None:
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:  # keep the whole image visible until the user zooms
        super().resizeEvent(event)
        if self._size[0] and self._scale() <= self._fit_scale() * 1.001:
            self.fit()

    def _fit_scale(self) -> float:
        if not self._size[0]:
            return 1.0
        view = self.viewport().size()
        return min(view.width() / self._size[0], view.height() / self._size[1])

    def wheelEvent(self, event) -> None:
        step = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        if step < 1 and self._scale() * step < self._fit_scale() * 0.5:
            return
        self.zoom(step)

    # ------------------------------------------------------------------ mouse
    def _point(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def _box_at(self, p: QPointF) -> str:
        hits = [s for s in self._boxes.values() if QRectF(*s.box).adjusted(-1, -1, 1, 1).contains(p)]
        return min(hits, key=lambda s: s.box[2] * s.box[3]).id if hits else ""  # the smallest box under the cursor

    def _edges_at(self, p: QPointF, box) -> tuple[bool, bool, bool, bool] | None:
        """Which edges of `box` the cursor is on (left, right, top, bottom), or None if not on the border."""
        x, y, w, h = box
        tol = GRAB_PX / self._scale()
        if not (x - tol <= p.x() <= x + w + tol and y - tol <= p.y() <= y + h + tol):
            return None
        left, right = abs(p.x() - x) <= tol, abs(p.x() - (x + w)) <= tol
        top, bottom = abs(p.y() - y) <= tol, abs(p.y() - (y + h)) <= tol
        return (left, right, top, bottom) if (left or right or top or bottom) else None

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or not self._size[0]:
            return super().mousePressEvent(event)
        p = self._point(event)
        if self._mode == "point":
            if self._inside(p):
                self.pointClicked.emit(p.x(), p.y())
            return
        if self._mode == "draw":
            if self._inside(p):
                self._drag = ("draw", "", p, QRectF(p, p), None)
            return
        if self._mode == "edit" and self._selected:
            spec = self._boxes[self._selected]
            edges = self._edges_at(p, spec.box)
            if edges:
                self._drag = ("resize", spec.id, p, QRectF(*spec.box), edges)
                return
            if QRectF(*spec.box).contains(p):
                self._drag = ("move", spec.id, p, QRectF(*spec.box), None)
                return
        hit = self._box_at(p)
        if hit:
            self.select(hit, emit=True)
            if self._mode == "edit":
                self._drag = ("move", hit, p, QRectF(*self._boxes[hit].box), None)
            return
        if self._selected:
            self.select("", emit=True)
        self._pan_last = event.position().toPoint()
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_last is not None:
            now = event.position().toPoint()
            delta = now - self._pan_last
            self._pan_last = now
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        if self._drag is None:
            self._hover_cursor(self._point(event))
            return super().mouseMoveEvent(event)
        kind, box_id, start, rect, edges = self._drag
        p = self._point(event)
        if kind == "draw":
            self._show_rubber(self._clamped_rect(QRectF(start, p).normalized()))
            return
        new = self._moved(rect, p - start) if kind == "move" else self._resized(rect, p - start, edges)
        spec = self._boxes[box_id]
        spec.box = (new.x(), new.y(), new.width(), new.height())
        self._redraw()

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_last is not None:
            self._pan_last = None
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor if self._mode in ("view", "edit") else Qt.CursorShape.CrossCursor)
            return
        drag, self._drag = self._drag, None
        if drag is None:
            return super().mouseReleaseEvent(event)
        kind, box_id = drag[0], drag[1]
        if kind == "draw":
            if self._rubber is not None:
                rect = self._rubber.rect()
                self._scene.removeItem(self._rubber)
                self._rubber = None
                if rect.width() >= MIN_SIZE and rect.height() >= MIN_SIZE:
                    self.boxDrawn.emit(tuple(int(round(v)) for v in (rect.x(), rect.y(), rect.width(), rect.height())))
            return
        spec = self._boxes[box_id]
        final = tuple(int(round(v)) for v in spec.box)
        spec.box = final  # type: ignore[assignment]
        self._redraw()
        if tuple(int(round(v)) for v in drag[3].getRect()) != final:
            self.boxChanged.emit(box_id, final)

    def _inside(self, p: QPointF) -> bool:
        return 0 <= p.x() <= self._size[0] and 0 <= p.y() <= self._size[1]

    def _clamped_rect(self, r: QRectF) -> QRectF:
        return r.intersected(QRectF(0, 0, *self._size))

    def _show_rubber(self, rect: QRectF) -> None:
        if self._rubber is None:
            self._rubber = QGraphicsRectItem()
            pen = QPen(SELECTED_COLOR, 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._rubber.setPen(pen)
            self._rubber.setZValue(5)
            self._scene.addItem(self._rubber)
        self._rubber.setRect(rect)

    def _moved(self, rect: QRectF, delta: QPointF) -> QRectF:
        x = min(max(rect.x() + delta.x(), 0), self._size[0] - rect.width())
        y = min(max(rect.y() + delta.y(), 0), self._size[1] - rect.height())
        return QRectF(x, y, rect.width(), rect.height())

    def _resized(self, rect: QRectF, delta: QPointF, edges) -> QRectF:
        left, right, top, bottom = edges
        x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        if left:
            x0 = min(max(x0 + delta.x(), 0), x1 - MIN_SIZE)
        if right:
            x1 = max(min(x1 + delta.x(), self._size[0]), x0 + MIN_SIZE)
        if top:
            y0 = min(max(y0 + delta.y(), 0), y1 - MIN_SIZE)
        if bottom:
            y1 = max(min(y1 + delta.y(), self._size[1]), y0 + MIN_SIZE)
        return QRectF(QPointF(x0, y0), QPointF(x1, y1))

    def _hover_cursor(self, p: QPointF) -> None:
        if self._mode != "edit" or not self._selected:
            return
        edges = self._edges_at(p, self._boxes[self._selected].box)
        shape = Qt.CursorShape.ArrowCursor
        if edges:
            left, right, top, bottom = edges
            if (left or right) and (top or bottom):
                shape = Qt.CursorShape.SizeFDiagCursor if (left and top) or (right and bottom) \
                    else Qt.CursorShape.SizeBDiagCursor
            else:
                shape = Qt.CursorShape.SizeHorCursor if (left or right) else Qt.CursorShape.SizeVerCursor
        elif QRectF(*self._boxes[self._selected].box).contains(p):
            shape = Qt.CursorShape.SizeAllCursor
        self.viewport().setCursor(shape)
