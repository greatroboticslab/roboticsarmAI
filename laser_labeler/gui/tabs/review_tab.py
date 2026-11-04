"""Review queue: look at each proposed label, fix it, answer questions, approve or reject (spec section 10)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QAbstractButton, QButtonGroup, QComboBox, QDoubleSpinBox, QHBoxLayout, QInputDialog,
                             QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressDialog, QPushButton,
                             QSpinBox, QSplitter, QStackedWidget, QTabWidget, QVBoxLayout, QWidget, QApplication)

from config import credentials
from config.credentials import CredentialError, MissingCredentialError
from core.models import ImageLabelSet, LaserBox, ObjectBox
from core.queue_db import QueueRow
from core.reports.sidecar import read_sidecar
from core.providers.hints import failure_hint
from core.review.service import (ReviewError, add_laser_box, add_object_box, delete_box)
from core.roboflow.upload import UploadSummary, build_upload_plan, upload_annotations
from gui.context import AppContext
from gui.tabs.label_run_tab import boxes_of
from gui.widgets.box_editor import BoxEditor
from gui.widgets.conversation_viewer import ConversationViewer
from gui.widgets.image_canvas import ImageCanvas
from gui.widgets.question_dialog import QuestionDialog
from gui.widgets.thumb_grid import GridItem, ThumbnailGrid
from gui.widgets.upload_dialog import UploadDialog

FILTERS = {"Needs your input": ["needs_input"], "Ready for review": ["pending"], "Failed": ["failed"],
           "Approved": ["approved", "edited"], "Rejected": ["rejected"], "Uploaded": ["uploaded"],
           "Waiting in the queue": ["queued", "labeling"], "Everything": None}
SORTS = {"Least confident first": "confidence_asc", "Most confident first": "confidence_desc",
         "Newest first": "recent", "Queue order": "order"}
PAGE = 150


class ReviewTab(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None, *, key_fn: Callable[[], str] | None = None,
                 upload_fn: Callable = upload_annotations) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._key_fn = key_fn or (lambda: credentials.require_key("roboflow"))
        self._upload_fn = upload_fn
        self.rows: list[QueueRow] = []
        self.page = 0
        self.image_id = ""
        self.draft: ImageLabelSet | None = None
        self.dirty = False
        self._question_id = ""  # a hit-point question waiting for a click on the image
        self._loading = False
        self.confirm_taxonomy: Callable[[list[str]], bool] = self._ask_taxonomy  # replaced in tests
        self._build()
        self._connect()
        self.reload()

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        self.filter_box, self.sort_box = QComboBox(), QComboBox()
        self.filter_box.addItems(list(FILTERS))
        self.filter_box.setCurrentText("Ready for review")
        self.sort_box.addItems(list(SORTS))
        self.list_btn, self.grid_btn = QPushButton("List"), QPushButton("Thumbnails")
        for b in (self.list_btn, self.grid_btn):
            b.setCheckable(True)
        self.list_btn.setChecked(True)
        view_group = QButtonGroup(self)
        view_group.addButton(self.list_btn)
        view_group.addButton(self.grid_btn)
        self.count_label = QLabel()
        self.list = QListWidget()
        self.grid = ThumbnailGrid(size=110)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.list)
        self.stack.addWidget(self.grid)
        self.prev_page, self.next_page = QPushButton("<"), QPushButton(">")
        self.page_label = QLabel()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.9)
        self.bulk_btn = QPushButton("Approve all ready images at or above")
        self.upload_btn = QPushButton("Upload approved...")
        left = QWidget()
        lc = QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        lc.addWidget(QLabel("Show"))
        lc.addWidget(self.filter_box)
        lc.addWidget(self.sort_box)
        views = QHBoxLayout()
        views.addWidget(self.list_btn)
        views.addWidget(self.grid_btn)
        lc.addLayout(views)
        lc.addWidget(self.count_label)
        lc.addWidget(self.stack, 1)
        pager = QHBoxLayout()
        for w in (self.prev_page, self.page_label, self.next_page):
            pager.addWidget(w)
        lc.addLayout(pager)
        bulk = QHBoxLayout()
        bulk.addWidget(self.bulk_btn, 1)
        bulk.addWidget(self.threshold)
        lc.addLayout(bulk)
        lc.addWidget(self.upload_btn)

        self.header = QLabel("Choose an image from the list.")
        self.header.setWordWrap(True)
        self.problems = QLabel()
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet("color: #bc4c00")
        self.canvas = ImageCanvas()
        self.mode_buttons: dict[str, QPushButton] = {}
        modes = QHBoxLayout()
        group = QButtonGroup(self)
        for key, label in (("view", "Look"), ("edit", "Move / resize boxes"), ("draw_object", "Draw object box"),
                           ("draw_laser", "Draw laser box")):
            b = QPushButton(label)
            b.setCheckable(True)
            group.addButton(b)
            modes.addWidget(b)
            self.mode_buttons[key] = b
        self.mode_buttons["view"].setChecked(True)
        self.delete_btn, self.fit_btn = QPushButton("Delete selected box"), QPushButton("Fit")
        modes.addWidget(self.delete_btn)
        modes.addWidget(self.fit_btn)
        modes.addStretch(1)
        self.approve_btn, self.reject_btn = QPushButton("Approve (A)"), QPushButton("Reject (R)")
        self.prev_btn, self.next_btn = QPushButton("Previous (P)"), QPushButton("Next (N)")
        self.reopen_btn = QPushButton("Undo: back to ready")
        self.rerun_btn = QPushButton("Re-run with a note...")
        self.pdf_btn, self.open_pdf_btn = QPushButton("Rebuild PDF"), QPushButton("Open PDF")
        actions = QHBoxLayout()
        for w in (self.approve_btn, self.reject_btn, self.reopen_btn, self.rerun_btn, self.pdf_btn, self.open_pdf_btn,
                  self.prev_btn, self.next_btn):
            actions.addWidget(w)
        center = QWidget()
        cc = QVBoxLayout(center)
        cc.setContentsMargins(0, 0, 0, 0)
        cc.addWidget(self.header)
        cc.addWidget(self.problems)
        cc.addLayout(modes)
        cc.addWidget(self.canvas, 1)
        cc.addLayout(actions)

        self.boxes = QListWidget()
        self.boxes.setMaximumHeight(120)
        self.editor = BoxEditor()
        self.apply_btn, self.revert_btn = QPushButton("Apply changes"), QPushButton("Discard changes")
        self.add_tax_btn = QPushButton("Add this box's values to the taxonomy")
        self.dirty_label = QLabel()
        labels = QWidget()
        lab = QVBoxLayout(labels)
        lab.addWidget(self.boxes)
        lab.addWidget(self.editor, 1)
        lab.addWidget(self.add_tax_btn)
        btns = QHBoxLayout()
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.revert_btn)
        lab.addLayout(btns)
        lab.addWidget(self.dirty_label)
        self.questions = QListWidget()
        self.answer_btn = QPushButton("Answer the selected question...")
        self.question_hint = QLabel()
        self.question_hint.setWordWrap(True)
        qs = QWidget()
        ql = QVBoxLayout(qs)
        ql.addWidget(self.questions, 1)
        ql.addWidget(self.answer_btn)
        ql.addWidget(self.question_hint)
        self.viewer = ConversationViewer()
        self.right = QTabWidget()
        self.right.addTab(labels, "Labels")
        self.right.addTab(qs, "Questions")
        self.right.addTab(self.viewer, "Conversation")

        split = QSplitter(Qt.Orientation.Horizontal)
        for w, size in ((left, 270), (center, 560), (self.right, 380)):
            split.addWidget(w)
            split.setStretchFactor(split.count() - 1, 1 if w is center else 0)
        split.setSizes([270, 560, 380])
        self.message = QLabel()
        self.message.setWordWrap(True)
        outer = QVBoxLayout(self)
        outer.addWidget(split, 1)
        outer.addWidget(self.message)

        for key, handler in (("A", self.approve), ("R", self.reject), ("N", self.next_image), ("Right", self.next_image),
                             ("P", self.previous_image), ("Left", self.previous_image), ("E", self.toggle_edit)):
            shortcut = QShortcut(QKeySequence(key), self, activated=handler)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.apply_changes).setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

    def _connect(self) -> None:
        self.filter_box.currentIndexChanged.connect(lambda _i: self._filter_changed())
        self.sort_box.currentIndexChanged.connect(lambda _i: self._filter_changed())
        self.list_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.grid_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.list.currentItemChanged.connect(lambda cur, _o: cur and self._picked(cur.data(Qt.ItemDataRole.UserRole)))
        self.grid.current_key.connect(lambda key: key and self._picked(key))
        self.prev_page.clicked.connect(lambda: self._set_page(self.page - 1))
        self.next_page.clicked.connect(lambda: self._set_page(self.page + 1))
        self.bulk_btn.clicked.connect(self.bulk_approve)
        self.upload_btn.clicked.connect(self.open_upload)
        for key, button in self.mode_buttons.items():
            button.clicked.connect(lambda _c, k=key: self.set_mode(k))
        self.delete_btn.clicked.connect(self.delete_selected)
        self.fit_btn.clicked.connect(self.canvas.fit)
        self.approve_btn.clicked.connect(self.approve)
        self.reject_btn.clicked.connect(self.reject)
        self.reopen_btn.clicked.connect(self.reopen)
        self.rerun_btn.clicked.connect(self.rerun)
        self.pdf_btn.clicked.connect(self.rebuild_pdf)
        self.open_pdf_btn.clicked.connect(self.open_pdf)
        self.prev_btn.clicked.connect(self.previous_image)
        self.next_btn.clicked.connect(self.next_image)
        self.canvas.boxChanged.connect(self._canvas_box_changed)
        self.canvas.boxSelected.connect(self._canvas_selected)
        self.canvas.boxDrawn.connect(self._box_drawn)
        self.canvas.pointClicked.connect(self._point_clicked)
        self.boxes.currentItemChanged.connect(lambda cur, _o: self._box_picked(cur.data(Qt.ItemDataRole.UserRole) if cur else ""))
        self.editor.edited.connect(self._editor_edited)
        self.apply_btn.clicked.connect(self.apply_changes)
        self.revert_btn.clicked.connect(self.discard_changes)
        self.add_tax_btn.clicked.connect(self.add_to_taxonomy)
        self.answer_btn.clicked.connect(self.answer_selected)
        self.questions.itemDoubleClicked.connect(lambda _i: self.answer_selected())
        self.ctx.open_image.connect(self.open_image)
        self.ctx.queue_changed.connect(self._queue_changed)

    def _emit_quietly(self) -> None:
        """Tell the other tabs the queue changed without this tab reloading itself in the middle of what it is doing."""
        was, self._loading = self._loading, True
        try:
            self.ctx.queue_changed.emit()
        finally:
            self._loading = was

    # ------------------------------------------------------------------ the list
    def _statuses(self):
        return FILTERS[self.filter_box.currentText()]

    def reload(self, select: str = "") -> None:
        """Re-read the list for the current filter and sort, keeping (or choosing) the selected image."""
        self._loading = True
        self.rows = self.ctx.db.list(self._statuses(), SORTS[self.sort_box.currentText()])
        ids = [r.image_id for r in self.rows]
        target = select or self.image_id
        if target in ids:
            self.page = ids.index(target) // PAGE
        self.page = max(0, min(self.page, max(len(self.rows) - 1, 0) // PAGE))
        self._fill_page()
        self._loading = False
        self.count_label.setText(f"{len(self.rows)} image(s)")
        c = self.ctx.db.counts()
        self.upload_btn.setText(f"Upload approved... ({c['approved'] + c['edited']})")
        if target in ids:
            self._select_in_widgets(target)
        elif not self.image_id or self.image_id not in ids:
            self._clear_selection_if_gone()

    def _fill_page(self) -> None:
        chunk = self.rows[self.page * PAGE:(self.page + 1) * PAGE]
        pages = max(1, -(-len(self.rows) // PAGE))
        self.page_label.setText(f"page {self.page + 1} of {pages}")
        self.prev_page.setEnabled(self.page > 0)
        self.next_page.setEnabled(self.page + 1 < pages)
        self.list.clear()
        for row in chunk:
            conf = f"{row.confidence:.2f}" if row.confidence is not None else "-"
            extra = f", {row.question_count} question(s)" if row.question_count else ""
            why = f"\n{row.error[:90]}" if row.status == "failed" and row.error else ""
            item = QListWidgetItem(f"{row.name}\n{row.status}, confidence {conf}{extra}{why}")
            item.setData(Qt.ItemDataRole.UserRole, row.image_id)
            self.list.addItem(item)
        self.grid.set_items([self._grid_item(r) for r in chunk], keep_selection=False)

    def _grid_item(self, row: QueueRow) -> GridItem:
        boxes = []
        if row.sidecar_path and Path(row.sidecar_path).is_file():
            try:
                ls = read_sidecar(row.sidecar_path)
                boxes = [("laser", b.box, ls.image_width, ls.image_height) for b in ls.laser_boxes]
                boxes += [("object", o.box, ls.image_width, ls.image_height) for o in ls.object_boxes]
            except (OSError, ValueError):
                pass
        conf = f"{row.confidence:.2f}" if row.confidence is not None else "-"
        stamp = f"{row.status}|{row.updated_at}"
        return GridItem(row.image_id, f"{row.name}\n{conf}", row.path if Path(row.path).is_file() else None,
                        f"{row.name}: {row.status}", boxes, cache_key=stamp)

    def _set_page(self, page: int) -> None:
        self.page = max(0, page)
        self._loading = True
        self._fill_page()
        self._loading = False

    def _select_in_widgets(self, image_id: str) -> None:
        self._loading = True
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == image_id:
                self.list.setCurrentRow(i)
        self.grid.select_key(image_id)
        self._loading = False

    def _clear_selection_if_gone(self) -> None:
        if self.image_id and not any(r.image_id == self.image_id for r in self.rows):
            self.image_id = ""
            self.draft = None
            self.dirty = False
            self._show_nothing()

    def _filter_changed(self) -> None:
        if not self._leave_ok():
            return
        self.page = 0
        self.reload()
        if self.rows:
            self.select_image(self.rows[0].image_id, force=True)
        else:
            self.image_id = ""
            self._show_nothing()

    def _queue_changed(self) -> None:
        if not self._loading:
            self.reload()

    def _picked(self, image_id: str) -> None:
        if self._loading or image_id == self.image_id:
            return
        if not self._leave_ok():
            self._select_in_widgets(self.image_id)
            return
        self.select_image(image_id, force=True)

    # ------------------------------------------------------------------ showing one image
    def open_image(self, image_id: str) -> None:
        """Show this image whatever the filter (used when another tab asks for it)."""
        row = self.ctx.db.get(image_id)
        if row is None or not self._leave_ok():
            return
        for name, statuses in FILTERS.items():
            if statuses is None or row.status in statuses:
                self.filter_box.blockSignals(True)
                self.filter_box.setCurrentText(name)
                self.filter_box.blockSignals(False)
                break
        self.reload(select=image_id)
        self.select_image(image_id, force=True)

    def _show_nothing(self) -> None:
        self.header.setText("Nothing to review here." if not self.rows else "Choose an image from the list.")
        self.problems.clear()
        self.canvas.set_image(None)
        self.boxes.clear()
        self.questions.clear()
        self.editor.set_box(None)
        self.viewer.clear_view()
        self.dirty_label.clear()
        self._update_buttons()

    def select_image(self, image_id: str, force: bool = False) -> bool:
        if not force and self.image_id == image_id:
            return True
        if not self._leave_ok():
            return False
        row = self.ctx.db.get(image_id)
        if row is None:
            return False
        self.image_id, self.dirty, self._question_id = image_id, False, ""
        self.mode_buttons["view"].setChecked(True)
        self.canvas.set_mode("view")
        self._select_in_widgets(image_id)
        image_ok = self.canvas.set_image(row.path)
        try:
            ls = self.ctx.service().load(image_id)
        except ReviewError as exc:
            self.draft = None
            self.canvas.set_boxes([])
            self.header.setText(f"<b>{row.name}</b>: {exc}")
            self.problems.setText("" if image_ok else f"The image file was not found at {row.path}.")
            self.boxes.clear()
            self.questions.clear()
            self.editor.set_box(None)
            self.viewer.load(row.transcript_path)
            hint = failure_hint(row.error, row.provider or "", row.model or "") if row.status == "failed" else ""
            self.message.setText(" ".join(t for t in (row.error or "", hint) if t))
            self._update_buttons()
            return True
        self.draft = copy.deepcopy(ls)
        self._refresh_all()
        if row.status == "failed":  # say why, and what to try
            hint = failure_hint(row.error, row.provider or "", row.model or "")
            self.message.setText(" ".join(t for t in (row.error or "", hint) if t))
        if not image_ok:
            self.problems.setText(f"The image file was not found at {row.path}.")
        self.viewer.load(row.transcript_path)
        if row.status == "needs_input":
            self.right.setCurrentIndex(1)
        elif self.right.currentIndex() == 1 and not self.draft.questions:
            self.right.setCurrentIndex(0)
        return True

    def _refresh_all(self, keep_box: str = "") -> None:
        ls, row = self.draft, self.ctx.db.get(self.image_id)
        if ls is None or row is None:
            return
        service = self.ctx.service()
        taxonomy = service.taxonomy()
        self.editor.set_taxonomy(taxonomy, service.placeholder)
        self.header.setText(f"<b>{ls.image_name}</b> &nbsp; {row.status}, confidence "
                            f"{(row.confidence if row.confidence is not None else 0):.2f}, prompt {ls.prompt_version or '-'}, "
                            f"{ls.provider or '-'}/{ls.model or '-'}, {ls.run_mode or '-'} mode"
                            + (f"<br>{ls.notes}" if ls.notes else ""))
        self._show_problems(service.taxonomy_problems(ls))
        selected = keep_box or self.canvas.selected()
        self.canvas.set_boxes(boxes_of(ls), selected)
        self._fill_box_list(selected)
        self._fill_questions()
        self._box_picked(selected, from_list=False)
        self._update_buttons()

    def _show_problems(self, problems: list[str]) -> None:
        self.problems.setText("Not in your taxonomy lists: " + "; ".join(problems) if problems else "")

    def _fill_box_list(self, selected: str) -> None:
        ls = self.draft
        self.boxes.blockSignals(True)
        self.boxes.clear()
        for b in sorted(ls.laser_boxes, key=lambda x: x.order):
            item = QListWidgetItem(f"laser {b.id}: {b.component or b.kind}")
            item.setData(Qt.ItemDataRole.UserRole, b.id)
            self.boxes.addItem(item)
        for o in ls.object_boxes:
            item = QListWidgetItem(f"object {o.id}: {o.general_object}/{o.sub_object}, {o.material}, {o.color}")
            item.setData(Qt.ItemDataRole.UserRole, o.id)
            self.boxes.addItem(item)
        for i in range(self.boxes.count()):
            if self.boxes.item(i).data(Qt.ItemDataRole.UserRole) == selected:
                self.boxes.setCurrentRow(i)
        self.boxes.blockSignals(False)

    def _find_box(self, box_id: str):
        if self.draft is None:
            return None
        return next((b for b in [*self.draft.object_boxes, *self.draft.laser_boxes] if b.id == box_id), None)

    def _box_picked(self, box_id: str, from_list: bool = True) -> None:
        self.editor.set_box(self._find_box(box_id))
        if from_list and box_id:
            self.canvas.select(box_id)
        self.add_tax_btn.setEnabled(isinstance(self._find_box(box_id), ObjectBox))

    def _canvas_selected(self, box_id: str) -> None:
        for i in range(self.boxes.count()):
            if self.boxes.item(i).data(Qt.ItemDataRole.UserRole) == box_id:
                self.boxes.setCurrentRow(i)
                return
        self.boxes.clearSelection()
        self._box_picked("", from_list=False)

    # ------------------------------------------------------------------ editing the draft
    def set_mode(self, key: str) -> None:
        if key.startswith("draw") and self.draft is None:
            self.mode_buttons["view"].setChecked(True)
            return
        self.mode_buttons[key].setChecked(True)
        self.canvas.set_mode({"view": "view", "edit": "edit", "draw_object": "draw", "draw_laser": "draw"}[key])
        self._draw_kind = "laser" if key == "draw_laser" else "object"

    def toggle_edit(self) -> None:
        self.set_mode("view" if self.canvas.mode() == "edit" else "edit")

    def _mark_dirty(self) -> None:
        self.dirty = True
        self.dirty_label.setText("You have changes that are not applied yet. Press Apply changes (Ctrl+S).")
        self._update_buttons()

    def _canvas_box_changed(self, box_id: str, box: tuple) -> None:
        item = self._find_box(box_id)
        if item is not None:
            item.box = tuple(box)  # type: ignore[assignment]
            self._mark_dirty()

    def _editor_edited(self) -> None:
        if self.draft is None:
            return
        selected = self.canvas.selected()
        self.canvas.set_boxes(boxes_of(self.draft), selected)
        self._fill_box_list(selected)
        self._show_problems(self.ctx.service().taxonomy_problems(self.draft))
        self._mark_dirty()

    def _box_drawn(self, rect: tuple) -> None:
        if self.draft is None:
            return
        row, service = self.ctx.db.get(self.image_id), self.ctx.service()
        try:
            if getattr(self, "_draw_kind", "object") == "laser":
                kind = "reflection" if self.draft.laser_boxes else "first_incidence"
                new = add_laser_box(self.draft, row.path, self.ctx.settings(), rect, kind)
            else:
                new = add_object_box(self.draft, row.path, service.taxonomy(), rect, self.ctx.settings())
        except (ReviewError, OSError, ValueError) as exc:
            self.message.setText(f"The box could not be added: {exc}")
            return
        self.set_mode("edit")
        self._mark_dirty()
        self._refresh_all(keep_box=new.id)
        self.canvas.select(new.id)

    def delete_selected(self) -> None:
        box_id = self.canvas.selected()
        if not box_id or self.draft is None:
            self.message.setText("Select a box first.")
            return
        delete_box(self.draft, box_id)
        self._mark_dirty()
        self._refresh_all(keep_box="")

    def add_to_taxonomy(self) -> None:
        box = self.editor.box()
        if not isinstance(box, ObjectBox):
            return
        service, added = self.ctx.service(), []
        tax = service.taxonomy()
        try:
            if box.sub_object not in tax.objects.get(box.general_object, []):
                service.add_to_taxonomy("sub_object", box.sub_object, parent_object=box.general_object)
                added.append(f"{box.general_object}/{box.sub_object}")
            if box.material not in tax.materials:
                service.add_to_taxonomy("material", box.material)
                added.append(box.material)
            if box.color != service.placeholder and box.color not in tax.colors:
                service.add_to_taxonomy("color", box.color)
                added.append(box.color)
        except ReviewError as exc:
            self.message.setText(str(exc))
            return
        self.message.setText("Added to the taxonomy: " + ", ".join(added) + ". Use Generate new prompt on the Prompts tab so the model learns it."
                             if added else "Everything on this box is already in the taxonomy.")
        self.ctx.settings_changed.emit()
        self._refresh_all(keep_box=self.canvas.selected())

    def apply_changes(self) -> bool:
        if self.draft is None or not self.image_id:
            return False
        try:
            result = self.ctx.service().save_edit(self.image_id, self.draft)
        except ReviewError as exc:
            self.message.setText(str(exc))
            return False
        self.dirty = False
        self.dirty_label.clear()
        note = f"Saved {len(result.changes)} change(s); status is now {result.status}." if result.changes else "No changes to save."
        self.message.setText(" ".join([note, *result.warnings]))
        self._emit_quietly()
        self.select_image(self.image_id, force=True)
        return True

    def discard_changes(self) -> None:
        self.dirty = False
        self.dirty_label.clear()
        if self.image_id:
            self.select_image(self.image_id, force=True)

    def _leave_ok(self) -> bool:
        """Before moving to another image: apply, discard or cancel unsaved changes."""
        if not self.dirty:
            return True
        choice = self._ask_unsaved()
        if choice == "apply":
            return self.apply_changes()
        if choice == "discard":
            self.dirty = False
            return True
        return False

    def _ask_unsaved(self) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved changes")
        box.setText("This image has changes that are not applied.")
        apply_btn = box.addButton("Apply them", QMessageBox.ButtonRole.AcceptRole)
        drop_btn = box.addButton("Discard them", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return "apply" if box.clickedButton() is apply_btn else ("discard" if box.clickedButton() is drop_btn else "cancel")

    # ------------------------------------------------------------------ questions
    def _fill_questions(self) -> None:
        self.questions.clear()
        for q in (self.draft.questions if self.draft else []):
            text = q.text if q.answer is None else f"{q.text}\nAnswered: {q.answer}"
            item = QListWidgetItem(("[open] " if q.answer is None else "[done] ") + text)
            item.setData(Qt.ItemDataRole.UserRole, q.id)
            self.questions.addItem(item)
        open_count = sum(q.answer is None for q in (self.draft.questions if self.draft else []))
        self.right.setTabText(1, f"Questions ({open_count})" if open_count else "Questions")
        self.question_hint.setText("" if self._question_id == "" else "Click on the image where the beam first hits.")

    def answer_selected(self) -> None:
        item = self.questions.currentItem()
        if item is None or self.draft is None:
            self.message.setText("Select a question first.")
            return
        question = next((q for q in self.draft.questions if q.id == item.data(Qt.ItemDataRole.UserRole)), None)
        if question is None or question.answer is not None:
            return
        if self.dirty and not self.apply_changes():
            return
        if question.field_name == "hit_point":
            self._question_id = question.id
            self.canvas.set_mode("point")
            self.question_hint.setText("Click on the image where the beam first hits.")
            return
        dialog = QuestionDialog(question, self)
        if dialog.exec():
            self._submit_answer(question.id, dialog.answer(), dialog.add_to_taxonomy())

    def _point_clicked(self, x: float, y: float) -> None:
        if self._question_id:
            qid, self._question_id = self._question_id, ""
            self.canvas.set_mode("view")
            self._submit_answer(qid, f"{x:.0f},{y:.0f}", False)

    def _submit_answer(self, question_id: str, answer: str, add_taxonomy: bool) -> None:
        try:
            warnings = self.ctx.service().answer_question(self.image_id, question_id, answer, add_to_taxonomy=add_taxonomy)
        except ReviewError as exc:
            self.message.setText(str(exc))
            return
        row, index = self.ctx.db.get(self.image_id), self._current_index()
        done = row.status == "pending"
        self.message.setText(" ".join(["Answer saved." + (" All questions answered: it is now under Ready for review." if done else ""),
                                       *warnings]))
        self._emit_quietly()
        statuses = self._statuses()
        if statuses is not None and row.status not in statuses:
            self._next_after_action(index)  # it left this list: carry on with the next one
        else:
            self.reload()
            self.select_image(self.image_id, force=True)

    # ------------------------------------------------------------------ actions
    def _next_after_action(self, index: int) -> None:
        """After an image leaves the current list, show the one that took its place."""
        self.reload()
        if self.rows:
            self.select_image(self.rows[min(index, len(self.rows) - 1)].image_id, force=True)
        else:
            self.image_id = ""
            self._show_nothing()

    def _current_index(self) -> int:
        return next((i for i, r in enumerate(self.rows) if r.image_id == self.image_id), 0)

    def approve(self) -> bool:
        if not self.image_id or self.draft is None:
            return False
        if self.dirty and not self.apply_changes():
            return False
        service = self.ctx.service()
        problems = service.taxonomy_problems(service.load(self.image_id))
        if problems and not self.confirm_taxonomy(problems):
            return False
        index = self._current_index()
        try:
            warnings = service.approve(self.image_id)
        except ReviewError as exc:
            self.message.setText(str(exc))
            return False
        self.message.setText(" ".join(["Approved.", *warnings]))
        self._emit_quietly()
        self._next_after_action(index)
        return True

    def _ask_taxonomy(self, problems: list[str]) -> bool:
        return QMessageBox.question(self, "Not in your taxonomy", "These values are not in your taxonomy lists:\n\n- "
                                    + "\n- ".join(problems) + "\n\nApprove anyway? (You can cancel and use 'Add this box's "
                                    "values to the taxonomy' first.)", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                                    QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Yes

    def reject(self) -> bool:
        if not self.image_id:
            return False
        index = self._current_index()
        self.dirty = False
        try:
            self.ctx.service().reject(self.image_id)
        except ReviewError as exc:
            self.message.setText(str(exc))
            return False
        self.message.setText("Rejected. Use the Rejected filter and 'Undo' to bring it back.")
        self._emit_quietly()
        self._next_after_action(index)
        return True

    def reopen(self) -> None:
        if not self.image_id:
            return
        index = self._current_index()
        try:
            self.ctx.service().reopen(self.image_id)
        except ReviewError as exc:
            self.message.setText(str(exc))
            return
        self.message.setText("Moved back to ready for review.")
        self._emit_quietly()
        self._next_after_action(index)

    def next_image(self) -> None:
        self._step(1)

    def previous_image(self) -> None:
        self._step(-1)

    def _step(self, delta: int) -> None:
        if not self.rows:
            return
        target = min(max(self._current_index() + delta, 0), len(self.rows) - 1)
        if self.rows[target].image_id != self.image_id:
            self.select_image(self.rows[target].image_id, force=True)

    def rerun(self) -> None:
        if not self.image_id:
            return
        note, ok = QInputDialog.getMultiLineText(self, "Re-run with a note", "Tell the model what to look at differently (optional):")
        if not ok:
            return
        self.rerun_with(note)

    def rerun_with(self, note: str) -> bool:
        index = self._current_index()
        self.dirty = False
        try:
            self.ctx.service().rerun(self.image_id, note)
        except ReviewError as exc:
            self.message.setText(str(exc))
            return False
        self.message.setText("Queued again. It will be labeled by the next run (or the one in progress), with your note.")
        self._emit_quietly()
        self._next_after_action(index)
        return True

    def rebuild_pdf(self) -> None:
        if not self.image_id:
            return
        if self.dirty and not self.apply_changes():
            return
        try:
            path = self.ctx.service().rebuild_pdf(self.image_id)
        except Exception as exc:
            self.message.setText(f"The PDF could not be built: {exc}")
            return
        self.message.setText(f"PDF saved: {path}")
        self._emit_quietly()

    def open_pdf(self) -> None:
        row = self.ctx.db.get(self.image_id) if self.image_id else None
        if row and row.pdf_path and Path(row.pdf_path).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(row.pdf_path))
        else:
            self.message.setText("This image has no PDF yet. Press Rebuild PDF first.")

    def bulk_approve(self) -> list[str]:
        if not self._leave_ok():
            return []
        ids = self.ctx.service().bulk_approve(self.threshold.value())
        self.message.setText(f"Approved {len(ids)} image(s) with confidence at or above {self.threshold.value():.2f} and no open questions.")
        self._emit_quietly()
        self.reload()
        if self.rows:
            self.select_image(self.rows[0].image_id, force=True)
        else:
            self.image_id = ""
            self._show_nothing()
        return ids

    # ------------------------------------------------------------------ upload
    def open_upload(self) -> UploadDialog | None:
        if not self._leave_ok():
            return None
        rf = self.ctx.settings()["roboflow"]
        if not (rf.get("workspace") and rf.get("project")):
            self.message.setText("Choose a Roboflow workspace and project on the Sign in tab first.")
            return None
        service = self.ctx.service()
        ids = [r.image_id for r in self.ctx.db.list(["approved", "edited"])]
        if not ids:
            self.message.setText("Nothing is approved yet.")
            return None
        stale = [i for i in ids if not service.pdf_is_current(self.ctx.db.get(i))]
        if stale and not self._build_pdfs(stale):
            return None
        placeholder = service.placeholder

        def plan_fn(overwrite: bool):
            return build_upload_plan(service.approved_sets(), workspace=rf["workspace"], project=rf["project"],
                                     color_placeholder=placeholder, overwrite=overwrite)

        def send_fn(plan, dry_run: bool) -> UploadSummary:
            key = self._key_fn()
            return self._upload_fn(plan, key, dry_run=dry_run, audit_log=self.ctx.data_dir / "upload_audit.jsonl",
                                   out=lambda _s: None)

        def done_fn(summary: UploadSummary) -> None:
            self.ctx.service().mark_uploaded(summary.uploaded)

        dialog = UploadDialog(plan_fn, send_fn, done_fn, self)
        dialog.uploaded.connect(lambda _ids: (self._emit_quietly(), self.reload()))
        dialog.show()
        self.upload_dialog = dialog
        return dialog

    def _build_pdfs(self, ids: list[str]) -> bool:
        progress = QProgressDialog("Building PDFs...", "Cancel", 0, len(ids), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        service, problems = self.ctx.service(), []
        for n, image_id in enumerate(ids):
            progress.setValue(n)
            QApplication.processEvents()
            if progress.wasCanceled():
                return False
            problems += service.ensure_pdfs([image_id])
        progress.setValue(len(ids))
        if problems:
            self.message.setText(f"{len(problems)} PDF(s) could not be built (for example {problems[0]}). The upload goes on without them.")
        return True

    # ------------------------------------------------------------------ button states
    def _update_buttons(self) -> None:
        row = self.ctx.db.get(self.image_id) if self.image_id else None
        status = row.status if row else ""
        have = self.draft is not None
        self.approve_btn.setEnabled(have and status in ("pending", "edited", "approved"))
        self.reject_btn.setEnabled(bool(row) and status in ("needs_input", "failed", "pending", "approved", "edited"))
        self.reopen_btn.setEnabled(status in ("approved", "edited", "rejected"))
        self.rerun_btn.setEnabled(bool(row) and status in ("needs_input", "failed", "pending", "approved", "edited", "rejected"))
        self.pdf_btn.setEnabled(have)
        self.open_pdf_btn.setEnabled(bool(row and row.pdf_path))
        self.apply_btn.setEnabled(have and self.dirty)
        self.revert_btn.setEnabled(have and self.dirty)
        for w in (self.delete_btn, *self.mode_buttons.values()):
            w.setEnabled(have)
        self.prev_btn.setEnabled(bool(self.rows))
        self.next_btn.setEnabled(bool(self.rows))
