"""Select images: choose local files or a Roboflow project's images, filter, tick, pick a chunk, add to the queue (spec section 10)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QProgressBar, QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget)

from config import credentials
from config.credentials import CredentialError, MissingCredentialError
from core.local_files import collect_files, make_image_id
from core.queue_db import NewImage
from core.roboflow.enqueue import enqueue_remote
from core.roboflow.images import download_image, list_images
from core.selection import (UNKNOWN, Entry, filter_entries, label_state, next_unticked, split_options, tag_options)
from gui.context import AppContext
from gui.widgets.thumb_grid import GridItem, ThumbnailGrid
from gui.workers.progress_worker import ProgressWorker

ALL = "All"


class SelectImagesTab(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None, *, list_fn: Callable = list_images,
                 download_fn: Callable = download_image, key_fn: Callable[[], str] | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._list_fn, self._download_fn = list_fn, download_fn
        self._key_fn = key_fn or (lambda: credentials.require_key("roboflow"))
        self.entries: list[Entry] = []
        self.ticked: set[str] = set()
        self._shown: list[Entry] = []
        self._worker: ProgressWorker | None = None
        self._syncing = False
        self.ask_unknown: Callable[[int], str] = self._ask_unknown  # replaced in tests

        self.local_radio, self.remote_radio = QRadioButton("Files on this computer"), QRadioButton("Images in my Roboflow project")
        self.local_radio.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(self.local_radio)
        group.addButton(self.remote_radio)
        self.choose_btn = QPushButton("Choose folder...")
        self.tag_edit = QLineEdit()
        self.tag_edit.setPlaceholderText("Tag for these images, e.g. camera:logitech (optional)")
        self.load_btn = QPushButton("Load images from Roboflow")
        self.project_label = QLabel()
        self.all_radio, self.unlabeled_radio = QRadioButton("All images"), QRadioButton("Unlabeled only")
        self.all_radio.setChecked(True)
        filter_group = QButtonGroup(self)
        filter_group.addButton(self.all_radio)
        filter_group.addButton(self.unlabeled_radio)
        self.split_box, self.tag_box = QComboBox(), QComboBox()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search by name")
        self.hide_queued = QCheckBox("Hide images already in the queue")
        self.hide_queued.setChecked(True)
        self.grid = ThumbnailGrid()
        self.tick_all_btn, self.untick_btn = QPushButton("Tick all shown"), QPushButton("Untick all")
        self.chunk = QSpinBox()
        self.chunk.setRange(1, 100000)
        self.chunk.setValue(250)
        self.chunk_btn = QPushButton("Tick the next")
        self.count_label = QLabel()
        self.estimate_label = QLabel()
        self.estimate_label.setWordWrap(True)
        self.add_btn = QPushButton("Add ticked images to the queue")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.hide()
        self.bar = QProgressBar()
        self.bar.hide()
        self.message = QLabel()
        self.message.setWordWrap(True)
        self._layout()
        self._connect()
        self._refresh_source()
        self.refresh_view()

    # ------------------------------------------------------------------ layout and wiring
    def _layout(self) -> None:
        col = QVBoxLayout(self)
        source = QHBoxLayout()
        for w in (self.local_radio, self.remote_radio):
            source.addWidget(w)
        source.addStretch(1)
        col.addLayout(source)
        local = QHBoxLayout()
        local.addWidget(self.choose_btn)
        local.addWidget(self.tag_edit, 1)
        local.addWidget(self.load_btn)
        local.addWidget(self.project_label)
        col.addLayout(local)
        filters = QHBoxLayout()
        for w in (self.all_radio, self.unlabeled_radio, QLabel("Split"), self.split_box, QLabel("Tag"), self.tag_box):
            filters.addWidget(w)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.hide_queued)
        col.addLayout(filters)
        col.addWidget(self.grid, 1)
        ticks = QHBoxLayout()
        for w in (self.tick_all_btn, self.untick_btn, self.chunk_btn, self.chunk, QLabel("images")):
            ticks.addWidget(w)
        ticks.addStretch(1)
        ticks.addWidget(self.count_label)
        col.addLayout(ticks)
        col.addWidget(self.estimate_label)
        run = QHBoxLayout()
        run.addWidget(self.add_btn)
        run.addWidget(self.bar, 1)
        run.addWidget(self.cancel_btn)
        col.addLayout(run)
        col.addWidget(self.message)

    def _connect(self) -> None:
        self.local_radio.toggled.connect(self._source_changed)
        self.choose_btn.clicked.connect(self.choose_files)
        self.load_btn.clicked.connect(self.load_remote)
        for radio in (self.all_radio, self.unlabeled_radio):
            radio.toggled.connect(lambda _v: self.refresh_view())
        for combo in (self.split_box, self.tag_box):
            combo.currentIndexChanged.connect(lambda _i: self.refresh_view())
        self.search.textChanged.connect(lambda _t: self.refresh_view())
        self.hide_queued.toggled.connect(lambda _v: self.refresh_view())
        self.grid.checks_changed.connect(self._ticks_changed)
        self.tick_all_btn.clicked.connect(lambda: self._set_ticks([e.key for e in self._shown], True))
        self.untick_btn.clicked.connect(lambda: self._set_ticks(list(self.ticked), False))
        self.chunk_btn.clicked.connect(self.tick_chunk)
        self.add_btn.clicked.connect(self.add_to_queue)
        self.cancel_btn.clicked.connect(lambda: self._worker and self._worker.cancel())
        self.ctx.queue_changed.connect(self.refresh_view)
        self.ctx.settings_changed.connect(self._refresh_source)

    def _source_changed(self) -> None:
        self.entries, self.ticked = [], set()
        self._refresh_source()
        self.refresh_view()

    def _refresh_source(self) -> None:
        local = self.local_radio.isChecked()
        for w in (self.choose_btn, self.tag_edit):
            w.setVisible(local)
        for w in (self.load_btn, self.project_label):
            w.setVisible(not local)
        rf = self.ctx.settings()["roboflow"]
        self.project_label.setText(f"{rf['workspace']}/{rf['project']}" if rf.get("workspace") and rf.get("project")
                                   else "No project chosen: pick one on the Sign in tab")

    # ------------------------------------------------------------------ filling the list
    def set_entries(self, entries: list[Entry]) -> None:
        self.entries = entries
        self.ticked &= {e.key for e in entries}
        for combo, options in ((self.split_box, split_options(entries)), (self.tag_box, tag_options(entries))):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems([ALL] + options)
            combo.blockSignals(False)
        self.refresh_view()

    def choose_files(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder of images")
        if folder:
            self.scan_paths([folder])

    def scan_paths(self, paths: list[str | Path]) -> int:
        """List the image files in these folders. Returns how many were found."""
        files = collect_files(paths)
        batch: dict[str, Path] = {}
        entries = [Entry(make_image_id(self.ctx.db, p, batch), p.name, str(p), tags=[]) for p in files]
        self.set_entries(entries)
        self.message.setText(f"Found {len(entries)} image(s)." if entries else "No image files were found there.")
        return len(entries)

    def load_remote(self) -> None:
        rf = self.ctx.settings()["roboflow"]
        if not (rf.get("workspace") and rf.get("project")):
            self.message.setText("Choose a Roboflow workspace and project on the Sign in tab first.")
            return
        try:
            key = self._key_fn()
        except (CredentialError, MissingCredentialError) as exc:
            self.message.setText(str(exc))
            return
        ws, project = rf["workspace"], rf["project"]

        def job(progress, cancelled):
            found = []
            for image in self._list_fn(ws, project, key):
                found.append(image)
                if len(found) % 50 == 0:
                    progress(len(found), 0)
                if cancelled():
                    break
            return found

        self._start_job(job, self._remote_loaded, "Loading the image list from Roboflow...")

    def _remote_loaded(self, images: list) -> None:
        self._job_done()
        self.set_entries([Entry(i.id, i.name or i.id, None, i.split, list(i.tags), i.labels, i) for i in images])
        self.message.setText(f"Loaded {len(images)} image(s) from Roboflow.")

    def refresh_view(self) -> None:
        queued = {r.image_id for r in self.ctx.db.list()}
        split = self.split_box.currentText() if self.split_box.currentText() != ALL else ""
        tag = self.tag_box.currentText() if self.tag_box.currentText() != ALL else ""
        self._shown = filter_entries(self.entries, only_unlabeled=self.unlabeled_radio.isChecked(), split=split, tag=tag,
                                     text=self.search.text(), queued=queued, hide_queued=self.hide_queued.isChecked())
        items = []
        for e in self._shown:
            state = label_state(e) if e.remote is not None else ""
            mark = {"unknown": "? ", "labeled": "* "}.get(state, "")
            tip = f"{e.name}\nsplit: {e.split or '-'}\ntags: {', '.join(e.tags) or '-'}"
            if state:
                tip += f"\nlabels: {state}" + (" (Roboflow did not say; you will be asked)" if state == UNKNOWN else "")
            items.append(GridItem(e.key, mark + e.name, e.path, tip, checked=e.key in self.ticked))
        self.grid.set_items(items)
        self._update_counts()

    # ------------------------------------------------------------------ ticking
    def _ticks_changed(self) -> None:
        if self._syncing:
            return
        shown = {e.key for e in self._shown}
        self.ticked = (self.ticked - shown) | set(self.grid.checked_keys())
        self._update_counts()

    def _set_ticks(self, keys: list[str], on: bool) -> None:
        self.ticked = (self.ticked | set(keys)) if on else (self.ticked - set(keys))
        self._syncing = True  # the grid reports its own changes; do not let that overwrite the new set
        try:
            self.grid.set_checked(self.grid.keys(), False)
            self.grid.set_checked([k for k in self.grid.keys() if k in self.ticked], True)
        finally:
            self._syncing = False
        self._update_counts()

    def tick_chunk(self) -> None:
        self._set_ticks(next_unticked(self._shown, self.ticked, self.chunk.value()), True)

    def _update_counts(self) -> None:
        total_queued = len(self.ctx.db.list())
        self.count_label.setText(f"{len(self.ticked)} ticked, {len(self._shown)} shown, {len(self.entries)} loaded, "
                                 f"{total_queued} already in the queue")
        n = len(self.ticked)
        self.estimate_label.setText(self.ctx.estimate(n).describe() if n else "Tick images to see a time and cost estimate.")
        self.add_btn.setEnabled(n > 0 and self._worker is None)

    # ------------------------------------------------------------------ adding
    def _ask_unknown(self, count: int) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("Some images have an unknown label state")
        box.setText(f"Roboflow did not say whether {count} of the ticked image(s) already have labels.")
        box.setInformativeText("Labeling them is safe: nothing is uploaded unless you approve it, and existing annotations "
                               "are never overwritten unless you turn that on. Include them?")
        include = box.addButton("Include them", QMessageBox.ButtonRole.AcceptRole)
        skip = box.addButton("Skip them", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        return "include" if box.clickedButton() is include else ("skip" if box.clickedButton() is skip else "cancel")

    def add_to_queue(self) -> int:
        """Queue the ticked images. Returns how many were added (remote downloads finish in the background)."""
        chosen = [e for e in self.entries if e.key in self.ticked]
        unknown = [e for e in chosen if e.remote is not None and label_state(e) == UNKNOWN]
        if unknown:
            answer = self.ask_unknown(len(unknown))
            if answer == "cancel":
                return 0
            if answer == "skip":
                chosen = [e for e in chosen if e not in unknown]
        if not chosen:
            self.message.setText("Nothing to add.")
            return 0
        local = [e for e in chosen if e.remote is None]
        remote = [e for e in chosen if e.remote is not None]
        tags = [t.strip() for t in self.tag_edit.text().split(",") if t.strip()]
        added = self.ctx.db.add_images([NewImage(e.key, e.name, e.path or "", "local", tags) for e in local]) if local else 0
        self.ticked -= {e.key for e in local}
        if remote:
            self._download(remote, added)
        else:
            self._finish_add(added, 0, {})
        return added

    def _download(self, remote: list[Entry], already_added: int) -> None:
        try:
            key = self._key_fn()
        except (CredentialError, MissingCredentialError) as exc:
            self.message.setText(str(exc))
            return
        images_dir = self.ctx.data_dir / "images"

        def job(progress, cancelled):
            return enqueue_remote(self.ctx.db, [e.remote for e in remote], lambda img: self._download_fn(img, images_dir, key),
                                  on_progress=progress, should_stop=cancelled)

        def done(result):
            self._job_done()
            self.ticked -= {e.key for e in remote}
            self._finish_add(already_added + result.added, result.already_known, result.failed)

        self._start_job(job, done, "Downloading images...")

    def _finish_add(self, added: int, known: int, failed: dict) -> None:
        text = f"Added {added} image(s) to the queue."
        if known:
            text += f" {known} were already there."
        if failed:
            first = next(iter(failed.items()))
            text += f" {len(failed)} could not be downloaded (for example {first[0]}: {first[1]}); tick them and add again to retry."
        self.message.setText(text)
        self.ctx.queue_changed.emit()
        self.refresh_view()

    # ------------------------------------------------------------------ background jobs
    def _start_job(self, job: Callable, on_done: Callable, text: str) -> None:
        if self._worker is not None:
            return
        self.message.setText(text)
        worker = ProgressWorker(job, self)
        worker.progress.connect(self._progress)
        worker.succeeded.connect(on_done)
        worker.failed.connect(self._job_failed)
        self._worker = worker
        for w in (self.load_btn, self.add_btn, self.choose_btn):
            w.setEnabled(False)
        self.bar.setRange(0, 0)
        self.bar.show()
        self.cancel_btn.show()
        worker.start()

    def _progress(self, done: int, total: int) -> None:
        self.bar.setRange(0, total)
        self.bar.setValue(done)
        self.message.setText(f"{done} of {total} done..." if total else f"{done} found so far...")

    def _job_done(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.wait()
            worker.deleteLater()
        self.bar.hide()
        self.cancel_btn.hide()
        for w in (self.load_btn, self.choose_btn):
            w.setEnabled(True)
        self._update_counts()

    def _job_failed(self, message: str) -> None:
        self._job_done()
        self.message.setText(f"That did not work: {message}")

    def busy(self) -> bool:
        return self._worker is not None
