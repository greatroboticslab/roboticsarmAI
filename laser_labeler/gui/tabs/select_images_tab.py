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
from core.schema.scheme_labels import scheme_example
from core.selection import (LABELED, OLDER, UNKNOWN, Entry, filter_entries, label_state, next_unticked, split_options,
                            tag_options)
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
        self._scheme = self.scheme()
        self._states: dict[str, str] = {}  # entry key -> label state in the current scheme (remote entries only)
        self._needs = 0  # how many remote entries are not labeled in the current system
        self.ask_unknown: Callable[[int], str] = self._ask_unknown  # replaced in tests
        self.ask_labeled: Callable[[int, str], str] = self._ask_labeled  # replaced in tests

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
        self.all_radio = QRadioButton("All images")
        self.unlabeled_radio = QRadioButton("Needs labeling (not labeled in the current system)")
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
        self.unlabeled_radio.toggled.connect(lambda _v: self.refresh_view())  # the pair toggles together: refresh once
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
        self.ctx.settings_changed.connect(self._settings_changed)

    def _source_changed(self) -> None:
        self.entries, self.ticked = [], set()
        self._compute_states()
        self._refresh_source()
        self.refresh_view()

    def scheme(self) -> str:
        return self.ctx.settings()["labeling"].get("scheme", "full")

    def _compute_states(self) -> None:
        """Work out every remote image's label state once. (Doing it per image, per refresh, reading the settings file
        each time, is what made big Roboflow lists freeze.)"""
        scheme = self.scheme()
        self._scheme = scheme
        self._states = {e.key: label_state(e, scheme) for e in self.entries if e.remote is not None}
        self._needs = sum(state != LABELED for state in self._states.values())

    def _state(self, entry: Entry) -> str:
        return self._states.get(entry.key, "")

    def _settings_changed(self) -> None:
        old = getattr(self, "_scheme", None)
        self._refresh_source()
        if self.scheme() != old:
            self._compute_states()
        self.refresh_view()

    def _refresh_source(self) -> None:
        self.unlabeled_radio.setToolTip(
            f"An image counts as labeled when Roboflow already shows a laser label written the way your current label "
            f"scheme ('{self.scheme()}') writes it, for example {scheme_example(self.scheme())}. Images with no labels, "
            "or only labels of an older kind, are shown here so they can be labeled in the current system.")
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
        self._compute_states()
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
        self.unlabeled_radio.blockSignals(True)  # start from the images that still need labels; "All images" shows the rest
        self.unlabeled_radio.setChecked(True)
        self.unlabeled_radio.blockSignals(False)
        self.set_entries([Entry(i.id, i.name or i.id, None, i.split, list(i.tags), i.labels, i) for i in images])
        states = list(self._states.values())
        self.message.setText(
            f"Loaded {len(images)} image(s) from Roboflow: {states.count(LABELED)} already labeled in the current system, "
            f"{states.count(OLDER)} labeled in an older style, {len(images) - states.count(LABELED) - states.count(OLDER)} "
            "without labels or of unknown state. Showing the ones that need labeling.")

    def refresh_view(self) -> None:
        queued = self.ctx.db.ids()
        split = self.split_box.currentText() if self.split_box.currentText() != ALL else ""
        tag = self.tag_box.currentText() if self.tag_box.currentText() != ALL else ""
        self._shown = filter_entries(self.entries, only_unlabeled=self.unlabeled_radio.isChecked(), split=split, tag=tag,
                                     text=self.search.text(), queued=queued, hide_queued=self.hide_queued.isChecked(),
                                     scheme=self._scheme, states=self._states)
        items = []
        for e in self._shown:
            state = self._state(e)
            mark = {"unknown": "? ", "labeled": "* ", "older": "~ "}.get(state, "")
            tip = f"{e.name}\nsplit: {e.split or '-'}\ntags: {', '.join(e.tags) or '-'}"
            if state:
                tip += f"\nlabels: {self._state_text(state, e)}"
            items.append(GridItem(e.key, mark + e.name, e.path, tip, checked=e.key in self.ticked))
        self.grid.set_items(items)
        self._update_counts()

    def _state_text(self, state: str, entry: Entry) -> str:
        if state == UNKNOWN:
            return "unknown (Roboflow did not say; you will be asked)"
        if state == LABELED:
            return f"labeled in the current system ('{self.scheme()}'); marked * so it is not labeled twice by accident"
        if state == OLDER:
            return (f"has labels ({', '.join((entry.labels or [])[:4])}) but none in the current system "
                    f"('{self.scheme()}'); marked ~, it still needs labeling")
        return "none"

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
            self.grid.set_checked_exactly(self.ticked)
        finally:
            self._syncing = False
        self._update_counts()

    def tick_chunk(self) -> None:
        self._set_ticks(next_unticked(self._shown, self.ticked, self.chunk.value()), True)

    def _update_counts(self) -> None:
        total_queued = sum(self.ctx.db.counts().values())
        extra = f", {self._needs} need labeling" if self._states else ""
        self.count_label.setText(f"{len(self.ticked)} ticked, {len(self._shown)} shown, {len(self.entries)} loaded{extra}, "
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

    def _ask_labeled(self, count: int, scheme: str) -> str:
        box = QMessageBox(self)
        box.setWindowTitle("Some images are already labeled")
        box.setText(f"{count} of the ticked image(s) already have labels in the current system ('{scheme}').")
        box.setInformativeText("Labeling them again would spend model time on work that is done, and replacing the labels "
                               "on Roboflow needs Overwrite. Skip them?")
        skip = box.addButton("Skip them", QMessageBox.ButtonRole.AcceptRole)
        include = box.addButton("Label them again", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        return "skip" if clicked is skip else ("include" if clicked is include else "cancel")

    def add_to_queue(self) -> int:
        """Queue the ticked images. Returns how many were added (remote downloads finish in the background)."""
        chosen = [e for e in self.entries if e.key in self.ticked]
        scheme = self.scheme()
        done = {e.key for e in chosen if e.remote is not None and self._state(e) == LABELED}
        if done:  # only label what is not labeled yet, unless the person says otherwise
            answer = self.ask_labeled(len(done), scheme)
            if answer == "cancel":
                return 0
            if answer == "skip":
                chosen = [e for e in chosen if e.key not in done]
                self.ticked -= done
                self.grid.set_checked(done, False)
        unknown = {e.key for e in chosen if e.remote is not None and label_state(e) == UNKNOWN}
        if unknown:
            answer = self.ask_unknown(len(unknown))
            if answer == "cancel":
                return 0
            if answer == "skip":
                chosen = [e for e in chosen if e.key not in unknown]
        if not chosen:
            self.message.setText("Nothing to add.")
            return 0
        local = [e for e in chosen if e.remote is None]
        remote = [e for e in chosen if e.remote is not None]
        self._older_added = sum(self._state(e) == OLDER for e in remote)
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
        older = getattr(self, "_older_added", 0)
        if older and added:
            text += (f" {older} of the downloaded image(s) already have labels on Roboflow in an older style: when you upload "
                     "them, tick Overwrite in the upload window to replace those labels.")
        self._older_added = 0
        self.message.setText(text)
        self.ctx.queue_changed.emit()  # this tab listens too, so it redraws once, not twice

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
