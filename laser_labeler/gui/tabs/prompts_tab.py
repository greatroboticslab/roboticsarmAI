"""Prompts and taxonomy: edit prompt parts and taxonomy files, save numbered versions, diff them, generate a new prompt (spec sections 7 and 10)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QInputDialog, QLabel,
                             QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                             QSplitter, QTabWidget, QVBoxLayout, QWidget)

from config.credentials import CredentialError, MissingCredentialError
from core.labeler.guard import TrackerGuard
from core.prompts import versions as versions_mod
from core.prompts.examples import apply_to_background, load_corrections, set_included
from core.prompts.generate import (PART_NAME, GenerateError, Proposal, apply_proposal, generate_new_prompt,
                                   old_taxonomy_for_active, proposal_pairs, read_parts)
from core.prompts.manager import MODES, build_system_prompt
from core.schema.taxonomy import LIST_FILES, OBJECTS_FILE, TaxonomyError, load_taxonomy
from gui.context import AppContext
from gui.widgets.diff_view import PartsDiff
from gui.workers.verify_worker import start_call

TAXONOMY_FILES = (OBJECTS_FILE, *LIST_FILES)
MONO = QFont("Consolas, Menlo, monospace", 10)


class PromptsTab(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None, *, provider_fn: Callable | None = None) -> None:
        super().__init__(parent)
        self.ctx = ctx
        self._provider_fn = provider_fn or ctx.provider
        self._unsaved: dict[str, str] = {}
        self._current_part = ""
        self._loading = False
        self.proposal: Proposal | None = None
        self._runner = None
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.message = QLabel()
        self.message.setWordWrap(True)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._parts_page(), "Prompt parts")
        self.tabs.addTab(self._versions_page(), "Versions")
        self.tabs.addTab(self._generate_page(), "Generate new prompt")
        self.tabs.addTab(self._taxonomy_page(), "Taxonomy")
        self.tabs.addTab(self._corrections_page(), "Corrections")
        col = QVBoxLayout(self)
        col.addWidget(self.banner)
        col.addWidget(self.tabs, 1)
        col.addWidget(self.message)
        ctx.prompt_changed.connect(self.refresh)
        ctx.settings_changed.connect(self._refresh_banner)
        self.refresh()

    # ------------------------------------------------------------------ prompt parts
    def _parts_page(self) -> QWidget:
        self.part_list = QListWidget()
        self.part_list.setMaximumWidth(240)
        self.editor = QPlainTextEdit()
        self.editor.setFont(MONO)
        self.save_draft_btn, self.new_part_btn = QPushButton("Save draft"), QPushButton("New part...")
        self.preview_btn = QPushButton("Preview the assembled prompt")
        self.save_version_btn = QPushButton("Save as a new version...")
        hint = QLabel("Parts are joined in order of their number. {{taxonomy}}, {{table_rule}} and {{color_placeholder}} are filled in by "
                      "the app. Files ending .agent.md or .pipeline.md are used only in that mode. Part 07 holds your project background: "
                      "the more detail the better.")
        hint.setWordWrap(True)
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.addWidget(hint)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.part_list)
        split.addWidget(self.editor)
        split.setSizes([200, 700])
        outer.addWidget(split, 1)
        row = QHBoxLayout()
        for w in (self.save_draft_btn, self.new_part_btn, self.preview_btn, self.save_version_btn):
            row.addWidget(w)
        row.addStretch(1)
        outer.addLayout(row)
        self.part_list.currentItemChanged.connect(self._part_picked)
        self.editor.textChanged.connect(self._editor_changed)
        self.save_draft_btn.clicked.connect(self.save_draft)
        self.new_part_btn.clicked.connect(self.new_part)
        self.preview_btn.clicked.connect(self.preview_prompt)
        self.save_version_btn.clicked.connect(self.save_version)
        return page

    def _fill_parts(self, keep: str = "") -> None:
        self._loading = True
        parts = read_parts(self.ctx.parts_dir)
        names = sorted(set(parts) | set(self._unsaved))
        self.part_list.clear()
        for name in names:
            item = QListWidgetItem(name + (" *" if name in self._unsaved else ""))
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.part_list.addItem(item)
        target = keep or self._current_part or (names[0] if names else "")
        for i in range(self.part_list.count()):
            if self.part_list.item(i).data(Qt.ItemDataRole.UserRole) == target:
                self.part_list.setCurrentRow(i)
        self._loading = False
        self._show_part(self.part_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.part_list.currentItem() else "")

    def _part_text(self, name: str) -> str:
        if name in self._unsaved:
            return self._unsaved[name]
        path = self.ctx.parts_dir / name
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def _show_part(self, name: str) -> None:
        self._loading = True
        self._current_part = name
        self.editor.setPlainText(self._part_text(name) if name else "")
        self.editor.setEnabled(bool(name))
        self._loading = False

    def _part_picked(self, cur, _old) -> None:
        if not self._loading and cur is not None:
            self._show_part(cur.data(Qt.ItemDataRole.UserRole))

    def _editor_changed(self) -> None:
        if self._loading or not self._current_part:
            return
        text = self.editor.toPlainText()
        saved = (self.ctx.parts_dir / self._current_part)
        if text == (saved.read_text(encoding="utf-8") if saved.is_file() else ""):
            self._unsaved.pop(self._current_part, None)
        else:
            self._unsaved[self._current_part] = text
        item = self.part_list.currentItem()
        if item is not None:
            item.setText(self._current_part + (" *" if self._current_part in self._unsaved else ""))
        self._refresh_banner()

    def save_draft(self) -> int:
        """Write every edited part to the draft folder. Returns how many files were written."""
        count = 0
        for name, text in list(self._unsaved.items()):
            if not PART_NAME.match(name):
                self.message.setText(f"{name} is not a valid part name (use a name like 05_questions.md).")
                continue
            (self.ctx.parts_dir / name).write_text(text.rstrip("\n") + "\n", encoding="utf-8")
            self._unsaved.pop(name)
            count += 1
        self.message.setText(f"Saved {count} part(s) to the draft. Runs use the ACTIVE version until you save and activate a new one."
                             if count else "No edits to save.")
        self._fill_parts()
        self._refresh_banner()
        return count

    def new_part(self) -> None:
        name, ok = QInputDialog.getText(self, "New prompt part", "File name, for example 08_ceramics.md (agent-only: 08_x.agent.md):")
        if ok:
            self.create_part(name.strip())

    def create_part(self, name: str) -> bool:
        if not PART_NAME.match(name):
            self.message.setText("Use a name like 08_ceramics.md: two digits, an underscore, lowercase words, .md")
            return False
        if (self.ctx.parts_dir / name).exists():
            self.message.setText(f"{name} already exists.")
            return False
        self._unsaved[name] = "## New part\n\n"
        self._fill_parts(keep=name)
        return True

    def preview_prompt(self) -> None:
        if self._unsaved and not self.save_draft():
            return
        settings = self.ctx.settings()
        mode, ok = QInputDialog.getItem(self, "Preview", "Which run mode?", list(MODES), 1, False)
        if not ok:
            return
        try:
            build = build_system_prompt(mode, self.ctx.service().taxonomy(), table_rule=settings["labeling"]["table_rule"],
                                        color_placeholder=settings["labeling"]["color_placeholder"], parts_dir=self.ctx.parts_dir)
        except (OSError, TaxonomyError, FileNotFoundError) as exc:
            self.message.setText(f"The prompt could not be built: {exc}")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Assembled {mode} prompt ({build.version}), {len(build.text)} characters")
        dialog.resize(800, 600)
        view = QPlainTextEdit(build.text)
        view.setReadOnly(True)
        view.setFont(MONO)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        box.rejected.connect(dialog.reject)
        lay = QVBoxLayout(dialog)
        lay.addWidget(view)
        lay.addWidget(box)
        dialog.exec()

    def save_version(self) -> str | None:
        if self._unsaved and not self.save_draft():
            return None
        note, ok = QInputDialog.getText(self, "Save a new version", "A short note about what changed (optional):")
        return self.make_version(note.strip()) if ok else None

    def make_version(self, note: str = "", activate: bool = True) -> str | None:
        try:
            tax = self.ctx.service().taxonomy().to_dict()
            version = versions_mod.save_version(self.ctx.parts_dir, self.ctx.versions_dir, note, taxonomy=tax)
            if activate:
                versions_mod.set_active(self.ctx.versions_dir, version)
        except (ValueError, OSError, TaxonomyError) as exc:
            self.message.setText(f"The version could not be saved: {exc}")
            return None
        self.message.setText(f"Saved as {version}" + (" and made active: the next run uses it." if activate else "."))
        self.ctx.prompt_changed.emit()
        return version

    # ------------------------------------------------------------------ versions
    def _versions_page(self) -> QWidget:
        self.version_list = QListWidget()
        self.version_list.setMaximumWidth(300)
        self.activate_btn = QPushButton("Make this version active")
        self.diff_active_btn = QPushButton("Compare with the active version")
        self.diff_draft_btn = QPushButton("Compare with my draft")
        self.restore_btn = QPushButton("Copy into my draft")
        self.version_diff = PartsDiff()
        left = QWidget()
        lc = QVBoxLayout(left)
        lc.setContentsMargins(0, 0, 0, 0)
        lc.addWidget(self.version_list, 1)
        for b in (self.activate_btn, self.diff_active_btn, self.diff_draft_btn, self.restore_btn):
            lc.addWidget(b)
        page = QWidget()
        row = QHBoxLayout(page)
        row.addWidget(left)
        row.addWidget(self.version_diff, 1)
        self.activate_btn.clicked.connect(self.activate_selected)
        self.diff_active_btn.clicked.connect(lambda: self.compare_selected("active"))
        self.diff_draft_btn.clicked.connect(lambda: self.compare_selected("draft"))
        self.restore_btn.clicked.connect(self.restore_selected)
        self.version_list.currentItemChanged.connect(lambda *_: self._update_version_buttons())
        return page

    def _fill_versions(self) -> None:
        active = versions_mod.active_version(self.ctx.versions_dir)
        self.version_list.clear()
        for info in reversed(versions_mod.list_versions(self.ctx.versions_dir)):
            label = f"{info.version}{'  (ACTIVE)' if info.version == active else ''}\n{info.created[:16].replace('T', ' ')}  {info.note}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, info.version)
            self.version_list.addItem(item)
        if self.version_list.count():
            self.version_list.setCurrentRow(0)
        self._update_version_buttons()

    def _selected_version(self) -> str:
        item = self.version_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else ""

    def _update_version_buttons(self) -> None:
        version = self._selected_version()
        active = versions_mod.active_version(self.ctx.versions_dir)
        self.activate_btn.setEnabled(bool(version) and version != active)
        self.diff_active_btn.setEnabled(bool(version) and bool(active) and version != active)
        self.diff_draft_btn.setEnabled(bool(version))
        self.restore_btn.setEnabled(bool(version))

    def activate_selected(self) -> None:
        version = self._selected_version()
        if version:
            versions_mod.set_active(self.ctx.versions_dir, version)
            self.message.setText(f"{version} is now the active version: the next run uses it.")
            self.ctx.prompt_changed.emit()

    def compare_selected(self, against: str) -> None:
        version = self._selected_version()
        if not version:
            return
        other = versions_mod.active_version(self.ctx.versions_dir) if against == "active" else "draft"
        if not other:
            self.message.setText("There is no active version to compare with.")
            return
        pairs = self._version_pairs(version, other)
        self.version_diff.set_pairs(pairs, version, "my draft" if other == "draft" else other)
        self.message.setText(f"{len(pairs)} part(s) differ between {version} and {'my draft' if other == 'draft' else other}.")

    def _version_pairs(self, a: str, b: str) -> list[tuple[str, str, str]]:
        def parts(which: str) -> dict[str, str]:
            return read_parts(self.ctx.parts_dir if which == "draft" else versions_mod.get_version_dir(self.ctx.versions_dir, which))
        left, right = parts(a), parts(b)
        return [(n, left.get(n, ""), right.get(n, "")) for n in sorted(set(left) | set(right)) if left.get(n, "") != right.get(n, "")]

    def restore_selected(self) -> None:
        version = self._selected_version()
        if not version:
            return
        if QMessageBox.question(self, "Copy into draft", f"Replace your draft prompt parts with those of {version}? "
                                "Your current draft is not saved anywhere else.") != QMessageBox.StandardButton.Yes:
            return
        self.copy_version_to_draft(version)

    def copy_version_to_draft(self, version: str) -> None:
        source = versions_mod.get_version_dir(self.ctx.versions_dir, version)
        for old in self.ctx.parts_dir.glob("*.md"):
            old.unlink()
        for part in source.glob("*.md"):
            shutil.copy2(part, self.ctx.parts_dir / part.name)
        self._unsaved.clear()
        self.message.setText(f"Your draft now matches {version}.")
        self.ctx.prompt_changed.emit()

    # ------------------------------------------------------------------ generate
    def _generate_page(self) -> QWidget:
        intro = QLabel("When you add classes or attributes, the AI you chose in AI settings drafts an updated prompt from the current "
                       "prompt, the old and new taxonomy, your past corrections and a few approved examples. Nothing changes until you "
                       "approve the result.")
        intro.setWordWrap(True)
        self.use_corrections = QCheckBox("Include the corrections I ticked on the Corrections tab")
        self.use_corrections.setChecked(True)
        self.example_count = QSpinBox()
        self.example_count.setRange(0, 10)
        self.example_count.setValue(3)
        self.generate_btn = QPushButton("Generate new prompt")
        self.approve_prop_btn, self.reject_prop_btn = QPushButton("Approve: save as a new version and make it active"), QPushButton("Reject")
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.prop_diff = PartsDiff()
        page = QWidget()
        col = QVBoxLayout(page)
        col.addWidget(intro)
        opts = QHBoxLayout()
        for w in (self.use_corrections, QLabel("Approved examples to include:"), self.example_count, self.generate_btn):
            opts.addWidget(w)
        opts.addStretch(1)
        col.addLayout(opts)
        col.addWidget(self.summary)
        col.addWidget(self.prop_diff, 1)
        btns = QHBoxLayout()
        btns.addWidget(self.approve_prop_btn)
        btns.addWidget(self.reject_prop_btn)
        btns.addStretch(1)
        col.addLayout(btns)
        self.approve_prop_btn.setEnabled(False)
        self.reject_prop_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.generate)
        self.approve_prop_btn.clicked.connect(self.approve_proposal)
        self.reject_prop_btn.clicked.connect(self.reject_proposal)
        return page

    def generate(self) -> bool:
        if self._unsaved and not self.save_draft():
            return False
        try:
            provider = self._provider_fn()
            taxonomy = self.ctx.service().taxonomy()
        except (MissingCredentialError, CredentialError, ValueError, TaxonomyError) as exc:
            self.message.setText(f"Cannot generate: {exc}")
            return False
        settings = self.ctx.settings()
        corrections = [c for c in load_corrections(self.ctx.data_dir / "corrections.jsonl") if c.get("included")] \
            if self.use_corrections.isChecked() else []
        examples = self.ctx.service().approved_sets()[:self.example_count.value()]
        old = old_taxonomy_for_active(self.ctx.versions_dir)
        guard = TrackerGuard(self.ctx.tracker())
        self.generate_btn.setEnabled(False)
        self.message.setText("Asking the model to draft an updated prompt...")

        def job() -> Proposal:
            return generate_new_prompt(provider, self.ctx.parts_dir, old, taxonomy, corrections=corrections, examples=examples,
                                       placeholder=settings["labeling"]["color_placeholder"], guard=guard)

        self._runner = start_call(job, (), self._generated, self._generate_failed, self)
        return True

    def _generated(self, proposal: Proposal) -> None:
        self.generate_btn.setEnabled(True)
        self.show_proposal(proposal)

    def show_proposal(self, proposal: Proposal) -> None:
        self.proposal = proposal
        pairs = proposal_pairs(self.ctx.parts_dir, proposal)
        self.summary.setText(("<b>What the model says it changed:</b> " + proposal.summary) if proposal.summary else "")
        self.prop_diff.set_pairs(pairs, "Current draft", "Proposed")
        self.approve_prop_btn.setEnabled(bool(pairs))
        self.reject_prop_btn.setEnabled(True)
        self.message.setText(f"The model proposes changes to {len(pairs)} part(s). Read the differences, then approve or reject."
                             if pairs else "The model proposed no changes.")

    def _generate_failed(self, message: str) -> None:
        self.generate_btn.setEnabled(True)
        self.message.setText(f"The prompt could not be generated: {message}")

    def approve_proposal(self) -> str | None:
        if self.proposal is None:
            return None
        try:
            version = apply_proposal(self.proposal, self.ctx.parts_dir, self.ctx.versions_dir, self.ctx.service().taxonomy())
        except (GenerateError, OSError, ValueError) as exc:
            self.message.setText(f"The proposal could not be applied: {exc}")
            return None
        self.proposal = None
        self.approve_prop_btn.setEnabled(False)
        self.reject_prop_btn.setEnabled(False)
        self.prop_diff.set_pairs([])
        self.message.setText(f"Saved as {version} and made active. Older versions are still in the Versions list.")
        self.ctx.prompt_changed.emit()
        return version

    def reject_proposal(self) -> None:
        self.proposal = None
        self.approve_prop_btn.setEnabled(False)
        self.reject_prop_btn.setEnabled(False)
        self.prop_diff.set_pairs([])
        self.summary.clear()
        self.message.setText("Proposal discarded. Nothing was changed.")

    # ------------------------------------------------------------------ taxonomy
    def _taxonomy_page(self) -> QWidget:
        self.tax_file = QComboBox()
        self.tax_file.addItems([f"{n}.yaml" for n in TAXONOMY_FILES])
        self.tax_editor = QPlainTextEdit()
        self.tax_editor.setFont(MONO)
        self.tax_save_btn, self.tax_reload_btn = QPushButton("Check and save"), QPushButton("Reload from disk")
        self.tax_message = QLabel()
        self.tax_message.setWordWrap(True)
        hint = QLabel("Every value is lowercase letters and digits only (no spaces or dashes), because class names are built from them. "
                      "Laser events may use a single dash. Removing a value does not change labels that already use it.")
        hint.setWordWrap(True)
        page = QWidget()
        col = QVBoxLayout(page)
        col.addWidget(hint)
        col.addWidget(self.tax_file)
        col.addWidget(self.tax_editor, 1)
        row = QHBoxLayout()
        row.addWidget(self.tax_save_btn)
        row.addWidget(self.tax_reload_btn)
        row.addStretch(1)
        col.addLayout(row)
        col.addWidget(self.tax_message)
        self.tax_file.currentIndexChanged.connect(lambda _i: self.load_taxonomy_file())
        self.tax_save_btn.clicked.connect(self.save_taxonomy_file)
        self.tax_reload_btn.clicked.connect(self.load_taxonomy_file)
        return page

    def load_taxonomy_file(self) -> None:
        path = self.ctx.taxonomy_dir / self.tax_file.currentText()
        self.tax_editor.setPlainText(path.read_text(encoding="utf-8") if path.is_file() else "")
        self.tax_message.clear()

    def save_taxonomy_file(self) -> bool:
        """Check the whole taxonomy with this file replaced; write it only if everything is valid."""
        name, text = self.tax_file.currentText(), self.tax_editor.toPlainText()
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "taxonomy"
            shutil.copytree(self.ctx.taxonomy_dir, copy)
            (copy / name).write_text(text, encoding="utf-8")
            try:
                load_taxonomy(copy)
            except TaxonomyError as exc:
                self.tax_message.setText("Not saved. Problems:\n- " + "\n- ".join(exc.problems))
                return False
        (self.ctx.taxonomy_dir / name).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        self.tax_message.setText(f"{name} saved. Use Generate new prompt so the model learns the change.")
        self.ctx.settings_changed.emit()
        self._refresh_banner()
        return True

    # ------------------------------------------------------------------ corrections
    def _corrections_page(self) -> QWidget:
        self.correction_list = QListWidget()
        self.apply_corrections_btn = QPushButton("Put the ticked corrections in the project background (part 07)")
        hint = QLabel("Every fix you make in review is kept here. Tick the ones that show a mistake the model keeps making. They are written "
                      "into a marked block of 07_background.md and are also sent to Generate new prompt.")
        hint.setWordWrap(True)
        page = QWidget()
        col = QVBoxLayout(page)
        col.addWidget(hint)
        col.addWidget(self.correction_list, 1)
        col.addWidget(self.apply_corrections_btn)
        self.apply_corrections_btn.clicked.connect(self.apply_corrections)
        return page

    def _fill_corrections(self) -> None:
        self.correction_list.clear()
        for item in reversed(load_corrections(self.ctx.data_dir / "corrections.jsonl")):
            row = QListWidgetItem(f"{item['text']}   [{item.get('image_name', '')}]")
            row.setData(Qt.ItemDataRole.UserRole, item["id"])
            row.setFlags(row.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            row.setCheckState(Qt.CheckState.Checked if item.get("included") else Qt.CheckState.Unchecked)
            self.correction_list.addItem(row)

    def apply_corrections(self) -> int:
        path = self.ctx.data_dir / "corrections.jsonl"
        everything = load_corrections(path)
        ticked = {self.correction_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.correction_list.count())
                  if self.correction_list.item(i).checkState() == Qt.CheckState.Checked}
        set_included(path, [c["id"] for c in everything if c["id"] in ticked], True)
        set_included(path, [c["id"] for c in everything if c["id"] not in ticked], False)
        chosen = [c for c in load_corrections(path) if c["id"] in ticked]
        changed = apply_to_background(self.ctx.parts_dir, chosen)
        self.message.setText((f"{len(chosen)} correction(s) written to 07_background.md in your draft. Save it as a new version to use it."
                              if changed else "The background part already had exactly these corrections."))
        self._unsaved.pop("07_background.md", None)
        self._fill_parts()
        self._refresh_banner()
        return len(chosen)

    # ------------------------------------------------------------------ shared
    def _refresh_banner(self) -> None:
        notes = []
        active = versions_mod.active_version(self.ctx.versions_dir)
        notes.append(f"Runs use prompt version <b>{active}</b>." if active else
                     "No version is saved yet: runs use your draft (shown as draft-xxxx on each label). Save a version to track changes.")
        if self._unsaved or versions_mod.draft_differs_from_active(self.ctx.parts_dir, self.ctx.versions_dir) and active:
            notes.append("Your draft has changes that are not in the active version.")
        try:
            snap = old_taxonomy_for_active(self.ctx.versions_dir)
            current = self.ctx.service().taxonomy().to_dict()
            if active and snap is not None and snap != current:
                notes.append("<span style='color:#bc4c00'>The taxonomy has changed since this version was saved: use Generate new prompt.</span>")
        except TaxonomyError:
            pass
        self.banner.setText(" ".join(notes))

    def refresh(self) -> None:
        self._fill_parts()
        self._fill_versions()
        self._fill_corrections()
        self.load_taxonomy_file()
        self._refresh_banner()
