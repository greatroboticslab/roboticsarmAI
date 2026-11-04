import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from core.prompts import versions as v  # noqa: E402
from core.prompts.examples import START, load_corrections, record_corrections  # noqa: E402
from core.review.diffs import Change  # noqa: E402
from gui.context import AppContext  # noqa: E402
from gui.tabs.prompts_tab import PromptsTab  # noqa: E402
from tests.fakes import ScriptedProvider, text  # noqa: E402
from tests.gui_support import make_project, wait_until  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ctx(app, tmp_path):
    context = AppContext(tmp_path / "user.yaml", make_project(tmp_path), tmp_path / "data")
    yield context
    context.close()


@pytest.fixture
def tab(app, ctx):
    t = PromptsTab(ctx, provider_fn=lambda: ScriptedProvider([]))
    t.resize(1100, 750)
    t.show()
    app.processEvents()
    yield t
    t.close()


def edit(tab, name, new_text):
    for i in range(tab.part_list.count()):
        if tab.part_list.item(i).data(Qt.ItemDataRole.UserRole) == name:  
            tab.part_list.setCurrentRow(i)
    assert tab._current_part == name
    tab.editor.setPlainText(new_text)


# ---------------------------------------------------------------- parts and draft
def test_parts_are_listed_and_edits_are_marked(tab, ctx):
    names = [tab.part_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(tab.part_list.count())]
    assert "01_role_task.md" in names and "07_background.md" in names
    edit(tab, "07_background.md", "## Project background\n\nLasers on velvets.\n")
    assert "07_background.md" in tab._unsaved and "*" in tab.part_list.currentItem().text()
    assert "not in the active version" in tab.banner.text() or "draft has changes" in tab.banner.text()
    assert "Lasers" not in (ctx.parts_dir / "07_background.md").read_text()  # nothing written yet
    assert tab.save_draft() == 1
    assert "Lasers on velvets" in (ctx.parts_dir / "07_background.md").read_text() and not tab._unsaved


def test_undoing_an_edit_clears_the_mark(tab, ctx):
    original = (ctx.parts_dir / "07_background.md").read_text()
    edit(tab, "07_background.md", "changed")
    assert tab._unsaved
    tab.editor.setPlainText(original)
    assert not tab._unsaved


def test_new_part_names_are_checked(tab, ctx):
    assert not tab.create_part("velvets.md") and "two digits" in tab.message.text()
    assert not tab.create_part("07_background.md") and "already exists" in tab.message.text()
    assert tab.create_part("08_velvets.md")
    tab.editor.setPlainText("Ceramic is glossy and white.\n")
    tab.save_draft()
    assert (ctx.parts_dir / "08_velvets.md").is_file()


# ---------------------------------------------------------------- versions
def test_save_version_makes_it_active_and_tracks_changes(tab, ctx, app):
    signals = []
    ctx.prompt_changed.connect(lambda: signals.append(1))
    assert tab.make_version("first") == "v0001"
    assert v.active_version(ctx.versions_dir) == "v0001" and signals
    assert "v0001" in tab.banner.text() and tab.version_list.count() == 1 and "ACTIVE" in tab.version_list.item(0).text()
    assert v.taxonomy_snapshot(ctx.versions_dir, "v0001")["materials"]
    edit(tab, "07_background.md", "## Project background\n\nmore\n")
    tab.save_draft()
    assert "not in the active version" in tab.banner.text()
    assert tab.make_version("second") == "v0002" and tab.version_list.count() == 2


def test_activate_compare_and_restore(tab, ctx):
    tab.make_version("first")
    edit(tab, "07_background.md", "## Project background\n\nversion two text\n")
    tab.save_draft()
    tab.make_version("second")
    tab.version_list.setCurrentRow(1)  # newest first: row 1 is v0001
    assert tab._selected_version() == "v0001" and tab.activate_btn.isEnabled()
    tab.compare_selected("active")
    assert "1 part(s) differ" in tab.message.text()
    tab.activate_selected()
    assert v.active_version(ctx.versions_dir) == "v0001"
    tab.version_list.setCurrentRow(0)
    tab.copy_version_to_draft("v0002")
    assert "version two text" in (ctx.parts_dir / "07_background.md").read_text()
    tab.copy_version_to_draft("v0001")
    assert "version two text" not in (ctx.parts_dir / "07_background.md").read_text()


def test_version_buttons_when_nothing_is_saved(tab):
    assert tab.version_list.count() == 0 and not tab.activate_btn.isEnabled() and not tab.restore_btn.isEnabled()
    assert "No version is saved" in tab.banner.text()


# ---------------------------------------------------------------- taxonomy
def test_invalid_taxonomy_is_refused_and_valid_is_saved(tab, ctx):
    tab.tax_file.setCurrentText("materials.yaml")
    original = (ctx.taxonomy_dir / "materials.yaml").read_text()
    tab.tax_editor.setPlainText("materials:\n  - glass\n  - Bad Value!\n")
    assert not tab.save_taxonomy_file()
    assert "Not saved" in tab.tax_message.text() and (ctx.taxonomy_dir / "materials.yaml").read_text() == original
    tab.tax_editor.setPlainText(original + "  - velvet\n")
    assert tab.save_taxonomy_file()
    assert "velvet" in ctx.service().taxonomy().materials
    tab.make_version("with velvet")
    tab.tax_editor.setPlainText(original + "  - velvet\n  - satin\n")
    assert tab.save_taxonomy_file()
    assert "taxonomy has changed" in tab.banner.text()


# ---------------------------------------------------------------- corrections
def test_ticked_corrections_go_into_the_background_part(tab, ctx):
    path = ctx.data_dir / "corrections.jsonl"
    record_corrections(path, "a", "a.png", [Change("attribute", "o1", "material", "rubber", "glass", "pen")], "v0001")
    record_corrections(path, "b", "b.png", [Change("attribute", "o1", "color", "red", "blue", "cup")], "v0001")
    tab.refresh()
    assert tab.correction_list.count() == 2
    tab.correction_list.item(0).setCheckState(Qt.CheckState.Checked)
    assert tab.apply_corrections() == 1
    background = (ctx.parts_dir / "07_background.md").read_text()
    assert START in background and "->" in background
    assert sum(1 for c in load_corrections(path) if c.get("included")) == 1
    tab.correction_list.item(0).setCheckState(Qt.CheckState.Unchecked)
    tab.apply_corrections()
    assert not any(c.get("included") for c in load_corrections(path))


# ---------------------------------------------------------------- generate new prompt
def parts_reply(ctx, extra):
    from core.prompts.generate import read_parts
    current = read_parts(ctx.parts_dir)
    return text({"summary": "added velvet guidance", "parts": {"07_background.md": current["07_background.md"] + extra}})


def test_generate_review_and_approve(app, ctx):
    tab = PromptsTab(ctx, provider_fn=lambda: ScriptedProvider([parts_reply(ctx, "\nCeramic is glossy.\n")]))
    tab.make_version("base")
    assert tab.generate()
    assert wait_until(app, lambda: tab.proposal is not None)
    assert "added velvet guidance" in tab.summary.text() and tab.approve_prop_btn.isEnabled()
    assert "Ceramic" not in (ctx.parts_dir / "07_background.md").read_text()  # nothing changes before approval
    assert v.active_version(ctx.versions_dir) == "v0001"
    assert tab.approve_proposal() == "v0002"
    assert v.active_version(ctx.versions_dir) == "v0002" and "Ceramic is glossy" in (ctx.parts_dir / "07_background.md").read_text()
    assert not tab.approve_prop_btn.isEnabled() and tab.proposal is None
    tab.close()


def test_generate_reject_changes_nothing(app, ctx):
    tab = PromptsTab(ctx, provider_fn=lambda: ScriptedProvider([parts_reply(ctx, "\nCeramic.\n")]))
    tab.generate()
    assert wait_until(app, lambda: tab.proposal is not None)
    tab.reject_prop_btn.click()
    assert tab.proposal is None and "Nothing was changed" in tab.message.text()
    assert "Ceramic" not in (ctx.parts_dir / "07_background.md").read_text() and v.list_versions(ctx.versions_dir) == []
    tab.close()


def test_generate_bad_reply_and_missing_key_are_explained(app, ctx):
    tab = PromptsTab(ctx, provider_fn=lambda: ScriptedProvider([text("not json"), text("still not"), text("nope"), text("no")]))
    tab.generate()
    assert wait_until(app, lambda: "could not be generated" in tab.message.text())
    assert tab.proposal is None and tab.generate_btn.isEnabled()

    from config.credentials import MissingCredentialError

    def refuse():
        raise MissingCredentialError("No key for gemini.")

    tab2 = PromptsTab(ctx, provider_fn=refuse)
    assert not tab2.generate() and "No key for gemini" in tab2.message.text()
    tab.close(), tab2.close()


def test_preview_builds_the_prompt_text(app, ctx, tab, monkeypatch):
    from PyQt6.QtWidgets import QDialog, QInputDialog
    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(lambda *a, **k: ("pipeline", True)))
    shown = []
    monkeypatch.setattr(QDialog, "exec", lambda self: shown.append(self.findChildren(type(tab.editor))[0].toPlainText()) or 0)
    tab.preview_prompt()
    assert shown and "{{taxonomy}}" not in shown[0] and "pen" in shown[0]
