import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from config import credentials  # noqa: E402
from core.keep_awake import KeepAwake  # noqa: E402
from core.models import Question  # noqa: E402
from core.prompts.diffview import align_diff, changed_count  # noqa: E402
from gui.context import AppContext  # noqa: E402
from gui.main_window import MainWindow, provider_status_text, spend_status_text  # noqa: E402
from gui.widgets.conversation_viewer import ConversationViewer  # noqa: E402
from gui.widgets.diff_view import PartsDiff, SideBySideDiff  # noqa: E402
from gui.widgets.question_dialog import QuestionDialog  # noqa: E402
from gui.widgets.thumb_grid import GridItem, ThumbnailGrid  # noqa: E402
from gui.workers.progress_worker import ProgressWorker  # noqa: E402
from tests.gui_support import make_images, make_project, wait_until  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ctx(app, tmp_path):
    context = AppContext(tmp_path / "user.yaml", make_project(tmp_path), tmp_path / "data")
    yield context
    context.close()


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    monkeypatch.setattr(credentials, "get_key", lambda service: None)


# ---------------------------------------------------------------- main window
def test_full_window_has_the_six_tabs_in_order(app, ctx, tmp_path):
    win = MainWindow(user_settings_path=ctx.settings_path, ctx=ctx)
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == [
        "Sign in", "Select images", "Label run", "Review queue", "Prompts and taxonomy", "AI settings"]
    assert win.statusBar().currentMessage() == "Roboflow: not signed in"
    assert win.prompt_label.text().startswith("Prompt: draft") and win.provider_label.text().startswith("AI: ")
    assert win.spend_label.text().startswith("Spent today: $0.00")
    win.close()


def test_status_bar_follows_settings_and_prompt_changes(app, ctx):
    win = MainWindow(user_settings_path=ctx.settings_path, ctx=ctx)
    ctx.update_settings(lambda s: s["provider"].update(model="my-model", daily_cost_cap_usd=2.0))
    assert "my-model" in win.provider_label.text() and "of $2.00" in win.spend_label.text()
    from core.prompts import versions
    version = versions.save_version(ctx.parts_dir, ctx.versions_dir, "x")
    versions.set_active(ctx.versions_dir, version)
    ctx.prompt_changed.emit()
    assert version in win.prompt_label.text()
    win.close()


def test_status_text_helpers():
    settings = {"provider": {"provider": "gemini", "model": "", "daily_cost_cap_usd": None}}
    assert provider_status_text(settings) == "AI: gemini / no model chosen"
    assert spend_status_text(1.234, settings) == "Spent today: $1.23"


def test_ai_settings_link_switches_to_sign_in_and_open_image_routes_to_review(app, ctx):
    win = MainWindow(user_settings_path=ctx.settings_path, ctx=ctx)
    win.show_tab("AI settings")
    ai = win.tabs.currentWidget()
    ai.go_sign_in.emit()
    assert win.tabs.tabText(win.tabs.currentIndex()) == "Sign in"
    win.show_tab("Label run")
    ctx.open_image.emit("ghost")  # unknown image: shown nowhere, but the Review tab is brought up
    assert win.tabs.tabText(win.tabs.currentIndex()) == "Review queue"
    win.close()


def test_closing_the_window_closes_the_database(app, ctx):
    win = MainWindow(user_settings_path=ctx.settings_path, ctx=ctx)
    assert ctx.db is not None and ctx._db is not None
    win.close()
    assert ctx._db is None


def test_screenshots_render_every_tab(app, ctx, tmp_path):
    win = MainWindow(user_settings_path=ctx.settings_path, ctx=ctx)
    win.show()
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        app.processEvents()
        out = tmp_path / f"tab{i}.png"
        assert win.grab().save(str(out)) and out.stat().st_size > 2000
    win.close()


# ---------------------------------------------------------------- context
def test_context_refuses_to_store_a_secret(app, ctx):
    from config.settings import SettingsError
    with pytest.raises(SettingsError):
        ctx.update_settings(lambda s: s["provider"].update(api_key="sk-abcdefghijklmnop1234"))
    assert "sk-abc" not in ctx.settings_path.read_text() if ctx.settings_path.exists() else True


def test_context_provider_needs_a_key_for_cloud_providers(app, ctx):
    ctx.update_settings(lambda s: s["provider"].update(provider="gemini", model="m"))
    with pytest.raises(credentials.MissingCredentialError):
        ctx.provider()


def test_context_estimate_and_prompt_text(app, ctx):
    assert ctx.estimate(10).describe()
    assert ctx.prompt_version_text().startswith("draft")


# ---------------------------------------------------------------- widgets
def test_question_dialog_answers_and_taxonomy_checkbox(app):
    q = Question("q1", "What material is the pen?", "o1", "material", ["glass", "metal"], True)
    d = QuestionDialog(q)
    d.show()
    assert d.answer() == "glass" and d.add_box.isVisible() and not d.add_to_taxonomy()
    d.choice.setCurrentText("  velvet ")
    d.add_box.setChecked(True)
    assert d.answer() == "velvet" and d.add_to_taxonomy()
    plain = QuestionDialog(Question("q2", "Where?", None, "hit_point", [], False))
    plain.show()
    assert plain.answer() == "" and not plain.add_box.isVisible()


def test_conversation_viewer_shows_messages_and_events(app):
    v = ConversationViewer()
    v.append_entry({"type": "message", "role": "user", "text": "Label this <image>", "time": "2026-01-01T10:00:00"})
    v.append_entry({"type": "message", "role": "assistant", "text": "", "tool_calls": [{"name": "crop", "arguments": {"x": 1}}],
                    "images": ["crop1.png"]})
    v.append_entry({"type": "reviewer_note", "text": "it is a gel pen"})
    text = v.toPlainText()
    assert "Label this <image>" in text and "crop" in text and "[image crop1.png]" in text and "gel pen" in text
    assert not v.load(None) and not v.load("/nonexistent.json")


def test_align_diff_rows():
    rows = align_diff("a\nb\nc", "a\nB\nc\nd")
    assert [r[2] for r in rows] == ["equal", "replace", "equal", "insert"] and changed_count(rows) == 2
    assert changed_count(align_diff("same", "same")) == 0
    assert align_diff("x\ny", "x") == [("x", "x", "equal"), ("y", "", "delete")]


def test_side_by_side_and_parts_diff(app):
    d = SideBySideDiff()
    assert d.set_texts("a\nb", "a\nc") == 1 and "1 line" in d.summary.text()
    assert d.set_texts("a", "a") == 0 and "No differences" in d.summary.text()
    p = PartsDiff()
    p.set_pairs([("01.md", "x", "y"), ("02.md", "same?", "different")])
    p.set_pairs([])


def test_thumbnail_grid_ticks_and_lazy_icons(app, tmp_path):
    paths = make_images(tmp_path / "t", ("a.png", "b.png", "c.png"))
    grid = ThumbnailGrid()
    grid.resize(500, 300)
    grid.show()
    grid.set_items([GridItem(p.stem, p.name, str(p), p.name, checked=(p.stem == "b")) for p in paths] + [GridItem("gone", "gone.png", None, "")])
    assert grid.keys() == ["a", "b", "c", "gone"] and grid.checked_keys() == ["b"]
    seen = []
    grid.checks_changed.connect(lambda: seen.append(1))
    grid.set_checked(["a", "c"], True)
    assert sorted(grid.checked_keys()) == ["a", "b", "c"] and seen
    assert wait_until(app, lambda: not grid.loading(), 10)
    grid.select_key("c")
    assert grid.current_key_value() == "c"


# ---------------------------------------------------------------- workers and helpers
def test_progress_worker_reports_result_progress_and_errors(app):
    results, progress, errors = [], [], []
    w = ProgressWorker(lambda report, stop: (report(1, 2), report(2, 2), "done")[2])
    w.progress.connect(lambda a, b: progress.append((a, b)))
    w.succeeded.connect(results.append)
    w.start()
    assert wait_until(app, lambda: results) and progress == [(1, 2), (2, 2)] and results == ["done"]
    w.wait()

    def boom(report, stop):
        raise RuntimeError("no network")

    bad = ProgressWorker(boom)
    bad.failed.connect(errors.append)
    bad.start()
    assert wait_until(app, lambda: errors) and errors == ["no network"]
    bad.wait()


def test_progress_worker_can_be_cancelled(app):
    out = []

    def loop(report, stop):
        n = 0
        while not stop() and n < 500:
            n += 1
            QApplication.instance().thread().msleep(5)
        return n

    w = ProgressWorker(loop)
    w.succeeded.connect(out.append)
    w.start()
    w.cancel()
    assert wait_until(app, lambda: out) and out[0] < 500
    w.wait()


def test_keep_awake_never_raises():
    with KeepAwake() as awake:  # may do nothing on this machine; it must not fail
        assert awake is not None
    KeepAwake().stop()
