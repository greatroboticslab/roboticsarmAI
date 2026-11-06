import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from config import credentials  # noqa: E402
from core.providers.base import ProviderReply, Usage  # noqa: E402
from gui.context import AppContext  # noqa: E402
from gui.tabs.ai_settings_tab import AISettingsTab  # noqa: E402
from gui.tabs.label_run_tab import LabelRunTab  # noqa: E402
from gui.tabs.select_images_tab import SelectImagesTab  # noqa: E402
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text  # noqa: E402
from tests.gui_support import make_images, make_project, wait_until  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ctx(app, tmp_path):
    context = AppContext(tmp_path / "user.yaml", make_project(tmp_path), tmp_path / "data")
    yield context
    context.close()


def good():
    return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0, "from_id": None,
                                "object": PEN_RUBBER}]))


# ---------------------------------------------------------------- AI settings
def test_ai_settings_save_and_reload(app, ctx):
    tab = AISettingsTab(ctx)
    tab.model.setCurrentText("some-model") if hasattr(tab.model, "setCurrentText") else tab.model.setText("some-model")
    assert tab.save()
    assert ctx.settings()["provider"]["model"] == "some-model"
    assert AISettingsTab(ctx).provider_block()["model"] == "some-model"


def test_ai_settings_never_writes_a_key(app, ctx, monkeypatch):
    monkeypatch.setattr(credentials, "get_key", lambda service: "sk-SECRET-VALUE-123456")
    tab = AISettingsTab(ctx)
    tab.save()
    assert "SECRET" not in ctx.settings_path.read_text()
    assert "SECRET" not in "".join(w.text() for w in tab.findChildren(type(tab.message)) if hasattr(w, "text"))


# ---------------------------------------------------------------- select images
def test_select_scan_filter_tick_and_add(app, ctx, tmp_path):
    make_images(tmp_path / "pics", ("a.png", "b.png", "c.png"))
    tab = SelectImagesTab(ctx)
    assert tab.scan_paths([tmp_path / "pics"]) == 3
    assert "3 shown" in tab.count_label.text() and not tab.add_btn.isEnabled()
    tab.search.setText("b")
    assert [e.name for e in tab._shown] == ["b.png"]
    tab.search.setText("")
    tab.chunk.setValue(2)
    tab.tick_chunk()
    assert len(tab.ticked) == 2 and tab.add_btn.isEnabled() and "ticked" in tab.count_label.text()
    tab.tick_chunk()  # the next chunk continues with what is not ticked yet
    assert len(tab.ticked) == 3
    tab.untick_btn.click()
    assert not tab.ticked
    tab.tick_all_btn.click()
    tab.tag_edit.setText("camera:a, day:1")
    assert tab.add_to_queue() == 3
    assert ctx.db.counts()["queued"] == 3
    assert ctx.db.get("a").tags == ["camera:a", "day:1"]
    assert tab._shown == []  # already queued images are hidden by default
    tab.hide_queued.setChecked(False)
    assert len(tab._shown) == 3


def test_select_empty_folder_and_nothing_ticked(app, ctx, tmp_path):
    (tmp_path / "empty").mkdir()
    tab = SelectImagesTab(ctx)
    assert tab.scan_paths([tmp_path / "empty"]) == 0 and "No image files" in tab.message.text()
    assert tab.add_to_queue() == 0


def test_select_remote_unknown_state_is_asked(app, ctx, tmp_path):
    from core.roboflow.images import RemoteImage
    images = [RemoteImage("r1", "r1.jpg", labels=[]), RemoteImage("r2", "r2.jpg", labels=["x"]), RemoteImage("r3", "r3.jpg")]
    fetched = []

    def download(img, folder, key):
        fetched.append(img.id)
        path = folder / f"{img.id}.png"
        folder.mkdir(parents=True, exist_ok=True)
        make_images(folder, (path.name,))
        return path

    ctx.update_settings(lambda s: s["roboflow"].update(workspace="ws", project="pr"))
    tab = SelectImagesTab(ctx, list_fn=lambda ws, p, k: images, download_fn=download, key_fn=lambda: "rf-key")
    tab.remote_radio.setChecked(True)
    tab.load_remote()
    assert wait_until(app, lambda: len(tab.entries) == 3)
    tab.unlabeled_radio.setChecked(True)  # r1 (empty) and r3 (unknown) stay; r2 is labeled
    assert {e.key for e in tab._shown} == {"r1", "r3"}
    tab.tick_all_btn.click()
    asked = []
    tab.ask_unknown = lambda n: (asked.append(n), "skip")[1]
    tab.add_to_queue()
    assert asked == [1]
    assert wait_until(app, lambda: ctx.db.counts()["queued"] == 1)
    assert fetched == ["r1"] and ctx.db.get("r3") is None


def test_select_remote_without_project_or_key(app, ctx):
    tab = SelectImagesTab(ctx, list_fn=lambda *a: [], key_fn=lambda: credentials.require_key("roboflow"))
    tab.load_remote()
    assert "Sign in" in tab.message.text()


# ---------------------------------------------------------------- label run
def queue_images(ctx, tmp_path, n=2):
    tab = SelectImagesTab(ctx)
    names = [f"{chr(97 + i)}.png" for i in range(n)]
    make_images(tmp_path / "pics", names)
    tab.scan_paths([tmp_path / "pics"])
    tab.tick_all_btn.click()
    assert tab.add_to_queue() == n


def test_label_run_nothing_queued(app, ctx):
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([]), confirm_fn=lambda t: True)
    assert not run.start() and "Nothing is queued" in run.message.text()


def test_label_run_start_and_finish(app, ctx, tmp_path):
    queue_images(ctx, tmp_path, 2)
    provider = ScriptedProvider([good(), good()])
    asked = []
    run = LabelRunTab(ctx, provider_fn=lambda: provider, confirm_fn=lambda t: (asked.append(t), True)[1])
    changes = []
    ctx.queue_changed.connect(lambda: changes.append(1))
    assert run.start() and "fake/fake-model" in asked[0]
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["pending"] == 2 and changes
    assert "Finished 2" in run.message.text() and run.start_btn.isEnabled()
    assert run.viewer.toPlainText().strip()
    assert (ctx.data_dir / "sidecars" / "a.json").is_file()


def test_label_run_declined_and_missing_credentials(app, ctx, tmp_path):
    queue_images(ctx, tmp_path, 1)
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([]), confirm_fn=lambda t: False)
    assert not run.start() and not run.running() and ctx.db.counts()["queued"] == 1

    def refuse():
        raise credentials.MissingCredentialError("No key for gemini. Run `python -m cli credentials set gemini`.")

    run = LabelRunTab(ctx, provider_fn=refuse, confirm_fn=lambda t: True)
    assert not run.start() and "No key for gemini" in run.message.text()


def test_label_run_stop_after_the_image_in_progress(app, ctx, tmp_path):
    queue_images(ctx, tmp_path, 3)
    run = LabelRunTab(ctx, provider_fn=lambda: None, confirm_fn=lambda t: True)

    def slow(messages):
        run.stop()  # asked for while the first image is being labeled
        return good()

    provider = ScriptedProvider([slow, good(), good()])
    run._provider_fn = lambda: provider
    assert run.start()
    assert wait_until(app, lambda: not run.running(), 20)
    counts = ctx.db.counts()
    assert counts["pending"] == 1 and counts["queued"] == 2
    assert "stopped" in run.message.text().lower()


def test_label_run_retry_failed(app, ctx, tmp_path):
    queue_images(ctx, tmp_path, 1)
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([]), confirm_fn=lambda t: True)
    assert run.start()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["failed"] == 1 and run.retry_btn.isEnabled()
    run.retry_btn.click()
    assert ctx.db.counts()["queued"] == 1


def test_ai_settings_test_provider_reports(app, ctx):
    from tests.fakes import tool
    tab = AISettingsTab(ctx)
    ctx.provider = lambda overrides=None: ScriptedProvider([text('{"dot": true}')] * 3 + [tool("get_image_info")])
    tab.test_provider()
    assert wait_until(app, lambda: "valid JSON" in tab.report.toPlainText())
    assert "3/3" in tab.report.toPlainText() and tab.test_btn.isEnabled() and tab.message.text() == "Test finished."


def test_ai_settings_test_provider_without_a_key_explains(app, ctx):
    tab = AISettingsTab(ctx)

    def refuse(overrides=None):
        raise credentials.MissingCredentialError("No key for gemini. Run `python -m cli credentials set gemini`.")

    ctx.provider = refuse
    tab.test_provider()
    assert "Cannot test yet" in tab.message.text() and "No key for gemini" in tab.message.text() and tab.test_btn.isEnabled()


def test_ai_settings_escalation_needs_a_model_name(app, ctx):
    tab = AISettingsTab(ctx)
    tab.esc_on.setChecked(True)
    tab.esc_model.setText("")
    assert not tab.save() and "no cloud model" in tab.message.text()
    assert ctx.settings()["escalation"]["enabled"] is False  # nothing half-saved


def test_ai_settings_label_scheme_is_saved_and_reloaded(app, ctx):
    tab = AISettingsTab(ctx)
    assert tab.scheme.currentData() == "full"
    tab.scheme.setCurrentIndex(tab.scheme.findData("components"))
    assert tab.save() and ctx.settings()["labeling"]["scheme"] == "components"
    assert AISettingsTab(ctx).scheme.currentData() == "components"
    assert "component1" in tab.scheme_help.text() and "halo" in tab.scheme_help.text()


def test_label_run_shows_why_an_image_failed_and_what_to_try(app, ctx, tmp_path):
    from core.providers.base import ProviderError
    queue_images(ctx, tmp_path, 2)

    def boom(messages):
        raise ProviderError("NotFoundError: Error code: 404 - model 'llama3:latest' not found")

    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([boom, boom]), confirm_fn=lambda t: True)
    assert run.start()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["failed"] == 2
    shown = run.failure.text()
    assert "Last failure" in shown and "404" in shown and "named exactly" in shown
    run.retry_btn.click()
    assert run.failure.text() == ""


def test_review_failed_list_shows_the_error(app, ctx, tmp_path):
    from core.providers.base import ProviderError
    from gui.tabs.review_tab import ReviewTab
    queue_images(ctx, tmp_path, 1)

    def boom(messages):
        raise ProviderError("APIConnectionError: Connection error.")

    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([boom]), confirm_fn=lambda t: True)
    run.start()
    assert wait_until(app, lambda: not run.running(), 20)
    review = ReviewTab(ctx)
    review.filter_box.setCurrentText("Failed")
    assert "Connection error" in review.list.item(0).text()
    review.select_image("a")
    assert "could not reach the server" in review.message.text()


def test_a_pasted_model_command_is_tidied_when_saved_and_used(app, ctx):
    tab = AISettingsTab(ctx)
    tab.model.setText('  ollama run qwen2.5vl:7b\n')
    assert tab.save() and ctx.settings()["provider"]["model"] == "qwen2.5vl:7b"
    ctx.update_settings(lambda s: s["provider"].update(model='"qwen2.5vl:7b" '))  # hand-edited settings file
    assert ctx.provider().model == "qwen2.5vl:7b"


def test_label_run_says_what_each_waiting_image_needs(app, ctx, tmp_path):
    from PIL import Image
    from tests.synth import finish, scene
    pics = tmp_path / "pics"
    pics.mkdir()
    for name in ("x.png", "y.png"):
        Image.fromarray(finish(scene())).save(pics / name)  # no laser in either
    tab = SelectImagesTab(ctx)
    tab.scan_paths([pics])
    tab.tick_all_btn.click()
    assert tab.add_to_queue() == 2
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([]), confirm_fn=lambda t: True)
    assert run.start()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["needs_input"] == 2
    assert "no laser found" in run.notice.text() and "Click where the beam first hits" in run.notice.text()
    assert "Waiting for you: 2 x no laser found" in run.message.text()
    assert "Needs input" in run.message.text() or "Review queue" in run.message.text()


def test_ai_settings_laser_box_size_saves_and_reloads(app, ctx):
    tab = AISettingsTab(ctx)
    assert tab.box_size.value() == 0.33
    tab.box_size.setValue(0.4)
    assert tab.save()
    assert ctx.settings()["labeling"]["laser_box_object_fraction"] == 0.4
    assert AISettingsTab(ctx).box_size.value() == 0.4


def test_ai_settings_offers_the_merged_scheme(app, ctx):
    tab = AISettingsTab(ctx)
    assert tab.scheme.findData("merged") >= 0
    tab.scheme.setCurrentIndex(tab.scheme.findData("merged"))
    assert tab.save()
    assert ctx.settings()["labeling"]["scheme"] == "merged"
    assert AISettingsTab(ctx).scheme.currentData() == "merged"
