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
    images = [RemoteImage("r1", "r1.jpg", labels=[]), RemoteImage("r2", "r2.jpg", labels=["las_first-incidence"]),
              RemoteImage("r3", "r3.jpg")]
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
    assert tab.unlabeled_radio.isChecked()  # a Roboflow load starts from the images that still need labels
    assert {e.key for e in tab._shown} == {"r1", "r3"}  # r1 (empty) and r3 (unknown) stay; r2 is labeled
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
    assert "direct hit" in tab.scheme_help.text() and "bounce" not in tab.scheme_help.text().lower().replace("every bounce", "")


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
    assert "marked failed and the run moved on to the next one" in shown
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


def limit_reply(retry_after=None):
    from core.providers.base import RateLimitError

    def reply(messages):
        raise RateLimitError("429 You exceeded your current quota", retry_after)
    return reply


def ask_instead_of_auto_resume(ctx):
    ctx.update_settings(lambda s: s["run"].update(auto_resume_on_limit=False))


def test_label_run_pauses_on_a_provider_limit_and_resumes_when_asked(app, ctx, tmp_path):
    ask_instead_of_auto_resume(ctx)
    queue_images(ctx, tmp_path, 2)
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([limit_reply(120), good(), good()]),
                      confirm_fn=lambda t: True)
    run.show()
    assert run.start()
    assert wait_until(app, lambda: run.limit_box.isVisible(), 20)
    # paused and asking: nothing failed, the image is back in the queue, the run is still alive
    assert run.running() and run.worker.waiting_for_limit
    assert ctx.db.counts()["failed"] == 0 and ctx.db.counts()["queued"] == 2
    assert "limit was reached" in run.limit_text.text() and "2 minutes" in run.limit_text.text()
    assert "Waiting for you" in run.message.text() and not run.pause_btn.isEnabled()
    run.resume_after_limit()
    assert not run.limit_box.isVisible()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["pending"] == 2 and ctx.db.counts()["failed"] == 0
    assert "Finished 2" in run.message.text()
    run.close()


def test_label_run_limit_then_stop_leaves_the_images_queued(app, ctx, tmp_path):
    ask_instead_of_auto_resume(ctx)
    queue_images(ctx, tmp_path, 2)
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([limit_reply()]), confirm_fn=lambda t: True)
    run.show()
    assert run.start()
    assert wait_until(app, lambda: run.limit_box.isVisible(), 20)
    run.limit_stop_btn.click()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["queued"] == 2 and ctx.db.counts()["failed"] == 0 and not run.limit_box.isVisible()
    assert "stopped" in run.message.text()
    run.close()


def test_window_close_ends_a_run_that_is_waiting_on_a_limit(app, ctx, tmp_path):
    from gui.main_window import MainWindow
    queue_images(ctx, tmp_path, 1)
    window = MainWindow(tab_factories=[("Label run", lambda: LabelRunTab(
        ctx, provider_fn=lambda: ScriptedProvider([limit_reply()]), confirm_fn=lambda t: True))])
    run = window.tabs.widget(0)
    assert run.start()
    assert wait_until(app, lambda: run.worker is not None and run.worker.waiting_for_limit, 20)
    window.close()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["queued"] == 1


def test_each_provider_remembers_its_own_model_and_prices(app, ctx):
    tab = AISettingsTab(ctx)
    tab._pick(tab.provider, "gemini")
    tab.model.setText("gemini-flash-lite-x")
    tab.price_in.setValue(0.3)
    tab.price_out.setValue(2.5)
    tab._pick(tab.provider, "openai_compat")
    assert tab.model.text() == "" and tab.price_in.value() == 0  # nothing remembered yet: no Gemini prices carried over
    tab.model.setText("qwen3-vl:8b")
    tab._pick(tab.provider, "anthropic")
    assert tab.model.text() == ""
    tab.model.setText("claude-x")
    tab._pick(tab.provider, "gemini")
    assert tab.model.text() == "gemini-flash-lite-x" and tab.price_in.value() == 0.3 and tab.price_out.value() == 2.5
    tab._pick(tab.provider, "openai_compat")
    assert tab.model.text() == "qwen3-vl:8b"
    assert tab.save()
    saved = ctx.settings()["provider"]
    assert saved["provider"] == "openai_compat" and saved["model"] == "qwen3-vl:8b"  # the active one is the main setting
    assert saved["profiles"]["gemini"]["model"] == "gemini-flash-lite-x"
    assert saved["profiles"]["anthropic"]["model"] == "claude-x"
    fresh = AISettingsTab(ctx)  # a restart
    fresh._pick(fresh.provider, "gemini")
    assert fresh.model.text() == "gemini-flash-lite-x" and fresh.price_out.value() == 2.5
    fresh._pick(fresh.provider, "openai_compat")
    assert fresh.model.text() == "qwen3-vl:8b"


def test_an_existing_model_setting_is_kept_for_its_provider(app, ctx):
    ctx.update_settings(lambda s: s["provider"].update(provider="gemini", model="old-model", profiles={}))
    tab = AISettingsTab(ctx)
    assert tab.model.text() == "old-model"
    tab._pick(tab.provider, "anthropic")
    assert tab.model.text() == ""
    tab._pick(tab.provider, "gemini")
    assert tab.model.text() == "old-model"


def test_the_main_model_still_wins_over_a_stale_profile(app, ctx):
    ctx.update_settings(lambda s: s["provider"].update(
        provider="gemini", model="new-model", profiles={"gemini": {"model": "stale-model"}}))
    assert AISettingsTab(ctx).model.text() == "new-model"


def test_bad_profiles_are_refused():
    from config.settings import SettingsError, load_settings, validate_settings
    for bad in ({"gemini": "text"}, {"mystery": {"model": "x"}}, ["x"], {"gemini": {"model": 5}}):
        s = load_settings()
        s["provider"]["profiles"] = bad
        with pytest.raises(SettingsError):
            validate_settings(s)


def test_ai_settings_speed_options_are_saved_and_validated(app, ctx):
    from config.settings import SettingsError, load_settings, validate_settings
    tab = AISettingsTab(ctx)
    tab.disable_thinking.setChecked(True)
    tab.timeout.setValue(240)
    tab.max_spots.setValue(3)
    tab.crop_px.setValue(256)
    assert tab.save()
    s = ctx.settings()
    assert s["provider"]["disable_thinking"] is True and s["provider"]["timeout_s"] == 240
    assert s["run"]["max_candidates"] == 3 and s["run"]["crop_px"] == 256
    again = AISettingsTab(ctx)
    assert again.max_spots.value() == 3 and again.crop_px.value() == 256 and again.disable_thinking.isChecked()
    bad = load_settings()
    bad["run"]["crop_px"] = 10
    with pytest.raises(SettingsError, match="crop_px"):
        validate_settings(bad)
    bad = load_settings()
    bad["provider"]["timeout_s"] = -5
    with pytest.raises(SettingsError, match="timeout_s"):
        validate_settings(bad)


def test_ai_settings_taxonomy_match_is_saved(app, ctx):
    tab = AISettingsTab(ctx)
    assert tab.match.currentData() == "object"  # the default: only the object name is matched
    tab.match.setCurrentIndex(tab.match.findData("all"))
    assert tab.save() and ctx.settings()["labeling"]["taxonomy_match"] == "all"
    assert AISettingsTab(ctx).match.currentData() == "all"


def test_ai_settings_can_turn_the_taxonomy_match_off(app, ctx):
    tab = AISettingsTab(ctx)
    tab.match.setCurrentIndex(tab.match.findData("off"))
    assert tab.save() and ctx.settings()["labeling"]["taxonomy_match"] == "off"


def test_ai_settings_auto_fit_option_is_saved(app, ctx):
    tab = AISettingsTab(ctx)
    assert tab.auto_fit.isChecked()  # on by default
    tab.auto_fit.setChecked(False)
    assert tab.save() and ctx.settings()["run"]["auto_fit_context"] is False
    assert not AISettingsTab(ctx).auto_fit.isChecked()


# ---------------------------------------------------------------- clearing the queue
def test_clear_queue_asks_first_and_removes_waiting_images(app, ctx, tmp_path):
    queue_images(ctx, tmp_path, 3)
    asked = []
    run = LabelRunTab(ctx, provider_fn=lambda: None, confirm_fn=lambda t: True,
                      clear_confirm_fn=lambda t: (asked.append(t), False)[1])
    assert run.clear_btn.isEnabled()
    assert run.clear_queue(("queued",)) == 0 and ctx.db.counts()["queued"] == 3  # said no
    assert "3 image(s)" in asked[0] and "stay on disk" in asked[0]
    run._confirm_clear = lambda t: True
    assert run.clear_queue(("queued",)) == 3
    assert ctx.db.counts()["queued"] == 0 and "Removed 3" in run.message.text()
    assert not run.clear_btn.isEnabled()  # nothing left to clear
    assert run.clear_queue(("queued",)) == 0 and "Nothing to remove" in run.message.text()


def test_clear_everything_warns_about_unreviewed_work_and_keeps_uploaded(app, ctx, tmp_path):
    queue_images(ctx, tmp_path, 3)
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([good(), good()]), confirm_fn=lambda t: True)
    run.limit.setValue(2)
    assert run.start() and wait_until(app, lambda: not run.running(), 20)
    ctx.db.set_status("a", "approved")
    ctx.db.set_status("a", "uploaded")
    asked = []
    run._confirm_clear = lambda t: (asked.append(t), True)[1]
    assert run.clear_queue(None) == 2  # one pending, one still queued; the uploaded one stays
    assert "waiting for review or upload" in asked[0]
    assert ctx.db.get("a").status == "uploaded" and len(ctx.db.list()) == 1


def test_clear_queue_is_not_offered_during_a_run(app, ctx, tmp_path):
    import threading

    queue_images(ctx, tmp_path, 1)
    run = LabelRunTab(ctx, provider_fn=lambda: None, confirm_fn=lambda t: True, clear_confirm_fn=lambda t: True)
    release = threading.Event()

    def held(messages):
        release.wait(10)
        return good()

    run._provider_fn = lambda: ScriptedProvider([held])
    assert run.start()
    assert not run.clear_btn.isEnabled()
    assert run.clear_queue(None) == 0 and "Stop the run" in run.message.text()
    release.set()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["pending"] == 1


# ---------------------------------------------------------------- auto-resume after a provider limit
def test_label_run_carries_on_by_itself_when_the_limit_reopens(app, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("gui.workers.run_worker.auto_wait_seconds", lambda retry_after, consecutive: 1.0)
    queue_images(ctx, tmp_path, 2)
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([limit_reply(), good(), good()]),
                      confirm_fn=lambda t: True)
    run.show()
    assert run.start()
    assert wait_until(app, lambda: run.limit_box.isVisible() and "by itself" in run.limit_text.text(), 20)
    assert ctx.db.counts()["failed"] == 0 and ctx.db.counts()["queued"] == 2  # nothing lost while waiting
    assert "Trying again by itself" in run.limit_text.text() and "Press Resume when" not in run.limit_text.text()
    # nobody presses anything: the wait runs out and the same image is tried again
    assert wait_until(app, lambda: not run.running(), 30)
    assert ctx.db.counts()["pending"] == 2 and ctx.db.counts()["failed"] == 0
    assert not run.limit_box.isVisible() and "Finished 2" in run.message.text()
    run.close()


def test_resume_now_skips_the_automatic_wait(app, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("gui.workers.run_worker.auto_wait_seconds", lambda retry_after, consecutive: 3600.0)
    queue_images(ctx, tmp_path, 1)
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([limit_reply(), good()]), confirm_fn=lambda t: True)
    run.show()
    assert run.start()
    assert wait_until(app, lambda: run.limit_box.isVisible() and "by itself" in run.limit_text.text(), 20)
    run.resume_after_limit()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["pending"] == 1
    run.close()


def test_stop_ends_an_automatic_wait(app, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("gui.workers.run_worker.auto_wait_seconds", lambda retry_after, consecutive: 3600.0)
    queue_images(ctx, tmp_path, 1)
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([limit_reply()]), confirm_fn=lambda t: True)
    run.show()
    assert run.start()
    assert wait_until(app, lambda: run.limit_box.isVisible(), 20)
    run.limit_stop_btn.click()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["queued"] == 1
    run.close()


def test_automatic_waiting_gives_up_after_many_limit_hits_in_a_row(app, ctx, tmp_path, monkeypatch):
    monkeypatch.setattr("gui.workers.run_worker.auto_wait_seconds", lambda retry_after, consecutive: 0.05)
    monkeypatch.setattr("gui.workers.run_worker.MAX_AUTO_RETRIES", 3)
    queue_images(ctx, tmp_path, 1)
    always = [limit_reply() for _ in range(10)]
    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider(always), confirm_fn=lambda t: True)
    run.show()
    assert run.start()
    # the fourth hit in a row is past the limit: the run waits for a person instead of retrying forever
    assert wait_until(app, lambda: run.worker is not None and run.worker.waiting_for_limit
                      and "by itself" not in run.limit_text.text() and run.limit_box.isVisible(), 20)
    run.limit_stop_btn.click()
    assert wait_until(app, lambda: not run.running(), 20)
    assert ctx.db.counts()["queued"] == 1 and ctx.db.counts()["failed"] == 0
    run.close()


def test_ai_settings_auto_resume_and_reply_cap_are_saved(app, ctx):
    tab = AISettingsTab(ctx)
    assert tab.auto_resume.isChecked() and tab.max_reply.value() == 0
    tab.auto_resume.setChecked(False)
    tab.max_reply.setValue(1200)
    assert tab.save()
    s = ctx.settings()
    assert s["run"]["auto_resume_on_limit"] is False and s["provider"]["reply_limit"] == 1200
    tab2 = AISettingsTab(ctx)
    assert not tab2.auto_resume.isChecked() and tab2.max_reply.value() == 1200


def test_a_stalled_image_is_skipped_and_the_run_carries_on_to_the_next(app, ctx, tmp_path):
    from core.providers.base import ProviderError
    queue_images(ctx, tmp_path, 3)

    def stalled(messages):
        raise ProviderError("Request timed out after 300 seconds: APITimeoutError: Request timed out.")

    run = LabelRunTab(ctx, provider_fn=lambda: ScriptedProvider([good(), stalled, good()]), confirm_fn=lambda t: True)
    assert run.start()
    assert wait_until(app, lambda: not run.running(), 20)
    counts = ctx.db.counts()
    assert counts["pending"] == 2 and counts["failed"] == 1 and counts["queued"] == 0
    shown = run.failure.text()
    assert "timed out after 300 seconds" in shown and "moved on to the next one" in shown
    assert "Finished 3" in run.message.text() and "The run" not in run.message.text()  # it did not stop early


# ---------------------------------------------------------------- only label what is not labeled in the current system
def _remote_tab(app, ctx, images, tmp_path, scheme="full"):
    fetched = []

    def download(img, folder, key):
        fetched.append(img.id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{img.id}.png"
        make_images(folder, (path.name,))
        return path

    ctx.update_settings(lambda s: (s["roboflow"].update(workspace="ws", project="pr"), s["labeling"].update(scheme=scheme)))
    tab = SelectImagesTab(ctx, list_fn=lambda ws, p, k: images, download_fn=download, key_fn=lambda: "rf-key")
    tab.remote_radio.setChecked(True)
    tab.load_remote()
    assert wait_until(app, lambda: len(tab.entries) == len(images))
    return tab, fetched


def test_needs_labeling_means_not_labeled_in_the_current_scheme(app, ctx, tmp_path):
    from core.roboflow.images import RemoteImage
    images = [RemoteImage("none", "a.jpg", labels=[]),
              RemoteImage("old", "b.jpg", labels=["component1", "component3"]),          # first components scheme
              RemoteImage("objs", "c.jpg", labels=["obj_pen_rubber_red_first"]),         # objects only: no laser label
              RemoteImage("done", "d.jpg", labels=["component1", "obj_pen_rubber_red_first"]),
              RemoteImage("unk", "e.jpg")]
    tab, _ = _remote_tab(app, ctx, images, tmp_path, scheme="components")
    assert {e.key for e in tab._shown} == {"none", "old", "objs", "unk"}  # only "done" is hidden
    assert "1 already labeled in the current system" in tab.message.text() and "2 labeled in an older style" in tab.message.text()
    assert "4 need labeling" in tab.count_label.text()
    tab.all_radio.setChecked(True)
    assert len(tab._shown) == 5


def test_changing_the_scheme_changes_what_counts_as_labeled(app, ctx, tmp_path):
    from core.roboflow.images import RemoteImage
    images = [RemoteImage("a", "a.jpg", labels=["las_first-incidence"]), RemoteImage("b", "b.jpg", labels=["component1"])]
    tab, _ = _remote_tab(app, ctx, images, tmp_path, scheme="full")
    assert {e.key for e in tab._shown} == {"b"}
    ctx.update_settings(lambda s: s["labeling"].update(scheme="components"))
    assert {e.key for e in tab._shown} == {"a"}  # the view follows the setting without a reload


def test_ticking_an_already_labeled_image_asks_and_skip_leaves_it_out(app, ctx, tmp_path):
    from core.roboflow.images import RemoteImage
    images = [RemoteImage("todo", "a.jpg", labels=[]), RemoteImage("done", "b.jpg", labels=["las_first-incidence"])]
    tab, fetched = _remote_tab(app, ctx, images, tmp_path)
    tab.all_radio.setChecked(True)
    tab.tick_all_btn.click()
    asked = []
    tab.ask_labeled = lambda n, scheme: (asked.append((n, scheme)), "skip")[1]
    tab.add_to_queue()
    assert asked == [(1, "full")]
    assert wait_until(app, lambda: ctx.db.counts()["queued"] == 1)
    assert fetched == ["todo"] and ctx.db.get("done") is None and "done" not in tab.ticked


def test_labeling_an_already_labeled_image_again_needs_a_yes(app, ctx, tmp_path):
    from core.roboflow.images import RemoteImage
    images = [RemoteImage("done", "b.jpg", labels=["las_first-incidence"])]
    tab, fetched = _remote_tab(app, ctx, images, tmp_path)
    tab.all_radio.setChecked(True)
    tab.tick_all_btn.click()
    tab.ask_labeled = lambda n, scheme: "cancel"
    assert tab.add_to_queue() == 0 and ctx.db.counts()["queued"] == 0
    tab.ask_labeled = lambda n, scheme: "include"
    tab.add_to_queue()
    assert wait_until(app, lambda: ctx.db.counts()["queued"] == 1) and fetched == ["done"]


def test_older_style_images_are_added_with_a_note_about_overwrite(app, ctx, tmp_path):
    from core.roboflow.images import RemoteImage
    images = [RemoteImage("old", "a.jpg", labels=["component1", "component3"])]
    tab, _ = _remote_tab(app, ctx, images, tmp_path, scheme="components")
    tab.tick_all_btn.click()
    tab.add_to_queue()
    assert wait_until(app, lambda: ctx.db.counts()["queued"] == 1)
    assert wait_until(app, lambda: "Overwrite" in tab.message.text())
    assert "older style" in tab.message.text()
