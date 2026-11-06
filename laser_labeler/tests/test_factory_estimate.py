import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from config import credentials
from config.settings import load_settings
from core.estimate import (DEFAULT_SECONDS_PER_IMAGE, average_tokens, estimate_run, save_seconds_per_image,
                           seconds_per_image)
from core.labeler import factory
from core.labeler.factory import Escalation, build_escalation, build_label_fn, needs_escalation
from core.labeler.guard import TrackerGuard
from core.labeler.runner import run_queue
from core.labeler.transcript import load_transcript
from core.models import ImageLabelSet, LaserBox, ObjectBox
from core.providers.base import ProviderError
from core.providers.cost import CostTracker, SpendCapExceeded
from core.queue_db import NewImage, QueueDB
from core.reports.sidecar import read_sidecar
from core.review.service import set_rerun_note
from core.schema.taxonomy import load_taxonomy
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot

ROOT = Path(__file__).resolve().parent.parent
TAX = load_taxonomy(ROOT / "taxonomy")


@pytest.fixture
def env(tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / "a.png"
    Image.fromarray(finish(img)).save(path)
    db = QueueDB(tmp_path / "data" / "queue.db")
    db.add_images([NewImage("a", "a.png", str(path))])
    yield type("E", (), {"db": db, "data": tmp_path / "data", "settings": load_settings(), "path": path})()
    db.close()


def reply(confidence=0.9, **obj):
    return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0, "from_id": None,
                                "object": {**PEN_RUBBER, "confidence": confidence, **obj}}]))


class Broken(ScriptedProvider):
    name, model = "broken", "b1"

    def chat(self, *a, **k):
        raise ProviderError("server down")


def run(env, fn):
    return run_queue(env.db, fn, sidecar_dir=env.data / "sidecars")


def labeled(env):
    return read_sidecar(env.db.get("a").sidecar_path)


# ---------------------------------------------------------------- label function
def test_plain_label_fn_saves_everything_and_passes_the_rerun_note(env):
    provider = ScriptedProvider([reply()])
    set_rerun_note(env.data, "a", "check the cap")
    seen = []
    fn = build_label_fn(env.settings, provider, TAX, env.data, on_entry=seen.append)
    summary = run(env, fn)
    assert summary.pending == 1 and "check the cap" in provider.calls[0][1].text
    assert any(e["type"] == "reviewer_note" for e in seen) and any(e["type"] == "message" for e in seen)  # live feed
    assert (env.data / "transcripts" / "a.json").is_file() and (env.data / "manifest.csv").is_file()
    assert not (env.data / "rerun_notes.json").read_text().strip("{}\n ")  # note used up


def test_broken_listener_cannot_break_labeling(env):
    def listener(entry):
        raise RuntimeError("viewer crashed")
    summary = run(env, build_label_fn(env.settings, ScriptedProvider([reply()]), TAX, env.data, on_entry=listener))
    assert summary.pending == 1


# ---------------------------------------------------------------- escalation
def escalation(replies, threshold=0.5, provider_cls=ScriptedProvider):
    second = provider_cls(replies)
    second.name, second.model = "cloud", "big"
    return Escalation(second, TrackerGuard(CostTracker()), threshold), second


def test_needs_escalation_rules():
    sure = ImageLabelSet("x", "x", "c", False, 0.0, laser_boxes=[LaserBox("l1", "first_incidence", (1, 1, 5, 5), 0, metrics={"confidence": 0.9})],
                         object_boxes=[ObjectBox("o1", (0, 0, 9, 9), "pen", "glass", "red", "first", confidence=0.3)])
    assert "below" in needs_escalation(sure, 0.5) and needs_escalation(sure, 0.2) == ""
    assert needs_escalation(ImageLabelSet("x", "x", "c", False, 0.0, status="failed"), 0.5)
    assert needs_escalation(ImageLabelSet("x", "x", "c", False, 0.0, status="needs_input"), 0.5) == ""  # nothing found


def test_unsure_result_is_re_labeled_by_the_escalation_model(env):
    esc, second = escalation([reply(confidence=0.95)])
    fn = build_label_fn(env.settings, ScriptedProvider([reply(confidence=0.1)]), TAX, env.data, escalation=esc)
    run(env, fn)
    ls = labeled(env)
    assert (ls.provider, ls.model) == ("cloud", "big") and ls.notes.startswith("escalated from fake/fake-model")
    assert len(second.calls) == 1
    entries = [e["type"] for e in load_transcript(env.db.get("a").transcript_path)["entries"]]
    assert entries.count("message") >= 3  # the saved transcript is the escalated one


def test_confident_result_is_not_escalated(env):
    esc, second = escalation([])
    run(env, build_label_fn(env.settings, ScriptedProvider([reply(confidence=0.95)]), TAX, env.data, escalation=esc))
    assert second.calls == [] and labeled(env).provider == "fake"


def test_failed_escalation_keeps_the_usable_first_answer(env):
    esc, _ = escalation([], provider_cls=Broken)
    run(env, build_label_fn(env.settings, ScriptedProvider([reply(confidence=0.1)]), TAX, env.data, escalation=esc))
    ls = labeled(env)
    assert ls.status != "failed" and ls.provider == "fake"
    assert load_transcript(env.db.get("a").transcript_path)["entries"][-1]["type"] in ("message", "escalation")


def test_a_failed_first_model_goes_to_the_escalation_model(env):
    esc, _ = escalation([reply(confidence=0.9)])
    run(env, build_label_fn(env.settings, Broken([]), TAX, env.data, escalation=esc))
    assert labeled(env).provider == "cloud" and env.db.get("a").status == "pending"


def test_both_models_failing_is_recorded_as_failed(env):
    esc, _ = escalation([], provider_cls=Broken)
    run(env, build_label_fn(env.settings, Broken([]), TAX, env.data, escalation=esc))
    assert env.db.get("a").status == "failed"


def test_spend_cap_hit_by_the_escalation_model_stops_the_run(env):
    class Capped(TrackerGuard):
        def before_call(self):
            raise SpendCapExceeded("cap")

    esc = Escalation(ScriptedProvider([]), Capped(CostTracker()), 0.5)
    summary = run(env, build_label_fn(env.settings, ScriptedProvider([reply(confidence=0.1)]), TAX, env.data, escalation=esc))
    assert summary.stopped == "spend_cap" and env.db.get("a").status == "queued"


def test_build_escalation_settings(env, monkeypatch):
    assert build_escalation(env.settings, env.data) is None  # off by default
    env.settings["escalation"].update(enabled=True, provider="openai_compat", model="m")
    with pytest.raises(ValueError, match="gemini, anthropic"):
        build_escalation(env.settings, env.data)
    env.settings["escalation"].update(provider="gemini", model="gem-model", price_in_per_mtok=1.0, price_out_per_mtok=2.0)
    monkeypatch.setattr(credentials, "require_key", lambda s: "k")
    monkeypatch.setattr(factory, "build_provider", lambda block, key: ScriptedProvider([]))
    esc = build_escalation(env.settings, env.data)
    assert esc.threshold == 0.5 and esc.guard.tracker.price_in_per_mtok == 1.0
    assert esc.guard.tracker.daily_cap_usd == env.settings["provider"]["daily_cost_cap_usd"]  # the one shared cap
    monkeypatch.setattr(credentials, "require_key", lambda s: (_ for _ in ()).throw(credentials.MissingCredentialError("no key")))
    with pytest.raises(credentials.MissingCredentialError):
        build_escalation(env.settings, env.data)


# ---------------------------------------------------------------- estimates
def test_seconds_per_image_default_and_saved(tmp_path):
    assert seconds_per_image(tmp_path) == DEFAULT_SECONDS_PER_IMAGE
    save_seconds_per_image(tmp_path, 7.3)
    save_seconds_per_image(tmp_path, 0)  # ignored
    assert seconds_per_image(tmp_path) == 7.3
    (tmp_path / "run_stats.json").write_text("garbage")
    assert seconds_per_image(tmp_path) == DEFAULT_SECONDS_PER_IMAGE


def test_average_tokens_reads_transcripts(tmp_path):
    folder = tmp_path / "transcripts"
    folder.mkdir()
    assert average_tokens(folder) is None
    (folder / "a.json").write_text(json.dumps({"usage": {"input_tokens": 1000, "output_tokens": 100}, "entries": []}))
    (folder / "b.json").write_text(json.dumps({"usage": {"input_tokens": 3000, "output_tokens": 300}, "entries": []}))
    (folder / "c.json").write_text("broken")
    (folder / "d.json").write_text(json.dumps({"usage": {}, "entries": []}))
    assert average_tokens(folder) == (2000, 200)


def test_estimate_free_model_and_priced_model_and_cap(tmp_path):
    settings = load_settings()
    free = estimate_run(100, settings, tmp_path)
    assert free.cost_usd is None and free.seconds == 1000 and "no price set" in free.describe() and not free.over_cap
    settings["provider"].update(price_in_per_mtok=10.0, price_out_per_mtok=20.0, daily_cost_cap_usd=0.5)
    priced = estimate_run(100, settings, tmp_path)
    assert priced.cost_usd == pytest.approx(100 * (3000 * 10 + 500 * 20) / 1e6) and priced.over_cap
    assert "rough" in priced.describe() and "$0.50" in priced.describe()
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "transcripts" / "a.json").write_text(json.dumps({"usage": {"input_tokens": 1000, "output_tokens": 0}}))
    measured = estimate_run(100, settings, tmp_path)
    assert measured.tokens_are_measured and measured.cost_usd == pytest.approx(1.0) and "rough" not in measured.describe()
    assert "hours" in estimate_run(5000, load_settings(), tmp_path).describe()
