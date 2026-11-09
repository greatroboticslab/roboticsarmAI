"""A provider limit pauses the run and asks, in the label function and in the Label run tab."""
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from config.settings import load_settings
from core.labeler.factory import Escalation, build_label_fn
from core.labeler.guard import TrackerGuard
from core.labeler.label_image import label_image_unsaved
from core.providers.base import RateLimitError
from core.providers.hints import limit_advice
from core.queue_db import NewImage, QueueDB
from core.schema.taxonomy import load_taxonomy
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot

TAX = load_taxonomy(Path(__file__).resolve().parent.parent / "taxonomy")


def hit_limit(retry_after=None):
    def reply(messages):
        raise RateLimitError("429 You exceeded your current quota", retry_after)
    return reply


def good():
    return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                "from_id": None, "object": PEN_RUBBER}]))


@pytest.fixture
def image(tmp_path):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / "cam1" / "red_object.png"
    path.parent.mkdir()
    Image.fromarray(finish(img)).save(path)
    return path


# ------------------------------------------------------------------ advice text
def test_advice_uses_the_providers_wait_and_says_nothing_is_lost():
    text_ = limit_advice("429 quota", 600, now=datetime(2026, 11, 5, 20, 0))
    assert "10 minutes" in text_ and "20:10" in text_ and "put back in the queue" in text_ and "Resume" in text_


def test_advice_without_a_time_explains_minute_and_daily_limits():
    assert "daily quota" in limit_advice("Quota exceeded for metric per day")
    unknown = limit_advice("429 too many requests")
    assert "per-minute" in unknown and "usage page" in unknown


# ------------------------------------------------------------------ label function
def test_label_image_raises_a_limit_only_when_asked(image, tmp_path):
    kwargs = dict(provider=ScriptedProvider([hit_limit()]), taxonomy=TAX, settings=load_settings())
    with pytest.raises(RateLimitError):
        label_image_unsaved(image, raise_on_limit=True, **kwargs)
    kwargs["provider"] = ScriptedProvider([hit_limit()])
    result, _ = label_image_unsaved(image, **kwargs)  # review, compare and single-image commands keep the old behaviour
    assert result.status == "failed" and "quota" in result.notes


def test_queue_label_function_lets_the_limit_through(image, tmp_path):
    db = QueueDB(tmp_path / "q.db")
    db.add_images([NewImage("a", "a.png", str(image))])
    fn = build_label_fn(load_settings(), ScriptedProvider([hit_limit(30)]), TAX, tmp_path / "data")
    with pytest.raises(RateLimitError) as exc:
        fn(db.claim_next())
    assert exc.value.retry_after == 30
    db.close()


def test_a_limit_on_the_escalation_model_keeps_the_first_answer(image, tmp_path):
    db = QueueDB(tmp_path / "q.db")
    db.add_images([NewImage("a", "a.png", str(image))])
    first = ScriptedProvider([good()])
    second = ScriptedProvider([hit_limit()])
    settings = load_settings()
    esc = Escalation(second, TrackerGuard.__new__(TrackerGuard), 1.01)  # threshold above 1: always escalate
    esc.guard = None
    fn = build_label_fn(settings, first, TAX, tmp_path / "data", escalation=esc)
    result = fn(db.claim_next())
    assert result.status in ("pending", "needs_input") and result.provider == "fake"
    assert second.calls  # it was asked, hit its limit, and the first result stood
    db.close()
