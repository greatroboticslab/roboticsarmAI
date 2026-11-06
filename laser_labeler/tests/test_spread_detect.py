"""Spread-out lasers, textured scenes, and plain-words explanations when nothing is found."""
import numpy as np
import pytest
from PIL import Image

from core.camera import image_stats
from core.labeler.label_image import label_image
from core.labeler.transcript import load_transcript
from core.review.reasons import (COLOR, LOW_CONFIDENCE, MODEL, NO_LASER, TAXONOMY, describe_summary, question_kind,
                                 reason_line, summarize)
from core.models import ImageLabelSet, Question
from core.tools.laser_detect import detect_laser
from config.settings import load_settings
from core.schema.taxonomy import load_taxonomy
from pathlib import Path
from tests.fakes import ScriptedProvider
from tests.synth import finish, rect, scene, spot

TAX = load_taxonomy(Path(__file__).resolve().parent.parent / "taxonomy")


def detect(img):
    return detect_laser(finish(img))


# ---------------------------------------------------------------- noise is not the picture's own contrast
def test_noise_is_not_inflated_by_gradients_or_objects():
    flat = image_stats(finish(scene(640, 480))).noise_level
    img = scene(640, 480)
    yy, xx = np.mgrid[0:480, 0:640]
    img += (xx / 640 * 60)[..., None]  # lighting gradient
    img[100:300, 100:400] = 200  # a large bright object
    busy = image_stats(finish(img)).noise_level
    assert busy == pytest.approx(flat, rel=0.35) and busy * 255 < 5


def test_white_noise_is_measured_close_to_its_true_size():
    rng = np.random.default_rng(3)
    plane = 100 + rng.normal(0, 4.0, (300, 300))
    rgb = finish(np.repeat(plane[..., None], 3, axis=2))
    assert image_stats(rgb).noise_level * 255 == pytest.approx(4.0, rel=0.15)


# ---------------------------------------------------------------- spread spots are found as one spot
@pytest.mark.parametrize("sigma", [3, 8, 15, 25, 40])
def test_gaussian_spots_of_any_spread_give_one_candidate(sigma):
    img = scene(640, 480)
    spot(img, 320, 240, sigma=sigma, amp=0.5, color=(1, 1, 1))
    found = detect(img).candidates
    assert len(found) == 1
    x, y, w, h = found[0].box
    assert abs(x + w / 2 - 320) < 6 and abs(y + h / 2 - 240) < 6


def test_a_spread_spot_on_a_bright_object_is_one_candidate_not_the_object():
    img = scene(640, 480)
    rect(img, 200, 150, 240, 180, (150, 30, 30))
    spot(img, 320, 240, sigma=15, amp=0.5, color=(1, 1, 1))
    found = detect(img).candidates
    assert len(found) == 1 and found[0].area < 4000


def test_a_lighting_gradient_does_not_hide_the_spot_or_add_false_ones():
    img = scene(640, 480)
    yy, xx = np.mgrid[0:480, 0:640]
    img += (xx / 640 * 60)[..., None]
    spot(img, 320, 240, sigma=18, amp=0.5, color=(1, 1, 1))
    found = detect(img).candidates
    assert len(found) == 1 and abs(found[0].center[0] - 320) < 8


def test_a_speckled_spread_spot_is_not_split_into_nested_candidates():
    rng = np.random.default_rng(1)
    from scipy.ndimage import gaussian_filter
    img = scene(640, 480)
    yy, xx = np.mgrid[0:480, 0:640]
    speck = gaussian_filter(rng.normal(1.0, 1.0, (480, 640)), 2.5)
    speck = np.clip(1 + (speck - 1) / (speck.std() / 0.3), 0, None)
    light = 0.6 * np.exp(-(((xx - 320) ** 2 + (yy - 240) ** 2) / (2 * 18 ** 2))) * speck * 255
    img += light[..., None]
    found = detect(img).candidates
    assert len(found) == 1 and found[0].area > 1000


def test_a_clipped_white_blob_larger_than_the_search_window_is_still_a_spot():
    img = scene(640, 480)
    spot(img, 320, 240, sigma=40, amp=4.0, color=(1, 1, 1))
    found = detect(img).candidates
    assert len(found) == 1 and found[0].saturated_fraction > 0.5


def test_streaks_still_count_as_spots_when_not_too_thin():
    img = scene(640, 480)
    spot(img, 320, 240, sigma=30, sigma_y=8, amp=0.5, color=(1, 1, 1))
    assert len(detect(img).candidates) == 1


# ---------------------------------------------------------------- saying why nothing was found
def test_empty_image_explanation_says_how_faint_it_is():
    d = detect(scene(320, 240))
    assert not d.candidates
    text = d.explain()
    assert "found no laser spot" in text and "noise levels" in text and "levels above the background" in text


def test_found_spots_are_summarized():
    img = scene(320, 240)
    spot(img, 160, 120, sigma=3, amp=0.5, color=(1, 1, 1))
    assert detect(img).explain().startswith("Image search found 1 candidate laser spot(s)")


def test_rejections_are_counted_and_named():
    img = scene(640, 480)
    yy, xx = np.mgrid[0:480, 0:640]
    img += (xx / 640 * 80)[..., None]  # bright end of a gradient: a bright area, not a spot
    d = detect(img)
    assert not d.candidates and d.seeds >= 1
    assert "too large" in d.explain() or "too small" in d.explain() or "none stood out" in d.explain()


# ---------------------------------------------------------------- reasons
def q(qid, field=None):
    return Question(qid, "text", None, field)


def test_question_kinds_and_summaries():
    assert question_kind(q("q_no_laser", "hit_point")) == NO_LASER
    assert question_kind(q("ql_o1")) == LOW_CONFIDENCE and question_kind(q("qc_o1")) == COLOR
    assert question_kind(q("qt_o1_0")) == TAXONOMY and question_kind(q("q7")) == MODEL
    a = ImageLabelSet("a", "a.png", "c", False, 0.1, questions=[q("q_no_laser", "hit_point")])
    b = ImageLabelSet("b", "b.png", "c", False, 0.1, questions=[q("ql_o1"), q("qc_o1"), q("ql_o2")])
    done = ImageLabelSet("c", "c.png", "c", False, 0.1, questions=[Question("ql_o1", "t", answer="x")])
    assert reason_line(a) == NO_LASER and reason_line(b) == f"{LOW_CONFIDENCE}, {COLOR}" and reason_line(done) == ""
    counts = summarize([a, b, done])
    assert counts[NO_LASER] == 1 and counts[LOW_CONFIDENCE] == 1 and counts[COLOR] == 1
    assert describe_summary(counts).startswith("1 x ")


# ---------------------------------------------------------------- the transcript explains the status
def test_transcript_records_detection_questions_and_result_even_with_no_model_call(tmp_path):
    path = tmp_path / "blank.png"
    Image.fromarray(finish(scene())).save(path)
    result = label_image(path, provider=ScriptedProvider([]), taxonomy=TAX, settings=load_settings(),
                         data_dir=tmp_path / "data")
    assert result.status == "needs_input"
    entries = load_transcript(result.transcript_path)["entries"]
    kinds = [e["type"] for e in entries]
    assert kinds[0] == "detection" and "question" in kinds and kinds[-1] == "result"
    detection = entries[0]["text"]
    assert "found no laser spot" in detection
    question = next(e for e in entries if e["type"] == "question")["text"]
    assert question.startswith(f"[{NO_LASER}]") and "Click where the beam first hits" in question
    assert "status needs_input" in entries[-1]["text"]
