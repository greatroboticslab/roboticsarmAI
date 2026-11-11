"""A click on a laser is only "a laser is here": it says where, never how big. These cover the marked places that a re-run
passes to the model and the several clicks that answer "where is the laser?"."""
import shutil
from pathlib import Path

import pytest
from PIL import Image

from config.settings import load_settings
from core.camera import image_stats, load_image
from core.labeler.agent_mode import _start
from core.labeler.context import build_context
from core.labeler.label_image import label_image
from core.labeler.pipeline_mode import MARKS_NOTE, _user_message
from core.labeler.runner import run_queue
from core.labeler.transcript import load_transcript
from core.queue_db import NewImage, QueueDB
from core.review.service import (ReviewError, ReviewService, add_lasers_at_points, clean_points, set_rerun_points,
                                 take_rerun_note, take_rerun_points)
from core.schema.taxonomy import Taxonomy, load_taxonomy
from core.tools.laser_detect import MARK_BOX, MARK_SNAP, apply_marks, candidate_at, detect_laser
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot

ROOT = Path(__file__).resolve().parent.parent
TAX = load_taxonomy(ROOT / "taxonomy")


def two_spot_image(tmp_path, name="two.png"):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    spot(img, 60, 200, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / name
    Image.fromarray(finish(img)).save(path)
    return path


def one_spot_image(tmp_path, name="one.png"):
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / name
    Image.fromarray(finish(img)).save(path)
    return path


def blank_image(tmp_path, name="blank.png"):
    path = tmp_path / name
    Image.fromarray(finish(scene())).save(path)
    return path


# ---------------------------------------------------------------- the detector side
def test_a_mark_near_a_spot_finds_that_spots_real_size_not_a_box_around_the_click(tmp_path):
    image = load_image(one_spot_image(tmp_path))
    cand = candidate_at(image.rgb, image_stats(image.rgb), 150, 118, 1)  # about 12 px off the spot
    assert cand.marked and cand.id == "c1"
    x, y, w, h = cand.box
    assert x <= 140 <= x + w and y <= 110 <= y + h and w < MARK_BOX  # the measured core, not a square around the click


def test_a_mark_on_nothing_gets_a_small_square_and_never_crashes(tmp_path):
    image = load_image(blank_image(tmp_path))
    for x, y in ((0, 0), (image.width - 1, image.height - 1), (-50, 9999)):
        cand = candidate_at(image.rgb, image_stats(image.rgb), x, y, 3)
        bx, by, bw, bh = cand.box
        assert cand.marked and 0 <= bx and 0 <= by and bx + bw <= image.width and by + bh <= image.height


def test_a_mark_on_a_found_spot_marks_that_spot_and_makes_no_second_box(tmp_path):
    image = load_image(two_spot_image(tmp_path))
    stats = image_stats(image.rgb)
    detection = detect_laser(image.rgb, stats)
    before = len(detection.candidates)
    marked = apply_marks(detection, image.rgb, stats, [(143, 112), (141, 109)])  # twice on the same spot
    assert len(detection.candidates) == before and len(marked) == 1 and marked[0].marked
    assert detection.candidates[0] is marked[0]  # marked ones come first


def test_a_mark_far_from_any_spot_adds_a_candidate_of_its_own_with_a_new_id(tmp_path):
    image = load_image(one_spot_image(tmp_path))
    stats = image_stats(image.rgb)
    detection = detect_laser(image.rgb, stats)
    ids = {c.id for c in detection.candidates}
    marked = apply_marks(detection, image.rgb, stats, [(300, 40), (20, 20)])
    assert len(marked) == 2 and len({c.id for c in detection.candidates}) == len(detection.candidates)
    assert all(c.id not in ids for c in marked) and [c.marked for c in detection.candidates[:2]] == [True, True]
    assert MARK_SNAP > 0


def test_marked_places_are_always_kept_even_when_more_than_the_candidate_limit(tmp_path):
    image = load_image(one_spot_image(tmp_path))
    ctx = build_context(image, Taxonomy(), max_candidates=1)
    ctx.hint_points = [(300, 40), (20, 20), (300, 200)]
    cands = ctx.detection.candidates
    assert sum(c.marked for c in cands) == 3 and ctx.max_candidates >= 3
    assert all(c.marked for c in cands[:ctx.max_candidates][:3])


# ---------------------------------------------------------------- what the model is told
def test_the_model_is_told_a_mark_is_only_a_point_not_a_box(tmp_path):
    image = load_image(one_spot_image(tmp_path))
    ctx = build_context(image, TAX)
    ctx.hint_points = [(300, 40)]
    candidates = ctx.detection.candidates
    message = _user_message(ctx, candidates[:ctx.max_candidates])
    marked_id = next(c.id for c in candidates if c.marked)
    assert "the reviewer pointed at" in message.text and marked_id in message.text
    assert "only a point on a laser, not its size or its box" in message.text
    assert MARKS_NOTE.split("{ids}")[0] in message.text
    plain = build_context(image, TAX)
    assert "reviewer pointed" not in _user_message(plain, plain.detection.candidates).text


def test_agent_mode_is_told_the_same(tmp_path):
    ctx = build_context(load_image(one_spot_image(tmp_path)), TAX)
    ctx.hint_points = [(300.4, 40.6)]
    text_ = _start(ctx).text
    assert "(300, 41)" in text_ and "not its box" in text_
    ctx.hint_points = []
    assert "pointed at" not in _start(ctx).text


def test_a_rerun_with_marks_labels_the_marked_place_and_records_it(tmp_path):
    path = one_spot_image(tmp_path)
    settings = load_settings()
    reply = text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0, "from_id": None,
                                 "object": PEN_RUBBER}]))
    provider = ScriptedProvider([reply])
    result = label_image(path, provider=provider, taxonomy=TAX, settings=settings, data_dir=tmp_path / "data",
                         hint_points=[(141, 111)])
    assert result.status == "pending" and len(result.laser_boxes) == 1  # the mark snapped to the spot: one box
    sent = provider.calls[0][-1].text
    assert "the reviewer pointed at" in sent
    entries = load_transcript(Path(result.transcript_path))["entries"]
    assert any(e.get("type") == "reviewer_marks" and "(141, 111)" in e.get("text", "") for e in entries)


# ---------------------------------------------------------------- storing marks for a re-run
def test_marks_are_stored_per_image_and_used_once(tmp_path):
    set_rerun_points(tmp_path, "a", [(10, 20), (30.5, 40.5)])
    set_rerun_points(tmp_path, "b", [(1, 2)])
    assert take_rerun_points(tmp_path, "a") == [(10.0, 20.0), (30.5, 40.5)]
    assert take_rerun_points(tmp_path, "a") == []
    assert take_rerun_points(tmp_path, "b") == [(1.0, 2.0)]
    assert take_rerun_points(tmp_path, "never") == []


def test_setting_no_marks_clears_old_ones(tmp_path):
    set_rerun_points(tmp_path, "a", [(1, 2)])
    set_rerun_points(tmp_path, "a", [])
    assert take_rerun_points(tmp_path, "a") == []


@pytest.mark.parametrize("bad", [[("x", 1)], [(1,)], [None], [(float("nan"), 1)], [(float("inf"), 2)]])
def test_clean_points_refuses_anything_that_is_not_a_pair_of_numbers(bad):
    with pytest.raises(ReviewError):
        clean_points(bad)


def test_a_corrupt_marks_file_is_ignored(tmp_path):
    (tmp_path / "rerun_points.json").write_text("{not json", encoding="utf-8")
    assert take_rerun_points(tmp_path, "a") == []
    (tmp_path / "rerun_points.json").write_text('{"a": [["x", 1]]}', encoding="utf-8")
    assert take_rerun_points(tmp_path, "a") == []


# ---------------------------------------------------------------- the service
@pytest.fixture
def env(tmp_path):
    taxonomy_dir = tmp_path / "taxonomy"
    shutil.copytree(ROOT / "taxonomy", taxonomy_dir)
    data = tmp_path / "data"
    db = QueueDB(data / "queue.db")
    settings = load_settings()
    path = two_spot_image(tmp_path)
    blank = blank_image(tmp_path)

    def label(image_path, image_id, replies):
        db.add_images([NewImage(image_id, image_path.name, str(image_path))])
        provider = ScriptedProvider(replies)
        run_queue(db, lambda row: label_image(row.path, provider=provider, taxonomy=TAX, settings=settings,
                                              data_dir=data, image_id=row.image_id), sidecar_dir=data / "sidecars")

    env = type("Env", (), {})()
    env.db, env.data, env.settings, env.path, env.blank, env.label = db, data, settings, path, blank, label
    env.service = ReviewService(db, data, taxonomy_dir, settings)
    yield env
    db.close()


def test_rerun_stores_the_note_and_the_marks_together_and_requeues(env):
    env.label(env.path, "a", [text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                                    "from_id": None, "object": PEN_RUBBER}]))])
    env.service.rerun("a", "look at the left one", [(60, 200), (140, 110)])
    assert env.db.get("a").status == "queued"
    assert take_rerun_note(env.data, "a") == "look at the left one"
    assert take_rerun_points(env.data, "a") == [(60.0, 200.0), (140.0, 110.0)]


def test_rerun_with_marks_but_no_note_is_fine(env):
    env.label(env.path, "a", [text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                                    "from_id": None, "object": PEN_RUBBER}]))])
    env.service.rerun("a", "", [(60, 200)])
    assert take_rerun_note(env.data, "a") == "" and take_rerun_points(env.data, "a") == [(60.0, 200.0)]


def test_a_bad_mark_stops_the_rerun_before_anything_changes(env):
    env.label(env.path, "a", [text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                                    "from_id": None, "object": PEN_RUBBER}]))])
    status = env.db.get("a").status
    with pytest.raises(ReviewError):
        env.service.rerun("a", "note", [("x", "y")])
    assert env.db.get("a").status == status and take_rerun_note(env.data, "a") == ""


def test_several_clicks_answer_where_is_the_laser_with_one_box_each(env):
    env.label(env.blank, "blank", [])
    env.service.answer_question("blank", "q_no_laser", "140,110;60,200")
    ls = env.service.load("blank")
    assert env.db.get("blank").status == "pending" and len(ls.laser_boxes) == 2
    first, second = sorted(ls.laser_boxes, key=lambda b: b.order)
    assert first.kind == "first_incidence" and second.kind == "other"
    assert abs(first.box[0] + first.box[2] / 2 - 140) < 6 and abs(second.box[0] + second.box[2] / 2 - 60) < 6
    assert all(q.answer for q in ls.questions)


def test_two_clicks_on_the_same_spot_make_one_box(env):
    env.label(env.blank, "blank", [])
    env.service.answer_question("blank", "q_no_laser", "140,110;142,112")
    assert len(env.service.load("blank").laser_boxes) == 1


def test_a_single_click_answer_still_works_as_before(env):
    env.label(env.blank, "blank", [])
    env.service.answer_question("blank", "q_no_laser", "140,110")
    ls = env.service.load("blank")
    assert len(ls.laser_boxes) == 1 and ls.laser_boxes[0].kind == "first_incidence"


@pytest.mark.parametrize("answer", ["not a point", "1,2;3", "1,2,3", ";", "a,b;c,d"])
def test_malformed_answers_are_refused_with_a_plain_message(env, answer):
    env.label(env.blank, "blank", [])
    with pytest.raises(ReviewError, match="Click on the image where a laser is"):
        env.service.answer_question("blank", "q_no_laser", answer)
    assert env.db.get("blank").status == "needs_input"


def test_add_lasers_at_points_gives_the_components_scheme_its_classes(env):
    env.label(env.blank, "blank", [])
    ls = env.service.load("blank")
    settings = {**env.settings, "labeling": {**env.settings["labeling"], "scheme": "components"}}
    made = add_lasers_at_points(ls, env.path, settings, [(140, 110), (60, 200)])
    assert [b.component for b in made] == ["component1", "component2"]
