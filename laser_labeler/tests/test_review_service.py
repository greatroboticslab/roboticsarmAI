import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from config.settings import load_settings
from core.labeler.label_image import label_image
from core.labeler.runner import run_queue
from core.labeler.transcript import load_transcript
from core.prompts import examples
from core.queue_db import NewImage, QueueDB
from core.reports import manifest
from core.reports.pdf import build_pdf, load_addenda, add_addendum, class_label
from core.review.diffs import diff_label_sets
from core.review.service import (ReviewError, ReviewService, add_laser_at_point, add_object_box, delete_box,
                                 next_id, take_rerun_note)
from core.schema.taxonomy import load_taxonomy
from tests.fakes import PEN_RUBBER, ScriptedProvider, pipeline_json, text
from tests.synth import finish, rect, scene, spot

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def env(tmp_path):
    taxonomy_dir = tmp_path / "taxonomy"
    shutil.copytree(ROOT / "taxonomy", taxonomy_dir)
    data = tmp_path / "data"
    img = scene()
    rect(img, 100, 80, 80, 60, (150, 30, 30))
    spot(img, 140, 110, sigma=2.5, amp=0.5, color=(1, 1, 1))
    path = tmp_path / "a.png"
    Image.fromarray(finish(img)).save(path)
    blank = tmp_path / "blank.png"
    Image.fromarray(finish(scene())).save(blank)
    db = QueueDB(data / "queue.db")
    settings = load_settings()
    tax = load_taxonomy(taxonomy_dir)

    def good(**changes):
        return text(pipeline_json([{"id": "c1", "is_laser": True, "kind": "first_incidence", "order": 0,
                                    "from_id": None, "object": {**PEN_RUBBER, **changes}}]))

    def label(image_path, image_id, replies):
        db.add_images([NewImage(image_id, image_path.name, str(image_path))])
        provider = ScriptedProvider(replies)
        run_queue(db, lambda row: label_image(row.path, provider=provider, taxonomy=tax, settings=settings,
                                              data_dir=data, image_id=row.image_id),
                  sidecar_dir=data / "sidecars")

    env = type("Env", (), {})()
    env.db, env.data, env.settings, env.path, env.blank, env.good, env.label = db, data, settings, path, blank, good, label
    env.service = ReviewService(db, data, taxonomy_dir, settings)
    env.taxonomy_dir = taxonomy_dir
    yield env
    db.close()


# ---------------------------------------------------------------- loading and approving
def test_load_uses_queue_status_and_approve_builds_pdf(env):
    env.label(env.path, "a", [env.good()])
    ls = env.service.load("a")
    assert ls.status == "pending" and len(ls.object_boxes) == 1
    assert env.service.approve("a") == []
    row = env.db.get("a")
    assert row.status == "approved" and Path(row.pdf_path).is_file() and Path(row.pdf_path).name == "a.pdf"
    assert env.service.load("a").status == "approved"
    assert [r["status"] for r in manifest.read_manifest(env.data / "manifest.csv")] == ["approved"]
    assert manifest.read_manifest(env.data / "manifest.csv")[0]["pdf_path"].endswith("a.pdf")
    assert env.service.approved_sets()[0].pdf_path.endswith("a.pdf")  # upload tags need it


def test_load_without_labels_and_unknown_image(env):
    env.db.add_images([NewImage("q", "q.png", "/x/q.png")])
    with pytest.raises(ReviewError, match="no labels yet"):
        env.service.load("q")
    with pytest.raises(ReviewError, match="not in the queue"):
        env.service.load("ghost")


def test_reject_reopen_and_blocked_moves(env):
    env.label(env.path, "a", [env.good()])
    env.service.reject("a")
    assert env.db.get("a").status == "rejected"
    with pytest.raises(ReviewError, match="cannot become"):
        env.service.approve("a")  # rejected -> approved is not allowed; reopen first
    env.service.reopen("a")
    assert env.db.get("a").status == "pending"
    env.service.approve("a")
    env.service.reopen("a")
    assert env.db.get("a").status == "pending"


def test_needs_input_cannot_be_approved(env):
    env.label(env.blank, "blank", [])  # no laser found: parked with a question, the model is never called
    assert env.db.get("blank").status == "needs_input"
    with pytest.raises(ReviewError, match="questions"):
        env.service.approve("blank")


def test_bulk_approve_updates_sidecars_and_defers_pdfs(env):
    env.label(env.path, "a", [env.good(confidence=0.99)])
    ids = env.service.bulk_approve(0.1)
    assert ids == ["a"] and env.service.load("a").status == "approved"
    assert env.db.get("a").pdf_path is None  # built later, only when needed
    assert env.service.ensure_pdfs(["a"]) == [] and Path(env.db.get("a").pdf_path).is_file()
    assert env.service.pdf_is_current(env.db.get("a"))


# ---------------------------------------------------------------- editing
def test_edit_attribute_marks_edited_and_records_everything(env):
    env.label(env.path, "a", [env.good()])
    ls = env.service.load("a")
    ls.object_boxes[0].material = "glass"
    result = env.service.save_edit("a", ls, note="it is glass")
    assert result.status == "edited" and [c.kind for c in result.changes] == ["attribute"]
    assert env.service.load("a").object_boxes[0].material == "glass"
    assert env.db.get("a").status == "edited" and Path(env.db.get("a").pdf_path).is_file()
    addenda = load_addenda(env.data / "pdfs" / "a.addenda.json")
    assert addenda[0]["note"] == "it is glass" and "material rubber -> glass" in addenda[0]["changes"][0]
    corrections = examples.load_corrections(env.data / "corrections.jsonl")
    assert len(corrections) == 1 and "rubber -> glass" in corrections[0]["text"] and not corrections[0]["included"]
    assert env.service.save_edit("a", env.service.load("a")).changes == []  # nothing changed: nothing recorded
    assert len(load_addenda(env.data / "pdfs" / "a.addenda.json")) == 1
    assert [r["labeled_at"] for r in manifest.read_manifest(env.data / "manifest.csv")][0]  # kept


def test_moving_a_laser_box_remeasures_it(env):
    env.label(env.path, "a", [env.good()])
    ls = env.service.load("a")
    old_area = ls.laser_boxes[0].metrics["core_area_px"]
    ls.laser_boxes[0].box = (20, 20, 12, 12)  # empty background
    result = env.service.save_edit("a", ls)
    new = env.service.load("a").laser_boxes[0].metrics
    assert result.warnings == [] and new["confidence"] == 0.0 and new["core_area_px"] != old_area


def test_edit_of_a_hand_typed_measurement_is_kept(env):
    env.label(env.path, "a", [env.good()])
    ls = env.service.load("a")
    ls.laser_boxes[0].metrics["peak_intensity_norm"] = 0.42
    result = env.service.save_edit("a", ls)
    assert [c.kind for c in result.changes] == ["metric"]
    assert env.service.load("a").laser_boxes[0].metrics["peak_intensity_norm"] == 0.42
    assert examples.load_corrections(env.data / "corrections.jsonl") == []  # not a teachable example


def test_diff_covers_add_remove_move_and_lasers(env):
    env.label(env.path, "a", [env.good()])
    before, after = env.service.load("a"), env.service.load("a")
    tax = env.service.taxonomy()
    after.object_boxes[0].box = (90, 70, 100, 80)
    added = add_object_box(after, env.path, tax, (10, 10, 30, 30), env.settings)
    delete_box(after, "l1")
    kinds = sorted((c.kind, c.box_id) for c in diff_label_sets(before, after))
    assert kinds == [("added", added.id), ("box", "o1"), ("removed", "l1")]
    assert all(c.describe() for c in diff_label_sets(before, after))
    assert after.object_boxes[0].laser_ids == []  # link to the deleted laser box is gone


def test_new_boxes_get_unique_ids_and_sensible_defaults(env):
    env.label(env.path, "a", [env.good()])
    ls, tax = env.service.load("a"), env.service.taxonomy()
    obj = add_object_box(ls, env.path, tax, (100, 80, 80, 60), env.settings)
    assert obj.id == "o2" and obj.color == "red" and obj.laser_ids == ["l1"]  # colour sampled, laser inside
    assert obj.general_object in tax.objects
    assert next_id("o", ["o1", "o2", "o4"]) == "o3"


# ---------------------------------------------------------------- questions
def test_answering_every_question_resumes_the_image(env):
    env.label(env.path, "a", [env.good(general_object="spoon", material="metal")])
    row = env.db.get("a")
    assert row.status == "needs_input" and row.question_count >= 1
    questions = env.service.load("a").questions
    for q in questions:
        env.service.answer_question("a", q.id, "checked")
    assert env.db.get("a").status == "pending" and env.db.get("a").question_count == 0
    events = [e for e in load_transcript(env.db.get("a").transcript_path)["entries"] if e["type"] == "answer"]
    assert len(events) == len(questions) and "-> checked" in events[0]["text"]


def test_answer_sets_the_field_and_can_extend_the_taxonomy(env):
    env.settings["labeling"]["taxonomy_match"] = "all"
    env.label(env.path, "a", [env.good(material="velvet")])
    ls = env.service.load("a")
    q = next(q for q in ls.questions if q.field_name is None and "material" in q.text)
    warnings = env.service.answer_question("a", q.id, "checked")
    assert warnings == []
    assert env.service.taxonomy_problems(env.service.load("a")) and "velvet" in env.service.taxonomy_problems(env.service.load("a"))[0]
    env.service.add_to_taxonomy("material", "velvet")
    assert env.service.taxonomy_problems(env.service.load("a")) == []


def test_field_answer_validation(env):
    env.settings["labeling"]["taxonomy_match"] = "all"
    env.label(env.path, "a", [env.good(general_object="spoon", material="metal")])
    # build a field question by hand on the saved label set
    ls = env.service.load("a")
    from core.models import Question
    ls.questions = [Question("qm", "Which material?", "o1", "material", ["glass", "metal"], True),
                    Question("qi", "Incidence?", "o1", "incidence", ["first"]),
                    Question("ql", "Which event?", "l1", "kind", ["reflection"])]
    env.service.save_edit("a", ls)  # an edit that only adds questions has no label changes, so write directly instead
    from core.reports.sidecar import write_sidecar
    write_sidecar(ls, env.data / "sidecars")
    with pytest.raises(ReviewError, match="Incidence must be"):
        env.service.answer_question("a", "qi", "sideways")
    with pytest.raises(ReviewError, match="Please choose"):
        env.service.answer_question("a", "qm", "  ")
    with pytest.raises(ReviewError, match="not found"):
        env.service.answer_question("a", "nope", "x")
    with pytest.raises(ReviewError):
        env.service.answer_question("a", "qm", "Bad Value!")  # not a valid class-name token
    warnings = env.service.answer_question("a", "qm", "glass")
    assert warnings == [] and env.service.load("a").object_boxes[0].material == "glass"
    assert env.service.answer_question("a", "qm", "ceramicx", add_to_taxonomy=False) == ["material 'ceramicx' is not in your taxonomy lists."]
    env.service.answer_question("a", "qm", "ceramicx", add_to_taxonomy=True)
    assert "ceramicx" in env.service.taxonomy().materials
    env.service.answer_question("a", "ql", "reflection")
    assert env.service.load("a").laser_boxes[0].kind == "reflection"


def test_hit_point_answer_adds_a_measured_laser_box(env):
    env.label(env.blank, "blank", [])
    ls = env.service.load("blank")
    assert [q.id for q in ls.questions] == ["q_no_laser"]
    with pytest.raises(ReviewError, match="Click on the image"):
        env.service.answer_question("blank", "q_no_laser", "not a point")
    env.service.answer_question("blank", "q_no_laser", "140,110")
    row = env.db.get("blank")
    assert row.status == "pending"
    ls = env.service.load("blank")
    assert len(ls.laser_boxes) == 1 and ls.laser_boxes[0].kind == "first_incidence"
    assert ls.laser_boxes[0].order == 0 and "confidence" in ls.laser_boxes[0].metrics


def test_hit_point_snaps_to_a_detected_spot(env):
    env.label(env.path, "a", [env.good()])
    ls = env.service.load("a")
    delete_box(ls, "l1")
    made = add_laser_at_point(ls, env.path, env.settings, 143, 113)  # a few pixels off the real spot
    assert abs(made.box[0] + made.box[2] / 2 - 140) < 6 and made.box[2] < 15  # snapped, not the default square
    far = add_laser_at_point(ls, env.path, env.settings, 20, 200)
    assert far.box[2] == 15 and far.kind == "first_incidence"


# ---------------------------------------------------------------- re-run, PDF, upload support
def test_rerun_requeues_and_the_note_is_used_once(env):
    env.label(env.path, "a", [env.good()])
    env.service.rerun("a", "the object is a gel pen, not ballpoint")
    assert env.db.get("a").status == "queued"
    assert take_rerun_note(env.data, "a") == "the object is a gel pen, not ballpoint"
    assert take_rerun_note(env.data, "a") == ""
    env.db.claim_next()
    env.db.fail("a", "x")
    with pytest.raises(ReviewError):
        env.service.rerun("ghost")


def test_rerun_note_reaches_the_model(env):
    provider = ScriptedProvider([env.good()])
    tax = env.service.taxonomy()
    label_image(env.path, provider=provider, taxonomy=tax, settings=env.settings, data_dir=env.data,
                image_id="n", user_note="ignore the reflection on the left")
    assert "ignore the reflection on the left" in provider.calls[0][1].text
    entries = load_transcript(env.data / "transcripts" / "n.json")["entries"]
    assert entries[0]["type"] == "reviewer_note"


def test_pdf_contents_and_rebuild_keeps_addenda(env, tmp_path):
    env.label(env.path, "a", [env.good()])
    ls = env.service.load("a")
    ls.object_boxes[0].material = "glass"
    env.service.save_edit("a", ls)
    first = Path(env.db.get("a").pdf_path)
    size = first.stat().st_size
    env.service.rebuild_pdf("a")
    assert first.read_bytes()[:5] == b"%PDF-" and abs(first.stat().st_size - size) < 2000
    assert len(load_addenda(env.data / "pdfs" / "a.addenda.json")) == 1  # rebuilding adds no fake edit
    assert class_label(ls, "object", ls.object_boxes[0]).startswith("obj_pen_glass")
    bad = env.service.load("a")
    bad.object_boxes[0].material = "Not Valid"
    assert class_label(bad, "object", bad.object_boxes[0]).startswith("(invalid class")


def test_pdf_survives_missing_transcript_image_and_odd_text(env, tmp_path):
    env.label(env.path, "a", [env.good()])
    ls = env.service.load("a")
    ls.notes = "ünïcode ✓ <b>&</b>"
    out = build_pdf(ls, env.path, tmp_path / "x.pdf", transcript_path=tmp_path / "missing.json")
    assert out.read_bytes()[:5] == b"%PDF-"
    gone = tmp_path / "gone.png"
    out2 = build_pdf(ls, gone, tmp_path / "y.pdf")  # image file missing: noted in the PDF, not fatal
    assert out2.is_file()


def test_ensure_pdfs_rebuilds_stale_ones(env):
    env.label(env.path, "a", [env.good()])
    env.service.approve("a")
    pdf = Path(env.db.get("a").pdf_path)
    assert env.service.ensure_pdfs(["a"]) == []
    import os, time
    future = time.time() + 100
    os.utime(env.db.get("a").sidecar_path, (future, future))  # sidecar newer than the PDF
    assert not env.service.pdf_is_current(env.db.get("a"))
    env.service.ensure_pdfs(["a"])
    assert env.service.pdf_is_current(env.db.get("a")) and pdf.is_file()


def test_mark_uploaded_updates_queue_and_manifest(env):
    env.label(env.path, "a", [env.good()])
    env.service.approve("a")
    assert env.service.mark_uploaded(["a"]) == 1
    assert env.db.get("a").status == "uploaded"
    assert manifest.read_manifest(env.data / "manifest.csv")[0]["status"] == "uploaded"
    with pytest.raises(ReviewError):
        env.service.reject("a")  # uploaded is final


# ---------------------------------------------------------------- corrections as prompt examples
def test_corrections_dedupe_toggle_and_write_background(env, tmp_path):
    env.label(env.path, "a", [env.good()])
    for material in ("glass", "metal"):
        ls = env.service.load("a")
        ls.object_boxes[0].material = material
        env.service.save_edit("a", ls)
    path = env.data / "corrections.jsonl"
    items = examples.load_corrections(path)
    assert len(items) == 2
    assert examples.record_corrections(path, "a", "a.png", diff_label_sets(
        env.service.load("a"), env.service.load("a"))) == 0
    assert examples.set_included(path, [items[0]["id"]]) == 1 and examples.set_included(path, [items[0]["id"]]) == 0
    parts = tmp_path / "parts"
    parts.mkdir()
    (parts / "07_background.md").write_text("## Project background\n\nsome text\n")
    chosen = [i for i in examples.load_corrections(path) if i["included"]]
    assert examples.apply_to_background(parts, chosen) is True
    text_now = (parts / "07_background.md").read_text()
    assert text_now.count(examples.START) == 1 and chosen[0]["text"] in text_now and "some text" in text_now
    assert examples.apply_to_background(parts, chosen) is False  # same block again: no change, no duplicate
    examples.apply_to_background(parts, examples.load_corrections(path))  # now both
    assert (parts / "07_background.md").read_text().count("- o1") == 2
    examples.apply_to_background(parts, [])
    assert examples.START not in (parts / "07_background.md").read_text()
    assert examples.apply_to_background(parts, []) is False


def test_damaged_corrections_line_is_skipped(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text('{"id": "a", "text": "x"}\nnot json\n{"no_id": 1}\n')
    assert [i["id"] for i in examples.load_corrections(path)] == ["a"]


def test_add_addendum_appends(tmp_path):
    path = tmp_path / "x.addenda.json"
    add_addendum(path, ["one"], "edited")
    items = add_addendum(path, ["two"], "approved", "note")
    assert [i["changes"] for i in items] == [["one"], ["two"]] and items[1]["note"] == "note"
    assert json.loads(path.read_text())[1]["status"] == "approved"


def test_taxonomy_off_adds_new_words_when_a_label_is_approved(env):
    env.settings["labeling"]["taxonomy_match"] = "off"
    env.label(env.path, "a", [env.good(general_object="spaceship", material="velvet")])
    assert env.db.get("a").status == "pending"  # off: no taxonomy question
    assert env.service.taxonomy_problems(env.service.load("a")) == []
    env.service.approve("a")
    tax = env.service.taxonomy()
    assert "spaceship" in tax.objects and "velvet" in tax.materials


def test_ticking_add_to_taxonomy_always_adds(env):
    env.label(env.path, "a", [env.good()])  # default mode: material is not matched
    ls = env.service.load("a")
    from core.models import Question
    ls.questions = [Question("qm", "Which material?", "o1", "material", ["glass"], True)]
    from core.reports.sidecar import write_sidecar
    write_sidecar(ls, env.data / "sidecars")
    env.service.answer_question("a", "qm", "velvet", add_to_taxonomy=True)
    assert "velvet" in env.service.taxonomy().materials


def test_the_run_saves_new_words_to_the_yaml_files(env):
    from core.labeler.factory import build_label_fn
    env.settings["labeling"]["taxonomy_match"] = "off"
    tax = load_taxonomy(env.taxonomy_dir)
    fn = build_label_fn(env.settings, ScriptedProvider([env.good(general_object="spaceship")]), tax, env.data,
                        taxonomy_dir=env.taxonomy_dir)
    env.db.add_images([NewImage("a", env.path.name, str(env.path))])
    run_queue(env.db, fn, sidecar_dir=env.data / "sidecars")
    assert "spaceship" in (env.taxonomy_dir / "objects.yaml").read_text()
    assert "spaceship" in load_taxonomy(env.taxonomy_dir).objects


def test_clear_taxonomy_questions_frees_stuck_images_and_adds_the_words(env):
    env.settings["labeling"]["taxonomy_match"] = "all"  # how the old runs were made
    env.label(env.path, "a", [env.good(general_object="spaceship", material="velvet")])
    assert env.db.get("a").status == "needs_input"
    env.label(env.blank, "b", [env.good(general_object="unknown")])
    freed, added = env.service.clear_taxonomy_questions()
    assert freed == 1 and "object spaceship" in added and "material velvet" in added
    assert env.db.get("a").status == "pending"
    tax = env.service.taxonomy()
    assert "spaceship" in tax.objects and "velvet" in tax.materials and "unknown" not in tax.objects
    assert all(q.answer == "added to taxonomy" for q in env.service.load("a").questions if q.id.startswith("qt_"))
    assert env.service.clear_taxonomy_questions() == (0, [])  # nothing left to do


def test_local_images_are_reported_so_they_are_not_sent_to_roboflow(env):
    env.label(env.path, "a", [env.good()])
    env.service.approve("a")
    assert env.service.local_image_ids() == ["a"]  # queued from a file, not downloaded from Roboflow
