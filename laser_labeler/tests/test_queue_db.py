import threading

import pytest

from core.models import STATUSES, ImageLabelSet, LaserBox, ObjectBox, Question
from core.queue_db import InvalidTransition, NewImage, QueueDB, overall_confidence


@pytest.fixture
def db(tmp_path):
    database = QueueDB(tmp_path / "data" / "queue.db")
    yield database
    database.close()


def images(n, start=1):
    return [NewImage(f"img{i}", f"img{i}.png", f"/x/img{i}.png", "local", ["camera:a"]) for i in range(start, start + n)]


def label_set(status="pending", obj_conf=0.9, laser_conf=0.8, questions=(), notes=""):
    return ImageLabelSet(
        "x", "x.png", "cam", False, 0.1, status=status, notes=notes, questions=list(questions),
        laser_boxes=[LaserBox("l1", "first_incidence", (1, 1, 5, 5), 0, metrics={"confidence": laser_conf})],
        object_boxes=[ObjectBox("o1", (0, 0, 9, 9), "pen", "glass", "clear", "first", confidence=obj_conf)])


def labeled(db, image_id, **kw):
    db.add_images(images(1, int(image_id[3:]))) if db.get(image_id) is None else None
    row = db.claim_next()
    assert row.image_id == image_id
    db.finish(image_id, label_set(**kw), f"/s/{image_id}.json")


def test_add_orders_and_ignores_duplicates(db):
    assert db.add_images(images(3)) == 3
    assert db.add_images(images(3, 2)) == 1  # img2 and img3 known, img4 new
    assert [r.image_id for r in db.list()] == ["img1", "img2", "img3", "img4"]
    row = db.get("img1")
    assert row.status == "queued" and row.tags == ["camera:a"] and row.source == "local"
    assert db.get("nope") is None


def test_counts_include_every_status(db):
    db.add_images(images(2))
    counts = db.counts()
    assert list(counts) == list(STATUSES) and counts["queued"] == 2 and counts["uploaded"] == 0


def test_claim_next_is_sequential_and_ends(db):
    db.add_images(images(2))
    a, b = db.claim_next(), db.claim_next()
    assert (a.image_id, b.image_id) == ("img1", "img2") and a.status == "labeling" and a.attempts == 1
    assert db.claim_next() is None


def test_claims_are_exclusive_across_threads(tmp_path):
    shared = QueueDB(tmp_path / "q.db")
    shared.add_images(images(40))
    taken, lock = [], threading.Lock()

    def worker():
        while (row := shared.claim_next()) is not None:
            with lock:
                taken.append(row.image_id)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(taken) == sorted(f"img{i}" for i in range(1, 41)) and len(taken) == len(set(taken))


def test_finish_records_result(db):
    db.add_images(images(1))
    db.claim_next()
    status = db.finish("img1", label_set(obj_conf=0.9, laser_conf=0.6, questions=[Question("q", "?")]), "/s/img1.json")
    row = db.get("img1")
    assert status == "pending" == row.status and row.confidence == 0.6  # weakest link
    assert row.question_count == 1 and row.sidecar_path == "/s/img1.json"


def test_finish_failed_keeps_error_and_needs_labeling_state(db):
    db.add_images(images(1))
    with pytest.raises(InvalidTransition):
        db.finish("img1", label_set(), "/s")  # still queued
    db.claim_next()
    db.finish("img1", label_set("failed", notes="server down"), "/s")
    assert db.get("img1").error == "server down" and db.get("img1").status == "failed"
    with pytest.raises(InvalidTransition):
        db.finish("img1", label_set("approved"), "/s")


def test_overall_confidence_without_boxes_is_zero():
    assert overall_confidence(ImageLabelSet("x", "x", "c", False, 0.0)) == 0.0


def test_transitions(db):
    db.add_images(images(1))
    with pytest.raises(InvalidTransition):
        db.set_status("img1", "approved")
    labeled(db, "img1")
    db.set_status("img1", "approved")
    db.set_status("img1", "uploaded")
    for target in ("queued", "pending", "approved"):
        with pytest.raises(InvalidTransition):
            db.set_status("img1", target)  # uploaded is final
    with pytest.raises(ValueError):
        db.set_status("img1", "bogus")
    with pytest.raises(KeyError):
        db.set_status("ghost", "queued")


def test_requeue_retry_and_rerun(db):
    db.add_images(images(2))
    labeled(db, "img1", status="failed", notes="boom")
    labeled(db, "img2")
    assert db.requeue(["img1", "img2"]) == 2
    assert db.counts()["queued"] == 2
    assert db.get("img1").error is None


def test_recover_interrupted_after_crash(tmp_path):
    first = QueueDB(tmp_path / "q.db")
    first.add_images(images(3))
    first.claim_next()
    first.close()  # "crash" with img1 mid-labeling
    again = QueueDB(tmp_path / "q.db")
    assert again.get("img1").status == "labeling"
    assert again.recover_interrupted() == 1
    assert again.claim_next().image_id == "img1"  # resumes where it left off


def test_bulk_approve_rules(db):
    db.add_images(images(4))
    labeled(db, "img1", obj_conf=0.95, laser_conf=0.95)                          # approvable
    labeled(db, "img2", obj_conf=0.5)                                            # too unsure
    labeled(db, "img3", obj_conf=0.99, laser_conf=0.99, questions=[Question("q", "?")])  # open question
    labeled(db, "img4", status="needs_input", obj_conf=0.99, laser_conf=0.99)
    assert db.bulk_approve(0.9) == ["img1"]
    assert [db.get(f"img{i}").status for i in range(1, 5)] == ["approved", "pending", "pending", "needs_input"]


def test_list_filters_and_sorts(db):
    db.add_images(images(3))
    labeled(db, "img1", obj_conf=0.9)
    labeled(db, "img2", obj_conf=0.3)
    labeled(db, "img3", obj_conf=0.6)
    assert [r.image_id for r in db.list("pending", "confidence_asc")] == ["img2", "img3", "img1"]
    assert [r.image_id for r in db.list("pending", "confidence_desc", limit=1)] == ["img1"]
    assert [r.image_id for r in db.list(["pending", "failed"], "order", offset=1)] == ["img2", "img3"]
    for bad in (lambda: db.list(sort="x"), lambda: db.list("nope")):
        with pytest.raises(ValueError):
            bad()


# ------------------------------------------------------------------ removing images
def test_remove_deletes_only_the_named_rows(db):
    db.add_images(images(4))
    assert db.remove(["img2", "img4", "nope"]) == 2
    assert [r.image_id for r in db.list()] == ["img1", "img3"]
    assert db.remove([]) == 0


def test_clear_by_status_and_default_keeps_uploaded(db):
    db.add_images(images(5))
    labeled(db, "img1")                   # pending
    labeled(db, "img2", status="failed", notes="boom")
    db.set_status("img1", "approved")
    db.set_status("img1", "uploaded")
    assert db.clear(["failed"]) == 1 and db.get("img2") is None
    assert db.clear(["queued"]) == 3 and db.counts()["queued"] == 0
    db.add_images(images(1, 9))
    assert db.clear() == 1                # default: everything not uploaded
    assert db.get("img1") is not None and db.get("img1").status == "uploaded"
    assert db.clear(["uploaded"]) == 1 and db.list() == []


def test_clear_rejects_unknown_status_and_empty_list_is_a_no_op(db):
    db.add_images(images(2))
    with pytest.raises(ValueError):
        db.clear(["bogus"])
    assert db.clear([]) == 0 and len(db.list()) == 2


def test_removed_images_can_be_added_again(db):
    db.add_images(images(1))
    db.remove(["img1"])
    assert db.add_images(images(1)) == 1 and db.get("img1").status == "queued"
