import pytest

from core.labeler.runner import run_queue
from core.models import ImageLabelSet
from core.providers.cost import SpendCapExceeded
from core.queue_db import NewImage, QueueDB


@pytest.fixture
def db(tmp_path):
    database = QueueDB(tmp_path / "q.db")
    database.add_images([NewImage(f"i{n}", f"i{n}.png", f"/x/i{n}.png") for n in range(1, 11)])
    yield database
    database.close()


def result(status="pending"):
    return ImageLabelSet("x", "x.png", "c", False, 0.1, status=status)


def run(db, fn, tmp_path, **kw):
    return run_queue(db, fn, sidecar_dir=tmp_path / "sc", **kw)


def test_processes_everything_and_reports(db, tmp_path):
    outcomes = iter(["pending", "needs_input", "failed"] + ["pending"] * 7)
    seen = []
    ticks = iter(range(0, 1000, 10))
    summary = run(db, lambda row: result(next(outcomes)), tmp_path, max_consecutive_failures=99,
                  on_progress=lambda s, row, res: seen.append((s.processed, row.image_id, res.status)),
                  clock=lambda: next(ticks))
    assert (summary.processed, summary.pending, summary.needs_input, summary.failed) == (10, 8, 1, 1)
    assert summary.stopped == "" and summary.queued_left == 0
    assert seen[0] == (1, "i1", "pending") and len(seen) == 10
    assert db.get("i1").sidecar_path == str(tmp_path / "sc" / "i1.json")


def test_eta_uses_average_time(db, tmp_path):
    ticks = iter([0, 10, 20, 20])  # start, after image 1, after image 2, final reading
    summary = run(db, lambda row: result(), tmp_path, limit=2, clock=lambda: next(ticks))
    assert summary.stopped == "limit" and summary.seconds_per_image == 10 and summary.queued_left == 8
    assert summary.eta_seconds == 80


def test_one_crash_does_not_end_the_run(db, tmp_path):
    def fn(row):
        if row.image_id == "i2":
            raise RuntimeError("corrupt image")
        return result()
    summary = run(db, fn, tmp_path)
    assert summary.processed == 10 and summary.failed == 1
    assert db.get("i2").status == "failed" and "RuntimeError: corrupt image" in db.get("i2").error


def test_failures_in_a_row_stop_the_run(db, tmp_path):
    summary = run(db, lambda row: result("failed"), tmp_path)
    assert summary.stopped == "too_many_failures" and summary.processed == 5 and summary.queued_left == 5


def test_success_resets_the_failure_streak(db, tmp_path):
    outcomes = iter(["failed"] * 4 + ["pending"] + ["failed"] * 4 + ["pending"])
    summary = run(db, lambda row: result(next(outcomes)), tmp_path)
    assert summary.stopped == "" and summary.processed == 10


def test_spend_cap_requeues_current_image_and_stops(db, tmp_path):
    def fn(row):
        if row.image_id == "i3":
            raise SpendCapExceeded("cap")
        return result()
    summary = run(db, fn, tmp_path)
    assert summary.stopped == "spend_cap" and summary.processed == 2
    assert db.get("i3").status == "queued" and db.counts()["queued"] == 8


def test_stop_flag_and_limit(db, tmp_path):
    calls = []
    summary = run(db, lambda row: (calls.append(1), result())[1], tmp_path, should_stop=lambda: len(calls) >= 3)
    assert summary.stopped == "stopped" and summary.processed == 3
    assert run(db, lambda row: result(), tmp_path, limit=2).stopped == "limit"


def test_resume_recovers_interrupted_image(db, tmp_path):
    db.claim_next()  # crash left i1 "labeling"
    summary = run(db, lambda row: result(), tmp_path)
    assert summary.processed == 10 and db.get("i1").status == "pending"


def test_ctrl_c_requeues_and_propagates(db, tmp_path):
    def fn(row):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        run(db, fn, tmp_path)
    assert db.get("i1").status == "queued"
