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


# ---------------------------------------------------------------- provider limits: pause and ask
from core.providers.base import RateLimitError  # noqa: E402


def limited_then_ok(failures, retry_after=None):
    """A labeling function that hits the provider limit `failures` times, then works."""
    state = {"left": failures}

    def fn(row):
        if state["left"] > 0:
            state["left"] -= 1
            raise RateLimitError("429 RESOURCE_EXHAUSTED: quota exceeded for this day", retry_after)
        return result()
    return fn


def test_a_rate_limit_puts_the_image_back_asks_and_carries_on(db, tmp_path):
    asked = []
    summary = run(db, limited_then_ok(2, retry_after=90), tmp_path, limit=3,
                  on_limit=lambda info: asked.append(info) or True)
    assert [a.times for a in asked] == [1, 2] and asked[0].retry_after == 90 and asked[0].image_id == "i1"
    assert "quota exceeded" in asked[0].message
    assert summary.processed == 3 and summary.failed == 0 and summary.stopped == "limit"  # the limit did not count
    assert db.counts()["failed"] == 0 and db.get("i1").status == "pending"  # the same image was tried again


def test_declining_to_resume_stops_with_the_image_still_queued(db, tmp_path):
    summary = run(db, limited_then_ok(1), tmp_path, on_limit=lambda info: False)
    assert summary.stopped == "rate_limit" and summary.processed == 0
    assert db.get("i1").status == "queued" and db.counts()["failed"] == 0


def test_without_anyone_to_ask_the_run_stops_cleanly(db, tmp_path):
    summary = run(db, limited_then_ok(1), tmp_path)
    assert summary.stopped == "rate_limit" and db.get("i1").status == "queued" and summary.queued_left == 10


def test_a_stop_request_while_waiting_reads_as_stopped(db, tmp_path):
    flag = {"stop": False}

    def on_limit(info):
        flag["stop"] = True
        return True  # even an answer of yes is ignored once the user pressed Stop

    summary = run(db, limited_then_ok(1), tmp_path, on_limit=on_limit, should_stop=lambda: flag["stop"])
    assert summary.stopped == "stopped" and db.get("i1").status == "queued"


def test_other_errors_are_still_recorded_as_failures(db, tmp_path):
    def boom(row):
        raise RuntimeError("bad image")
    summary = run(db, boom, tmp_path, limit=2, on_limit=lambda info: pytest.fail("not a limit"), max_consecutive_failures=99)
    assert summary.failed == 2


# ------------------------------------------------------------------ a stalled image is passed over
def failed_with(notes):
    r = result("failed")
    r.notes = notes
    return r


TIMED_OUT = "failed: Request timed out after 300 seconds: APITimeoutError: Request timed out."
CUT_OFF = "failed: The model ran out of tokens: its reply was cut off after 2000 tokens, with the request at 6000 tokens."


def test_timeouts_do_not_trip_the_five_failures_guard_but_have_their_own_limit(db, tmp_path):
    summary = run(db, lambda row: failed_with(TIMED_OUT), tmp_path)
    assert summary.processed == 10 and summary.failed == 10 and summary.stopped == "too_many_timeouts"
    # with a tighter limit the stop reason shows up before the queue is empty
    db.requeue([r.image_id for r in db.list("failed")])
    summary = run(db, lambda row: failed_with(TIMED_OUT), tmp_path, max_consecutive_timeouts=3)
    assert summary.stopped == "too_many_timeouts" and summary.processed == 3 and summary.queued_left == 7


def test_a_stalled_image_is_marked_failed_and_the_run_moves_on(db, tmp_path):
    def fn(row):
        return failed_with(TIMED_OUT) if row.image_id == "i2" else result()
    summary = run(db, fn, tmp_path)
    assert summary.stopped == "" and summary.processed == 10 and summary.failed == 1
    assert db.get("i2").status == "failed" and "timed out" in db.get("i2").error
    assert db.get("i3").status == "pending"


def test_running_out_of_tokens_never_counts_toward_a_stop(db, tmp_path):
    summary = run(db, lambda row: failed_with(CUT_OFF), tmp_path)
    assert summary.stopped == "" and summary.processed == 10 and summary.failed == 10


def test_a_timeout_streak_is_broken_by_a_success_and_by_another_failure(db, tmp_path):
    outcomes = iter([TIMED_OUT] * 2 + [None] + [TIMED_OUT] * 2 + ["failed: some other thing"] + [TIMED_OUT] * 2 + [None] * 2)
    summary = run(db, lambda row: (lambda n: result() if n is None else failed_with(n))(next(outcomes)), tmp_path,
                  max_consecutive_timeouts=3)
    assert summary.stopped == "" and summary.processed == 10


def test_limit_info_counts_hits_in_a_row_and_resets_after_an_image(db, tmp_path):
    seen = []
    script = iter(["limit", "limit", "ok", "limit", "ok"])

    def fn(row):
        step = next(script, "ok")
        if step == "limit":
            raise RateLimitError("429 quota", None)
        return result()

    run(db, fn, tmp_path, limit=3, on_limit=lambda info: seen.append((info.times, info.consecutive)) or True)
    assert seen == [(1, 1), (2, 2), (3, 1)]
