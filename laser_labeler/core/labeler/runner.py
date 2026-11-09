"""Sequential queue worker (spec sections 5 and 11). One image at a time, results saved at once.

The GUI will wrap this in a QThread; the CLI calls it directly. Everything the loop needs
from outside is passed in, so it is easy to test with a fake labeler.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.models import ImageLabelSet
from core.providers.base import RateLimitError
from core.providers.cost import SpendCapExceeded
from core.queue_db import QueueDB, QueueRow
from core.reports.sidecar import sidecar_path

STOP_TEXT = {"stopped": "stopped", "limit": "reached the limit", "spend_cap": "daily spend cap reached (the image was re-queued)",
             "too_many_failures": "stopped after several failures in a row: check your key, model name and server, "
                                  "then retry the failed images",
             "too_many_timeouts": "stopped because several images in a row timed out: the model is too slow for these "
                                  "settings (see the failure message for what to change), then retry the failed images",
             "error": "stopped because of an unexpected error",
             "rate_limit": "stopped because the provider's rate limit or quota was reached (the image was put back in "
                           "the queue); start the run again when the limit has reopened"}

Progress = Callable[["RunSummary", QueueRow, ImageLabelSet | None], None]


@dataclass
class LimitInfo:
    """A provider limit that stopped one image. `retry_after` is the provider's own hint in seconds, if it gave one."""
    message: str
    retry_after: float | None
    image_id: str
    times: int  # how often this run has hit a limit, counting this one
    consecutive: int = 1  # limit hits in a row with no image finished in between, counting this one


OnLimit = Callable[[LimitInfo], bool]  # blocks until the user decides; True = try again, False = end the run


@dataclass
class RunSummary:
    processed: int = 0
    pending: int = 0
    needs_input: int = 0
    failed: int = 0
    stopped: str = ""  # empty | stopped | limit | spend_cap | too_many_failures | too_many_timeouts | rate_limit
    seconds: float = 0.0
    seconds_per_image: float = 0.0
    queued_left: int = 0

    @property
    def eta_seconds(self) -> float:
        return self.seconds_per_image * self.queued_left


def failure_kind(error: str | None) -> str:
    """"timeout" (the model was too slow for one image), "image" (this image alone did not fit or was cut off) or
    "setup" (anything else, which usually means the same thing will happen to every image)."""
    text = (error or "").lower()
    if "request timed out" in text:
        return "timeout"
    if "ran out of tokens" in text:
        return "image"
    return "setup"


def run_queue(
    db: QueueDB,
    label_fn: Callable[[QueueRow], ImageLabelSet],
    *,
    sidecar_dir: str | Path,
    should_stop: Callable[[], bool] = lambda: False,
    limit: int | None = None,
    max_consecutive_failures: int = 5,
    max_consecutive_timeouts: int = 10,
    on_progress: Progress | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_limit: OnLimit | None = None,
) -> RunSummary:
    """Label queued images until the queue is empty, `limit` is reached, or something stops the run.

    A run of failures in a row (wrong key, server down) stops it, so 2,000 images are not marked
    failed one by one. A spend-cap hit puts the current image back in the queue and stops.

    One image that stalls or does not fit is passed over, not a reason to stop: a timeout or a "ran out of tokens"
    failure marks that image failed and the run moves on to the next. Timeouts have their own, looser limit
    (`max_consecutive_timeouts`) because a hung server times out on every image; "ran out of tokens" never counts.

    A rate-limit or quota error (after the provider's own retries) is not a failure of the image: it goes back in the
    queue and `on_limit` is asked. It blocks while the user waits for the limit to reopen and answers True to carry on
    with the same image or False to end the run. Without `on_limit` the run stops with `stopped == "rate_limit"`.
    """
    summary, started, streak, timeouts, limit_hits, limit_streak = RunSummary(), clock(), 0, 0, 0, 0
    db.recover_interrupted()
    while True:
        if should_stop():
            summary.stopped = "stopped"
        elif limit is not None and summary.processed >= limit:
            summary.stopped = "limit"
        elif streak >= max_consecutive_failures:
            summary.stopped = "too_many_failures"
        elif timeouts >= max_consecutive_timeouts:
            summary.stopped = "too_many_timeouts"
        if summary.stopped:
            break
        row = db.claim_next()
        if row is None:
            break
        try:
            result = label_fn(row)
        except SpendCapExceeded:
            db.requeue([row.image_id])
            summary.stopped = "spend_cap"
            break
        except RateLimitError as exc:
            db.requeue([row.image_id])
            limit_hits += 1
            limit_streak += 1
            if on_limit is not None and on_limit(LimitInfo(str(exc), exc.retry_after, row.image_id, limit_hits,
                                                           limit_streak)) \
                    and not should_stop():
                continue
            summary.stopped = "stopped" if should_stop() else "rate_limit"
            break
        except KeyboardInterrupt:
            db.requeue([row.image_id])
            raise
        except Exception as exc:  # one bad image must not end a 2,000-image run
            db.fail(row.image_id, f"{type(exc).__name__}: {exc}")
            result, status = None, "failed"
        else:
            status = db.finish(row.image_id, result, str(sidecar_path(sidecar_dir, row.image_id)))
        summary.processed += 1
        setattr(summary, status, getattr(summary, status) + 1)
        limit_streak = 0
        kind = failure_kind(result.notes if result is not None else db.get(row.image_id).error) \
            if status == "failed" else ""
        if status != "failed":
            streak = timeouts = 0
        elif kind == "timeout":
            timeouts += 1
        elif kind == "setup":
            streak, timeouts = streak + 1, 0
        summary.seconds = clock() - started
        summary.seconds_per_image = summary.seconds / summary.processed
        summary.queued_left = db.counts()["queued"]
        if on_progress:
            on_progress(summary, row, result)
    summary.seconds = clock() - started
    summary.queued_left = db.counts()["queued"]
    return summary
