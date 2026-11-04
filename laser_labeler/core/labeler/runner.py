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
from core.providers.cost import SpendCapExceeded
from core.queue_db import QueueDB, QueueRow
from core.reports.sidecar import sidecar_path

STOP_TEXT = {"stopped": "stopped", "limit": "reached the limit", "spend_cap": "daily spend cap reached (the image was re-queued)",
             "too_many_failures": "stopped after several failures in a row: check your key, model name and server, "
                                  "then retry the failed images", "error": "stopped because of an unexpected error"}

Progress = Callable[["RunSummary", QueueRow, ImageLabelSet | None], None]


@dataclass
class RunSummary:
    processed: int = 0
    pending: int = 0
    needs_input: int = 0
    failed: int = 0
    stopped: str = ""  # empty | stopped | limit | spend_cap | too_many_failures
    seconds: float = 0.0
    seconds_per_image: float = 0.0
    queued_left: int = 0

    @property
    def eta_seconds(self) -> float:
        return self.seconds_per_image * self.queued_left


def run_queue(
    db: QueueDB,
    label_fn: Callable[[QueueRow], ImageLabelSet],
    *,
    sidecar_dir: str | Path,
    should_stop: Callable[[], bool] = lambda: False,
    limit: int | None = None,
    max_consecutive_failures: int = 5,
    on_progress: Progress | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunSummary:
    """Label queued images until the queue is empty, `limit` is reached, or something stops the run.

    A run of failures in a row (wrong key, server down) stops it, so 2,000 images are not marked
    failed one by one. A spend-cap hit puts the current image back in the queue and stops.
    """
    summary, started, streak = RunSummary(), clock(), 0
    db.recover_interrupted()
    while True:
        if should_stop():
            summary.stopped = "stopped"
        elif limit is not None and summary.processed >= limit:
            summary.stopped = "limit"
        elif streak >= max_consecutive_failures:
            summary.stopped = "too_many_failures"
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
        streak = streak + 1 if status == "failed" else 0
        summary.seconds = clock() - started
        summary.seconds_per_image = summary.seconds / summary.processed
        summary.queued_left = db.counts()["queued"]
        if on_progress:
            on_progress(summary, row, result)
    summary.seconds = clock() - started
    summary.queued_left = db.counts()["queued"]
    return summary
