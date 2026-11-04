"""SQLite queue: every image has a status and the app resumes after a crash (spec section 11).

queued -> labeling -> (needs_input | failed | pending) -> (approved | edited | rejected) -> uploaded

Extra moves are allowed so work can be redone (re-run, retry, undo an approval); only
`uploaded` is final. The worker takes one `queued` image at a time (claim_next).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from core.models import STATUSES, ImageLabelSet

ALLOWED: dict[str, set[str]] = {
    "queued": {"labeling"},
    "labeling": {"needs_input", "failed", "pending", "queued"},
    "needs_input": {"queued", "rejected", "pending"},  # pending: the user answered every question
    "failed": {"queued", "rejected"},
    "pending": {"approved", "edited", "rejected", "queued"},
    "approved": {"uploaded", "edited", "pending", "rejected", "queued"},
    "edited": {"uploaded", "approved", "pending", "rejected", "queued"},
    "rejected": {"queued", "pending"},
    "uploaded": set(),
}
SORTS = {"order": "order_idx", "confidence_asc": "confidence IS NULL, confidence, order_idx",
         "confidence_desc": "confidence IS NULL, confidence DESC, order_idx", "recent": "updated_at DESC"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'local',
    status TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]', order_idx INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0, error TEXT, sidecar_path TEXT, transcript_path TEXT, pdf_path TEXT,
    prompt_version TEXT, provider TEXT, model TEXT, run_mode TEXT, confidence REAL,
    question_count INTEGER NOT NULL DEFAULT 0, added_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_images_status ON images (status, order_idx);
"""


class InvalidTransition(ValueError):
    """A status change that the workflow does not allow."""


@dataclass
class NewImage:
    image_id: str
    name: str
    path: str
    source: str = "local"  # "local" or "roboflow"
    tags: Sequence[str] = ()


@dataclass
class QueueRow:
    image_id: str
    name: str
    path: str
    source: str
    status: str
    tags: list[str] = field(default_factory=list)
    order_idx: int = 0
    attempts: int = 0
    error: str | None = None
    sidecar_path: str | None = None
    transcript_path: str | None = None
    pdf_path: str | None = None
    prompt_version: str | None = None
    provider: str | None = None
    model: str | None = None
    run_mode: str | None = None
    confidence: float | None = None
    question_count: int = 0
    added_at: str = ""
    updated_at: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def overall_confidence(label_set: ImageLabelSet) -> float:
    """Weakest link: the lowest confidence of any box. 0 when there are no boxes."""
    values = [o.confidence for o in label_set.object_boxes]
    values += [ls.metrics.get("confidence", 0.0) for ls in label_set.laser_boxes]
    return float(min(values)) if values else 0.0


def _row(record: sqlite3.Row) -> QueueRow:
    data = dict(record)
    data["tags"] = json.loads(data["tags"])
    return QueueRow(**data)


class QueueDB:
    """Thread-safe: the worker thread and the GUI thread can share one instance."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- adding
    def add_images(self, items: Iterable[NewImage]) -> int:
        """Add new images as `queued`. Existing image ids are left alone. Returns how many were added."""
        added = 0
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                order = self._db.execute("SELECT COALESCE(MAX(order_idx), 0) FROM images").fetchone()[0]
                for item in items:
                    order += 1
                    cur = self._db.execute(
                        "INSERT OR IGNORE INTO images (image_id, name, path, source, status, tags, order_idx, "
                        "added_at, updated_at) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)",
                        (item.image_id, item.name, str(item.path), item.source, json.dumps(list(item.tags)),
                         order, _now(), _now()))
                    added += cur.rowcount
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return added

    # ---- reading
    def get(self, image_id: str) -> QueueRow | None:
        with self._lock:
            record = self._db.execute("SELECT * FROM images WHERE image_id = ?", (image_id,)).fetchone()
        return _row(record) if record else None

    def counts(self) -> dict[str, int]:
        with self._lock:
            found = dict(self._db.execute("SELECT status, COUNT(*) FROM images GROUP BY status").fetchall())
        return {status: found.get(status, 0) for status in STATUSES}

    def list(self, status: str | Sequence[str] | None = None, sort: str = "order",
             limit: int | None = None, offset: int = 0) -> list[QueueRow]:
        if sort not in SORTS:
            raise ValueError(f"sort must be one of {sorted(SORTS)}")
        statuses = [status] if isinstance(status, str) else list(status or [])
        for s in statuses:
            if s not in STATUSES:
                raise ValueError(f"unknown status {s!r}")
        where = f"WHERE status IN ({','.join('?' * len(statuses))})" if statuses else ""
        sql = f"SELECT * FROM images {where} ORDER BY {SORTS[sort]} LIMIT ? OFFSET ?"
        with self._lock:
            records = self._db.execute(sql, (*statuses, -1 if limit is None else limit, offset)).fetchall()
        return [_row(r) for r in records]

    # ---- state changes
    def _update(self, image_id: str, **fields) -> None:
        fields["updated_at"] = _now()
        assignments = ", ".join(f"{k} = ?" for k in fields)
        self._db.execute(f"UPDATE images SET {assignments} WHERE image_id = ?", (*fields.values(), image_id))

    def set_status(self, image_id: str, status: str, *, error: str | None = None) -> None:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        with self._lock:
            row = self.get(image_id)
            if row is None:
                raise KeyError(image_id)
            if status not in ALLOWED[row.status]:
                raise InvalidTransition(f"{image_id}: {row.status} -> {status} is not allowed")
            self._update(image_id, status=status, error=error)

    def claim_next(self) -> QueueRow | None:
        """Take the oldest `queued` image and mark it `labeling`. None when the queue is empty."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                record = self._db.execute("SELECT * FROM images WHERE status = 'queued' "
                                          "ORDER BY order_idx LIMIT 1").fetchone()
                if record is not None:
                    self._db.execute("UPDATE images SET status = 'labeling', attempts = attempts + 1, "
                                     "updated_at = ? WHERE image_id = ?", (_now(), record["image_id"]))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return self.get(record["image_id"]) if record else None

    def finish(self, image_id: str, label_set: ImageLabelSet, sidecar_path: str) -> str:
        """Record a labeling result. Returns the new status (needs_input, failed or pending)."""
        status = label_set.status
        if status not in ("needs_input", "failed", "pending"):
            raise InvalidTransition(f"a labeling run cannot end in {status!r}")
        with self._lock:
            row = self.get(image_id)
            if row is None:
                raise KeyError(image_id)
            if row.status != "labeling":
                raise InvalidTransition(f"{image_id} is {row.status}, not labeling")
            self._update(
                image_id, status=status, sidecar_path=sidecar_path, transcript_path=label_set.transcript_path,
                pdf_path=label_set.pdf_path, prompt_version=label_set.prompt_version, provider=label_set.provider,
                model=label_set.model, run_mode=label_set.run_mode, confidence=overall_confidence(label_set),
                question_count=sum(q.answer is None for q in label_set.questions),
                error=label_set.notes if status == "failed" else None)
        return status

    def record_review(self, image_id: str, label_set: ImageLabelSet, status: str | None = None,
                      sidecar_path: str | None = None) -> str:
        """Store the numbers from a reviewed label set (confidence, open questions, PDF) and optionally move
        the image to `status`. Returns the status the image now has."""
        with self._lock:
            row = self.get(image_id)
            if row is None:
                raise KeyError(image_id)
            if status is not None and status != row.status and status not in ALLOWED[row.status]:
                raise InvalidTransition(f"{image_id}: {row.status} -> {status} is not allowed")
            fields = dict(confidence=overall_confidence(label_set), pdf_path=label_set.pdf_path,
                          question_count=sum(q.answer is None for q in label_set.questions))
            if sidecar_path:
                fields["sidecar_path"] = sidecar_path
            if status is not None:
                fields["status"] = status
                if status != "failed":
                    fields["error"] = None
            self._update(image_id, **fields)
            return status or row.status

    def fail(self, image_id: str, error: str) -> None:
        self.set_status(image_id, "failed", error=error[:500])

    def requeue(self, image_ids: Iterable[str]) -> int:
        """Send images back to `queued` (retry, re-run, or after the user answered a question)."""
        count = 0
        for image_id in image_ids:
            self.set_status(image_id, "queued")
            count += 1
        return count

    def recover_interrupted(self) -> int:
        """After a crash, pause or reboot: anything left `labeling` goes back to `queued`."""
        with self._lock:
            cur = self._db.execute("UPDATE images SET status = 'queued', updated_at = ? WHERE status = 'labeling'",
                                   (_now(),))
        return cur.rowcount

    def bulk_approve(self, min_confidence: float) -> list[str]:
        """Approve `pending` images with no open questions and confidence at or above the threshold."""
        with self._lock:
            ids = [r["image_id"] for r in self._db.execute(
                "SELECT image_id FROM images WHERE status = 'pending' AND question_count = 0 "
                "AND confidence >= ?", (min_confidence,))]
            for image_id in ids:
                self._update(image_id, status="approved")
        return ids
