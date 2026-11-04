"""data/manifest.csv: joins PDFs and sidecars to the dataset after export (spec section 8, 12)."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from core.models import ImageLabelSet

COLUMNS = ["image_id", "image_name", "pdf_path", "sidecar_path", "prompt_version", "provider", "model",
           "status", "labeled_at"]


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)  # never leave a half-written manifest
    return path


def update_manifest(label_set: ImageLabelSet, path: str | Path, sidecar_path: str,
                    labeled_at: str | None = None) -> Path:
    """Add or replace this image's row, keeping other rows in order."""
    path = Path(path)
    row = {"image_id": label_set.image_id, "image_name": label_set.image_name,
           "pdf_path": label_set.pdf_path or "", "sidecar_path": str(sidecar_path),
           "prompt_version": label_set.prompt_version, "provider": label_set.provider, "model": label_set.model,
           "status": label_set.status,
           "labeled_at": labeled_at or datetime.now(timezone.utc).isoformat(timespec="seconds")}
    rows, replaced = [], False
    for old in read_manifest(path):
        if old["image_id"] == label_set.image_id:
            rows.append(row)
            replaced = True
        else:
            rows.append(old)
    if not replaced:
        rows.append(row)
    return _write(path, rows)


def set_status(path: str | Path, image_id: str, status: str) -> bool:
    """Change only the status of an existing row (for example after upload). False if no such row."""
    path = Path(path)
    rows = read_manifest(path)
    found = False
    for row in rows:
        if row["image_id"] == image_id:
            row["status"], found = status, True
    if found:
        _write(path, rows)
    return found
