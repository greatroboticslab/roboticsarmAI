"""Find image files on disk and give them queue ids (shared by the CLI and the Select images tab)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from core.queue_db import NewImage, QueueDB
from core.reports.sidecar import safe_id

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_files(paths: Iterable[str | Path]) -> list[Path]:
    """Image files in the given files and folders (folders are searched including subfolders), in a stable order."""
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).resolve()
        found = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_TYPES) if path.is_dir() else [path]
        files += [p for p in found if p.is_file() and p.suffix.lower() in IMAGE_TYPES]
    return files


def make_image_id(db: QueueDB, path: Path, batch: dict[str, Path] | None = None) -> str:
    """File stem, made unique if a different file with the same stem is already queued or is
    earlier in this same batch (`batch` maps ids handed out so far to their files)."""
    batch = batch if batch is not None else {}
    base = safe_id(path.stem)
    known = db.get(base)
    clash = (known is not None and Path(known.path) != path) or (base in batch and batch[base] != path)
    image_id = f"{base}-{hashlib.sha1(str(path).encode()).hexdigest()[:6]}" if clash else base
    batch[image_id] = path
    return image_id


def new_images_for(db: QueueDB, files: Iterable[Path], tags: Iterable[str] = ()) -> list[NewImage]:
    batch: dict[str, Path] = {}
    tag_list = list(tags)
    return [NewImage(make_image_id(db, p, batch), p.name, str(p), "local", tag_list) for p in files]
