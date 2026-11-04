"""Per-image sidecar JSON (spec section 12): the full ImageLabelSet, numbers and links included."""
from __future__ import annotations

import re
from pathlib import Path

from core.models import ImageLabelSet

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_id(image_id: str) -> str:
    """File-system safe form of an image id (also used for transcript and PDF names)."""
    cleaned = _UNSAFE.sub("_", image_id).strip("._")
    if not cleaned:
        raise ValueError(f"image id {image_id!r} has no usable characters")
    return cleaned


def sidecar_path(directory: str | Path, image_id: str) -> Path:
    return Path(directory) / f"{safe_id(image_id)}.json"


def write_sidecar(label_set: ImageLabelSet, directory: str | Path) -> Path:
    path = sidecar_path(directory, label_set.image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(label_set.to_json(), encoding="utf-8")
    tmp.replace(path)  # never leave a half-written sidecar after a crash
    return path


def read_sidecar(path: str | Path) -> ImageLabelSet:
    return ImageLabelSet.from_json(Path(path).read_text(encoding="utf-8"))
