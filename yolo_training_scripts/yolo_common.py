"""
yolo_common.py

Small shared helpers used by train_yolo.py, evaluate.py, and
resplit_dataset.py, so the three scripts agree on how to find the right
data.yaml, what counts as an image file, and how to map a class id back to
the PDF object it came from.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

PDFNAME_RE = re.compile(r"Pdfname\s+(\S+)\s*$")


def load_data_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_data_yaml(dataset_dir: Path | None, explicit_data: Path | None) -> Path:
    """
    Prefer dataset/data.corrected.yaml (the output of build_dataset.py) over
    the raw dataset/data.yaml, unless the caller passed an explicit path.
    """
    if explicit_data is not None:
        return explicit_data
    if dataset_dir is None:
        raise SystemExit("Provide either --dataset-dir or --data")
    corrected = dataset_dir / "data.corrected.yaml"
    raw = dataset_dir / "data.yaml"
    if corrected.exists():
        print(f"[info] using reconciled config: {corrected}")
        print("        (run build_dataset.py again any time the PDFs change)")
        return corrected
    if raw.exists():
        print(f"[warn] no data.corrected.yaml found -- using raw {raw} as-is.")
        print("        Run build_dataset.py first to reconcile class names against the")
        print("        latest PDF corrections before training/evaluating on real data.")
        return raw
    raise SystemExit(f"Neither data.corrected.yaml nor data.yaml found under {dataset_dir}")


def extract_pdf_id_from_name(name: str) -> str | None:
    m = PDFNAME_RE.search(name)
    return m.group(1).strip() if m else None


def class_id_to_pdf_id_map(names: list[str]) -> dict[int, str | None]:
    """Map each class index to the PDF object id encoded in its name (or None if unparseable)."""
    return {idx: extract_pdf_id_from_name(name) for idx, name in enumerate(names)}


def find_split_dirs(dataset_dir: Path) -> dict[str, Path]:
    """Return {split_name: split_dir} for whichever of train/valid/test actually exist."""
    found = {}
    for split in ("train", "valid", "test"):
        d = dataset_dir / split
        if (d / "images").exists() and (d / "labels").exists():
            found[split] = d
    return found


def list_image_label_pairs(split_dir: Path) -> list[tuple[Path, Path]]:
    """Return [(image_path, label_path), ...] for every image with a matching label file
    (label_path may not exist for a background/no-object image -- caller decides how to handle)."""
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        pairs.append((img_path, lbl_dir / f"{img_path.stem}.txt"))
    return pairs


def read_label_class_ids(label_path: Path) -> list[int]:
    """Return the list of class ids present in a YOLO label file (empty list if missing/background)."""
    if not label_path.exists():
        return []
    ids = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            ids.append(int(parts[0]))
        except (ValueError, IndexError):
            continue
    return ids
