"""data/measurements.csv: one row per laser box, to join with the dataset after export (spec 12)."""
from __future__ import annotations

import csv
from pathlib import Path

from core.models import ImageLabelSet

BASE_COLUMNS = [
    "image_id", "image_name", "camera_id", "box_id", "class", "order", "background_level",
    "peak_intensity", "peak_intensity_norm", "mean_intensity", "mean_intensity_norm", "core_area_px",
    "aspect_ratio", "scatter_ratio", "halo_radius_px", "reflectivity", "angle_deg", "color_r", "color_g",
    "color_b", "gray_level", "clipped_fraction", "confidence",
]
METRIC_COLUMNS = BASE_COLUMNS[7:]  # all measured numbers; blank when not applicable


def rows_for(label_set: ImageLabelSet) -> list[dict]:
    rows = []
    for box in sorted(label_set.laser_boxes, key=lambda b: b.order):
        row = {"image_id": label_set.image_id, "image_name": label_set.image_name,
               "camera_id": label_set.camera_id, "box_id": box.id,
               "class": label_set.laser_class_name(box), "order": box.order,
               "background_level": label_set.background_level}
        row.update({k: box.metrics[k] for k in METRIC_COLUMNS if k in box.metrics})
        row.update({f"exposure_{k}": v for k, v in (label_set.exposure or {}).items()})
        rows.append(row)
    return rows


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def update_measurements(label_set: ImageLabelSet, path: str | Path) -> Path:
    """Replace this image's rows (re-labeling must not duplicate) and rewrite the file."""
    path = Path(path)
    existing = [r for r in _read(path) if r["image_id"] != label_set.image_id]
    rows = existing + rows_for(label_set)
    extra = sorted({k for r in rows for k in r if k not in BASE_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_COLUMNS + extra, restval="")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return path
