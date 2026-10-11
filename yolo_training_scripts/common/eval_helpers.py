"""
eval_helpers.py

Pure helpers shared by the YOLO and RF-DETR evaluation scripts (composite
weights, the object->category map, label-group normalisation). They used to
live in yolo/evaluate.py; they are here so the transformer/ scripts don't have
to import from the yolo/ folder. yolo/evaluate.py re-imports them unchanged.
"""

from __future__ import annotations

from pathlib import Path

from yolo_common import parse_class_name


def _normalize_group_key(value: str) -> str:
    """Case/whitespace-insensitive grouping key -- e.g. 'Metal' and 'metal' (a data.yaml
    typo, not a real distinction) should end up in the same group, not two."""
    return " ".join(value.split()).lower()



# Default composite weighting: object is the most important thing to get right,
# then material, then color (deliberately the least weighted of the three).
DEFAULT_OBJECT_WEIGHT = 0.6
DEFAULT_MATERIAL_WEIGHT = 0.3
DEFAULT_COLOR_WEIGHT = 0.1


def normalize_weights(object_weight: float, material_weight: float, color_weight: float) -> tuple[float, float, float]:
    """Scale the three weights so they sum to 1. Negative weights are rejected; all-zero
    falls back to the defaults. Warns (but doesn't block) if the weights aren't ordered
    object >= material >= color, since that's the intended priority."""
    if min(object_weight, material_weight, color_weight) < 0:
        raise ValueError("object/material/color weights must be non-negative")
    total = object_weight + material_weight + color_weight
    if total <= 0:
        object_weight, material_weight, color_weight = DEFAULT_OBJECT_WEIGHT, DEFAULT_MATERIAL_WEIGHT, DEFAULT_COLOR_WEIGHT
        total = 1.0
    o, m, c = object_weight / total, material_weight / total, color_weight / total
    if not (o >= m >= c):
        print(f"[warn] weights are not ordered object >= material >= color "
              f"({o:.2f}/{m:.2f}/{c:.2f}); using them as given.")
    return o, m, c


def load_category_map(path: Path | None) -> dict[str, str]:
    """
    Load an object-name -> category-name mapping from a small YAML/JSON file, e.g.:
        ballpoint pen: pen
        gel pen: pen
        precision screwdriver: screwdriver
        flathead screwdriver: screwdriver
    Any object not listed simply isn't grouped with anything (see build_category_index).
    Returns {} if path is None or the file doesn't exist.
    """
    if path is None or not Path(path).exists():
        return {}
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {str(k).strip(): str(v).strip() for k, v in data.items()}


def build_category_index(names: dict[int, str], category_map: dict[str, str]) -> dict[int, str]:
    """Map each class id -> its category name. Objects not in category_map fall back to
    their own object name as a singleton category, so this is always safe to call even
    with an empty/partial mapping -- it just won't group anything extra. Lookup is
    case/whitespace-insensitive (see _normalize_group_key), same reasoning as _aggregate."""
    normalized_map = {_normalize_group_key(k): v for k, v in category_map.items()}
    index = {}
    for cls_id, name in names.items():
        obj = parse_class_name(name)["object"] or name
        index[cls_id] = normalized_map.get(_normalize_group_key(obj), obj)
    return index
