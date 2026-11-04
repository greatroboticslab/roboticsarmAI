"""Build COCO JSON boxes from approved label sets (build spec section 12).

Only class names and boxes are sent to Roboflow. Measurements and links stay in
the sidecar and CSV. Category ids start at 1 and are assigned in sorted class
order so the same classes always get the same ids.
"""
from __future__ import annotations

from typing import Iterable

from core.models import ImageLabelSet
from core.schema.class_names import encode_object_class, resolve_color

EXPORTABLE_STATUSES = ("approved", "edited")


def class_names_for(label_set: ImageLabelSet, color_placeholder: str = "na") -> list[tuple[str, tuple]]:
    """Return (class_name, box) for every box in the set."""
    items: list[tuple[str, tuple]] = []
    for laser in label_set.laser_boxes:
        items.append((laser.class_name, laser.box))
    for obj in label_set.object_boxes:
        color = resolve_color(obj.color, label_set.is_grayscale, color_placeholder)
        name = encode_object_class(
            obj.general_object, obj.sub_object, obj.material, color, obj.incidence
        )
        items.append((name, obj.box))
    return items


def _clamp(box: tuple, width: int, height: int) -> list[int]:
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    if width and height:
        w, h = min(w, width - x), min(h, height - y)
    return [x, y, max(w, 0), max(h, 0)]


def build_coco(
    label_sets: Iterable[ImageLabelSet],
    color_placeholder: str = "na",
    only_exportable: bool = True,
    category_names: Iterable[str] | None = None,
) -> dict:
    """Return a COCO dict. Raises ValueError if an image has no size and boxes need it."""
    sets = [s for s in label_sets if not only_exportable or s.status in EXPORTABLE_STATUSES]
    per_set = [class_names_for(s, color_placeholder) for s in sets]
    names = set(category_names or ())
    names.update(name for items in per_set for name, _ in items)
    categories = [{"id": i, "name": n, "supercategory": n.split("_", 1)[0]}
                  for i, n in enumerate(sorted(names), start=1)]
    ids = {c["name"]: c["id"] for c in categories}

    images, annotations = [], []
    for image_index, (label_set, items) in enumerate(zip(sets, per_set), start=1):
        images.append({"id": image_index, "file_name": label_set.image_name,
                       "width": label_set.image_width, "height": label_set.image_height})
        for name, box in items:
            bbox = _clamp(box, label_set.image_width, label_set.image_height)
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"{label_set.image_name}: empty box for class {name}")
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_index,
                "category_id": ids[name], "bbox": bbox,
                "area": bbox[2] * bbox[3], "iscrowd": 0,
            })
    return {"images": images, "categories": categories, "annotations": annotations}
