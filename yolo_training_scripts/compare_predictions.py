#!/usr/bin/env python3
"""
compare_predictions.py

Picks ONE representative image per distinct object in a split, and renders
it twice into two separate folders:
  ground_truth/      -- the human-labeled ("control") boxes, i.e. what
                         visualize_annotations.py already draws
  model_predictions/ -- what your trained model actually detects on that
                         same image, drawn in the same visual style so the
                         two are directly comparable side by side

This needs a trained checkpoint (--weights) to generate the predictions --
it's not just a relabeling of visualize_annotations.py, it actually runs
inference.

Usage:
    python compare_predictions.py --weights runs/detect/runs/train/exp/weights/best.pt --dataset-dir ../dataset

    # lower the confidence threshold if the model is under-confident and predictions look empty
    python compare_predictions.py --weights .../best.pt --dataset-dir ../dataset --conf 0.1

    # use the test split instead, and a specific image size
    python compare_predictions.py --weights .../best.pt --dataset-dir ../dataset --split test --imgsz 640
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from yolo_common import (
    resolve_data_yaml, load_data_yaml, find_split_dirs, list_image_label_pairs,
    read_label_class_ids, parse_class_name,
)
from visualize_annotations import (
    load_font, build_color_map, short_label, draw_annotated_image, draw_single_box,
)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def safe_filename(name: str, max_len: int = 60) -> str:
    return _SAFE_NAME_RE.sub("_", name).strip("_")[:max_len] or "unnamed"


def pick_one_image_per_object(
    pairs: list[tuple[Path, Path]], class_names: list[str], seed: int,
) -> list[tuple[str, Path, Path]]:
    """
    Returns [(object_name, image_path, label_path), ...], one entry per distinct
    object found anywhere in this split. Rarest objects (fewest candidate images)
    get first pick of an unused image, so common objects don't crowd out one
    that only appears in a couple of photos; images are reused across objects
    only if there's truly no unused candidate left for that object.
    """
    object_to_pairs: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img_path, label_path in pairs:
        cls_ids = read_label_class_ids(label_path)
        objects_in_image = {
            parse_class_name(class_names[c])["object"] or class_names[c]
            for c in cls_ids if c < len(class_names)
        }
        for obj in objects_in_image:
            object_to_pairs[obj].append((img_path, label_path))

    rng = random.Random(seed)
    for obj in object_to_pairs:
        rng.shuffle(object_to_pairs[obj])

    ordered_objects = sorted(object_to_pairs, key=lambda o: len(object_to_pairs[o]))

    used: set[Path] = set()
    selection: list[tuple[str, Path, Path]] = []
    for obj in ordered_objects:
        candidates = object_to_pairs[obj]
        pick = next((c for c in candidates if c[0] not in used), candidates[0])
        used.add(pick[0])
        selection.append((obj, pick[0], pick[1]))

    selection.sort(key=lambda s: s[0])
    return selection


def draw_predictions(
    img_path: Path, boxes, class_names: list[str], color_map: dict, font, box_width: int = 3,
) -> tuple[Image.Image, int]:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    n = 0
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        label = f"{short_label(cls_name)} {conf:.2f}"
        color = color_map.get(cls_id, (220, 40, 40))
        draw_single_box(draw, (x1, y1, x2, y2), color, label, font, box_width)
        n += 1
    return img, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True, help="Trained checkpoint (.pt) to generate predictions from")
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml, overrides --dataset-dir auto-detection")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: <dataset-dir>/gt_vs_pred_preview")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions (default: 0.25)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--box-width", type=int, default=3)
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--font", type=Path, default=None, help="Path to a .ttf font to use instead of auto-detecting one")
    ap.add_argument("--seed", type=int, default=42, help="Random seed used when an object has multiple candidate images")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights file not found: {args.weights}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install it with:\n    pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)
    class_names = load_data_yaml(data_yaml)["names"]
    color_map = build_color_map(class_names)
    font = load_font(args.font_size, args.font)

    split_dirs = find_split_dirs(args.dataset_dir)
    if args.split not in split_dirs:
        raise SystemExit(f"Split '{args.split}' not found under {args.dataset_dir} (found: {sorted(split_dirs)})")
    pairs = list_image_label_pairs(split_dirs[args.split])

    selection = pick_one_image_per_object(pairs, class_names, args.seed)
    print(f"Selected {len(selection)} image(s), one per distinct object, from '{args.split}'.")

    output_dir = args.output_dir or (args.dataset_dir / "gt_vs_pred_preview")
    gt_dir = output_dir / "ground_truth"
    pred_dir = output_dir / "model_predictions"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))

    print(f"\n{'object':<40}{'file':<20}{'gt_boxes':>10}{'pred_boxes':>12}")
    print("-" * 82)
    for obj, img_path, label_path in selection:
        fname = safe_filename(obj) + img_path.suffix.lower()

        gt_img, gt_labels = draw_annotated_image(img_path, label_path, class_names, color_map, font, args.box_width)
        gt_img.save(gt_dir / fname, quality=92)

        results = model.predict(source=str(img_path), conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)
        pred_img, n_pred = draw_predictions(img_path, results[0].boxes, class_names, color_map, font, args.box_width)
        pred_img.save(pred_dir / fname, quality=92)

        print(f"{obj[:39]:<40}{fname[:19]:<20}{len(gt_labels):>10}{n_pred:>12}")

    print(f"\nGround-truth images: {gt_dir}")
    print(f"Model-prediction images: {pred_dir}")
    print("Matching filenames in both folders make it easy to open them side by side.")


if __name__ == "__main__":
    main()
