#!/usr/bin/env python3
"""
sort_yolo_errors.py

Runs a trained YOLO checkpoint over a dataset split and SORTS every image
into folders:

    <output>/correct/                      every object found with the exact right class, nothing extra
    <output>/failure/<failure_type>/       anything else, filed under its most serious failure

Each image is saved with the ground truth (green, "TRUE: ...") and the model's
predictions (red, "PRED: ... conf") drawn on it.

Failure types: wrong_object, wrong_object_same_category, wrong_material,
wrong_color, wrong_material_and_color, wrong_class_same_attributes (all "wrong
class"), missed_no_detection, missed_low_confidence, poor_localization (false
negatives), false_positive_background, false_positive_duplicate (false
positives). The rules live in ../common/error_sorting.py and are IDENTICAL to
the RF-DETR sorter (../transformer/sort_rfdetr_errors.py), so the two models
are directly comparable. Also writes image_summary.csv, errors_report.csv,
confusions.csv, summary.txt and summary.json.

This is the finer-grained, folder-sorting successor to find_prediction_errors.py
(which only distinguishes MISSED / WRONGCLASS / FALSEPOS and is kept as-is).

Usage:
    python sort_yolo_errors.py --dataset-dir ../../dataset --weights runs/detect/runs/train/exp/weights/best.pt

    # test split, looser confidence cut-off
    python sort_yolo_errors.py --dataset-dir ../../dataset --weights .../best.pt --split test --conf 0.2
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "common"))  # shared helpers live in ../common

import argparse
import sys
from pathlib import Path

from error_sorting import LOW_FLOOR, add_common_args, load_class_names, run_sort
from yolo_common import resolve_data_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--weights", type=Path, required=True, help="YOLO checkpoint (best.pt)")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size (use what you trained with)")
    ap.add_argument("--device", default=None, help="e.g. 0, cpu, mps (default: let ultralytics choose)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install it with:\n    pip install ultralytics", file=sys.stderr)
        sys.exit(1)
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}")

    names = load_class_names(resolve_data_yaml(args.dataset_dir, args.data))
    split = "valid" if args.split == "val" else args.split
    model = YOLO(str(args.weights))
    if len(model.names) != len(names):
        print(f"[warn] model has {len(model.names)} classes but the dataset yaml has {len(names)} -- "
              "are you sorting with the right dataset/weights pair?")

    def predict(img_path, img):
        r = model.predict(source=str(img_path), conf=LOW_FLOOR, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        return [(int(b.cls[0]), float(b.conf[0]), *[float(v) for v in b.xyxy[0]]) for b in r.boxes]

    run_sort(args, predict, model_label="YOLO", checkpoint_label=str(args.weights),
             default_out_dir=args.weights.parent.parent / f"error_sort_{split}")


if __name__ == "__main__":
    main()
