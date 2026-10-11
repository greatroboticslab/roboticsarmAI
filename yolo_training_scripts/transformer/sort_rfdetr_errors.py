#!/usr/bin/env python3
"""
sort_rfdetr_errors.py

Runs a trained RF-DETR checkpoint over a dataset split and SORTS every image
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
the YOLO sorter (../yolo/sort_yolo_errors.py), so the two models are directly
comparable. Also writes image_summary.csv, errors_report.csv, confusions.csv,
summary.txt and summary.json.

Usage:
    python sort_rfdetr_errors.py --dataset-dir ../../dataset --weights runs/rfdetr/pretrained

    # test split, looser confidence cut-off
    python sort_rfdetr_errors.py --dataset-dir ../../dataset --weights runs/rfdetr/pretrained --split test --conf 0.2
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


def resolve_checkpoint(path: Path) -> Path:
    if path.is_dir():
        for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
            if (path / name).exists():
                return path / name
        raise SystemExit(f"No checkpoint_best_*.pth found in {path}")
    if not path.exists():
        raise SystemExit(f"Weights not found: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--weights", type=Path, required=True, help="RF-DETR checkpoint (.pth), or a run folder containing checkpoint_best_total.pth")
    args = ap.parse_args()

    try:
        from rfdetr import RFDETR
    except ImportError:
        print('rfdetr is not installed. Install it with:\n    pip install "rfdetr[train]"', file=sys.stderr)
        sys.exit(1)

    names = load_class_names(resolve_data_yaml(args.dataset_dir, args.data))
    name_to_id = {n: i for i, n in enumerate(names)}
    ckpt = resolve_checkpoint(args.weights)
    split = "valid" if args.split == "val" else args.split
    print(f"[info] checkpoint: {ckpt}")
    model = RFDETR.from_checkpoint(str(ckpt))

    def predict(img_path, img):
        det = model.predict(img, threshold=LOW_FLOOR)
        det_names = det.data.get("class_name") if getattr(det, "data", None) else None
        out = []
        for k in range(len(det)):
            # map by class NAME so the ids always match data.yaml's order
            if det_names is not None and str(det_names[k]) in name_to_id:
                cid = name_to_id[str(det_names[k])]
            else:
                cid = int(det.class_id[k])
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[k])
            out.append((cid, float(det.confidence[k]), x1, y1, x2, y2))
        return out

    run_sort(args, predict, model_label="RF-DETR", checkpoint_label=str(ckpt),
             default_out_dir=ckpt.parent / f"error_sort_{split}")


if __name__ == "__main__":
    main()
