#!/usr/bin/env python3
"""
evaluate.py

Runs standard object-detection evaluation (precision, recall, mAP50,
mAP50-95, per class) for an already-trained YOLO checkpoint against a split
of the dataset (val, test, or even train, e.g. to sanity-check for
overfitting).

Like train_yolo.py, this prefers dataset/data.corrected.yaml over the raw
dataset/data.yaml unless you pass --data explicitly.

Usage:
    # evaluate the best checkpoint from a training run against the validation split
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset

    # evaluate against the held-out test split instead
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset --split test

    # evaluate against train, e.g. to compare against val/test and check for overfitting
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset --split train
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from yolo_common import resolve_data_yaml, load_data_yaml


def run_evaluation(
    weights: Path,
    data_yaml: Path,
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    conf: float = 0.001,
    iou: float = 0.6,
    project: str = "runs/val",
    name: str = "exp",
) -> dict:
    """Run model.val() and write a per-class report. Returns a dict of overall metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install it with:\n    pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    model = YOLO(str(weights))
    val_kwargs = dict(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        project=project,
        name=name,
        save_json=False,
    )
    if device is not None:
        val_kwargs["device"] = device

    metrics = model.val(**val_kwargs)

    names = metrics.names if hasattr(metrics, "names") else load_data_yaml(data_yaml).get("names", {})
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    box = metrics.box
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "fitness": float(getattr(metrics, "fitness", box.map)),
    }

    # Per-class rows: index-aligned with box.ap_class_index for the classes that
    # actually appeared in this split (classes with zero instances are omitted by ultralytics).
    per_class_rows = []
    ap_class_index = list(getattr(box, "ap_class_index", []))
    ap50_per_class = list(getattr(box, "ap50", []))
    ap_per_class = list(getattr(box, "ap", []))
    p_per_class = list(getattr(box, "p", []))
    r_per_class = list(getattr(box, "r", []))
    nt_per_class = getattr(metrics, "nt_per_class", None)

    for row_i, cls_idx in enumerate(ap_class_index):
        per_class_rows.append({
            "class_id": int(cls_idx),
            "class_name": names.get(int(cls_idx), str(cls_idx)),
            "instances": int(nt_per_class[cls_idx]) if nt_per_class is not None else "",
            "precision": float(p_per_class[row_i]) if row_i < len(p_per_class) else float("nan"),
            "recall": float(r_per_class[row_i]) if row_i < len(r_per_class) else float("nan"),
            "mAP50": float(ap50_per_class[row_i]) if row_i < len(ap50_per_class) else float("nan"),
            "mAP50-95": float(ap_per_class[row_i]) if row_i < len(ap_per_class) else float("nan"),
        })

    out_dir = Path(project) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_report(out_dir, weights, data_yaml, split, overall, per_class_rows)
    return {"overall": overall, "per_class": per_class_rows, "output_dir": str(out_dir)}


def _write_report(out_dir: Path, weights: Path, data_yaml: Path, split: str, overall: dict, per_class_rows: list[dict]):
    txt_path = out_dir / "evaluation_report.txt"
    csv_path = out_dir / "evaluation_per_class.csv"

    lines = [
        "=" * 70,
        "YOLO EVALUATION REPORT",
        "=" * 70,
        f"weights: {weights}",
        f"data:    {data_yaml}",
        f"split:   {split}",
        "",
        "Overall:",
        f"  precision  : {overall['precision']:.4f}",
        f"  recall     : {overall['recall']:.4f}",
        f"  mAP50      : {overall['mAP50']:.4f}",
        f"  mAP50-95   : {overall['mAP50-95']:.4f}",
        "",
        "Per-class:",
    ]
    header = f"{'class':<55}{'instances':>10}{'precision':>11}{'recall':>9}{'mAP50':>9}{'mAP50-95':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in sorted(per_class_rows, key=lambda r: r["mAP50-95"]):
        lines.append(
            f"{row['class_name'][:54]:<55}{row['instances']:>10}"
            f"{row['precision']:>11.4f}{row['recall']:>9.4f}{row['mAP50']:>9.4f}{row['mAP50-95']:>10.4f}"
        )
    lowest = sorted(per_class_rows, key=lambda r: r["mAP50-95"])[:5]
    if lowest:
        lines.append("")
        lines.append("Lowest mAP50-95 classes (worth a closer look):")
        for row in lowest:
            lines.append(f"  {row['class_name']}: mAP50-95={row['mAP50-95']:.4f}, instances={row['instances']}")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "instances", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        for row in per_class_rows:
            writer.writerow(row)

    print("\n".join(lines))
    print(f"\nFull report: {txt_path}")
    print(f"Per-class CSV: {csv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True, help="Path to a trained .pt checkpoint")
    ap.add_argument("--dataset-dir", type=Path, default=None, help="Path to the dataset/ folder (used to auto-pick data.corrected.yaml)")
    ap.add_argument("--data", type=Path, default=None, help="Explicit path to a data.yaml, overrides --dataset-dir auto-detection")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"], help="Which split to evaluate against (default: val)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu, mps. Default: let ultralytics auto-select")
    ap.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for evaluation (low, standard for mAP calc)")
    ap.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold used during evaluation")
    ap.add_argument("--project", default="runs/val", help="Where to save evaluation outputs")
    ap.add_argument("--name", default="exp", help="Run name (subfolder under --project)")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights file not found: {args.weights}")

    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)

    if args.split == "test" and args.dataset_dir is not None and not (args.dataset_dir / "test" / "images").exists():
        raise SystemExit(f"--split test was requested but {args.dataset_dir / 'test'} doesn't exist.")

    run_evaluation(
        weights=args.weights,
        data_yaml=data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
