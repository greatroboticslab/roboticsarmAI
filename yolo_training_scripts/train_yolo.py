#!/usr/bin/env python3
"""
train_yolo.py

Trains a YOLO model on the dataset/ folder.

Run build_dataset.py FIRST. This script will refuse to train on data.yaml
directly if a data.corrected.yaml exists and hasn't been reconciled (i.e. it
prefers data.corrected.yaml, the output of the PDF-label reconciliation step,
over the raw data.yaml, unless you explicitly pass --data to override).

Usage:
    # 1) reconcile labels from the PDFs first
    python build_dataset.py --dataset-dir ../dataset

    # 2) train
    python train_yolo.py --dataset-dir ../dataset --model yolov8n.pt --epochs 100

    # override everything explicitly
    python train_yolo.py --data ../dataset/data.yaml --model yolov8s.pt --epochs 150 --imgsz 640 --batch 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def resolve_data_yaml(dataset_dir: Path | None, explicit_data: Path | None) -> Path:
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
        print("        latest PDF corrections before training on real data.")
        return raw
    raise SystemExit(f"Neither data.corrected.yaml nor data.yaml found under {dataset_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=None, help="Path to the dataset/ folder (used to auto-pick data.corrected.yaml)")
    ap.add_argument("--data", type=Path, default=None, help="Explicit path to a data.yaml, overrides --dataset-dir auto-detection")
    ap.add_argument("--model", default="yolov8n.pt", help="Base checkpoint or model yaml to start from (default: yolov8n.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu, mps. Default: let ultralytics auto-select")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", default="runs/train", help="Where to save run outputs")
    ap.add_argument("--name", default="exp", help="Run name (subfolder under --project)")
    ap.add_argument("--patience", type=int, default=50, help="Early-stopping patience (epochs with no improvement)")
    ap.add_argument("--resume", action="store_true", help="Resume the most recent interrupted run")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install it with:\n    pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)
    print("\nTraining complete.")
    print(f"Best weights: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    main()
