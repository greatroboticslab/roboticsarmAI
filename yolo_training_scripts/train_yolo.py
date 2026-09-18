#!/usr/bin/env python3
"""
train_yolo.py

Trains a YOLO model on the dataset/ folder.

Run build_dataset.py FIRST. This script prefers dataset/data.corrected.yaml
(the output of the PDF-label reconciliation step) over the raw
dataset/data.yaml, unless you explicitly pass --data to override.

Usage:
    # 1) reconcile labels from the PDFs first
    python build_dataset.py --dataset-dir ../dataset

    # 2) train
    python train_yolo.py --dataset-dir ../dataset --model yolov8n.pt --epochs 100

    # train, then immediately run validation-set evaluation on the best checkpoint
    python train_yolo.py --dataset-dir ../dataset --epochs 100 --evaluate

    # override everything explicitly
    python train_yolo.py --data ../dataset/data.yaml --model yolov8s.pt --epochs 150 --imgsz 640 --batch 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from yolo_common import resolve_data_yaml


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
    ap.add_argument(
        "--evaluate", action="store_true",
        help="After training, run evaluate.py's validation pass on the best checkpoint "
             "(validation split, plus test split too if the dataset has one) and write a report.",
    )
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

    train_results = model.train(**train_kwargs)
    save_dir = Path(getattr(train_results, "save_dir", Path(args.project) / args.name))
    best_weights = save_dir / "weights" / "best.pt"
    print("\nTraining complete.")
    print(f"Run directory: {save_dir}")
    print(f"Best weights: {best_weights}")

    if args.evaluate:
        if not best_weights.exists():
            print(f"[warn] --evaluate was set but {best_weights} doesn't exist; skipping evaluation.")
            return
        print("\nRunning post-training evaluation...")
        from evaluate import run_evaluation

        splits_to_run = ["val"]
        if args.dataset_dir is not None and (args.dataset_dir / "test" / "images").exists():
            splits_to_run.append("test")

        for split in splits_to_run:
            print(f"\n--- evaluating on '{split}' split ---")
            run_evaluation(
                weights=best_weights,
                data_yaml=data_yaml,
                split=split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=args.project,
                name=f"{args.name}_eval_{split}",
            )


if __name__ == "__main__":
    main()
