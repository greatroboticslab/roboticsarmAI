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

    # slow training? make sure it's actually using the GPU, and cache images in RAM
    python train_yolo.py --dataset-dir ../dataset --device 0 --cache ram --workers 4

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
    ap.add_argument("--workers", type=int, default=8, help="Dataloader worker processes. On Windows, high values can add overhead on small datasets -- try 0 or 4 if training seems slow.")
    ap.add_argument(
        "--cache", default=None, choices=["ram", "disk"],
        help="Cache images after the first read instead of re-decoding from disk every epoch "
             "(ram: fastest, needs enough RAM to hold the dataset; disk: slower than ram but still "
             "much faster than no caching). Off by default; strongly recommended if training feels slow.",
    )
    ap.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed-precision (fp16) training. Use this if training crashes with a cuDNN/CUBLAS "
             "error (e.g. CUDNN_STATUS_EXECUTION_FAILED_CUBLAS), which is common on GTX 16-series cards. "
             "Slower and uses more memory, but stable.",
    )
    aug = ap.add_argument_group(
        "augmentation (applied on-the-fly during training; unset flags keep ultralytics' defaults)"
    )
    aug.add_argument(
        "--no-color-aug", action="store_true",
        help="Turn off all color/brightness augmentation (hsv_h, hsv_s, hsv_v = 0). Use this if color, "
             "brightness or laser glow/diffraction patterns are part of what distinguishes your classes.",
    )
    aug.add_argument("--hsv-h", type=float, default=None, help="Hue jitter (ultralytics default 0.015)")
    aug.add_argument("--hsv-s", type=float, default=None, help="Saturation jitter (ultralytics default 0.7)")
    aug.add_argument("--hsv-v", type=float, default=None, help="Brightness jitter (ultralytics default 0.4)")
    aug.add_argument("--degrees", type=float, default=None, help="Random rotation range in degrees, +/- (default 0)")
    aug.add_argument("--translate", type=float, default=None, help="Random shift as a fraction of image size (default 0.1)")
    aug.add_argument("--scale", type=float, default=None, help="Random zoom range (default 0.5)")
    aug.add_argument("--shear", type=float, default=None, help="Random shear in degrees (default 0)")
    aug.add_argument("--fliplr", type=float, default=None, help="Probability of a left-right flip (default 0.5). Set 0 if your laser/camera geometry is fixed.")
    aug.add_argument("--flipud", type=float, default=None, help="Probability of an up-down flip (default 0)")
    aug.add_argument("--mosaic", type=float, default=None, help="Probability of mosaic (4-image stitching) augmentation (default 1.0)")
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

    if args.device is None or str(args.device).lower() != "cpu":
        try:
            import torch
            if not torch.cuda.is_available():
                print(
                    "[warn] torch.cuda.is_available() is False -- training will run on CPU, which is "
                    "typically 10-50x slower than GPU even with the smallest model (yolov8n). If you have "
                    "an NVIDIA GPU, this usually means PyTorch installed without CUDA support; reinstall it "
                    "using the command for your CUDA version from https://pytorch.org/get-started/locally/ "
                    "then re-run with --device 0."
                )
        except ImportError:
            pass  # torch not importable standalone in this env; ultralytics import above already succeeded, so let it proceed

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
    if args.cache is not None:
        train_kwargs["cache"] = args.cache
    if args.no_amp:
        train_kwargs["amp"] = False

    aug_overrides = {
        "hsv_h": args.hsv_h, "hsv_s": args.hsv_s, "hsv_v": args.hsv_v,
        "degrees": args.degrees, "translate": args.translate, "scale": args.scale,
        "shear": args.shear, "fliplr": args.fliplr, "flipud": args.flipud, "mosaic": args.mosaic,
    }
    if args.no_color_aug:
        for k in ("hsv_h", "hsv_s", "hsv_v"):
            aug_overrides[k] = 0.0  # explicit --hsv-* flags are overridden by --no-color-aug
    aug_overrides = {k: v for k, v in aug_overrides.items() if v is not None}
    if aug_overrides:
        train_kwargs.update(aug_overrides)
        print(f"[info] augmentation overrides: {aug_overrides}")

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

        category_map_path = None
        if args.dataset_dir is not None:
            default_map = args.dataset_dir / "object_categories.yaml"
            if default_map.exists():
                print(f"[info] using category map: {default_map}")
                category_map_path = default_map

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
                category_map_path=category_map_path,
            )


if __name__ == "__main__":
    main()
