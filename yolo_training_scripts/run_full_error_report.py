#!/usr/bin/env python3
"""
run_full_error_report.py

Runs find_prediction_errors.py for BOTH checkpoints and compare_model_errors.py
for the two of them together, across BOTH the train and valid splits (and
test, if you have one) -- all in one command, all packaged into a single zip.

For two checkpoints A and B and two splits (train, valid), this produces:
  <label_a>_errors/train/   + <label_a>_errors/valid/    (find_prediction_errors, model A alone)
  <label_b>_errors/train/   + <label_b>_errors/valid/    (find_prediction_errors, model B alone)
  <label_a>_vs_<label_b>/train/ + <label_a>_vs_<label_b>/valid/   (compare_model_errors, A vs B)
all under one output folder, zipped together at the end.

If you only want one model's errors (no comparison), just omit --weights-b.

Each checkpoint is loaded once and reused across both splits, rather than
reloading it per split.

Usage:
    # both models, both splits, plus the A-vs-B comparison for each split
    python run_full_error_report.py \\
        --weights-a runs/detect/runs/train/exp-4/weights/best.pt --label-a baseline \\
        --weights-b runs/detect/runs/train/noColorAug/weights/best.pt --label-b no_color_aug \\
        --dataset-dir ../dataset

    # just one model's errors across both splits, no comparison
    python run_full_error_report.py --weights-a .../best.pt --label-a baseline --dataset-dir ../dataset
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from yolo_common import resolve_data_yaml, find_split_dirs
from find_prediction_errors import run_find_errors
from compare_model_errors import run_compare_models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-a", type=Path, required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--weights-b", type=Path, default=None, help="Optional -- if given, also runs the A-vs-B comparison")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml, overrides --dataset-dir auto-detection")
    ap.add_argument(
        "--splits", nargs="+", default=None, choices=["train", "valid", "test"],
        help="Which splits to run against (default: every split that actually exists under --dataset-dir)",
    )
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: <dataset-dir>/full_error_report")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--box-width", type=int, default=3)
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--font", type=Path, default=None)
    args = ap.parse_args()

    if not args.weights_a.exists():
        raise SystemExit(f"Weights file not found: {args.weights_a}")
    if args.weights_b is not None and not args.weights_b.exists():
        raise SystemExit(f"Weights file not found: {args.weights_b}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed. Install it with:\n    pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)

    if args.splits:
        splits = args.splits
    else:
        splits = list(find_split_dirs(args.dataset_dir))
        if not splits:
            raise SystemExit(f"No train/valid/test split found under {args.dataset_dir}")
    print(f"Running against splits: {splits}")

    output_dir = args.output_dir or (args.dataset_dir / "full_error_report")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print(f"Loading model A ({args.label_a}): {args.weights_a}")
    model_a = YOLO(str(args.weights_a))
    model_b = None
    if args.weights_b is not None:
        print(f"Loading model B ({args.label_b}): {args.weights_b}")
        model_b = YOLO(str(args.weights_b))

    common = dict(
        dataset_dir=args.dataset_dir, data_yaml=data_yaml, conf=args.conf, iou_match=args.iou_match,
        imgsz=args.imgsz, device=args.device, box_width=args.box_width, font_size=args.font_size,
        font_path=args.font, make_zip=False,  # we zip everything together once at the very end
    )

    summary_lines = ["=" * 70, "FULL ERROR REPORT -- SUMMARY", "=" * 70]

    for split in splits:
        print(f"\n{'#' * 70}\n# split: {split}  --  model {args.label_a}\n{'#' * 70}")
        res_a = run_find_errors(
            weights=args.weights_a, split=split,
            output_dir=output_dir / f"{args.label_a}_errors" / split,
            model=model_a, **common,
        )
        summary_lines.append(
            f"[{split}] {args.label_a}: {res_a['n_images_with_errors']}/{res_a['n_images_checked']} image(s) "
            f"with errors -- {res_a['error_type_counts']}"
        )

        if model_b is not None:
            print(f"\n{'#' * 70}\n# split: {split}  --  model {args.label_b}\n{'#' * 70}")
            res_b = run_find_errors(
                weights=args.weights_b, split=split,
                output_dir=output_dir / f"{args.label_b}_errors" / split,
                model=model_b, **common,
            )
            summary_lines.append(
                f"[{split}] {args.label_b}: {res_b['n_images_with_errors']}/{res_b['n_images_checked']} image(s) "
                f"with errors -- {res_b['error_type_counts']}"
            )

            print(f"\n{'#' * 70}\n# split: {split}  --  {args.label_a} vs {args.label_b}\n{'#' * 70}")
            res_cmp = run_compare_models(
                weights_a=args.weights_a, weights_b=args.weights_b, label_a=args.label_a, label_b=args.label_b,
                split=split, output_dir=output_dir / f"{args.label_a}_vs_{args.label_b}" / split,
                model_a=model_a, model_b=model_b, **common,
            )
            summary_lines.append(
                f"[{split}] {args.label_a} vs {args.label_b}: {res_cmp['outcome_counts']} (net={res_cmp['net']:+d})"
            )

    summary_path = output_dir / "SUMMARY.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n" + "\n".join(summary_lines))

    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=f"{output_dir.name}/{f.relative_to(output_dir)}")

    print(f"\nEverything written to: {output_dir}")
    print(f"Zipped: {zip_path}")


if __name__ == "__main__":
    main()
