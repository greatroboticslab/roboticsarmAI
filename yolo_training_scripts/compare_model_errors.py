#!/usr/bin/env python3
"""
compare_model_errors.py

Answers "did this change actually help?" by running TWO checkpoints (e.g.
your baseline vs. a --no-color-aug retrain) against the same split and
classifying every image into:

  FIXED_BY_B      -- model A got this wrong, model B got it right
  REGRESSED_BY_B  -- model A got this right, model B got it wrong (a real
                     problem worth noticing even if B looks better overall)
  STILL_WRONG     -- both models got this wrong (maybe differently)
  (images both models get right are skipped entirely -- not interesting)

Each exported image overlays: the ground truth in green, model A's
prediction in amber, model B's prediction in red -- all on the same photo,
so you can see exactly what changed instead of flipping between two
separate folders.

Usage:
    python compare_model_errors.py \\
        --weights-a runs/detect/runs/train/baseline/weights/best.pt \\
        --weights-b runs/detect/runs/train/noColorAug/weights/best.pt \\
        --label-a baseline --label-b no_color_aug \\
        --dataset-dir ../dataset --split valid
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from yolo_common import (
    resolve_data_yaml, load_data_yaml, find_split_dirs, list_image_label_pairs,
    read_label_boxes,
)
from visualize_annotations import load_font, short_label, draw_single_box
from find_prediction_errors import match_predictions_to_truth

TRUE_COLOR = (30, 160, 60)     # green: ground truth
COLOR_A = (230, 150, 20)       # amber: model A
COLOR_B = (215, 30, 30)        # red: model B


def run_inference(model, img_path: Path, conf: float, imgsz: int, device):
    results = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False)
    return [
        (int(b.cls[0]), float(b.conf[0]), *[float(v) for v in b.xyxy[0]])
        for b in results[0].boxes
    ]


def classify_errors(gt_boxes, pred_boxes, iou_thresh: float, class_names: list[str]) -> list[str]:
    """Returns a list of short error descriptions, e.g. ['MISSED rubber duck', 'WRONGCLASS ...'].
    Empty list means every ground-truth box was correctly matched and there were no extra predictions."""
    matches, unmatched_gt, unmatched_pred = match_predictions_to_truth(gt_boxes, pred_boxes, iou_thresh)
    errors = []
    for gi, pi, _ in matches:
        gt_cls, pred_cls = gt_boxes[gi][0], pred_boxes[pi][0]
        if gt_cls != pred_cls:
            errors.append(f"WRONGCLASS true={class_names[gt_cls]} pred={class_names[pred_cls]}")
    for gi in unmatched_gt:
        errors.append(f"MISSED {class_names[gt_boxes[gi][0]]}")
    for pi in unmatched_pred:
        errors.append(f"FALSEPOS {class_names[pred_boxes[pi][0]]}")
    return errors


def run_compare_models(
    weights_a: Path, weights_b: Path, label_a: str, label_b: str,
    dataset_dir: Path, data_yaml: Path, split: str, output_dir: Path,
    conf: float = 0.25, iou_match: float = 0.5, imgsz: int = 640, device=None,
    box_width: int = 3, font_size: int = 16, font_path: Path | None = None,
    make_zip: bool = True, model_a=None, model_b=None,
) -> dict:
    """
    Core logic behind compare_model_errors.py's CLI, factored out so other
    scripts (e.g. run_full_error_report.py) can call it directly for several
    splits without shelling back out to this file. Pass already-loaded
    `model_a`/`model_b` to avoid reloading the same checkpoints repeatedly
    across multiple splits. Returns a summary dict.
    """
    class_names = load_data_yaml(data_yaml)["names"]
    font = load_font(font_size, font_path)

    split_dirs = find_split_dirs(dataset_dir)
    if split not in split_dirs:
        raise SystemExit(f"Split '{split}' not found under {dataset_dir} (found: {sorted(split_dirs)})")
    pairs = list_image_label_pairs(split_dirs[split])

    if output_dir.exists():
        shutil.rmtree(output_dir)
    for sub in ("fixed_by_b", "regressed_by_b", "still_wrong"):
        (output_dir / sub).mkdir(parents=True)

    if model_a is None or model_b is None:
        from ultralytics import YOLO
        if model_a is None:
            print(f"Loading model A ({label_a}): {weights_a}")
            model_a = YOLO(str(weights_a))
        if model_b is None:
            print(f"Loading model B ({label_b}): {weights_b}")
            model_b = YOLO(str(weights_b))

    outcome_counts = Counter()
    report_rows = []

    for img_path, label_path in pairs:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        gt_boxes = read_label_boxes(label_path, w, h)

        pred_a = run_inference(model_a, img_path, conf, imgsz, device)
        pred_b = run_inference(model_b, img_path, conf, imgsz, device)

        errors_a = classify_errors(gt_boxes, pred_a, iou_match, class_names)
        errors_b = classify_errors(gt_boxes, pred_b, iou_match, class_names)

        if not errors_a and not errors_b:
            continue  # both correct -- not interesting

        if errors_a and not errors_b:
            outcome = "fixed_by_b"
        elif not errors_a and errors_b:
            outcome = "regressed_by_b"
        else:
            outcome = "still_wrong"
        outcome_counts[outcome] += 1

        draw = ImageDraw.Draw(img, "RGBA")
        for cls_id, x1, y1, x2, y2 in gt_boxes:
            draw_single_box(draw, (x1, y1, x2, y2), TRUE_COLOR, f"TRUE: {short_label(class_names[cls_id])}", font, box_width)
        for cls_id, pconf, x1, y1, x2, y2 in pred_a:
            draw_single_box(draw, (x1, y1, x2, y2), COLOR_A, f"{label_a}: {short_label(class_names[cls_id])} {pconf:.2f}", font, box_width)
        for cls_id, pconf, x1, y1, x2, y2 in pred_b:
            draw_single_box(draw, (x1, y1, x2, y2), COLOR_B, f"{label_b}: {short_label(class_names[cls_id])} {pconf:.2f}", font, box_width)

        out_name = f"{img_path.stem}{img_path.suffix.lower()}"
        img.save(output_dir / outcome / out_name, quality=92)

        report_rows.append({
            "image": out_name, "outcome": outcome,
            f"{label_a}_errors": "; ".join(errors_a),
            f"{label_b}_errors": "; ".join(errors_b),
        })

    csv_path = output_dir / "comparison_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "outcome", f"{label_a}_errors", f"{label_b}_errors"])
        writer.writeheader()
        writer.writerows(report_rows)

    n_total_interesting = sum(outcome_counts.values())
    txt_lines = [
        "=" * 70, "MODEL COMPARISON REPORT", "=" * 70,
        f"model A ({label_a}): {weights_a}",
        f"model B ({label_b}): {weights_b}",
        f"split: {split}   conf: {conf}   iou-match: {iou_match}",
        "",
        f"images checked: {len(pairs)}",
        f"images where both models were correct (skipped): {len(pairs) - n_total_interesting}",
        "",
        f"fixed_by_b     ({label_a} wrong, {label_b} right): {outcome_counts['fixed_by_b']}",
        f"regressed_by_b ({label_a} right, {label_b} wrong): {outcome_counts['regressed_by_b']}",
        f"still_wrong    (both wrong):                        {outcome_counts['still_wrong']}",
        "",
    ]
    net = outcome_counts["fixed_by_b"] - outcome_counts["regressed_by_b"]
    if net > 0:
        txt_lines.append(f"Net effect: model B fixed {net} more image(s) than it broke -- looks like an improvement.")
    elif net < 0:
        txt_lines.append(f"Net effect: model B broke {-net} more image(s) than it fixed -- looks like a regression.")
    else:
        txt_lines.append("Net effect: a wash on this split -- fixed and regressed counts are equal.")
    txt_lines.append("Check regressed_by_b/ specifically -- even a net improvement can hide a new problem worth knowing about.")

    txt_path = output_dir / "comparison_report.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    print("\n".join(txt_lines))

    zip_path = None
    if make_zip:
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in output_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, arcname=f"{output_dir.name}/{f.relative_to(output_dir)}")
        print(f"Zipped: {zip_path}")

    print(f"\nAnnotated comparison images + report: {output_dir}")

    return {
        "n_images_checked": len(pairs),
        "outcome_counts": dict(outcome_counts),
        "net": net,
        "output_dir": output_dir,
        "zip_path": zip_path,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights-a", type=Path, required=True, help="Baseline checkpoint")
    ap.add_argument("--weights-b", type=Path, required=True, help="The checkpoint you're comparing against A (e.g. trained with different augmentation)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml, overrides --dataset-dir auto-detection")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: <dataset-dir>/model_comparison")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--box-width", type=int, default=3)
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--font", type=Path, default=None)
    args = ap.parse_args()

    for w in (args.weights_a, args.weights_b):
        if not w.exists():
            raise SystemExit(f"Weights file not found: {w}")

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ultralytics is not installed. Install it with:\n    pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)
    output_dir = args.output_dir or (args.dataset_dir / "model_comparison")

    run_compare_models(
        weights_a=args.weights_a, weights_b=args.weights_b, label_a=args.label_a, label_b=args.label_b,
        dataset_dir=args.dataset_dir, data_yaml=data_yaml, split=args.split, output_dir=output_dir,
        conf=args.conf, iou_match=args.iou_match, imgsz=args.imgsz, device=args.device,
        box_width=args.box_width, font_size=args.font_size, font_path=args.font,
    )


if __name__ == "__main__":
    main()
