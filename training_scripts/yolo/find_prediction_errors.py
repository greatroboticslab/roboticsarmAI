#!/usr/bin/env python3
"""
find_prediction_errors.py

Runs your trained model against a split, compares every prediction to the
ground truth via IoU matching, and exports ONLY the images where something
went wrong -- each one showing the true (green) and predicted (red) boxes
overlaid on the same image, so you can see exactly what it should have
said versus what it actually said. Correctly-detected objects are skipped
entirely; this is specifically "what did it get wrong."

Three kinds of error, all rendered the same way but categorized separately:
  MISSED      -- a real object with no matching prediction at all (false negative)
  WRONGCLASS  -- a prediction overlaps a real object, but called it the wrong
                 class (e.g. predicted "gel pen" where the true label is
                 "ballpoint pen")
  FALSEPOS    -- a prediction with no matching ground-truth object nearby at
                 all (the model detected something that isn't there)

Writes annotated images to --output-dir (default: dataset/prediction_errors/),
a detailed errors_report.csv + errors_report.txt, and zips the whole folder
so it's ready to share.

Usage:
    python find_prediction_errors.py --weights runs/detect/runs/train/exp/weights/best.pt --dataset-dir ../dataset

    # loosen matching / lower confidence if you're not sure what threshold is right
    python find_prediction_errors.py --weights .../best.pt --dataset-dir ../dataset --conf 0.1 --iou-match 0.3
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "common"))  # shared helpers live in ../common

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
from error_sorting import TRUE_COLOR, PRED_COLOR, iou, match_predictions_to_truth  # shared matching logic

def run_find_errors(
    weights: Path, dataset_dir: Path, data_yaml: Path, split: str, output_dir: Path,
    conf: float = 0.25, iou_match: float = 0.5, imgsz: int = 640, device=None,
    box_width: int = 3, font_size: int = 16, font_path: Path | None = None,
    limit: int | None = None, make_zip: bool = True, model=None,
) -> dict:
    """
    Core logic behind find_prediction_errors.py's CLI, factored out so other
    scripts (e.g. run_full_error_report.py) can call it directly for several
    (weights, split) combinations without shelling back out to this file.
    Pass an already-loaded `model` to avoid reloading the same checkpoint
    repeatedly across multiple splits. Returns a summary dict.
    """
    class_names = load_data_yaml(data_yaml)["names"]
    font = load_font(font_size, font_path)

    split_dirs = find_split_dirs(dataset_dir)
    if split not in split_dirs:
        raise SystemExit(f"Split '{split}' not found under {dataset_dir} (found: {sorted(split_dirs)})")
    pairs = list_image_label_pairs(split_dirs[split])

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    if model is None:
        from ultralytics import YOLO
        model = YOLO(str(weights))

    report_rows = []
    error_type_counts = Counter()
    per_class_error_counts = Counter()
    n_images_with_errors = 0
    n_images_checked = 0

    for img_path, label_path in pairs:
        if limit is not None and n_images_with_errors >= limit:
            break
        n_images_checked += 1

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        gt_boxes = read_label_boxes(label_path, w, h)

        results = model.predict(source=str(img_path), conf=conf, imgsz=imgsz, device=device, verbose=False)
        pred_boxes = [
            (int(b.cls[0]), float(b.conf[0]), *[float(v) for v in b.xyxy[0]])
            for b in results[0].boxes
        ]

        matches, unmatched_gt, unmatched_pred = match_predictions_to_truth(gt_boxes, pred_boxes, iou_match)

        image_errors = []  # (error_type, true_cls_name, pred_cls_name, conf, iou_val)
        for gi, pi, iou_val in matches:
            gt_cls = gt_boxes[gi][0]
            pred_cls, pred_conf = pred_boxes[pi][0], pred_boxes[pi][1]
            if gt_cls != pred_cls:
                image_errors.append(("WRONGCLASS", class_names[gt_cls], class_names[pred_cls], pred_conf, iou_val))
        for gi in unmatched_gt:
            gt_cls = gt_boxes[gi][0]
            image_errors.append(("MISSED", class_names[gt_cls], None, None, None))
        for pi in unmatched_pred:
            pred_cls, pred_conf = pred_boxes[pi][0], pred_boxes[pi][1]
            image_errors.append(("FALSEPOS", None, class_names[pred_cls], pred_conf, None))

        if not image_errors:
            continue

        n_images_with_errors += 1
        draw = ImageDraw.Draw(img, "RGBA")
        for cls_id, x1, y1, x2, y2 in gt_boxes:
            draw_single_box(draw, (x1, y1, x2, y2), TRUE_COLOR, f"TRUE: {short_label(class_names[cls_id])}", font, box_width)
        for cls_id, pconf, x1, y1, x2, y2 in pred_boxes:
            draw_single_box(draw, (x1, y1, x2, y2), PRED_COLOR, f"PRED: {short_label(class_names[cls_id])} {pconf:.2f}", font, box_width)

        types_present = {e[0] for e in image_errors}
        primary = "WRONGCLASS" if "WRONGCLASS" in types_present else ("MISSED" if "MISSED" in types_present else "FALSEPOS")
        out_name = f"{primary}__{img_path.stem}{img_path.suffix.lower()}"
        img.save(output_dir / out_name, quality=92)

        for error_type, true_name, pred_name, econf, iou_val in image_errors:
            error_type_counts[error_type] += 1
            per_class_error_counts[true_name or pred_name] += 1
            report_rows.append({
                "image": out_name,
                "error_type": error_type,
                "true_class": true_name or "",
                "predicted_class": pred_name or "",
                "confidence": f"{econf:.4f}" if econf is not None else "",
                "iou": f"{iou_val:.4f}" if iou_val is not None else "",
            })

    csv_path = output_dir / "errors_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "error_type", "true_class", "predicted_class", "confidence", "iou"])
        writer.writeheader()
        writer.writerows(report_rows)

    txt_lines = [
        "=" * 70, "PREDICTION ERROR REPORT", "=" * 70,
        f"weights: {weights}", f"split: {split}",
        f"conf threshold: {conf}  iou-match threshold: {iou_match}", "",
        f"images checked: {n_images_checked}",
        f"images with at least one error: {n_images_with_errors}",
        "",
        "error counts:",
        f"  MISSED (real object, no prediction):        {error_type_counts['MISSED']}",
        f"  WRONGCLASS (predicted the wrong class):      {error_type_counts['WRONGCLASS']}",
        f"  FALSEPOS (predicted something not there):    {error_type_counts['FALSEPOS']}",
        "",
        "errors by class (both as the true class and as a wrongly-predicted class):",
    ]
    for name, count in per_class_error_counts.most_common():
        txt_lines.append(f"  {count:>4}  {name}")
    txt_path = output_dir / "errors_report.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print("\n".join(txt_lines))

    zip_path = None
    if make_zip:
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in output_dir.iterdir():
                zf.write(f, arcname=f"{output_dir.name}/{f.name}")
        print(f"Zipped: {zip_path}")

    print(f"\nAnnotated error images + report: {output_dir}")

    return {
        "n_images_checked": n_images_checked,
        "n_images_with_errors": n_images_with_errors,
        "error_type_counts": dict(error_type_counts),
        "output_dir": output_dir,
        "zip_path": zip_path,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml, overrides --dataset-dir auto-detection")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--output-dir", type=Path, default=None, help="Default: <dataset-dir>/prediction_errors")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions (default: 0.25)")
    ap.add_argument("--iou-match", type=float, default=0.5, help="IoU threshold to consider a prediction a match for a ground-truth box (default: 0.5)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--box-width", type=int, default=3)
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--font", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None, help="Only render the first N erroring images (default: all)")
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights file not found: {args.weights}")

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ultralytics is not installed. Install it with:\n    pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)
    output_dir = args.output_dir or (args.dataset_dir / "prediction_errors")

    run_find_errors(
        weights=args.weights, dataset_dir=args.dataset_dir, data_yaml=data_yaml, split=args.split,
        output_dir=output_dir, conf=args.conf, iou_match=args.iou_match, imgsz=args.imgsz,
        device=args.device, box_width=args.box_width, font_size=args.font_size, font_path=args.font,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
