#!/usr/bin/env python3
"""
evaluate.py

Runs standard object-detection evaluation (precision, recall, mAP50,
mAP50-95) for an already-trained YOLO checkpoint against a split of the
dataset (val, test, or even train, e.g. to sanity-check for overfitting).

Since every class here is really an (object, material, color) combination
(e.g. "Object ballpoint pen - Material plastic - Color red"), a plain
per-class table doesn't directly answer two questions people usually care
about:
  - How well does the model find THIS OBJECT, regardless of which material
    variant it is?
  - How well does the model recognize THIS MATERIAL, regardless of which
    object it's on?
So this script also aggregates the per-class numbers into a per-object
summary and a per-material summary, and combines the two into a single
weighted composite score you can tune with --object-weight/--material-weight.

Like train_yolo.py, this prefers dataset/data.corrected.yaml over the raw
dataset/data.yaml unless you pass --data explicitly.

Usage:
    # evaluate the best checkpoint from a training run against the validation split
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset

    # evaluate against the held-out test split instead
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset --split test

    # weight material recognition more heavily in the composite score
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset --material-weight 0.7 --object-weight 0.3
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from yolo_common import resolve_data_yaml, load_data_yaml, parse_class_name

METRIC_KEYS = ("precision", "recall", "mAP50", "mAP50-95")


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
    object_weight: float = 0.5,
    material_weight: float = 0.5,
) -> dict:
    """Run model.val(), aggregate per-object and per-material summaries, and write a report.
    Returns a dict with overall metrics, per-class/object/material breakdowns, and the composite score."""
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
        class_name = names.get(int(cls_idx), str(cls_idx))
        components = parse_class_name(class_name)
        per_class_rows.append({
            "class_id": int(cls_idx),
            "class_name": class_name,
            "object": components["object"] or class_name,
            "material": components["material"] or "(unspecified)",
            "color": components["color"],
            "instances": int(nt_per_class[cls_idx]) if nt_per_class is not None else 0,
            "precision": float(p_per_class[row_i]) if row_i < len(p_per_class) else float("nan"),
            "recall": float(r_per_class[row_i]) if row_i < len(r_per_class) else float("nan"),
            "mAP50": float(ap50_per_class[row_i]) if row_i < len(ap50_per_class) else float("nan"),
            "mAP50-95": float(ap_per_class[row_i]) if row_i < len(ap_per_class) else float("nan"),
        })

    per_object_rows = _aggregate(per_class_rows, "object")
    per_material_rows = _aggregate(per_class_rows, "material")

    total_w = object_weight + material_weight
    object_weight_n = object_weight / total_w if total_w else 0.5
    material_weight_n = material_weight / total_w if total_w else 0.5

    macro_class_avg = _macro_mean_all(per_class_rows)
    macro_object_avg = _macro_mean_all(per_object_rows)
    macro_material_avg = _macro_mean_all(per_material_rows)
    composite_per_metric = {
        m: object_weight_n * macro_object_avg[m] + material_weight_n * macro_material_avg[m]
        for m in METRIC_KEYS
    }
    composite = {
        "object_weight": object_weight_n,
        "material_weight": material_weight_n,
        "object_macro_mAP50-95": macro_object_avg["mAP50-95"],
        "material_macro_mAP50-95": macro_material_avg["mAP50-95"],
        "weighted_composite_score": composite_per_metric["mAP50-95"],
        "per_metric": composite_per_metric,
    }
    summary = {
        "overall (instance-weighted)": overall,
        "macro avg across classes": macro_class_avg,
        "macro avg across objects": macro_object_avg,
        "macro avg across materials": macro_material_avg,
        "weighted composite (object+material)": composite_per_metric,
    }

    out_dir = Path(getattr(metrics, "save_dir", Path(project) / name))
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_report(out_dir, weights, data_yaml, split, overall, per_class_rows, per_object_rows, per_material_rows, composite, summary)
    return {
        "overall": overall,
        "per_class": per_class_rows,
        "per_object": per_object_rows,
        "per_material": per_material_rows,
        "composite": composite,
        "summary": summary,
        "output_dir": str(out_dir),
    }


def _aggregate(rows: list[dict], key: str) -> list[dict]:
    """
    Macro-average the per-class rows by `key` (object or material): every
    row in a group counts equally, regardless of how many instances it has.
    This is deliberate -- it's what makes the per-material summary a fair
    read on material recognition instead of being dominated by whichever
    object happens to have the most photos. Total instance count is still
    reported alongside for context.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)

    out = []
    for group_val, group_rows in groups.items():
        n = len(group_rows)
        agg = {key: group_val, "n_classes": n, "instances": sum(r["instances"] for r in group_rows)}
        for m in METRIC_KEYS:
            agg[m] = sum(r[m] for r in group_rows) / n
        out.append(agg)
    return out


def _macro_mean(rows: list[dict], metric: str) -> float:
    if not rows:
        return 0.0
    return sum(r[metric] for r in rows) / len(rows)


def _macro_mean_all(rows: list[dict]) -> dict:
    return {m: _macro_mean(rows, m) for m in METRIC_KEYS}


def _format_table(rows: list[dict], key: str, key_label: str, key_width: int) -> list[str]:
    header = f"{key_label:<{key_width}}{'classes':>8}{'instances':>10}{'precision':>11}{'recall':>9}{'mAP50':>9}{'mAP50-95':>10}"
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda r: r["mAP50-95"]):
        lines.append(
            f"{str(row[key])[:key_width - 1]:<{key_width}}{row['n_classes']:>8}{row['instances']:>10}"
            f"{row['precision']:>11.4f}{row['recall']:>9.4f}{row['mAP50']:>9.4f}{row['mAP50-95']:>10.4f}"
        )
    return lines


def _write_report(
    out_dir: Path, weights: Path, data_yaml: Path, split: str,
    overall: dict, per_class_rows: list[dict], per_object_rows: list[dict],
    per_material_rows: list[dict], composite: dict, summary: dict,
):
    txt_path = out_dir / "evaluation_report.txt"
    csv_path = out_dir / "evaluation_per_class.csv"
    object_csv_path = out_dir / "evaluation_per_object.csv"
    material_csv_path = out_dir / "evaluation_per_material.csv"

    lines = [
        "=" * 70,
        "YOLO EVALUATION REPORT",
        "=" * 70,
        f"weights: {weights}",
        f"data:    {data_yaml}",
        f"split:   {split}",
        "",
        "Overall (standard YOLO detection metrics, instance-weighted across all classes):",
        f"  precision  : {overall['precision']:.4f}",
        f"  recall     : {overall['recall']:.4f}",
        f"  mAP50      : {overall['mAP50']:.4f}",
        f"  mAP50-95   : {overall['mAP50-95']:.4f}",
        "",
        f"Weighted composite score: {composite['weighted_composite_score']:.4f}",
        f"  = {composite['object_weight']:.2f} * object_macro_mAP50-95 ({composite['object_macro_mAP50-95']:.4f})"
        f" + {composite['material_weight']:.2f} * material_macro_mAP50-95 ({composite['material_macro_mAP50-95']:.4f})",
        "  (macro-averaged: every object/material counts equally regardless of instance count."
        " Tune with --object-weight/--material-weight. This is a custom summary, not a standard YOLO metric.)",
        "",
        "Per-class (every object+material+color class):",
    ]
    header = f"{'class':<55}{'instances':>10}{'precision':>11}{'recall':>9}{'mAP50':>9}{'mAP50-95':>10}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in sorted(per_class_rows, key=lambda r: r["mAP50-95"]):
        lines.append(
            f"{row['class_name'][:54]:<55}{row['instances']:>10}"
            f"{row['precision']:>11.4f}{row['recall']:>9.4f}{row['mAP50']:>9.4f}{row['mAP50-95']:>10.4f}"
        )

    lines.append("")
    lines.append("Per-object summary (material/color variants of the same object combined, macro-averaged):")
    lines.extend(_format_table(per_object_rows, "object", "object", 45))

    lines.append("")
    lines.append("Per-material summary (all objects sharing a material combined, macro-averaged):")
    lines.extend(_format_table(per_material_rows, "material", "material", 25))

    lowest_classes = sorted(per_class_rows, key=lambda r: r["mAP50-95"])[:5]
    if lowest_classes:
        lines.append("")
        lines.append("Lowest mAP50-95 classes (worth a closer look):")
        for row in lowest_classes:
            lines.append(f"  {row['class_name']}: mAP50-95={row['mAP50-95']:.4f}, instances={row['instances']}")

    lowest_materials = sorted(per_material_rows, key=lambda r: r["mAP50-95"])[:5]
    if lowest_materials:
        lines.append("")
        lines.append("Lowest mAP50-95 materials (worth a closer look):")
        for row in lowest_materials:
            lines.append(f"  {row['material']}: mAP50-95={row['mAP50-95']:.4f}, classes={row['n_classes']}, instances={row['instances']}")

    # ---- final summary: every metric, every level of aggregation, in one table ----
    lines.append("")
    lines.append("=" * 70)
    lines.append("SUMMARY -- COMPLETE RESULTS FOR EVERY METRIC")
    lines.append("=" * 70)
    summary_header = f"{'':<38}{'precision':>11}{'recall':>9}{'mAP50':>9}{'mAP50-95':>10}"
    lines.append(summary_header)
    lines.append("-" * len(summary_header))
    for label, vals in summary.items():
        lines.append(
            f"{label:<38}{vals['precision']:>11.4f}{vals['recall']:>9.4f}{vals['mAP50']:>9.4f}{vals['mAP50-95']:>10.4f}"
        )
    lines.append("")
    lines.append("  'overall (instance-weighted)'          -- standard YOLO metrics, larger classes count more")
    lines.append("  'macro avg across classes'             -- every object+material+color class counts equally")
    lines.append("  'macro avg across objects'             -- every distinct object counts equally, materials combined")
    lines.append("  'macro avg across materials'           -- every distinct material counts equally, objects combined")
    lines.append("  'weighted composite (object+material)' -- object/material macro averages blended per --object-weight/--material-weight")

    txt_path.write_text("\n".join(lines), encoding="utf-8")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "object", "material", "color", "instances", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        for row in per_class_rows:
            writer.writerow(row)

    with open(object_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["object", "n_classes", "instances", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        for row in per_object_rows:
            writer.writerow(row)

    with open(material_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["material", "n_classes", "instances", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        for row in per_material_rows:
            writer.writerow(row)

    summary_csv_path = out_dir / "evaluation_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        for label, vals in summary.items():
            writer.writerow({"level": label, **{m: vals[m] for m in METRIC_KEYS}})

    print("\n".join(lines))
    print(f"\nFull report: {txt_path}")
    print(f"Per-class CSV: {csv_path}")
    print(f"Per-object CSV: {object_csv_path}")
    print(f"Per-material CSV: {material_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")


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
    ap.add_argument("--object-weight", type=float, default=0.5, help="Weight given to object-level (macro) mAP50-95 in the composite score (default: 0.5)")
    ap.add_argument("--material-weight", type=float, default=0.5, help="Weight given to material-level (macro) mAP50-95 in the composite score (default: 0.5)")
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
        object_weight=args.object_weight,
        material_weight=args.material_weight,
    )


if __name__ == "__main__":
    main()

