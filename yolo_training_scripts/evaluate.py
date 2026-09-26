#!/usr/bin/env python3
"""
evaluate.py

Runs standard object-detection evaluation (precision, recall, mAP50,
mAP50-95) for an already-trained YOLO checkpoint against a split of the
dataset (val, test, or even train, e.g. to sanity-check for overfitting).

Since every class here is really an (object, material, color) combination
(e.g. "Object ballpoint pen - Material plastic - Color red"), a plain
per-class table doesn't directly answer questions people usually care about:
  - How well does the model find THIS OBJECT, regardless of which material
    variant it is? (per-object summary)
  - How well does the model recognize THIS MATERIAL, regardless of which
    object it's on? (per-material summary)
  - How well does the model get the general CATEGORY right, even if it
    mixes up two similar objects within that category (e.g. calling a gel
    pen a ballpoint pen is still useful -- it correctly found "a pen")?
    (per-category summary, see --category-map below)
The first two are macro-averaged mAP-based summaries; the composite score
combines them and is tunable with --object-weight/--material-weight.

The category summary is a genuinely different kind of metric: it's built
from the confusion matrix (what got predicted as what), not from mAP, so it
can give credit for "right category, wrong exact object" -- something a
per-class mAP table structurally can't express, since each class is scored
independently of what any other class predicted. It requires you to define
which objects belong to which category (this can't be reliably guessed from
object names alone) via a small YAML/JSON mapping file -- see --category-map.

Like train_yolo.py, this prefers dataset/data.corrected.yaml over the raw
dataset/data.yaml unless you pass --data explicitly.

Usage:
    # evaluate the best checkpoint from a training run against the validation split
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset

    # evaluate against the held-out test split instead
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset --split test

    # weight material recognition more heavily in the composite score
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset --material-weight 0.7 --object-weight 0.3

    # also report category-level accuracy (e.g. "pen" covering both ballpoint and gel pen)
    python evaluate.py --weights runs/train/exp/weights/best.pt --dataset-dir ../dataset --category-map ../dataset/object_categories.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from yolo_common import resolve_data_yaml, load_data_yaml, parse_class_name

METRIC_KEYS = ("precision", "recall", "mAP50", "mAP50-95")


def load_category_map(path: Path | None) -> dict[str, str]:
    """
    Load an object-name -> category-name mapping from a small YAML/JSON file, e.g.:
        ballpoint pen: pen
        gel pen: pen
        precision screwdriver: screwdriver
        flathead screwdriver: screwdriver
    Any object not listed simply isn't grouped with anything (see build_category_index).
    Returns {} if path is None or the file doesn't exist.
    """
    if path is None or not Path(path).exists():
        return {}
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {str(k).strip(): str(v).strip() for k, v in data.items()}


def build_category_index(names: dict[int, str], category_map: dict[str, str]) -> dict[int, str]:
    """Map each class id -> its category name. Objects not in category_map fall back to
    their own object name as a singleton category, so this is always safe to call even
    with an empty/partial mapping -- it just won't group anything extra."""
    index = {}
    for cls_id, name in names.items():
        obj = parse_class_name(name)["object"] or name
        index[cls_id] = category_map.get(obj, obj)
    return index


def compute_category_confusion(matrix, category_index: dict[int, str], nc: int) -> list[dict]:
    """
    Aggregate ultralytics' confusion matrix (matrix[predicted, true], with index `nc`
    reserved for "background"/no-detection) into per-category precision/recall/F1.
    A prediction counts as a category true positive whenever the predicted class and
    the true class map to the same category, even if they're different exact classes --
    that's the whole point: "right general category, wrong exact object" still counts.
    """
    categories = sorted(set(category_index.values()))
    rows = []
    for cat in categories:
        cat_class_ids = {cid for cid, c in category_index.items() if c == cat}
        tp = fp = fn = 0
        for i in range(nc + 1):  # predicted axis, including background (nc)
            i_in_cat = i in cat_class_ids
            for j in range(nc + 1):  # true axis, including background (nc)
                if i == nc and j == nc:
                    continue  # background-vs-background isn't a real event
                j_in_cat = j in cat_class_ids
                count = matrix[i, j]
                if count == 0:
                    continue
                if i_in_cat and j_in_cat:
                    tp += count
                elif i_in_cat and not j_in_cat:
                    fp += count
                elif not i_in_cat and j_in_cat:
                    fn += count
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "category": cat,
            "n_classes": len(cat_class_ids),
            "instances": int(tp + fn),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall, "f1": f1,
        })
    return rows


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
    category_map_path: Path | None = None,
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

    # Category summary: confusion-matrix based (see module docstring), only computed
    # when a mapping is supplied -- without one, there's nothing extra to group by.
    per_category_rows: list[dict] = []
    category_map = load_category_map(category_map_path)
    if category_map:
        cm = getattr(metrics, "confusion_matrix", None)
        if cm is not None and hasattr(cm, "matrix"):
            nc = len(names)
            category_index = build_category_index(names, category_map)
            per_category_rows = compute_category_confusion(cm.matrix, category_index, nc)
        else:
            print("[warn] --category-map was given but this ultralytics version didn't return a confusion matrix; skipping category summary.")

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
    _write_report(out_dir, weights, data_yaml, split, overall, per_class_rows, per_object_rows, per_material_rows, composite, summary, per_category_rows)
    return {
        "overall": overall,
        "per_class": per_class_rows,
        "per_object": per_object_rows,
        "per_material": per_material_rows,
        "per_category": per_category_rows,
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
    per_category_rows: list[dict],
):
    txt_path = out_dir / "evaluation_report.txt"
    csv_path = out_dir / "evaluation_per_class.csv"
    object_csv_path = out_dir / "evaluation_per_object.csv"
    material_csv_path = out_dir / "evaluation_per_material.csv"
    category_csv_path = out_dir / "evaluation_per_category.csv"

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

    if per_category_rows:
        lines.append("")
        lines.append("Per-category summary (confusion-matrix based, NOT mAP -- a detection still counts")
        lines.append("as correct if it gets the general category right, even if it mixes up the exact")
        lines.append("object within that category, e.g. calling a gel pen a ballpoint pen still counts):")
        cat_header = f"{'category':<30}{'classes':>8}{'instances':>10}{'TP':>6}{'FP':>6}{'FN':>6}{'precision':>11}{'recall':>9}{'F1':>8}"
        lines.append(cat_header)
        lines.append("-" * len(cat_header))
        for row in sorted(per_category_rows, key=lambda r: r["f1"]):
            lines.append(
                f"{row['category'][:29]:<30}{row['n_classes']:>8}{row['instances']:>10}"
                f"{row['tp']:>6}{row['fp']:>6}{row['fn']:>6}"
                f"{row['precision']:>11.4f}{row['recall']:>9.4f}{row['f1']:>8.4f}"
            )

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

    if per_category_rows:
        with open(category_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["category", "n_classes", "instances", "tp", "fp", "fn", "precision", "recall", "f1"])
            writer.writeheader()
            for row in per_category_rows:
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
    if per_category_rows:
        print(f"Per-category CSV: {category_csv_path}")
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
    ap.add_argument(
        "--category-map", type=Path, default=None,
        help="YAML/JSON file mapping object name -> category name (e.g. 'ballpoint pen: pen', "
             "'gel pen: pen') to additionally report confusion-matrix-based category accuracy, "
             "which gives credit for getting the general category right even if the exact object "
             "is wrong. Default: auto-use <dataset-dir>/object_categories.yaml if it exists.",
    )
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights file not found: {args.weights}")

    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)

    if args.split == "test" and args.dataset_dir is not None and not (args.dataset_dir / "test" / "images").exists():
        raise SystemExit(f"--split test was requested but {args.dataset_dir / 'test'} doesn't exist.")

    category_map_path = args.category_map
    if category_map_path is None and args.dataset_dir is not None:
        default_map = args.dataset_dir / "object_categories.yaml"
        if default_map.exists():
            print(f"[info] using category map: {default_map}")
            category_map_path = default_map

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
        category_map_path=category_map_path,
    )


if __name__ == "__main__":
    main()

