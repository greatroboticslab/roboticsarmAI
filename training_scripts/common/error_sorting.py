"""
error_sorting.py

The ONE place that decides what counts as a correct detection and what kind of
mistake every wrong one is. Both model-specific sorters use it, so YOLO and
RF-DETR are judged by identical rules and their results are directly
comparable:

    yolo/sort_yolo_errors.py            (ultralytics YOLO checkpoint)
    transformer/sort_rfdetr_errors.py   (RF-DETR checkpoint)

Each of those only has to say "given this image, what does the model predict?".
Everything else -- matching predictions to labels, naming the mistake, filing
the annotated image into a folder, writing the CSVs / summary.txt / summary.json
-- happens here.

Output folder layout (identical for both models):

    <out>/correct/                      every object found with the exact right class, nothing extra
    <out>/failure/<failure_type>/       any other image, filed under its most serious failure
    <out>/image_summary.csv             one row per image
    <out>/errors_report.csv             one row per individual mistake
    <out>/confusions.csv                true class -> predicted class counts
    <out>/summary.txt / summary.json    totals (json is what compare_models.py reads)

Failure types (most serious first -- an image is filed under the first one it contains):

    wrong_object                 different object (pen called a hammer)
    wrong_object_same_category   different object, same category in object_categories.yaml (gel pen vs ballpoint pen)
    missed_no_detection          real object, no usable prediction on it           (false negative)
    wrong_material_and_color     right object, material AND color wrong
    wrong_material               right object, wrong material
    wrong_color                  right object, wrong color
    wrong_class_same_attributes  different class id, identical object/material/color
    poor_localization            right class, but box overlaps too little          (false negative)
    missed_low_confidence        right class and place, confidence under --conf    (false negative)
    false_positive_background    a prediction with no real object there            (false positive)
    false_positive_duplicate     second box on an object that was already found    (false positive)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from eval_helpers import _normalize_group_key, build_category_index, load_category_map
from visualize_annotations import draw_single_box, load_font, short_label
from yolo_common import (
    find_split_dirs, list_image_label_pairs, load_data_yaml, parse_class_name,
    read_label_boxes, resolve_data_yaml,
)

TRUE_COLOR = (30, 160, 60)   # green: ground truth
PRED_COLOR = (215, 30, 30)   # red: model prediction

# Most serious first. An image is filed under the first type in this list that it contains.
PRIORITY = [
    "wrong_object",
    "wrong_object_same_category",
    "missed_no_detection",
    "wrong_material_and_color",
    "wrong_material",
    "wrong_color",
    "wrong_class_same_attributes",
    "poor_localization",
    "missed_low_confidence",
    "false_positive_background",
    "false_positive_duplicate",
]

# Plain-English roll-up of the types above into the headline groups. wrong_material_and_color
# counts under BOTH wrong_material and wrong_color; poor_localization counts as a false negative
# because the real object was not successfully found.
HEADLINE = {
    "wrong class (any kind)": ["wrong_object", "wrong_object_same_category", "wrong_material_and_color",
                               "wrong_material", "wrong_color", "wrong_class_same_attributes"],
    "wrong object": ["wrong_object", "wrong_object_same_category"],
    "wrong material": ["wrong_material", "wrong_material_and_color"],
    "wrong color": ["wrong_color", "wrong_material_and_color"],
    "false negative (object missed)": ["missed_no_detection", "missed_low_confidence", "poor_localization"],
    "false positive (extra box)": ["false_positive_background", "false_positive_duplicate"],
}

LOW_FLOOR = 0.05          # predictions below this are never looked at when explaining a miss
LOCALIZE_MIN_IOU = 0.1    # overlap needed to call a same-class box "poor localization" rather than unrelated

Pred = tuple  # (class_id, confidence, x1, y1, x2, y2)


# --------------------------------------------------------------------------- #
# geometry + matching
# --------------------------------------------------------------------------- #

def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_truth(
    gt_boxes: list[tuple[int, float, float, float, float]],
    pred_boxes: list[tuple[int, float, float, float, float, float]],  # (cls, conf, x1,y1,x2,y2)
    iou_thresh: float,
):
    """
    Greedy IoU matching (class-agnostic), highest-confidence predictions matched first.
    Returns (matches, unmatched_gt, unmatched_pred):
      matches: [(gt_index, pred_index, iou), ...] -- includes both correct AND wrong-class matches
      unmatched_gt: indices of ground-truth boxes with no matching prediction (missed detections)
      unmatched_pred: indices of predictions with no matching ground-truth box (false positives)
    """
    pred_order = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i][1], reverse=True)
    gt_used = [False] * len(gt_boxes)
    pred_used = [False] * len(pred_boxes)
    matches = []

    for pi in pred_order:
        _, _, *pbox = pred_boxes[pi]
        best_gi, best_iou = None, 0.0
        for gi, (_, *gbox) in enumerate(gt_boxes):
            if gt_used[gi]:
                continue
            score = iou(tuple(gbox), tuple(pbox))
            if score >= iou_thresh and score > best_iou:
                best_gi, best_iou = gi, score
        if best_gi is not None:
            gt_used[best_gi] = True
            pred_used[pi] = True
            matches.append((best_gi, pi, best_iou))

    unmatched_gt = [i for i, used in enumerate(gt_used) if not used]
    unmatched_pred = [i for i, used in enumerate(pred_used) if not used]
    return matches, unmatched_gt, unmatched_pred


# --------------------------------------------------------------------------- #
# classification (pure functions, no model needed)
# --------------------------------------------------------------------------- #

def classify_wrong_class(gt_cls: int, pred_cls: int, parsed: dict[int, dict], cat_index: dict[int, str] | None) -> str:
    g, p = parsed[gt_cls], parsed[pred_cls]
    n = _normalize_group_key
    if n(g["object"]) != n(p["object"]):
        if cat_index and cat_index.get(gt_cls) is not None and cat_index.get(gt_cls) == cat_index.get(pred_cls):
            return "wrong_object_same_category"
        return "wrong_object"
    bad_material = n(g["material"]) != n(p["material"])
    bad_color = n(g["color"]) != n(p["color"])
    if bad_material and bad_color:
        return "wrong_material_and_color"
    if bad_material:
        return "wrong_material"
    if bad_color:
        return "wrong_color"
    return "wrong_class_same_attributes"


def classify_image(gt, preds, conf: float, iou_thr: float, parsed, cat_index=None) -> dict:
    """
    gt:    [(cls, x1, y1, x2, y2)]
    preds: [(cls, confidence, x1, y1, x2, y2)]   -- ALL predictions down to LOW_FLOOR
    Returns {"errors": [...], "n_gt": int, "n_correct": int, "shown_preds": [...]}; each error is
    a dict with type / true_cls / pred_cls / conf / iou.
    """
    shown = [p for p in preds if p[1] >= conf]
    low = [p for p in preds if LOW_FLOOR <= p[1] < conf]
    matches, unmatched_gt, unmatched_pred = match_predictions_to_truth(gt, shown, iou_thr)

    errors, n_correct = [], 0
    for gi, pi, v in matches:
        if gt[gi][0] == shown[pi][0]:
            n_correct += 1
        else:
            errors.append({"type": classify_wrong_class(gt[gi][0], shown[pi][0], parsed, cat_index),
                           "true_cls": gt[gi][0], "pred_cls": shown[pi][0], "conf": shown[pi][1], "iou": v})

    explained_preds: set[int] = set()   # unmatched predictions already accounted for by a miss explanation
    for gi in unmatched_gt:
        gcls, gbox = gt[gi][0], tuple(gt[gi][1:])
        lc = [p for p in low if p[0] == gcls and iou(gbox, p[2:]) >= iou_thr]
        if lc:
            best = max(lc, key=lambda p: p[1])
            errors.append({"type": "missed_low_confidence", "true_cls": gcls, "pred_cls": gcls,
                           "conf": best[1], "iou": iou(gbox, best[2:])})
            continue
        loc = [(pi, iou(gbox, shown[pi][2:])) for pi in unmatched_pred
               if shown[pi][0] == gcls and LOCALIZE_MIN_IOU <= iou(gbox, shown[pi][2:]) < iou_thr]
        if loc:
            pi, v = max(loc, key=lambda t: t[1])
            explained_preds.add(pi)
            errors.append({"type": "poor_localization", "true_cls": gcls, "pred_cls": gcls,
                           "conf": shown[pi][1], "iou": v})
            continue
        errors.append({"type": "missed_no_detection", "true_cls": gcls, "pred_cls": None, "conf": None, "iou": None})

    for pi in unmatched_pred:
        if pi in explained_preds:
            continue
        pbox = tuple(shown[pi][2:])
        dup = any(iou(tuple(g[1:]), pbox) >= iou_thr for g in gt)
        errors.append({"type": "false_positive_duplicate" if dup else "false_positive_background",
                       "true_cls": None, "pred_cls": shown[pi][0], "conf": shown[pi][1], "iou": None})
    return {"errors": errors, "n_gt": len(gt), "n_correct": n_correct, "shown_preds": shown}


def primary_type(types: set[str]) -> str:
    for t in PRIORITY:
        if t in types:
            return t
    return "other"


# --------------------------------------------------------------------------- #
# shared CLI + driver
# --------------------------------------------------------------------------- #

def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dataset-dir", type=Path, required=True, help="Path to dataset/ (split folders + data.corrected.yaml)")
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml for class names (default: auto from --dataset-dir)")
    ap.add_argument("--split", default="valid", choices=["valid", "val", "test", "train"], help="Which split to sort (default valid)")
    ap.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for a prediction to count (default 0.3)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed to match a prediction to a real object (default 0.5)")
    ap.add_argument("--category-map", type=Path, default=None, help="object->category YAML (default: <dataset>/object_categories.yaml if present)")
    ap.add_argument("--output-dir", type=Path, default=None, help="Where to write the sorted images and reports")
    ap.add_argument("--plain", action="store_true", help="Copy the original images instead of drawing TRUE/PRED boxes on them")
    ap.add_argument("--box-width", type=int, default=3)
    ap.add_argument("--font-size", type=int, default=16)
    ap.add_argument("--font", type=Path, default=None, help="Path to a .ttf font (e.g. from C:\\Windows\\Fonts)")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N images (quick test)")


def load_class_names(data_yaml: Path) -> list[str]:
    raw = load_data_yaml(data_yaml)["names"]
    return [raw[k] for k in sorted(raw, key=int)] if isinstance(raw, dict) else list(raw)


def run_sort(
    args: argparse.Namespace,
    predict_fn: Callable[[Path, Image.Image], list[Pred]],
    model_label: str,
    checkpoint_label: str,
    default_out_dir: Path,
) -> dict:
    """
    predict_fn(image_path, PIL_image) -> [(class_id, confidence, x1, y1, x2, y2), ...] with every
    prediction down to LOW_FLOOR confidence (this function applies --conf itself).
    """
    split = "valid" if args.split == "val" else args.split
    split_dirs = find_split_dirs(args.dataset_dir)
    if split not in split_dirs:
        raise SystemExit(f"Split '{split}' not found under {args.dataset_dir} (found: {sorted(split_dirs)})")

    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)
    names = load_class_names(data_yaml)
    parsed = {i: parse_class_name(n) for i, n in enumerate(names)}

    cat_path = args.category_map
    if cat_path is None and (args.dataset_dir / "object_categories.yaml").exists():
        cat_path = args.dataset_dir / "object_categories.yaml"
    category_map = load_category_map(cat_path)
    cat_index = build_category_index(dict(enumerate(names)), category_map) if category_map else None

    out_dir = args.output_dir or default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("correct", "failure"):          # only ever clear what this script itself creates
        if (out_dir / sub).exists():
            shutil.rmtree(out_dir / sub)
    (out_dir / "correct").mkdir()

    font = load_font(args.font_size, args.font)
    pairs = list_image_label_pairs(split_dirs[split])
    if args.limit:
        pairs = pairs[: args.limit]

    image_rows, error_rows = [], []
    inst_counts, img_counts = Counter(), Counter()
    confusions = Counter()
    per_class = defaultdict(Counter)   # true class name -> Counter(error type)
    n_gt_total = n_correct_total = 0

    for n, (img_path, lbl_path) in enumerate(pairs, 1):
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        gt = [g for g in read_label_boxes(lbl_path, w, h) if 0 <= g[0] < len(names)]
        preds = [p for p in predict_fn(img_path, img) if 0 <= p[0] < len(names)]
        res = classify_image(gt, preds, args.conf, args.iou, parsed, cat_index)

        types = {e["type"] for e in res["errors"]}
        n_gt_total += res["n_gt"]
        n_correct_total += res["n_correct"]

        if not types:
            status, primary, folder = "correct", "", out_dir / "correct"
        else:
            primary = primary_type(types)
            status, folder = "failure", out_dir / "failure" / primary
            img_counts[primary] += 1
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / f"{img_path.stem}{img_path.suffix.lower()}"

        if args.plain:
            shutil.copy2(img_path, out_path)
        else:
            canvas = img.copy()
            draw = ImageDraw.Draw(canvas, "RGBA")
            for cid, x1, y1, x2, y2 in gt:
                draw_single_box(draw, (x1, y1, x2, y2), TRUE_COLOR, f"TRUE: {short_label(names[cid])}", font, args.box_width)
            for cid, pc, x1, y1, x2, y2 in res["shown_preds"]:
                draw_single_box(draw, (x1, y1, x2, y2), PRED_COLOR, f"PRED: {short_label(names[cid])} {pc:.2f}", font, args.box_width)
            canvas.save(out_path, quality=92)

        for e in res["errors"]:
            inst_counts[e["type"]] += 1
            t = names[e["true_cls"]] if e["true_cls"] is not None else ""
            p = names[e["pred_cls"]] if e["pred_cls"] is not None else ""
            if e["true_cls"] is not None:
                per_class[t][e["type"]] += 1
            if e["true_cls"] is not None and e["pred_cls"] is not None and e["true_cls"] != e["pred_cls"]:
                confusions[(t, p)] += 1
            error_rows.append({
                "image": img_path.name, "filed_under": primary or "correct", "error_type": e["type"],
                "true_class": t, "predicted_class": p,
                "confidence": f"{e['conf']:.4f}" if e["conf"] is not None else "",
                "iou": f"{e['iou']:.4f}" if e["iou"] is not None else "",
            })
        image_rows.append({
            "image": img_path.name, "status": status, "filed_under": primary or "correct",
            "all_error_types": ";".join(sorted(types, key=PRIORITY.index)),
            "objects": res["n_gt"], "objects_correct": res["n_correct"], "n_errors": len(res["errors"]),
        })
        if n % 25 == 0 or n == len(pairs):
            print(f"    processed {n}/{len(pairs)} images")

    def write_csv(name, fields, rows):
        with open(out_dir / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)

    write_csv("image_summary.csv", ["image", "status", "filed_under", "all_error_types", "objects", "objects_correct", "n_errors"], image_rows)
    write_csv("errors_report.csv", ["image", "filed_under", "error_type", "true_class", "predicted_class", "confidence", "iou"], error_rows)
    write_csv("confusions.csv", ["true_class", "predicted_class", "count"],
              [{"true_class": t, "predicted_class": p, "count": c} for (t, p), c in confusions.most_common()])

    n_img = len(image_rows)
    n_fail = sum(img_counts.values())
    n_ok = n_img - n_fail
    headline = {k: sum(inst_counts[t] for t in v) for k, v in HEADLINE.items()}
    summary = {
        "model": model_label, "checkpoint": checkpoint_label, "split": split,
        "conf": args.conf, "iou": args.iou,
        "images": n_img, "images_correct": n_ok, "images_with_failure": n_fail,
        "objects": n_gt_total, "objects_correct": n_correct_total,
        "images_by_failure_type": {t: img_counts[t] for t in PRIORITY},
        "mistakes_by_type": {t: inst_counts[t] for t in PRIORITY},
        "headline_mistakes": headline,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "  n/a"
    lines = [
        f"{model_label} error sort -- split: {split}   conf >= {args.conf}   IoU >= {args.iou}",
        f"checkpoint: {checkpoint_label}",
        "=" * 78, "",
        f"IMAGES:  {n_img} total   {n_ok} fully correct ({pct(n_ok, n_img)})   {n_fail} with at least one failure",
        f"OBJECTS: {n_gt_total} labelled   {n_correct_total} found with exact class ({pct(n_correct_total, n_gt_total)})",
        "", "HEADLINE (individual mistakes)",
    ]
    lines += [f"  {k:<34}{v:>6}" for k, v in headline.items()]
    lines += ["  (a wrong material AND color counts under both 'wrong material' and 'wrong color')", "",
              "FAILURE TYPES                  images filed here   individual mistakes"]
    for t in PRIORITY:
        if img_counts[t] or inst_counts[t]:
            lines.append(f"  {t:<30}{img_counts[t]:>6}{inst_counts[t]:>16}")
    lines += ["", "(an image is filed under its most serious failure; 'individual mistakes' counts every one, "
                  "so it can exceed the image count)", ""]
    if confusions:
        lines.append("MOST COMMON CONFUSIONS (true -> predicted)")
        for (t, p), c in confusions.most_common(15):
            lines.append(f"  {c:>3}x  {short_label(t)}  ->  {short_label(p)}")
        lines.append("")
    if per_class:
        lines.append("CLASSES WITH THE MOST MISTAKES")
        for t, cnt in sorted(per_class.items(), key=lambda kv: -sum(kv[1].values()))[:15]:
            detail = ", ".join(f"{k} {v}" for k, v in cnt.most_common())
            lines.append(f"  {sum(cnt.values()):>3}  {short_label(t)}   [{detail}]")
        lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[info] sorted images + reports written to {out_dir}")
    return summary
