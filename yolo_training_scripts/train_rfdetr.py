#!/usr/bin/env python3
"""
train_rfdetr.py

Trains Roboflow's RF-DETR (a DINOv2 vision-transformer detector) on the same
dataset/ folder the YOLO scripts use, then evaluates it.

It reads the same inputs as train_yolo.py / evaluate.py:
  - dataset/data.corrected.yaml if present (output of build_dataset.py),
    otherwise dataset/data.yaml
  - dataset/{train,valid,test}/{images,labels}  (YOLO txt labels)
Class names, class order and label files are used exactly as-is.

RF-DETR insists on a folder whose data.yaml is literally called data.yaml, so
this script stages a small working folder (a copy of your split folders)
under <output>/dataset_staged. Your real dataset/ folder is never modified.

Evaluation is done two ways:
  1. RF-DETR's own COCO metrics (mAP50, mAP50-95, ...) on the val/test split.
  2. The same breakdown evaluate.py gives you, computed from the model's
     actual predictions: precision / recall / F1 at one confidence threshold
     and IoU 0.5, per class, per OBJECT, per MATERIAL, per COLOR, and (if a
     category map exists) per CATEGORY, plus the weighted composite
     (object > material > color). Note these are P/R/F1 numbers, not mAP, so
     they are not directly comparable to evaluate.py's mAP columns -- but they
     ARE comparable between RF-DETR runs and, if you want, a YOLO model scored
     the same way.

Usage:
    pip install "rfdetr[train]"

    # train + evaluate (val split, plus test split if the dataset has one)
    python train_rfdetr.py --dataset-dir ../dataset

    # evaluate an already-trained checkpoint only
    python train_rfdetr.py --dataset-dir ../dataset --eval-only --weights runs/rfdetr/exp/checkpoint_best_total.pth

    # on a 6GB GPU: small model, fp32 (avoids fp16 issues on GTX 16-series)
    python train_rfdetr.py --dataset-dir ../dataset --model small --device cuda --no-amp --batch 4
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from yolo_common import (
    find_split_dirs,
    list_image_label_pairs,
    load_data_yaml,
    parse_class_name,
    read_label_boxes,
    resolve_data_yaml,
)
from evaluate import (
    DEFAULT_COLOR_WEIGHT,
    DEFAULT_MATERIAL_WEIGHT,
    DEFAULT_OBJECT_WEIGHT,
    _normalize_group_key,
    build_category_index,
    load_category_map,
    normalize_weights,
)

MODEL_SIZES = ["nano", "small", "medium", "base", "large"]

# Rows of the summary table: (label, level key)
LEVELS = ("class", "object", "material", "color", "category")


# --------------------------------------------------------------------------- #
# dataset staging
# --------------------------------------------------------------------------- #

def get_class_names(data_yaml: Path) -> list[str]:
    names = load_data_yaml(data_yaml).get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    if isinstance(names, list):
        return list(names)
    raise SystemExit(f"{data_yaml} has no usable 'names' entry")


def _copy_split(src: Path, dst: Path) -> None:
    # Copy rather than symlink: RF-DETR rejects split folders that resolve outside the dataset root.
    # The dataset is small, so a fresh copy each run is cheap and keeps the staged folder in sync.
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def stage_dataset(dataset_dir: Path, data_yaml: Path, out_dir: Path) -> Path:
    """Build <out_dir>/dataset_staged with data.yaml + train/valid/test in the layout RF-DETR expects."""
    import yaml

    splits = find_split_dirs(dataset_dir)
    for needed in ("train", "valid"):
        if needed not in splits:
            raise SystemExit(f"{dataset_dir} is missing {needed}/images and {needed}/labels")
    staged = out_dir / "dataset_staged"
    staged.mkdir(parents=True, exist_ok=True)
    for split, path in splits.items():
        _copy_split(path, staged / split)
    cfg = {
        "train": "train/images",
        "val": "valid/images",
        "nc": len(get_class_names(data_yaml)),
        "names": get_class_names(data_yaml),
    }
    if "test" in splits:
        cfg["test"] = "test/images"
    with open(staged / "data.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[info] staged dataset at {staged} ({', '.join(splits)})")
    return staged


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #

def model_class(size: str):
    import rfdetr

    name = {
        "nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
        "base": "RFDETRBase", "large": "RFDETRLarge",
    }[size]
    cls = getattr(rfdetr, name, None)
    if cls is None:
        raise SystemExit(f"This rfdetr version has no {name}; upgrade with: pip install -U rfdetr")
    return cls


def find_best_checkpoint(out_dir: Path) -> Path | None:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        p = out_dir / name
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- #
# prediction-based evaluation (P / R / F1 at fixed conf + IoU)
# --------------------------------------------------------------------------- #

def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_counts(per_image: list[tuple[list, list]], group_of: dict[int, str], iou_thr: float) -> dict[str, dict]:
    """
    per_image: [(gt, preds)] where gt = [(cls, x1,y1,x2,y2)], preds = [(cls, conf, x1,y1,x2,y2)].
    A prediction counts as a TP when it overlaps an unmatched GT box with IoU >= iou_thr AND both
    map to the same group (the group is the class itself for the per-class level, the object for
    the per-object level, and so on). Highest-confidence predictions match first.
    """
    stats: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "gt": 0})
    for gt, preds in per_image:
        gt_g = [(group_of.get(c, f"class{c}"), box) for c, *box in gt]
        for g, _ in gt_g:
            stats[g]["gt"] += 1
        used = [False] * len(gt_g)
        for c, conf, *box in sorted(preds, key=lambda p: -p[1]):
            g = group_of.get(c, f"class{c}")
            best, best_iou = -1, iou_thr
            for i, (gg, gbox) in enumerate(gt_g):
                if used[i] or gg != g:
                    continue
                v = _iou(box, gbox)
                if v >= best_iou:
                    best, best_iou = i, v
            if best >= 0:
                used[best] = True
                stats[g]["tp"] += 1
            else:
                stats[g]["fp"] += 1
        for i, (g, _) in enumerate(gt_g):
            if not used[i]:
                stats[g]["fn"] += 1
    return stats


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def rows_from_stats(stats: dict[str, dict], key: str) -> list[dict]:
    rows = []
    for g, s in sorted(stats.items()):
        if s["gt"] == 0 and s["fp"] == 0:
            continue
        p, r, f = prf(s["tp"], s["fp"], s["fn"])
        rows.append({key: g, "instances": s["gt"], "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                     "precision": p, "recall": r, "f1": f})
    return rows


def macro(rows: list[dict], metric: str) -> float:
    # only groups that actually have ground truth count toward the macro average
    vals = [r[metric] for r in rows if r["instances"] > 0]
    return sum(vals) / len(vals) if vals else 0.0


def predict_split(model, split_dir: Path, names: list[str], conf: float) -> list[tuple[list, list]]:
    from PIL import Image

    name_to_id = {n: i for i, n in enumerate(names)}
    out = []
    pairs = list_image_label_pairs(split_dir)
    for n, (img_path, lbl_path) in enumerate(pairs, 1):
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            det = model.predict(im, threshold=conf)
        gt = read_label_boxes(lbl_path, w, h)
        preds = []
        det_names = det.data.get("class_name") if getattr(det, "data", None) else None
        for k in range(len(det)):
            if det_names is not None and str(det_names[k]) in name_to_id:
                cid = name_to_id[str(det_names[k])]
            else:
                cid = int(det.class_id[k])
            if not 0 <= cid < len(names):
                continue  # not one of this dataset's classes (e.g. an untrained head's spare slot)
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[k])
            preds.append((cid, float(det.confidence[k]), x1, y1, x2, y2))
        out.append((gt, preds))
        if n % 25 == 0 or n == len(pairs):
            print(f"    predicted {n}/{len(pairs)} images")
    return out


def analyse(per_image, names, category_map, weights, iou_thr) -> dict:
    parsed = {i: parse_class_name(n) for i, n in enumerate(names)}
    # group labels are normalized (case/whitespace) so "Red" and "red " are one color
    maps = {
        "class": {i: n for i, n in enumerate(names)},
        "object": {i: _normalize_group_key(p["object"] or names[i]) for i, p in parsed.items()},
        "material": {i: _normalize_group_key(p["material"] or "(unknown)") for i, p in parsed.items()},
        "color": {i: _normalize_group_key(p["color"] or "(unknown)") for i, p in parsed.items()},
    }
    if category_map:
        maps["category"] = build_category_index(dict(enumerate(names)), category_map)

    results = {lvl: rows_from_stats(match_counts(per_image, m, iou_thr), lvl) for lvl, m in maps.items()}

    # overall (micro) = every class-correct match
    overall = {"tp": 0, "fp": 0, "fn": 0, "gt": 0}
    for s in match_counts(per_image, maps["class"], iou_thr).values():
        for k in overall:
            overall[k] += s[k]
    p, r, f = prf(overall["tp"], overall["fp"], overall["fn"])

    o, m, c = weights
    composite = o * macro(results["object"], "f1") + m * macro(results["material"], "f1") + c * macro(results["color"], "f1")
    return {"rows": results, "overall": {**overall, "precision": p, "recall": r, "f1": f},
            "composite_f1": composite, "weights": weights}


def write_analysis(res: dict, out_dir: Path, split: str, conf: float, iou_thr: float, coco: dict | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"RF-DETR evaluation -- split: {split}  (conf >= {conf}, IoU >= {iou_thr})", "=" * 78, ""]
    if coco:
        lines += ["COCO METRICS (from RF-DETR's own evaluator)"]
        lines += [f"  {k:<28}{v:.4f}" if isinstance(v, (int, float)) else f"  {k:<28}{v}" for k, v in coco.items()]
        lines.append("")
    o = res["overall"]
    lines += [
        "OVERALL (instance-weighted, exact class must match)",
        f"  precision {o['precision']:.4f}   recall {o['recall']:.4f}   F1 {o['f1']:.4f}",
        f"  TP {o['tp']}  FP {o['fp']}  FN {o['fn']}  ground-truth instances {o['gt']}",
        "",
        "WEIGHTED COMPOSITE (macro F1; object/material/color weights "
        f"{res['weights'][0]:.2f}/{res['weights'][1]:.2f}/{res['weights'][2]:.2f})",
        f"  {res['composite_f1']:.4f}",
        "",
    ]
    summary = [("overall (instance-weighted)", o["precision"], o["recall"], o["f1"])]
    for lvl in LEVELS:
        rows = res["rows"].get(lvl)
        if not rows:
            continue
        w = max(len(str(r[lvl])) for r in rows)
        w = min(max(w, len(lvl)), 70)
        lines.append(f"PER {lvl.upper()}  (sorted by F1, worst first)")
        lines.append(f"  {lvl:<{w}} {'n':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'prec':>7} {'rec':>7} {'F1':>7}")
        for r in sorted(rows, key=lambda r: r["f1"]):
            lines.append(f"  {str(r[lvl])[:w]:<{w}} {r['instances']:>5} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
                         f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")
        lines.append("")
        summary.append((f"macro avg across {lvl}es" if lvl == "class" else f"macro avg across {lvl}s",
                        macro(rows, "precision"), macro(rows, "recall"), macro(rows, "f1")))
        with open(out_dir / f"evaluation_per_{lvl}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=[lvl, "instances", "tp", "fp", "fn", "precision", "recall", "f1"])
            wr.writeheader()
            wr.writerows(rows)
    lines += ["SUMMARY -- COMPLETE RESULTS FOR EVERY LEVEL",
              f"  {'':<40}{'precision':>10}{'recall':>9}{'F1':>9}"]
    lines += [f"  {n:<40}{p:>10.4f}{r:>9.4f}{f:>9.4f}" for n, p, r, f in summary]
    lines.append(f"  {'weighted composite (object/material/color)':<40}{'':>10}{'':>9}{res['composite_f1']:>9.4f}")
    with open(out_dir / "evaluation_summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["level", "precision", "recall", "f1"])
        wr.writerows(summary)
        wr.writerow(["weighted composite", "", "", res["composite_f1"]])
    report = out_dir / f"evaluation_report_{split}.txt"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[info] wrote {report}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, default=None, help="Path to dataset/ (auto-picks data.corrected.yaml)")
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml (class names); the split folders still come from --dataset-dir")
    ap.add_argument("--model", default="small", choices=MODEL_SIZES,
                    help="RF-DETR size (default small). nano/small suit a 6GB GPU and a ~250 image dataset.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4, help="Per-step batch size (default 4; lower it if you run out of GPU memory)")
    ap.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps; effective batch = batch * grad-accum (default 4 -> 16)")
    ap.add_argument("--lr", type=float, default=None, help="Learning rate (default: RF-DETR's own, 1e-4)")
    ap.add_argument("--resolution", type=int, default=None, help="Input resolution (must be divisible by the model's patch size * windows; default: model's own)")
    ap.add_argument("--device", default=None, help="cuda, cuda:0, cpu or mps. Default: RF-DETR auto-selects")
    ap.add_argument("--workers", type=int, default=2, help="Dataloader workers (use 0 if training hangs on Windows)")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Random init instead of COCO-pretrained weights. Almost always much worse on a small dataset; "
                         "mainly for offline smoke tests / ablations.")
    ap.add_argument("--no-amp", action="store_true", help="Train in full fp32 (use on GTX 16-series cards if you hit fp16/cuDNN errors)")
    ap.add_argument("--patience", type=int, default=15, help="Early-stopping patience in epochs (0 disables)")
    ap.add_argument("--project", default="runs/rfdetr", help="Parent folder for run outputs")
    ap.add_argument("--name", default="exp", help="Run name (subfolder under --project)")
    ap.add_argument("--eval-only", action="store_true", help="Skip training; evaluate --weights (or the run's best checkpoint)")
    ap.add_argument("--weights", type=Path, default=None, help="Checkpoint to evaluate with --eval-only (default: <run>/checkpoint_best_total.pth)")
    ap.add_argument("--splits", nargs="+", default=None, choices=["val", "test"], help="Splits to evaluate (default: val, plus test if present)")
    ap.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for the P/R/F1 breakdown (default 0.3)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for the P/R/F1 breakdown (default 0.5)")
    ap.add_argument("--category-map", type=Path, default=None, help="object->category YAML (default: <dataset>/object_categories.yaml if it exists)")
    ap.add_argument("--object-weight", type=float, default=DEFAULT_OBJECT_WEIGHT)
    ap.add_argument("--material-weight", type=float, default=DEFAULT_MATERIAL_WEIGHT)
    ap.add_argument("--color-weight", type=float, default=DEFAULT_COLOR_WEIGHT)
    args = ap.parse_args()

    if args.dataset_dir is None:
        raise SystemExit("--dataset-dir is required (the split folders are read from it)")
    try:
        import rfdetr  # noqa: F401
    except ImportError:
        print('rfdetr is not installed. Install it with:\n    pip install "rfdetr[train]"', file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = resolve_data_yaml(args.dataset_dir, args.data)
    names = get_class_names(data_yaml)
    staged = stage_dataset(args.dataset_dir, data_yaml, out_dir)
    weights = normalize_weights(args.object_weight, args.material_weight, args.color_weight)

    cat_path = args.category_map
    if cat_path is None and (args.dataset_dir / "object_categories.yaml").exists():
        cat_path = args.dataset_dir / "object_categories.yaml"
        print(f"[info] using category map: {cat_path}")
    category_map = load_category_map(cat_path)

    ModelCls = model_class(args.model)
    common = dict(dataset_dir=str(staged), output_dir=str(out_dir))
    if args.device:
        common["device"] = args.device
    if args.resolution:
        common["resolution"] = args.resolution

    if not args.eval_only:
        try:
            import torch
            has_gpu = torch.cuda.is_available() or (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
            if args.device is None and not has_gpu:
                print("[warn] no CUDA/MPS device found -- training will run on CPU and be very slow. "
                      "Install a CUDA build of PyTorch (https://pytorch.org/get-started/locally/) and pass --device cuda.")
        except ImportError:
            pass
        model = ModelCls(pretrain_weights=None) if args.from_scratch else ModelCls()
        train_kwargs = dict(
            common,
            epochs=args.epochs,
            batch_size=args.batch,
            grad_accum_steps=args.grad_accum,
            num_workers=args.workers,
            early_stopping=args.patience > 0,
            early_stopping_patience=max(args.patience, 1),
            tensorboard=False,
        )
        if args.lr is not None:
            train_kwargs["lr"] = args.lr
        if args.no_amp:
            train_kwargs["amp_dtype"] = None
        model.train(**train_kwargs)
        print("\nTraining complete.")
        ckpt = find_best_checkpoint(out_dir)
    else:
        ckpt = args.weights or find_best_checkpoint(out_dir)

    if ckpt is None or not Path(ckpt).exists():
        raise SystemExit(f"No checkpoint found to evaluate (looked in {out_dir}); pass --weights")
    print(f"[info] evaluating checkpoint: {ckpt}")
    model = ModelCls.from_checkpoint(str(ckpt)) if hasattr(ModelCls, "from_checkpoint") else ModelCls(pretrain_weights=str(ckpt))
    # class names come from the dataset yaml, so reports always use the full "Object ... Pdfname" names
    splits = args.splits or (["val"] + (["test"] if (staged / "test").exists() else []))
    split_dir_name = {"val": "valid", "test": "test"}

    for split in splits:
        print(f"\n--- evaluating on '{split}' split ---")
        coco = None
        try:
            eval_kwargs = dict(common, num_workers=args.workers, batch_size=args.batch)
            raw = model.evaluate(split=split, **eval_kwargs)
            coco = {k: v for k, v in (raw or {}).items() if isinstance(v, (int, float)) and "/AP/" not in k}
        except Exception as e:  # keep going: the prediction-based report below doesn't need it
            print(f"[warn] RF-DETR COCO evaluation failed ({type(e).__name__}: {e}); continuing with P/R/F1 report")
        per_image = predict_split(model, staged / split_dir_name[split], names, args.conf)
        res = analyse(per_image, names, category_map, weights, args.iou)
        write_analysis(res, out_dir / f"eval_{split}", split, args.conf, args.iou, coco)

    print(f"\nRun directory: {out_dir}")
    print(f"Best weights:  {ckpt}")


if __name__ == "__main__":
    main()
