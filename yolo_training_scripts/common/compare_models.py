#!/usr/bin/env python3
"""
compare_models.py

Side-by-side comparison of the YOLO and RF-DETR results, from the folders
written by sort_yolo_errors.py and sort_rfdetr_errors.py (run both on the SAME
split with the SAME --conf / --iou, or the comparison isn't fair).

Writes to --out-dir (default ./model_comparison):
    comparison.txt   headline accuracy, every failure type side by side, and
                     which individual images each model got right or wrong
    comparison.csv   the same numbers as a table
    comparison.png   a large-print bar chart of failure types for both models

Usage:
    python compare_models.py --yolo-dir ../yolo/runs/detect/runs/train/exp/error_sort_valid --transformer-dir ../runs/rfdetr/pretrained/error_sort_valid

You can also compare two runs of the same model (e.g. pretrained vs scratch) by
passing them as --yolo-dir / --transformer-dir and renaming with --label-a / --label-b.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from error_sorting import HEADLINE, PRIORITY


def load(d: Path):
    sj = d / "summary.json"
    if not sj.exists():
        raise SystemExit(f"{sj} not found -- is {d} the output folder of a sort_*_errors.py run?")
    summary = json.loads(sj.read_text())
    status = {}
    with open(d / "image_summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            status[row["image"]] = row["status"] == "correct"
    return summary, status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo-dir", type=Path, required=True, help="sorted-output folder for model A (YOLO)")
    ap.add_argument("--transformer-dir", type=Path, required=True, help="sorted-output folder for model B (RF-DETR)")
    ap.add_argument("--label-a", default="YOLO")
    ap.add_argument("--label-b", default="RF-DETR")
    ap.add_argument("--out-dir", type=Path, default=Path("model_comparison"))
    args = ap.parse_args()

    a, sa = load(args.yolo_dir)
    b, sb = load(args.transformer_dir)
    A, B = args.label_a, args.label_b
    if (a["split"], a["conf"], a["iou"]) != (b["split"], b["conf"], b["iou"]):
        print(f"[warn] the two runs used different settings (A: {a['split']} conf {a['conf']} IoU {a['iou']}; "
              f"B: {b['split']} conf {b['conf']} IoU {b['iou']}) -- the comparison is not apples-to-apples.")

    pct = lambda x, y: f"{100 * x / y:.1f}%" if y else "n/a"
    rows = [
        ("images fully correct", a["images_correct"], b["images_correct"]),
        ("objects found with exact class", a["objects_correct"], b["objects_correct"]),
    ]
    rows += [(k, a["headline_mistakes"][k], b["headline_mistakes"][k]) for k in HEADLINE]
    type_rows = [(t, a["mistakes_by_type"][t], b["mistakes_by_type"][t]) for t in PRIORITY]

    common_imgs = sorted(set(sa) & set(sb))
    both_ok = sum(sa[i] and sb[i] for i in common_imgs)
    only_a = sum(sa[i] and not sb[i] for i in common_imgs)
    only_b = sum(sb[i] and not sa[i] for i in common_imgs)
    both_bad = sum((not sa[i]) and (not sb[i]) for i in common_imgs)

    L = [f"{A} vs {B} -- split {a['split']}, conf >= {a['conf']}, IoU >= {a['iou']}", "=" * 78, "",
         f"{'':<38}{A:>14}{B:>14}",
         f"{'images (total)':<38}{a['images']:>14}{b['images']:>14}",
         f"{'images fully correct':<38}{a['images_correct']:>9} ({pct(a['images_correct'], a['images']):>6}){b['images_correct']:>5} ({pct(b['images_correct'], b['images']):>6})",
         f"{'objects labelled':<38}{a['objects']:>14}{b['objects']:>14}",
         f"{'objects found, exact class':<38}{a['objects_correct']:>9} ({pct(a['objects_correct'], a['objects']):>6}){b['objects_correct']:>5} ({pct(b['objects_correct'], b['objects']):>6})",
         "", "HEADLINE MISTAKES (lower is better)"]
    L += [f"  {k:<36}{a['headline_mistakes'][k]:>14}{b['headline_mistakes'][k]:>14}" for k in HEADLINE]
    L += ["", "EVERY FAILURE TYPE (individual mistakes)"]
    L += [f"  {t:<36}{x:>14}{y:>14}" for t, x, y in type_rows]
    L += ["", f"IMAGE BY IMAGE ({len(common_imgs)} images in both)",
          f"  both correct              {both_ok}",
          f"  only {A} correct".ljust(28) + f"{only_a}",
          f"  only {B} correct".ljust(28) + f"{only_b}",
          f"  both have a failure       {both_bad}"]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    with open(args.out_dir / "comparison.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["metric", A, B])
        wr.writerows(rows + type_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [(t, x, y) for t, x, y in type_rows if x or y] or type_rows[:1]
    fig, ax = plt.subplots(figsize=(16, 1.3 * len(shown) + 3.5))
    ys = range(len(shown))
    ax.barh([i - 0.2 for i in ys], [r[1] for r in shown], height=0.38, label=A, color="#2a6fdb")
    ax.barh([i + 0.2 for i in ys], [r[2] for r in shown], height=0.38, label=B, color="#e8590c")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0].replace("_", " ") for r in shown], fontsize=22)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=20)
    ax.set_xlabel("individual mistakes (fewer is better)", fontsize=22)
    ax.set_title(f"{A} vs {B}: mistakes by type", fontsize=28, pad=14)
    ax.legend(fontsize=22)
    for i, (_, x, y) in enumerate(shown):
        ax.text(x, i - 0.2, f" {x}", va="center", fontsize=18)
        ax.text(y, i + 0.2, f" {y}", va="center", fontsize=18)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "comparison.png", dpi=110)
    print(f"[info] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
