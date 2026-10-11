#!/usr/bin/env python3
"""
collect_examples.py

Fills in the picture slots in a model's README. After sort_yolo_errors.py or
sort_rfdetr_errors.py has finished, this copies a few example images per error
type out of the sorted folders into that model's results/ folder under the
fixed names the README already links to:

    results/correct/example_1.jpg, example_2.jpg
    results/wrong_material/example_1.jpg, example_2.jpg
    ...

It also writes results/examples_index.csv (which source image each example is)
and results/examples_captions.md (a caption for each example -- the true class
vs. what the model said -- ready to paste under the pictures in the README).

Types that have no failures keep their placeholder picture. The same example
images are re-picked identically every time (--seed), so re-running won't
shuffle your README; use --pick to choose differently.

Usage:
    python collect_examples.py --sorted-dir ../runs/rfdetr/pretrained/error_sort_valid --out-dir ../transformer/results
    python collect_examples.py --sorted-dir ../yolo/runs/detect/runs/train/exp/error_sort_valid --out-dir ../yolo/results --per-type 3
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from error_sorting import PRIORITY

TYPES = ["correct"] + PRIORITY
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sorted-dir", type=Path, required=True, help="Output folder of sort_*_errors.py (has correct/ and failure/)")
    ap.add_argument("--out-dir", type=Path, required=True, help="The model's results/ folder (yolo/results or transformer/results)")
    ap.add_argument("--per-type", type=int, default=2, help="Examples to copy per error type (default 2 -- matches the README slots)")
    ap.add_argument("--pick", choices=["random", "first", "most-confident"], default="random",
                    help="Which images to take from each folder (default random, seeded)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1000, help="Shrink examples so their longest side is at most this many px")
    args = ap.parse_args()

    if not (args.sorted_dir / "correct").exists() and not (args.sorted_dir / "failure").exists():
        raise SystemExit(f"{args.sorted_dir} doesn't look like a sort_*_errors.py output (no correct/ or failure/)")

    errors_by_image: dict[str, list[dict]] = {}
    ep = args.sorted_dir / "errors_report.csv"
    if ep.exists():
        with open(ep, newline="") as f:
            for row in csv.DictReader(f):
                errors_by_image.setdefault(row["image"], []).append(row)

    rng = random.Random(args.seed)
    index_rows, captions = [], []

    for t in TYPES:
        src = args.sorted_dir / ("correct" if t == "correct" else f"failure/{t}")
        imgs = sorted(p for p in src.glob("*") if p.suffix.lower() in IMG_EXTS) if src.exists() else []
        if not imgs:
            print(f"  {t:<30} no images -> keeping placeholder")
            continue
        if args.pick == "random":
            chosen = rng.sample(imgs, min(args.per_type, len(imgs)))
        elif args.pick == "first":
            chosen = imgs[: args.per_type]
        else:  # most-confident: highest-confidence mistake of this type first
            def best_conf(p):
                rows = [r for r in errors_by_image.get(p.name, []) if r["error_type"] == t and r["confidence"]]
                return max((float(r["confidence"]) for r in rows), default=0.0)
            chosen = sorted(imgs, key=best_conf, reverse=True)[: args.per_type]

        dest = args.out_dir / t
        dest.mkdir(parents=True, exist_ok=True)
        for k, p in enumerate(chosen, 1):
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((args.max_side, args.max_side))
                im.save(dest / f"example_{k}.jpg", quality=90)
            rows = [r for r in errors_by_image.get(p.name, []) if t == "correct" or r["error_type"] == t]
            if t == "correct":
                cap = "every object found with the exact right class"
            else:
                bits = []
                for r in rows[:3]:
                    tr = r["true_class"].replace("Object ", "").split(" - Pdfname")[0].replace(" - Material ", " / ").replace(" - Color ", " / ")
                    pr = r["predicted_class"].replace("Object ", "").split(" - Pdfname")[0].replace(" - Material ", " / ").replace(" - Color ", " / ")
                    conf = f" ({float(r['confidence']):.2f})" if r["confidence"] else ""
                    bits.append(f"true **{tr or '-'}** -> predicted **{pr or '-'}**{conf}")
                cap = "; ".join(bits) or "see image"
            index_rows.append({"type": t, "example": f"{t}/example_{k}.jpg", "source_image": p.name})
            captions.append((t, k, p.name, cap))
        print(f"  {t:<30} copied {len(chosen)} of {len(imgs)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "examples_index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["type", "example", "source_image"])
        wr.writeheader()
        wr.writerows(index_rows)
    md = ["# Example captions (paste under the pictures in the README)", ""]
    for t, k, name, cap in captions:
        md.append(f"- `results/{t}/example_{k}.jpg` ({name}): {cap}")
    (args.out_dir / "examples_captions.md").write_text("\n".join(md) + "\n")
    print(f"[info] wrote examples to {args.out_dir}")


if __name__ == "__main__":
    main()
