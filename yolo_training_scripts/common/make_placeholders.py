#!/usr/bin/env python3
"""
make_placeholders.py

Creates the placeholder pictures the READMEs link to:

    <results-dir>/<type>/example_1.jpg
    <results-dir>/<type>/example_2.jpg

for "correct" and every failure type. Each placeholder is a grey card with the
error type written on it in very large text, so the README renders cleanly
before any real results exist. collect_examples.py later overwrites them with
real sorted examples.

Existing files are never overwritten unless you pass --force (so it can't
clobber real results).

Usage:
    python make_placeholders.py --results-dir ../yolo/results
    python make_placeholders.py --results-dir ../transformer/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from error_sorting import PRIORITY
from visualize_annotations import load_font

TYPES = ["correct"] + PRIORITY


def make_card(label: str, k: int, size=(900, 600)) -> Image.Image:
    img = Image.new("RGB", size, (232, 234, 238))
    d = ImageDraw.Draw(img)
    d.rectangle([6, 6, size[0] - 7, size[1] - 7], outline=(150, 156, 168), width=6)
    big, mid = load_font(64), load_font(40)
    words = label.replace("_", " ").upper().split()
    lines, cur = [], ""
    for w in words:                                  # wrap the type name to fit the card
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=big) > size[0] - 80 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    lines.append(cur)
    y = 90
    for ln in lines:
        d.text(((size[0] - d.textlength(ln, font=big)) / 2, y), ln, fill=(40, 44, 52), font=big)
        y += 84
    for txt in (f"example {k}", "PLACEHOLDER", "run collect_examples.py to fill"):
        d.text(((size[0] - d.textlength(txt, font=mid)) / 2, y + 30), txt, fill=(90, 96, 108), font=mid)
        y += 56
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="Overwrite existing files (will replace real results!)")
    args = ap.parse_args()
    made = 0
    for t in TYPES:
        (args.results_dir / t).mkdir(parents=True, exist_ok=True)
        for k in range(1, args.per_type + 1):
            p = args.results_dir / t / f"example_{k}.jpg"
            if p.exists() and not args.force:
                continue
            make_card(t, k).save(p, quality=88)
            made += 1
    print(f"[info] wrote {made} placeholder images under {args.results_dir}")


if __name__ == "__main__":
    main()
