#!/usr/bin/env python3
"""
resplit_dataset.py

Rebuilds the train/valid/test splits from scratch at a chosen ratio.

Two modes:

  --mode object-disjoint (default)
      Every object (every distinct `Pdfname` id, i.e. every physical item
      that was photographed) ends up ENTIRELY in one split -- never split
      across train and valid/test. If two different objects happen to
      co-occur in the same photo, they're treated as one group and kept
      together too. This avoids leakage from near-duplicate photos of the
      *same physical item* appearing in both train and validation, which
      otherwise inflates validation metrics without actually testing
      generalization to new objects. It also lets you dial the ratio (e.g.
      more objects into train) to widen the training set when you have
      few images per object.

  --mode random
      Classic per-image random shuffle, ignoring object identity. Images of
      the same object can land in both train and valid/test.

This script is non-destructive by default: it writes the new split into
--output-dir (a fresh folder next to dataset/, never overwriting your
current train/valid/test) so you can inspect it first. Pass
--apply-in-place to have it back up your existing train/valid/test folders
(renamed with a timestamp suffix) and replace them with the new split.

Usage:
    # preview a new 70/20/10 split, grouped so no object leaks across splits
    python resplit_dataset.py --dataset-dir ../dataset --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

    # actually replace train/valid/test with the new split (old ones backed up)
    python resplit_dataset.py --dataset-dir ../dataset --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1 --apply-in-place

    # plain random split instead, for comparison
    python resplit_dataset.py --dataset-dir ../dataset --mode random --train-ratio 0.8 --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from yolo_common import (
    load_data_yaml,
    resolve_data_yaml,
    class_id_to_pdf_id_map,
    find_split_dirs,
    list_image_label_pairs,
    read_label_class_ids,
)


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def gather_all_pairs(dataset_dir: Path) -> list[tuple[Path, Path]]:
    """Pool every image/label pair from every existing split -- we're rebuilding all of them."""
    splits = find_split_dirs(dataset_dir)
    if not splits:
        raise SystemExit(f"No train/valid/test split with images/ and labels/ found under {dataset_dir}")
    pairs = []
    for split_dir in splits.values():
        pairs.extend(list_image_label_pairs(split_dir))
    return pairs


def build_object_groups(
    pairs: list[tuple[Path, Path]], cls_to_pdf: dict[int, str | None]
) -> tuple[dict[str, list[tuple[Path, Path]]], list[tuple[Path, Path]]]:
    """
    Returns (component_id -> [(image, label), ...], background_pairs) where
    background_pairs are images with no labels / no resolvable pdf id at all
    (free to place anywhere).
    """
    image_pdf_ids: dict[Path, set[str]] = {}
    all_pdf_ids: set[str] = set()
    background: list[tuple[Path, Path]] = []

    for img_path, lbl_path in pairs:
        cls_ids = read_label_class_ids(lbl_path)
        pdf_ids = {cls_to_pdf.get(c) for c in cls_ids}
        pdf_ids.discard(None)
        if not pdf_ids:
            background.append((img_path, lbl_path))
            continue
        image_pdf_ids[img_path] = pdf_ids
        all_pdf_ids.update(pdf_ids)

    uf = UnionFind(all_pdf_ids)
    for pdf_ids in image_pdf_ids.values():
        pdf_ids = list(pdf_ids)
        for other in pdf_ids[1:]:
            uf.union(pdf_ids[0], other)

    components: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for (img_path, lbl_path), pdf_ids in zip(
        [p for p in pairs if p[0] in image_pdf_ids], image_pdf_ids.values()
    ):
        root = uf.find(next(iter(pdf_ids)))
        components[root].append((img_path, lbl_path))

    return components, background


def assign_object_disjoint(
    pairs: list[tuple[Path, Path]],
    cls_to_pdf: dict[int, str | None],
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, list[tuple[Path, Path]]]:
    components, background = build_object_groups(pairs, cls_to_pdf)

    total = len(pairs)
    targets = {split: ratio * total for split, ratio in ratios.items()}
    assigned: dict[str, list[tuple[Path, Path]]] = {split: [] for split in ratios}

    # Largest components first (most constrained), shuffling within equal sizes for fairness.
    comp_items = list(components.items())
    rng.shuffle(comp_items)
    comp_items.sort(key=lambda kv: len(kv[1]), reverse=True)

    for _, imgs in comp_items:
        # Assign to whichever split is furthest below its target (relative to size, so
        # small splits like test aren't starved early by one big component).
        best_split = max(ratios, key=lambda s: targets[s] - len(assigned[s]))
        assigned[best_split].extend(imgs)

    # Background / unresolvable images: no object constraint, so distribute freely to fill deficits.
    rng.shuffle(background)
    for img in background:
        best_split = max(ratios, key=lambda s: targets[s] - len(assigned[s]))
        assigned[best_split].append(img)

    return assigned


def assign_random(
    pairs: list[tuple[Path, Path]], ratios: dict[str, float], rng: random.Random
) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    total = len(shuffled)
    assigned: dict[str, list[tuple[Path, Path]]] = {}
    cursor = 0
    split_names = list(ratios.keys())
    for i, split in enumerate(split_names):
        if i == len(split_names) - 1:
            assigned[split] = shuffled[cursor:]
        else:
            n = round(ratios[split] * total)
            assigned[split] = shuffled[cursor:cursor + n]
            cursor += n
    return assigned


def write_split(output_dir: Path, split: str, items: list[tuple[Path, Path]]):
    img_out = output_dir / split / "images"
    lbl_out = output_dir / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path, lbl_path in items:
        shutil.copy2(img_path, img_out / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_out / lbl_path.name)
        else:
            (lbl_out / f"{img_path.stem}.txt").write_text("")  # background image, empty label


def summarize(
    assigned: dict[str, list[tuple[Path, Path]]], cls_to_pdf: dict[int, str | None], ratios: dict[str, float]
) -> list[str]:
    total = sum(len(v) for v in assigned.values())
    lines = ["=" * 70, "RESPLIT SUMMARY", "=" * 70]
    all_objects_seen: dict[str, set[str]] = {}
    for split, items in assigned.items():
        objs = set()
        n_instances = 0
        for img_path, lbl_path in items:
            cls_ids = read_label_class_ids(lbl_path)
            n_instances += len(cls_ids)
            for c in cls_ids:
                pid = cls_to_pdf.get(c)
                if pid:
                    objs.add(pid)
        all_objects_seen[split] = objs
        achieved_pct = 100 * len(items) / total if total else 0
        target_pct = 100 * ratios.get(split, 0)
        lines.append(
            f"[{split}] images={len(items)} ({achieved_pct:.1f}%, target {target_pct:.1f}%) "
            f"label_instances={n_instances} distinct_objects={len(objs)}"
        )
        lines.append(f"    objects: {sorted(objs)}")

    # cross-split leakage check
    split_names = list(assigned.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = all_objects_seen[a] & all_objects_seen[b]
            if overlap:
                lines.append(f"[NOTE] objects present in BOTH {a} and {b}: {sorted(overlap)}")
    if not any(
        all_objects_seen[split_names[i]] & all_objects_seen[split_names[j]]
        for i in range(len(split_names)) for j in range(i + 1, len(split_names))
    ):
        lines.append("[OK] no object appears in more than one split")

    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml (default: auto-pick data.corrected.yaml/data.yaml under --dataset-dir)")
    ap.add_argument("--mode", choices=["object-disjoint", "random"], default="object-disjoint")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--test-ratio", type=float, default=0.0, help="Set > 0 to also produce a test split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=None, help="Where to write the new split (default: <dataset-dir>/resplit_preview)")
    ap.add_argument(
        "--apply-in-place", action="store_true",
        help="Back up existing train/valid/test (renamed with a timestamp suffix) and replace them with the new split.",
    )
    args = ap.parse_args()

    ratios = {"train": args.train_ratio, "valid": args.val_ratio}
    if args.test_ratio > 0:
        ratios["test"] = args.test_ratio
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-3:
        raise SystemExit(f"--train-ratio + --val-ratio + --test-ratio must sum to 1.0 (got {total_ratio})")

    data_yaml_path = resolve_data_yaml(args.dataset_dir, args.data)
    names = load_data_yaml(data_yaml_path)["names"]
    cls_to_pdf = class_id_to_pdf_id_map(names)

    pairs = gather_all_pairs(args.dataset_dir)
    print(f"Found {len(pairs)} image/label pairs across existing splits.")

    rng = random.Random(args.seed)
    if args.mode == "object-disjoint":
        assigned = assign_object_disjoint(pairs, cls_to_pdf, ratios, rng)
    else:
        assigned = assign_random(pairs, ratios, rng)

    summary_lines = summarize(assigned, cls_to_pdf, ratios)
    print("\n".join(summary_lines))

    output_dir = args.output_dir or (args.dataset_dir / "resplit_preview")

    if args.apply_in_place:
        # Stage the new split fully in a temp directory FIRST, using the original
        # image/label paths (which still exist at this point) -- only after
        # staging succeeds do we touch the existing train/valid/test folders.
        # This avoids ever being caught with the old files half-moved and the
        # new files half-written.
        staging_dir = args.dataset_dir / f"_resplit_staging_{time.strftime('%Y%m%d_%H%M%S')}"
        for split, items in assigned.items():
            write_split(staging_dir, split, items)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for split in find_split_dirs(args.dataset_dir):
            src = args.dataset_dir / split
            backup = args.dataset_dir / f"{split}.bak_{timestamp}"
            shutil.move(str(src), str(backup))
            print(f"Backed up existing '{split}' -> {backup}")

        for split in assigned:
            shutil.move(str(staging_dir / split), str(args.dataset_dir / split))
        staging_dir.rmdir()

        output_dir = args.dataset_dir
        report_path = output_dir / "resplit_report.txt"
        report_path.write_text("\n".join(summary_lines), encoding="utf-8")
    else:
        for split, items in assigned.items():
            write_split(output_dir, split, items)
        report_path = output_dir / "resplit_report.txt"
        report_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\nNew split written to: {output_dir}")
    print(f"Report written to: {report_path}")
    if not args.apply_in_place:
        print("\nThis was a preview (original train/valid/test untouched). Re-run with --apply-in-place to use it for training.")


if __name__ == "__main__":
    main()
