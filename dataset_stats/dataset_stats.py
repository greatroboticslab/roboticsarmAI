#!/usr/bin/env python3
"""
dataset_stats.py  -- one-command audit of a YOLO-format dataset.

    python dataset_stats.py --dataset-dir ..\\..\\dataset
    (or just double-click / run run_stats.bat)

What it reports (text report + CSV files + PNG charts, all in --out, default dataset_stats_out):
  1. Split sizes: images, labelled images, empty/background images, instances.
  2. Per class: images + instances in train / valid / test, flagged when under-represented.
  3. Per object / per material / per color roll-ups (the three things your score is built from).
  4. Box statistics: size as a share of the image, small-box share, per split.
  5. Image statistics: resolutions, brightness (and optional blur) - shows lighting / focus gaps.
  6. Label problems: missing label files, bad class ids, malformed lines, boxes outside the
     image, zero-size boxes, duplicate boxes, labels without images.
  7. Duplicate and near-duplicate images, and TRAIN<->VALID leakage. Near-identical photos of the
     same object in both splits make validation scores look better than they really are.

It only reads your dataset - nothing is modified. Needs: pillow, numpy, pyyaml (matplotlib optional for charts).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")
OBJECT_RE = re.compile(r"Object\s+(.*?)\s*-\s*Material", re.IGNORECASE)
MATERIAL_RE = re.compile(r"Material\s+([^-]+?)\s*(?:-\s*Color|-\s*Pdfname)", re.IGNORECASE)
COLOR_RE = re.compile(r"Color\s+([^-]+?)\s*-\s*Pdfname", re.IGNORECASE)
# fallback for plain "object - Material X - Color Y" style names
ALT_OBJECT_RE = re.compile(r"^(.*?)\s*-\s*Material", re.IGNORECASE)
ALT_MATERIAL_RE = re.compile(r"Material\s+([^-]+?)\s*(?:-|$)", re.IGNORECASE)
ALT_COLOR_RE = re.compile(r"Color\s+([^-]+?)\s*(?:-|$)", re.IGNORECASE)


def parse_name(name: str) -> dict:
    def pick(*rxs):
        for rx in rxs:
            m = rx.search(name)
            if m and m.group(1).strip():
                return m.group(1).strip().replace("Object ", "", 1) if rx is OBJECT_RE else m.group(1).strip()
        return ""
    return {
        "object": pick(OBJECT_RE, ALT_OBJECT_RE) or name,
        "material": pick(MATERIAL_RE, ALT_MATERIAL_RE) or "(unspecified)",
        "color": pick(COLOR_RE, ALT_COLOR_RE) or "(unspecified)",
    }


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def load_names(dataset: Path, explicit: Path | None) -> tuple[list[str], Path]:
    cands = [explicit] if explicit else [dataset / "data.corrected.yaml", dataset / "data.yaml"]
    for p in cands:
        if p and p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            names = data["names"]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names, key=int)]
            return list(names), p
    raise SystemExit(f"No data.corrected.yaml / data.yaml found in {dataset}")


def find_splits(dataset: Path) -> dict[str, Path]:
    return {s: dataset / s for s in SPLITS if (dataset / s / "images").is_dir()}


def percentiles(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float)
    return {p: float(np.percentile(a, p)) for p in ps} if len(a) else {p: float("nan") for p in ps}


# ---------------------------------------------------------------- image hashing
def dhash(img: Image.Image) -> int:
    g = img.convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,) uint64, b: (m,) uint64 -> (n,m) hamming distances."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    out = np.zeros(x.shape, dtype=np.uint16)
    for shift in range(0, 64, 8):
        out += _POP[((x >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint8)]
    return out


# ---------------------------------------------------------------- scanning
def scan_split(split: str, root: Path, nc: int, pixel_stats: bool, max_pixel: int, issues: list, rng):
    img_dir, lbl_dir = root / "images", root / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    stems = {p.stem for p in images}
    info = {"images": [], "boxes": []}  # per-image dicts, per-box tuples
    for lf in sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []:
        if lf.stem not in stems:
            issues.append((split, lf.name, "label file has no matching image", ""))
    pix_idx = set(rng.choice(len(images), size=min(max_pixel, len(images)), replace=False).tolist()) if pixel_stats and images else set()
    for i, ip in enumerate(images):
        try:
            with Image.open(ip) as im:
                w, h = im.size
                rec = {"split": split, "name": ip.name, "path": str(ip), "w": w, "h": h}
                if i in pix_idx:
                    g = np.asarray(im.convert("L").resize((min(w, 320), min(h, 320))), dtype=np.float32)
                    rec["brightness"] = float(g.mean())
                    rec["contrast"] = float(g.std())
                    gy, gx = np.gradient(g)
                    rec["sharpness"] = float((gx ** 2 + gy ** 2).mean())
                rec["hash"] = dhash(im)
        except Exception as e:
            issues.append((split, ip.name, "unreadable image", str(e)))
            continue
        lp = lbl_dir / f"{ip.stem}.txt"
        rec["n_boxes"] = 0
        rec["classes"] = set()
        if not lp.exists():
            issues.append((split, ip.name, "image has no label file", "treated as background"))
            rec["no_label_file"] = True
        else:
            seen = set()
            for ln, line in enumerate(lp.read_text(errors="ignore").splitlines(), 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5:
                    issues.append((split, lp.name, "malformed line", f"line {ln}: {line[:60]}"))
                    continue
                try:
                    c = int(parts[0]); xc, yc, bw, bh = (float(v) for v in parts[1:])
                except ValueError:
                    issues.append((split, lp.name, "non-numeric line", f"line {ln}: {line[:60]}"))
                    continue
                if not 0 <= c < nc:
                    issues.append((split, lp.name, "class id out of range", f"line {ln}: class {c} (nc={nc})"))
                    continue
                if bw <= 0 or bh <= 0:
                    issues.append((split, lp.name, "zero/negative-size box", f"line {ln}"))
                    continue
                if min(xc - bw / 2, yc - bh / 2) < -0.005 or max(xc + bw / 2, yc + bh / 2) > 1.005 or not (0 <= xc <= 1 and 0 <= yc <= 1):
                    issues.append((split, lp.name, "box extends outside image", f"line {ln}: {xc:.3f} {yc:.3f} {bw:.3f} {bh:.3f}"))
                key = (c, round(xc, 4), round(yc, 4), round(bw, 4), round(bh, 4))
                if key in seen:
                    issues.append((split, lp.name, "duplicate box", f"line {ln}"))
                    continue
                seen.add(key)
                rec["n_boxes"] += 1
                rec["classes"].add(c)
                info["boxes"].append((c, bw * w, bh * h, bw, bh, split, ip.name))
        info["images"].append(rec)
    return info


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="Explicit data yaml (default: data.corrected.yaml, else data.yaml)")
    ap.add_argument("--out", type=Path, default=Path("dataset_stats_out"))
    ap.add_argument("--min-train-images", type=int, default=30, help="Flag classes with fewer train images than this (default 30)")
    ap.add_argument("--min-valid-images", type=int, default=5, help="Flag classes with fewer valid images than this (default 5)")
    ap.add_argument("--near-dup-distance", type=int, default=6, help="Max hash distance (0-64) counted as a near-duplicate (default 6; lower = stricter)")
    ap.add_argument("--no-pixel-stats", action="store_true", help="Skip brightness/sharpness sampling (faster)")
    ap.add_argument("--pixel-sample", type=int, default=600, help="Images per split sampled for brightness/sharpness")
    ap.add_argument("--no-hash", action="store_true", help="Skip exact/near-duplicate detection")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    ds = args.dataset_dir.resolve()
    names, yaml_path = load_names(ds, args.data)
    nc = len(names)
    splits = find_splits(ds)
    if "train" not in splits:
        raise SystemExit(f"No train/images folder under {ds}")
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    comps = [parse_name(n) for n in names]

    print(f"Dataset {ds}\nClasses: {nc} (from {yaml_path.name}); splits: {', '.join(splits)}")
    issues: list[tuple] = []
    data = {}
    for s, root in splits.items():
        print(f"Scanning {s} ...")
        data[s] = scan_split(s, root, nc, not args.no_pixel_stats, args.pixel_sample, issues, rng)

    # ---------- per class
    cls_inst = {s: Counter() for s in splits}
    cls_imgs = {s: Counter() for s in splits}
    for s, d in data.items():
        for c, *_ in d["boxes"]:
            cls_inst[s][c] += 1
        for r in d["images"]:
            for c in r["classes"]:
                cls_imgs[s][c] += 1
    class_rows = []
    for c in range(nc):
        row = {"class_id": c, "class": names[c], **{k: comps[c][k] for k in ("object", "material", "color")}}
        for s in splits:
            row[f"{s}_images"] = cls_imgs[s][c]
            row[f"{s}_instances"] = cls_inst[s][c]
        tot = sum(cls_inst[s][c] for s in splits)
        row["total_instances"] = tot
        flags = []
        if cls_imgs["train"][c] == 0:
            flags.append("NO TRAIN IMAGES")
        elif cls_imgs["train"][c] < args.min_train_images:
            flags.append(f"few train images (<{args.min_train_images})")
        if "valid" in splits:
            if cls_imgs["valid"][c] == 0:
                flags.append("NOT IN VALID (can't be scored)")
            elif cls_imgs["valid"][c] < args.min_valid_images:
                flags.append(f"few valid images (<{args.min_valid_images})")
            if cls_imgs["train"][c] and cls_imgs["valid"][c]:
                share = cls_imgs["valid"][c] / (cls_imgs["train"][c] + cls_imgs["valid"][c])
                if share < 0.08 or share > 0.35:
                    flags.append(f"odd valid share {share:.0%}")
        row["flags"] = "; ".join(flags)
        class_rows.append(row)

    def rollup(key):
        agg = defaultdict(lambda: defaultdict(int))
        members = defaultdict(set)
        for r in class_rows:
            for s in splits:
                agg[r[key]][f"{s}_images"] += r[f"{s}_images"]
                agg[r[key]][f"{s}_instances"] += r[f"{s}_instances"]
            members[r[key]].add(r["class_id"])
        rows = [{key: k, "classes": len(members[k]), **dict(v)} for k, v in agg.items()]
        return sorted(rows, key=lambda r: r.get("train_instances", 0))

    obj_rows, mat_rows, col_rows = rollup("object"), rollup("material"), rollup("color")

    # ---------- boxes
    all_boxes = [b for d in data.values() for b in d["boxes"]]
    bw_px = np.array([b[1] for b in all_boxes]); bh_px = np.array([b[2] for b in all_boxes])
    bw_rel = np.array([b[3] for b in all_boxes]); bh_rel = np.array([b[4] for b in all_boxes])
    area_rel = bw_rel * bh_rel
    small = float(np.mean(np.sqrt(bw_px * bh_px) < 32)) if len(all_boxes) else float("nan")
    tiny_rel = float(np.mean(np.sqrt(area_rel) < 0.05)) if len(all_boxes) else float("nan")

    # ---------- images
    all_imgs = [r for d in data.values() for r in d["images"]]
    res = Counter((r["w"], r["h"]) for r in all_imgs)

    # ---------- duplicates / leakage
    dup_rows, leak_rows = [], []
    exact_groups = []
    leak_summary = {}
    if not args.no_hash and all_imgs:
        print("Looking for duplicate / near-duplicate images ...")
        by_md5 = defaultdict(list)
        for r in all_imgs:
            by_md5[md5_of(r["path"])].append(r)
        for grp in by_md5.values():
            if len(grp) > 1:
                exact_groups.append(grp)
                dup_rows.append({"kind": "exact", "files": " | ".join(f"{g['split']}/{g['name']}" for g in grp)})
        hashes = {s: np.array([r["hash"] for r in d["images"]], dtype=np.uint64) for s, d in data.items() if d["images"]}
        for other in [s for s in splits if s != "train"]:
            if other not in hashes or "train" not in hashes:
                continue
            dist = hamming_matrix(hashes[other], hashes["train"])
            nearest = dist.min(axis=1)
            idx = dist.argmin(axis=1)
            near = nearest <= args.near_dup_distance
            leak_summary[other] = (int(near.sum()), len(nearest), int((nearest == 0).sum()))
            for i in np.where(near)[0]:
                a, b = data[other]["images"][i], data["train"]["images"][idx[i]]
                leak_rows.append({"split": other, "image": a["name"], "closest_train_image": b["name"], "hash_distance": int(nearest[i])})
        # near-duplicates inside train
        if "train" in hashes and len(hashes["train"]) > 1:
            n = len(hashes["train"])
            step = 2000
            same = 0
            for i0 in range(0, n, step):
                dm = hamming_matrix(hashes["train"][i0:i0 + step], hashes["train"])
                for k in range(dm.shape[0]):
                    dm[k, i0 + k] = 99
                same += int((dm.min(axis=1) <= args.near_dup_distance).sum())
            leak_summary["train (within)"] = (same, n, 0)

    # ---------- write CSVs
    def write_csv(name, rows):
        if not rows:
            return
        with open(args.out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    write_csv("per_class.csv", class_rows)
    write_csv("per_object.csv", obj_rows)
    write_csv("per_material.csv", mat_rows)
    write_csv("per_color.csv", col_rows)
    write_csv("label_issues.csv", [dict(split=a, file=b, problem=c, detail=d) for a, b, c, d in issues])
    write_csv("duplicates_exact.csv", dup_rows)
    write_csv("train_valid_near_duplicates.csv", leak_rows)

    # ---------- text report
    L = []
    P = L.append
    P("=" * 78); P("DATASET STATISTICS"); P("=" * 78)
    P(f"dataset: {ds}"); P(f"classes: {nc}   (names from {yaml_path.name})"); P("")
    P("SPLITS")
    P(f"  {'split':<8}{'images':>8}{'with boxes':>12}{'background':>12}{'instances':>11}{'inst/img':>10}")
    for s, d in data.items():
        n = len(d["images"]); wb = sum(1 for r in d["images"] if r["n_boxes"])
        P(f"  {s:<8}{n:>8}{wb:>12}{n - wb:>12}{len(d['boxes']):>11}{(len(d['boxes']) / max(n, 1)):>10.2f}")
    tot_imgs = sum(len(d["images"]) for d in data.values())
    if "valid" in data and tot_imgs:
        P(f"  valid share of all images: {len(data['valid']['images']) / tot_imgs:.1%}")
    P("")

    inst = np.array([r["train_instances"] for r in class_rows])
    P("CLASS BALANCE (train instances)")
    if inst.size:
        nz = inst[inst > 0]
        P(f"  min {inst.min()}   median {int(np.median(inst))}   max {inst.max()}   "
          f"imbalance (max/min of non-empty) {inst.max() / max(nz.min(), 1) if nz.size else float('nan'):.1f}x")
    flagged = [r for r in class_rows if r["flags"]]
    P(f"  classes with a warning: {len(flagged)} of {nc}"); P("")

    def table(rows, key, title):
        P(title)
        P(f"  {key:<34}{'classes':>8}" + "".join(f"{s + ' img':>11}{s + ' inst':>11}" for s in splits))
        for r in rows:
            P(f"  {str(r[key])[:33]:<34}{r['classes']:>8}" + "".join(f"{r.get(f'{s}_images', 0):>11}{r.get(f'{s}_instances', 0):>11}" for s in splits))
        P("")
    table(obj_rows, "object", "PER OBJECT (fewest train instances first)")
    table(mat_rows, "material", "PER MATERIAL")
    table(col_rows, "color", "PER COLOR")

    P("CLASSES THAT NEED ATTENTION (fewest train images first)")
    P(f"  {'class':<58}{'train img':>10}{'valid img':>10}  warning")
    for r in sorted(flagged, key=lambda r: (r["train_images"], r["valid_images"] if "valid_images" in r else 0))[:40]:
        P(f"  {r['class'][:57]:<58}{r['train_images']:>10}{r.get('valid_images', 0):>10}  {r['flags']}")
    if len(flagged) > 40:
        P(f"  ... {len(flagged) - 40} more in per_class.csv")
    P("")

    P("BOXES")
    if all_boxes:
        rel_w, rel_h = percentiles(bw_rel * 100), percentiles(bh_rel * 100)
        P(f"  total boxes {len(all_boxes)}")
        P("  width  as % of image (p5/p25/p50/p75/p95): " + " / ".join(f"{rel_w[p]:.1f}" for p in rel_w))
        P("  height as % of image (p5/p25/p50/p75/p95): " + " / ".join(f"{rel_h[p]:.1f}" for p in rel_h))
        P(f"  boxes under 32 px (on the original image): {small:.1%}   covering under 5% of image side: {tiny_rel:.1%}")
        P("  (many tiny boxes => use a larger --imgsz / RF-DETR resolution, otherwise they get shrunk away)")
    P("")

    P("IMAGES")
    P(f"  total {len(all_imgs)}; distinct sizes {len(res)}")
    for (w, h), n in res.most_common(6):
        P(f"    {w}x{h}: {n}")
    for key, label in (("brightness", "brightness (0-255)"), ("contrast", "contrast (std)"), ("sharpness", "sharpness (higher = crisper)")):
        by = {s: [r[key] for r in d["images"] if key in r] for s, d in data.items()}
        if any(by.values()):
            P(f"  {label}: " + "   ".join(f"{s} p5/p50/p95 = " + "/".join(f"{v:.0f}" for v in percentiles(x, (5, 50, 95)).values()) for s, x in by.items() if x))
    P("")

    P("LABEL PROBLEMS")
    cnt = Counter(i[2] for i in issues)
    if not cnt:
        P("  none found")
    for k, v in cnt.most_common():
        P(f"  {v:>6}  {k}")
    P("  (every one is listed in label_issues.csv)"); P("")

    P("DUPLICATES AND TRAIN/VALID LEAKAGE")
    if args.no_hash:
        P("  skipped (--no-hash)")
    else:
        P(f"  exact duplicate groups: {len(exact_groups)}")
        for s, (n_near, n_tot, n_same) in leak_summary.items():
            what = "have a near-duplicate in train" if s != "train (within)" else "have a near-duplicate elsewhere in train"
            P(f"  {s}: {n_near} of {n_tot} images ({n_near / max(n_tot, 1):.0%}) {what}"
              + (f"  [{n_same} identical-looking]" if s != "train (within)" else ""))
        if any(s in leak_summary for s in ("valid", "test")):
            worst = max((v[0] / max(v[1], 1) for k, v in leak_summary.items() if k != "train (within)"), default=0)
            if worst > 0.25:
                P("  !! A large share of validation images look almost the same as training images, so validation scores")
                P("     are probably higher than the model will score on genuinely new photos. See train_valid_near_duplicates.csv.")
                P("     Fix: split by object/capture session rather than by random image (resplit_dataset.py has object-disjoint modes).")
        P(f"  (near-duplicate = image fingerprint distance <= {args.near_dup_distance} of 64; a rotating/shuffled shoot of one object trips this)")
    P("")
    report = "\n".join(L)
    (args.out / "dataset_stats_report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)

    # ---------- charts
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            order = sorted(range(nc), key=lambda c: class_rows[c]["train_images"])
            fig, ax = plt.subplots(figsize=(10, max(4, nc * 0.2)))
            y = np.arange(nc)
            ax.barh(y - 0.2, [class_rows[c]["train_images"] for c in order], 0.4, label="train")
            if "valid" in splits:
                ax.barh(y + 0.2, [class_rows[c]["valid_images"] for c in order], 0.4, label="valid")
            ax.set_yticks(y); ax.set_yticklabels([names[c][:50] for c in order], fontsize=6)
            ax.set_xlabel("images containing the class"); ax.legend(); ax.set_title("Images per class")
            fig.tight_layout(); fig.savefig(args.out / "images_per_class.png", dpi=130); plt.close(fig)
            for rows, key in ((obj_rows, "object"), (mat_rows, "material"), (col_rows, "color")):
                fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.3)))
                ax.barh([str(r[key])[:30] for r in rows], [r["train_instances"] for r in rows])
                ax.set_title(f"Train instances per {key}"); fig.tight_layout()
                fig.savefig(args.out / f"instances_per_{key}.png", dpi=130); plt.close(fig)
            if all_boxes:
                fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
                axs[0].hist(bw_rel * 100, bins=40); axs[0].set_title("box width (% of image)")
                axs[1].hist(bh_rel * 100, bins=40); axs[1].set_title("box height (% of image)")
                fig.tight_layout(); fig.savefig(args.out / "box_sizes.png", dpi=130); plt.close(fig)
        except ImportError:
            print("[info] matplotlib not installed - skipped charts (pip install matplotlib)")
    print(f"\nAll files written to: {args.out.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
