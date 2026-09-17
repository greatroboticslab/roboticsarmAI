#!/usr/bin/env python3
"""
build_dataset.py

Prepares the `dataset/` folder for YOLO training.

IMPORTANT: dataset/data.yaml is treated as the AUTHORITATIVE, human-curated
class list. People often hand-edit it in ways that deliberately diverge from
a literal reading of the PDFs -- e.g. collapsing several parts into one
class because only one material is actually visible/relevant in-frame (a
claw hammer photographed such that only its metal head/neck and its rubber
grip are distinguishable classes, even though the PDF might list the head
and neck as two separate "metal" parts). That's a legitimate curation
choice, not a bug, and this script must never silently overwrite it.

So by default this script is READ-ONLY / advisory: it never edits
data.yaml, and data.corrected.yaml starts as an exact copy of it. Pass
--apply-renames to actually let a PDF's current answer overwrite a class's
name text (never its index) -- and even then, only for the safe case where
the material *count* still matches what's already in data.yaml.

What it does:

  1. Parses every PDF in dataset/pdf_for_labels/. Each PDF is a Gemini
     labeling transcript that may contain one or more human corrections
     appended after the original answer. We always take the MOST RECENT
     answer -- i.e. we look from the bottom of the PDF upward -- so a
     correction ("thats a pen ands its plastic rubber near the tip...")
     always overrides the original guess above it.
  2. Compares that answer against the class names already in
     dataset/data.yaml (matched via the `Pdfname` suffix baked into every
     class name).
       - If the material *count* matches, it's a candidate rename --
         reported always, applied to data.corrected.yaml only with
         --apply-renames. Class indices are never reordered or renumbered
         either way, so existing label .txt files always stay valid.
       - If the material count differs from what's currently in data.yaml,
         that's reported as [COUNT DIFFERS FROM PDF] -- purely informational.
         This is common and often intentional (manual simplification, only
         labeling the material of the part that's actually visible, etc.).
         These are NEVER auto-applied regardless of --apply-renames; if the
         difference really does reflect a missed update, apply it by hand.
  3. Writes dataset/data.corrected.yaml and a human-readable
     dataset/dataset_build_report.txt.
  4. Audits image/label pairing and class-id validity across the
     train/valid(/test) splits and reports any problems.

Usage:
    python build_dataset.py --dataset-dir /path/to/dataset
    python build_dataset.py --dataset-dir /path/to/dataset --apply-renames
    python build_dataset.py --dataset-dir /path/to/dataset --strict   # exit 1 if anything differs from data.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from pdf_label_parser import parse_pdf_label, canonical_class_names, PdfLabel

PDFNAME_RE = re.compile(r"Pdfname\s+(\S+)\s*$")
MATERIAL_RE = re.compile(r"Material\s+([^-]+?)\s*(?:-\s*Color|-\s*Pdfname)", re.IGNORECASE)
COLOR_RE = re.compile(r"Color\s+([^-]+?)\s*-\s*Pdfname", re.IGNORECASE)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_data_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_pdf_id_from_name(name: str) -> str | None:
    m = PDFNAME_RE.search(name)
    return m.group(1).strip() if m else None


def extract_material_hint_from_name(name: str) -> str:
    m = MATERIAL_RE.search(name)
    return m.group(1).strip().lower() if m else ""


def extract_color_hint_from_name(name: str) -> str:
    m = COLOR_RE.search(name)
    return m.group(1).strip().lower() if m else ""


def group_class_indices_by_pdf(names: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(names):
        pdf_id = extract_pdf_id_from_name(name)
        if pdf_id:
            groups[pdf_id].append(idx)
        else:
            print(f"  [warn] class {idx} ({name!r}) has no parseable Pdfname suffix; leaving untouched")
    return groups


def _text_matches(a: str, b: str) -> bool:
    """Loose equality: exact, or one contains the other (handles typos/plurals
    like 'orange' vs 'oranges', or 'plastic' vs 'hard plastic')."""
    if not a or not b:
        return False
    return a == b or a in b or b in a


def align_new_names_to_indices(
    old_indices: list[int], old_names: list[str], new_names: list[str], new_label: PdfLabel
) -> tuple[dict[int, str], list[str]]:
    """
    Map each existing class index to its corrected name, using a fallback
    chain from most to least confident so a manually-curated entry (which
    may have its own deliberately chosen material/color) is matched to the
    PDF entry it actually corresponds to, rather than to whichever PDF
    entry happens to come first:

        1. exact/near match on BOTH material and color
        2. match on material only (color may have been manually adjusted)
        3. match on color only (material may have been manually adjusted)
        4. positional fallback, flagged as low-confidence

    Each pass only considers old indices / new entries not already claimed,
    so a confident match never gets displaced by a later, weaker pass.
    Returns (index -> new_name mapping, list of warning strings).
    Only called when len(old_indices) == len(new_names).
    """
    warnings: list[str] = []
    mapping: dict[int, str] = {}
    remaining_new = list(zip(new_names, new_label.materials))  # [(name, MaterialEntry), ...]
    remaining_old = list(old_indices)

    def hints(idx: int) -> tuple[str, str]:
        return extract_material_hint_from_name(old_names[idx]), extract_color_hint_from_name(old_names[idx])

    def run_pass(match_fn) -> list[int]:
        """Match remaining_old against remaining_new using match_fn(old_mat, old_color, new_mat_entry) -> bool.
        Mutates remaining_old/remaining_new/mapping in place. Returns indices matched this pass."""
        matched_this_pass = []
        still_unmatched = []
        for idx in remaining_old:
            old_mat, old_color = hints(idx)
            match_pos = None
            for pos, (_, mat) in enumerate(remaining_new):
                if match_fn(old_mat, old_color, mat):
                    match_pos = pos
                    break
            if match_pos is not None:
                mapping[idx] = remaining_new.pop(match_pos)[0]
                matched_this_pass.append(idx)
            else:
                still_unmatched.append(idx)
        remaining_old[:] = still_unmatched
        return matched_this_pass

    # Pass 1: material AND color both match.
    run_pass(lambda om, oc, mat: _text_matches(om, mat.material.lower()) and _text_matches(oc, mat.color.lower()))

    # Pass 2: material matches (color may have been manually adjusted -- fall back to it).
    pass2 = run_pass(lambda om, oc, mat: _text_matches(om, mat.material.lower()))
    if pass2:
        warnings.append(
            f"indices {pass2} matched by material only -- color text differs from the PDF's current "
            f"answer, which may be a manual color correction; verify the color is still right"
        )

    # Pass 3: color matches (material may have been manually adjusted -- fall back to it).
    pass3 = run_pass(lambda om, oc, mat: oc and _text_matches(oc, mat.color.lower()))
    if pass3:
        warnings.append(
            f"indices {pass3} matched by color only -- material text differs from the PDF's current "
            f"answer, which may be a manual material correction; verify the material is still right"
        )

    # Pass 4: whatever's left, align positionally as a last resort.
    if remaining_old and remaining_new:
        warnings.append(
            f"could not match material or color keywords for indices {remaining_old}; "
            f"aligned by position instead -- please double check the resulting names"
        )
        for idx, (name, _) in zip(remaining_old, remaining_new):
            mapping[idx] = name

    return mapping, warnings


def build_corrected_yaml(dataset_dir: Path, apply_renames: bool = False, strict: bool = False) -> int:
    data_yaml_path = dataset_dir / "data.yaml"
    pdf_dir = dataset_dir / "pdf_for_labels"

    data = load_data_yaml(data_yaml_path)
    names: list[str] = list(data["names"])
    original_names = list(names)

    groups = group_class_indices_by_pdf(names)

    pdf_files = {p.stem: p for p in pdf_dir.glob("*.pdf")}

    report_lines: list[str] = []
    count_diffs: list[str] = []      # material count differs from data.yaml -- informational only, never touched
    proposed_renames: list[str] = [] # count matches, text differs -- applied only if apply_renames
    unchanged: list[str] = []

    referenced_pdf_ids = set(groups.keys())
    available_pdf_ids = set(pdf_files.keys())

    missing_pdfs = referenced_pdf_ids - available_pdf_ids
    orphan_pdfs = available_pdf_ids - referenced_pdf_ids
    for pid in sorted(missing_pdfs):
        report_lines.append(f"[MISSING PDF] classes reference Pdfname {pid} but no {pid}.pdf found in pdf_for_labels/")
    for pid in sorted(orphan_pdfs):
        report_lines.append(f"[ORPHAN PDF] {pid}.pdf exists but no class in data.yaml references it")

    for pdf_id in sorted(referenced_pdf_ids & available_pdf_ids):
        old_indices = groups[pdf_id]
        try:
            label = parse_pdf_label(pdf_files[pdf_id])
        except Exception as e:
            report_lines.append(f"[PARSE ERROR] {pdf_id}.pdf: {e}")
            count_diffs.append(pdf_id)
            continue

        new_names = canonical_class_names(label)
        turn_note = (
            f"(used most-recent turn; {label.num_turns_found} total response(s) in PDF, "
            f"walked {label.turn_index_used} turn(s) up from the bottom)"
        )

        if len(old_indices) != len(new_names):
            count_diffs.append(pdf_id)
            report_lines.append(
                f"[COUNT DIFFERS FROM PDF] Pdfname {pdf_id}: data.yaml has {len(old_indices)} class(es) "
                f"for this object, PDF's current answer has {len(new_names)} material(s) {turn_note}."
            )
            report_lines.append(f"    data.yaml classes (indices {old_indices}, treated as authoritative):")
            for idx in old_indices:
                report_lines.append(f"      [{idx}] {original_names[idx]!r}")
            report_lines.append(f"    PDF's current answer -> object={label.object_name!r}")
            for n in new_names:
                report_lines.append(f"      -> {n!r}")
            report_lines.append(
                "    INFO ONLY, nothing changed: this is often intentional (e.g. only the material "
                "of the part actually visible/relevant was kept as its own class, or several parts "
                "sharing a material were deliberately merged). Re-check by hand only if this looks "
                "like a genuine missed update, not just a curation choice."
            )
            continue

        mapping, warns = align_new_names_to_indices(old_indices, original_names, new_names, label)
        any_diff = False
        diff_pairs = []
        for idx in old_indices:
            new_name = mapping.get(idx)
            if new_name and new_name != names[idx]:
                diff_pairs.append((idx, names[idx], new_name))
                any_diff = True
                if apply_renames:
                    names[idx] = new_name

        if any_diff:
            proposed_renames.append(pdf_id)
            tag = "[APPLIED RENAME]" if apply_renames else "[PROPOSED RENAME] (not written -- pass --apply-renames to apply)"
            report_lines.append(f"{tag} Pdfname {pdf_id} {turn_note}:")
            for idx, old_name, new_name in diff_pairs:
                report_lines.append(f"    [{idx}] {old_name!r}")
                report_lines.append(f"        -> {new_name!r}")
            for w in warns:
                report_lines.append(f"    [warn] {w}")
        else:
            unchanged.append(pdf_id)

    # data.corrected.yaml: exact copy of data.yaml unless --apply-renames was given,
    # in which case only count-matched renames are baked in. Never touches indices.
    corrected = dict(data)
    corrected["names"] = names
    corrected_path = dataset_dir / "data.corrected.yaml"
    with open(corrected_path, "w") as f:
        yaml.safe_dump(corrected, f, sort_keys=False, allow_unicode=True)

    # ---- image / label integrity audit ----
    integrity_lines = audit_image_label_integrity(dataset_dir, nc=data.get("nc", len(names)))

    summary = [
        "=" * 70,
        "PDF LABEL RECONCILIATION SUMMARY",
        "=" * 70,
        f"mode: {'apply-renames (data.corrected.yaml updated)' if apply_renames else 'dry-run / advisory only (data.corrected.yaml == data.yaml)'}",
        f"pdfs unchanged (already matches PDF): {len(unchanged)}",
        f"pdfs with a proposed rename (count matches, text differs): {len(proposed_renames)} -> {sorted(proposed_renames)}",
        f"pdfs where material count differs from data.yaml (informational only): {len(count_diffs)} -> {sorted(count_diffs)}",
        f"missing pdf files: {sorted(missing_pdfs)}",
        f"orphan pdf files: {sorted(orphan_pdfs)}",
        "",
    ]

    full_report = summary + report_lines + [""] + integrity_lines
    report_path = dataset_dir / "dataset_build_report.txt"
    report_path.write_text("\n".join(full_report), encoding="utf-8")

    print("\n".join(summary))
    print(f"Full report written to: {report_path}")
    print(f"Corrected data.yaml written to: {corrected_path}")

    if strict and (count_diffs or (proposed_renames and not apply_renames)):
        print("\n--strict was set and there are unresolved differences from data.yaml -- exiting with error.")
        return 1
    return 0


def audit_image_label_integrity(dataset_dir: Path, nc: int) -> list[str]:
    lines = ["=" * 70, "IMAGE / LABEL INTEGRITY AUDIT", "=" * 70]
    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if not split_dir.exists():
            lines.append(f"[{split}] split directory not found -- skipped")
            continue
        if not img_dir.exists() or not lbl_dir.exists():
            lines.append(f"[{split}] missing images/ or labels/ subfolder -- skipped")
            continue

        images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")}

        images_without_labels = sorted(set(images) - set(labels))
        labels_without_images = sorted(set(labels) - set(images))

        bad_class_ids = []
        malformed_lines = []
        for stem, lbl_path in labels.items():
            for lineno, line in enumerate(lbl_path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    malformed_lines.append(f"{lbl_path.name}:{lineno} -> expected 5 fields, got {len(parts)}")
                    continue
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    malformed_lines.append(f"{lbl_path.name}:{lineno} -> non-integer class id {parts[0]!r}")
                    continue
                if not (0 <= cls_id < nc):
                    bad_class_ids.append(f"{lbl_path.name}:{lineno} -> class id {cls_id} out of range [0,{nc})")

        lines.append(f"[{split}] images={len(images)} labels={len(labels)}")
        if images_without_labels:
            lines.append(f"  {len(images_without_labels)} image(s) with no matching label file (background images -- OK if intentional):")
            for s in images_without_labels[:10]:
                lines.append(f"    - {s}")
            if len(images_without_labels) > 10:
                lines.append(f"    ... and {len(images_without_labels) - 10} more")
        if labels_without_images:
            lines.append(f"  [ERROR] {len(labels_without_images)} label file(s) with no matching image:")
            for s in labels_without_images[:10]:
                lines.append(f"    - {s}")
        if bad_class_ids:
            lines.append(f"  [ERROR] {len(bad_class_ids)} label line(s) reference an out-of-range class id:")
            for s in bad_class_ids[:10]:
                lines.append(f"    - {s}")
        if malformed_lines:
            lines.append(f"  [ERROR] {len(malformed_lines)} malformed label line(s):")
            for s in malformed_lines[:10]:
                lines.append(f"    - {s}")
        if not (images_without_labels or labels_without_images or bad_class_ids or malformed_lines):
            lines.append("  OK -- no issues found")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True, help="Path to the dataset/ folder")
    ap.add_argument(
        "--apply-renames", action="store_true",
        help="Actually write count-matched name refreshes into data.corrected.yaml (indices are never "
             "changed either way). Default is a dry run: data.corrected.yaml is an exact copy of data.yaml "
             "and the report just shows what would change.",
    )
    ap.add_argument("--strict", action="store_true", help="Exit with code 1 if anything still differs from data.yaml")
    args = ap.parse_args()
    sys.exit(build_corrected_yaml(args.dataset_dir, apply_renames=args.apply_renames, strict=args.strict))


if __name__ == "__main__":
    main()
