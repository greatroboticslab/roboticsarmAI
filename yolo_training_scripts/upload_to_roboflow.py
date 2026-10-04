#!/usr/bin/env python3
"""
upload_to_roboflow.py

Uploads this local dataset (images + YOLO-format labels) to an existing
Roboflow project, so the images you've added locally (and the class names
reconciled from the PDFs) end up there too -- e.g. to train using Roboflow's
own hosted training instead of (or in addition to) train_yolo.py locally.

This only uploads images+annotations into the project's raw pool. To
actually train on Roboflow afterward, you still need to "Generate a new
Version" in the Roboflow web UI (choose your split/preprocessing/
augmentation settings there) and hit Train.

Non-destructive by default: it copies just the standard train/valid/test +
data.yaml pieces into a clean staging folder (leaving out pdf_for_labels/,
backup folders, reports, etc. that would confuse Roboflow's uploader) and
PREVIEWS what would be uploaded without calling the Roboflow API. Pass
--upload once you've checked the preview and are ready to actually push it.

Setup:
    pip install roboflow
    Get an API key from your Roboflow account settings, then either:
      - set it as an environment variable: setx ROBOFLOW_API_KEY "..." (Windows) / export ROBOFLOW_API_KEY=... (bash)
      - or pass it directly with --api-key

Usage:
    # preview only -- stages the clean folder, prints counts/classes, uploads nothing
    python upload_to_roboflow.py --dataset-dir ../dataset --project your-project-slug

    # try a small batch first (recommended before uploading everything)
    python upload_to_roboflow.py --dataset-dir ../dataset --project your-project-slug --limit 5 --upload

    # actually upload everything
    python upload_to_roboflow.py --dataset-dir ../dataset --project your-project-slug --upload
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

from yolo_common import resolve_data_yaml, load_data_yaml, find_split_dirs, list_image_label_pairs


def stage_clean_dataset(dataset_dir: Path, data_yaml_path: Path, staging_dir: Path, limit: int | None) -> dict:
    """Copy just data.yaml + train/valid/test images+labels into a clean staging
    folder Roboflow's uploader expects, leaving out pdf_for_labels/, backups,
    reports, and anything else that isn't part of the standard layout."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    data = load_data_yaml(data_yaml_path)
    with open(staging_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    split_dirs = find_split_dirs(dataset_dir)
    counts = {}
    for split, split_dir in split_dirs.items():
        pairs = list_image_label_pairs(split_dir)
        if limit is not None:
            pairs = pairs[:limit]
        img_out = staging_dir / split / "images"
        lbl_out = staging_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, img_out / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)
            else:
                (lbl_out / f"{img_path.stem}.txt").write_text("")
        counts[split] = len(pairs)

    return {"counts": counts, "nc": data.get("nc", len(data.get("names", []))), "names": data.get("names", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", type=Path, required=True, help="Path to the dataset/ folder")
    ap.add_argument("--data", type=Path, default=None, help="Explicit data.yaml, overrides --dataset-dir auto-detection (data.corrected.yaml preferred)")
    ap.add_argument("--project", required=True, help="Roboflow project slug/id (from the project's URL) -- images are ADDED to this project if it already exists")
    ap.add_argument("--workspace", default=None, help="Roboflow workspace name/id. Default: your account's default workspace")
    ap.add_argument("--api-key", default=None, help="Roboflow API key. Default: the ROBOFLOW_API_KEY environment variable")
    ap.add_argument("--project-type", default="object-detection")
    ap.add_argument("--project-license", default="MIT", help="Only used if --project doesn't exist yet and gets created")
    ap.add_argument("--num-workers", type=int, default=10)
    ap.add_argument("--batch-name", default=None, help="Optional label to group this upload in the Roboflow web UI")
    ap.add_argument("--limit", type=int, default=None, help="Only stage/upload the first N images per split -- use this to try a small batch before uploading everything")
    ap.add_argument("--upload", action="store_true", help="Actually call the Roboflow API. Without this, only stages the clean folder and previews counts (uploads nothing).")
    ap.add_argument("--cleanup", action="store_true", help="Delete the staging folder after a successful upload")
    args = ap.parse_args()

    data_yaml_path = resolve_data_yaml(args.dataset_dir, args.data)

    staging_dir = args.dataset_dir / f"_roboflow_upload_staging_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"Staging a clean copy at: {staging_dir}")
    info = stage_clean_dataset(args.dataset_dir, data_yaml_path, staging_dir, args.limit)

    print("\n" + "=" * 70)
    print("UPLOAD PREVIEW")
    print("=" * 70)
    print(f"nc: {info['nc']}")
    print("classes (first 5):")
    for n in info["names"][:5]:
        print(f"  - {n}")
    if len(info["names"]) > 5:
        print(f"  ... and {len(info['names']) - 5} more")
    print("\nimages staged per split:")
    for split, n in info["counts"].items():
        print(f"  {split}: {n}")
    if args.limit is not None:
        print(f"\n(--limit {args.limit} applied -- this is a partial batch, not the full dataset)")

    if not args.upload:
        print(f"\nThis was a preview only -- nothing was uploaded. Staged files are at: {staging_dir}")
        print("Review them, then re-run with --upload to actually push this to Roboflow.")
        return

    try:
        import roboflow
    except ImportError:
        print("roboflow is not installed. Install it with:\n    pip install roboflow", file=sys.stderr)
        sys.exit(1)

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    rf = roboflow.Roboflow(api_key=api_key)
    workspace = rf.workspace(args.workspace) if args.workspace else rf.workspace()

    print(f"\nUploading to project '{args.project}'...")
    result = workspace.upload_dataset(
        str(staging_dir),
        args.project,
        num_workers=args.num_workers,
        project_license=args.project_license,
        project_type=args.project_type,
        batch_name=args.batch_name,
    )
    print(f"Upload result: {result}")
    print(
        "\nImages are now in the project's raw pool. To train on Roboflow, go to the project in the "
        "web UI, 'Generate' a new Version (pick your split/preprocessing/augmentation settings there), "
        "then hit Train."
    )

    if args.cleanup:
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"Cleaned up staging folder: {staging_dir}")
    else:
        print(f"Staging folder left at: {staging_dir} (pass --cleanup next time to remove it automatically)")


if __name__ == "__main__":
    main()
