"""`camera-check`: report background brightness and drift per camera."""
from __future__ import annotations

from pathlib import Path

from cli.common import load_cli_settings
from core.tools.camera_check import camera_check, format_report

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def add_parser(subparsers):
    p = subparsers.add_parser("camera-check", help="does background brightness drift per camera?")
    p.add_argument("folder")
    p.add_argument("--sample", type=int, default=50, help="images to measure (0 = all)")
    p.add_argument("--camera-from", choices=["tag", "folder"], help="where the camera id comes from "
                   "(tags are not available offline, so folder is the useful choice here)")
    p.add_argument("--settings")
    p.set_defaults(func=run)


def run(args) -> int:
    settings = load_cli_settings(args.settings)
    files = sorted(p for p in Path(args.folder).rglob("*") if p.suffix.lower() in IMAGE_TYPES)
    if not files:
        print(f"no images found in {args.folder}")
        return 1
    source = args.camera_from or settings["labeling"]["camera_source"]
    reports = camera_check(files, sample=args.sample or None, source=source,
                           tag_prefix=settings["labeling"]["camera_tag_prefix"])
    print(format_report(reports))
    print("\nTip: if a camera is drifting, lock exposure, gain, focus and white balance in the webcam "
          "software if your model allows it.")
    return 0
