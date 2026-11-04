"""Phase 2 acceptance check against YOUR real Roboflow project.

Read-only by default. Run from the project folder:

    python scripts/roboflow_check.py --workspace my-ws --project my-project

Only with --upload-test --image-id <id> does it write: one hand-made single-box
annotation (class las_other, saved with addToDataset=false) plus the tag
laser-labeler-test, on that ONE image. It will not replace an existing annotation
unless you also pass --overwrite. The key is read from the keyring/environment
and is never printed. Paste the block between the === lines back to the builder.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.credentials import MissingCredentialError, require_key  # noqa: E402
from core.models import ImageLabelSet, LaserBox  # noqa: E402
from core.roboflow.auth import RoboflowError, list_projects, verify_roboflow_key  # noqa: E402
from core.roboflow.http import build_url, quote_seg, request_json  # noqa: E402
from core.roboflow.images import (RemoteImage, count_images, download_image, is_unlabeled,  # noqa: E402
                                  list_images, parse_image)
from core.roboflow.upload import build_upload_plan, upload_annotations  # noqa: E402

TEST_TAG = "laser-labeler-test"
report: list[str] = []


def note(key: str, value: object) -> None:
    report.append(f"{key}: {value}")
    print(f"  {key}: {value}")


def step(title: str):
    """Decorator: run a step, record RoboflowError instead of crashing."""
    def wrap(fn):
        def run(*a, **kw):
            print(f"\n[{title}]")
            try:
                return fn(*a, **kw)
            except RoboflowError as exc:
                note(f"{title} FAILED", exc)
            except Exception as exc:  # noqa: BLE001 - keep going, report it
                note(f"{title} CRASHED", f"{type(exc).__name__}: {exc}")
        return run
    return wrap


@step("1 verify key")
def check_key(key: str) -> None:
    res = verify_roboflow_key(key)
    note("key state", res.state)
    note("workspace ids", res.workspace_ids or "(none returned)")
    note("message", res.message)


@step("2 list projects")
def check_projects(ws: str, project: str, key: str) -> None:
    projects = list_projects(ws, key)
    note("project count", len(projects))
    note("target project listed", any(p.id == project for p in projects))


@step("3 count images")
def check_counts(ws: str, project: str, key: str) -> None:
    note("total images", count_images(ws, project, key))
    note("unlabeled images (client-side, full paging)", count_images(ws, project, key, only_unlabeled=True))


@step("4 raw search response shape (keys only, no values)")
def check_search_shape(ws: str, project: str, key: str) -> None:
    url = build_url(f"{quote_seg(ws)}/{quote_seg(project)}/search", key)
    body = {"offset": 0, "limit": 2, "fields": ["id", "name", "split", "tags", "labels", "created"]}
    data = request_json("POST", url, body=body, key_for_masking=key)
    note("top-level keys", sorted(data) if isinstance(data, dict) else type(data).__name__)
    rows = data.get("results") if isinstance(data, dict) else None
    note("rows under 'results'", isinstance(rows, list))
    if rows:
        note("row keys", sorted(rows[0]))
        note("labels value type", type(rows[0].get("labels")).__name__)
    note("rows returned for limit=2", len(rows) if isinstance(rows, list) else "n/a")


@step("5 list first 20 images")
def check_first_20(ws: str, project: str, key: str) -> list[RemoteImage]:
    images = list(list_images(ws, project, key, limit=20, page_size=20))
    states = [is_unlabeled(i) for i in images]
    note("fetched", len(images))
    note("labeled / unlabeled / unknown", f"{states.count(False)} / {states.count(True)} / {states.count(None)}")
    note("with width+height in search rows", sum(1 for i in images if i.width and i.height))
    note("with a url in search rows", sum(1 for i in images if i.url))
    return images


@step("6 download one image")
def check_download(image: RemoteImage, key: str, dest: Path) -> Path:
    path = download_image(image, dest, key)
    head = path.read_bytes()[:12]
    note("saved as", path.name)
    note("bytes", path.stat().st_size)
    note("looks like an image", head[:3] == b"\xff\xd8\xff" or head[:4] == b"\x89PNG" or head[:2] == b"BM"
         or head[8:12] == b"WEBP")
    return path


def image_size(path: Path, args: argparse.Namespace) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except ImportError:
        if args.width and args.height:
            return args.width, args.height
        raise RoboflowError("Pillow is not installed: pass --width and --height of the test image.")


def fetch_details(ws: str, project: str, image_id: str, key: str) -> RemoteImage:
    url = build_url(f"{quote_seg(ws)}/{quote_seg(project)}/images/{quote_seg(image_id)}", key)
    data = request_json("GET", url, key_for_masking=key)
    image = parse_image(data.get("image") if isinstance(data, dict) else None, ws, project)
    if image is None:
        raise RoboflowError(f"Image {image_id} was not found (details response had no image).")
    return image


def test_label_set(image: RemoteImage, width: int, height: int) -> ImageLabelSet:
    box = LaserBox("t1", "other", (10, 10, 50, 50), 0)
    return ImageLabelSet(image.id, image.name, "test", False, 0.0, laser_boxes=[box], status="approved",
                         image_width=width, image_height=height)


@step("7 upload test (writes to ONE image)")
def check_upload(args: argparse.Namespace, key: str, dest: Path) -> None:
    before = fetch_details(args.workspace, args.project, args.image_id, key)
    note("image name", before.name)
    note("labels before", before.labels)
    if before.labels and not args.overwrite:
        note("SKIPPED", "image already has labels; pick an unlabeled image or add --overwrite")
        return
    path = download_image(before, dest, key)
    width, height = image_size(path, args)
    plan = build_upload_plan([test_label_set(before, width, height)], workspace=args.workspace,
                             project=args.project, overwrite=args.overwrite, add_to_dataset=False,
                             remote_labels={before.id: before.labels})
    for item in plan.items:
        item.tags = [TEST_TAG]
    summary = upload_annotations(plan, key, dry_run=False, audit_log=dest / "audit.jsonl")
    note("uploaded", summary.uploaded)
    note("failed", summary.failed or "none")
    note("tag_failed", summary.tag_failed or "none")
    note("skipped", summary.skipped or "none")
    after = fetch_details(args.workspace, args.project, args.image_id, key)
    note("labels after", after.labels)
    note("new class las_other visible on image", "las_other" in (after.labels or []))
    note("tags after", after.tags)
    note("test tag present", TEST_TAG in after.tags)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--upload-test", action="store_true", help="WRITE one test annotation + tag")
    ap.add_argument("--image-id", help="the ONE image for --upload-test")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing that image's annotation")
    ap.add_argument("--width", type=int, help="test image width if Pillow is missing")
    ap.add_argument("--height", type=int)
    ap.add_argument("--download-dir", type=Path, help="default: a temporary folder")
    args = ap.parse_args(argv)
    if args.upload_test and not args.image_id:
        ap.error("--upload-test needs --image-id")
    try:
        key = require_key("roboflow")
    except MissingCredentialError as exc:
        print(exc)
        return 2
    dest = args.download_dir or Path(tempfile.mkdtemp(prefix="roboflow_check_"))
    print(f"Mode: {'READ + WRITE one image' if args.upload_test else 'read-only'}. Files go to {dest}")
    check_key(key)
    check_projects(args.workspace, args.project, key)
    check_counts(args.workspace, args.project, key)
    check_search_shape(args.workspace, args.project, key)
    images = check_first_20(args.workspace, args.project, key) or []
    if images:
        check_download(images[0], key, dest)
    if args.upload_test:
        check_upload(args, key, dest)
    print("\n=== PASTE THIS BACK ===")
    print("\n".join(report))
    print("=== END ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
