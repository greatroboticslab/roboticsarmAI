"""`queue`: add images, run the labeler over them, inspect and approve results, dry-run uploads.

Nothing is uploaded unless `queue upload --send` is used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from cli.common import (data_dir_for, load_cli_settings, load_project_taxonomy, make_guard, make_provider)
from config import credentials
from config.credentials import MissingCredentialError
from core.local_files import collect_files, make_image_id
from core.labeler.factory import build_escalation, build_label_fn
from core.labeler.runner import STOP_TEXT, run_queue
from core.queue_db import NewImage, QueueDB
from core.reports import manifest
from core.reports.sidecar import read_sidecar, safe_id
from core.roboflow.auth import RoboflowError
from core.roboflow.enqueue import enqueue_remote
from core.roboflow.images import download_image, list_images
from core.roboflow.upload import build_upload_plan, upload_annotations
from core.schema.coco_export import build_coco

def add_parser(subparsers):
    p = subparsers.add_parser("queue", help="resumable labeling queue")
    p.set_defaults(func=run)
    p.add_argument("--data-dir")
    p.add_argument("--settings")
    sub = p.add_subparsers(dest="queue_action", required=True, metavar="action")
    add = sub.add_parser("add", help="add local image files or folders")
    add.add_argument("paths", nargs="+")
    add.add_argument("--tag", action="append", default=[], help="e.g. camera:logitech")
    rf = sub.add_parser("add-roboflow", help="download images from your Roboflow project and add them")
    rf.add_argument("--workspace")
    rf.add_argument("--project")
    rf.add_argument("--unlabeled", action="store_true", help="only images without labels (plus any whose state is unknown)")
    rf.add_argument("--split")
    rf.add_argument("--tag")
    rf.add_argument("--limit", type=int, help="stop after this many NEW images (for example 250)")
    go = sub.add_parser("run", help="label queued images, one at a time")
    go.add_argument("--limit", type=int, help="label at most this many images")
    go.add_argument("--mode", choices=["pipeline", "agent"])
    go.add_argument("--model")
    go.add_argument("--max-failures", type=int, default=5, help="stop after this many failures in a row")
    sub.add_parser("status", help="counts per status")
    ls = sub.add_parser("list", help="list images")
    ls.add_argument("--status")
    ls.add_argument("--sort", default="confidence_asc", choices=["order", "confidence_asc", "confidence_desc", "recent"])
    ls.add_argument("--limit", type=int, default=30)
    sub.add_parser("retry-failed", help="put failed images back in the queue")
    rq = sub.add_parser("requeue", help="label these images again")
    rq.add_argument("image_ids", nargs="+")
    ap = sub.add_parser("approve", help="approve pending images with no questions at or above a confidence")
    ap.add_argument("--min-confidence", type=float, required=True)
    ex = sub.add_parser("export-coco", help="write approved/edited labels as COCO JSON")
    ex.add_argument("--out", required=True)
    up = sub.add_parser("upload", help="upload approved labels to Roboflow (dry run unless --send)")
    up.add_argument("--workspace")
    up.add_argument("--project")
    up.add_argument("--send", action="store_true", help="really upload (default is a dry run)")
    up.add_argument("--overwrite", action="store_true")
    return p


def _add(db, args, settings, data_dir) -> int:
    batch: dict[str, Path] = {}
    items = [NewImage(make_image_id(db, p, batch), p.name, str(p), "local", args.tag) for p in collect_files(args.paths)]
    added = db.add_images(items)
    print(f"added {added} new image(s); {len(items) - added} already in the queue")
    return 0 if items else 1


def _add_roboflow(db, args, settings, data_dir) -> int:
    ws = args.workspace or settings["roboflow"]["workspace"]
    proj = args.project or settings["roboflow"]["project"]
    if not ws or not proj:
        print("give --workspace and --project (or save them on the Sign in tab)", file=sys.stderr)
        return 1
    try:
        key = credentials.require_key("roboflow")
        images = list_images(ws, proj, key, only_unlabeled=args.unlabeled, tag=args.tag, split=args.split)
        result = enqueue_remote(db, images, lambda img: download_image(img, data_dir / "images", key), limit=args.limit)
    except (MissingCredentialError, RoboflowError) as exc:
        print(f"Roboflow: {exc}", file=sys.stderr)
        return 1
    print(f"added {result.added}, already known {result.already_known}, failed {len(result.failed)}")
    if result.unknown_label_state:
        print(f"{result.unknown_label_state} added image(s) have an unknown label state (Roboflow did not say).")
    for image_id, why in list(result.failed.items())[:5]:
        print(f"  failed {image_id}: {why}")
    return 0


def _run(db, args, settings, data_dir) -> int:
    overrides = {"model": args.model} if args.model else None
    try:
        provider = make_provider(settings, overrides)
    except (MissingCredentialError, ValueError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 1
    guard, taxonomy = make_guard(settings, data_dir, overrides), load_project_taxonomy()
    try:
        escalation = build_escalation(settings, data_dir)
    except (MissingCredentialError, ValueError) as exc:
        print(f"cannot start (escalation setting): {exc}", file=sys.stderr)
        return 1
    label_fn = build_label_fn(settings, provider, taxonomy, data_dir, guard=guard, mode=args.mode, escalation=escalation)

    def progress(summary, row, result):
        eta = f"~{summary.eta_seconds / 60:.0f} min left" if summary.queued_left else "queue empty"
        status = result.status if result else "failed"
        print(f"[{summary.processed}] {row.name}: {status}  ({summary.seconds_per_image:.1f}s/image, {eta})", flush=True)

    summary = run_queue(db, label_fn, sidecar_dir=data_dir / "sidecars", limit=args.limit,
                        max_consecutive_failures=args.max_failures, on_progress=progress)
    print(f"\nlabeled {summary.processed}: {summary.pending} ready for review, {summary.needs_input} need your input, "
          f"{summary.failed} failed; {summary.queued_left} still queued")
    if summary.stopped:
        print(f"run {STOP_TEXT.get(summary.stopped, summary.stopped)}")
    return 3 if summary.stopped in ("spend_cap", "too_many_failures") else 0


def _status(db, args, settings, data_dir) -> int:
    for status, count in db.counts().items():
        print(f"{status:<12}{count:>7}")
    return 0


def _list(db, args, settings, data_dir) -> int:
    for r in db.list(args.status, args.sort, args.limit):
        conf = f"{r.confidence:.2f}" if r.confidence is not None else "  - "
        print(f"{r.image_id:<28}{r.status:<12}conf {conf}  q={r.question_count}  {r.run_mode or '-':<9}{r.error or ''}")
    return 0


def _retry(db, args, settings, data_dir) -> int:
    print(f"re-queued {db.requeue([r.image_id for r in db.list('failed')])} failed image(s)")
    return 0


def _requeue(db, args, settings, data_dir) -> int:
    print(f"re-queued {db.requeue(args.image_ids)} image(s)")
    return 0


def _approve(db, args, settings, data_dir) -> int:
    ids = db.bulk_approve(args.min_confidence)
    for image_id in ids:
        manifest.set_status(data_dir / "manifest.csv", image_id, "approved")
    print(f"approved {len(ids)} image(s) at confidence >= {args.min_confidence}")
    return 0


def _approved_sets(db, data_dir):
    sets = []
    for row in db.list(["approved", "edited"]):
        if row.sidecar_path and Path(row.sidecar_path).is_file():
            ls = read_sidecar(row.sidecar_path)
            ls.status = row.status  # the queue is the source of truth for approval
            sets.append(ls)
    return sets


def _export(db, args, settings, data_dir) -> int:
    sets = _approved_sets(db, data_dir)
    Path(args.out).write_text(json.dumps(build_coco(sets, settings["labeling"]["color_placeholder"])), encoding="utf-8")
    print(f"wrote {len(sets)} image(s) to {args.out}")
    return 0 if sets else 1


def _upload(db, args, settings, data_dir) -> int:
    ws = args.workspace or settings["roboflow"]["workspace"]
    proj = args.project or settings["roboflow"]["project"]
    if not ws or not proj:
        print("give --workspace and --project", file=sys.stderr)
        return 1
    try:
        key = credentials.require_key("roboflow")
    except MissingCredentialError as exc:
        print(exc, file=sys.stderr)
        return 1
    plan = build_upload_plan(_approved_sets(db, data_dir), workspace=ws, project=proj,
                             color_placeholder=settings["labeling"]["color_placeholder"], overwrite=args.overwrite)
    summary = upload_annotations(plan, key, dry_run=not args.send, audit_log=data_dir / "upload_audit.jsonl")
    for image_id in summary.uploaded:
        db.set_status(image_id, "uploaded")
        manifest.set_status(data_dir / "manifest.csv", image_id, "uploaded")
    if not args.send:
        print("\nDRY RUN: nothing was sent. Add --send to upload.")
    else:
        print(f"uploaded {len(summary.uploaded)}, failed {len(summary.failed)}, skipped {len(summary.skipped)}")
    return 1 if summary.failed else 0


ACTIONS = {"add": _add, "add-roboflow": _add_roboflow, "run": _run, "status": _status, "list": _list,
           "retry-failed": _retry, "requeue": _requeue, "approve": _approve, "export-coco": _export,
           "upload": _upload}


def run(args) -> int:
    settings = load_cli_settings(args.settings)
    data_dir = data_dir_for(settings, args.data_dir)
    db = QueueDB(data_dir / "queue.db")
    try:
        return ACTIONS[args.queue_action](db, args, settings, data_dir)
    finally:
        db.close()
