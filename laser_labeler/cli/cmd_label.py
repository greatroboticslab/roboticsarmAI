"""`label`: label images one at a time and write sidecars, transcripts and the measurements CSV."""
from __future__ import annotations

import sys

from cli.common import data_dir_for, load_cli_settings, load_project_taxonomy, make_guard, make_provider
from config.credentials import MissingCredentialError
from core.labeler.label_image import label_image
from core.providers.cost import SpendCapExceeded


def add_parser(subparsers):
    p = subparsers.add_parser("label", help="label images (nothing is uploaded)")
    p.add_argument("images", nargs="+")
    p.add_argument("--mode", choices=["pipeline", "agent"], help="override the run mode in settings")
    p.add_argument("--tag", action="append", default=[], help="image tag, e.g. camera:logitech (repeatable)")
    p.add_argument("--model", help="override the model name for this run")
    p.add_argument("--data-dir")
    p.add_argument("--settings")
    p.set_defaults(func=run)


def summarize(result) -> str:
    parts = [f"{len(result.laser_boxes)} laser, {len(result.object_boxes)} object boxes"]
    if result.questions:
        parts.append(f"{len(result.questions)} question(s)")
    if result.notes:
        parts.append(result.notes)
    return f"{result.image_name}: {result.status} ({result.run_mode or 'no mode'}) - " + "; ".join(parts)


def run(args) -> int:
    settings = load_cli_settings(args.settings)
    data_dir = data_dir_for(settings, args.data_dir)
    overrides = {"model": args.model} if args.model else None
    try:
        provider = make_provider(settings, overrides)
    except (MissingCredentialError, ValueError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 1
    guard, taxonomy = make_guard(settings, data_dir, overrides), load_project_taxonomy()
    failures = 0
    for path in args.images:
        try:
            result = label_image(path, provider=provider, taxonomy=taxonomy, settings=settings, data_dir=data_dir,
                                 tags=args.tag, mode=args.mode, guard=guard)
        except SpendCapExceeded as exc:
            print(f"stopped: {exc}", file=sys.stderr)
            return 3
        failures += result.status == "failed"
        print(summarize(result))
    print(f"output in {data_dir}")
    return 1 if failures else 0
