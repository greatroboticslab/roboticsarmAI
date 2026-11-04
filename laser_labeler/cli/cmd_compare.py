"""`compare`: score several provider/mode runs against the owner's hand-labeled golden set.

runs.yaml example:
  runs:
    - {name: local-pipeline, provider: openai_compat, model: your-local-model, base_url: "http://localhost:11434/v1", mode: pipeline}
    - {name: gemini-agent, provider: gemini, model: your-gemini-model, mode: agent}
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from cli.common import data_dir_for, load_cli_settings, load_project_taxonomy, make_guard, make_provider
from config.credentials import MissingCredentialError
from core.labeler.golden import format_report, load_truth_coco, run_golden
from core.labeler.label_image import label_image
from core.reports.sidecar import safe_id

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PROVIDER_KEYS = ("provider", "model", "base_url", "temperature", "max_image_px", "price_in_per_mtok",
                 "price_out_per_mtok", "daily_cost_cap_usd")


def add_parser(subparsers):
    p = subparsers.add_parser("compare", help="compare providers/modes on the golden set")
    p.add_argument("--images", required=True, help="folder with the golden images")
    p.add_argument("--truth", required=True, help="COCO JSON export of your hand-labeled boxes")
    p.add_argument("--runs", required=True, help="YAML file listing the runs to compare")
    p.add_argument("--out", help="write the Markdown report here")
    p.add_argument("--settings")
    p.set_defaults(func=run)


def run(args) -> int:
    settings = load_cli_settings(args.settings)
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_TYPES)
    truth = load_truth_coco(args.truth)
    runs = (yaml.safe_load(Path(args.runs).read_text(encoding="utf-8")) or {}).get("runs") or []
    if not files or not runs:
        print("need at least one golden image and one run", file=sys.stderr)
        return 1
    taxonomy, scores = load_project_taxonomy(), []
    for spec in runs:
        overrides = {k: spec[k] for k in PROVIDER_KEYS if k in spec}
        mode = spec.get("mode", settings["provider"]["run_mode"])
        data_dir = data_dir_for(settings) / "golden" / safe_id(spec["name"])
        try:
            provider = make_provider(settings, overrides)
        except (MissingCredentialError, ValueError) as exc:
            print(f"skipping {spec['name']}: {exc}", file=sys.stderr)
            continue
        guard = make_guard(settings, data_dir, overrides)
        run_settings = {**settings, "provider": {**settings["provider"], **overrides}}

        def label_one(path, provider=provider, guard=guard, data_dir=data_dir, run_settings=run_settings, mode=mode):
            print(f"  [{spec['name']}] {path.name}")
            return label_image(path, provider=provider, taxonomy=taxonomy, settings=run_settings,
                               data_dir=data_dir, mode=mode, guard=guard)

        scores.append(run_golden(spec["name"], files, label_one, truth, mode, settings["labeling"]["color_placeholder"]))
    if not scores:
        return 1
    report = format_report(scores)
    print("\n" + report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
    return 0
