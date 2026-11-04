"""`test-provider`: check that the configured model returns valid JSON and calls tools."""
from __future__ import annotations

import sys

from cli.common import load_cli_settings, make_provider
from config.credentials import MissingCredentialError
from core.camera import load_image
from core.providers.base import ImagePart
from core.providers.check import check_provider, format_check
from core.tools.crop import downscale_to_max, encode_png


def add_parser(subparsers):
    p = subparsers.add_parser("test-provider", help="check JSON validity and tool calls of the model")
    p.add_argument("images", nargs="*", help="a few sample images (a built-in test picture is used if none)")
    p.add_argument("--model")
    p.add_argument("--settings")
    p.set_defaults(func=run)


def run(args) -> int:
    settings = load_cli_settings(args.settings)
    overrides = {"model": args.model} if args.model else None
    try:
        provider = make_provider(settings, overrides)
    except (MissingCredentialError, ValueError) as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 1
    images = [ImagePart(encode_png(downscale_to_max(load_image(p).rgb, settings["provider"]["max_image_px"])[0]))
              for p in args.images]
    result = check_provider(provider, images or None)
    print(format_check(f"{provider.name}/{provider.model}", result))
    return 0 if result.valid_json_rate > 0 else 1
