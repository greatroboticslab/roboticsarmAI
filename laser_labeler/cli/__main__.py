"""Command line: python -m cli <command> ...  (see `python -m cli --help`)."""
from __future__ import annotations

import argparse
import sys

from cli import cmd_camera, cmd_compare, cmd_credentials, cmd_detect, cmd_label, cmd_provider, cmd_queue

COMMANDS = (cmd_credentials, cmd_detect, cmd_label, cmd_queue, cmd_camera, cmd_provider, cmd_compare)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cli", description="Laser Labeler command line")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for module in COMMANDS:
        module.add_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
