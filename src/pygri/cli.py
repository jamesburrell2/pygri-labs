"""Command line entry point: `pygri`."""

from __future__ import annotations

import argparse
import sys

from pygri import __version__, targets


def _cmd_targets(_: argparse.Namespace) -> int:
    registered = targets.all_targets()
    if not registered:
        print("No targets registered yet.")
        return 0
    for cls in registered:
        ok, reason = cls().available()
        status = "ready" if ok else f"unavailable: {reason}"
        print(f"{cls.info.name:<20} {cls.info.kind.value:<15} {status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pygri", description=__doc__)
    parser.add_argument("--version", action="version", version=f"pygri {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("targets", help="list registered targets and whether they are usable")
    args = parser.parse_args(argv)

    if args.command == "targets":
        return _cmd_targets(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
