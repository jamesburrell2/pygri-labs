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


def _cmd_demo(args: argparse.Namespace) -> int:
    from pathlib import Path

    from pygri.demo import run_demo
    from pygri.report import render_html

    report = run_demo(timesteps=args.timesteps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.to_json())
    (out / "report.html").write_text(render_html(report))
    for r in report.results:
        acc = r.metrics.get("accuracy")
        status = f"accuracy {acc.value:.3f}" if acc else f"failed: {r.error}"
        print(f"{r.target:<26} {status}")
    print(f"Report written to {out / 'report.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pygri", description=__doc__)
    parser.add_argument("--version", action="version", version=f"pygri {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("targets", help="list registered targets and whether they are usable")
    demo = sub.add_parser("demo", help="run the end-to-end demo on synthetic data")
    demo.add_argument("--timesteps", type=int, default=50)
    demo.add_argument("--out", default="reports/demo")
    args = parser.parse_args(argv)

    if args.command == "targets":
        return _cmd_targets(args)
    if args.command == "demo":
        return _cmd_demo(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
