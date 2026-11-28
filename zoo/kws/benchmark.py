"""Benchmark the converted keyword-spotting graph on any registered Pygri target.

Needs only `nir` plus the target's own dependencies (torch is not required), so the
same script runs in the snnTorch environment and in the Python 3.10 Lava environment.

    python zoo/kws/benchmark.py --build build/kws --target snntorch --timesteps 10 20 50 100
    python zoo/kws/benchmark.py --build build/kws --target lava --timesteps 50 --limit 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nir
import numpy as np

from pygri import targets
from pygri.targets.base import Workload


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--build", type=Path, default=Path("build/kws"))
    p.add_argument("--target", required=True)
    p.add_argument("--timesteps", type=int, nargs="+", default=[50])
    p.add_argument("--limit", type=int, default=None, help="use the first N test clips")
    args = p.parse_args()

    graph = nir.read(str(args.build / "kws.nir"))
    d = np.load(args.build / "test.npz")
    x, y = d["x"][: args.limit], d["y"][: args.limit]
    samples = x
    if args.target == "snntorch":
        import torch

        samples = torch.as_tensor(x)

    rows = []
    for t in args.timesteps:
        r = targets.get(args.target).run(
            Workload(name="kws-mlp", graph=graph, samples=samples, labels=y, timesteps=t)
        )
        row = {"target": args.target, "timesteps": t, "samples": len(y), "ok": r.ok, "error": r.error}
        row.update({k: m.value for k, m in r.metrics.items()})
        rows.append(row)
        print(json.dumps(row))
    out = args.build / f"bench_{args.target}.json"
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
