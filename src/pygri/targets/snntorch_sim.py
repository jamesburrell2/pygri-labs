"""snnTorch simulator target.

Loads a NIR graph into snnTorch neurons with Pygri's own loader, so the
semantics are explicit: `IF` nodes become non-leaky neurons (beta = 1) that
reset to zero, inputs are a constant current every timestep, and the class is
the output neuron with the most spikes.

Supported NIR nodes: Input, Output, Flatten, Affine, Linear, IF. Graphs must
be a single chain. Anything else returns `ok=False` with the reason.
"""

from __future__ import annotations

import time

from pygri.results import Metric, Provenance, TargetResult
from pygri.targets import register
from pygri.targets.base import Target, TargetInfo, TargetKind, Workload


class UnsupportedGraphError(ValueError):
    pass


def _chain(graph) -> list[str]:
    nexts = dict(graph.edges)
    if len(nexts) != len(graph.edges):
        raise UnsupportedGraphError("only single-chain graphs are supported")
    inputs = [k for k, n in graph.nodes.items() if type(n).__name__ == "Input"]
    if len(inputs) != 1:
        raise UnsupportedGraphError("graph must have exactly one Input node")
    order, cur = [], inputs[0]
    while cur is not None:
        order.append(cur)
        cur = nexts.get(cur)
    if len(order) != len(graph.nodes):
        raise UnsupportedGraphError("graph is not a single chain")
    return order


def build_network(graph):
    """Return a list of torch modules implementing the graph, in order."""
    import nir
    import numpy as np
    import snntorch as snn
    import torch

    modules: list[torch.nn.Module] = []
    for key in _chain(graph):
        node = graph.nodes[key]
        if isinstance(node, (nir.Input, nir.Output)):
            continue
        if isinstance(node, nir.Flatten):
            modules.append(torch.nn.Flatten(start_dim=1))
        elif isinstance(node, (nir.Affine, nir.Linear)):
            w = torch.as_tensor(np.asarray(node.weight), dtype=torch.float32)
            has_bias = isinstance(node, nir.Affine)
            lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
            with torch.no_grad():
                lin.weight.copy_(w)
                if has_bias:
                    lin.bias.copy_(torch.as_tensor(np.asarray(node.bias), dtype=torch.float32))
            modules.append(lin)
        elif isinstance(node, nir.IF):
            thr = np.unique(np.asarray(node.v_threshold) * np.asarray(node.r))
            reset = np.zeros(1) if node.v_reset is None else np.unique(np.asarray(node.v_reset))
            if thr.size != 1 or reset.size != 1 or float(reset[0]) != 0.0:
                raise UnsupportedGraphError(
                    f"node '{key}': only uniform thresholds and reset-to-zero are supported"
                )
            modules.append(
                snn.Leaky(
                    beta=1.0,
                    threshold=float(thr[0]),
                    reset_mechanism="zero",
                    init_hidden=True,
                )
            )
        else:
            raise UnsupportedGraphError(f"node '{key}': {type(node).__name__} is not supported")
    return modules


def simulate(modules, x, timesteps: int):
    """Run `timesteps` steps with constant input `x`; return (spike counts, total spikes)."""
    import snntorch as snn
    import torch
    from snntorch import utils

    net = torch.nn.Sequential(*modules)
    utils.reset(net)
    counts = None
    total_spikes = 0.0
    with torch.no_grad():
        for _ in range(timesteps):
            h = x
            for m in modules:
                h = m(h)
                if isinstance(h, tuple):
                    h = h[0]
                if isinstance(m, snn.Leaky):
                    total_spikes += float(h.sum())
            counts = h if counts is None else counts + h
    return counts, total_spikes


@register
class SnnTorchSimulator(Target):
    info = TargetInfo(
        name="snntorch",
        kind=TargetKind.SIMULATOR,
        vendor="snnTorch (open source)",
        description="Runs NIR graphs as snnTorch integrate-and-fire neurons on CPU.",
        requires=("torch", "snntorch", "nir"),
    )

    def available(self) -> tuple[bool, str]:
        try:
            import nir  # noqa: F401
            import snntorch  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            return False, f"missing package: {exc.name}"
        return True, ""

    def run(self, workload: Workload) -> TargetResult:
        ok, reason = self.available()
        if not ok:
            return TargetResult(target=self.info.name, model=workload.name, ok=False, error=reason)
        import torch

        try:
            modules = build_network(workload.graph)
        except UnsupportedGraphError as exc:
            return TargetResult(
                target=self.info.name, model=workload.name, ok=False, error=str(exc)
            )

        x = workload.samples
        x = x if isinstance(x, torch.Tensor) else torch.stack(list(x))
        y = workload.labels
        y = y if isinstance(y, torch.Tensor) else torch.as_tensor(list(y))
        n = int(x.shape[0])

        start = time.perf_counter()
        counts, total_spikes = simulate(modules, x.float(), workload.timesteps)
        elapsed = time.perf_counter() - start

        acc = float((counts.argmax(dim=1) == y).float().mean())
        sim = Provenance.MEASURED_SIMULATION
        return TargetResult(
            target=self.info.name,
            model=workload.name,
            ok=True,
            metrics={
                TargetResult.ACCURACY: Metric(acc, "fraction", sim),
                TargetResult.LATENCY: Metric(
                    elapsed / n * 1e3,
                    "ms",
                    sim,
                    note="CPU wall-clock of the simulator per sample (batched); "
                    "not a hardware latency",
                ),
                "spikes_per_inference": Metric(total_spikes / n, "spikes", sim),
            },
            details={"samples": n, "timesteps": workload.timesteps},
        )
