"""Lava simulator target (Intel's open-source neuromorphic framework).

Runs a NIR graph with Lava's floating-point CPU models (`Loihi2SimCfg`). This is a
software simulation of Loihi-style dynamics, not a run on Loihi hardware.

Mapping:

- `Affine` / `Linear` become `Dense` processes. Biases are fed through a second
  `Dense` from a bias source, so they line up in time with the synaptic input.
- `IF` becomes `LIFReset` with `du = 1` (the current is the input of this step),
  `dv = 0` (no leak) and reset to zero, which is integrate-and-fire.
- Samples run back to back in a single Lava run. Each `Dense` adds one timestep of
  delay, so layer k's window starts k steps after the input; every layer resets its
  state at the start of its own window.

Lava (`lava-nc`) supports Python 3.10 only and pins `numpy < 2`, so this target is an
optional extra that usually lives in its own environment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pygri.results import Metric, Provenance, TargetResult
from pygri.targets import register
from pygri.targets._nir_chain import UnsupportedGraphError, chain, if_threshold
from pygri.targets.base import Target, TargetInfo, TargetKind, Workload


@dataclass
class _Layer:
    weight: object  # np.ndarray, shape (out, in)
    bias: object  # np.ndarray, shape (out,)
    threshold: float


def parse_layers(graph) -> list[_Layer]:
    """Turn a chain of (Affine|Linear, IF) pairs into layers, or raise UnsupportedGraphError."""
    import nir
    import numpy as np

    layers: list[_Layer] = []
    pending = None
    for key in chain(graph):
        node = graph.nodes[key]
        if isinstance(node, (nir.Input, nir.Output, nir.Flatten)):
            continue
        if isinstance(node, (nir.Affine, nir.Linear)):
            if pending is not None:
                raise UnsupportedGraphError(f"node '{key}': two weight layers in a row")
            w = np.asarray(node.weight, dtype=float)
            b = np.asarray(node.bias, dtype=float) if isinstance(node, nir.Affine) else None
            pending = (w, np.zeros(w.shape[0]) if b is None else b)
        elif isinstance(node, nir.IF):
            if pending is None:
                raise UnsupportedGraphError(f"node '{key}': IF must follow a weight layer")
            layers.append(_Layer(pending[0], pending[1], if_threshold(key, node)))
            pending = None
        else:
            raise UnsupportedGraphError(f"node '{key}': {type(node).__name__} is not supported")
    if pending is not None or not layers:
        raise UnsupportedGraphError("graph must end with an IF layer")
    return layers


def simulate(layers: list[_Layer], x, timesteps: int):
    """Run every sample in `x` (shape (n, features)) for `timesteps` steps of constant input.

    Returns (output spike counts with shape (n, classes), total spikes over all layers).
    """
    import numpy as np
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.proc.dense.process import Dense
    from lava.proc.io.sink import RingBuffer as Sink
    from lava.proc.io.source import RingBuffer as Source
    from lava.proc.lif.process import LIFReset

    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    n, depth = x.shape[0], len(layers)
    period = timesteps + depth + 1  # steps per sample, with room for the pipeline delay
    total = n * period

    # Input: sample i is presented during [i * period, i * period + timesteps).
    data = np.zeros((x.shape[1], total))
    for i in range(n):
        data[:, i * period : i * period + timesteps] = x[i][:, None]
    source = Source(data=data)

    prev_out = source.s_out
    sinks = []
    for k, layer in enumerate(layers, start=1):
        size = layer.weight.shape[0]
        # Layer k sees its input during [i * period + k, ... + timesteps). Lava counts
        # time steps from 1, so array index j is time step j + 1.
        neurons = LIFReset(
            shape=(size,),
            du=1.0,
            dv=0.0,
            vth=layer.threshold,
            reset_interval=period,
            reset_offset=(k + 1) % period,
        )
        dense = Dense(weights=layer.weight, num_message_bits=24 if k == 1 else 0)
        prev_out.connect(dense.s_in)
        dense.a_out.connect(neurons.a_in)

        if np.any(layer.bias):
            on = np.zeros((1, total))
            for i in range(n):
                start = i * period + k - 1
                on[0, start : start + timesteps] = 1.0
            bias_src = Source(data=on)
            bias_dense = Dense(weights=layer.bias[:, None], num_message_bits=24)
            bias_src.s_out.connect(bias_dense.s_in)
            bias_dense.a_out.connect(neurons.a_in)

        sink = Sink(shape=(size,), buffer=total)
        neurons.s_out.connect(sink.a_in)
        sinks.append(sink)
        prev_out = neurons.s_out

    source.run(condition=RunSteps(num_steps=total), run_cfg=Loihi2SimCfg(select_tag="floating_pt"))
    try:
        spikes = [s.data.get() for s in sinks]  # each (neurons, total)
    finally:
        source.stop()

    total_spikes = float(sum(s.sum() for s in spikes))
    out = spikes[-1]
    counts = np.zeros((n, out.shape[0]))
    for i in range(n):
        start = i * period + depth
        counts[i] = out[:, start : start + timesteps].sum(axis=1)
    return counts, total_spikes


@register
class LavaSimulator(Target):
    info = TargetInfo(
        name="lava",
        kind=TargetKind.SIMULATOR,
        vendor="Intel Lava (open source)",
        description="Runs NIR graphs as Lava integrate-and-fire processes on CPU "
        "(floating-point simulation, not Loihi hardware).",
        requires=("lava-nc", "nir"),
    )

    def available(self) -> tuple[bool, str]:
        try:
            import lava.proc.lif.process  # noqa: F401
            import nir  # noqa: F401
        except ImportError as exc:
            return False, f"missing package: {exc.name} (lava-nc needs Python 3.10)"
        return True, ""

    def run(self, workload: Workload) -> TargetResult:
        ok, reason = self.available()
        if not ok:
            return TargetResult(target=self.info.name, model=workload.name, ok=False, error=reason)
        import numpy as np

        try:
            layers = parse_layers(workload.graph)
        except UnsupportedGraphError as exc:
            return TargetResult(
                target=self.info.name, model=workload.name, ok=False, error=str(exc)
            )

        x = np.asarray(workload.samples, dtype=float)
        y = np.asarray(workload.labels)
        n = int(x.shape[0])

        start = time.perf_counter()
        counts, total_spikes = simulate(layers, x, workload.timesteps)
        elapsed = time.perf_counter() - start

        acc = float((counts.argmax(axis=1) == y).mean())
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
                    note="CPU wall-clock of the Lava simulation per sample, including "
                    "process start-up; not a hardware latency",
                ),
                "spikes_per_inference": Metric(total_spikes / n, "spikes", sim),
            },
            details={"samples": n, "timesteps": workload.timesteps, "run_config": "floating_pt"},
        )
