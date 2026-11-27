"""Rate-coded ANN-to-SNN conversion for simple feed-forward networks.

Method: data-based weight normalisation (Diehl et al., 2015). Each ReLU is
replaced by a non-leaky integrate-and-fire neuron with threshold 1. Layer
weights are rescaled by the ratio of the previous and current layer's maximum
activation on calibration data, so firing rates approximate the original
activations. Inputs are fed as a constant current every timestep, and the
class is the output neuron with the most spikes.

Supported layers in this first version: Flatten, Linear, ReLU. Anything else
raises `UnsupportedLayerError` with the layer name, so callers can report it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any


class UnsupportedLayerError(ValueError):
    pass


@dataclass
class ConversionResult:
    graph: Any  # nir.NIRGraph
    scales: list[float]  # per-layer activation maxima used for normalisation


def _activation_maxima(model, calibration, torch, percentile: float) -> list[float]:
    """Return the chosen percentile of each ReLU's output (and the final layer's)."""
    maxima: list[float] = []
    x = calibration
    with torch.no_grad():
        for layer in model:
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                maxima.append(float(torch.quantile(x.flatten().float(), percentile)))
        maxima.append(float(torch.quantile(x.flatten().float(), percentile)))
    return [m if m > 0 else 1.0 for m in maxima]


def convert_sequential(model, calibration, percentile: float = 0.999) -> ConversionResult:
    """Convert a trained `torch.nn.Sequential` (Flatten/Linear/ReLU) into a NIR graph.

    The graph uses NIR `Affine` and `IF` (integrate-and-fire, r=1, threshold=1,
    reset to 0) nodes, with one discrete timestep per simulator step
    (`metadata["dt"] = 1`). `calibration` is a batch of representative inputs.
    """
    import nir
    import numpy as np
    import torch

    allowed = (torch.nn.Flatten, torch.nn.Linear, torch.nn.ReLU)
    for layer in model:
        if not isinstance(layer, allowed):
            raise UnsupportedLayerError(f"unsupported layer: {type(layer).__name__}")
    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    if not linears:
        raise UnsupportedLayerError("model has no Linear layers")

    model = model.eval()
    maxima = _activation_maxima(model, calibration, torch, percentile)
    if len(maxima) != len(linears):
        raise UnsupportedLayerError("expected one ReLU after every Linear except the last")

    in_shape = np.array(calibration.shape[1:])
    nodes: dict[str, Any] = {"input": nir.Input(input_type={"input": in_shape})}
    order = ["input"]
    prev_scale = 1.0
    li = 0
    for layer in model:
        if isinstance(layer, torch.nn.Flatten):
            nodes["flatten"] = nir.Flatten(
                input_type={"input": in_shape}, start_dim=0, end_dim=-1
            )
            order.append("flatten")
        elif isinstance(layer, torch.nn.Linear):
            cur = maxima[li]
            w = layer.weight.detach().cpu().numpy() * prev_scale / cur
            b = (
                layer.bias.detach().cpu().numpy() / cur
                if layer.bias is not None
                else np.zeros(layer.out_features, dtype=np.float32)
            )
            n = layer.out_features
            nodes[f"fc{li}"] = nir.Affine(weight=w.astype(np.float32), bias=b.astype(np.float32))
            nodes[f"if{li}"] = nir.IF(
                r=np.ones(n, dtype=np.float32),
                v_threshold=np.ones(n, dtype=np.float32),
                v_reset=np.zeros(n, dtype=np.float32),
            )
            order += [f"fc{li}", f"if{li}"]
            prev_scale = cur
            li += 1
    nodes["output"] = nir.Output(output_type={"output": np.array([linears[-1].out_features])})
    order.append("output")
    edges = list(itertools.pairwise(order))
    graph = nir.NIRGraph(nodes=nodes, edges=edges, metadata={"dt": 1.0, "source": "pygri"})
    return ConversionResult(graph=graph, scales=maxima)
