"""Tests for the Lava target. Skipped unless lava-nc (Python 3.10 only) is installed.

They use NumPy-built NIR graphs, so torch is not needed.
"""

import pytest

np = pytest.importorskip("numpy")
nir = pytest.importorskip("nir")
pytest.importorskip("lava.proc.lif.process")

import itertools

from pygri import targets
from pygri.results import Provenance
from pygri.targets.base import Workload
from pygri.targets.lava_sim import parse_layers, simulate


def _graph(rng, sizes=(6, 12, 8, 3)):
    nodes = {"input": nir.Input(input_type={"input": np.array([sizes[0]])})}
    order = ["input"]
    for i, (a, b) in enumerate(itertools.pairwise(sizes)):
        nodes[f"fc{i}"] = nir.Affine(
            weight=rng.normal(scale=0.6, size=(b, a)), bias=rng.normal(scale=0.1, size=b)
        )
        nodes[f"if{i}"] = nir.IF(r=np.ones(b), v_threshold=np.ones(b), v_reset=np.zeros(b))
        order += [f"fc{i}", f"if{i}"]
    nodes["output"] = nir.Output(output_type={"output": np.array([sizes[-1]])})
    order.append("output")
    return nir.NIRGraph(nodes=nodes, edges=list(itertools.pairwise(order)))


def _reference(layers, x, timesteps):
    """Independent NumPy integrate-and-fire simulation (reset to zero)."""
    counts = np.zeros((len(x), layers[-1].weight.shape[0]))
    total = 0.0
    for i, sample in enumerate(x):
        v = [np.zeros(layer.weight.shape[0]) for layer in layers]
        for _ in range(timesteps):
            h = sample
            for j, layer in enumerate(layers):
                v[j] += layer.weight @ h + layer.bias
                h = (v[j] > layer.threshold).astype(float)
                v[j][h > 0] = 0.0
                total += h.sum()
            counts[i] += h
    return counts, total


def test_lava_matches_independent_reference():
    rng = np.random.default_rng(1)
    layers = parse_layers(_graph(rng))
    x = rng.uniform(0, 1, size=(5, 6))
    counts, total = simulate(layers, x, timesteps=15)
    ref_counts, ref_total = _reference(layers, x, timesteps=15)
    assert ref_counts.sum() > 0
    np.testing.assert_array_equal(counts, ref_counts)
    assert total == ref_total


def test_lava_target_reports_simulation_provenance():
    rng = np.random.default_rng(2)
    graph = _graph(rng)
    x = rng.uniform(0, 1, size=(4, 6))
    result = targets.get("lava").run(
        Workload(name="t", graph=graph, samples=x, labels=[0, 1, 2, 0], timesteps=10)
    )
    assert result.ok, result.error
    for metric in result.metrics.values():
        assert metric.provenance is Provenance.MEASURED_SIMULATION


def test_lava_unsupported_graph_returns_not_ok():
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "lif": nir.LIF(tau=np.ones(2), r=np.ones(2), v_leak=np.zeros(2), v_threshold=np.ones(2)),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "lif"), ("lif", "output")],
    )
    result = targets.get("lava").run(
        Workload(name="t", graph=graph, samples=np.zeros((1, 2)), labels=[0], timesteps=2)
    )
    assert not result.ok and "LIF" in result.error
