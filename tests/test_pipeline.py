"""End-to-end tests for the converter and the snnTorch target.

Skipped when the optional simulator dependencies are not installed.
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("snntorch")
pytest.importorskip("nir")

from pygri import targets
from pygri.convert import convert_sequential
from pygri.convert.ann_to_snn import UnsupportedLayerError
from pygri.demo import make_dataset, run_demo, train_mlp
from pygri.report import render_html
from pygri.results import Provenance
from pygri.targets.base import Workload
from pygri.targets.snntorch_sim import build_network, simulate


def _reference_if_counts(graph, x: np.ndarray, timesteps: int) -> np.ndarray:
    """Independent NumPy simulation of the NIR graph (IF neurons, reset to zero)."""
    layers = []
    key = "input"
    nexts = dict(graph.edges)
    while key in nexts:
        key = nexts[key]
        node = graph.nodes[key]
        name = type(node).__name__
        if name == "Affine":
            layers.append(("affine", np.asarray(node.weight), np.asarray(node.bias)))
        elif name == "IF":
            layers.append(("if", np.asarray(node.v_threshold) * np.asarray(node.r)))
    v = [None] * len(layers)
    counts = None
    for _ in range(timesteps):
        h = x
        for i, layer in enumerate(layers):
            if layer[0] == "affine":
                h = h @ layer[1].T + layer[2]
            else:
                v[i] = h if v[i] is None else v[i] + h
                spikes = (v[i] > layer[1]).astype(np.float32)
                v[i] = v[i] * (1 - spikes)
                h = spikes
        counts = h if counts is None else counts + h
    return counts


def _trained():
    x, y = make_dataset(n_per_class=60)
    return train_mlp(x, y, epochs=150), x, y


def test_snntorch_matches_independent_reference():
    model, x, _ = _trained()
    graph = convert_sequential(model, x).graph
    counts, _ = simulate(build_network(graph), x[:40], timesteps=30)
    ref = _reference_if_counts(graph, x[:40].numpy(), timesteps=30)
    mismatch = np.abs(counts.numpy() - ref).sum()
    # Threshold comparisons can differ on float ties; allow a tiny number of spikes.
    assert mismatch <= 0.005 * ref.sum() + 1


def test_conversion_preserves_accuracy_and_labels_provenance():
    model, x, y = _trained()
    graph = convert_sequential(model, x).graph
    result = targets.get("snntorch").run(
        Workload(name="t", graph=graph, samples=x, labels=y, timesteps=50)
    )
    assert result.ok, result.error
    with torch.no_grad():
        ann_acc = float((model(x).argmax(1) == y).float().mean())
    assert result.metrics["accuracy"].value >= ann_acc - 0.05
    for metric in result.metrics.values():
        assert metric.provenance is Provenance.MEASURED_SIMULATION


def test_unsupported_layer_is_reported():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Sigmoid(), torch.nn.Linear(4, 2))
    with pytest.raises(UnsupportedLayerError):
        convert_sequential(model, torch.randn(8, 4))


def test_unsupported_graph_returns_not_ok():
    import nir

    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "lif": nir.LIF(
                tau=np.ones(2), r=np.ones(2), v_leak=np.zeros(2), v_threshold=np.ones(2)
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "lif"), ("lif", "output")],
    )
    result = targets.get("snntorch").run(
        Workload(name="t", graph=graph, samples=torch.zeros(1, 2), labels=[0], timesteps=2)
    )
    assert not result.ok and "LIF" in result.error


def test_demo_report_renders():
    report = run_demo(timesteps=20)
    html = render_html(report)
    assert "measured in simulation" in html
    assert report.results[1].ok
