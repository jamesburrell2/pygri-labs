"""End-to-end demo on synthetic data: train, convert, benchmark, report.

The dataset is synthetic (Gaussian clusters), so results only show the pipeline
works; they say nothing about real workloads.
"""

from __future__ import annotations

from pygri.results import BenchmarkReport, Metric, Provenance, TargetResult


def make_dataset(n_per_class: int = 200, classes: int = 3, dims: int = 8, seed: int = 0):
    import torch

    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(classes, dims, generator=g) * 1.0
    x = torch.cat([centers[c] + torch.randn(n_per_class, dims, generator=g) for c in range(classes)])
    y = torch.arange(classes).repeat_interleave(n_per_class)
    perm = torch.randperm(len(y), generator=g)
    return x[perm], y[perm]


def train_mlp(x, y, hidden: int = 32, epochs: int = 300, seed: int = 0):
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, int(y.max()) + 1)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    return model.eval()


def run_demo(timesteps: int = 50, target_names: list[str] | None = None) -> BenchmarkReport:
    import torch

    from pygri import targets
    from pygri.convert import convert_sequential
    from pygri.targets.base import Workload

    x, y = make_dataset()
    split = int(0.7 * len(y))
    model = train_mlp(x[:split], y[:split])
    x_test, y_test = x[split:], y[split:]

    with torch.no_grad():
        ann_acc = float((model(x_test).argmax(1) == y_test).float().mean())
    reference = TargetResult(
        target="pytorch (original ANN)",
        model="synthetic-mlp",
        ok=True,
        metrics={
            TargetResult.ACCURACY: Metric(
                ann_acc, "fraction", Provenance.MEASURED_SIMULATION, note="non-spiking reference"
            )
        },
    )

    graph = convert_sequential(model, x[:split]).graph
    workload = Workload(
        name="synthetic-mlp", graph=graph, samples=x_test, labels=y_test, timesteps=timesteps
    )
    names = target_names or [cls.info.name for cls in targets.all_targets()]
    results = [reference] + [targets.get(n).run(workload) for n in names]
    return BenchmarkReport(model="synthetic-mlp", results=results)
