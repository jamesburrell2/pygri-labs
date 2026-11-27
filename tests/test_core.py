import json

import pytest

from pygri import cli, targets
from pygri.results import BenchmarkReport, Metric, Provenance, TargetResult
from pygri.targets import Target, TargetInfo, TargetKind, Workload


def test_metric_requires_provenance_enum():
    with pytest.raises(TypeError):
        Metric(value=1.0, unit="ms", provenance="estimated")  # type: ignore[arg-type]


def test_metric_requires_unit():
    with pytest.raises(ValueError):
        Metric(value=1.0, unit="", provenance=Provenance.ESTIMATED)


def test_report_json_round_trip_keeps_provenance():
    result = TargetResult(
        target="fake",
        model="kws",
        ok=True,
        metrics={
            TargetResult.ACCURACY: Metric(0.91, "fraction", Provenance.MEASURED_SIMULATION),
            TargetResult.ENERGY_PER_INFERENCE: Metric(2.5, "uJ", Provenance.ESTIMATED),
        },
    )
    data = json.loads(BenchmarkReport(model="kws", results=[result]).to_json())
    metrics = data["results"][0]["metrics"]
    assert metrics["accuracy"]["provenance"] == "measured_simulation"
    assert metrics["energy_per_inference"]["provenance"] == "estimated"
    assert data["schema_version"] == "0.1"


class _FakeTarget(Target):
    info = TargetInfo(
        name="fake-sim",
        kind=TargetKind.SIMULATOR,
        vendor="test",
        description="A target used only in tests.",
    )

    def available(self):
        return True, ""

    def run(self, workload: Workload) -> TargetResult:
        return TargetResult(target=self.info.name, model=workload.name, ok=True)


def test_registry_and_cli(capsys):
    targets.register(_FakeTarget)
    assert isinstance(targets.get("fake-sim"), _FakeTarget)
    with pytest.raises(KeyError):
        targets.get("does-not-exist")
    assert cli.main(["targets"]) == 0
    assert "fake-sim" in capsys.readouterr().out


def test_registering_a_different_class_under_same_name_fails():
    class Other(_FakeTarget):
        pass

    targets.register(_FakeTarget)
    with pytest.raises(ValueError):
        targets.register(Other)
