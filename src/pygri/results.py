"""Result schema shared by every target adapter and the report generator.

Every number Pygri reports carries its provenance, so a reader can always tell
whether a figure was measured on real hardware, measured in a simulator, or
estimated from a model. This is the project's core honesty rule.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

SCHEMA_VERSION = "0.1"


class Provenance(str, Enum):
    """Where a number came from."""

    MEASURED_HARDWARE = "measured_hardware"
    MEASURED_SIMULATION = "measured_simulation"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class Metric:
    """One reported number, with its unit and provenance."""

    value: float
    unit: str
    provenance: Provenance
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, Provenance):
            raise TypeError("Metric.provenance must be a Provenance value")
        if not self.unit:
            raise ValueError("Metric.unit must not be empty")


@dataclass
class TargetResult:
    """The outcome of running one model on one target."""

    target: str
    model: str
    ok: bool
    metrics: dict[str, Metric] = field(default_factory=dict)
    error: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    # Standard metric keys. Adapters may add others.
    ACCURACY = "accuracy"
    LATENCY = "latency"
    ENERGY_PER_INFERENCE = "energy_per_inference"


@dataclass
class BenchmarkReport:
    """All target results for one model, ready to render as JSON or HTML."""

    model: str
    results: list[TargetResult]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for result in data["results"]:
            for metric in result["metrics"].values():
                metric["provenance"] = Provenance(metric["provenance"]).value
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
