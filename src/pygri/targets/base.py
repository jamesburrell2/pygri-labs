"""The interface every target (simulator or chip) implements.

Agents writing new adapters implement `Target` and register it with
`pygri.targets.register`. The rest of Pygri never imports an adapter directly,
so new chips plug in without changes elsewhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pygri.results import TargetResult


class TargetKind(str, Enum):
    SIMULATOR = "simulator"
    HARDWARE = "hardware"
    CLOUD_HARDWARE = "cloud_hardware"


@dataclass(frozen=True)
class TargetInfo:
    """Static description of a target, shown in `pygri targets`."""

    name: str
    kind: TargetKind
    vendor: str
    description: str
    requires: tuple[str, ...] = ()  # Python packages or credentials it needs


@dataclass
class Workload:
    """What to run: a network in NIR form plus evaluation data.

    `graph` is a `nir.NIRGraph` once the converter exists. It is typed loosely
    here so the core package does not depend on `nir` at import time.
    """

    name: str
    graph: Any
    samples: Iterable[Any]
    labels: Iterable[Any]
    timesteps: int


class Target(ABC):
    """A place a spiking network can run."""

    info: TargetInfo

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """Return (True, "") if usable now, else (False, reason)."""

    @abstractmethod
    def run(self, workload: Workload) -> TargetResult:
        """Run the workload and return metrics with provenance.

        Must not raise for expected failures (missing credentials, unsupported
        layers). Return `TargetResult(ok=False, error=...)` instead.
        """
