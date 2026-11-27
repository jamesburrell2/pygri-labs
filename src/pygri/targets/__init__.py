"""Target registry. Adapters register themselves here."""

from __future__ import annotations

from pygri.targets.base import Target, TargetInfo, TargetKind, Workload

_REGISTRY: dict[str, type[Target]] = {}


def register(cls: type[Target]) -> type[Target]:
    """Class decorator that adds a Target to the registry."""
    name = cls.info.name
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"target '{name}' is already registered")
    _REGISTRY[name] = cls
    return cls


def get(name: str) -> Target:
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown target '{name}' (known: {known})") from exc


def all_targets() -> list[type[Target]]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


__all__ = ["Target", "TargetInfo", "TargetKind", "Workload", "all_targets", "get", "register"]
