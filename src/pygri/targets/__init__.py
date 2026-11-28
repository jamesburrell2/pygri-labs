"""Target registry. Adapters register themselves here."""

from __future__ import annotations

from pygri.targets.base import Target, TargetInfo, TargetKind, Workload

_REGISTRY: dict[str, type[Target]] = {}

# Built-in adapters, imported on first use. Their heavy dependencies are imported
# inside methods, so loading them here is cheap.
_BUILTIN_MODULES = ("pygri.targets.snntorch_sim", "pygri.targets.lava_sim")
_builtins_loaded = False


def _load_builtins() -> None:
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    import importlib

    for module in _BUILTIN_MODULES:
        importlib.import_module(module)


def register(cls: type[Target]) -> type[Target]:
    """Class decorator that adds a Target to the registry."""
    name = cls.info.name
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"target '{name}' is already registered")
    _REGISTRY[name] = cls
    return cls


def get(name: str) -> Target:
    _load_builtins()
    try:
        return _REGISTRY[name]()
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"unknown target '{name}' (known: {known})") from exc


def all_targets() -> list[type[Target]]:
    _load_builtins()
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


__all__ = ["Target", "TargetInfo", "TargetKind", "Workload", "all_targets", "get", "register"]
