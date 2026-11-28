"""Helpers shared by adapters that accept single-chain NIR graphs."""

from __future__ import annotations


class UnsupportedGraphError(ValueError):
    pass


def chain(graph) -> list[str]:
    """Return the node keys of a single-chain graph, from Input to Output."""
    nexts = dict(graph.edges)
    if len(nexts) != len(graph.edges):
        raise UnsupportedGraphError("only single-chain graphs are supported")
    inputs = [k for k, n in graph.nodes.items() if type(n).__name__ == "Input"]
    if len(inputs) != 1:
        raise UnsupportedGraphError("graph must have exactly one Input node")
    order, cur = [], inputs[0]
    while cur is not None:
        order.append(cur)
        cur = nexts.get(cur)
    if len(order) != len(graph.nodes):
        raise UnsupportedGraphError("graph is not a single chain")
    return order


def if_threshold(key: str, node) -> float:
    """Uniform firing threshold of an IF node; only reset-to-zero is supported."""
    import numpy as np

    thr = np.unique(np.asarray(node.v_threshold) * np.asarray(node.r))
    reset = np.zeros(1) if node.v_reset is None else np.unique(np.asarray(node.v_reset))
    if thr.size != 1 or reset.size != 1 or float(reset[0]) != 0.0:
        raise UnsupportedGraphError(
            f"node '{key}': only uniform thresholds and reset-to-zero are supported"
        )
    return float(thr[0])
