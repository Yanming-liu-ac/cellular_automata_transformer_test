"""Propagation utilities for CA lattice experiments.

The first architectural question is simple: can a CA-style lattice move
information across long contexts faster than a plain 1D local rule while keeping
only local physical links? This module models the logical graph and counts the
minimum number of update ticks needed for information to travel between cells.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil, log2
from typing import Dict, Iterable, List, Tuple

import numpy as np

Node = Tuple[int, int]
EdgeMap = Dict[Node, List[Node]]


@dataclass(frozen=True)
class PropagationResult:
    """Shortest propagation result on a CA lattice graph."""

    source: Node
    target: Node
    steps: int | None
    visited_nodes: int


@dataclass(frozen=True)
class DynamicPropagationPoint:
    """Low-bit rollout result for signal propagation on a CA graph."""

    topology: str
    rule: str
    length: int
    bits: int
    ticks: int
    graph_nodes: int
    graph_edges: int
    source: Node
    target: Node
    threshold: int
    target_reach_tick: int | None
    all_token_reach_tick: int | None
    final_token_reach_fraction: float
    final_active_fraction: float
    final_saturation_fraction: float
    peak_saturation_fraction: float
    final_mean_level: float
    peak_mean_level: float
    checksum: int


@dataclass(frozen=True)
class DynamicPropagationResult:
    """Sweep of low-bit dynamic propagation and stability proxies."""

    lengths: Tuple[int, ...]
    topologies: Tuple[str, ...]
    rules: Tuple[str, ...]
    bits: int
    ticks: int
    points: Tuple[DynamicPropagationPoint, ...]


def _connect(edges: EdgeMap, a: Node, b: Node) -> None:
    edges.setdefault(a, []).append(b)
    edges.setdefault(b, []).append(a)


def line_edges(length: int, radius: int = 1) -> EdgeMap:
    """Return local 1D CA edges for level-0 token cells."""

    if length <= 0:
        raise ValueError("length must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    edges: EdgeMap = {(0, i): [] for i in range(length)}
    for i in range(length):
        for delta in range(1, radius + 1):
            j = i + delta
            if j < length:
                _connect(edges, (0, i), (0, j))
    return edges


def harc_ca_edges(length: int, radius: int = 1) -> EdgeMap:
    """Return logical HARC-CA multiscale edges.

    Nodes are `(level, index)`. Level 0 contains token cells. Each higher level
    contains block-summary cells with fan-in 2. Same-level local edges and
    parent-child edges are both local in a folded hardware layout.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    levels = ceil(log2(length)) + 1
    edges: EdgeMap = {}
    for level in range(levels):
        width = ceil(length / (2**level))
        for index in range(width):
            edges.setdefault((level, index), [])

            for delta in range(1, radius + 1):
                j = index + delta
                if j < width:
                    _connect(edges, (level, index), (level, j))

            if level + 1 < levels:
                parent = (level + 1, index // 2)
                _connect(edges, (level, index), parent)

    return edges


def shortest_propagation_steps(edges: EdgeMap, source: Node, target: Node) -> PropagationResult:
    """Breadth-first shortest path length in update ticks."""

    if source not in edges:
        raise ValueError(f"source node does not exist: {source}")
    if target not in edges:
        raise ValueError(f"target node does not exist: {target}")

    frontier = deque([(source, 0)])
    visited = {source}

    while frontier:
        node, steps = frontier.popleft()
        if node == target:
            return PropagationResult(source, target, steps, len(visited))
        for neighbor in edges.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, steps + 1))

    return PropagationResult(source, target, None, len(visited))


def propagation_distances(edges: EdgeMap, source: Node) -> Dict[Node, int]:
    """Shortest propagation distance from one source to all reachable nodes."""

    if source not in edges:
        raise ValueError(f"source node does not exist: {source}")

    frontier = deque([(source, 0)])
    distances = {source: 0}

    while frontier:
        node, steps = frontier.popleft()
        for neighbor in edges.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = steps + 1
                frontier.append((neighbor, steps + 1))

    return distances


def edge_count(edges: EdgeMap) -> int:
    """Count undirected edges."""

    return sum(len(v) for v in edges.values()) // 2


def max_distance_from_tail(edges: EdgeMap, length: int) -> int:
    """Maximum shortest-path distance from the final token to all token cells."""

    source = (0, length - 1)
    distances = propagation_distances(edges, source)
    max_distance = 0
    for i in range(length):
        distance = distances.get((0, i))
        if distance is None:
            raise RuntimeError(f"unreachable token cell: {i}")
        max_distance = max(max_distance, distance)
    return max_distance


def summarize_lengths(lengths: Iterable[int], radius: int = 1) -> List[dict[str, int]]:
    """Compare plain local CA and HARC-CA propagation over several lengths."""

    rows: List[dict[str, int]] = []
    for length in lengths:
        line = line_edges(length, radius)
        harc = harc_ca_edges(length, radius)
        rows.append(
            {
                "length": length,
                "line_steps": max_distance_from_tail(line, length),
                "harc_steps": max_distance_from_tail(harc, length),
                "line_nodes": len(line),
                "harc_nodes": len(harc),
                "line_edges": edge_count(line),
                "harc_edges": edge_count(harc),
            }
        )
    return rows


def _ordered_nodes(edges: EdgeMap) -> List[Node]:
    return sorted(edges)


def _neighbor_indices(edges: EdgeMap, nodes: List[Node]) -> List[np.ndarray]:
    node_index = {node: index for index, node in enumerate(nodes)}
    return [
        np.array([node_index[neighbor] for neighbor in edges[node]], dtype=np.int32)
        for node in nodes
    ]


def _graph_for_topology(topology: str, length: int, radius: int) -> EdgeMap:
    topology = topology.lower()
    if topology == "line":
        return line_edges(length, radius)
    if topology == "harc":
        return harc_ca_edges(length, radius)
    raise ValueError("topology must be one of: line, harc")


def _aggregate_neighbor_mean(
    values: np.ndarray,
    neighbors: List[np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    out = np.empty_like(fallback)
    for index, neighbor in enumerate(neighbors):
        if len(neighbor) == 0:
            out[index] = fallback[index]
        else:
            out[index] = int(np.mean(values[neighbor]))
    return out


def _aggregate_neighbor_max(
    values: np.ndarray,
    neighbors: List[np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    out = np.empty_like(fallback)
    for index, neighbor in enumerate(neighbors):
        if len(neighbor) == 0:
            out[index] = fallback[index]
        else:
            out[index] = int(np.max(values[neighbor]))
    return out


def _step_dynamic_state(
    state: np.ndarray,
    rule: str,
    neighbors: List[np.ndarray],
    max_value: int,
    source_index: int,
) -> np.ndarray:
    rule = rule.lower()
    if rule == "residual_avg":
        current = state[:, 0].astype(np.int16)
        neighbor_mean = _aggregate_neighbor_mean(current, neighbors, current)
        delta = np.sign(neighbor_mean - current).astype(np.int16)
        next_state = np.clip(current + delta, 0, max_value).astype(np.uint8)[:, None]
        next_state[source_index, 0] = max_value
        return next_state

    if rule == "route_max":
        current = state[:, 0].astype(np.int16)
        neighbor_max = _aggregate_neighbor_max(current, neighbors, current)
        delta = np.sign(neighbor_max - current).astype(np.int16)
        next_state = np.clip(current + delta, 0, max_value).astype(np.uint8)[:, None]
        next_state[source_index, 0] = max_value
        return next_state

    if rule == "mhc_grouped":
        local = state[:, 0].astype(np.int16)
        route = state[:, 1].astype(np.int16)
        envelope = state[:, 2].astype(np.int16)
        local_mean = _aggregate_neighbor_mean(local, neighbors, local)
        route_max = _aggregate_neighbor_max(route, neighbors, route)

        route_target = np.maximum(route_max, local_mean)
        route_delta = np.clip(route_target - route, -1, 2)
        next_route = np.clip(route + route_delta, 0, max_value)

        local_target = (local_mean + next_route) // 2
        next_local = np.clip(local + np.sign(local_target - local), 0, max_value)

        envelope_target = np.maximum(next_local, next_route) // 2
        next_envelope = np.clip(envelope + np.sign(envelope_target - envelope), 0, max_value)

        budget = max_value * 2
        total = next_local + next_route + next_envelope
        overflow = np.maximum(total - budget, 0)
        next_envelope = np.maximum(0, next_envelope - overflow)
        total = next_local + next_route + next_envelope
        overflow = np.maximum(total - budget, 0)
        next_route = np.maximum(0, next_route - overflow)

        next_state = np.stack([next_local, next_route, next_envelope], axis=1).astype(np.uint8)
        next_state[source_index] = np.array([max_value, max_value, max_value // 2], dtype=np.uint8)
        return next_state

    raise ValueError("rule must be one of: residual_avg, route_max, mhc_grouped")


def _signal_level(state: np.ndarray, rule: str) -> np.ndarray:
    if rule == "mhc_grouped":
        return np.maximum(state[:, 0], state[:, 1])
    return state[:, 0]


def evaluate_dynamic_propagation(
    length: int,
    topology: str,
    rule: str,
    bits: int = 4,
    ticks: int = 128,
    radius: int = 1,
) -> DynamicPropagationPoint:
    """Roll out a low-bit signal and measure reach latency and state stability.

    This is a dynamic complement to shortest-path propagation. It keeps the
    update local and low-bit, so a short graph path only counts if the signal
    actually survives the integer rollout without saturating the whole state.
    """

    if length <= 0:
        raise ValueError("length must be positive")
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, 8")
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    rule = rule.lower()
    if rule not in {"residual_avg", "route_max", "mhc_grouped"}:
        raise ValueError("rule must be one of: residual_avg, route_max, mhc_grouped")

    edges = _graph_for_topology(topology, length, radius)
    nodes = _ordered_nodes(edges)
    node_index = {node: index for index, node in enumerate(nodes)}
    neighbors = _neighbor_indices(edges, nodes)
    source = (0, length - 1)
    target = (0, 0)
    source_index = node_index[source]
    target_index = node_index[target]
    token_indices = np.array([node_index[(0, index)] for index in range(length)], dtype=np.int32)
    max_value = (1 << bits) - 1
    threshold = max(1, max_value // 2)
    channels = 3 if rule == "mhc_grouped" else 1
    state = np.zeros((len(nodes), channels), dtype=np.uint8)
    if rule == "mhc_grouped":
        state[source_index] = np.array([max_value, max_value, max_value // 2], dtype=np.uint8)
    else:
        state[source_index, 0] = max_value

    target_reach_tick: int | None = 0 if _signal_level(state, rule)[target_index] >= threshold else None
    all_token_reach_tick: int | None = None
    peak_saturation_fraction = float(np.count_nonzero(state == max_value)) / float(state.size)
    peak_mean_level = float(np.mean(state)) / float(max_value)

    for tick in range(1, ticks + 1):
        state = _step_dynamic_state(
            state=state,
            rule=rule,
            neighbors=neighbors,
            max_value=max_value,
            source_index=source_index,
        )
        signal = _signal_level(state, rule)
        token_signal = signal[token_indices]
        if target_reach_tick is None and int(signal[target_index]) >= threshold:
            target_reach_tick = tick
        if all_token_reach_tick is None and bool(np.all(token_signal >= threshold)):
            all_token_reach_tick = tick
        saturation_fraction = float(np.count_nonzero(state == max_value)) / float(state.size)
        mean_level = float(np.mean(state)) / float(max_value)
        peak_saturation_fraction = max(peak_saturation_fraction, saturation_fraction)
        peak_mean_level = max(peak_mean_level, mean_level)

    signal = _signal_level(state, rule)
    token_signal = signal[token_indices]
    checksum_weights = np.arange(1, channels + 1, dtype=np.uint64)
    checksum = int(np.sum(state.astype(np.uint64) * checksum_weights) % (2**63 - 1))

    return DynamicPropagationPoint(
        topology=topology.lower(),
        rule=rule,
        length=length,
        bits=bits,
        ticks=ticks,
        graph_nodes=len(nodes),
        graph_edges=edge_count(edges),
        source=source,
        target=target,
        threshold=threshold,
        target_reach_tick=target_reach_tick,
        all_token_reach_tick=all_token_reach_tick,
        final_token_reach_fraction=float(np.count_nonzero(token_signal >= threshold)) / float(length),
        final_active_fraction=float(np.count_nonzero(state)) / float(state.size),
        final_saturation_fraction=float(np.count_nonzero(state == max_value)) / float(state.size),
        peak_saturation_fraction=peak_saturation_fraction,
        final_mean_level=float(np.mean(state)) / float(max_value),
        peak_mean_level=peak_mean_level,
        checksum=checksum,
    )


def run_dynamic_propagation_sweep(
    lengths: Tuple[int, ...] = (128, 512, 2048),
    topologies: Tuple[str, ...] = ("line", "harc"),
    rules: Tuple[str, ...] = ("residual_avg", "route_max", "mhc_grouped"),
    bits: int = 4,
    ticks: int = 128,
    radius: int = 1,
) -> DynamicPropagationResult:
    """Compare low-bit rollout propagation rules over line and HARC graphs."""

    if len(lengths) == 0:
        raise ValueError("lengths must not be empty")
    clean_lengths = tuple(int(length) for length in lengths)
    if any(length <= 0 for length in clean_lengths):
        raise ValueError("lengths must be positive")
    if len(topologies) == 0:
        raise ValueError("topologies must not be empty")
    clean_topologies = tuple(dict.fromkeys(str(topology).lower() for topology in topologies))
    if len(rules) == 0:
        raise ValueError("rules must not be empty")
    clean_rules = tuple(dict.fromkeys(str(rule).lower() for rule in rules))

    points = []
    for length in clean_lengths:
        for topology in clean_topologies:
            for rule in clean_rules:
                points.append(
                    evaluate_dynamic_propagation(
                        length=length,
                        topology=topology,
                        rule=rule,
                        bits=bits,
                        ticks=ticks,
                        radius=radius,
                    )
                )

    return DynamicPropagationResult(
        lengths=clean_lengths,
        topologies=clean_topologies,
        rules=clean_rules,
        bits=bits,
        ticks=ticks,
        points=tuple(points),
    )
