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


@dataclass(frozen=True)
class LongRolloutStabilityPoint:
    """Long low-bit rollout stability result from an unforced initial state."""

    topology: str
    rule: str
    init_mode: str
    length: int
    bits: int
    ticks: int
    seed: int
    graph_nodes: int
    graph_edges: int
    initial_active_fraction: float
    final_active_fraction: float
    initial_saturation_fraction: float
    final_saturation_fraction: float
    peak_saturation_fraction: float
    initial_mean_level: float
    final_mean_level: float
    peak_mean_level: float
    initial_entropy_bits: float
    final_entropy_bits: float
    min_entropy_bits: float
    mean_abs_step: float
    checksum: int
    collapsed: bool
    saturated: bool


@dataclass(frozen=True)
class LongRolloutStabilityResult:
    """Sweep of unforced long-rollout stability checks."""

    lengths: Tuple[int, ...]
    topologies: Tuple[str, ...]
    rules: Tuple[str, ...]
    init_modes: Tuple[str, ...]
    bits: int
    ticks: int
    points: Tuple[LongRolloutStabilityPoint, ...]


@dataclass(frozen=True)
class ContentRetentionPoint:
    """Content-retention result for a stable CA carrier plus optional latch plane."""

    topology: str
    policy: str
    length: int
    bits: int
    ticks: int
    seed: int
    graph_nodes: int
    graph_edges: int
    state_bits_per_token: int
    refresh_interval: int
    refresh_events: int
    refresh_channel_writes_per_token_tick: float
    initial_content_entropy_bits: float
    final_content_entropy_bits: float
    content_exact_retention_rate: float
    content_mean_abs_error: float
    carrier_exact_retention_rate: float
    mean_carrier_exact_retention_rate: float
    carrier_mean_abs_error: float
    mean_carrier_mean_abs_error: float
    carrier_final_entropy_bits: float
    carrier_final_saturation_fraction: float
    carrier_final_mean_level: float
    checksum: int


@dataclass(frozen=True)
class ContentRetentionResult:
    """Sweep of content retention policies on top of the mHC carrier."""

    lengths: Tuple[int, ...]
    topologies: Tuple[str, ...]
    policies: Tuple[str, ...]
    bits: int
    ticks: int
    points: Tuple[ContentRetentionPoint, ...]


@dataclass(frozen=True)
class ContentGatePoint:
    """Local content-to-carrier gate result."""

    topology: str
    policy: str
    length: int
    bits: int
    ticks: int
    seed: int
    graph_nodes: int
    graph_edges: int
    state_bits_per_token: int
    gate_token_writes: int
    gate_channel_writes_per_token_tick: float
    mean_gate_fraction: float
    content_exact_retention_rate: float
    carrier_exact_retention_rate: float
    mean_carrier_exact_retention_rate: float
    carrier_mean_abs_error: float
    mean_carrier_mean_abs_error: float
    carrier_final_entropy_bits: float
    carrier_final_saturation_fraction: float
    carrier_final_mean_level: float
    checksum: int


@dataclass(frozen=True)
class ContentGateResult:
    """Sweep of content-to-carrier gate policies."""

    lengths: Tuple[int, ...]
    topologies: Tuple[str, ...]
    policies: Tuple[str, ...]
    bits: int
    ticks: int
    points: Tuple[ContentGatePoint, ...]


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
    source_index: int | None,
) -> np.ndarray:
    rule = rule.lower()
    if rule == "residual_avg":
        current = state[:, 0].astype(np.int16)
        neighbor_mean = _aggregate_neighbor_mean(current, neighbors, current)
        delta = np.sign(neighbor_mean - current).astype(np.int16)
        next_state = np.clip(current + delta, 0, max_value).astype(np.uint8)[:, None]
        if source_index is not None:
            next_state[source_index, 0] = max_value
        return next_state

    if rule == "route_max":
        current = state[:, 0].astype(np.int16)
        neighbor_max = _aggregate_neighbor_max(current, neighbors, current)
        delta = np.sign(neighbor_max - current).astype(np.int16)
        next_state = np.clip(current + delta, 0, max_value).astype(np.uint8)[:, None]
        if source_index is not None:
            next_state[source_index, 0] = max_value
        return next_state

    if rule in {"mhc_grouped", "mhc_damped"}:
        local = state[:, 0].astype(np.int16)
        route = state[:, 1].astype(np.int16)
        envelope = state[:, 2].astype(np.int16)
        local_mean = _aggregate_neighbor_mean(local, neighbors, local)
        route_max = _aggregate_neighbor_max(route, neighbors, route)

        route_target = np.maximum(route_max, local_mean)
        if rule == "mhc_damped":
            route_delta = np.clip(route_target - route, 0, 2)
            next_route = np.clip(route + route_delta - 1, 0, max_value)
        else:
            route_delta = np.clip(route_target - route, -1, 2)
            next_route = np.clip(route + route_delta, 0, max_value)

        local_target = (local_mean + next_route) // 2
        next_local = np.clip(local + np.sign(local_target - local), 0, max_value)

        if rule == "mhc_damped":
            envelope_target = (next_local + next_route) // 3
        else:
            envelope_target = np.maximum(next_local, next_route) // 2
        next_envelope = np.clip(envelope + np.sign(envelope_target - envelope), 0, max_value)

        budget = max_value + max_value // 2 if rule == "mhc_damped" else max_value * 2
        total = next_local + next_route + next_envelope
        overflow = np.maximum(total - budget, 0)
        next_envelope = np.maximum(0, next_envelope - overflow)
        total = next_local + next_route + next_envelope
        overflow = np.maximum(total - budget, 0)
        next_route = np.maximum(0, next_route - overflow)

        next_state = np.stack([next_local, next_route, next_envelope], axis=1).astype(np.uint8)
        if source_index is not None:
            next_state[source_index] = np.array([max_value, max_value, max_value // 2], dtype=np.uint8)
        return next_state

    raise ValueError("rule must be one of: residual_avg, route_max, mhc_grouped, mhc_damped")


def _signal_level(state: np.ndarray, rule: str) -> np.ndarray:
    if rule in {"mhc_grouped", "mhc_damped"}:
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
    if rule not in {"residual_avg", "route_max", "mhc_grouped", "mhc_damped"}:
        raise ValueError("rule must be one of: residual_avg, route_max, mhc_grouped, mhc_damped")

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
    channels = 3 if rule in {"mhc_grouped", "mhc_damped"} else 1
    state = np.zeros((len(nodes), channels), dtype=np.uint8)
    if rule in {"mhc_grouped", "mhc_damped"}:
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


def _level_entropy_bits(state: np.ndarray, max_value: int) -> float:
    counts = np.bincount(state.reshape(-1).astype(np.int16), minlength=max_value + 1)
    total = int(np.sum(counts))
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0].astype(np.float64) / float(total)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _initialize_dynamic_state(
    init_mode: str,
    rule: str,
    nodes: List[Node],
    node_index: Dict[Node, int],
    length: int,
    max_value: int,
    rng: np.random.Generator,
) -> np.ndarray:
    init_mode = init_mode.lower()
    channels = 3 if rule in {"mhc_grouped", "mhc_damped"} else 1
    state = np.zeros((len(nodes), channels), dtype=np.uint8)

    if init_mode == "sparse_random":
        active = rng.random(state.shape) < 0.08
        values = rng.integers(1, max_value + 1, size=state.shape, dtype=np.uint8)
        state[:] = np.where(active, values, 0).astype(np.uint8)
        return state

    if init_mode == "dense_random":
        state[:] = rng.integers(0, max_value + 1, size=state.shape, dtype=np.uint8)
        return state

    if init_mode == "structured_pulses":
        pulse_positions = tuple(sorted({0, length // 3, (2 * length) // 3, length - 1}))
        for token_index, position in enumerate(pulse_positions):
            node = (0, position)
            if node not in node_index:
                continue
            row = node_index[node]
            if channels == 1:
                state[row, 0] = max_value
            else:
                local = max(1, max_value - token_index)
                route = max_value if token_index % 2 == 0 else max_value // 2
                envelope = max_value // 2
                state[row] = np.array([local, route, envelope], dtype=np.uint8)
        return state

    raise ValueError("init_mode must be one of: sparse_random, dense_random, structured_pulses")


def _state_checksum(state: np.ndarray) -> int:
    weights = np.arange(1, state.shape[1] + 1, dtype=np.uint64)
    return int(np.sum(state.astype(np.uint64) * weights) % (2**63 - 1))


def evaluate_long_rollout_stability(
    length: int,
    topology: str,
    rule: str,
    init_mode: str,
    bits: int = 4,
    ticks: int = 1000,
    radius: int = 1,
    seed: int = 0,
) -> LongRolloutStabilityPoint:
    """Run an unforced low-bit CA rollout and measure collapse/saturation risk."""

    if length <= 0:
        raise ValueError("length must be positive")
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, 8")
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    rule = rule.lower()
    if rule not in {"residual_avg", "route_max", "mhc_grouped", "mhc_damped"}:
        raise ValueError("rule must be one of: residual_avg, route_max, mhc_grouped, mhc_damped")

    edges = _graph_for_topology(topology, length, radius)
    nodes = _ordered_nodes(edges)
    node_index = {node: index for index, node in enumerate(nodes)}
    neighbors = _neighbor_indices(edges, nodes)
    max_value = (1 << bits) - 1
    rng = np.random.default_rng(seed)
    state = _initialize_dynamic_state(
        init_mode=init_mode,
        rule=rule,
        nodes=nodes,
        node_index=node_index,
        length=length,
        max_value=max_value,
        rng=rng,
    )

    initial_active_fraction = float(np.count_nonzero(state)) / float(state.size)
    initial_saturation_fraction = float(np.count_nonzero(state == max_value)) / float(state.size)
    initial_mean_level = float(np.mean(state)) / float(max_value)
    initial_entropy = _level_entropy_bits(state, max_value)
    peak_saturation_fraction = initial_saturation_fraction
    peak_mean_level = initial_mean_level
    min_entropy = initial_entropy
    total_abs_step = 0.0

    for _ in range(ticks):
        previous = state
        state = _step_dynamic_state(
            state=state,
            rule=rule,
            neighbors=neighbors,
            max_value=max_value,
            source_index=None,
        )
        total_abs_step += float(
            np.mean(np.abs(state.astype(np.int16) - previous.astype(np.int16))) / float(max_value)
        )
        saturation_fraction = float(np.count_nonzero(state == max_value)) / float(state.size)
        mean_level = float(np.mean(state)) / float(max_value)
        entropy = _level_entropy_bits(state, max_value)
        peak_saturation_fraction = max(peak_saturation_fraction, saturation_fraction)
        peak_mean_level = max(peak_mean_level, mean_level)
        min_entropy = min(min_entropy, entropy)

    final_active_fraction = float(np.count_nonzero(state)) / float(state.size)
    final_saturation_fraction = float(np.count_nonzero(state == max_value)) / float(state.size)
    final_mean_level = float(np.mean(state)) / float(max_value)
    final_entropy = _level_entropy_bits(state, max_value)
    collapsed = final_active_fraction < 0.01 or final_entropy < 0.10
    saturated = peak_saturation_fraction > 0.80

    return LongRolloutStabilityPoint(
        topology=topology.lower(),
        rule=rule,
        init_mode=init_mode.lower(),
        length=length,
        bits=bits,
        ticks=ticks,
        seed=seed,
        graph_nodes=len(nodes),
        graph_edges=edge_count(edges),
        initial_active_fraction=initial_active_fraction,
        final_active_fraction=final_active_fraction,
        initial_saturation_fraction=initial_saturation_fraction,
        final_saturation_fraction=final_saturation_fraction,
        peak_saturation_fraction=peak_saturation_fraction,
        initial_mean_level=initial_mean_level,
        final_mean_level=final_mean_level,
        peak_mean_level=peak_mean_level,
        initial_entropy_bits=initial_entropy,
        final_entropy_bits=final_entropy,
        min_entropy_bits=min_entropy,
        mean_abs_step=total_abs_step / float(ticks),
        checksum=_state_checksum(state),
        collapsed=collapsed,
        saturated=saturated,
    )


def run_long_rollout_stability_sweep(
    lengths: Tuple[int, ...] = (512,),
    topologies: Tuple[str, ...] = ("harc",),
    rules: Tuple[str, ...] = ("residual_avg", "route_max", "mhc_grouped", "mhc_damped"),
    init_modes: Tuple[str, ...] = ("sparse_random", "dense_random", "structured_pulses"),
    bits: int = 4,
    ticks: int = 1000,
    radius: int = 1,
    seed: int = 101,
) -> LongRolloutStabilityResult:
    """Sweep long unforced low-bit rollouts for stability diagnostics."""

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
    if len(init_modes) == 0:
        raise ValueError("init_modes must not be empty")
    clean_init_modes = tuple(dict.fromkeys(str(init_mode).lower() for init_mode in init_modes))

    points = []
    for length_index, length in enumerate(clean_lengths):
        for topology_index, topology in enumerate(clean_topologies):
            for rule_index, rule in enumerate(clean_rules):
                for init_index, init_mode in enumerate(clean_init_modes):
                    point_seed = seed + 1009 * length_index + 131 * topology_index + 17 * rule_index + init_index
                    points.append(
                        evaluate_long_rollout_stability(
                            length=length,
                            topology=topology,
                            rule=rule,
                            init_mode=init_mode,
                            bits=bits,
                            ticks=ticks,
                            radius=radius,
                            seed=point_seed,
                        )
                    )

    return LongRolloutStabilityResult(
        lengths=clean_lengths,
        topologies=clean_topologies,
        rules=clean_rules,
        init_modes=clean_init_modes,
        bits=bits,
        ticks=ticks,
        points=tuple(points),
    )


def _parse_content_policy(policy: str) -> Tuple[bool, int]:
    policy = str(policy).lower()
    if policy == "shared_mhc":
        return False, 0
    if policy == "content_latch":
        return True, 0
    prefix = "content_latch_refresh"
    if policy.startswith(prefix):
        suffix = policy[len(prefix) :]
        if not suffix:
            raise ValueError("content_latch_refresh policy must include an interval")
        interval = int(suffix)
        if interval <= 0:
            raise ValueError("refresh interval must be positive")
        return True, interval
    raise ValueError(
        "policy must be one of: shared_mhc, content_latch, content_latch_refresh<N>"
    )


def _inject_content_into_carrier(
    carrier: np.ndarray,
    content_values: np.ndarray,
    token_indices: np.ndarray,
) -> None:
    token_content = content_values[token_indices]
    carrier[token_indices, 0] = token_content
    carrier[token_indices, 1] = token_content
    carrier[token_indices, 2] = (token_content // 2).astype(np.uint8)


def evaluate_content_retention(
    length: int,
    topology: str,
    policy: str,
    bits: int = 4,
    ticks: int = 1000,
    radius: int = 1,
    seed: int = 0,
) -> ContentRetentionPoint:
    """Measure whether low-bit content survives on or beside the mHC carrier."""

    if length <= 0:
        raise ValueError("length must be positive")
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, 8")
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    policy = str(policy).lower()
    has_latch, refresh_interval = _parse_content_policy(policy)
    edges = _graph_for_topology(topology, length, radius)
    nodes = _ordered_nodes(edges)
    node_index = {node: index for index, node in enumerate(nodes)}
    neighbors = _neighbor_indices(edges, nodes)
    token_indices = np.array([node_index[(0, index)] for index in range(length)], dtype=np.int32)
    max_value = (1 << bits) - 1
    rng = np.random.default_rng(seed)

    content_values = np.zeros(len(nodes), dtype=np.uint8)
    content_values[token_indices] = rng.integers(
        0, max_value + 1, size=length, dtype=np.uint8
    )
    initial_token_content = content_values[token_indices].copy()
    carrier = np.zeros((len(nodes), 3), dtype=np.uint8)
    _inject_content_into_carrier(carrier, content_values, token_indices)

    refresh_events = 0
    carrier_exact_sum = 0.0
    carrier_error_sum = 0.0
    for tick in range(1, ticks + 1):
        carrier = _step_dynamic_state(
            state=carrier,
            rule="mhc_grouped",
            neighbors=neighbors,
            max_value=max_value,
            source_index=None,
        )
        if has_latch and refresh_interval > 0 and tick % refresh_interval == 0:
            _inject_content_into_carrier(carrier, content_values, token_indices)
            refresh_events += 1
        carrier_token_content = carrier[token_indices, 0]
        carrier_error = np.abs(
            carrier_token_content.astype(np.int16) - initial_token_content.astype(np.int16)
        )
        carrier_exact_sum += float(np.mean(carrier_token_content == initial_token_content))
        carrier_error_sum += float(np.mean(carrier_error) / float(max_value))

    carrier_token_content = carrier[token_indices, 0]
    if has_latch:
        final_token_content = content_values[token_indices]
    else:
        final_token_content = carrier_token_content

    content_error = np.abs(
        final_token_content.astype(np.int16) - initial_token_content.astype(np.int16)
    )
    carrier_error = np.abs(
        carrier_token_content.astype(np.int16) - initial_token_content.astype(np.int16)
    )
    content_channel_count = 1 if has_latch else 0
    state_bits_per_token = (3 + content_channel_count) * bits
    refresh_channel_writes_per_token_tick = (
        refresh_events * 3 / float(ticks)
        if has_latch and refresh_interval > 0
        else 0.0
    )

    checksum = (
        _state_checksum(carrier)
        + int(
            np.sum(
                content_values[token_indices].astype(np.uint64)
                * np.arange(1, length + 1, dtype=np.uint64)
            )
            % (2**63 - 1)
        )
    ) % (2**63 - 1)

    return ContentRetentionPoint(
        topology=topology.lower(),
        policy=policy,
        length=length,
        bits=bits,
        ticks=ticks,
        seed=seed,
        graph_nodes=len(nodes),
        graph_edges=edge_count(edges),
        state_bits_per_token=state_bits_per_token,
        refresh_interval=refresh_interval,
        refresh_events=refresh_events,
        refresh_channel_writes_per_token_tick=refresh_channel_writes_per_token_tick,
        initial_content_entropy_bits=_level_entropy_bits(initial_token_content, max_value),
        final_content_entropy_bits=_level_entropy_bits(final_token_content, max_value),
        content_exact_retention_rate=float(np.mean(final_token_content == initial_token_content)),
        content_mean_abs_error=float(np.mean(content_error) / float(max_value)),
        carrier_exact_retention_rate=float(np.mean(carrier_token_content == initial_token_content)),
        mean_carrier_exact_retention_rate=carrier_exact_sum / float(ticks),
        carrier_mean_abs_error=float(np.mean(carrier_error) / float(max_value)),
        mean_carrier_mean_abs_error=carrier_error_sum / float(ticks),
        carrier_final_entropy_bits=_level_entropy_bits(carrier, max_value),
        carrier_final_saturation_fraction=float(np.count_nonzero(carrier == max_value))
        / float(carrier.size),
        carrier_final_mean_level=float(np.mean(carrier)) / float(max_value),
        checksum=int(checksum),
    )


def run_content_retention_sweep(
    lengths: Tuple[int, ...] = (512,),
    topologies: Tuple[str, ...] = ("harc",),
    policies: Tuple[str, ...] = (
        "shared_mhc",
        "content_latch",
        "content_latch_refresh64",
        "content_latch_refresh16",
        "content_latch_refresh8",
    ),
    bits: int = 4,
    ticks: int = 1000,
    radius: int = 1,
    seed: int = 211,
) -> ContentRetentionResult:
    """Sweep content-latch policies over the stable mHC carrier."""

    if len(lengths) == 0:
        raise ValueError("lengths must not be empty")
    clean_lengths = tuple(int(length) for length in lengths)
    if any(length <= 0 for length in clean_lengths):
        raise ValueError("lengths must be positive")
    if len(topologies) == 0:
        raise ValueError("topologies must not be empty")
    clean_topologies = tuple(dict.fromkeys(str(topology).lower() for topology in topologies))
    if len(policies) == 0:
        raise ValueError("policies must not be empty")
    clean_policies = tuple(dict.fromkeys(str(policy).lower() for policy in policies))

    points = []
    for length_index, length in enumerate(clean_lengths):
        for topology_index, topology in enumerate(clean_topologies):
            for policy in clean_policies:
                point_seed = seed + 1009 * length_index + 131 * topology_index
                points.append(
                    evaluate_content_retention(
                        length=length,
                        topology=topology,
                        policy=policy,
                        bits=bits,
                        ticks=ticks,
                        radius=radius,
                        seed=point_seed,
                    )
                )

    return ContentRetentionResult(
        lengths=clean_lengths,
        topologies=clean_topologies,
        policies=clean_policies,
        bits=bits,
        ticks=ticks,
        points=tuple(points),
    )


def _content_gate_selection(
    policy: str,
    tick: int,
    carrier: np.ndarray,
    content_values: np.ndarray,
    token_indices: np.ndarray,
    max_value: int,
) -> np.ndarray:
    policy = str(policy).lower()
    if policy == "none":
        return np.empty(0, dtype=np.int32)

    if policy.startswith("fixed_refresh"):
        suffix = policy[len("fixed_refresh") :]
        if not suffix:
            raise ValueError("fixed_refresh policy must include an interval")
        interval = int(suffix)
        if interval <= 0:
            raise ValueError("fixed_refresh interval must be positive")
        if tick % interval == 0:
            return token_indices
        return np.empty(0, dtype=np.int32)

    carrier_values = carrier[token_indices, 0].astype(np.int16)
    content = content_values[token_indices].astype(np.int16)
    error = np.abs(carrier_values - content)

    if policy.startswith("mismatch_ge"):
        threshold = int(policy[len("mismatch_ge") :])
        if threshold <= 0:
            raise ValueError("mismatch threshold must be positive")
        return token_indices[error >= threshold]

    if policy.startswith("budget_top") and policy.endswith("pct"):
        percent = int(policy[len("budget_top") : -len("pct")])
        if not 0 < percent <= 100:
            raise ValueError("budget_top percent must be in (0, 100]")
        positive = np.flatnonzero(error > 0)
        if len(positive) == 0:
            return np.empty(0, dtype=np.int32)
        budget = max(1, int(round(len(token_indices) * percent / 100.0)))
        if len(positive) <= budget:
            return token_indices[positive]
        selected = np.argpartition(error[positive], -budget)[-budget:]
        return token_indices[np.sort(positive[selected])]

    raise ValueError(
        "content gate policy must be one of: none, fixed_refresh<N>, "
        "mismatch_ge<N>, budget_top<PCT>pct"
    )


def evaluate_content_gate(
    length: int,
    topology: str,
    policy: str,
    bits: int = 4,
    ticks: int = 1000,
    radius: int = 1,
    seed: int = 0,
) -> ContentGatePoint:
    """Measure local content-to-carrier write gating over a stable mHC carrier."""

    if length <= 0:
        raise ValueError("length must be positive")
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, 8")
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")

    policy = str(policy).lower()
    edges = _graph_for_topology(topology, length, radius)
    nodes = _ordered_nodes(edges)
    node_index = {node: index for index, node in enumerate(nodes)}
    neighbors = _neighbor_indices(edges, nodes)
    token_indices = np.array([node_index[(0, index)] for index in range(length)], dtype=np.int32)
    max_value = (1 << bits) - 1
    rng = np.random.default_rng(seed)

    content_values = np.zeros(len(nodes), dtype=np.uint8)
    content_values[token_indices] = rng.integers(
        0, max_value + 1, size=length, dtype=np.uint8
    )
    initial_token_content = content_values[token_indices].copy()
    carrier = np.zeros((len(nodes), 3), dtype=np.uint8)
    _inject_content_into_carrier(carrier, content_values, token_indices)

    gate_token_writes = 0
    carrier_exact_sum = 0.0
    carrier_error_sum = 0.0
    for tick in range(1, ticks + 1):
        carrier = _step_dynamic_state(
            state=carrier,
            rule="mhc_grouped",
            neighbors=neighbors,
            max_value=max_value,
            source_index=None,
        )
        selected = _content_gate_selection(
            policy=policy,
            tick=tick,
            carrier=carrier,
            content_values=content_values,
            token_indices=token_indices,
            max_value=max_value,
        )
        if len(selected) > 0:
            _inject_content_into_carrier(carrier, content_values, selected)
            gate_token_writes += int(len(selected))

        carrier_token_content = carrier[token_indices, 0]
        carrier_error = np.abs(
            carrier_token_content.astype(np.int16) - initial_token_content.astype(np.int16)
        )
        carrier_exact_sum += float(np.mean(carrier_token_content == initial_token_content))
        carrier_error_sum += float(np.mean(carrier_error) / float(max_value))

    final_content = content_values[token_indices]
    carrier_token_content = carrier[token_indices, 0]
    carrier_error = np.abs(
        carrier_token_content.astype(np.int16) - initial_token_content.astype(np.int16)
    )
    checksum = (
        _state_checksum(carrier)
        + int(
            np.sum(
                content_values[token_indices].astype(np.uint64)
                * np.arange(1, length + 1, dtype=np.uint64)
            )
            % (2**63 - 1)
        )
    ) % (2**63 - 1)

    return ContentGatePoint(
        topology=topology.lower(),
        policy=policy,
        length=length,
        bits=bits,
        ticks=ticks,
        seed=seed,
        graph_nodes=len(nodes),
        graph_edges=edge_count(edges),
        state_bits_per_token=4 * bits,
        gate_token_writes=gate_token_writes,
        gate_channel_writes_per_token_tick=3 * gate_token_writes / float(length * ticks),
        mean_gate_fraction=gate_token_writes / float(length * ticks),
        content_exact_retention_rate=float(np.mean(final_content == initial_token_content)),
        carrier_exact_retention_rate=float(np.mean(carrier_token_content == initial_token_content)),
        mean_carrier_exact_retention_rate=carrier_exact_sum / float(ticks),
        carrier_mean_abs_error=float(np.mean(carrier_error) / float(max_value)),
        mean_carrier_mean_abs_error=carrier_error_sum / float(ticks),
        carrier_final_entropy_bits=_level_entropy_bits(carrier, max_value),
        carrier_final_saturation_fraction=float(np.count_nonzero(carrier == max_value))
        / float(carrier.size),
        carrier_final_mean_level=float(np.mean(carrier)) / float(max_value),
        checksum=int(checksum),
    )


def run_content_gate_sweep(
    lengths: Tuple[int, ...] = (512,),
    topologies: Tuple[str, ...] = ("harc",),
    policies: Tuple[str, ...] = (
        "none",
        "fixed_refresh16",
        "mismatch_ge12",
        "mismatch_ge8",
        "mismatch_ge6",
        "mismatch_ge4",
        "budget_top5pct",
        "budget_top10pct",
    ),
    bits: int = 4,
    ticks: int = 1000,
    radius: int = 1,
    seed: int = 307,
) -> ContentGateResult:
    """Sweep local and budgeted content-to-carrier gate policies."""

    if len(lengths) == 0:
        raise ValueError("lengths must not be empty")
    clean_lengths = tuple(int(length) for length in lengths)
    if any(length <= 0 for length in clean_lengths):
        raise ValueError("lengths must be positive")
    if len(topologies) == 0:
        raise ValueError("topologies must not be empty")
    clean_topologies = tuple(dict.fromkeys(str(topology).lower() for topology in topologies))
    if len(policies) == 0:
        raise ValueError("policies must not be empty")
    clean_policies = tuple(dict.fromkeys(str(policy).lower() for policy in policies))

    points = []
    for length_index, length in enumerate(clean_lengths):
        for topology_index, topology in enumerate(clean_topologies):
            for policy in clean_policies:
                point_seed = seed + 1009 * length_index + 131 * topology_index
                points.append(
                    evaluate_content_gate(
                        length=length,
                        topology=topology,
                        policy=policy,
                        bits=bits,
                        ticks=ticks,
                        radius=radius,
                        seed=point_seed,
                    )
                )

    return ContentGateResult(
        lengths=clean_lengths,
        topologies=clean_topologies,
        policies=clean_policies,
        bits=bits,
        ticks=ticks,
        points=tuple(points),
    )
