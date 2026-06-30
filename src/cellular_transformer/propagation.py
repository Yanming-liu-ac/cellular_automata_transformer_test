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

Node = Tuple[int, int]
EdgeMap = Dict[Node, List[Node]]


@dataclass(frozen=True)
class PropagationResult:
    """Shortest propagation result on a CA lattice graph."""

    source: Node
    target: Node
    steps: int | None
    visited_nodes: int


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
