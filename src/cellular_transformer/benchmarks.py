"""Sequence-memory benchmarks for CA-first retrieval lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import numpy as np

from .retrieval import (
    HashRouteCAM,
    HashRouteCAMConfig,
    TieredHashRouteCAM,
    TieredHashRouteCAMConfig,
    keyed_hash,
)

PairFactory = Callable[[int, int], List[Tuple[int, int]]]


@dataclass(frozen=True)
class MemoryTaskResult:
    """Aggregate result for one sequence-memory task."""

    task: str
    context_length: int
    buckets: int
    ways: int
    routes: int
    tag_bits: int
    load_factor: float
    evictions: int
    queries: int
    correct_rate: float
    false_positive_rate: float
    avg_visited_cells: float
    full_scan_cells: int
    memory_bytes: float
    memory: str = "single"
    overflow_buckets: int = 0
    overflow_ways: int = 0
    overflow_routes: int = 0
    overflow_evictions: int = 0
    overflow_insertions: int = 0
    overflow_queries: int = 0

    @property
    def scan_avoidance_ratio(self) -> float:
        if self.avg_visited_cells == 0.0:
            return 0.0
        return self.full_scan_cells / self.avg_visited_cells

    @property
    def overflow_query_rate(self) -> float:
        if self.queries == 0:
            return 0.0
        return self.overflow_queries / self.queries


def make_key_value_pairs(context_length: int, seed: int) -> List[Tuple[int, int]]:
    """Random key/value recall task."""

    rng = np.random.default_rng(seed)
    keys = set()
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < context_length:
        key = int(rng.integers(1, np.iinfo(np.uint64).max, dtype=np.uint64))
        if key in keys:
            continue
        keys.add(key)
        value = keyed_hash(key, 31) & ((1 << 32) - 1)
        pairs.append((key, value))
    return pairs


def make_position_copy_pairs(context_length: int, seed: int) -> List[Tuple[int, int]]:
    """Copy task: query a position and return the symbol stored there."""

    rng = np.random.default_rng(seed)
    symbols = rng.integers(0, 256, size=context_length, dtype=np.uint16)
    return [(index, int(symbols[index])) for index in range(context_length)]


def make_induction_pairs(context_length: int, seed: int) -> List[Tuple[int, int]]:
    """Induction task: after seeing A B, query A and recover B."""

    rng = np.random.default_rng(seed)
    keys = set()
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < context_length:
        a_token = int(rng.integers(1, 1 << 48, dtype=np.uint64))
        if a_token in keys:
            continue
        keys.add(a_token)
        b_token = int(rng.integers(0, 1 << 16, dtype=np.uint32))
        pairs.append((a_token, b_token))
    return pairs


TASK_FACTORIES: dict[str, PairFactory] = {
    "key_value": make_key_value_pairs,
    "copy": make_position_copy_pairs,
    "induction": make_induction_pairs,
}


def run_memory_task(
    task: str,
    context_length: int,
    buckets: int,
    ways: int = 4,
    routes: int = 2,
    tag_bits: int = 24,
    overflow_bucket_multiplier: float | None = None,
    overflow_ways: int = 4,
    overflow_routes: int = 2,
    query_count: int | None = None,
    seed: int = 0,
) -> MemoryTaskResult:
    """Run one exact-recall sequence-memory task."""

    if task not in TASK_FACTORIES:
        raise ValueError(f"unknown task: {task}")

    pairs = TASK_FACTORIES[task](context_length, seed)
    expected = dict(pairs)
    primary_config = HashRouteCAMConfig(
        buckets=buckets,
        ways=ways,
        routes=routes,
        tag_bits=tag_bits,
    )
    if overflow_bucket_multiplier is None:
        cam = HashRouteCAM(primary_config)
        memory = "single"
        overflow_buckets = 0
    else:
        overflow_buckets = max(1, int(round(context_length * overflow_bucket_multiplier)))
        overflow_config = HashRouteCAMConfig(
            buckets=overflow_buckets,
            ways=overflow_ways,
            routes=overflow_routes,
            tag_bits=tag_bits,
        )
        cam = TieredHashRouteCAM(TieredHashRouteCAMConfig(primary_config, overflow_config))
        memory = "tiered"

    for key, value in pairs:
        cam.insert(key, value)

    rng = np.random.default_rng(seed + 101)
    if query_count is None or query_count >= context_length:
        query_indices = np.arange(context_length)
    else:
        query_indices = rng.choice(context_length, size=query_count, replace=False)

    correct = 0
    false_positive = 0
    visited = 0
    overflow_queries = 0
    for index in query_indices:
        key, _ = pairs[int(index)]
        result = cam.lookup(key)
        value_correct = result.found and result.correct and result.value == expected[key]
        correct += int(value_correct)
        false_positive += int(result.found and not value_correct)
        visited += result.visited_cells
        overflow_queries += int(getattr(result, "used_overflow", False))

    queries = len(query_indices)
    return MemoryTaskResult(
        task=task,
        context_length=context_length,
        buckets=buckets,
        ways=ways,
        routes=routes,
        tag_bits=tag_bits,
        load_factor=context_length / primary_config.capacity,
        evictions=cam.evictions,
        queries=queries,
        correct_rate=correct / queries if queries else 0.0,
        false_positive_rate=false_positive / queries if queries else 0.0,
        avg_visited_cells=visited / queries if queries else 0.0,
        full_scan_cells=context_length,
        memory_bytes=cam.memory_bytes(),
        memory=memory,
        overflow_buckets=overflow_buckets,
        overflow_ways=overflow_ways if overflow_bucket_multiplier is not None else 0,
        overflow_routes=overflow_routes if overflow_bucket_multiplier is not None else 0,
        overflow_evictions=getattr(cam, "overflow_evictions", 0),
        overflow_insertions=getattr(cam, "overflow_insertions", 0),
        overflow_queries=overflow_queries,
    )


def sweep_memory_tasks(
    tasks: Iterable[str] = ("copy", "induction", "key_value"),
    lengths: Iterable[int] = (1024, 4096, 16384),
    routes_options: Iterable[int] = (1, 2),
    bucket_multiplier: float = 0.25,
    ways: int = 4,
    tag_bits: int = 24,
    overflow_bucket_multiplier: float | None = None,
    overflow_ways: int = 4,
    overflow_routes: int = 2,
    query_count: int = 1000,
    seed: int = 0,
) -> List[MemoryTaskResult]:
    """Sweep task, context, and route count at a fixed memory budget."""

    results: List[MemoryTaskResult] = []
    for task in tasks:
        for length in lengths:
            buckets = max(1, int(round(length * bucket_multiplier)))
            for routes in routes_options:
                results.append(
                    run_memory_task(
                        task=task,
                        context_length=length,
                        buckets=buckets,
                        ways=ways,
                        routes=routes,
                        tag_bits=tag_bits,
                        overflow_bucket_multiplier=overflow_bucket_multiplier,
                        overflow_ways=overflow_ways,
                        overflow_routes=overflow_routes,
                        query_count=min(query_count, length),
                        seed=seed,
                    )
                )
    return results
