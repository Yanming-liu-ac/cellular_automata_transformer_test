"""Associative retrieval prototypes for CA-first sequence models.

Long-range exact recall is the first hard gate for a CA language architecture.
This module models a hardware-shaped associative lane: a query routes through a
local hash tree to a small set-associative bucket, then compares low-bit tags.

This is not a Transformer attention replacement yet. It is the smallest
component that could keep a CA model from scanning the full context for copy and
induction-style tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import Iterable, List, Tuple

import numpy as np

U64_MASK = (1 << 64) - 1
GOLDEN64 = 0x9E3779B97F4A7C15


def splitmix64(value: int) -> int:
    """Stable 64-bit integer mixer."""

    z = (value + GOLDEN64) & U64_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & U64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & U64_MASK
    return (z ^ (z >> 31)) & U64_MASK


def keyed_hash(key: int, salt: int) -> int:
    """Stable salted hash for integer keys."""

    return splitmix64((int(key) + salt * GOLDEN64) & U64_MASK)


@dataclass(frozen=True)
class HashRouteCAMConfig:
    """Configuration for the hash-routed associative CA lane."""

    buckets: int
    ways: int = 4
    routes: int = 1
    tag_bits: int = 24
    value_bits: int = 32

    def __post_init__(self) -> None:
        if self.buckets <= 0:
            raise ValueError("buckets must be positive")
        if self.ways <= 0:
            raise ValueError("ways must be positive")
        if self.routes <= 0:
            raise ValueError("routes must be positive")
        if not 1 <= self.tag_bits <= 63:
            raise ValueError("tag_bits must be in [1, 63]")
        if not 1 <= self.value_bits <= 63:
            raise ValueError("value_bits must be in [1, 63]")

    @property
    def capacity(self) -> int:
        return self.buckets * self.ways

    @property
    def route_steps(self) -> int:
        return ceil(log2(self.buckets)) if self.buckets > 1 else 0

    @property
    def entry_bits(self) -> int:
        return self.tag_bits + self.value_bits + 1


@dataclass(frozen=True)
class LookupResult:
    """One CAM lookup trace."""

    key: int
    bucket: int
    found: bool
    correct: bool
    value: int | None
    route_steps: int
    way_reads: int
    tag_matches: int
    routes: int

    @property
    def visited_cells(self) -> int:
        return self.route_steps * self.routes + self.way_reads


@dataclass(frozen=True)
class RecallTrialResult:
    """Aggregate recall metrics for the associative lane."""

    context_length: int
    buckets: int
    ways: int
    routes: int
    tag_bits: int
    load_factor: float
    evictions: int
    queries: int
    found_rate: float
    correct_rate: float
    false_positive_rate: float
    avg_visited_cells: float
    full_scan_cells: int
    memory_bytes: float


class HashRouteCAM:
    """Set-associative hash-routed memory with low-bit tags.

    The chip interpretation is a small CA fabric:

    - hash bits route a query through one or more binary local trees;
    - each route reaches one small set-associative bucket;
    - ways compare tags in parallel with XNOR/equality logic;
    - values are returned without scanning sequence cells.
    """

    def __init__(self, config: HashRouteCAMConfig) -> None:
        self.config = config
        shape = (config.buckets, config.ways)
        self.valid = np.zeros(shape, dtype=np.bool_)
        self.tags = np.zeros(shape, dtype=np.uint64)
        self.values = np.zeros(shape, dtype=np.uint64)
        self.debug_keys = np.zeros(shape, dtype=np.uint64)
        self.ages = np.zeros(shape, dtype=np.uint64)
        self.clock = np.uint64(0)
        self.evictions = 0

    def _bucket(self, key: int) -> int:
        return keyed_hash(key, 1) % self.config.buckets

    def _buckets(self, key: int) -> Tuple[int, ...]:
        buckets = []
        seen = set()
        for route in range(self.config.routes):
            bucket = keyed_hash(key, route + 1) % self.config.buckets
            if bucket not in seen:
                seen.add(bucket)
                buckets.append(bucket)
        return tuple(buckets)

    def _tag(self, key: int) -> int:
        mask = (1 << self.config.tag_bits) - 1
        tag = keyed_hash(key, 2) & mask
        return tag if tag != 0 else 1

    def _value_mask(self) -> int:
        return (1 << self.config.value_bits) - 1

    def insert(self, key: int, value: int) -> bool:
        """Insert or update one key/value pair.

        Returns `True` if an occupied entry was evicted.
        """

        buckets = self._buckets(key)
        tag = self._tag(key)
        value_u64 = int(value) & self._value_mask()
        self.clock = np.uint64(int(self.clock) + 1)

        for bucket in buckets:
            valid = self.valid[bucket]
            same_tag = valid & (self.tags[bucket] == tag)
            same_key = same_tag & (self.debug_keys[bucket] == int(key))
            if np.any(same_key):
                way = int(np.argmax(same_key))
                self.values[bucket, way] = value_u64
                self.ages[bucket, way] = self.clock
                return False

        best_empty: Tuple[int, int] | None = None
        lowest_occupancy = self.config.ways + 1
        for bucket in buckets:
            valid = self.valid[bucket]
            occupancy = int(np.count_nonzero(valid))
            empty = np.flatnonzero(~valid)
            if len(empty) > 0 and occupancy < lowest_occupancy:
                best_empty = (bucket, int(empty[0]))
                lowest_occupancy = occupancy

        if best_empty is not None:
            bucket, way = best_empty
            evicted = False
        else:
            bucket, way = min(
                (
                    (bucket, int(np.argmin(self.ages[bucket])))
                    for bucket in buckets
                ),
                key=lambda item: int(self.ages[item[0], item[1]]),
            )
            evicted = True
            self.evictions += 1

        self.valid[bucket, way] = True
        self.tags[bucket, way] = tag
        self.values[bucket, way] = value_u64
        self.debug_keys[bucket, way] = int(key)
        self.ages[bucket, way] = self.clock
        return evicted

    def lookup(self, key: int) -> LookupResult:
        """Lookup one key using only bucket routing and tag comparison."""

        buckets = self._buckets(key)
        tag = self._tag(key)
        total_tag_matches = 0
        for bucket in buckets:
            matches = self.valid[bucket] & (self.tags[bucket] == tag)
            tag_matches = int(np.count_nonzero(matches))
            total_tag_matches += tag_matches
            if tag_matches == 0:
                continue

            way = int(np.argmax(matches))
            value = int(self.values[bucket, way])
            correct = int(self.debug_keys[bucket, way]) == int(key)
            return LookupResult(
                key=int(key),
                bucket=bucket,
                found=True,
                correct=correct,
                value=value,
                route_steps=self.config.route_steps,
                way_reads=self.config.ways * len(buckets),
                tag_matches=total_tag_matches,
                routes=len(buckets),
            )

        return LookupResult(
            key=int(key),
            bucket=buckets[0],
            found=False,
            correct=False,
            value=None,
            route_steps=self.config.route_steps,
            way_reads=self.config.ways * len(buckets),
            tag_matches=total_tag_matches,
            routes=len(buckets),
        )

    def memory_bytes(self) -> float:
        """Deployment-shaped storage estimate, excluding debug keys and ages."""

        return self.config.capacity * self.config.entry_bits / 8


def make_induction_pairs(context_length: int, seed: int = 0) -> List[Tuple[int, int]]:
    """Generate deterministic key/value pairs for an induction-style task."""

    rng = np.random.default_rng(seed)
    keys = set()
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < context_length:
        key = int(rng.integers(1, np.iinfo(np.uint64).max, dtype=np.uint64))
        if key in keys:
            continue
        keys.add(key)
        value = keyed_hash(key, 3) & ((1 << 32) - 1)
        pairs.append((key, value))
    return pairs


def run_recall_trial(
    context_length: int,
    buckets: int,
    ways: int = 4,
    routes: int = 1,
    tag_bits: int = 24,
    query_count: int | None = None,
    seed: int = 0,
) -> RecallTrialResult:
    """Run a random induction-style recall trial."""

    pairs = make_induction_pairs(context_length, seed)
    expected = dict(pairs)
    config = HashRouteCAMConfig(buckets=buckets, ways=ways, routes=routes, tag_bits=tag_bits)
    cam = HashRouteCAM(config)
    for key, value in pairs:
        cam.insert(key, value)

    rng = np.random.default_rng(seed + 17)
    if query_count is None or query_count >= context_length:
        query_indices = np.arange(context_length)
    else:
        query_indices = rng.choice(context_length, size=query_count, replace=False)

    found = 0
    correct = 0
    false_positive = 0
    visited = 0
    for index in query_indices:
        key, _ = pairs[int(index)]
        result = cam.lookup(key)
        found += int(result.found)
        value_correct = result.found and result.value == expected[key] and result.correct
        correct += int(value_correct)
        false_positive += int(result.found and not value_correct)
        visited += result.visited_cells

    queries = len(query_indices)
    found_rate = found / queries if queries else 0.0
    correct_rate = correct / queries if queries else 0.0
    false_positive_rate = false_positive / queries if queries else 0.0
    avg_visited = visited / queries if queries else 0.0

    return RecallTrialResult(
        context_length=context_length,
        buckets=buckets,
        ways=ways,
        routes=routes,
        tag_bits=tag_bits,
        load_factor=context_length / config.capacity,
        evictions=cam.evictions,
        queries=queries,
        found_rate=found_rate,
        correct_rate=correct_rate,
        false_positive_rate=false_positive_rate,
        avg_visited_cells=avg_visited,
        full_scan_cells=context_length,
        memory_bytes=cam.memory_bytes(),
    )


def sweep_recall_trials(
    lengths: Iterable[int],
    bucket_multipliers: Iterable[float] = (0.5, 1.0, 2.0),
    ways: int = 4,
    routes: int = 1,
    tag_bits: int = 24,
    query_count: int = 1000,
    seed: int = 0,
) -> List[RecallTrialResult]:
    """Sweep context lengths and bucket budgets."""

    results: List[RecallTrialResult] = []
    for length in lengths:
        for multiplier in bucket_multipliers:
            buckets = max(1, int(round(length * multiplier)))
            results.append(
                run_recall_trial(
                    context_length=length,
                    buckets=buckets,
                    ways=ways,
                    routes=routes,
                    tag_bits=tag_bits,
                    query_count=min(query_count, length),
                    seed=seed,
                )
            )
    return results
