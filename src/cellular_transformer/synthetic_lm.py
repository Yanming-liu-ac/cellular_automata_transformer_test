"""Synthetic next-token benchmark for the HARC-CA dual memory path.

This is not a trained language model. It is a controlled inference prototype
that combines:

- exact sparse memory for induction-style key -> value prediction;
- compressed dense context for topic-like next-token candidate ranking.

The benchmark is useful because it forces both memory paths to serve a
next-token interface instead of remaining isolated memory demos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .dense_context import DenseContextConfig, LowBitDenseContext
from .retrieval import HashRouteCAMConfig, TieredHashRouteCAM, TieredHashRouteCAMConfig, keyed_hash


@dataclass(frozen=True)
class SyntheticLMConfig:
    """Configuration for the dual-path synthetic next-token benchmark."""

    vocab_size: int = 65536
    hot_tokens: int = 256
    topic_top_k: int = 64
    candidate_pool_size: int = 512
    fact_count: int = 16384
    topic_events: int = 8192
    query_events: int = 4096
    topic_probability: float = 0.85
    zipf_exponent: float = 1.15
    dense_banks: int = 4
    dense_width: int = 2048
    dense_bits: int = 4
    dense_decay_interval: int = 256
    primary_buckets: int = 4096
    primary_ways: int = 4
    primary_routes: int = 2
    overflow_buckets: int = 1024
    overflow_ways: int = 4
    overflow_routes: int = 2
    tag_bits: int = 32

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 < self.hot_tokens < self.vocab_size:
            raise ValueError("hot_tokens must be in (0, vocab_size)")
        if not 0 < self.topic_top_k <= self.candidate_pool_size:
            raise ValueError("topic_top_k must be in (0, candidate_pool_size]")
        if self.candidate_pool_size < self.hot_tokens:
            raise ValueError("candidate_pool_size must include all hot tokens")
        if self.fact_count <= 0:
            raise ValueError("fact_count must be positive")
        if self.topic_events <= 0:
            raise ValueError("topic_events must be positive")
        if self.query_events <= 0:
            raise ValueError("query_events must be positive")
        if not 0.0 <= self.topic_probability <= 1.0:
            raise ValueError("topic_probability must be in [0, 1]")


@dataclass(frozen=True)
class SyntheticLMResult:
    """Aggregate dual-path next-token benchmark metrics."""

    vocab_size: int
    fact_count: int
    topic_events: int
    query_events: int
    topic_top_k: int
    candidate_pool_size: int
    induction_accuracy: float
    topic_topk_hit_rate: float
    exact_avg_visited_cells: float
    overflow_query_rate: float
    dense_update_cells_per_event: float
    avg_cells_per_event: float
    exact_memory_bytes: float
    dense_memory_bytes: float
    total_memory_bytes: float


def make_fact_pairs(config: SyntheticLMConfig, seed: int) -> List[Tuple[int, int]]:
    """Create deterministic key/value facts within the vocabulary."""

    rng = np.random.default_rng(seed)
    reserved = set(range(config.hot_tokens))
    keys = set()
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < config.fact_count:
        key = int(rng.integers(config.hot_tokens, config.vocab_size, dtype=np.uint32))
        if key in reserved or key in keys:
            continue
        keys.add(key)
        value = int(keyed_hash(key, 7001) % config.vocab_size)
        if value in reserved:
            value = config.hot_tokens + (value % (config.vocab_size - config.hot_tokens))
        pairs.append((key, value))
    return pairs


def make_candidate_pool(config: SyntheticLMConfig, seed: int) -> np.ndarray:
    """Candidate output shortlist for dense topic prediction."""

    rng = np.random.default_rng(seed)
    pool = list(range(config.hot_tokens))
    used = set(pool)
    while len(pool) < config.candidate_pool_size:
        token = int(rng.integers(config.hot_tokens, config.vocab_size, dtype=np.uint32))
        if token in used:
            continue
        used.add(token)
        pool.append(token)
    return np.array(pool, dtype=np.int32)


def sample_topic_token(config: SyntheticLMConfig, rng: np.random.Generator) -> int:
    """Sample a topic/noise token from a simple Zipf mixture."""

    if rng.random() > config.topic_probability:
        return int(rng.integers(config.hot_tokens, config.vocab_size, dtype=np.uint32))

    ranks = np.arange(1, config.hot_tokens + 1, dtype=np.float64)
    probabilities = 1.0 / np.power(ranks, config.zipf_exponent)
    probabilities /= probabilities.sum()
    return int(rng.choice(config.hot_tokens, p=probabilities))


class DualPathSyntheticLM:
    """Non-trained dual-path next-token predictor."""

    def __init__(self, config: SyntheticLMConfig, seed: int = 0) -> None:
        self.config = config
        dense_config = DenseContextConfig(
            vocab_size=config.vocab_size,
            banks=config.dense_banks,
            width=config.dense_width,
            bits=config.dense_bits,
            decay_interval=config.dense_decay_interval,
        )
        primary = HashRouteCAMConfig(
            buckets=config.primary_buckets,
            ways=config.primary_ways,
            routes=config.primary_routes,
            tag_bits=config.tag_bits,
        )
        overflow = HashRouteCAMConfig(
            buckets=config.overflow_buckets,
            ways=config.overflow_ways,
            routes=config.overflow_routes,
            tag_bits=config.tag_bits,
        )
        self.dense = LowBitDenseContext(dense_config)
        self.exact = TieredHashRouteCAM(TieredHashRouteCAMConfig(primary, overflow))
        self.rng = np.random.default_rng(seed)
        self.candidates = make_candidate_pool(config, seed + 1)
        self.candidate_slots = np.array(
            [
                [
                    keyed_hash(int(token), 1000 + bank) % config.dense_width
                    for token in self.candidates
                ]
                for bank in range(config.dense_banks)
            ],
            dtype=np.int32,
        )
        self.facts = make_fact_pairs(config, seed + 2)

    def prefill(self) -> int:
        """Insert facts and update dense state with observed key/value tokens."""

        touched = 0
        for key, value in self.facts:
            self.exact.insert(key, value)
            touched += self.dense.update(key)
            touched += self.dense.update(value)
        return touched

    def predict_topic_topk(self) -> set[int]:
        """Rank dense candidates by compressed-context estimate."""

        bank_indices = np.arange(self.config.dense_banks)[:, None]
        scores = self.dense.counters[bank_indices, self.candidate_slots].min(axis=0)
        top_indices = np.argsort(scores)[-self.config.topic_top_k :]
        return {int(self.candidates[index]) for index in top_indices}

    def run(self) -> SyntheticLMResult:
        """Run a mixed topic/induction next-token benchmark."""

        dense_touched = self.prefill()
        exact_visited = 0
        overflow_queries = 0
        correct_queries = 0
        topic_hits = 0

        query_indices = self.rng.choice(
            len(self.facts),
            size=self.config.query_events,
            replace=True,
        )
        event_types = np.array(
            ["topic"] * self.config.topic_events + ["query"] * self.config.query_events
        )
        self.rng.shuffle(event_types)

        query_cursor = 0
        for event_type in event_types:
            if event_type == "topic":
                token = sample_topic_token(self.config, self.rng)
                topic_hits += int(token in self.predict_topic_topk())
                dense_touched += self.dense.update(token)
                continue

            key, expected = self.facts[int(query_indices[query_cursor])]
            query_cursor += 1
            result = self.exact.lookup(key)
            exact_visited += result.visited_cells
            overflow_queries += int(result.used_overflow)
            correct_queries += int(result.found and result.correct and result.value == expected)
            dense_touched += self.dense.update(key)
            dense_touched += self.dense.update(expected)

        total_events = self.config.topic_events + self.config.query_events
        exact_memory = self.exact.memory_bytes()
        dense_memory = self.dense.memory_bytes()
        return SyntheticLMResult(
            vocab_size=self.config.vocab_size,
            fact_count=self.config.fact_count,
            topic_events=self.config.topic_events,
            query_events=self.config.query_events,
            topic_top_k=self.config.topic_top_k,
            candidate_pool_size=self.config.candidate_pool_size,
            induction_accuracy=correct_queries / self.config.query_events,
            topic_topk_hit_rate=topic_hits / self.config.topic_events,
            exact_avg_visited_cells=exact_visited / self.config.query_events,
            overflow_query_rate=overflow_queries / self.config.query_events,
            dense_update_cells_per_event=dense_touched / (total_events + 2 * self.config.fact_count),
            avg_cells_per_event=(dense_touched + exact_visited) / total_events,
            exact_memory_bytes=exact_memory,
            dense_memory_bytes=dense_memory,
            total_memory_bytes=exact_memory + dense_memory,
        )


def run_synthetic_lm_trial(seed: int = 0, config: SyntheticLMConfig | None = None) -> SyntheticLMResult:
    """Convenience wrapper for one synthetic dual-path LM trial."""

    lm = DualPathSyntheticLM(config or SyntheticLMConfig(), seed=seed)
    return lm.run()
