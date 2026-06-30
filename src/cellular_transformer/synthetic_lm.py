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

from .candidate_cache import CandidateCacheConfig, LowBitCandidateCache
from .dense_context import DenseContextConfig, LowBitDenseContext
from .retrieval import HashRouteCAMConfig, TieredHashRouteCAM, TieredHashRouteCAMConfig, keyed_hash


@dataclass(frozen=True)
class SyntheticLMConfig:
    """Configuration for the dual-path synthetic next-token benchmark."""

    vocab_size: int = 65536
    hot_tokens: int = 256
    topic_top_k: int = 64
    candidate_pool_size: int = 512
    candidate_strategy: str = "static"
    candidate_cache_ways: int = 4
    candidate_cache_routes: int = 2
    candidate_cache_score_bits: int = 4
    candidate_cache_decay_interval: int = 256
    candidate_cache_decay_shift: int = 1
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
        if self.candidate_strategy not in ("static", "online_cache"):
            raise ValueError("candidate_strategy must be 'static' or 'online_cache'")
        if self.candidate_strategy == "static" and self.candidate_pool_size < self.hot_tokens:
            raise ValueError("candidate_pool_size must include all hot tokens")
        if self.candidate_strategy == "online_cache":
            if self.candidate_cache_ways <= 0:
                raise ValueError("candidate_cache_ways must be positive")
            if self.candidate_pool_size % self.candidate_cache_ways != 0:
                raise ValueError("candidate_pool_size must be divisible by candidate_cache_ways")
            if self.candidate_cache_routes <= 0:
                raise ValueError("candidate_cache_routes must be positive")
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
    candidate_strategy: str
    induction_accuracy: float
    topic_topk_hit_rate: float
    exact_avg_visited_cells: float
    overflow_query_rate: float
    dense_update_cells_per_event: float
    candidate_update_cells_per_event: float
    candidate_cache_hit_rate: float
    candidate_cache_replacements: int
    candidate_cache_resident_tokens: int
    avg_cells_per_event: float
    exact_memory_bytes: float
    dense_memory_bytes: float
    candidate_memory_bytes: float
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
        self.candidate_cache: LowBitCandidateCache | None = None
        self.candidates: np.ndarray | None = None
        self.candidate_slots: np.ndarray | None = None
        if config.candidate_strategy == "static":
            self.candidates = make_candidate_pool(config, seed + 1)
            self.candidate_slots = self._candidate_slots(self.candidates)
        else:
            cache_config = CandidateCacheConfig(
                vocab_size=config.vocab_size,
                capacity=config.candidate_pool_size,
                ways=config.candidate_cache_ways,
                routes=config.candidate_cache_routes,
                score_bits=config.candidate_cache_score_bits,
                token_bits=max(1, (config.vocab_size - 1).bit_length()),
                decay_interval=config.candidate_cache_decay_interval,
                decay_shift=config.candidate_cache_decay_shift,
            )
            self.candidate_cache = LowBitCandidateCache(cache_config)
        self.facts = make_fact_pairs(config, seed + 2)

    def _candidate_slots(self, candidates: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [
                    keyed_hash(int(token), 1000 + bank) % self.config.dense_width
                    for token in candidates
                ]
                for bank in range(self.config.dense_banks)
            ],
            dtype=np.int32,
        )

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

        if self.config.candidate_strategy == "online_cache":
            if self.candidate_cache is None:
                raise RuntimeError("candidate cache is not initialized")
            candidates = np.array(
                self.candidate_cache.topk(self.config.candidate_pool_size),
                dtype=np.int32,
            )
            if len(candidates) == 0:
                return set()
            candidate_slots = self._candidate_slots(candidates)
            top_k = min(self.config.topic_top_k, len(candidates))
        else:
            if self.candidates is None or self.candidate_slots is None:
                raise RuntimeError("static candidates are not initialized")
            candidates = self.candidates
            candidate_slots = self.candidate_slots
            top_k = self.config.topic_top_k

        bank_indices = np.arange(self.config.dense_banks)[:, None]
        scores = self.dense.counters[bank_indices, candidate_slots].min(axis=0)
        top_indices = np.argsort(scores)[-top_k:]
        return {int(candidates[index]) for index in top_indices}

    def run(self) -> SyntheticLMResult:
        """Run a mixed topic/induction next-token benchmark."""

        dense_touched = self.prefill()
        exact_visited = 0
        overflow_queries = 0
        correct_queries = 0
        topic_hits = 0
        candidate_touched = 0

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
                if self.candidate_cache is not None:
                    candidate_touched += self.candidate_cache.observe(token).total_touched_cells
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
        candidate_memory = (
            self.candidate_cache.memory_bytes() if self.candidate_cache is not None else 0.0
        )
        candidate_cache_hit_rate = (
            self.candidate_cache.cache_update_hit_rate() if self.candidate_cache is not None else 0.0
        )
        candidate_cache_replacements = (
            self.candidate_cache.replacements if self.candidate_cache is not None else 0
        )
        candidate_cache_resident = (
            self.candidate_cache.resident_count() if self.candidate_cache is not None else 0
        )
        return SyntheticLMResult(
            vocab_size=self.config.vocab_size,
            fact_count=self.config.fact_count,
            topic_events=self.config.topic_events,
            query_events=self.config.query_events,
            topic_top_k=self.config.topic_top_k,
            candidate_pool_size=self.config.candidate_pool_size,
            candidate_strategy=self.config.candidate_strategy,
            induction_accuracy=correct_queries / self.config.query_events,
            topic_topk_hit_rate=topic_hits / self.config.topic_events,
            exact_avg_visited_cells=exact_visited / self.config.query_events,
            overflow_query_rate=overflow_queries / self.config.query_events,
            dense_update_cells_per_event=dense_touched / (total_events + 2 * self.config.fact_count),
            candidate_update_cells_per_event=candidate_touched / total_events,
            candidate_cache_hit_rate=candidate_cache_hit_rate,
            candidate_cache_replacements=candidate_cache_replacements,
            candidate_cache_resident_tokens=candidate_cache_resident,
            avg_cells_per_event=(dense_touched + exact_visited + candidate_touched) / total_events,
            exact_memory_bytes=exact_memory,
            dense_memory_bytes=dense_memory,
            candidate_memory_bytes=candidate_memory,
            total_memory_bytes=exact_memory + dense_memory + candidate_memory,
        )


def run_synthetic_lm_trial(seed: int = 0, config: SyntheticLMConfig | None = None) -> SyntheticLMResult:
    """Convenience wrapper for one synthetic dual-path LM trial."""

    lm = DualPathSyntheticLM(config or SyntheticLMConfig(), seed=seed)
    return lm.run()
