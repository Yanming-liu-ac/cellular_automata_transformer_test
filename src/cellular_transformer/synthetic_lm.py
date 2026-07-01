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
from .propagation import (
    DemandContentGateLUT,
    DemandContentGatePoint,
    evaluate_lut_trace_demand_content_gate,
    evaluate_trace_demand_content_gate,
    train_trace_demand_content_gate_lut,
)
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
    candidate_admission_threshold: int = 0
    candidate_admission_lut: Tuple[int, ...] | None = None
    candidate_scorer_lut: Tuple[int, ...] | None = None
    candidate_scorer_dense_bins: int = 16
    candidate_scorer_cache_bins: int = 16
    candidate_scorer_dense_weight: int = 0
    candidate_scorer_cache_weight: int = 0
    candidate_score_source: str = "dense"
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
        if self.candidate_score_source not in (
            "dense",
            "topic_phase",
            "dense_topic_sum",
            "topic_cache",
            "dense_topic_cache",
            "topic_denoised",
        ):
            raise ValueError("candidate_score_source is not supported")
        if self.candidate_strategy == "static" and self.candidate_pool_size < self.hot_tokens:
            raise ValueError("candidate_pool_size must include all hot tokens")
        if self.candidate_strategy == "online_cache":
            if self.candidate_cache_ways <= 0:
                raise ValueError("candidate_cache_ways must be positive")
            if self.candidate_pool_size % self.candidate_cache_ways != 0:
                raise ValueError("candidate_pool_size must be divisible by candidate_cache_ways")
            if self.candidate_cache_routes <= 0:
                raise ValueError("candidate_cache_routes must be positive")
            if self.candidate_admission_threshold < 0:
                raise ValueError("candidate_admission_threshold must be non-negative")
            if self.candidate_admission_lut is not None and len(self.candidate_admission_lut) == 0:
                raise ValueError("candidate_admission_lut must not be empty")
            if self.candidate_scorer_lut is not None:
                expected = self.candidate_scorer_dense_bins * self.candidate_scorer_cache_bins
                if len(self.candidate_scorer_lut) != expected:
                    raise ValueError("candidate_scorer_lut length must equal dense_bins * cache_bins")
                if self.candidate_scorer_dense_weight < 0:
                    raise ValueError("candidate_scorer_dense_weight must be non-negative")
                if self.candidate_scorer_cache_weight < 0:
                    raise ValueError("candidate_scorer_cache_weight must be non-negative")
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
    candidate_admission_mode: str
    candidate_scorer_mode: str
    candidate_score_source: str
    induction_accuracy: float
    topic_topk_hit_rate: float
    exact_avg_visited_cells: float
    overflow_query_rate: float
    dense_update_cells_per_event: float
    candidate_update_cells_per_event: float
    candidate_gate_cells_per_event: float
    candidate_score_cells_per_event: float
    candidate_score_update_cells_per_event: float
    candidate_admission_rate: float
    candidate_admission_skips: int
    candidate_cache_hit_rate: float
    candidate_cache_replacements: int
    candidate_cache_resident_tokens: int
    avg_cells_per_event: float
    exact_memory_bytes: float
    dense_memory_bytes: float
    candidate_score_memory_bytes: float
    candidate_memory_bytes: float
    total_memory_bytes: float


@dataclass(frozen=True)
class SyntheticLMDemandGateResult:
    """Content-gate diagnostic driven by synthetic exact-query demand."""

    fact_count: int
    topic_events: int
    query_events: int
    total_events: int
    bits: int
    train_seed: int
    eval_seed: int
    write_cost: float
    lut_state_bytes: float
    lut_write_state_count: int
    lut: DemandContentGateLUT
    points: Tuple[DemandContentGatePoint, ...]


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


def make_exact_query_demand_trace(
    config: SyntheticLMConfig,
    seed: int,
) -> Tuple[np.ndarray, ...]:
    """Return one demand trace for exact-memory query events.

    Topic events do not demand exact fact rows. Query events demand the fact row
    selected by the same event-order process as ``DualPathSyntheticLM.run``.
    """

    rng = np.random.default_rng(seed)
    query_indices = rng.choice(
        config.fact_count,
        size=config.query_events,
        replace=True,
    )
    event_types = np.array(
        ["topic"] * config.topic_events + ["query"] * config.query_events
    )
    rng.shuffle(event_types)

    trace = []
    query_cursor = 0
    for event_type in event_types:
        if event_type == "query":
            trace.append(np.array([int(query_indices[query_cursor])], dtype=np.int32))
            query_cursor += 1
        else:
            trace.append(np.empty(0, dtype=np.int32))
    return tuple(trace)


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
        self.candidate_score_dense: LowBitDenseContext | None = None
        if config.candidate_score_source != "dense":
            self.candidate_score_dense = LowBitDenseContext(dense_config)
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

    def predict_topic_topk(self) -> tuple[set[int], int]:
        """Rank dense candidates by compressed-context estimate."""

        if self.config.candidate_strategy == "online_cache":
            if self.candidate_cache is None:
                raise RuntimeError("candidate cache is not initialized")
            entries = self.candidate_cache.resident_entries()
            entries = sorted(entries, key=lambda item: (-item[1], item[0]))[
                : self.config.candidate_pool_size
            ]
            candidates = np.array([token for token, _ in entries], dtype=np.int32)
            cache_scores = np.array([score for _, score in entries], dtype=np.int32)
            if len(candidates) == 0:
                return set(), 0
            candidate_slots = self._candidate_slots(candidates)
            top_k = min(self.config.topic_top_k, len(candidates))
        else:
            if self.candidates is None or self.candidate_slots is None:
                raise RuntimeError("static candidates are not initialized")
            candidates = self.candidates
            cache_scores = np.zeros(len(candidates), dtype=np.int32)
            candidate_slots = self.candidate_slots
            top_k = self.config.topic_top_k

        bank_indices = np.arange(self.config.dense_banks)[:, None]
        scores, score_cells = self._candidate_scores(
            candidate_slots=candidate_slots,
            cache_scores=cache_scores,
            bank_indices=bank_indices,
        )
        if self.config.candidate_scorer_lut is not None:
            learned_scores = np.empty(len(candidates), dtype=np.int32)
            for index, (dense_score, cache_score) in enumerate(zip(scores, cache_scores)):
                dense_index = min(int(dense_score), self.config.candidate_scorer_dense_bins - 1)
                cache_index = min(int(cache_score), self.config.candidate_scorer_cache_bins - 1)
                lut_index = dense_index * self.config.candidate_scorer_cache_bins + cache_index
                learned_scores[index] = (
                    self.config.candidate_scorer_dense_weight * int(dense_score)
                    + self.config.candidate_scorer_cache_weight * int(cache_score)
                    + int(self.config.candidate_scorer_lut[lut_index])
                )
            top_indices = np.lexsort((scores, learned_scores))[-top_k:]
        else:
            top_indices = np.argsort(scores)[-top_k:]
        return {int(candidates[index]) for index in top_indices}, score_cells

    def run(self) -> SyntheticLMResult:
        """Run a mixed topic/induction next-token benchmark."""

        dense_touched = self.prefill()
        exact_visited = 0
        overflow_queries = 0
        correct_queries = 0
        topic_hits = 0
        candidate_touched = 0
        candidate_gate_touched = 0
        candidate_score_touched = 0
        candidate_score_update_touched = 0
        candidate_admitted = 0
        candidate_skipped = 0

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
                prediction, score_cells = self.predict_topic_topk()
                candidate_score_touched += score_cells
                topic_hits += int(token in prediction)
                admit_candidate = True
                needs_admission_gate = self.candidate_cache is not None and (
                    self.config.candidate_admission_lut is not None
                    or self.config.candidate_admission_threshold > 0
                )
                if needs_admission_gate:
                    candidate_gate_touched += self.config.dense_banks
                    estimate = self.dense.estimate(token)
                    if self.config.candidate_admission_lut is not None:
                        index = min(estimate, len(self.config.candidate_admission_lut) - 1)
                        admit_candidate = self.config.candidate_admission_lut[index] >= 0
                    else:
                        admit_candidate = estimate >= self.config.candidate_admission_threshold
                dense_touched += self.dense.update(token)
                if self.candidate_score_dense is not None:
                    candidate_score_update_touched += self.candidate_score_dense.update(token)
                if self.candidate_cache is not None:
                    if admit_candidate:
                        candidate_admitted += 1
                        candidate_touched += self.candidate_cache.observe(token).total_touched_cells
                    else:
                        candidate_skipped += 1
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
        candidate_score_memory = (
            self.candidate_score_dense.memory_bytes()
            if self.candidate_score_dense is not None
            else 0.0
        )
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
        candidate_observations = candidate_admitted + candidate_skipped
        candidate_admission_rate = (
            candidate_admitted / candidate_observations if candidate_observations else 0.0
        )
        return SyntheticLMResult(
            vocab_size=self.config.vocab_size,
            fact_count=self.config.fact_count,
            topic_events=self.config.topic_events,
            query_events=self.config.query_events,
            topic_top_k=self.config.topic_top_k,
            candidate_pool_size=self.config.candidate_pool_size,
            candidate_strategy=self.config.candidate_strategy,
            candidate_admission_mode=self._candidate_admission_mode(),
            candidate_scorer_mode=self._candidate_scorer_mode(),
            candidate_score_source=self.config.candidate_score_source,
            induction_accuracy=correct_queries / self.config.query_events,
            topic_topk_hit_rate=topic_hits / self.config.topic_events,
            exact_avg_visited_cells=exact_visited / self.config.query_events,
            overflow_query_rate=overflow_queries / self.config.query_events,
            dense_update_cells_per_event=dense_touched / (total_events + 2 * self.config.fact_count),
            candidate_update_cells_per_event=candidate_touched / total_events,
            candidate_gate_cells_per_event=candidate_gate_touched / total_events,
            candidate_score_cells_per_event=candidate_score_touched / total_events,
            candidate_score_update_cells_per_event=candidate_score_update_touched / total_events,
            candidate_admission_rate=candidate_admission_rate,
            candidate_admission_skips=candidate_skipped,
            candidate_cache_hit_rate=candidate_cache_hit_rate,
            candidate_cache_replacements=candidate_cache_replacements,
            candidate_cache_resident_tokens=candidate_cache_resident,
            avg_cells_per_event=(
                dense_touched
                + exact_visited
                + candidate_touched
                + candidate_gate_touched
                + candidate_score_touched
                + candidate_score_update_touched
            )
            / total_events,
            exact_memory_bytes=exact_memory,
            dense_memory_bytes=dense_memory,
            candidate_score_memory_bytes=candidate_score_memory,
            candidate_memory_bytes=candidate_memory,
            total_memory_bytes=exact_memory + dense_memory + candidate_score_memory + candidate_memory,
        )

    def _candidate_scores(
        self,
        candidate_slots: np.ndarray,
        cache_scores: np.ndarray,
        bank_indices: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        dense_scores = self.dense.counters[bank_indices, candidate_slots].min(axis=0)
        read_cells = candidate_slots.size
        source = self.config.candidate_score_source
        if source == "dense":
            return dense_scores, read_cells

        if self.candidate_score_dense is None:
            raise RuntimeError("candidate score dense state is not initialized")
        topic_scores = self.candidate_score_dense.counters[bank_indices, candidate_slots].min(axis=0)
        topic_read_cells = candidate_slots.size

        if source == "topic_phase":
            return topic_scores, topic_read_cells
        dense_scores = dense_scores.astype(np.int32)
        topic_scores = topic_scores.astype(np.int32)
        if source == "dense_topic_sum":
            return dense_scores + topic_scores, read_cells + topic_read_cells
        if source == "topic_cache":
            return 2 * topic_scores + cache_scores.astype(np.int32), topic_read_cells
        if source == "dense_topic_cache":
            return dense_scores + topic_scores + cache_scores.astype(np.int32), read_cells + topic_read_cells
        if source == "topic_denoised":
            contamination = np.maximum(dense_scores - topic_scores, 0)
            return 2 * topic_scores - contamination, read_cells + topic_read_cells
        raise RuntimeError("unsupported candidate score source")

    def _candidate_admission_mode(self) -> str:
        if self.config.candidate_strategy != "online_cache":
            return "none"
        if self.config.candidate_admission_lut is not None:
            return "learned_lut"
        if self.config.candidate_admission_threshold > 0:
            return f"threshold_{self.config.candidate_admission_threshold}"
        return "always"

    def _candidate_scorer_mode(self) -> str:
        if self.config.candidate_scorer_lut is not None:
            if (
                self.config.candidate_scorer_dense_weight > 0
                or self.config.candidate_scorer_cache_weight > 0
            ):
                return "learned_residual"
            return "learned_lut"
        return "dense_min"


def run_synthetic_lm_trial(seed: int = 0, config: SyntheticLMConfig | None = None) -> SyntheticLMResult:
    """Convenience wrapper for one synthetic dual-path LM trial."""

    lm = DualPathSyntheticLM(config or SyntheticLMConfig(), seed=seed)
    return lm.run()


def run_synthetic_lm_demand_gate_sweep(
    config: SyntheticLMConfig | None = None,
    policies: Tuple[str, ...] = (
        "none",
        "fixed_refresh16",
        "mismatch_ge8",
        "demand_mismatch_ge1",
        "demand_mismatch_ge4",
    ),
    bits: int = 4,
    train_seed: int = 31,
    eval_seed: int = 37,
    write_cost: float = 0.15,
    route_weight: float = 0.50,
    envelope_weight: float = 0.25,
) -> SyntheticLMDemandGateResult:
    """Train/evaluate content gates on synthetic exact-query demand."""

    gate_config = config or SyntheticLMConfig(
        fact_count=512,
        topic_events=512,
        query_events=256,
        dense_width=512,
        primary_buckets=512,
        overflow_buckets=128,
    )
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, 8")
    if len(policies) == 0:
        raise ValueError("policies must not be empty")

    train_trace = make_exact_query_demand_trace(gate_config, seed=train_seed)
    eval_trace = make_exact_query_demand_trace(gate_config, seed=eval_seed)
    lut = train_trace_demand_content_gate_lut(
        demand_trace=train_trace,
        length=gate_config.fact_count,
        bits=bits,
        seed=train_seed + 8192,
        write_cost=write_cost,
        route_weight=route_weight,
        envelope_weight=envelope_weight,
    )

    points = []
    for policy in tuple(dict.fromkeys(str(policy).lower() for policy in policies)):
        points.append(
            evaluate_trace_demand_content_gate(
                policy=policy,
                demand_trace=eval_trace,
                length=gate_config.fact_count,
                bits=bits,
                seed=eval_seed + 8192,
            )
        )
    points.append(
        evaluate_lut_trace_demand_content_gate(
            lut=lut,
            demand_trace=eval_trace,
            length=gate_config.fact_count,
            bits=bits,
            seed=eval_seed + 8192,
            policy=f"learned_exact_trace_lut_c{write_cost:0.2f}",
        )
    )

    return SyntheticLMDemandGateResult(
        fact_count=gate_config.fact_count,
        topic_events=gate_config.topic_events,
        query_events=gate_config.query_events,
        total_events=gate_config.topic_events + gate_config.query_events,
        bits=bits,
        train_seed=train_seed,
        eval_seed=eval_seed,
        write_cost=write_cost,
        lut_state_bytes=lut.state_bytes,
        lut_write_state_count=lut.write_state_count,
        lut=lut,
        points=tuple(points),
    )
