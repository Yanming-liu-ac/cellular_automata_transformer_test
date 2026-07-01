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
    _demand_indices_to_mask,
    _graph_for_topology,
    _inject_content_into_carrier,
    _level_entropy_bits,
    _neighbor_indices,
    _ordered_nodes,
    _state_checksum,
    _step_dynamic_state,
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
    """Content-gate diagnostic driven by synthetic LM demand traces."""

    demand_trace: str
    fact_count: int
    candidate_rows: int
    topic_events: int
    query_events: int
    total_events: int
    content_rows: int
    bits: int
    train_seed: int
    eval_seed: int
    write_cost: float
    lut_state_bytes: float
    lut_write_state_count: int
    lut: DemandContentGateLUT
    points: Tuple[DemandContentGatePoint, ...]


@dataclass(frozen=True)
class SyntheticLMCandidateDemandSweepPoint:
    """One candidate-output demand sparsity point."""

    candidate_rows: int
    content_rows: int
    mean_demand_fraction: float
    fixed_refresh_writes_per_token_tick: float
    fixed_refresh_demand_exact_rate: float
    demand_mismatch_writes_per_token_tick: float
    demand_mismatch_demand_exact_rate: float
    learned_writes_per_token_tick: float
    learned_demand_exact_rate: float
    learned_demand_mean_abs_error: float
    lut_state_bytes: float
    lut_write_state_count: int
    phase_lut_state_bytes: float
    phase_lut_write_state_count: int
    phase_writes_per_token_tick: float
    phase_demand_exact_rate: float
    phase_demand_mean_abs_error: float


@dataclass(frozen=True)
class SyntheticLMCandidateDemandSweepResult:
    """Candidate-output demand sparsity sweep for the synthetic LM."""

    fact_count: int
    topic_events: int
    query_events: int
    total_events: int
    bits: int
    train_seed: int
    eval_seed: int
    write_cost: float
    points: Tuple[SyntheticLMCandidateDemandSweepPoint, ...]


@dataclass(frozen=True)
class SyntheticLMCandidateReducerTrace:
    """Demand trace produced by a low-bit candidate reducer."""

    demand_trace: Tuple[np.ndarray, ...]
    fact_count: int
    candidate_pool_size: int
    content_rows: int
    reducer_rows: int
    base_top_k: int
    topic_events: int
    query_events: int
    total_events: int
    candidate_score_source: str
    base_topic_hit_rate: float
    reduced_topic_hit_rate: float
    score_cells_per_topic_event: float
    score_cells_per_event: float


@dataclass(frozen=True)
class SyntheticLMCandidateReducerPoint:
    """One low-bit candidate reducer point with content-gate cost."""

    reducer_rows: int
    content_rows: int
    candidate_score_source: str
    base_topic_hit_rate: float
    reduced_topic_hit_rate: float
    hit_retention_rate: float
    mean_demand_fraction: float
    score_cells_per_topic_event: float
    score_cells_per_event: float
    phase_lut_state_bytes: float
    phase_lut_write_state_count: int
    phase_writes_per_token_tick: float
    phase_channel_writes_per_event: float
    phase_demand_exact_rate: float
    phase_demand_mean_abs_error: float


@dataclass(frozen=True)
class SyntheticLMCandidateReducerResult:
    """Low-bit candidate reducer plus phase/rank content-gate sweep."""

    fact_count: int
    candidate_pool_size: int
    base_top_k: int
    topic_events: int
    query_events: int
    total_events: int
    bits: int
    train_seed: int
    eval_seed: int
    write_cost: float
    points: Tuple[SyntheticLMCandidateReducerPoint, ...]


@dataclass(frozen=True)
class SyntheticLMHierarchicalCandidateReducerTrace:
    """Demand trace produced by a group-summary candidate reducer."""

    demand_trace: Tuple[np.ndarray, ...]
    fact_count: int
    candidate_pool_size: int
    content_rows: int
    reducer_rows: int
    base_top_k: int
    group_size: int
    selected_groups: int
    total_groups: int
    topic_events: int
    query_events: int
    total_events: int
    candidate_score_source: str
    base_topic_hit_rate: float
    reduced_topic_hit_rate: float
    group_score_cells_per_topic_event: float
    fine_score_cells_per_topic_event: float
    score_cells_per_topic_event: float
    score_cells_per_event: float


@dataclass(frozen=True)
class SyntheticLMHierarchicalCandidateReducerPoint:
    """One group-summary candidate reducer point with content-gate cost."""

    reducer_rows: int
    group_size: int
    selected_groups: int
    candidate_rows_scored: int
    total_groups: int
    content_rows: int
    candidate_score_source: str
    base_topic_hit_rate: float
    reduced_topic_hit_rate: float
    hit_retention_rate: float
    mean_demand_fraction: float
    group_score_cells_per_topic_event: float
    fine_score_cells_per_topic_event: float
    score_cells_per_topic_event: float
    score_cells_per_event: float
    score_cell_reduction_rate: float
    phase_lut_state_bytes: float
    phase_lut_write_state_count: int
    phase_channel_writes_per_event: float
    phase_demand_exact_rate: float
    phase_demand_mean_abs_error: float


@dataclass(frozen=True)
class SyntheticLMHierarchicalCandidateReducerResult:
    """Group-summary candidate reducer plus exact content-gate sweep."""

    fact_count: int
    candidate_pool_size: int
    base_top_k: int
    topic_events: int
    query_events: int
    total_events: int
    bits: int
    train_seed: int
    eval_seed: int
    write_cost: float
    points: Tuple[SyntheticLMHierarchicalCandidateReducerPoint, ...]


@dataclass(frozen=True)
class SyntheticLMPhasedDemandGateLUT:
    """Synthetic LM LUT with exact/candidate phase and candidate-rank buckets."""

    writes: Tuple[bool, ...]
    phase_count: int = 3
    rank_thresholds: Tuple[int, ...] = (1, 4, 8, 16, 32)
    mismatch_thresholds: Tuple[int, ...] = (1, 4, 8)
    route_thresholds: Tuple[int, ...] = ()
    envelope_thresholds: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        expected = (
            self.phase_count
            * (len(self.rank_thresholds) + 1)
            * (len(self.mismatch_thresholds) + 1)
            * (len(self.route_thresholds) + 1)
            * (len(self.envelope_thresholds) + 1)
        )
        if len(self.writes) != expected:
            raise ValueError("write table length does not match phased demand dimensions")

    @property
    def state_bytes(self) -> float:
        return len(self.writes) / 8

    @property
    def write_state_count(self) -> int:
        return sum(1 for value in self.writes if value)

    def indices(
        self,
        phase: np.ndarray,
        rank: np.ndarray,
        mismatch: np.ndarray,
        route: np.ndarray,
        envelope: np.ndarray,
    ) -> np.ndarray:
        phase_bucket = np.clip(phase.astype(np.int64), 0, self.phase_count - 1)
        rank_bucket = np.searchsorted(self.rank_thresholds, rank, side="right")
        mismatch_bucket = np.searchsorted(self.mismatch_thresholds, mismatch, side="right")
        route_bucket = np.searchsorted(self.route_thresholds, route, side="right")
        envelope_bucket = np.searchsorted(self.envelope_thresholds, envelope, side="right")
        rank_bucket_count = len(self.rank_thresholds) + 1
        mismatch_bucket_count = len(self.mismatch_thresholds) + 1
        route_bucket_count = len(self.route_thresholds) + 1
        envelope_bucket_count = len(self.envelope_thresholds) + 1
        return (
            (
                (
                    (phase_bucket * rank_bucket_count + rank_bucket)
                    * mismatch_bucket_count
                    + mismatch_bucket
                )
                * route_bucket_count
                + route_bucket
            )
            * envelope_bucket_count
            + envelope_bucket
        )

    def gate_mask(
        self,
        phase: np.ndarray,
        rank: np.ndarray,
        mismatch: np.ndarray,
        route: np.ndarray,
        envelope: np.ndarray,
    ) -> np.ndarray:
        table = np.array(self.writes, dtype=np.bool_)
        return table[self.indices(phase, rank, mismatch, route, envelope)]


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


def make_mixed_exact_candidate_demand_trace(
    config: SyntheticLMConfig,
    seed: int,
    candidate_rows: int | None = None,
) -> Tuple[np.ndarray, ...]:
    """Demand exact fact rows on query events and candidate rows on topic events."""

    rows = config.topic_top_k if candidate_rows is None else int(candidate_rows)
    if rows <= 0:
        raise ValueError("candidate_rows must be positive")
    rows = min(rows, config.candidate_pool_size)

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

    candidate_offset = config.fact_count
    candidate_indices = np.arange(
        candidate_offset,
        candidate_offset + rows,
        dtype=np.int32,
    )
    trace = []
    query_cursor = 0
    for event_type in event_types:
        if event_type == "query":
            trace.append(np.array([int(query_indices[query_cursor])], dtype=np.int32))
            query_cursor += 1
        else:
            trace.append(candidate_indices)
    return tuple(trace)


def _blank_phased_demand_lut() -> SyntheticLMPhasedDemandGateLUT:
    phase_count = 3
    rank_thresholds = (1, 4, 8, 16, 32)
    mismatch_thresholds = (1, 4, 8)
    route_thresholds: Tuple[int, ...] = ()
    envelope_thresholds: Tuple[int, ...] = ()
    size = (
        phase_count
        * (len(rank_thresholds) + 1)
        * (len(mismatch_thresholds) + 1)
        * (len(route_thresholds) + 1)
        * (len(envelope_thresholds) + 1)
    )
    return SyntheticLMPhasedDemandGateLUT(
        writes=tuple(False for _ in range(size)),
        phase_count=phase_count,
        rank_thresholds=rank_thresholds,
        mismatch_thresholds=mismatch_thresholds,
        route_thresholds=route_thresholds,
        envelope_thresholds=envelope_thresholds,
    )


def _demand_candidate_rank_vector(
    demanded: np.ndarray,
    length: int,
    fact_count: int,
) -> np.ndarray:
    rank = np.zeros(length, dtype=np.int16)
    if len(demanded) == 0:
        return rank
    indices = np.asarray(demanded, dtype=np.int64)
    candidate_indices = indices[(indices >= fact_count) & (indices < length)]
    for candidate_rank, index in enumerate(candidate_indices):
        rank[int(index)] = int(candidate_rank)
    return rank


def _demand_phase_vector(
    demanded: np.ndarray,
    length: int,
    fact_count: int,
) -> np.ndarray:
    phase = np.zeros(length, dtype=np.uint8)
    if len(demanded) == 0:
        return phase
    indices = np.asarray(demanded, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < length)]
    if len(indices) == 0:
        return phase
    exact_indices = indices[indices < fact_count]
    candidate_indices = indices[indices >= fact_count]
    phase[exact_indices] = 1
    phase[candidate_indices] = 2
    return phase


def train_synthetic_lm_phased_demand_gate_lut(
    demand_trace: Tuple[np.ndarray, ...],
    fact_count: int,
    candidate_rows: int,
    topology: str = "harc",
    bits: int = 4,
    radius: int = 1,
    seed: int = 911,
    write_cost: float = 0.15,
) -> SyntheticLMPhasedDemandGateLUT:
    """Train a phase/rank-aware content gate for synthetic LM demand."""

    if fact_count <= 0:
        raise ValueError("fact_count must be positive")
    if candidate_rows <= 0:
        raise ValueError("candidate_rows must be positive")
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, 8")
    if len(demand_trace) == 0:
        raise ValueError("demand_trace must not be empty")
    if write_cost < 0.0:
        raise ValueError("write_cost must be non-negative")

    length = fact_count + candidate_rows
    edges = _graph_for_topology(topology, length, radius)
    nodes = _ordered_nodes(edges)
    node_index = {node: index for index, node in enumerate(nodes)}
    neighbors = _neighbor_indices(edges, nodes)
    token_indices = np.array([node_index[(0, index)] for index in range(length)], dtype=np.int32)
    max_value = (1 << bits) - 1
    content_rng = np.random.default_rng(seed)

    content_values = np.zeros(len(nodes), dtype=np.uint8)
    content_values[token_indices] = content_rng.integers(
        0,
        max_value + 1,
        size=length,
        dtype=np.uint8,
    )
    carrier = np.zeros((len(nodes), 3), dtype=np.uint8)
    _inject_content_into_carrier(carrier, content_values, token_indices)

    template = _blank_phased_demand_lut()
    benefit_sums = np.zeros(len(template.writes), dtype=np.float64)
    counts = np.zeros(len(template.writes), dtype=np.int64)

    for demanded in demand_trace:
        carrier = _step_dynamic_state(
            state=carrier,
            rule="mhc_grouped",
            neighbors=neighbors,
            max_value=max_value,
            source_index=None,
        )
        phase = _demand_phase_vector(demanded, length, fact_count)
        rank = _demand_candidate_rank_vector(demanded, length, fact_count)
        carrier_values = carrier[token_indices, 0].astype(np.int16)
        content = content_values[token_indices].astype(np.int16)
        mismatch = np.abs(carrier_values - content)
        route = carrier[token_indices, 1].astype(np.int16)
        envelope = carrier[token_indices, 2].astype(np.int16)
        indices = template.indices(phase, rank, mismatch, route, envelope)
        benefit = (
            (phase > 0).astype(np.float64)
            * (mismatch > 0).astype(np.float64)
        )
        np.add.at(benefit_sums, indices, benefit)
        np.add.at(counts, indices, 1)

    mean_benefit = np.divide(
        benefit_sums,
        counts,
        out=np.zeros_like(benefit_sums),
        where=counts > 0,
    )
    writes = tuple(
        bool(count > 0 and value >= write_cost)
        for count, value in zip(counts, mean_benefit)
    )
    return SyntheticLMPhasedDemandGateLUT(
        writes=writes,
        phase_count=template.phase_count,
        rank_thresholds=template.rank_thresholds,
        mismatch_thresholds=template.mismatch_thresholds,
        route_thresholds=template.route_thresholds,
        envelope_thresholds=template.envelope_thresholds,
    )


def evaluate_synthetic_lm_phased_demand_gate(
    lut: SyntheticLMPhasedDemandGateLUT,
    demand_trace: Tuple[np.ndarray, ...],
    fact_count: int,
    candidate_rows: int,
    topology: str = "harc",
    bits: int = 4,
    radius: int = 1,
    seed: int = 911,
    policy: str = "learned_phase_rank_exact_lut",
) -> DemandContentGatePoint:
    """Evaluate a phase/rank-aware synthetic LM content gate."""

    if fact_count <= 0:
        raise ValueError("fact_count must be positive")
    if candidate_rows <= 0:
        raise ValueError("candidate_rows must be positive")
    if bits not in (2, 4, 8):
        raise ValueError("bits must be one of 2, 4, 8")
    if len(demand_trace) == 0:
        raise ValueError("demand_trace must not be empty")

    length = fact_count + candidate_rows
    ticks = len(demand_trace)
    edges = _graph_for_topology(topology, length, radius)
    nodes = _ordered_nodes(edges)
    node_index = {node: index for index, node in enumerate(nodes)}
    neighbors = _neighbor_indices(edges, nodes)
    token_indices = np.array([node_index[(0, index)] for index in range(length)], dtype=np.int32)
    max_value = (1 << bits) - 1
    content_rng = np.random.default_rng(seed)

    content_values = np.zeros(len(nodes), dtype=np.uint8)
    content_values[token_indices] = content_rng.integers(
        0,
        max_value + 1,
        size=length,
        dtype=np.uint8,
    )
    initial_token_content = content_values[token_indices].copy()
    carrier = np.zeros((len(nodes), 3), dtype=np.uint8)
    _inject_content_into_carrier(carrier, content_values, token_indices)

    gate_token_writes = 0
    demand_token_count = 0
    demand_exact_count = 0.0
    demand_error_sum = 0.0
    carrier_exact_sum = 0.0
    carrier_error_sum = 0.0

    for demanded in demand_trace:
        carrier = _step_dynamic_state(
            state=carrier,
            rule="mhc_grouped",
            neighbors=neighbors,
            max_value=max_value,
            source_index=None,
        )
        phase = _demand_phase_vector(demanded, length, fact_count)
        rank = _demand_candidate_rank_vector(demanded, length, fact_count)
        demand_mask = _demand_indices_to_mask(demanded, length)
        carrier_values = carrier[token_indices, 0].astype(np.int16)
        content = content_values[token_indices].astype(np.int16)
        mismatch = np.abs(carrier_values - content)
        route = carrier[token_indices, 1].astype(np.int16)
        envelope = carrier[token_indices, 2].astype(np.int16)
        selected = token_indices[lut.gate_mask(phase, rank, mismatch, route, envelope)]
        if len(selected) > 0:
            _inject_content_into_carrier(carrier, content_values, selected)
            gate_token_writes += int(len(selected))

        carrier_token_content = carrier[token_indices, 0]
        carrier_error = np.abs(
            carrier_token_content.astype(np.int16)
            - initial_token_content.astype(np.int16)
        )
        carrier_exact_sum += float(np.mean(carrier_token_content == initial_token_content))
        carrier_error_sum += float(np.mean(carrier_error) / float(max_value))

        if bool(np.any(demand_mask)):
            demand_values = carrier_token_content[demand_mask]
            target_values = initial_token_content[demand_mask]
            demand_errors = np.abs(
                demand_values.astype(np.int16) - target_values.astype(np.int16)
            )
            demand_token_count += int(np.count_nonzero(demand_mask))
            demand_exact_count += float(np.count_nonzero(demand_values == target_values))
            demand_error_sum += float(np.sum(demand_errors) / float(max_value))

    carrier_token_content = carrier[token_indices, 0]
    carrier_error = np.abs(
        carrier_token_content.astype(np.int16) - initial_token_content.astype(np.int16)
    )
    demand_denominator = demand_token_count if demand_token_count else 1
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

    return DemandContentGatePoint(
        policy=policy,
        length=length,
        topology=topology.lower(),
        bits=bits,
        ticks=ticks,
        demand_rate=demand_token_count / float(length * ticks),
        seed=seed,
        state_bits_per_token=4 * bits,
        gate_token_writes=gate_token_writes,
        gate_channel_writes_per_token_tick=3 * gate_token_writes / float(length * ticks),
        demand_token_count=demand_token_count,
        mean_demand_fraction=demand_token_count / float(length * ticks),
        demand_exact_rate=demand_exact_count / float(demand_denominator),
        demand_mean_abs_error=demand_error_sum / float(demand_denominator),
        carrier_exact_retention_rate=float(np.mean(carrier_token_content == initial_token_content)),
        mean_carrier_exact_retention_rate=carrier_exact_sum / float(ticks),
        carrier_mean_abs_error=float(np.mean(carrier_error) / float(max_value)),
        mean_carrier_mean_abs_error=carrier_error_sum / float(ticks),
        carrier_final_entropy_bits=_level_entropy_bits(carrier, max_value),
        carrier_final_saturation_fraction=float(np.count_nonzero(carrier == max_value))
        / float(carrier.size),
        checksum=int(checksum),
    )


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

    def score_static_candidates(self) -> tuple[np.ndarray, int]:
        """Score the static candidate pool using the current low-bit scorer."""

        if self.config.candidate_strategy != "static":
            raise ValueError("score_static_candidates requires static candidates")
        if self.candidates is None or self.candidate_slots is None:
            raise RuntimeError("static candidates are not initialized")

        cache_scores = np.zeros(len(self.candidates), dtype=np.int32)
        bank_indices = np.arange(self.config.dense_banks)[:, None]
        return self._candidate_scores(
            candidate_slots=self.candidate_slots,
            cache_scores=cache_scores,
            bank_indices=bank_indices,
        )

    def order_static_candidate_scores(self, scores: np.ndarray) -> np.ndarray:
        """Return descending static candidate indices for already-computed scores."""

        if self.candidates is None:
            raise RuntimeError("static candidates are not initialized")
        index_tiebreaker = np.arange(len(self.candidates), dtype=np.int32)
        if self.config.candidate_scorer_lut is not None:
            cache_scores = np.zeros(len(self.candidates), dtype=np.int32)
            learned_scores = np.empty(len(self.candidates), dtype=np.int32)
            for index, (dense_score, cache_score) in enumerate(zip(scores, cache_scores)):
                dense_index = min(int(dense_score), self.config.candidate_scorer_dense_bins - 1)
                cache_index = min(int(cache_score), self.config.candidate_scorer_cache_bins - 1)
                lut_index = dense_index * self.config.candidate_scorer_cache_bins + cache_index
                learned_scores[index] = (
                    self.config.candidate_scorer_dense_weight * int(dense_score)
                    + self.config.candidate_scorer_cache_weight * int(cache_score)
                    + int(self.config.candidate_scorer_lut[lut_index])
                )
            order = np.lexsort((index_tiebreaker, scores, learned_scores))[::-1]
        else:
            order = np.lexsort((index_tiebreaker, scores))[::-1]
        return order.astype(np.int32)

    def rank_static_candidate_indices(self) -> tuple[np.ndarray, int]:
        """Rank the static candidate pool using the current low-bit scorer."""

        scores, score_cells = self.score_static_candidates()
        return self.order_static_candidate_scores(scores), score_cells

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
            order, score_cells = self.rank_static_candidate_indices()
            top_indices = order[: self.config.topic_top_k]
            return {int(self.candidates[index]) for index in top_indices}, score_cells

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


def make_lowbit_candidate_reducer_demand_trace(
    config: SyntheticLMConfig | None = None,
    reducer_rows: int = 16,
    seed: int = 37,
    base_top_k: int | None = None,
) -> SyntheticLMCandidateReducerTrace:
    """Build a mixed exact+candidate trace using low-bit top-M candidate reduction."""

    trace_config = config or SyntheticLMConfig(
        fact_count=512,
        topic_events=512,
        query_events=256,
        dense_width=512,
        primary_buckets=512,
        overflow_buckets=128,
        candidate_score_source="topic_phase",
    )
    if trace_config.candidate_strategy != "static":
        raise ValueError("candidate reducer trace currently requires static candidates")
    rows = min(max(1, int(reducer_rows)), trace_config.candidate_pool_size)
    top_k = trace_config.topic_top_k if base_top_k is None else int(base_top_k)
    top_k = min(max(1, top_k), trace_config.candidate_pool_size)

    lm = DualPathSyntheticLM(trace_config, seed=seed)
    lm.prefill()
    if lm.candidates is None:
        raise RuntimeError("static candidates are not initialized")

    query_indices = lm.rng.choice(
        len(lm.facts),
        size=trace_config.query_events,
        replace=True,
    )
    event_types = np.array(
        ["topic"] * trace_config.topic_events + ["query"] * trace_config.query_events
    )
    lm.rng.shuffle(event_types)

    trace = []
    query_cursor = 0
    base_topic_hits = 0
    reduced_topic_hits = 0
    score_cells = 0
    for event_type in event_types:
        if event_type == "topic":
            token = sample_topic_token(trace_config, lm.rng)
            order, touched = lm.rank_static_candidate_indices()
            score_cells += touched
            base_indices = order[:top_k]
            reduced_indices = order[:rows]
            base_topic_hits += int(token in set(int(lm.candidates[index]) for index in base_indices))
            reduced_topic_hits += int(
                token in set(int(lm.candidates[index]) for index in reduced_indices)
            )
            trace.append((trace_config.fact_count + reduced_indices).astype(np.int32))
            lm.dense.update(token)
            if lm.candidate_score_dense is not None:
                lm.candidate_score_dense.update(token)
            continue

        fact_index = int(query_indices[query_cursor])
        query_cursor += 1
        key, expected = lm.facts[fact_index]
        trace.append(np.array([fact_index], dtype=np.int32))
        lm.dense.update(key)
        lm.dense.update(expected)

    total_events = trace_config.topic_events + trace_config.query_events
    return SyntheticLMCandidateReducerTrace(
        demand_trace=tuple(trace),
        fact_count=trace_config.fact_count,
        candidate_pool_size=trace_config.candidate_pool_size,
        content_rows=trace_config.fact_count + trace_config.candidate_pool_size,
        reducer_rows=rows,
        base_top_k=top_k,
        topic_events=trace_config.topic_events,
        query_events=trace_config.query_events,
        total_events=total_events,
        candidate_score_source=trace_config.candidate_score_source,
        base_topic_hit_rate=base_topic_hits / float(trace_config.topic_events),
        reduced_topic_hit_rate=reduced_topic_hits / float(trace_config.topic_events),
        score_cells_per_topic_event=score_cells / float(trace_config.topic_events),
        score_cells_per_event=score_cells / float(total_events),
    )


def _hierarchical_candidate_reducer_indices(
    scores: np.ndarray,
    reducer_rows: int,
    group_size: int,
    selected_groups: int,
) -> tuple[np.ndarray, int, int]:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if selected_groups <= 0:
        raise ValueError("selected_groups must be positive")

    candidate_count = len(scores)
    group_count = (candidate_count + group_size - 1) // group_size
    group_scores = np.full(group_count, np.iinfo(np.int32).min, dtype=np.int32)
    for group in range(group_count):
        start = group * group_size
        end = min(start + group_size, candidate_count)
        group_scores[group] = int(np.max(scores[start:end]))

    group_tiebreaker = np.arange(group_count, dtype=np.int32)
    group_order = np.lexsort((group_tiebreaker, group_scores))[::-1]
    chosen_groups = group_order[: min(selected_groups, group_count)]
    selected = []
    for group in chosen_groups:
        start = int(group) * group_size
        end = min(start + group_size, candidate_count)
        selected.extend(range(start, end))
    selected_indices = np.array(selected, dtype=np.int32)
    selected_tiebreaker = selected_indices
    selected_order = np.lexsort((selected_tiebreaker, scores[selected_indices]))[::-1]
    reduced = selected_indices[selected_order[: min(reducer_rows, len(selected_indices))]]
    return reduced.astype(np.int32), group_count, int(len(selected_indices))


def make_hierarchical_candidate_reducer_demand_trace(
    config: SyntheticLMConfig | None = None,
    reducer_rows: int = 16,
    group_size: int = 16,
    selected_groups: int = 4,
    seed: int = 37,
    base_top_k: int | None = None,
) -> SyntheticLMHierarchicalCandidateReducerTrace:
    """Build a trace using group-summary selection before candidate scoring."""

    trace_config = config or SyntheticLMConfig(
        fact_count=512,
        topic_events=512,
        query_events=256,
        dense_width=512,
        primary_buckets=512,
        overflow_buckets=128,
        candidate_score_source="topic_phase",
    )
    if trace_config.candidate_strategy != "static":
        raise ValueError("hierarchical reducer trace currently requires static candidates")
    rows = min(max(1, int(reducer_rows)), trace_config.candidate_pool_size)
    top_k = trace_config.topic_top_k if base_top_k is None else int(base_top_k)
    top_k = min(max(1, top_k), trace_config.candidate_pool_size)
    clean_group_size = max(1, int(group_size))
    clean_selected_groups = max(1, int(selected_groups))

    lm = DualPathSyntheticLM(trace_config, seed=seed)
    lm.prefill()
    if lm.candidates is None:
        raise RuntimeError("static candidates are not initialized")

    query_indices = lm.rng.choice(
        len(lm.facts),
        size=trace_config.query_events,
        replace=True,
    )
    event_types = np.array(
        ["topic"] * trace_config.topic_events + ["query"] * trace_config.query_events
    )
    lm.rng.shuffle(event_types)

    trace = []
    query_cursor = 0
    base_topic_hits = 0
    reduced_topic_hits = 0
    group_score_cells = 0.0
    fine_score_cells = 0.0
    total_groups = 0
    for event_type in event_types:
        if event_type == "topic":
            token = sample_topic_token(trace_config, lm.rng)
            scores, full_score_cells = lm.score_static_candidates()
            full_order = lm.order_static_candidate_scores(scores)
            reduced_indices, group_count, selected_candidate_count = (
                _hierarchical_candidate_reducer_indices(
                    scores=scores,
                    reducer_rows=rows,
                    group_size=clean_group_size,
                    selected_groups=clean_selected_groups,
                )
            )
            total_groups = group_count
            score_cells_per_candidate = full_score_cells / float(trace_config.candidate_pool_size)
            group_score_cells += group_count * score_cells_per_candidate
            fine_score_cells += selected_candidate_count * score_cells_per_candidate
            base_indices = full_order[:top_k]
            base_topic_hits += int(bool(np.any(lm.candidates[base_indices] == token)))
            reduced_topic_hits += int(bool(np.any(lm.candidates[reduced_indices] == token)))
            trace.append((trace_config.fact_count + reduced_indices).astype(np.int32))
            lm.dense.update(token)
            if lm.candidate_score_dense is not None:
                lm.candidate_score_dense.update(token)
            continue

        fact_index = int(query_indices[query_cursor])
        query_cursor += 1
        key, expected = lm.facts[fact_index]
        trace.append(np.array([fact_index], dtype=np.int32))
        lm.dense.update(key)
        lm.dense.update(expected)

    total_events = trace_config.topic_events + trace_config.query_events
    score_cells = group_score_cells + fine_score_cells
    return SyntheticLMHierarchicalCandidateReducerTrace(
        demand_trace=tuple(trace),
        fact_count=trace_config.fact_count,
        candidate_pool_size=trace_config.candidate_pool_size,
        content_rows=trace_config.fact_count + trace_config.candidate_pool_size,
        reducer_rows=rows,
        base_top_k=top_k,
        group_size=clean_group_size,
        selected_groups=clean_selected_groups,
        total_groups=total_groups,
        topic_events=trace_config.topic_events,
        query_events=trace_config.query_events,
        total_events=total_events,
        candidate_score_source=trace_config.candidate_score_source,
        base_topic_hit_rate=base_topic_hits / float(trace_config.topic_events),
        reduced_topic_hit_rate=reduced_topic_hits / float(trace_config.topic_events),
        group_score_cells_per_topic_event=group_score_cells / float(trace_config.topic_events),
        fine_score_cells_per_topic_event=fine_score_cells / float(trace_config.topic_events),
        score_cells_per_topic_event=score_cells / float(trace_config.topic_events),
        score_cells_per_event=score_cells / float(total_events),
    )


def run_synthetic_lm_trial(seed: int = 0, config: SyntheticLMConfig | None = None) -> SyntheticLMResult:
    """Convenience wrapper for one synthetic dual-path LM trial."""

    lm = DualPathSyntheticLM(config or SyntheticLMConfig(), seed=seed)
    return lm.run()


def run_synthetic_lm_demand_gate_sweep(
    config: SyntheticLMConfig | None = None,
    demand_trace: str = "exact_query",
    candidate_rows: int | None = None,
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
    """Train/evaluate content gates on synthetic event demand."""

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
    demand_trace = str(demand_trace).lower()
    if demand_trace == "exact_query":
        train_trace = make_exact_query_demand_trace(gate_config, seed=train_seed)
        eval_trace = make_exact_query_demand_trace(gate_config, seed=eval_seed)
        content_rows = gate_config.fact_count
        candidate_row_count = 0
    elif demand_trace == "mixed_candidate_topk":
        candidate_row_count = (
            gate_config.topic_top_k if candidate_rows is None else int(candidate_rows)
        )
        candidate_row_count = min(candidate_row_count, gate_config.candidate_pool_size)
        train_trace = make_mixed_exact_candidate_demand_trace(
            gate_config,
            seed=train_seed,
            candidate_rows=candidate_row_count,
        )
        eval_trace = make_mixed_exact_candidate_demand_trace(
            gate_config,
            seed=eval_seed,
            candidate_rows=candidate_row_count,
        )
        content_rows = gate_config.fact_count + candidate_row_count
    else:
        raise ValueError("demand_trace must be one of: exact_query, mixed_candidate_topk")

    lut = train_trace_demand_content_gate_lut(
        demand_trace=train_trace,
        length=content_rows,
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
                length=content_rows,
                bits=bits,
                seed=eval_seed + 8192,
            )
        )
    points.append(
        evaluate_lut_trace_demand_content_gate(
            lut=lut,
            demand_trace=eval_trace,
            length=content_rows,
            bits=bits,
            seed=eval_seed + 8192,
            policy=f"learned_{demand_trace}_lut_c{write_cost:0.2f}",
        )
    )

    return SyntheticLMDemandGateResult(
        demand_trace=demand_trace,
        fact_count=gate_config.fact_count,
        candidate_rows=candidate_row_count,
        topic_events=gate_config.topic_events,
        query_events=gate_config.query_events,
        total_events=gate_config.topic_events + gate_config.query_events,
        content_rows=content_rows,
        bits=bits,
        train_seed=train_seed,
        eval_seed=eval_seed,
        write_cost=write_cost,
        lut_state_bytes=lut.state_bytes,
        lut_write_state_count=lut.write_state_count,
        lut=lut,
        points=tuple(points),
    )


def run_synthetic_lm_candidate_demand_sparsity_sweep(
    config: SyntheticLMConfig | None = None,
    candidate_rows: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    bits: int = 4,
    train_seed: int = 31,
    eval_seed: int = 37,
    write_cost: float = 0.15,
) -> SyntheticLMCandidateDemandSweepResult:
    """Measure how candidate-output demand density affects content-gate writes."""

    gate_config = config or SyntheticLMConfig(
        fact_count=512,
        topic_events=512,
        query_events=256,
        dense_width=512,
        primary_buckets=512,
        overflow_buckets=128,
    )
    row_counts = tuple(
        dict.fromkeys(
            min(int(rows), gate_config.candidate_pool_size)
            for rows in candidate_rows
            if int(rows) > 0
        )
    )
    if len(row_counts) == 0:
        raise ValueError("candidate_rows must contain at least one positive value")

    points = []
    for rows in row_counts:
        result = run_synthetic_lm_demand_gate_sweep(
            config=gate_config,
            demand_trace="mixed_candidate_topk",
            candidate_rows=rows,
            policies=("fixed_refresh16", "demand_mismatch_ge1"),
            bits=bits,
            train_seed=train_seed,
            eval_seed=eval_seed,
            write_cost=write_cost,
        )
        fixed_refresh = next(
            point for point in result.points if point.policy == "fixed_refresh16"
        )
        demand_mismatch = next(
            point for point in result.points if point.policy == "demand_mismatch_ge1"
        )
        learned = result.points[-1]
        train_trace = make_mixed_exact_candidate_demand_trace(
            gate_config,
            seed=train_seed,
            candidate_rows=rows,
        )
        eval_trace = make_mixed_exact_candidate_demand_trace(
            gate_config,
            seed=eval_seed,
            candidate_rows=rows,
        )
        phase_lut = train_synthetic_lm_phased_demand_gate_lut(
            demand_trace=train_trace,
            fact_count=gate_config.fact_count,
            candidate_rows=rows,
            bits=bits,
            seed=train_seed + 8192,
            write_cost=write_cost,
        )
        phase_point = evaluate_synthetic_lm_phased_demand_gate(
            lut=phase_lut,
            demand_trace=eval_trace,
            fact_count=gate_config.fact_count,
            candidate_rows=rows,
            bits=bits,
            seed=eval_seed + 8192,
            policy=f"learned_phase_rank_exact_lut_c{write_cost:0.2f}",
        )
        points.append(
            SyntheticLMCandidateDemandSweepPoint(
                candidate_rows=result.candidate_rows,
                content_rows=result.content_rows,
                mean_demand_fraction=learned.mean_demand_fraction,
                fixed_refresh_writes_per_token_tick=(
                    fixed_refresh.gate_channel_writes_per_token_tick
                ),
                fixed_refresh_demand_exact_rate=fixed_refresh.demand_exact_rate,
                demand_mismatch_writes_per_token_tick=(
                    demand_mismatch.gate_channel_writes_per_token_tick
                ),
                demand_mismatch_demand_exact_rate=demand_mismatch.demand_exact_rate,
                learned_writes_per_token_tick=(
                    learned.gate_channel_writes_per_token_tick
                ),
                learned_demand_exact_rate=learned.demand_exact_rate,
                learned_demand_mean_abs_error=learned.demand_mean_abs_error,
                lut_state_bytes=result.lut_state_bytes,
                lut_write_state_count=result.lut_write_state_count,
                phase_lut_state_bytes=phase_lut.state_bytes,
                phase_lut_write_state_count=phase_lut.write_state_count,
                phase_writes_per_token_tick=(
                    phase_point.gate_channel_writes_per_token_tick
                ),
                phase_demand_exact_rate=phase_point.demand_exact_rate,
                phase_demand_mean_abs_error=phase_point.demand_mean_abs_error,
            )
        )

    return SyntheticLMCandidateDemandSweepResult(
        fact_count=gate_config.fact_count,
        topic_events=gate_config.topic_events,
        query_events=gate_config.query_events,
        total_events=gate_config.topic_events + gate_config.query_events,
        bits=bits,
        train_seed=train_seed,
        eval_seed=eval_seed,
        write_cost=write_cost,
        points=tuple(points),
    )


def run_synthetic_lm_candidate_reducer_sweep(
    config: SyntheticLMConfig | None = None,
    reducer_rows: Tuple[int, ...] = (8, 16, 32, 64),
    bits: int = 4,
    train_seed: int = 31,
    eval_seed: int = 37,
    write_cost: float = 0.15,
) -> SyntheticLMCandidateReducerResult:
    """Evaluate low-bit candidate reduction before exact content exposure."""

    reducer_config = config or SyntheticLMConfig(
        fact_count=512,
        topic_events=512,
        query_events=256,
        dense_width=512,
        primary_buckets=512,
        overflow_buckets=128,
        candidate_score_source="topic_phase",
    )
    row_counts = tuple(
        dict.fromkeys(
            min(int(rows), reducer_config.candidate_pool_size)
            for rows in reducer_rows
            if int(rows) > 0
        )
    )
    if len(row_counts) == 0:
        raise ValueError("reducer_rows must contain at least one positive value")

    points = []
    base_top_k = reducer_config.topic_top_k
    for rows in row_counts:
        train_trace = make_lowbit_candidate_reducer_demand_trace(
            config=reducer_config,
            reducer_rows=rows,
            seed=train_seed,
            base_top_k=base_top_k,
        )
        eval_trace = make_lowbit_candidate_reducer_demand_trace(
            config=reducer_config,
            reducer_rows=rows,
            seed=eval_seed,
            base_top_k=base_top_k,
        )
        phase_lut = train_synthetic_lm_phased_demand_gate_lut(
            demand_trace=train_trace.demand_trace,
            fact_count=reducer_config.fact_count,
            candidate_rows=reducer_config.candidate_pool_size,
            bits=bits,
            seed=train_seed + 8192,
            write_cost=write_cost,
        )
        phase_point = evaluate_synthetic_lm_phased_demand_gate(
            lut=phase_lut,
            demand_trace=eval_trace.demand_trace,
            fact_count=reducer_config.fact_count,
            candidate_rows=reducer_config.candidate_pool_size,
            bits=bits,
            seed=eval_seed + 8192,
            policy=f"learned_reducer_phase_rank_lut_c{write_cost:0.2f}",
        )
        hit_retention = (
            eval_trace.reduced_topic_hit_rate / eval_trace.base_topic_hit_rate
            if eval_trace.base_topic_hit_rate > 0.0
            else 0.0
        )
        points.append(
            SyntheticLMCandidateReducerPoint(
                reducer_rows=rows,
                content_rows=eval_trace.content_rows,
                candidate_score_source=eval_trace.candidate_score_source,
                base_topic_hit_rate=eval_trace.base_topic_hit_rate,
                reduced_topic_hit_rate=eval_trace.reduced_topic_hit_rate,
                hit_retention_rate=hit_retention,
                mean_demand_fraction=phase_point.mean_demand_fraction,
                score_cells_per_topic_event=eval_trace.score_cells_per_topic_event,
                score_cells_per_event=eval_trace.score_cells_per_event,
                phase_lut_state_bytes=phase_lut.state_bytes,
                phase_lut_write_state_count=phase_lut.write_state_count,
                phase_writes_per_token_tick=(
                    phase_point.gate_channel_writes_per_token_tick
                ),
                phase_channel_writes_per_event=(
                    phase_point.gate_channel_writes_per_token_tick
                    * eval_trace.content_rows
                ),
                phase_demand_exact_rate=phase_point.demand_exact_rate,
                phase_demand_mean_abs_error=phase_point.demand_mean_abs_error,
            )
        )

    return SyntheticLMCandidateReducerResult(
        fact_count=reducer_config.fact_count,
        candidate_pool_size=reducer_config.candidate_pool_size,
        base_top_k=base_top_k,
        topic_events=reducer_config.topic_events,
        query_events=reducer_config.query_events,
        total_events=reducer_config.topic_events + reducer_config.query_events,
        bits=bits,
        train_seed=train_seed,
        eval_seed=eval_seed,
        write_cost=write_cost,
        points=tuple(points),
    )


def run_synthetic_lm_hierarchical_candidate_reducer_sweep(
    config: SyntheticLMConfig | None = None,
    settings: Tuple[Tuple[int, int, int], ...] = (
        (16, 16, 2),
        (16, 16, 4),
        (32, 16, 4),
        (32, 16, 8),
    ),
    bits: int = 4,
    train_seed: int = 31,
    eval_seed: int = 37,
    write_cost: float = 0.15,
) -> SyntheticLMHierarchicalCandidateReducerResult:
    """Evaluate group-summary candidate reduction before exact content exposure."""

    reducer_config = config or SyntheticLMConfig(
        fact_count=512,
        topic_events=512,
        query_events=256,
        dense_width=512,
        primary_buckets=512,
        overflow_buckets=128,
        candidate_score_source="topic_phase",
    )
    clean_settings = tuple(
        dict.fromkeys(
            (
                min(max(1, int(rows)), reducer_config.candidate_pool_size),
                max(1, int(group_size)),
                max(1, int(selected_groups)),
            )
            for rows, group_size, selected_groups in settings
        )
    )
    if len(clean_settings) == 0:
        raise ValueError("settings must not be empty")

    points = []
    base_top_k = reducer_config.topic_top_k
    full_score_cells = reducer_config.candidate_pool_size * reducer_config.dense_banks
    if reducer_config.candidate_score_source in {"dense_topic_sum", "dense_topic_cache"}:
        full_score_cells *= 2

    for rows, group_size, selected_groups in clean_settings:
        train_trace = make_hierarchical_candidate_reducer_demand_trace(
            config=reducer_config,
            reducer_rows=rows,
            group_size=group_size,
            selected_groups=selected_groups,
            seed=train_seed,
            base_top_k=base_top_k,
        )
        eval_trace = make_hierarchical_candidate_reducer_demand_trace(
            config=reducer_config,
            reducer_rows=rows,
            group_size=group_size,
            selected_groups=selected_groups,
            seed=eval_seed,
            base_top_k=base_top_k,
        )
        phase_lut = train_synthetic_lm_phased_demand_gate_lut(
            demand_trace=train_trace.demand_trace,
            fact_count=reducer_config.fact_count,
            candidate_rows=reducer_config.candidate_pool_size,
            bits=bits,
            seed=train_seed + 8192,
            write_cost=write_cost,
        )
        phase_point = evaluate_synthetic_lm_phased_demand_gate(
            lut=phase_lut,
            demand_trace=eval_trace.demand_trace,
            fact_count=reducer_config.fact_count,
            candidate_rows=reducer_config.candidate_pool_size,
            bits=bits,
            seed=eval_seed + 8192,
            policy=f"learned_hier_phase_rank_lut_c{write_cost:0.2f}",
        )
        hit_retention = (
            eval_trace.reduced_topic_hit_rate / eval_trace.base_topic_hit_rate
            if eval_trace.base_topic_hit_rate > 0.0
            else 0.0
        )
        score_reduction = 1.0 - (
            eval_trace.score_cells_per_topic_event / float(full_score_cells)
        )
        points.append(
            SyntheticLMHierarchicalCandidateReducerPoint(
                reducer_rows=rows,
                group_size=group_size,
                selected_groups=selected_groups,
                candidate_rows_scored=min(
                    reducer_config.candidate_pool_size,
                    group_size * selected_groups,
                ),
                total_groups=eval_trace.total_groups,
                content_rows=eval_trace.content_rows,
                candidate_score_source=eval_trace.candidate_score_source,
                base_topic_hit_rate=eval_trace.base_topic_hit_rate,
                reduced_topic_hit_rate=eval_trace.reduced_topic_hit_rate,
                hit_retention_rate=hit_retention,
                mean_demand_fraction=phase_point.mean_demand_fraction,
                group_score_cells_per_topic_event=(
                    eval_trace.group_score_cells_per_topic_event
                ),
                fine_score_cells_per_topic_event=(
                    eval_trace.fine_score_cells_per_topic_event
                ),
                score_cells_per_topic_event=eval_trace.score_cells_per_topic_event,
                score_cells_per_event=eval_trace.score_cells_per_event,
                score_cell_reduction_rate=score_reduction,
                phase_lut_state_bytes=phase_lut.state_bytes,
                phase_lut_write_state_count=phase_lut.write_state_count,
                phase_channel_writes_per_event=(
                    phase_point.gate_channel_writes_per_token_tick
                    * eval_trace.content_rows
                ),
                phase_demand_exact_rate=phase_point.demand_exact_rate,
                phase_demand_mean_abs_error=phase_point.demand_mean_abs_error,
            )
        )

    return SyntheticLMHierarchicalCandidateReducerResult(
        fact_count=reducer_config.fact_count,
        candidate_pool_size=reducer_config.candidate_pool_size,
        base_top_k=base_top_k,
        topic_events=reducer_config.topic_events,
        query_events=reducer_config.query_events,
        total_events=reducer_config.topic_events + reducer_config.query_events,
        bits=bits,
        train_seed=train_seed,
        eval_seed=eval_seed,
        write_cost=write_cost,
        points=tuple(points),
    )
