"""Compressed block indexing for CSA-shaped context retrieval.

DeepSeek-V4's CSA path suggests a hardware pattern that maps cleanly onto a
cellular fabric: split the context into blocks, keep a low-bit summary inside
each block cell, and route a query to only a few high-scoring blocks instead of
reading the full KV cache. This module tests that pattern without requiring a
Transformer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .dense_context import DenseContextConfig, LowBitDenseContext, exact_decayed_counts
from .retrieval import keyed_hash


@dataclass(frozen=True)
class CompressedBlockIndexConfig:
    """Configuration for a low-bit per-block summary index."""

    context_length: int = 65536
    vocab_size: int = 65536
    hot_tokens: int = 256
    topic_probability: float = 0.85
    zipf_exponent: float = 1.15
    block_size: int = 64
    selected_blocks: int = 8
    tail_blocks: int = 2
    banks: int = 4
    summary_width: int = 256
    bits: int = 4
    queries: int = 4096

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not 0 < self.hot_tokens < self.vocab_size:
            raise ValueError("hot_tokens must be in (0, vocab_size)")
        if not 0.0 <= self.topic_probability <= 1.0:
            raise ValueError("topic_probability must be in [0, 1]")
        if self.zipf_exponent <= 0.0:
            raise ValueError("zipf_exponent must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.context_length % self.block_size != 0:
            raise ValueError("context_length must be divisible by block_size")
        if self.selected_blocks <= 0:
            raise ValueError("selected_blocks must be positive")
        if self.tail_blocks < 0:
            raise ValueError("tail_blocks must be non-negative")
        if self.selected_blocks + self.tail_blocks > self.blocks:
            raise ValueError("selected_blocks + tail_blocks cannot exceed block count")
        if self.banks <= 0:
            raise ValueError("banks must be positive")
        if self.summary_width <= 0:
            raise ValueError("summary_width must be positive")
        if self.bits not in (2, 4, 8):
            raise ValueError("bits must be one of 2, 4, 8")
        if self.queries <= 0:
            raise ValueError("queries must be positive")

    @property
    def blocks(self) -> int:
        return self.context_length // self.block_size

    @property
    def max_value(self) -> int:
        return (1 << self.bits) - 1

    @property
    def summary_state_bytes(self) -> float:
        return self.blocks * self.banks * self.summary_width * self.bits / 8

    @property
    def global_state_bytes(self) -> float:
        return self.banks * self.summary_width * self.bits / 8

    @property
    def score_cells_per_query(self) -> int:
        return self.blocks * self.banks

    @property
    def score_bytes_per_query(self) -> float:
        return self.score_cells_per_query * self.bits / 8


@dataclass(frozen=True)
class CompressedBlockIndexResult:
    """Aggregate metrics for compressed block selection."""

    context_length: int
    block_size: int
    blocks: int
    selected_blocks: int
    tail_blocks: int
    banks: int
    summary_width: int
    bits: int
    queries: int
    relevant_query_rate: float
    hot_relevant_query_rate: float
    cold_relevant_query_rate: float
    summary_state_bytes: float
    global_state_bytes: float
    score_cells_per_query: float
    score_bytes_per_query: float
    avg_update_cells_per_token: float
    index_block_hit_rate: float
    hot_index_block_hit_rate: float
    cold_index_block_hit_rate: float
    index_occurrence_coverage: float
    recent_block_hit_rate: float
    recent_occurrence_coverage: float
    combined_block_hit_rate: float
    hot_combined_block_hit_rate: float
    cold_combined_block_hit_rate: float
    combined_occurrence_coverage: float
    oracle_block_hit_rate: float
    oracle_occurrence_coverage: float
    dense_token_reads_per_query: float
    index_token_reads_per_query: float
    combined_token_reads_per_query: float

    @property
    def index_token_read_reduction(self) -> float:
        if self.index_token_reads_per_query == 0.0:
            return 0.0
        return self.dense_token_reads_per_query / self.index_token_reads_per_query

    @property
    def combined_token_read_reduction(self) -> float:
        if self.combined_token_reads_per_query == 0.0:
            return 0.0
        return self.dense_token_reads_per_query / self.combined_token_reads_per_query


@dataclass(frozen=True)
class CompressedBlockBudgetPoint:
    """One point on the sparse block-read budget curve."""

    selected_blocks: int
    tail_blocks: int
    avg_blocks_read: float
    block_hit_rate: float
    hot_block_hit_rate: float
    cold_block_hit_rate: float
    occurrence_coverage: float
    oracle_occurrence_coverage: float
    token_reads_per_query: float
    token_read_reduction: float

    @property
    def oracle_coverage_gap(self) -> float:
        return self.oracle_occurrence_coverage - self.occurrence_coverage


@dataclass(frozen=True)
class CompressedBlockBudgetSweepResult:
    """Coverage/read-budget curve for a fixed compressed block index."""

    context_length: int
    block_size: int
    blocks: int
    banks: int
    summary_width: int
    bits: int
    queries: int
    relevant_query_rate: float
    summary_state_bytes: float
    score_bytes_per_query: float
    points: Tuple[CompressedBlockBudgetPoint, ...]


@dataclass(frozen=True)
class CsaHcaPolicyPoint:
    """One low-bit global-summary threshold for CSA/HCA path selection."""

    hca_threshold: int
    csa_blocks: int
    tail_blocks: int
    hca_query_rate: float
    csa_query_rate: float
    hot_to_hca_rate: float
    cold_to_csa_rate: float
    csa_relevant_hit_rate: float
    csa_relevant_coverage: float
    policy_sparse_coverage: float
    token_reads_per_query: float
    token_read_reduction: float
    block_score_bytes_per_query: float


@dataclass(frozen=True)
class CsaHcaPolicyResult:
    """Adaptive sparse/dense path policy driven by a low-bit global summary."""

    context_length: int
    block_size: int
    blocks: int
    banks: int
    summary_width: int
    global_width: int
    bits: int
    queries: int
    relevant_query_rate: float
    block_summary_state_bytes: float
    global_summary_state_bytes: float
    global_summary_read_bytes_per_query: float
    points: Tuple[CsaHcaPolicyPoint, ...]


@dataclass(frozen=True)
class HcaSummaryQualityPoint:
    """One global-summary width in the HCA-like quality sweep."""

    global_width: int
    state_bytes: float
    read_bytes_per_query: float
    saturation_rate: float
    clipped_mean_abs_error: float
    top64_recall: float
    top256_recall: float
    threshold_precision: float
    threshold_recall: float
    query_route_accuracy: float
    query_false_hca_rate: float
    query_missed_hca_rate: float


@dataclass(frozen=True)
class HcaSummaryQualityResult:
    """Quality diagnostics for the HCA-like global low-bit summary."""

    context_length: int
    vocab_size: int
    hot_tokens: int
    bits: int
    threshold: int
    queries: int
    points: Tuple[HcaSummaryQualityPoint, ...]


@dataclass(frozen=True)
class HcaDecayQualityPoint:
    """One decay interval for the HCA-like global summary."""

    decay_interval: int
    state_bytes: float
    read_bytes_per_query: float
    avg_decay_cells_per_token: float
    saturation_rate: float
    clipped_mean_abs_error: float
    top64_recall: float
    top256_recall: float
    threshold_precision: float
    threshold_recall: float
    query_route_accuracy: float
    query_false_hca_rate: float
    query_missed_hca_rate: float


@dataclass(frozen=True)
class HcaDecayQualityResult:
    """Anti-saturation diagnostics for decayed HCA-like global summaries."""

    context_length: int
    vocab_size: int
    hot_tokens: int
    global_width: int
    bits: int
    threshold: int
    queries: int
    points: Tuple[HcaDecayQualityPoint, ...]


class LowBitCompressedBlockIndex:
    """Per-block low-bit count-min summaries for candidate block routing."""

    def __init__(self, config: CompressedBlockIndexConfig) -> None:
        self.config = config
        self.summaries = np.zeros(
            (config.blocks, config.banks, config.summary_width),
            dtype=np.uint8,
        )
        self.updates = 0

    def _slots(self, token: int) -> list[int]:
        return [
            keyed_hash(int(token), 12000 + bank) % self.config.summary_width
            for bank in range(self.config.banks)
        ]

    def update(self, token: int, position: int) -> int:
        """Insert one token into its block summary and return touched cells."""

        if not 0 <= int(token) < self.config.vocab_size:
            raise ValueError("token outside vocab")
        if not 0 <= int(position) < self.config.context_length:
            raise ValueError("position outside context")
        block = int(position) // self.config.block_size
        for bank, slot in enumerate(self._slots(token)):
            value = int(self.summaries[block, bank, slot])
            if value < self.config.max_value:
                self.summaries[block, bank, slot] = value + 1
        self.updates += 1
        return self.config.banks

    def estimate_blocks(self, token: int) -> np.ndarray:
        """Return one low-bit score per context block for a query token."""

        if not 0 <= int(token) < self.config.vocab_size:
            raise ValueError("token outside vocab")
        slots = self._slots(token)
        per_bank = [
            self.summaries[:, bank, slot].astype(np.int32)
            for bank, slot in enumerate(slots)
        ]
        return np.min(np.stack(per_bank, axis=0), axis=0)

    @property
    def state_bytes(self) -> float:
        return self.config.summary_state_bytes


def run_compressed_block_index_trial(
    summary_width: int = 256,
    selected_blocks: int = 8,
    tail_blocks: int = 2,
    block_size: int = 64,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> CompressedBlockIndexResult:
    """Evaluate low-bit compressed block selection on a topic/noise stream."""

    config = CompressedBlockIndexConfig(
        context_length=context_length,
        block_size=block_size,
        selected_blocks=selected_blocks,
        tail_blocks=tail_blocks,
        summary_width=summary_width,
        queries=queries,
    )
    stream = _make_zipf_topic_stream(config, seed=seed)
    index = LowBitCompressedBlockIndex(config)
    touched = 0
    for position, token in enumerate(stream):
        touched += index.update(int(token), position)

    exact_counts = _build_exact_block_counts(stream, config.block_size)
    query_tokens = _make_zipf_topic_stream(config, seed=seed + 1, length=config.queries)
    recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)

    relevant_queries = 0
    hot_queries = 0
    cold_queries = 0
    hot_relevant_queries = 0
    cold_relevant_queries = 0
    index_hits = 0
    hot_index_hits = 0
    cold_index_hits = 0
    recent_hits = 0
    combined_hits = 0
    hot_combined_hits = 0
    cold_combined_hits = 0
    oracle_hits = 0
    index_coverage = 0.0
    recent_coverage = 0.0
    combined_coverage = 0.0
    oracle_coverage = 0.0
    combined_block_reads = 0

    for token in query_tokens:
        is_hot = int(token) < config.hot_tokens
        if is_hot:
            hot_queries += 1
        else:
            cold_queries += 1
        block_counts = exact_counts.get(int(token))
        if not block_counts:
            continue
        relevant_queries += 1
        if is_hot:
            hot_relevant_queries += 1
        else:
            cold_relevant_queries += 1
        scores = index.estimate_blocks(int(token))
        selected = _top_blocks(scores, config.selected_blocks)
        oracle = _top_blocks(_exact_score_vector(block_counts, config.blocks), config.selected_blocks)
        combined = np.union1d(selected, recent_blocks)
        combined_block_reads += len(combined)

        index_hit = _block_hit(selected, block_counts)
        combined_hit = _block_hit(combined, block_counts)
        index_hits += index_hit
        recent_hits += _block_hit(recent_blocks, block_counts)
        combined_hits += combined_hit
        oracle_hits += _block_hit(oracle, block_counts)
        if is_hot:
            hot_index_hits += index_hit
            hot_combined_hits += combined_hit
        else:
            cold_index_hits += index_hit
            cold_combined_hits += combined_hit
        index_coverage += _occurrence_coverage(selected, block_counts)
        recent_coverage += _occurrence_coverage(recent_blocks, block_counts)
        combined_coverage += _occurrence_coverage(combined, block_counts)
        oracle_coverage += _occurrence_coverage(oracle, block_counts)

    denominator = relevant_queries if relevant_queries else 1
    avg_combined_blocks = combined_block_reads / denominator
    return CompressedBlockIndexResult(
        context_length=config.context_length,
        block_size=config.block_size,
        blocks=config.blocks,
        selected_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        banks=config.banks,
        summary_width=config.summary_width,
        bits=config.bits,
        queries=config.queries,
        relevant_query_rate=relevant_queries / config.queries,
        hot_relevant_query_rate=_safe_divide(hot_relevant_queries, hot_queries),
        cold_relevant_query_rate=_safe_divide(cold_relevant_queries, cold_queries),
        summary_state_bytes=index.state_bytes,
        global_state_bytes=config.global_state_bytes,
        score_cells_per_query=float(config.score_cells_per_query),
        score_bytes_per_query=config.score_bytes_per_query,
        avg_update_cells_per_token=touched / config.context_length,
        index_block_hit_rate=index_hits / denominator,
        hot_index_block_hit_rate=_safe_divide(hot_index_hits, hot_relevant_queries),
        cold_index_block_hit_rate=_safe_divide(cold_index_hits, cold_relevant_queries),
        index_occurrence_coverage=index_coverage / denominator,
        recent_block_hit_rate=recent_hits / denominator,
        recent_occurrence_coverage=recent_coverage / denominator,
        combined_block_hit_rate=combined_hits / denominator,
        hot_combined_block_hit_rate=_safe_divide(hot_combined_hits, hot_relevant_queries),
        cold_combined_block_hit_rate=_safe_divide(cold_combined_hits, cold_relevant_queries),
        combined_occurrence_coverage=combined_coverage / denominator,
        oracle_block_hit_rate=oracle_hits / denominator,
        oracle_occurrence_coverage=oracle_coverage / denominator,
        dense_token_reads_per_query=float(config.context_length),
        index_token_reads_per_query=float(config.selected_blocks * config.block_size),
        combined_token_reads_per_query=float(avg_combined_blocks * config.block_size),
    )


def run_compressed_block_budget_sweep(
    summary_width: int = 256,
    block_budgets: Tuple[int, ...] = (4, 8, 16, 32, 64, 128),
    tail_blocks: int = 2,
    block_size: int = 64,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> CompressedBlockBudgetSweepResult:
    """Measure coverage and read traffic as more compressed blocks are read."""

    if len(block_budgets) == 0:
        raise ValueError("block_budgets must not be empty")
    if any(int(budget) <= 0 for budget in block_budgets):
        raise ValueError("all block budgets must be positive")
    max_budget = max(int(budget) for budget in block_budgets)
    config = CompressedBlockIndexConfig(
        context_length=context_length,
        block_size=block_size,
        selected_blocks=max_budget,
        tail_blocks=tail_blocks,
        summary_width=summary_width,
        queries=queries,
    )

    stream = _make_zipf_topic_stream(config, seed=seed)
    index = LowBitCompressedBlockIndex(config)
    for position, token in enumerate(stream):
        index.update(int(token), position)

    exact_counts = _build_exact_block_counts(stream, config.block_size)
    query_tokens = _make_zipf_topic_stream(config, seed=seed + 1, length=config.queries)
    recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
    budgets = tuple(sorted({int(budget) for budget in block_budgets}))
    metrics = {
        budget: {
            "hits": 0,
            "hot_hits": 0,
            "cold_hits": 0,
            "coverage": 0.0,
            "oracle": 0.0,
            "blocks": 0,
        }
        for budget in budgets
    }

    relevant_queries = 0
    hot_relevant_queries = 0
    cold_relevant_queries = 0
    for token in query_tokens:
        block_counts = exact_counts.get(int(token))
        if not block_counts:
            continue
        relevant_queries += 1
        is_hot = int(token) < config.hot_tokens
        hot_relevant_queries += int(is_hot)
        cold_relevant_queries += int(not is_hot)
        scores = index.estimate_blocks(int(token))
        block_order = _top_blocks(scores, max_budget)
        oracle_order = _top_blocks(_exact_score_vector(block_counts, config.blocks), max_budget)
        for budget in budgets:
            selected = block_order[:budget]
            combined = np.union1d(selected, recent_blocks)
            oracle = np.union1d(oracle_order[:budget], recent_blocks)
            hit = _block_hit(combined, block_counts)
            metrics[budget]["hits"] += hit
            metrics[budget]["hot_hits"] += hit if is_hot else 0
            metrics[budget]["cold_hits"] += hit if not is_hot else 0
            metrics[budget]["coverage"] += _occurrence_coverage(combined, block_counts)
            metrics[budget]["oracle"] += _occurrence_coverage(oracle, block_counts)
            metrics[budget]["blocks"] += len(combined)

    denominator = relevant_queries if relevant_queries else 1
    points = []
    for budget in budgets:
        token_reads = metrics[budget]["blocks"] / denominator * config.block_size
        points.append(
            CompressedBlockBudgetPoint(
                selected_blocks=budget,
                tail_blocks=config.tail_blocks,
                avg_blocks_read=metrics[budget]["blocks"] / denominator,
                block_hit_rate=metrics[budget]["hits"] / denominator,
                hot_block_hit_rate=_safe_divide(
                    metrics[budget]["hot_hits"],
                    hot_relevant_queries,
                ),
                cold_block_hit_rate=_safe_divide(
                    metrics[budget]["cold_hits"],
                    cold_relevant_queries,
                ),
                occurrence_coverage=metrics[budget]["coverage"] / denominator,
                oracle_occurrence_coverage=metrics[budget]["oracle"] / denominator,
                token_reads_per_query=token_reads,
                token_read_reduction=_safe_divide(config.context_length, token_reads),
            )
        )

    return CompressedBlockBudgetSweepResult(
        context_length=config.context_length,
        block_size=config.block_size,
        blocks=config.blocks,
        banks=config.banks,
        summary_width=config.summary_width,
        bits=config.bits,
        queries=config.queries,
        relevant_query_rate=relevant_queries / config.queries,
        summary_state_bytes=index.state_bytes,
        score_bytes_per_query=config.score_bytes_per_query,
        points=tuple(points),
    )


def run_csa_hca_policy_trial(
    summary_width: int = 256,
    global_width: int = 2048,
    thresholds: Tuple[int, ...] = (1, 2, 4, 8, 12),
    csa_blocks: int = 4,
    tail_blocks: int = 2,
    block_size: int = 64,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> CsaHcaPolicyResult:
    """Route high-frequency queries to HCA and low-frequency queries to CSA."""

    if len(thresholds) == 0:
        raise ValueError("thresholds must not be empty")
    if any(int(threshold) <= 0 for threshold in thresholds):
        raise ValueError("all thresholds must be positive")
    config = CompressedBlockIndexConfig(
        context_length=context_length,
        block_size=block_size,
        selected_blocks=csa_blocks,
        tail_blocks=tail_blocks,
        summary_width=summary_width,
        queries=queries,
    )
    global_config = DenseContextConfig(
        vocab_size=config.vocab_size,
        banks=config.banks,
        width=global_width,
        bits=config.bits,
        decay_interval=config.context_length + 1,
    )

    stream = _make_zipf_topic_stream(config, seed=seed)
    index = LowBitCompressedBlockIndex(config)
    global_summary = LowBitDenseContext(global_config)
    for position, token in enumerate(stream):
        index.update(int(token), position)
        global_summary.update(int(token))

    exact_counts = _build_exact_block_counts(stream, config.block_size)
    query_tokens = _make_zipf_topic_stream(config, seed=seed + 1, length=config.queries)
    recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
    thresholds = tuple(sorted({int(threshold) for threshold in thresholds}))
    metrics = {
        threshold: {
            "hca": 0,
            "csa": 0,
            "hot_hca": 0,
            "cold_csa": 0,
            "csa_relevant": 0,
            "csa_hits": 0,
            "csa_coverage": 0.0,
            "policy_coverage": 0.0,
            "token_reads": 0.0,
            "score_bytes": 0.0,
        }
        for threshold in thresholds
    }

    relevant_queries = 0
    hot_relevant_queries = 0
    cold_relevant_queries = 0
    for token in query_tokens:
        token = int(token)
        global_estimate = global_summary.estimate(token)
        is_hot = token < config.hot_tokens
        block_counts = exact_counts.get(token)
        is_relevant = block_counts is not None
        if is_relevant:
            relevant_queries += 1
            hot_relevant_queries += int(is_hot)
            cold_relevant_queries += int(not is_hot)
        for threshold in thresholds:
            route_hca = global_estimate >= threshold
            if route_hca:
                selected = recent_blocks
                metrics[threshold]["hca"] += 1
            else:
                scores = index.estimate_blocks(token)
                selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
                metrics[threshold]["csa"] += 1
                metrics[threshold]["score_bytes"] += config.score_bytes_per_query

            metrics[threshold]["token_reads"] += len(selected) * config.block_size
            if not is_relevant:
                continue
            if route_hca and is_hot:
                metrics[threshold]["hot_hca"] += 1
            if (not route_hca) and (not is_hot):
                metrics[threshold]["cold_csa"] += 1
            coverage = _occurrence_coverage(selected, block_counts)
            metrics[threshold]["policy_coverage"] += coverage
            if not route_hca:
                metrics[threshold]["csa_relevant"] += 1
                metrics[threshold]["csa_hits"] += _block_hit(selected, block_counts)
                metrics[threshold]["csa_coverage"] += coverage

    query_denominator = config.queries if config.queries else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    points = []
    for threshold in thresholds:
        token_reads = metrics[threshold]["token_reads"] / query_denominator
        csa_relevant = metrics[threshold]["csa_relevant"]
        points.append(
            CsaHcaPolicyPoint(
                hca_threshold=threshold,
                csa_blocks=config.selected_blocks,
                tail_blocks=config.tail_blocks,
                hca_query_rate=metrics[threshold]["hca"] / query_denominator,
                csa_query_rate=metrics[threshold]["csa"] / query_denominator,
                hot_to_hca_rate=_safe_divide(
                    metrics[threshold]["hot_hca"],
                    hot_relevant_queries,
                ),
                cold_to_csa_rate=_safe_divide(
                    metrics[threshold]["cold_csa"],
                    cold_relevant_queries,
                ),
                csa_relevant_hit_rate=_safe_divide(
                    metrics[threshold]["csa_hits"],
                    csa_relevant,
                ),
                csa_relevant_coverage=_safe_divide(
                    metrics[threshold]["csa_coverage"],
                    csa_relevant,
                ),
                policy_sparse_coverage=metrics[threshold]["policy_coverage"]
                / relevant_denominator,
                token_reads_per_query=token_reads,
                token_read_reduction=_safe_divide(config.context_length, token_reads),
                block_score_bytes_per_query=metrics[threshold]["score_bytes"]
                / query_denominator,
            )
        )

    global_summary_read_bytes = config.banks * config.bits / 8
    return CsaHcaPolicyResult(
        context_length=config.context_length,
        block_size=config.block_size,
        blocks=config.blocks,
        banks=config.banks,
        summary_width=config.summary_width,
        global_width=global_width,
        bits=config.bits,
        queries=config.queries,
        relevant_query_rate=relevant_queries / query_denominator,
        block_summary_state_bytes=index.state_bytes,
        global_summary_state_bytes=global_summary.memory_bytes(),
        global_summary_read_bytes_per_query=global_summary_read_bytes,
        points=tuple(points),
    )


def run_hca_summary_quality_sweep(
    global_widths: Tuple[int, ...] = (512, 1024, 2048, 4096),
    threshold: int = 8,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> HcaSummaryQualityResult:
    """Check whether the HCA-like global summary can support path routing."""

    if len(global_widths) == 0:
        raise ValueError("global_widths must not be empty")
    if any(int(width) <= 0 for width in global_widths):
        raise ValueError("all global widths must be positive")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    config = CompressedBlockIndexConfig(context_length=context_length, queries=queries)
    stream = _make_zipf_topic_stream(config, seed=seed)
    exact_counts = np.bincount(stream, minlength=config.vocab_size).astype(np.int32)
    query_tokens = _make_zipf_topic_stream(config, seed=seed + 1, length=config.queries)
    exact_query_hca = exact_counts[query_tokens] >= threshold

    points = []
    for width in tuple(sorted({int(width) for width in global_widths})):
        global_config = DenseContextConfig(
            vocab_size=config.vocab_size,
            banks=config.banks,
            width=width,
            bits=config.bits,
            decay_interval=config.context_length + 1,
        )
        summary = LowBitDenseContext(global_config)
        for token in stream:
            summary.update(int(token))
        estimates = summary.estimate_all().astype(np.int32)
        clipped_exact = np.minimum(exact_counts, global_config.max_value)
        exact_frequent = exact_counts >= threshold
        estimated_frequent = estimates >= threshold
        query_estimated_hca = estimated_frequent[query_tokens]

        true_positive = int(np.count_nonzero(exact_frequent & estimated_frequent))
        predicted_positive = int(np.count_nonzero(estimated_frequent))
        actual_positive = int(np.count_nonzero(exact_frequent))
        query_correct = int(np.count_nonzero(query_estimated_hca == exact_query_hca))
        query_false_hca = int(np.count_nonzero(query_estimated_hca & ~exact_query_hca))
        query_missed_hca = int(np.count_nonzero(~query_estimated_hca & exact_query_hca))
        query_actual_cold = int(np.count_nonzero(~exact_query_hca))
        query_actual_hot = int(np.count_nonzero(exact_query_hca))

        points.append(
            HcaSummaryQualityPoint(
                global_width=width,
                state_bytes=summary.memory_bytes(),
                read_bytes_per_query=config.banks * config.bits / 8,
                saturation_rate=float(
                    np.count_nonzero(summary.counters == global_config.max_value)
                    / summary.counters.size
                ),
                clipped_mean_abs_error=float(np.mean(np.abs(estimates - clipped_exact))),
                top64_recall=_topk_recall(estimates, exact_counts, 64),
                top256_recall=_topk_recall(estimates, exact_counts, 256),
                threshold_precision=_safe_divide(true_positive, predicted_positive),
                threshold_recall=_safe_divide(true_positive, actual_positive),
                query_route_accuracy=query_correct / config.queries,
                query_false_hca_rate=_safe_divide(query_false_hca, query_actual_cold),
                query_missed_hca_rate=_safe_divide(query_missed_hca, query_actual_hot),
            )
        )

    return HcaSummaryQualityResult(
        context_length=config.context_length,
        vocab_size=config.vocab_size,
        hot_tokens=config.hot_tokens,
        bits=config.bits,
        threshold=threshold,
        queries=config.queries,
        points=tuple(points),
    )


def run_hca_decay_quality_sweep(
    global_width: int = 2048,
    decay_intervals: Tuple[int, ...] = (64, 128, 256, 512, 1024, 65537),
    threshold: int = 2,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> HcaDecayQualityResult:
    """Evaluate periodic decay as the first HCA anti-saturation mechanism."""

    if global_width <= 0:
        raise ValueError("global_width must be positive")
    if len(decay_intervals) == 0:
        raise ValueError("decay_intervals must not be empty")
    if any(int(interval) <= 0 for interval in decay_intervals):
        raise ValueError("all decay intervals must be positive")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    config = CompressedBlockIndexConfig(context_length=context_length, queries=queries)
    stream = _make_zipf_topic_stream(config, seed=seed)
    query_tokens = _make_zipf_topic_stream(config, seed=seed + 1, length=config.queries)

    points = []
    for decay_interval in tuple(sorted({int(interval) for interval in decay_intervals})):
        global_config = DenseContextConfig(
            vocab_size=config.vocab_size,
            banks=config.banks,
            width=global_width,
            bits=config.bits,
            decay_interval=decay_interval,
        )
        summary = LowBitDenseContext(global_config)
        for token in stream:
            summary.update(int(token))
        exact = exact_decayed_counts(stream, global_config).astype(np.int32)
        estimates = summary.estimate_all().astype(np.int32)
        exact_frequent = exact >= threshold
        estimated_frequent = estimates >= threshold
        exact_query_hca = exact_frequent[query_tokens]
        query_estimated_hca = estimated_frequent[query_tokens]

        true_positive = int(np.count_nonzero(exact_frequent & estimated_frequent))
        predicted_positive = int(np.count_nonzero(estimated_frequent))
        actual_positive = int(np.count_nonzero(exact_frequent))
        query_correct = int(np.count_nonzero(query_estimated_hca == exact_query_hca))
        query_false_hca = int(np.count_nonzero(query_estimated_hca & ~exact_query_hca))
        query_missed_hca = int(np.count_nonzero(~query_estimated_hca & exact_query_hca))
        query_actual_cold = int(np.count_nonzero(~exact_query_hca))
        query_actual_hot = int(np.count_nonzero(exact_query_hca))
        decay_events = config.context_length // decay_interval
        decay_cells = decay_events * config.banks * global_width

        points.append(
            HcaDecayQualityPoint(
                decay_interval=decay_interval,
                state_bytes=summary.memory_bytes(),
                read_bytes_per_query=config.banks * config.bits / 8,
                avg_decay_cells_per_token=decay_cells / config.context_length,
                saturation_rate=float(
                    np.count_nonzero(summary.counters == global_config.max_value)
                    / summary.counters.size
                ),
                clipped_mean_abs_error=float(np.mean(np.abs(estimates - exact))),
                top64_recall=_topk_recall(estimates, exact, 64),
                top256_recall=_topk_recall(estimates, exact, 256),
                threshold_precision=_safe_divide(true_positive, predicted_positive),
                threshold_recall=_safe_divide(true_positive, actual_positive),
                query_route_accuracy=query_correct / config.queries,
                query_false_hca_rate=_safe_divide(query_false_hca, query_actual_cold),
                query_missed_hca_rate=_safe_divide(query_missed_hca, query_actual_hot),
            )
        )

    return HcaDecayQualityResult(
        context_length=config.context_length,
        vocab_size=config.vocab_size,
        hot_tokens=config.hot_tokens,
        global_width=global_width,
        bits=config.bits,
        threshold=threshold,
        queries=config.queries,
        points=tuple(points),
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _make_zipf_topic_stream(
    config: CompressedBlockIndexConfig,
    seed: int,
    length: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stream_length = config.context_length if length is None else int(length)
    stream = np.empty(stream_length, dtype=np.int32)
    ranks = np.arange(1, config.hot_tokens + 1, dtype=np.float64)
    probabilities = 1.0 / np.power(ranks, config.zipf_exponent)
    probabilities /= probabilities.sum()
    for index in range(stream_length):
        if rng.random() < config.topic_probability:
            stream[index] = int(rng.choice(config.hot_tokens, p=probabilities))
        else:
            stream[index] = int(rng.integers(config.hot_tokens, config.vocab_size))
    return stream


def _build_exact_block_counts(
    stream: np.ndarray,
    block_size: int,
) -> Dict[int, Dict[int, int]]:
    token_to_blocks: Dict[int, Dict[int, int]] = {}
    for position, token in enumerate(stream):
        block = position // block_size
        counts = token_to_blocks.setdefault(int(token), {})
        counts[block] = counts.get(block, 0) + 1
    return token_to_blocks


def _exact_score_vector(block_counts: Dict[int, int], blocks: int) -> np.ndarray:
    scores = np.zeros(blocks, dtype=np.int32)
    for block, count in block_counts.items():
        scores[int(block)] = int(count)
    return scores


def _top_blocks(scores: np.ndarray, count: int) -> np.ndarray:
    block_ids = np.arange(len(scores), dtype=np.int32)
    order = np.lexsort((block_ids, -scores))
    return order[: min(count, len(order))].astype(np.int32)


def _topk_recall(estimated: np.ndarray, exact: np.ndarray, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    count = min(k, len(exact))
    exact_top = set(np.argsort(exact)[-count:].tolist())
    estimated_top = set(np.argsort(estimated)[-count:].tolist())
    return len(exact_top & estimated_top) / count


def _recent_blocks(blocks: int, count: int) -> np.ndarray:
    if count == 0:
        return np.empty(0, dtype=np.int32)
    return np.arange(max(0, blocks - count), blocks, dtype=np.int32)


def _block_hit(selected: np.ndarray, block_counts: Dict[int, int]) -> int:
    selected_set = {int(block) for block in selected}
    return int(any(int(block) in selected_set for block in block_counts))


def _occurrence_coverage(selected: np.ndarray, block_counts: Dict[int, int]) -> float:
    total = sum(int(value) for value in block_counts.values())
    if total == 0:
        return 0.0
    selected_set = {int(block) for block in selected}
    covered = sum(
        int(count)
        for block, count in block_counts.items()
        if int(block) in selected_set
    )
    return covered / total
