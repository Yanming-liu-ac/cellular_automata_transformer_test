"""Compressed block indexing for CSA-shaped context retrieval.

DeepSeek-V4's CSA path suggests a hardware pattern that maps cleanly onto a
cellular fabric: split the context into blocks, keep a low-bit summary inside
each block cell, and route a query to only a few high-scoring blocks instead of
reading the full KV cache. This module tests that pattern without requiring a
Transformer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

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
