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
class CsaHcaBlockStatePoint:
    """One CSA block-summary geometry under a fixed HCA routing policy."""

    block_size: int
    blocks: int
    summary_width: int
    csa_blocks: int
    tail_blocks: int
    block_summary_state_bytes: float
    block_score_bytes_per_query: float
    hca_query_rate: float
    csa_query_rate: float
    hot_to_hca_rate: float
    cold_to_csa_rate: float
    csa_relevant_hit_rate: float
    csa_relevant_coverage: float
    policy_sparse_coverage: float
    token_reads_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaBlockStateSweepResult:
    """State/read/quality sweep for CSA summaries behind an HCA gate."""

    context_length: int
    banks: int
    global_width: int
    bits: int
    hca_threshold: int
    queries: int
    relevant_query_rate: float
    global_summary_state_bytes: float
    global_summary_read_bytes_per_query: float
    points: Tuple[CsaHcaBlockStatePoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryPoint:
    """One CSA geometry with a small exact rare-token block directory."""

    block_size: int
    blocks: int
    summary_width: int
    csa_blocks: int
    tail_blocks: int
    directory_blocks_per_token: int
    directory_entries: int
    directory_entry_bytes: float
    directory_state_bytes: float
    block_summary_state_bytes: float
    block_plus_directory_state_bytes: float
    block_score_bytes_per_query: float
    directory_read_bytes_per_query: float
    hca_query_rate: float
    csa_query_rate: float
    directory_query_rate: float
    base_csa_relevant_hit_rate: float
    repaired_csa_relevant_hit_rate: float
    directory_repair_rate: float
    base_csa_relevant_coverage: float
    repaired_csa_relevant_coverage: float
    token_reads_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectorySweepResult:
    """Exact rare-token directory sweep for low-state CSA block summaries."""

    context_length: int
    banks: int
    global_width: int
    bits: int
    hca_threshold: int
    queries: int
    relevant_query_rate: float
    global_summary_state_bytes: float
    global_summary_read_bytes_per_query: float
    points: Tuple[CsaHcaRareDirectoryPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryStressPoint:
    """One rare-directory stress scenario and directory size."""

    scenario: str
    directory_guard: bool
    directory_blocks_per_token: int
    directory_read_blocks_per_token: int
    stress_token_count: int
    directory_entries: int
    directory_state_bytes: float
    block_plus_directory_state_bytes: float
    hca_query_rate: float
    csa_query_rate: float
    rare_false_hca_rate: float
    directory_guard_hit_rate: float
    directory_query_rate: float
    base_relevant_hit_rate: float
    repaired_relevant_hit_rate: float
    base_relevant_coverage: float
    repaired_relevant_coverage: float
    base_csa_relevant_hit_rate: float
    repaired_csa_relevant_hit_rate: float
    base_csa_relevant_coverage: float
    repaired_csa_relevant_coverage: float
    directory_read_bytes_per_query: float
    token_reads_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryStressResult:
    """Stress diagnostics for the exact rare-token block directory."""

    context_length: int
    block_size: int
    blocks: int
    banks: int
    summary_width: int
    global_width: int
    bits: int
    hca_threshold: int
    csa_blocks: int
    tail_blocks: int
    directory_guard: bool
    queries: int
    block_summary_state_bytes: float
    global_summary_state_bytes: float
    global_summary_read_bytes_per_query: float
    points: Tuple[CsaHcaRareDirectoryStressPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryPolicyPoint:
    """One admission/fanout policy on a rare-directory stress scenario."""

    policy: str
    scenario: str
    hca_threshold: int
    directory_guard: bool
    directory_blocks_per_token: int
    directory_read_blocks_per_token: int
    directory_state_bytes: float
    rare_false_hca_rate: float
    repaired_relevant_hit_rate: float
    repaired_relevant_coverage: float
    directory_read_bytes_per_query: float
    token_reads_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryPolicyResult:
    """Policy comparison for rare-directory admission and fanout."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    queries: int
    points: Tuple[CsaHcaRareDirectoryPolicyPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryAdaptivePolicyPoint:
    """One metadata-driven rare-directory fanout policy on a stress scenario."""

    policy: str
    scenario: str
    hca_threshold: int
    directory_guard: bool
    directory_blocks_per_token: int
    base_read_blocks_per_token: int
    expanded_read_blocks_per_token: int
    spread_threshold_blocks: int
    fanout_metadata_state_bytes: float
    directory_state_bytes: float
    avg_directory_entries_per_hit: float
    avg_directory_read_blocks_per_hit: float
    expanded_read_rate: float
    rare_false_hca_rate: float
    repaired_relevant_hit_rate: float
    repaired_relevant_coverage: float
    directory_read_bytes_per_query: float
    token_reads_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryAdaptivePolicyResult:
    """Metadata-driven rare-directory fanout policy comparison."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    queries: int
    points: Tuple[CsaHcaRareDirectoryAdaptivePolicyPoint, ...]


@dataclass(frozen=True)
class LowBitRareDirectoryFanoutLUT:
    """Low-bit LUT for rare-directory read fanout.

    The LUT is indexed by metadata visible to a CA memory tile: stored directory
    entry count, coarse block-span class, and overlap with the CSA-selected
    blocks. Values are fanout counts, not scores.
    """

    fanouts: Tuple[int, ...]
    max_entries: int = 6
    span_thresholds: Tuple[int, ...] = (64, 128, 256)
    max_overlap_bucket: int = 3
    fanout_bits: int = 3

    def __post_init__(self) -> None:
        if self.max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        if self.max_overlap_bucket < 0:
            raise ValueError("max_overlap_bucket must be non-negative")
        if self.fanout_bits not in (2, 3, 4, 8):
            raise ValueError("fanout_bits must be one of 2, 3, 4, 8")
        if any(int(threshold) < 0 for threshold in self.span_thresholds):
            raise ValueError("span thresholds must be non-negative")
        expected = (self.max_entries + 1) * (len(self.span_thresholds) + 1) * (
            self.max_overlap_bucket + 1
        )
        if len(self.fanouts) != expected:
            raise ValueError("fanout table length does not match metadata dimensions")
        max_value = (1 << self.fanout_bits) - 1
        for fanout in self.fanouts:
            if not 0 <= int(fanout) <= max_value:
                raise ValueError("fanout outside fanout_bits range")
            if int(fanout) > self.max_entries:
                raise ValueError("fanout cannot exceed max_entries")

    @property
    def state_bytes(self) -> float:
        return len(self.fanouts) * self.fanout_bits / 8

    def predict(self, directory_blocks: np.ndarray, base_selected: np.ndarray) -> int:
        index = _rare_fanout_lut_index(
            directory_blocks=directory_blocks,
            base_selected=base_selected,
            max_entries=self.max_entries,
            span_thresholds=self.span_thresholds,
            max_overlap_bucket=self.max_overlap_bucket,
        )
        entry_count = min(len(directory_blocks), self.max_entries)
        return min(int(self.fanouts[index]), entry_count)


@dataclass(frozen=True)
class LowBitRareDirectoryProbeLUT:
    """Low-bit LUT deciding whether an HCA-routed token needs a directory probe."""

    probes: Tuple[bool, ...]
    max_counter: int = 15
    spread_thresholds: Tuple[int, ...] = (1, 3, 7)
    max_saturation_bucket: int = 4

    def __post_init__(self) -> None:
        if self.max_counter <= 0:
            raise ValueError("max_counter must be positive")
        if self.max_saturation_bucket < 0:
            raise ValueError("max_saturation_bucket must be non-negative")
        if any(int(threshold) < 0 for threshold in self.spread_thresholds):
            raise ValueError("spread thresholds must be non-negative")
        expected = (self.max_counter + 1) * (len(self.spread_thresholds) + 1) * (
            self.max_saturation_bucket + 1
        )
        if len(self.probes) != expected:
            raise ValueError("probe table length does not match metadata dimensions")

    @property
    def state_bytes(self) -> float:
        return len(self.probes) / 8

    def probe(self, counter_values: Tuple[int, ...]) -> bool:
        index = _hca_probe_lut_index(
            counter_values=counter_values,
            max_counter=self.max_counter,
            spread_thresholds=self.spread_thresholds,
            max_saturation_bucket=self.max_saturation_bucket,
        )
        return bool(self.probes[index])


@dataclass(frozen=True)
class LowBitHcaRouteLUT:
    """Low-bit LUT deciding whether a query should use the HCA path."""

    routes_hca: Tuple[bool, ...]
    max_counter: int = 15
    spread_thresholds: Tuple[int, ...] = (1, 3, 7)
    max_saturation_bucket: int = 4

    def __post_init__(self) -> None:
        if self.max_counter <= 0:
            raise ValueError("max_counter must be positive")
        if self.max_saturation_bucket < 0:
            raise ValueError("max_saturation_bucket must be non-negative")
        if any(int(threshold) < 0 for threshold in self.spread_thresholds):
            raise ValueError("spread thresholds must be non-negative")
        expected = (self.max_counter + 1) * (len(self.spread_thresholds) + 1) * (
            self.max_saturation_bucket + 1
        )
        if len(self.routes_hca) != expected:
            raise ValueError("route table length does not match metadata dimensions")

    @property
    def state_bytes(self) -> float:
        return len(self.routes_hca) / 8

    def route_hca(self, counter_values: Tuple[int, ...]) -> bool:
        index = _hca_control_lut_index(
            counter_values=counter_values,
            max_counter=self.max_counter,
            spread_thresholds=self.spread_thresholds,
            max_saturation_bucket=self.max_saturation_bucket,
        )
        return bool(self.routes_hca[index])


@dataclass(frozen=True)
class LowBitDirectoryAwareHcaRouteLUT:
    """HCA route LUT with one rare-directory presence feature bit."""

    routes_hca: Tuple[bool, ...]
    max_counter: int = 15
    spread_thresholds: Tuple[int, ...] = (1, 3, 7)
    max_saturation_bucket: int = 4

    def __post_init__(self) -> None:
        if self.max_counter <= 0:
            raise ValueError("max_counter must be positive")
        if self.max_saturation_bucket < 0:
            raise ValueError("max_saturation_bucket must be non-negative")
        if any(int(threshold) < 0 for threshold in self.spread_thresholds):
            raise ValueError("spread thresholds must be non-negative")
        hca_buckets = (self.max_counter + 1) * (len(self.spread_thresholds) + 1) * (
            self.max_saturation_bucket + 1
        )
        expected = hca_buckets * 2
        if len(self.routes_hca) != expected:
            raise ValueError("route table length does not match metadata dimensions")

    @property
    def state_bytes(self) -> float:
        return len(self.routes_hca) / 8

    def route_hca(
        self,
        counter_values: Tuple[int, ...],
        directory_blocks: np.ndarray,
    ) -> bool:
        index = _directory_aware_hca_route_lut_index(
            counter_values=counter_values,
            directory_blocks=directory_blocks,
            max_counter=self.max_counter,
            spread_thresholds=self.spread_thresholds,
            max_saturation_bucket=self.max_saturation_bucket,
        )
        return bool(self.routes_hca[index])


class LowBitPresenceBloomSidecar:
    """Low-bit Bloom-style sidecar for rare-directory token presence."""

    def __init__(
        self,
        bit_count: int,
        hash_count: int,
        bank_count: int = 8,
        bank_mode: str = "modulo",
        salt: int = 6113,
    ) -> None:
        if bit_count <= 0:
            raise ValueError("bit_count must be positive")
        if hash_count <= 0:
            raise ValueError("hash_count must be positive")
        if bank_count <= 0:
            raise ValueError("bank_count must be positive")
        if bank_mode not in ("modulo", "by_hash", "hash_slot"):
            raise ValueError("bank_mode must be one of: modulo, by_hash, hash_slot")
        self.bit_count = int(bit_count)
        self.hash_count = int(hash_count)
        self.bank_count = int(bank_count)
        self.bank_mode = str(bank_mode)
        self.salt = int(salt)
        self.bits = np.zeros(self.bit_count, dtype=np.bool_)
        self.insert_count = 0

    @property
    def state_bytes(self) -> float:
        return self.bit_count / 8

    @property
    def read_bytes_per_query(self) -> float:
        return self.hash_count / 8

    @property
    def write_bytes_per_insert(self) -> float:
        return self.hash_count / 8

    def slots(self, token: int) -> Tuple[int, ...]:
        return tuple(
            int(keyed_hash(int(token), self.salt + index * 131) % self.bit_count)
            for index in range(self.hash_count)
        )

    def banks(self, slots: Tuple[int, ...]) -> Tuple[int, ...]:
        if self.bank_mode == "by_hash":
            return tuple(index % self.bank_count for index, _ in enumerate(slots))
        if self.bank_mode == "hash_slot":
            return tuple(
                int(keyed_hash(int(slot), self.salt + 7919)) % self.bank_count
                for slot in slots
            )
        return tuple(int(slot) % self.bank_count for slot in slots)

    def insert(self, token: int) -> None:
        for slot in self.slots(token):
            self.bits[slot] = True
        self.insert_count += 1

    def query(self, token: int) -> bool:
        return all(bool(self.bits[slot]) for slot in self.slots(token))


class LowBitCountingBloomSidecar(LowBitPresenceBloomSidecar):
    """Counting Bloom sidecar with a fast 1-bit visible plane for queries."""

    def __init__(
        self,
        bit_count: int,
        hash_count: int,
        counter_bits: int = 4,
        bank_count: int = 8,
        bank_mode: str = "modulo",
        salt: int = 6113,
    ) -> None:
        if counter_bits <= 0:
            raise ValueError("counter_bits must be positive")
        if counter_bits > 16:
            raise ValueError("counter_bits must be at most 16")
        super().__init__(
            bit_count=bit_count,
            hash_count=hash_count,
            bank_count=bank_count,
            bank_mode=bank_mode,
            salt=salt,
        )
        self.counter_bits = int(counter_bits)
        self.max_counter = (1 << self.counter_bits) - 1
        self.counters = np.zeros(self.bit_count, dtype=np.uint16)
        self.delete_count = 0

    @property
    def state_bytes(self) -> float:
        return self.bit_count * (self.counter_bits + 1) / 8

    @property
    def read_bytes_per_query(self) -> float:
        return self.hash_count / 8

    @property
    def write_bytes_per_insert(self) -> float:
        return self.hash_count * (self.counter_bits + 1) / 8

    @property
    def write_bytes_per_delete(self) -> float:
        return self.write_bytes_per_insert

    def insert(self, token: int) -> None:
        for slot in self.slots(token):
            if self.counters[slot] < self.max_counter:
                self.counters[slot] += 1
            self.bits[slot] = self.counters[slot] > 0
        self.insert_count += 1

    def delete(self, token: int) -> None:
        for slot in self.slots(token):
            if self.counters[slot] > 0:
                self.counters[slot] -= 1
            self.bits[slot] = self.counters[slot] > 0
        self.delete_count += 1


@dataclass(frozen=True)
class CsaHcaRareDirectoryLearnedFanoutResult:
    """Trained fanout-LUT evaluation for the rare-token directory."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_guard: bool
    directory_blocks_per_token: int
    coverage_target: float
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryAdaptivePolicyPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryJointPolicyPoint:
    """One joint probe/admission/fanout policy on a rare-directory scenario."""

    policy: str
    scenario: str
    hca_threshold: int
    probe_mode: str
    directory_blocks_per_token: int
    probe_lut_state_bytes: float
    fanout_lut_state_bytes: float
    fanout_metadata_state_bytes: float
    directory_state_bytes: float
    directory_probe_rate: float
    directory_hit_rate: float
    avg_directory_entries_per_hit: float
    avg_directory_read_blocks_per_hit: float
    expanded_read_rate: float
    hca_query_rate: float
    csa_query_rate: float
    rare_false_hca_rate: float
    repaired_relevant_hit_rate: float
    repaired_relevant_coverage: float
    directory_read_bytes_per_query: float
    token_reads_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryJointPolicyResult:
    """Joint rare-directory probe/admission/fanout control comparison."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    coverage_target: float
    probe_positive_rate_threshold: float
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    lut: LowBitRareDirectoryFanoutLUT
    probe_lut: LowBitRareDirectoryProbeLUT
    points: Tuple[CsaHcaRareDirectoryJointPolicyPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryJointThresholdResult:
    """HCA-threshold sweep under the joint probe/fanout control policy."""

    thresholds: Tuple[int, ...]
    policy: str
    probe_mode: str
    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    directory_blocks_per_token: int
    coverage_target: float
    points: Tuple[CsaHcaRareDirectoryJointPolicyPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryRouteLutResult:
    """Trained HCA-route LUT evaluation with rare-directory fanout control."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    coverage_target: float
    route_positive_rate_threshold: float
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryJointPolicyPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryAwareRouteLutResult:
    """Directory-aware HCA-route LUT evaluation with fanout control."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    coverage_target: float
    route_positive_rate_threshold: float
    route_feature_read_bytes: float
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryJointPolicyPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryPresenceSidecarPoint:
    """One false-positive point for rare-directory presence sidecar routing."""

    false_positive_rate: float
    scenario: str
    hca_threshold: int
    route_lut_state_bytes: float
    sidecar_state_bytes: float
    fanout_lut_state_bytes: float
    directory_state_bytes: float
    route_feature_read_bytes: float
    sidecar_false_positive_query_rate: float
    directory_hit_rate: float
    hca_query_rate: float
    csa_query_rate: float
    rare_false_hca_rate: float
    repaired_relevant_hit_rate: float
    repaired_relevant_coverage: float
    directory_read_bytes_per_query: float
    token_reads_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryPresenceSidecarResult:
    """Presence-sidecar false-positive sweep for directory-aware HCA routing."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    coverage_target: float
    route_positive_rate_threshold: float
    route_feature_read_bytes: float
    false_positive_rates: Tuple[float, ...]
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryPresenceSidecarPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomSidecarPoint:
    """One physical Bloom-sidecar point for directory-aware HCA routing."""

    bits_per_entry: int
    hash_count: int
    bank_count: int
    bank_mode: str
    scenario: str
    hca_threshold: int
    route_lut_state_bytes: float
    sidecar_state_bytes: float
    fanout_lut_state_bytes: float
    directory_state_bytes: float
    sidecar_entries: int
    read_bytes_per_query: float
    write_bytes_per_insert: float
    update_bytes_per_context_token: float
    sidecar_false_positive_query_rate: float
    query_bank_conflict_rate: float
    update_bank_conflict_rate: float
    avg_query_unique_banks: float
    avg_update_unique_banks: float
    directory_hit_rate: float
    hca_query_rate: float
    csa_query_rate: float
    rare_false_hca_rate: float
    repaired_relevant_hit_rate: float
    repaired_relevant_coverage: float
    directory_read_bytes_per_query: float
    token_reads_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomSidecarResult:
    """Physical Bloom-sidecar sweep for directory-aware HCA routing."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    coverage_target: float
    route_positive_rate_threshold: float
    bits_per_entry_options: Tuple[int, ...]
    hash_count_options: Tuple[int, ...]
    bank_count: int
    bank_mode: str
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryBloomSidecarPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomSaltPoint:
    """One hash-salt robustness point for a physical Bloom sidecar."""

    salt_index: int
    sidecar_salt: int
    bits_per_entry: int
    hash_count: int
    bank_count: int
    bank_mode: str
    scenario: str
    sidecar_state_bytes: float
    read_bytes_per_query: float
    write_bytes_per_insert: float
    sidecar_false_positive_query_rate: float
    hot_query_rate: float
    hot_sidecar_false_positive_rate: float
    query_bank_conflict_rate: float
    hca_query_rate: float
    repaired_relevant_coverage: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomSaltResult:
    """Hash-salt robustness sweep for the Bloom presence sidecar."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    bits_per_entry: int
    hash_count: int
    bank_count: int
    bank_mode: str
    salt_count: int
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryBloomSaltPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomBankResult:
    """Bank-mapping comparison for the physical Bloom presence sidecar."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    bits_per_entry: int
    hash_count: int
    bank_count: int
    bank_modes: Tuple[str, ...]
    salt_count: int
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryBloomSaltPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomSaltSelectionResult:
    """Selected Bloom-sidecar salt evaluated after hot-token scoring."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    bits_per_entry: int
    hash_count: int
    bank_count: int
    bank_mode: str
    salt_count: int
    selected_salt_index: int
    selected_sidecar_salt: int
    selection_metric: str
    route_train_scenarios: Tuple[str, ...]
    selection_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    selection_points: Tuple[CsaHcaRareDirectoryBloomSaltPoint, ...]
    eval_points: Tuple[CsaHcaRareDirectoryBloomSaltPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomStreamingPoint:
    """One streaming-update policy for the selected Bloom presence sidecar."""

    policy: str
    insert_count_threshold: int
    scenario: str
    bits_per_entry: int
    hash_count: int
    bank_count: int
    bank_mode: str
    sidecar_salt: int
    sidecar_state_bytes: float
    sidecar_fill_rate: float
    inserted_tokens: int
    final_rare_tokens: int
    final_hot_tokens: int
    inserted_final_rare_rate: float
    hot_polluted_token_rate: float
    update_bytes_per_context_token: float
    max_bank_update_bytes_per_context_token: float
    update_bank_conflict_rate: float
    avg_update_unique_banks: float
    sidecar_false_positive_query_rate: float
    hot_sidecar_false_positive_rate: float
    hca_query_rate: float
    rare_false_hca_rate: float
    repaired_relevant_coverage: float
    directory_read_bytes_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomStreamingResult:
    """Streaming update-pressure sweep for the selected Bloom sidecar."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    bits_per_entry: int
    hash_count: int
    bank_count: int
    bank_mode: str
    sidecar_salt: int
    policies: Tuple[int, ...]
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryBloomStreamingPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomRetirementPoint:
    """One deletable Bloom sidecar policy with hot-token retirement."""

    policy: str
    insert_count_threshold: int
    retire_count_threshold: int
    scenario: str
    bits_per_entry: int
    hash_count: int
    counter_bits: int
    bank_count: int
    bank_mode: str
    sidecar_salt: int
    sidecar_state_bytes: float
    sidecar_fill_rate: float
    inserted_tokens: int
    active_tokens: int
    deleted_tokens: int
    final_rare_tokens: int
    final_hot_tokens: int
    inserted_final_rare_rate: float
    active_final_rare_rate: float
    visible_active_rare_rate: float
    hot_retired_token_rate: float
    hot_polluted_token_rate: float
    update_bytes_per_context_token: float
    max_bank_update_bytes_per_context_token: float
    update_bank_conflict_rate: float
    avg_update_unique_banks: float
    sidecar_false_positive_query_rate: float
    hot_sidecar_false_positive_rate: float
    hca_query_rate: float
    rare_false_hca_rate: float
    repaired_relevant_coverage: float
    directory_read_bytes_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomRetirementResult:
    """Counting/deletable Bloom sidecar sweep with hot-token retirement."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    bits_per_entry: int
    hash_count: int
    counter_bits: int
    bank_count: int
    bank_mode: str
    sidecar_salt: int
    insert_count_thresholds: Tuple[int, ...]
    retire_count_threshold: int
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryBloomRetirementPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomRetirementCompressionResult:
    """Compressed counting-Bloom geometry sweep for hot-token retirement."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    bits_per_entries: Tuple[int, ...]
    hash_count: int
    counter_bits_values: Tuple[int, ...]
    bank_count: int
    bank_mode: str
    sidecar_salt: int
    insert_count_thresholds: Tuple[int, ...]
    retire_count_threshold: int
    train_scenarios: Tuple[str, ...]
    eval_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryBloomRetirementPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomRetirementCollisionPoint:
    """Adversarial Bloom-collision point for retirement sidecar deletion."""

    scenario: str
    bits_per_entry: int
    hash_count: int
    counter_bits: int
    bank_count: int
    bank_mode: str
    sidecar_salt: int
    rare_occurrences_per_token: int
    colliders_per_rare: int
    rare_tokens: int
    collider_tokens: int
    missing_colliders: int
    mean_slot_overlap: float
    sidecar_state_bytes: float
    visible_active_rare_rate: float
    hot_polluted_token_rate: float
    update_bytes_per_context_token: float
    hca_query_rate: float
    rare_false_hca_rate: float
    repaired_relevant_coverage: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomRetirementCollisionResult:
    """Adversarial collision sweep for counting-Bloom retirement counters."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    rare_token_count: int
    rare_occurrences_per_token_values: Tuple[int, ...]
    colliders_per_rare_values: Tuple[int, ...]
    bits_per_entries: Tuple[int, ...]
    hash_count: int
    counter_bits_values: Tuple[int, ...]
    bank_count: int
    bank_mode: str
    sidecar_salt: int
    train_scenarios: Tuple[str, ...]
    training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    fanout_lut: LowBitRareDirectoryFanoutLUT
    points: Tuple[CsaHcaRareDirectoryBloomRetirementCollisionPoint, ...]


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomRetirementCollisionFanoutPoint:
    """Fanout-budget point on the repeated-key adversarial collision stress."""

    scenario: str
    min_read_blocks_per_token: int
    coverage_target: float
    directory_blocks_per_token: int
    bits_per_entry: int
    hash_count: int
    counter_bits: int
    bank_count: int
    bank_mode: str
    sidecar_salt: int
    rare_occurrences_per_token: int
    colliders_per_rare: int
    rare_tokens: int
    collider_tokens: int
    missing_colliders: int
    mean_slot_overlap: float
    sidecar_state_bytes: float
    fanout_lut_state_bytes: float
    fanout_training_samples: int
    visible_active_rare_rate: float
    rare_false_hca_rate: float
    repaired_relevant_coverage: float
    directory_read_bytes_per_query: float
    directory_entries_read_per_query: float
    token_read_reduction: float


@dataclass(frozen=True)
class CsaHcaRareDirectoryBloomRetirementCollisionFanoutResult:
    """Read-fanout budget sweep for the repeated-key collision case."""

    context_length: int
    block_size: int
    summary_width: int
    global_width: int
    csa_blocks: int
    tail_blocks: int
    hca_threshold: int
    directory_blocks_per_token: int
    rare_token_count: int
    rare_occurrences_per_token: int
    colliders_per_rare: int
    bits_per_entry: int
    hash_count: int
    counter_bits: int
    bank_count: int
    bank_mode: str
    sidecar_salt: int
    min_read_blocks_per_token_values: Tuple[int, ...]
    coverage_targets: Tuple[float, ...]
    train_scenarios: Tuple[str, ...]
    route_training_samples: int
    route_lut: LowBitDirectoryAwareHcaRouteLUT
    points: Tuple[CsaHcaRareDirectoryBloomRetirementCollisionFanoutPoint, ...]


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


@dataclass(frozen=True)
class HcaLazyDecayResult:
    """Quality/cost for lazy epoch-based HCA decay."""

    context_length: int
    vocab_size: int
    hot_tokens: int
    global_width: int
    bits: int
    epoch_bits: int
    decay_interval: int
    threshold: int
    queries: int
    state_bytes: float
    read_bytes_per_query: float
    avg_update_cells_per_token: float
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
    explicit_decay_cells_per_token: float


@dataclass(frozen=True)
class HcaLazyMetadataSweepResult:
    """Sweep of lazy-decay epoch metadata widths and decay intervals."""

    context_length: int
    vocab_size: int
    hot_tokens: int
    global_width: int
    bits: int
    threshold: int
    queries: int
    points: Tuple[HcaLazyDecayResult, ...]


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


class LazyDecayedDenseSummary:
    """Count-min dense summary with per-counter lazy epoch decay."""

    def __init__(
        self,
        vocab_size: int,
        banks: int,
        width: int,
        bits: int,
        decay_interval: int,
        decay_shift: int = 1,
        epoch_bits: int = 16,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if banks <= 0:
            raise ValueError("banks must be positive")
        if width <= 0:
            raise ValueError("width must be positive")
        if bits not in (2, 4, 8):
            raise ValueError("bits must be one of 2, 4, 8")
        if decay_interval <= 0:
            raise ValueError("decay_interval must be positive")
        if decay_shift <= 0:
            raise ValueError("decay_shift must be positive")
        if epoch_bits not in (4, 8, 16):
            raise ValueError("epoch_bits must be one of 4, 8, 16")
        self.vocab_size = vocab_size
        self.banks = banks
        self.width = width
        self.bits = bits
        self.decay_interval = decay_interval
        self.decay_shift = decay_shift
        self.epoch_bits = epoch_bits
        self.max_value = (1 << bits) - 1
        self.counters = np.zeros((banks, width), dtype=np.uint8)
        self.epochs = np.zeros((banks, width), dtype=np.uint16)
        self.steps = 0
        self.touched_update_cells = 0

    @property
    def current_epoch(self) -> int:
        return self.steps // self.decay_interval

    @property
    def state_bytes(self) -> float:
        return self.banks * self.width * (self.bits + self.epoch_bits) / 8

    @property
    def read_bytes_per_query(self) -> float:
        return self.banks * (self.bits + self.epoch_bits) / 8

    def _slots(self, token: int) -> list[int]:
        return [
            keyed_hash(int(token), 1000 + bank) % self.width
            for bank in range(self.banks)
        ]

    def _decay_delta(self, stored_epoch: int, current_epoch: int) -> int:
        return max(0, current_epoch - int(stored_epoch))

    def _effective_value(self, bank: int, slot: int, current_epoch: int) -> int:
        value = int(self.counters[bank, slot])
        delta = self._decay_delta(int(self.epochs[bank, slot]), current_epoch)
        shift = delta * self.decay_shift
        if shift >= self.bits:
            return 0
        return value >> shift

    def update(self, token: int) -> int:
        if not 0 <= int(token) < self.vocab_size:
            raise ValueError("token outside vocab")
        current_epoch = self.current_epoch
        for bank, slot in enumerate(self._slots(token)):
            value = self._effective_value(bank, slot, current_epoch)
            if value < self.max_value:
                value += 1
            self.counters[bank, slot] = value
            self.epochs[bank, slot] = current_epoch
        self.steps += 1
        self.touched_update_cells += self.banks
        return self.banks

    def estimate(self, token: int) -> int:
        if not 0 <= int(token) < self.vocab_size:
            raise ValueError("token outside vocab")
        current_epoch = self.current_epoch
        return min(
            self._effective_value(bank, slot, current_epoch)
            for bank, slot in enumerate(self._slots(token))
        )

    def estimate_all(self) -> np.ndarray:
        estimates = np.zeros(self.vocab_size, dtype=np.uint16)
        for token in range(self.vocab_size):
            estimates[token] = self.estimate(token)
        return estimates

    def effective_counters(self) -> np.ndarray:
        current_epoch = self.current_epoch
        effective = np.zeros_like(self.counters, dtype=np.uint8)
        for bank in range(self.banks):
            for slot in range(self.width):
                effective[bank, slot] = self._effective_value(bank, slot, current_epoch)
        return effective


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
    thresholds: Tuple[int, ...] = (1, 2, 4, 8, 12, 15),
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


def run_csa_hca_block_state_sweep(
    candidates: Tuple[Tuple[int, int, int], ...] = (
        (64, 128, 4),
        (64, 256, 4),
        (128, 128, 4),
        (128, 256, 4),
        (256, 128, 4),
        (256, 256, 4),
    ),
    global_width: int = 2048,
    hca_threshold: int = 8,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> CsaHcaBlockStateSweepResult:
    """Sweep CSA block-summary state under a fixed HCA/CSA routing gate.

    Candidate tuples are ``(block_size, summary_width, csa_blocks)``. The HCA
    global summary and threshold are intentionally held fixed, so differences in
    the output isolate the CSA geometry tradeoff: SRAM state versus score reads,
    selected token reads, and cold-query reliability.
    """

    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")
    if hca_threshold <= 0:
        raise ValueError("hca_threshold must be positive")

    points = []
    reference: CsaHcaPolicyResult | None = None
    for block_size, summary_width, csa_blocks in candidates:
        trial = run_csa_hca_policy_trial(
            summary_width=int(summary_width),
            global_width=global_width,
            thresholds=(hca_threshold,),
            csa_blocks=int(csa_blocks),
            tail_blocks=tail_blocks,
            block_size=int(block_size),
            context_length=context_length,
            queries=queries,
            seed=seed,
        )
        reference = reference or trial
        policy_point = trial.points[0]
        points.append(
            CsaHcaBlockStatePoint(
                block_size=trial.block_size,
                blocks=trial.blocks,
                summary_width=trial.summary_width,
                csa_blocks=policy_point.csa_blocks,
                tail_blocks=policy_point.tail_blocks,
                block_summary_state_bytes=trial.block_summary_state_bytes,
                block_score_bytes_per_query=policy_point.block_score_bytes_per_query,
                hca_query_rate=policy_point.hca_query_rate,
                csa_query_rate=policy_point.csa_query_rate,
                hot_to_hca_rate=policy_point.hot_to_hca_rate,
                cold_to_csa_rate=policy_point.cold_to_csa_rate,
                csa_relevant_hit_rate=policy_point.csa_relevant_hit_rate,
                csa_relevant_coverage=policy_point.csa_relevant_coverage,
                policy_sparse_coverage=policy_point.policy_sparse_coverage,
                token_reads_per_query=policy_point.token_reads_per_query,
                token_read_reduction=policy_point.token_read_reduction,
            )
        )

    assert reference is not None
    return CsaHcaBlockStateSweepResult(
        context_length=reference.context_length,
        banks=reference.banks,
        global_width=reference.global_width,
        bits=reference.bits,
        hca_threshold=hca_threshold,
        queries=reference.queries,
        relevant_query_rate=reference.relevant_query_rate,
        global_summary_state_bytes=reference.global_summary_state_bytes,
        global_summary_read_bytes_per_query=reference.global_summary_read_bytes_per_query,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_sweep(
    candidates: Tuple[Tuple[int, int, int, int], ...] = (
        (128, 128, 4, 0),
        (128, 128, 4, 1),
        (128, 128, 4, 2),
        (128, 128, 4, 6),
        (256, 256, 4, 1),
        (256, 256, 4, 6),
    ),
    global_width: int = 2048,
    hca_threshold: int = 15,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> CsaHcaRareDirectorySweepResult:
    """Repair low-state CSA misses with an exact rare-token block directory.

    Candidate tuples are ``(block_size, summary_width, csa_blocks,
    directory_blocks_per_token)``. The directory stores exact block ids only for
    tokens whose exact count is below the HCA threshold, matching the intended
    split between dense recurrent context and exact sparse recall.
    """

    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")
    if hca_threshold <= 0:
        raise ValueError("hca_threshold must be positive")
    if any(int(candidate[3]) < 0 for candidate in candidates):
        raise ValueError("directory_blocks_per_token must be non-negative")

    points = []
    reference: CsaHcaPolicyResult | None = None
    for block_size, summary_width, csa_blocks, directory_blocks_per_token in candidates:
        config = CompressedBlockIndexConfig(
            context_length=context_length,
            block_size=int(block_size),
            selected_blocks=int(csa_blocks),
            tail_blocks=tail_blocks,
            summary_width=int(summary_width),
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
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=int(directory_blocks_per_token),
        )
        directory_entries = sum(len(blocks) for blocks in directory.values())
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        directory_state_bytes = directory_entries * directory_entry_bytes

        query_tokens = _make_zipf_topic_stream(config, seed=seed + 1, length=config.queries)
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)

        hca_queries = 0
        csa_queries = 0
        directory_queries = 0
        relevant_queries = 0
        csa_relevant = 0
        base_hits = 0
        repaired_hits = 0
        repaired_misses = 0
        base_coverage = 0.0
        repaired_coverage = 0.0
        token_reads = 0.0
        score_bytes = 0.0
        directory_read_bytes = 0.0

        for token in query_tokens:
            token = int(token)
            global_estimate = global_summary.estimate(token)
            route_hca = global_estimate >= hca_threshold
            block_counts = exact_counts.get(token)
            is_relevant = block_counts is not None

            if route_hca:
                selected = recent_blocks
                hca_queries += 1
            else:
                scores = index.estimate_blocks(token)
                base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
                directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
                selected = np.union1d(base_selected, directory_blocks)
                csa_queries += 1
                score_bytes += config.score_bytes_per_query
                if int(directory_blocks_per_token) > 0:
                    directory_read_bytes += directory_entry_bytes * max(1, len(directory_blocks))
                if len(directory_blocks) > 0:
                    directory_queries += 1

            token_reads += len(selected) * config.block_size
            if not is_relevant:
                continue
            relevant_queries += 1
            if route_hca:
                continue

            csa_relevant += 1
            base_hit = _block_hit(base_selected, block_counts)
            repaired_hit = _block_hit(selected, block_counts)
            base_hits += base_hit
            repaired_hits += repaired_hit
            repaired_misses += int((not base_hit) and repaired_hit)
            base_coverage += _occurrence_coverage(base_selected, block_counts)
            repaired_coverage += _occurrence_coverage(selected, block_counts)

        query_denominator = config.queries if config.queries else 1
        csa_denominator = csa_relevant if csa_relevant else 1
        token_reads_per_query = token_reads / query_denominator
        if reference is None:
            reference = CsaHcaPolicyResult(
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
                global_summary_read_bytes_per_query=config.banks * config.bits / 8,
                points=(),
            )
        points.append(
            CsaHcaRareDirectoryPoint(
                block_size=config.block_size,
                blocks=config.blocks,
                summary_width=config.summary_width,
                csa_blocks=config.selected_blocks,
                tail_blocks=config.tail_blocks,
                directory_blocks_per_token=int(directory_blocks_per_token),
                directory_entries=directory_entries,
                directory_entry_bytes=directory_entry_bytes,
                directory_state_bytes=directory_state_bytes,
                block_summary_state_bytes=index.state_bytes,
                block_plus_directory_state_bytes=index.state_bytes + directory_state_bytes,
                block_score_bytes_per_query=score_bytes / query_denominator,
                directory_read_bytes_per_query=directory_read_bytes / query_denominator,
                hca_query_rate=hca_queries / query_denominator,
                csa_query_rate=csa_queries / query_denominator,
                directory_query_rate=directory_queries / query_denominator,
                base_csa_relevant_hit_rate=base_hits / csa_denominator,
                repaired_csa_relevant_hit_rate=repaired_hits / csa_denominator,
                directory_repair_rate=repaired_misses / csa_denominator,
                base_csa_relevant_coverage=base_coverage / csa_denominator,
                repaired_csa_relevant_coverage=repaired_coverage / csa_denominator,
                token_reads_per_query=token_reads_per_query,
                token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
            )
        )

    assert reference is not None
    return CsaHcaRareDirectorySweepResult(
        context_length=reference.context_length,
        banks=reference.banks,
        global_width=reference.global_width,
        bits=reference.bits,
        hca_threshold=hca_threshold,
        queries=reference.queries,
        relevant_query_rate=reference.relevant_query_rate,
        global_summary_state_bytes=reference.global_summary_state_bytes,
        global_summary_read_bytes_per_query=reference.global_summary_read_bytes_per_query,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_stress_sweep(
    scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    directory_blocks: Tuple[int, ...] = (0, 2, 4, 6),
    directory_read_blocks: Tuple[int, ...] | None = None,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    hca_threshold: int = 15,
    directory_guard: bool = False,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> CsaHcaRareDirectoryStressResult:
    """Stress the rare-token directory beyond the smooth topic/noise stream."""

    if len(scenarios) == 0:
        raise ValueError("scenarios must not be empty")
    if len(directory_blocks) == 0:
        raise ValueError("directory_blocks must not be empty")
    if any(int(blocks) < 0 for blocks in directory_blocks):
        raise ValueError("directory block counts must be non-negative")
    if directory_read_blocks is not None and len(directory_read_blocks) != len(directory_blocks):
        raise ValueError("directory_read_blocks must match directory_blocks length")
    if hca_threshold <= 0:
        raise ValueError("hca_threshold must be positive")

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

    points = []
    for scenario_index, scenario in enumerate(scenarios):
        stream, query_tokens, stress_token_count = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
        directory_pairs = []
        if directory_read_blocks is None:
            directory_pairs = [(int(blocks), None) for blocks in directory_blocks]
        else:
            directory_pairs = [
                (int(blocks), int(read_blocks))
                for blocks, read_blocks in zip(directory_blocks, directory_read_blocks)
            ]
        for directory_blocks_per_token, directory_read_blocks_per_token in sorted(set(directory_pairs)):
            directory = _build_rare_block_directory(
                exact_counts=exact_counts,
                hca_threshold=hca_threshold,
                max_blocks_per_token=directory_blocks_per_token,
            )
            directory_entries = sum(len(blocks) for blocks in directory.values())
            directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
            directory_state_bytes = directory_entries * directory_entry_bytes
            points.append(
                _evaluate_rare_directory_stress_point(
                    scenario=scenario,
                    config=config,
                    index=index,
                    global_summary=global_summary,
                    exact_counts=exact_counts,
                    directory=directory,
                    directory_blocks_per_token=directory_blocks_per_token,
                    directory_entry_bytes=directory_entry_bytes,
                    directory_state_bytes=directory_state_bytes,
                    hca_threshold=hca_threshold,
                    directory_guard=directory_guard,
                    directory_read_blocks_per_token=directory_read_blocks_per_token,
                    recent_blocks=recent_blocks,
                    query_tokens=query_tokens,
                    stress_token_count=stress_token_count,
                )
            )

    return CsaHcaRareDirectoryStressResult(
        context_length=config.context_length,
        block_size=config.block_size,
        blocks=config.blocks,
        banks=config.banks,
        summary_width=config.summary_width,
        global_width=global_width,
        bits=config.bits,
        hca_threshold=hca_threshold,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        directory_guard=directory_guard,
        queries=config.queries,
        block_summary_state_bytes=CompressedBlockIndexConfig(
            context_length=context_length,
            block_size=block_size,
            selected_blocks=csa_blocks,
            tail_blocks=tail_blocks,
            summary_width=summary_width,
            queries=queries,
        ).summary_state_bytes,
        global_summary_state_bytes=global_config.state_bytes,
        global_summary_read_bytes_per_query=config.banks * config.bits / 8,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_policy_sweep(
    scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    policies: Tuple[Tuple[str, int, bool, int, int], ...] = (
        ("cheap_t15_read6", 15, False, 6, 6),
        ("guard_t8_read6", 8, True, 6, 6),
        ("guard_t8_read2", 8, True, 6, 2),
        ("cheap_t15_read2", 15, False, 6, 2),
    ),
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    seed: int = 37,
) -> CsaHcaRareDirectoryPolicyResult:
    """Compare hand policy points for rare-directory admission and fanout."""

    if len(policies) == 0:
        raise ValueError("policies must not be empty")
    points = []
    for policy, threshold, guard, stored_blocks, read_blocks in policies:
        result = run_csa_hca_rare_directory_stress_sweep(
            scenarios=scenarios,
            directory_blocks=(int(stored_blocks),),
            directory_read_blocks=(int(read_blocks),),
            block_size=block_size,
            summary_width=summary_width,
            csa_blocks=csa_blocks,
            global_width=global_width,
            hca_threshold=int(threshold),
            directory_guard=bool(guard),
            tail_blocks=tail_blocks,
            context_length=context_length,
            queries=queries,
            seed=seed,
        )
        for point in result.points:
            points.append(
                CsaHcaRareDirectoryPolicyPoint(
                    policy=policy,
                    scenario=point.scenario,
                    hca_threshold=int(threshold),
                    directory_guard=bool(guard),
                    directory_blocks_per_token=point.directory_blocks_per_token,
                    directory_read_blocks_per_token=point.directory_read_blocks_per_token,
                    directory_state_bytes=point.directory_state_bytes,
                    rare_false_hca_rate=point.rare_false_hca_rate,
                    repaired_relevant_hit_rate=point.repaired_relevant_hit_rate,
                    repaired_relevant_coverage=point.repaired_relevant_coverage,
                    directory_read_bytes_per_query=point.directory_read_bytes_per_query,
                    token_reads_per_query=point.token_reads_per_query,
                    token_read_reduction=point.token_read_reduction,
                )
            )

    return CsaHcaRareDirectoryPolicyResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=csa_blocks,
        tail_blocks=tail_blocks,
        queries=queries,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_adaptive_policy_sweep(
    scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    policies: Tuple[Tuple[str, int, bool, int, int, int, int], ...] = (
        ("cheap_t15_span2to6", 15, False, 6, 2, 6, 128),
        ("guard_t8_span2to6", 8, True, 6, 2, 6, 128),
        ("guard_t8_span2to5", 8, True, 6, 2, 5, 128),
        ("guard_t8_span2to4", 8, True, 6, 2, 4, 128),
    ),
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    seed: int = 37,
) -> CsaHcaRareDirectoryAdaptivePolicyResult:
    """Compare metadata-driven directory read fanout policies.

    Each policy stores up to ``directory_blocks_per_token`` exact rare-token
    block IDs, but starts from a small read budget and expands only when the
    stored IDs are spread across enough blocks. The spread class is a compact
    directory-header metadata proxy for a future trained fanout LUT.
    """

    if len(scenarios) == 0:
        raise ValueError("scenarios must not be empty")
    if len(policies) == 0:
        raise ValueError("policies must not be empty")

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

    points = []
    for scenario_index, scenario in enumerate(scenarios):
        max_threshold = max(int(policy[1]) for policy in policies)
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=max_threshold,
            seed=seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        for (
            policy,
            threshold,
            guard,
            stored_blocks,
            base_read_blocks,
            expanded_read_blocks,
            spread_threshold_blocks,
        ) in policies:
            if int(threshold) <= 0:
                raise ValueError("policy thresholds must be positive")
            if int(stored_blocks) < 0:
                raise ValueError("stored directory blocks must be non-negative")
            if int(base_read_blocks) < 0 or int(expanded_read_blocks) < 0:
                raise ValueError("directory read blocks must be non-negative")
            if int(spread_threshold_blocks) < 0:
                raise ValueError("spread thresholds must be non-negative")

            directory = _build_rare_block_directory(
                exact_counts=exact_counts,
                hca_threshold=int(threshold),
                max_blocks_per_token=int(stored_blocks),
            )
            directory_entries = sum(len(blocks) for blocks in directory.values())
            directory_state_bytes = directory_entries * directory_entry_bytes
            points.append(
                _evaluate_rare_directory_adaptive_policy_point(
                    policy=policy,
                    scenario=scenario,
                    config=config,
                    index=index,
                    global_summary=global_summary,
                    exact_counts=exact_counts,
                    directory=directory,
                    directory_blocks_per_token=int(stored_blocks),
                    directory_entry_bytes=directory_entry_bytes,
                    directory_state_bytes=directory_state_bytes,
                    hca_threshold=int(threshold),
                    directory_guard=bool(guard),
                    base_read_blocks_per_token=int(base_read_blocks),
                    expanded_read_blocks_per_token=int(expanded_read_blocks),
                    spread_threshold_blocks=int(spread_threshold_blocks),
                    recent_blocks=recent_blocks,
                    query_tokens=query_tokens,
                )
            )

    return CsaHcaRareDirectoryAdaptivePolicyResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        queries=config.queries,
        points=tuple(points),
    )


def train_rare_directory_fanout_lut(
    train_scenarios: Tuple[str, ...] = (
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 8,
    directory_guard: bool = True,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    seed: int = 19,
) -> Tuple[LowBitRareDirectoryFanoutLUT, int]:
    """Train a low-bit fanout LUT from self-supervised coverage labels."""

    if len(train_scenarios) == 0:
        raise ValueError("train_scenarios must not be empty")
    if hca_threshold <= 0:
        raise ValueError("hca_threshold must be positive")
    if directory_blocks_per_token < 0:
        raise ValueError("directory_blocks_per_token must be non-negative")
    if min_read_blocks_per_token < 0:
        raise ValueError("min_read_blocks_per_token must be non-negative")
    if not 0.0 <= coverage_target <= 1.0:
        raise ValueError("coverage_target must be in [0, 1]")

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
    table_size = (directory_blocks_per_token + 1) * (len(span_thresholds) + 1) * (
        max_overlap_bucket + 1
    )
    coverage_sums = np.zeros((table_size, directory_blocks_per_token + 1), dtype=np.float64)
    sample_counts = np.zeros(table_size, dtype=np.int32)

    for scenario_index, scenario in enumerate(train_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)

        for token in query_tokens:
            token = int(token)
            directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
            if len(directory_blocks) == 0:
                continue
            route_hca = global_summary.estimate(token) >= hca_threshold
            if directory_guard:
                route_hca = False
            if route_hca:
                continue

            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            block_counts = exact_counts.get(token)
            if block_counts is None:
                continue

            feature = _rare_fanout_lut_index(
                directory_blocks=directory_blocks,
                base_selected=base_selected,
                max_entries=directory_blocks_per_token,
                span_thresholds=span_thresholds,
                max_overlap_bucket=max_overlap_bucket,
            )
            sample_counts[feature] += 1
            entry_count = min(len(directory_blocks), directory_blocks_per_token)
            last_coverage = 0.0
            for read_limit in range(entry_count + 1):
                selected = np.union1d(base_selected, directory_blocks[:read_limit])
                last_coverage = _occurrence_coverage(selected, block_counts)
                coverage_sums[feature, read_limit] += last_coverage
            for read_limit in range(entry_count + 1, directory_blocks_per_token + 1):
                coverage_sums[feature, read_limit] += last_coverage

    fanouts = []
    training_samples = int(np.sum(sample_counts))
    for feature in range(table_size):
        entry_count, _, _ = _decode_rare_fanout_lut_index(
            feature,
            span_bucket_count=len(span_thresholds) + 1,
            overlap_bucket_count=max_overlap_bucket + 1,
        )
        max_read = min(entry_count, directory_blocks_per_token)
        min_read = min(max_read, min_read_blocks_per_token)
        if sample_counts[feature] == 0:
            fanouts.append(min_read)
            continue

        avg_coverage = coverage_sums[feature] / sample_counts[feature]
        chosen = max_read
        for read_limit in range(min_read, max_read + 1):
            if avg_coverage[read_limit] >= coverage_target:
                chosen = read_limit
                break
        fanouts.append(chosen)

    return (
        LowBitRareDirectoryFanoutLUT(
            fanouts=tuple(int(value) for value in fanouts),
            max_entries=directory_blocks_per_token,
            span_thresholds=span_thresholds,
            max_overlap_bucket=max_overlap_bucket,
        ),
        training_samples,
    )


def train_rare_directory_probe_lut(
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 8,
    directory_blocks_per_token: int = 6,
    probe_positive_rate_threshold: float = 0.25,
    hca_spread_thresholds: Tuple[int, ...] = (1, 3, 7),
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    seed: int = 23,
) -> Tuple[LowBitRareDirectoryProbeLUT, int]:
    """Train a low-bit HCA-confidence LUT for selective directory probes."""

    if len(train_scenarios) == 0:
        raise ValueError("train_scenarios must not be empty")
    if hca_threshold <= 0:
        raise ValueError("hca_threshold must be positive")
    if directory_blocks_per_token < 0:
        raise ValueError("directory_blocks_per_token must be non-negative")
    if not 0.0 <= probe_positive_rate_threshold <= 1.0:
        raise ValueError("probe_positive_rate_threshold must be in [0, 1]")

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
    max_counter = global_config.max_value
    max_saturation_bucket = config.banks
    table_size = (max_counter + 1) * (len(hca_spread_thresholds) + 1) * (
        max_saturation_bucket + 1
    )
    positive = np.zeros(table_size, dtype=np.int32)
    negative = np.zeros(table_size, dtype=np.int32)

    for scenario_index, scenario in enumerate(train_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=seed + scenario_index * 997,
        )
        global_summary = LowBitDenseContext(global_config)
        for token in stream:
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )

        for token in query_tokens:
            token = int(token)
            counter_values = _dense_counter_values(global_summary, token)
            if min(counter_values) < hca_threshold:
                continue
            feature = _hca_probe_lut_index(
                counter_values=counter_values,
                max_counter=max_counter,
                spread_thresholds=hca_spread_thresholds,
                max_saturation_bucket=max_saturation_bucket,
            )
            if token in directory:
                positive[feature] += 1
            else:
                negative[feature] += 1

    probes = []
    for pos, neg in zip(positive, negative):
        if pos == 0 and neg == 0:
            probes.append(True)
        else:
            probes.append(_safe_divide(pos, pos + neg) >= probe_positive_rate_threshold)

    return (
        LowBitRareDirectoryProbeLUT(
            probes=tuple(bool(value) for value in probes),
            max_counter=max_counter,
            spread_thresholds=hca_spread_thresholds,
            max_saturation_bucket=max_saturation_bucket,
        ),
        int(np.sum(positive + negative)),
    )


def train_hca_route_lut(
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 15,
    route_positive_rate_threshold: float = 0.50,
    hca_spread_thresholds: Tuple[int, ...] = (1, 3, 7),
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    seed: int = 29,
) -> Tuple[LowBitHcaRouteLUT, int]:
    """Train an HCA route LUT from exact frequent-token labels."""

    if len(train_scenarios) == 0:
        raise ValueError("train_scenarios must not be empty")
    if hca_threshold <= 0:
        raise ValueError("hca_threshold must be positive")
    if not 0.0 <= route_positive_rate_threshold <= 1.0:
        raise ValueError("route_positive_rate_threshold must be in [0, 1]")

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
    max_counter = global_config.max_value
    max_saturation_bucket = config.banks
    table_size = (max_counter + 1) * (len(hca_spread_thresholds) + 1) * (
        max_saturation_bucket + 1
    )
    positive = np.zeros(table_size, dtype=np.int32)
    negative = np.zeros(table_size, dtype=np.int32)

    for scenario_index, scenario in enumerate(train_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=seed + scenario_index * 997,
        )
        global_summary = LowBitDenseContext(global_config)
        for token in stream:
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        for token in query_tokens:
            token = int(token)
            counter_values = _dense_counter_values(global_summary, token)
            feature = _hca_control_lut_index(
                counter_values=counter_values,
                max_counter=max_counter,
                spread_thresholds=hca_spread_thresholds,
                max_saturation_bucket=max_saturation_bucket,
            )
            exact_total = sum(int(value) for value in exact_counts.get(token, {}).values())
            if exact_total >= hca_threshold:
                positive[feature] += 1
            else:
                negative[feature] += 1

    routes_hca = []
    for pos, neg in zip(positive, negative):
        if pos == 0 and neg == 0:
            routes_hca.append(False)
        else:
            routes_hca.append(_safe_divide(pos, pos + neg) >= route_positive_rate_threshold)

    return (
        LowBitHcaRouteLUT(
            routes_hca=tuple(bool(value) for value in routes_hca),
            max_counter=max_counter,
            spread_thresholds=hca_spread_thresholds,
            max_saturation_bucket=max_saturation_bucket,
        ),
        int(np.sum(positive + negative)),
    )


def train_directory_aware_hca_route_lut(
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    route_positive_rate_threshold: float = 0.50,
    hca_spread_thresholds: Tuple[int, ...] = (1, 3, 7),
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    seed: int = 31,
) -> Tuple[LowBitDirectoryAwareHcaRouteLUT, int]:
    """Train an HCA route LUT with a rare-directory presence sidecar bit."""

    if len(train_scenarios) == 0:
        raise ValueError("train_scenarios must not be empty")
    if hca_threshold <= 0:
        raise ValueError("hca_threshold must be positive")
    if directory_blocks_per_token < 0:
        raise ValueError("directory_blocks_per_token must be non-negative")
    if not 0.0 <= route_positive_rate_threshold <= 1.0:
        raise ValueError("route_positive_rate_threshold must be in [0, 1]")

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
    max_counter = global_config.max_value
    max_saturation_bucket = config.banks
    hca_buckets = (max_counter + 1) * (len(hca_spread_thresholds) + 1) * (
        max_saturation_bucket + 1
    )
    table_size = hca_buckets * 2
    positive = np.zeros(table_size, dtype=np.int32)
    negative = np.zeros(table_size, dtype=np.int32)

    for scenario_index, scenario in enumerate(train_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=seed + scenario_index * 997,
        )
        global_summary = LowBitDenseContext(global_config)
        for token in stream:
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        for token in query_tokens:
            token = int(token)
            counter_values = _dense_counter_values(global_summary, token)
            directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
            feature = _directory_aware_hca_route_lut_index(
                counter_values=counter_values,
                directory_blocks=directory_blocks,
                max_counter=max_counter,
                spread_thresholds=hca_spread_thresholds,
                max_saturation_bucket=max_saturation_bucket,
            )
            exact_total = sum(int(value) for value in exact_counts.get(token, {}).values())
            if exact_total >= hca_threshold:
                positive[feature] += 1
            else:
                negative[feature] += 1

    routes_hca = []
    for pos, neg in zip(positive, negative):
        if pos == 0 and neg == 0:
            routes_hca.append(False)
        else:
            routes_hca.append(_safe_divide(pos, pos + neg) >= route_positive_rate_threshold)

    return (
        LowBitDirectoryAwareHcaRouteLUT(
            routes_hca=tuple(bool(value) for value in routes_hca),
            max_counter=max_counter,
            spread_thresholds=hca_spread_thresholds,
            max_saturation_bucket=max_saturation_bucket,
        ),
        int(np.sum(positive + negative)),
    )


def run_csa_hca_rare_directory_learned_fanout_sweep(
    train_scenarios: Tuple[str, ...] = (
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 8,
    directory_guard: bool = True,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 19,
    eval_seed: int = 37,
) -> CsaHcaRareDirectoryLearnedFanoutResult:
    """Train and evaluate a low-bit fanout LUT for rare-directory reads."""

    lut, training_samples = train_rare_directory_fanout_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=directory_guard,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        directory_state_bytes = sum(len(blocks) for blocks in directory.values()) * directory_entry_bytes
        points.append(
            _evaluate_rare_directory_lut_fanout_point(
                policy="learned_lut",
                scenario=scenario,
                config=config,
                index=index,
                global_summary=global_summary,
                exact_counts=exact_counts,
                directory=directory,
                directory_blocks_per_token=directory_blocks_per_token,
                directory_entry_bytes=directory_entry_bytes,
                directory_state_bytes=directory_state_bytes,
                hca_threshold=hca_threshold,
                directory_guard=directory_guard,
                lut=lut,
                min_read_blocks_per_token=min_read_blocks_per_token,
                recent_blocks=_recent_blocks(config.blocks, config.tail_blocks),
                query_tokens=query_tokens,
            )
        )

    return CsaHcaRareDirectoryLearnedFanoutResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_guard=directory_guard,
        directory_blocks_per_token=directory_blocks_per_token,
        coverage_target=coverage_target,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=training_samples,
        lut=lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_joint_policy_sweep(
    train_scenarios: Tuple[str, ...] = (
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    policies: Tuple[Tuple[str, str], ...] = (
        ("never_probe", "never"),
        ("confidence_probe", "confidence"),
        ("hca_probe", "hca_only"),
        ("always_probe", "always"),
    ),
    hca_threshold: int = 8,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    probe_positive_rate_threshold: float = 0.25,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 19,
    eval_seed: int = 37,
) -> CsaHcaRareDirectoryJointPolicyResult:
    """Evaluate joint rare-directory probe/admission/fanout policies."""

    if len(policies) == 0:
        raise ValueError("policies must not be empty")

    lut, training_samples = train_rare_directory_fanout_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    probe_train_scenarios = tuple(dict.fromkeys(("zipf_reference",) + train_scenarios))
    probe_lut, probe_training_samples = train_rare_directory_probe_lut(
        train_scenarios=probe_train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        probe_positive_rate_threshold=probe_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        directory_state_bytes = sum(len(blocks) for blocks in directory.values()) * directory_entry_bytes
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)

        for policy, probe_mode in policies:
            points.append(
                _evaluate_rare_directory_joint_policy_point(
                    policy=policy,
                    scenario=scenario,
                    config=config,
                    index=index,
                    global_summary=global_summary,
                    exact_counts=exact_counts,
                    directory=directory,
                    directory_blocks_per_token=directory_blocks_per_token,
                    directory_entry_bytes=directory_entry_bytes,
                    directory_state_bytes=directory_state_bytes,
                    hca_threshold=hca_threshold,
                    probe_mode=probe_mode,
                    lut=lut,
                    probe_lut=probe_lut,
                    min_read_blocks_per_token=min_read_blocks_per_token,
                    recent_blocks=recent_blocks,
                    query_tokens=query_tokens,
                )
            )

    return CsaHcaRareDirectoryJointPolicyResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        coverage_target=coverage_target,
        probe_positive_rate_threshold=probe_positive_rate_threshold,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=training_samples + probe_training_samples,
        lut=lut,
        probe_lut=probe_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_joint_threshold_sweep(
    thresholds: Tuple[int, ...] = (6, 8, 10, 12, 15),
    policy: Tuple[str, str] = ("confidence_probe", "confidence"),
    train_scenarios: Tuple[str, ...] = (
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "split_rare",
        "repeated_name",
    ),
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    probe_positive_rate_threshold: float = 0.25,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 19,
    eval_seed: int = 37,
) -> CsaHcaRareDirectoryJointThresholdResult:
    """Sweep HCA thresholds after joint probe/fanout control is available."""

    if len(thresholds) == 0:
        raise ValueError("thresholds must not be empty")
    if len(policy) != 2:
        raise ValueError("policy must be a (name, probe_mode) tuple")
    sorted_thresholds = tuple(sorted({int(threshold) for threshold in thresholds}))
    if any(threshold <= 0 for threshold in sorted_thresholds):
        raise ValueError("thresholds must be positive")

    points = []
    for threshold in sorted_thresholds:
        result = run_csa_hca_rare_directory_joint_policy_sweep(
            train_scenarios=train_scenarios,
            eval_scenarios=eval_scenarios,
            policies=(policy,),
            hca_threshold=threshold,
            directory_blocks_per_token=directory_blocks_per_token,
            min_read_blocks_per_token=min_read_blocks_per_token,
            coverage_target=coverage_target,
            probe_positive_rate_threshold=probe_positive_rate_threshold,
            span_thresholds=span_thresholds,
            max_overlap_bucket=max_overlap_bucket,
            block_size=block_size,
            summary_width=summary_width,
            csa_blocks=csa_blocks,
            global_width=global_width,
            tail_blocks=tail_blocks,
            context_length=context_length,
            queries=queries,
            train_seed=train_seed,
            eval_seed=eval_seed,
        )
        points.extend(result.points)

    return CsaHcaRareDirectoryJointThresholdResult(
        thresholds=sorted_thresholds,
        policy=policy[0],
        probe_mode=policy[1],
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=csa_blocks,
        tail_blocks=tail_blocks,
        directory_blocks_per_token=directory_blocks_per_token,
        coverage_target=coverage_target,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_route_lut_sweep(
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 29,
    eval_seed: int = 37,
) -> CsaHcaRareDirectoryRouteLutResult:
    """Train an HCA-route LUT and evaluate it with rare-directory fanout."""

    route_lut, route_training_samples = train_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        directory_state_bytes = sum(len(blocks) for blocks in directory.values()) * directory_entry_bytes
        points.append(
            _evaluate_rare_directory_hca_route_lut_point(
                scenario=scenario,
                config=config,
                index=index,
                global_summary=global_summary,
                exact_counts=exact_counts,
                directory=directory,
                directory_blocks_per_token=directory_blocks_per_token,
                directory_entry_bytes=directory_entry_bytes,
                directory_state_bytes=directory_state_bytes,
                hca_threshold=hca_threshold,
                route_lut=route_lut,
                fanout_lut=fanout_lut,
                min_read_blocks_per_token=min_read_blocks_per_token,
                recent_blocks=_recent_blocks(config.blocks, config.tail_blocks),
                query_tokens=query_tokens,
            )
        )

    return CsaHcaRareDirectoryRouteLutResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        coverage_target=coverage_target,
        route_positive_rate_threshold=route_positive_rate_threshold,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_aware_route_lut_sweep(
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    route_feature_read_bytes: float = 1 / 8,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 37,
) -> CsaHcaRareDirectoryAwareRouteLutResult:
    """Train a directory-aware HCA-route LUT and evaluate it with fanout."""

    if route_feature_read_bytes < 0:
        raise ValueError("route_feature_read_bytes must be non-negative")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        directory_presence_state_bytes = len(directory) / 8
        directory_state_bytes = (
            sum(len(blocks) for blocks in directory.values()) * directory_entry_bytes
            + directory_presence_state_bytes
        )
        points.append(
            _evaluate_rare_directory_aware_hca_route_lut_point(
                scenario=scenario,
                config=config,
                index=index,
                global_summary=global_summary,
                exact_counts=exact_counts,
                directory=directory,
                directory_blocks_per_token=directory_blocks_per_token,
                directory_entry_bytes=directory_entry_bytes,
                directory_state_bytes=directory_state_bytes,
                hca_threshold=hca_threshold,
                route_lut=route_lut,
                fanout_lut=fanout_lut,
                min_read_blocks_per_token=min_read_blocks_per_token,
                recent_blocks=_recent_blocks(config.blocks, config.tail_blocks),
                query_tokens=query_tokens,
                route_feature_read_bytes=route_feature_read_bytes,
            )
        )

    return CsaHcaRareDirectoryAwareRouteLutResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        coverage_target=coverage_target,
        route_positive_rate_threshold=route_positive_rate_threshold,
        route_feature_read_bytes=route_feature_read_bytes,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_presence_sidecar_sweep(
    false_positive_rates: Tuple[float, ...] = (0.0, 0.01, 0.10, 0.25),
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    route_feature_read_bytes: float = 1 / 8,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 37,
    sidecar_salt: int = 6113,
) -> CsaHcaRareDirectoryPresenceSidecarResult:
    """Stress the directory-presence sidecar with Bloom-like false positives."""

    if len(false_positive_rates) == 0:
        raise ValueError("false_positive_rates must not be empty")
    sorted_rates = tuple(sorted({float(rate) for rate in false_positive_rates}))
    if any(rate < 0.0 or rate >= 1.0 for rate in sorted_rates):
        raise ValueError("false_positive_rates must be in [0, 1)")
    if route_feature_read_bytes < 0:
        raise ValueError("route_feature_read_bytes must be non-negative")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        directory_entry_state_bytes = sum(len(blocks) for blocks in directory.values()) * directory_entry_bytes
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
        for rate in sorted_rates:
            sidecar_state_bytes = _presence_sidecar_state_bytes(len(directory), rate)
            points.append(
                _evaluate_rare_directory_presence_sidecar_point(
                    false_positive_rate=rate,
                    scenario=scenario,
                    config=config,
                    index=index,
                    global_summary=global_summary,
                    exact_counts=exact_counts,
                    directory=directory,
                    directory_blocks_per_token=directory_blocks_per_token,
                    directory_entry_bytes=directory_entry_bytes,
                    directory_state_bytes=directory_entry_state_bytes + sidecar_state_bytes,
                    sidecar_state_bytes=sidecar_state_bytes,
                    hca_threshold=hca_threshold,
                    route_lut=route_lut,
                    fanout_lut=fanout_lut,
                    min_read_blocks_per_token=min_read_blocks_per_token,
                    recent_blocks=recent_blocks,
                    query_tokens=query_tokens,
                    route_feature_read_bytes=route_feature_read_bytes,
                    sidecar_salt=sidecar_salt + scenario_index * 997,
                )
            )

    return CsaHcaRareDirectoryPresenceSidecarResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        coverage_target=coverage_target,
        route_positive_rate_threshold=route_positive_rate_threshold,
        route_feature_read_bytes=route_feature_read_bytes,
        false_positive_rates=sorted_rates,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_bloom_sidecar_sweep(
    bits_per_entry_options: Tuple[int, ...] = (4, 8, 12),
    hash_count_options: Tuple[int, ...] = (2, 3, 4),
    bank_count: int = 8,
    bank_mode: str = "modulo",
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 37,
    sidecar_salt: int = 9173,
) -> CsaHcaRareDirectoryBloomSidecarResult:
    """Evaluate a concrete Bloom-style rare-directory presence sidecar."""

    if len(bits_per_entry_options) == 0:
        raise ValueError("bits_per_entry_options must not be empty")
    if len(hash_count_options) == 0:
        raise ValueError("hash_count_options must not be empty")
    bits_options = tuple(sorted({int(value) for value in bits_per_entry_options}))
    hash_options = tuple(sorted({int(value) for value in hash_count_options}))
    if any(value <= 0 for value in bits_options):
        raise ValueError("bits_per_entry_options must be positive")
    if any(value <= 0 for value in hash_options):
        raise ValueError("hash_count_options must be positive")
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if bank_mode not in ("modulo", "by_hash", "hash_slot"):
        raise ValueError("bank_mode must be one of: modulo, by_hash, hash_slot")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        directory_entry_state_bytes = sum(len(blocks) for blocks in directory.values()) * directory_entry_bytes
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
        directory_tokens = tuple(sorted(int(token) for token in directory))

        for bits_per_entry in bits_options:
            for hash_count in hash_options:
                bit_count = max(1, int(len(directory_tokens) * bits_per_entry))
                sidecar = LowBitPresenceBloomSidecar(
                    bit_count=bit_count,
                    hash_count=hash_count,
                    bank_count=bank_count,
                    bank_mode=bank_mode,
                    salt=sidecar_salt + scenario_index * 997 + bits_per_entry * 31 + hash_count,
                )
                for token in directory_tokens:
                    sidecar.insert(token)
                points.append(
                    _evaluate_rare_directory_bloom_sidecar_point(
                        bits_per_entry=bits_per_entry,
                        scenario=scenario,
                        config=config,
                        index=index,
                        global_summary=global_summary,
                        exact_counts=exact_counts,
                        directory=directory,
                        directory_blocks_per_token=directory_blocks_per_token,
                        directory_entry_bytes=directory_entry_bytes,
                        directory_state_bytes=directory_entry_state_bytes + sidecar.state_bytes,
                        hca_threshold=hca_threshold,
                        route_lut=route_lut,
                        fanout_lut=fanout_lut,
                        sidecar=sidecar,
                        min_read_blocks_per_token=min_read_blocks_per_token,
                        recent_blocks=recent_blocks,
                        query_tokens=query_tokens,
                        directory_tokens=directory_tokens,
                    )
                )

    return CsaHcaRareDirectoryBloomSidecarResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        coverage_target=coverage_target,
        route_positive_rate_threshold=route_positive_rate_threshold,
        bits_per_entry_options=bits_options,
        hash_count_options=hash_options,
        bank_count=bank_count,
        bank_mode=bank_mode,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_bloom_salt_sweep(
    salt_count: int = 16,
    bits_per_entry: int = 8,
    hash_count: int = 3,
    bank_count: int = 8,
    bank_mode: str = "modulo",
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = ("zipf_reference",),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 37,
    sidecar_salt: int = 9173,
    sidecar_salt_stride: int = 1543,
) -> CsaHcaRareDirectoryBloomSaltResult:
    """Sweep Bloom-sidecar hash salts to expose hot-token false positives."""

    if salt_count <= 0:
        raise ValueError("salt_count must be positive")
    if bits_per_entry <= 0:
        raise ValueError("bits_per_entry must be positive")
    if hash_count <= 0:
        raise ValueError("hash_count must be positive")
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if bank_mode not in ("modulo", "by_hash", "hash_slot"):
        raise ValueError("bank_mode must be one of: modulo, by_hash, hash_slot")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
        directory_tokens = tuple(sorted(int(token) for token in directory))
        bit_count = max(1, int(len(directory_tokens) * bits_per_entry))

        for salt_index in range(salt_count):
            current_salt = sidecar_salt + scenario_index * 997 + salt_index * sidecar_salt_stride
            sidecar = LowBitPresenceBloomSidecar(
                bit_count=bit_count,
                hash_count=hash_count,
                bank_count=bank_count,
                bank_mode=bank_mode,
                salt=current_salt,
            )
            for token in directory_tokens:
                sidecar.insert(token)
            points.append(
                _evaluate_rare_directory_bloom_salt_point(
                    salt_index=salt_index,
                    sidecar_salt=current_salt,
                    bits_per_entry=bits_per_entry,
                    scenario=scenario,
                    config=config,
                    index=index,
                    global_summary=global_summary,
                    exact_counts=exact_counts,
                    directory=directory,
                    directory_blocks_per_token=directory_blocks_per_token,
                    directory_entry_bytes=directory_entry_bytes,
                    hca_threshold=hca_threshold,
                    route_lut=route_lut,
                    fanout_lut=fanout_lut,
                    sidecar=sidecar,
                    min_read_blocks_per_token=min_read_blocks_per_token,
                    recent_blocks=recent_blocks,
                    query_tokens=query_tokens,
                )
            )

    return CsaHcaRareDirectoryBloomSaltResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        bits_per_entry=bits_per_entry,
        hash_count=hash_count,
        bank_count=bank_count,
        bank_mode=bank_mode,
        salt_count=salt_count,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_bloom_bank_sweep(
    bank_modes: Tuple[str, ...] = ("modulo", "by_hash", "hash_slot"),
    salt_count: int = 16,
    bits_per_entry: int = 8,
    hash_count: int = 3,
    bank_count: int = 8,
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = ("zipf_reference",),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 37,
    sidecar_salt: int = 9173,
    sidecar_salt_stride: int = 1543,
) -> CsaHcaRareDirectoryBloomBankResult:
    """Compare physical bank mappings for the Bloom presence sidecar."""

    if len(bank_modes) == 0:
        raise ValueError("bank_modes must not be empty")
    clean_modes = tuple(dict.fromkeys(str(mode) for mode in bank_modes))
    if any(mode not in ("modulo", "by_hash", "hash_slot") for mode in clean_modes):
        raise ValueError("bank_modes must contain only: modulo, by_hash, hash_slot")
    if salt_count <= 0:
        raise ValueError("salt_count must be positive")
    if bits_per_entry <= 0:
        raise ValueError("bits_per_entry must be positive")
    if hash_count <= 0:
        raise ValueError("hash_count must be positive")
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
        directory_tokens = tuple(sorted(int(token) for token in directory))
        bit_count = max(1, int(len(directory_tokens) * bits_per_entry))

        for bank_mode in clean_modes:
            for salt_index in range(salt_count):
                current_salt = sidecar_salt + scenario_index * 997 + salt_index * sidecar_salt_stride
                sidecar = LowBitPresenceBloomSidecar(
                    bit_count=bit_count,
                    hash_count=hash_count,
                    bank_count=bank_count,
                    bank_mode=bank_mode,
                    salt=current_salt,
                )
                for token in directory_tokens:
                    sidecar.insert(token)
                points.append(
                    _evaluate_rare_directory_bloom_salt_point(
                        salt_index=salt_index,
                        sidecar_salt=current_salt,
                        bits_per_entry=bits_per_entry,
                        scenario=scenario,
                        config=config,
                        index=index,
                        global_summary=global_summary,
                        exact_counts=exact_counts,
                        directory=directory,
                        directory_blocks_per_token=directory_blocks_per_token,
                        directory_entry_bytes=directory_entry_bytes,
                        hca_threshold=hca_threshold,
                        route_lut=route_lut,
                        fanout_lut=fanout_lut,
                        sidecar=sidecar,
                        min_read_blocks_per_token=min_read_blocks_per_token,
                        recent_blocks=recent_blocks,
                        query_tokens=query_tokens,
                    )
                )

    return CsaHcaRareDirectoryBloomBankResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        bits_per_entry=bits_per_entry,
        hash_count=hash_count,
        bank_count=bank_count,
        bank_modes=clean_modes,
        salt_count=salt_count,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_bloom_salt_selection_sweep(
    salt_count: int = 16,
    bits_per_entry: int = 8,
    hash_count: int = 3,
    bank_count: int = 8,
    bank_mode: str = "by_hash",
    route_train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    selection_scenarios: Tuple[str, ...] = ("zipf_reference",),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    selection_seed: int = 41,
    eval_seed: int = 37,
    sidecar_salt: int = 9173,
    sidecar_salt_stride: int = 1543,
) -> CsaHcaRareDirectoryBloomSaltSelectionResult:
    """Select a Bloom-sidecar salt by minimizing hot-token false positives."""

    if salt_count <= 0:
        raise ValueError("salt_count must be positive")
    if bits_per_entry <= 0:
        raise ValueError("bits_per_entry must be positive")
    if hash_count <= 0:
        raise ValueError("hash_count must be positive")
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if bank_mode not in ("modulo", "by_hash", "hash_slot"):
        raise ValueError("bank_mode must be one of: modulo, by_hash, hash_slot")
    if len(selection_scenarios) == 0:
        raise ValueError("selection_scenarios must not be empty")
    if len(eval_scenarios) == 0:
        raise ValueError("eval_scenarios must not be empty")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=route_train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in route_train_scenarios if s != "zipf_reference")
        or route_train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    def evaluate_scenario(
        scenario: str,
        seed: int,
        salts: Tuple[Tuple[int, int], ...],
    ) -> Tuple[CsaHcaRareDirectoryBloomSaltPoint, ...]:
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=seed,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
        directory_tokens = tuple(sorted(int(token) for token in directory))
        bit_count = max(1, int(len(directory_tokens) * bits_per_entry))

        points = []
        for salt_index, current_salt in salts:
            sidecar = LowBitPresenceBloomSidecar(
                bit_count=bit_count,
                hash_count=hash_count,
                bank_count=bank_count,
                bank_mode=bank_mode,
                salt=current_salt,
            )
            for token in directory_tokens:
                sidecar.insert(token)
            points.append(
                _evaluate_rare_directory_bloom_salt_point(
                    salt_index=salt_index,
                    sidecar_salt=current_salt,
                    bits_per_entry=bits_per_entry,
                    scenario=scenario,
                    config=config,
                    index=index,
                    global_summary=global_summary,
                    exact_counts=exact_counts,
                    directory=directory,
                    directory_blocks_per_token=directory_blocks_per_token,
                    directory_entry_bytes=directory_entry_bytes,
                    hca_threshold=hca_threshold,
                    route_lut=route_lut,
                    fanout_lut=fanout_lut,
                    sidecar=sidecar,
                    min_read_blocks_per_token=min_read_blocks_per_token,
                    recent_blocks=recent_blocks,
                    query_tokens=query_tokens,
                )
            )
        return tuple(points)

    candidate_salts = tuple(
        (salt_index, sidecar_salt + salt_index * sidecar_salt_stride)
        for salt_index in range(salt_count)
    )
    selection_points = []
    for scenario_index, scenario in enumerate(selection_scenarios):
        selection_points.extend(
            evaluate_scenario(
                scenario=scenario,
                seed=selection_seed + scenario_index * 997,
                salts=candidate_salts,
            )
        )

    score_by_salt = {}
    for salt_index, _ in candidate_salts:
        salt_points = [point for point in selection_points if point.salt_index == salt_index]
        mean_hot_fp = sum(point.hot_sidecar_false_positive_rate for point in salt_points) / len(salt_points)
        mean_fp = sum(point.sidecar_false_positive_query_rate for point in salt_points) / len(salt_points)
        mean_hca = sum(point.hca_query_rate for point in salt_points) / len(salt_points)
        mean_conflict = sum(point.query_bank_conflict_rate for point in salt_points) / len(salt_points)
        score_by_salt[salt_index] = (mean_hot_fp, mean_fp, -mean_hca, mean_conflict)
    selected_salt_index = min(score_by_salt, key=lambda salt_index: score_by_salt[salt_index])
    selected_sidecar_salt = sidecar_salt + selected_salt_index * sidecar_salt_stride

    eval_points = []
    selected = ((selected_salt_index, selected_sidecar_salt),)
    for scenario_index, scenario in enumerate(eval_scenarios):
        eval_points.extend(
            evaluate_scenario(
                scenario=scenario,
                seed=eval_seed + scenario_index * 997,
                salts=selected,
            )
        )

    return CsaHcaRareDirectoryBloomSaltSelectionResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        bits_per_entry=bits_per_entry,
        hash_count=hash_count,
        bank_count=bank_count,
        bank_mode=bank_mode,
        salt_count=salt_count,
        selected_salt_index=selected_salt_index,
        selected_sidecar_salt=selected_sidecar_salt,
        selection_metric="min mean hot_fp, then fp_q, then max HCA",
        route_train_scenarios=route_train_scenarios,
        selection_scenarios=selection_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        selection_points=tuple(selection_points),
        eval_points=tuple(eval_points),
    )


def run_csa_hca_rare_directory_bloom_streaming_update_sweep(
    policies: Tuple[int, ...] = (0, 1, 2, 4, 8, 14),
    bits_per_entry: int = 8,
    hash_count: int = 3,
    bank_count: int = 8,
    bank_mode: str = "by_hash",
    sidecar_salt: int = 30775,
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "split_rare",
        "repeated_name",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 37,
) -> CsaHcaRareDirectoryBloomStreamingResult:
    """Measure streaming update pressure and pollution for the selected sidecar."""

    if len(policies) == 0:
        raise ValueError("policies must not be empty")
    clean_policies = tuple(sorted({int(policy) for policy in policies}))
    if any(policy < 0 for policy in clean_policies):
        raise ValueError("policies must be non-negative")
    if bits_per_entry <= 0:
        raise ValueError("bits_per_entry must be positive")
    if hash_count <= 0:
        raise ValueError("hash_count must be positive")
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if bank_mode not in ("modulo", "by_hash", "hash_slot"):
        raise ValueError("bank_mode must be one of: modulo, by_hash, hash_slot")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)

        for policy in clean_policies:
            points.append(
                _evaluate_rare_directory_bloom_streaming_point(
                    policy=policy,
                    scenario=scenario,
                    config=config,
                    index=index,
                    global_summary=global_summary,
                    exact_counts=exact_counts,
                    directory=directory,
                    directory_blocks_per_token=directory_blocks_per_token,
                    directory_entry_bytes=directory_entry_bytes,
                    hca_threshold=hca_threshold,
                    route_lut=route_lut,
                    fanout_lut=fanout_lut,
                    min_read_blocks_per_token=min_read_blocks_per_token,
                    recent_blocks=recent_blocks,
                    query_tokens=query_tokens,
                    stream=stream,
                    bits_per_entry=bits_per_entry,
                    hash_count=hash_count,
                    bank_count=bank_count,
                    bank_mode=bank_mode,
                    sidecar_salt=sidecar_salt,
                )
            )

    return CsaHcaRareDirectoryBloomStreamingResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        bits_per_entry=bits_per_entry,
        hash_count=hash_count,
        bank_count=bank_count,
        bank_mode=bank_mode,
        sidecar_salt=sidecar_salt,
        policies=clean_policies,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_bloom_retirement_sweep(
    insert_count_thresholds: Tuple[int, ...] = (1, 2, 4, 8, 14),
    retire_count_threshold: int = 15,
    bits_per_entry: int = 8,
    hash_count: int = 3,
    counter_bits: int = 4,
    bank_count: int = 8,
    bank_mode: str = "by_hash",
    sidecar_salt: int = 30775,
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "split_rare",
        "repeated_name",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 37,
) -> CsaHcaRareDirectoryBloomRetirementResult:
    """Measure a counting Bloom sidecar that retires tokens when they become hot."""

    if len(insert_count_thresholds) == 0:
        raise ValueError("insert_count_thresholds must not be empty")
    clean_thresholds = tuple(sorted({int(threshold) for threshold in insert_count_thresholds}))
    if any(threshold <= 0 for threshold in clean_thresholds):
        raise ValueError("insert_count_thresholds must be positive")
    if retire_count_threshold <= 0:
        raise ValueError("retire_count_threshold must be positive")
    if any(threshold >= retire_count_threshold for threshold in clean_thresholds):
        raise ValueError("insert thresholds must be below retire_count_threshold")
    if bits_per_entry <= 0:
        raise ValueError("bits_per_entry must be positive")
    if hash_count <= 0:
        raise ValueError("hash_count must be positive")
    if counter_bits <= 0:
        raise ValueError("counter_bits must be positive")
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if bank_mode not in ("modulo", "by_hash", "hash_slot"):
        raise ValueError("bank_mode must be one of: modulo, by_hash, hash_slot")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)

        for threshold in clean_thresholds:
            points.append(
                _evaluate_rare_directory_bloom_retirement_point(
                    insert_count_threshold=threshold,
                    retire_count_threshold=retire_count_threshold,
                    scenario=scenario,
                    config=config,
                    index=index,
                    global_summary=global_summary,
                    exact_counts=exact_counts,
                    directory=directory,
                    directory_blocks_per_token=directory_blocks_per_token,
                    directory_entry_bytes=directory_entry_bytes,
                    hca_threshold=hca_threshold,
                    route_lut=route_lut,
                    fanout_lut=fanout_lut,
                    min_read_blocks_per_token=min_read_blocks_per_token,
                    recent_blocks=recent_blocks,
                    query_tokens=query_tokens,
                    stream=stream,
                    bits_per_entry=bits_per_entry,
                    hash_count=hash_count,
                    counter_bits=counter_bits,
                    bank_count=bank_count,
                    bank_mode=bank_mode,
                    sidecar_salt=sidecar_salt,
                )
            )

    return CsaHcaRareDirectoryBloomRetirementResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        bits_per_entry=bits_per_entry,
        hash_count=hash_count,
        counter_bits=counter_bits,
        bank_count=bank_count,
        bank_mode=bank_mode,
        sidecar_salt=sidecar_salt,
        insert_count_thresholds=clean_thresholds,
        retire_count_threshold=retire_count_threshold,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_bloom_retirement_compression_sweep(
    bits_per_entries: Tuple[int, ...] = (4, 6, 8),
    counter_bits_values: Tuple[int, ...] = (1, 2, 3, 4),
    insert_count_thresholds: Tuple[int, ...] = (1, 2),
    retire_count_threshold: int = 15,
    hash_count: int = 3,
    bank_count: int = 8,
    bank_mode: str = "by_hash",
    sidecar_salt: int = 30775,
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    eval_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "split_rare",
        "repeated_name",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 37,
) -> CsaHcaRareDirectoryBloomRetirementCompressionResult:
    """Sweep smaller counting-Bloom geometries for the retirement sidecar."""

    if len(bits_per_entries) == 0:
        raise ValueError("bits_per_entries must not be empty")
    clean_bits = tuple(sorted({int(bits) for bits in bits_per_entries}))
    if any(bits <= 0 for bits in clean_bits):
        raise ValueError("bits_per_entries must be positive")
    if len(counter_bits_values) == 0:
        raise ValueError("counter_bits_values must not be empty")
    clean_counter_bits = tuple(sorted({int(bits) for bits in counter_bits_values}))
    if any(bits <= 0 for bits in clean_counter_bits):
        raise ValueError("counter_bits_values must be positive")
    if any(bits > 16 for bits in clean_counter_bits):
        raise ValueError("counter_bits_values must be at most 16")
    if len(insert_count_thresholds) == 0:
        raise ValueError("insert_count_thresholds must not be empty")
    clean_thresholds = tuple(sorted({int(threshold) for threshold in insert_count_thresholds}))
    if any(threshold <= 0 for threshold in clean_thresholds):
        raise ValueError("insert_count_thresholds must be positive")
    if retire_count_threshold <= 0:
        raise ValueError("retire_count_threshold must be positive")
    if any(threshold >= retire_count_threshold for threshold in clean_thresholds):
        raise ValueError("insert thresholds must be below retire_count_threshold")
    if hash_count <= 0:
        raise ValueError("hash_count must be positive")
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if bank_mode not in ("modulo", "by_hash", "hash_slot"):
        raise ValueError("bank_mode must be one of: modulo, by_hash, hash_slot")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for scenario_index, scenario in enumerate(eval_scenarios):
        stream, query_tokens, _ = _make_rare_directory_stress_case(
            config=config,
            scenario=scenario,
            hca_threshold=hca_threshold,
            seed=eval_seed + scenario_index * 997,
        )
        index = LowBitCompressedBlockIndex(config)
        global_summary = LowBitDenseContext(global_config)
        for position, token in enumerate(stream):
            index.update(int(token), position)
            global_summary.update(int(token))

        exact_counts = _build_exact_block_counts(stream, config.block_size)
        directory = _build_rare_block_directory(
            exact_counts=exact_counts,
            hca_threshold=hca_threshold,
            max_blocks_per_token=directory_blocks_per_token,
        )
        directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
        recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)

        for bits_per_entry in clean_bits:
            for counter_bits in clean_counter_bits:
                for threshold in clean_thresholds:
                    points.append(
                        _evaluate_rare_directory_bloom_retirement_point(
                            insert_count_threshold=threshold,
                            retire_count_threshold=retire_count_threshold,
                            scenario=scenario,
                            config=config,
                            index=index,
                            global_summary=global_summary,
                            exact_counts=exact_counts,
                            directory=directory,
                            directory_blocks_per_token=directory_blocks_per_token,
                            directory_entry_bytes=directory_entry_bytes,
                            hca_threshold=hca_threshold,
                            route_lut=route_lut,
                            fanout_lut=fanout_lut,
                            min_read_blocks_per_token=min_read_blocks_per_token,
                            recent_blocks=recent_blocks,
                            query_tokens=query_tokens,
                            stream=stream,
                            bits_per_entry=bits_per_entry,
                            hash_count=hash_count,
                            counter_bits=counter_bits,
                            bank_count=bank_count,
                            bank_mode=bank_mode,
                            sidecar_salt=sidecar_salt,
                        )
                    )

    return CsaHcaRareDirectoryBloomRetirementCompressionResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        bits_per_entries=clean_bits,
        hash_count=hash_count,
        counter_bits_values=clean_counter_bits,
        bank_count=bank_count,
        bank_mode=bank_mode,
        sidecar_salt=sidecar_salt,
        insert_count_thresholds=clean_thresholds,
        retire_count_threshold=retire_count_threshold,
        train_scenarios=train_scenarios,
        eval_scenarios=eval_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_bloom_retirement_collision_sweep(
    bits_per_entries: Tuple[int, ...] = (4, 6, 8),
    counter_bits_values: Tuple[int, ...] = (1, 2, 3, 4),
    rare_occurrences_per_token_values: Tuple[int, ...] = (1, 3),
    colliders_per_rare_values: Tuple[int, ...] = (1, 2, 4, 8),
    rare_token_count: int = 128,
    retire_count_threshold: int = 15,
    hash_count: int = 3,
    bank_count: int = 8,
    bank_mode: str = "by_hash",
    sidecar_salt: int = 30775,
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    min_read_blocks_per_token: int = 2,
    coverage_target: float = 0.95,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 41,
) -> CsaHcaRareDirectoryBloomRetirementCollisionResult:
    """Stress counting-Bloom deletion with hot tokens chosen to collide with rare tokens."""

    if len(bits_per_entries) == 0:
        raise ValueError("bits_per_entries must not be empty")
    clean_bits = tuple(sorted({int(bits) for bits in bits_per_entries}))
    if any(bits <= 0 for bits in clean_bits):
        raise ValueError("bits_per_entries must be positive")
    if len(counter_bits_values) == 0:
        raise ValueError("counter_bits_values must not be empty")
    clean_counter_bits = tuple(sorted({int(bits) for bits in counter_bits_values}))
    if any(bits <= 0 for bits in clean_counter_bits):
        raise ValueError("counter_bits_values must be positive")
    if any(bits > 16 for bits in clean_counter_bits):
        raise ValueError("counter_bits_values must be at most 16")
    if len(rare_occurrences_per_token_values) == 0:
        raise ValueError("rare_occurrences_per_token_values must not be empty")
    clean_rare_occurrences = tuple(
        sorted({int(value) for value in rare_occurrences_per_token_values})
    )
    if any(value <= 0 for value in clean_rare_occurrences):
        raise ValueError("rare_occurrences_per_token_values must be positive")
    if any(value >= retire_count_threshold for value in clean_rare_occurrences):
        raise ValueError("rare occurrences must stay below retire_count_threshold")
    if len(colliders_per_rare_values) == 0:
        raise ValueError("colliders_per_rare_values must not be empty")
    clean_colliders = tuple(sorted({int(value) for value in colliders_per_rare_values}))
    if any(value <= 0 for value in clean_colliders):
        raise ValueError("colliders_per_rare_values must be positive")
    if rare_token_count <= 0:
        raise ValueError("rare_token_count must be positive")
    if retire_count_threshold <= 0:
        raise ValueError("retire_count_threshold must be positive")
    if hash_count <= 0:
        raise ValueError("hash_count must be positive")
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if bank_mode not in ("modulo", "by_hash", "hash_slot"):
        raise ValueError("bank_mode must be one of: modulo, by_hash, hash_slot")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )
    fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
        train_scenarios=tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios,
        hca_threshold=hca_threshold,
        directory_guard=True,
        directory_blocks_per_token=directory_blocks_per_token,
        min_read_blocks_per_token=min_read_blocks_per_token,
        coverage_target=coverage_target,
        span_thresholds=span_thresholds,
        max_overlap_bucket=max_overlap_bucket,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed + 4096,
    )

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

    points = []
    for bits_per_entry in clean_bits:
        for rare_occurrences in clean_rare_occurrences:
            for colliders_per_rare in clean_colliders:
                stream, query_tokens, found_colliders, missing_colliders, mean_overlap = (
                    _make_bloom_collision_retirement_stress_case(
                        config=config,
                        hca_threshold=hca_threshold,
                        rare_token_count=rare_token_count,
                        rare_occurrences_per_token=rare_occurrences,
                        colliders_per_rare=colliders_per_rare,
                        bits_per_entry=bits_per_entry,
                        hash_count=hash_count,
                        bank_count=bank_count,
                        bank_mode=bank_mode,
                        sidecar_salt=sidecar_salt,
                        seed=(
                            eval_seed
                            + bits_per_entry * 997
                            + rare_occurrences * 313
                            + colliders_per_rare * 131
                        ),
                    )
                )
                index = LowBitCompressedBlockIndex(config)
                global_summary = LowBitDenseContext(global_config)
                for position, token in enumerate(stream):
                    index.update(int(token), position)
                    global_summary.update(int(token))

                exact_counts = _build_exact_block_counts(stream, config.block_size)
                directory = _build_rare_block_directory(
                    exact_counts=exact_counts,
                    hca_threshold=hca_threshold,
                    max_blocks_per_token=directory_blocks_per_token,
                )
                directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
                recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)

                for counter_bits in clean_counter_bits:
                    point = _evaluate_rare_directory_bloom_retirement_point(
                        insert_count_threshold=1,
                        retire_count_threshold=retire_count_threshold,
                        scenario="adversarial_collision",
                        config=config,
                        index=index,
                        global_summary=global_summary,
                        exact_counts=exact_counts,
                        directory=directory,
                        directory_blocks_per_token=directory_blocks_per_token,
                        directory_entry_bytes=directory_entry_bytes,
                        hca_threshold=hca_threshold,
                        route_lut=route_lut,
                        fanout_lut=fanout_lut,
                        min_read_blocks_per_token=min_read_blocks_per_token,
                        recent_blocks=recent_blocks,
                        query_tokens=query_tokens,
                        stream=stream,
                        bits_per_entry=bits_per_entry,
                        hash_count=hash_count,
                        counter_bits=counter_bits,
                        bank_count=bank_count,
                        bank_mode=bank_mode,
                        sidecar_salt=sidecar_salt,
                    )
                    points.append(
                        CsaHcaRareDirectoryBloomRetirementCollisionPoint(
                            scenario=point.scenario,
                            bits_per_entry=bits_per_entry,
                            hash_count=hash_count,
                            counter_bits=counter_bits,
                            bank_count=bank_count,
                            bank_mode=bank_mode,
                            sidecar_salt=sidecar_salt,
                            rare_occurrences_per_token=rare_occurrences,
                            colliders_per_rare=colliders_per_rare,
                            rare_tokens=point.final_rare_tokens,
                            collider_tokens=found_colliders,
                            missing_colliders=missing_colliders,
                            mean_slot_overlap=mean_overlap,
                            sidecar_state_bytes=point.sidecar_state_bytes,
                            visible_active_rare_rate=point.visible_active_rare_rate,
                            hot_polluted_token_rate=point.hot_polluted_token_rate,
                            update_bytes_per_context_token=point.update_bytes_per_context_token,
                            hca_query_rate=point.hca_query_rate,
                            rare_false_hca_rate=point.rare_false_hca_rate,
                            repaired_relevant_coverage=point.repaired_relevant_coverage,
                            token_read_reduction=point.token_read_reduction,
                        )
                    )

    return CsaHcaRareDirectoryBloomRetirementCollisionResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        rare_token_count=rare_token_count,
        rare_occurrences_per_token_values=clean_rare_occurrences,
        colliders_per_rare_values=clean_colliders,
        bits_per_entries=clean_bits,
        hash_count=hash_count,
        counter_bits_values=clean_counter_bits,
        bank_count=bank_count,
        bank_mode=bank_mode,
        sidecar_salt=sidecar_salt,
        train_scenarios=train_scenarios,
        training_samples=route_training_samples + fanout_training_samples,
        route_lut=route_lut,
        fanout_lut=fanout_lut,
        points=tuple(points),
    )


def run_csa_hca_rare_directory_bloom_retirement_collision_fanout_sweep(
    min_read_blocks_per_token_values: Tuple[int, ...] = (2, 3),
    coverage_targets: Tuple[float, ...] = (0.95, 1.0),
    bits_per_entry: int = 8,
    counter_bits: int = 3,
    rare_occurrences_per_token: int = 3,
    colliders_per_rare: int = 8,
    rare_token_count: int = 128,
    retire_count_threshold: int = 15,
    hash_count: int = 3,
    bank_count: int = 8,
    bank_mode: str = "by_hash",
    sidecar_salt: int = 30775,
    train_scenarios: Tuple[str, ...] = (
        "zipf_reference",
        "rare_burst",
        "split_rare",
        "repeated_name",
        "collision_noise",
    ),
    hca_threshold: int = 15,
    directory_blocks_per_token: int = 6,
    route_positive_rate_threshold: float = 0.50,
    span_thresholds: Tuple[int, ...] = (64, 128, 256),
    max_overlap_bucket: int = 3,
    block_size: int = 128,
    summary_width: int = 128,
    csa_blocks: int = 4,
    global_width: int = 2048,
    tail_blocks: int = 2,
    context_length: int = 65536,
    queries: int = 2048,
    train_seed: int = 31,
    eval_seed: int = 41,
) -> CsaHcaRareDirectoryBloomRetirementCollisionFanoutResult:
    """Sweep low-bit fanout budgets after sidecar visibility is robust."""

    if len(min_read_blocks_per_token_values) == 0:
        raise ValueError("min_read_blocks_per_token_values must not be empty")
    clean_min_reads = tuple(sorted({int(value) for value in min_read_blocks_per_token_values}))
    if any(value < 0 for value in clean_min_reads):
        raise ValueError("min_read_blocks_per_token_values must be non-negative")
    if any(value > directory_blocks_per_token for value in clean_min_reads):
        raise ValueError("min_read_blocks_per_token_values cannot exceed directory_blocks_per_token")
    if len(coverage_targets) == 0:
        raise ValueError("coverage_targets must not be empty")
    clean_targets = tuple(sorted({float(value) for value in coverage_targets}))
    if any(value < 0.0 or value > 1.0 for value in clean_targets):
        raise ValueError("coverage_targets must be in [0, 1]")
    if bits_per_entry <= 0:
        raise ValueError("bits_per_entry must be positive")
    if counter_bits <= 0 or counter_bits > 16:
        raise ValueError("counter_bits must be in [1, 16]")
    if rare_token_count <= 0:
        raise ValueError("rare_token_count must be positive")
    if rare_occurrences_per_token <= 0:
        raise ValueError("rare_occurrences_per_token must be positive")
    if rare_occurrences_per_token >= retire_count_threshold:
        raise ValueError("rare_occurrences_per_token must stay below retire_count_threshold")
    if colliders_per_rare <= 0:
        raise ValueError("colliders_per_rare must be positive")
    if retire_count_threshold <= 0:
        raise ValueError("retire_count_threshold must be positive")
    if hash_count <= 0:
        raise ValueError("hash_count must be positive")
    if bank_count <= 0:
        raise ValueError("bank_count must be positive")
    if bank_mode not in ("modulo", "by_hash", "hash_slot"):
        raise ValueError("bank_mode must be one of: modulo, by_hash, hash_slot")

    route_lut, route_training_samples = train_directory_aware_hca_route_lut(
        train_scenarios=train_scenarios,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        route_positive_rate_threshold=route_positive_rate_threshold,
        block_size=block_size,
        summary_width=summary_width,
        csa_blocks=csa_blocks,
        global_width=global_width,
        tail_blocks=tail_blocks,
        context_length=context_length,
        queries=queries,
        seed=train_seed,
    )

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
    stream, query_tokens, found_colliders, missing_colliders, mean_overlap = (
        _make_bloom_collision_retirement_stress_case(
            config=config,
            hca_threshold=hca_threshold,
            rare_token_count=rare_token_count,
            rare_occurrences_per_token=rare_occurrences_per_token,
            colliders_per_rare=colliders_per_rare,
            bits_per_entry=bits_per_entry,
            hash_count=hash_count,
            bank_count=bank_count,
            bank_mode=bank_mode,
            sidecar_salt=sidecar_salt,
            seed=(
                eval_seed
                + bits_per_entry * 997
                + rare_occurrences_per_token * 313
                + colliders_per_rare * 131
            ),
        )
    )

    index = LowBitCompressedBlockIndex(config)
    global_summary = LowBitDenseContext(global_config)
    for position, token in enumerate(stream):
        index.update(int(token), position)
        global_summary.update(int(token))

    exact_counts = _build_exact_block_counts(stream, config.block_size)
    directory = _build_rare_block_directory(
        exact_counts=exact_counts,
        hca_threshold=hca_threshold,
        max_blocks_per_token=directory_blocks_per_token,
    )
    directory_entry_bytes = _rare_directory_entry_bytes(config.vocab_size, config.blocks)
    recent_blocks = _recent_blocks(config.blocks, config.tail_blocks)
    sidecar_read_bytes_per_query = hash_count / 8

    points = []
    fanout_train_scenarios = tuple(s for s in train_scenarios if s != "zipf_reference") or train_scenarios
    for min_read_blocks_per_token in clean_min_reads:
        for coverage_target in clean_targets:
            fanout_lut, fanout_training_samples = train_rare_directory_fanout_lut(
                train_scenarios=fanout_train_scenarios,
                hca_threshold=hca_threshold,
                directory_guard=True,
                directory_blocks_per_token=directory_blocks_per_token,
                min_read_blocks_per_token=min_read_blocks_per_token,
                coverage_target=coverage_target,
                span_thresholds=span_thresholds,
                max_overlap_bucket=max_overlap_bucket,
                block_size=block_size,
                summary_width=summary_width,
                csa_blocks=csa_blocks,
                global_width=global_width,
                tail_blocks=tail_blocks,
                context_length=context_length,
                queries=queries,
                seed=train_seed + 4096,
            )
            point = _evaluate_rare_directory_bloom_retirement_point(
                insert_count_threshold=1,
                retire_count_threshold=retire_count_threshold,
                scenario="adversarial_collision_fanout",
                config=config,
                index=index,
                global_summary=global_summary,
                exact_counts=exact_counts,
                directory=directory,
                directory_blocks_per_token=directory_blocks_per_token,
                directory_entry_bytes=directory_entry_bytes,
                hca_threshold=hca_threshold,
                route_lut=route_lut,
                fanout_lut=fanout_lut,
                min_read_blocks_per_token=min_read_blocks_per_token,
                recent_blocks=recent_blocks,
                query_tokens=query_tokens,
                stream=stream,
                bits_per_entry=bits_per_entry,
                hash_count=hash_count,
                counter_bits=counter_bits,
                bank_count=bank_count,
                bank_mode=bank_mode,
                sidecar_salt=sidecar_salt,
            )
            directory_entries_read = _safe_divide(
                max(0.0, point.directory_read_bytes_per_query - sidecar_read_bytes_per_query),
                directory_entry_bytes,
            )
            points.append(
                CsaHcaRareDirectoryBloomRetirementCollisionFanoutPoint(
                    scenario=point.scenario,
                    min_read_blocks_per_token=min_read_blocks_per_token,
                    coverage_target=coverage_target,
                    directory_blocks_per_token=directory_blocks_per_token,
                    bits_per_entry=bits_per_entry,
                    hash_count=hash_count,
                    counter_bits=counter_bits,
                    bank_count=bank_count,
                    bank_mode=bank_mode,
                    sidecar_salt=sidecar_salt,
                    rare_occurrences_per_token=rare_occurrences_per_token,
                    colliders_per_rare=colliders_per_rare,
                    rare_tokens=point.final_rare_tokens,
                    collider_tokens=found_colliders,
                    missing_colliders=missing_colliders,
                    mean_slot_overlap=mean_overlap,
                    sidecar_state_bytes=point.sidecar_state_bytes,
                    fanout_lut_state_bytes=fanout_lut.state_bytes,
                    fanout_training_samples=fanout_training_samples,
                    visible_active_rare_rate=point.visible_active_rare_rate,
                    rare_false_hca_rate=point.rare_false_hca_rate,
                    repaired_relevant_coverage=point.repaired_relevant_coverage,
                    directory_read_bytes_per_query=point.directory_read_bytes_per_query,
                    directory_entries_read_per_query=directory_entries_read,
                    token_read_reduction=point.token_read_reduction,
                )
            )

    return CsaHcaRareDirectoryBloomRetirementCollisionFanoutResult(
        context_length=context_length,
        block_size=block_size,
        summary_width=summary_width,
        global_width=global_width,
        csa_blocks=config.selected_blocks,
        tail_blocks=config.tail_blocks,
        hca_threshold=hca_threshold,
        directory_blocks_per_token=directory_blocks_per_token,
        rare_token_count=rare_token_count,
        rare_occurrences_per_token=rare_occurrences_per_token,
        colliders_per_rare=colliders_per_rare,
        bits_per_entry=bits_per_entry,
        hash_count=hash_count,
        counter_bits=counter_bits,
        bank_count=bank_count,
        bank_mode=bank_mode,
        sidecar_salt=sidecar_salt,
        min_read_blocks_per_token_values=clean_min_reads,
        coverage_targets=clean_targets,
        train_scenarios=train_scenarios,
        route_training_samples=route_training_samples,
        route_lut=route_lut,
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


def run_hca_lazy_decay_trial(
    global_width: int = 2048,
    decay_interval: int = 256,
    threshold: int = 2,
    epoch_bits: int = 16,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> HcaLazyDecayResult:
    """Evaluate per-counter epoch metadata as low-maintenance HCA decay."""

    if global_width <= 0:
        raise ValueError("global_width must be positive")
    if decay_interval <= 0:
        raise ValueError("decay_interval must be positive")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    _validate_epoch_capacity(context_length, decay_interval, epoch_bits)
    config = CompressedBlockIndexConfig(context_length=context_length, queries=queries)
    stream = _make_zipf_topic_stream(config, seed=seed)
    query_tokens = _make_zipf_topic_stream(config, seed=seed + 1, length=config.queries)
    summary = LazyDecayedDenseSummary(
        vocab_size=config.vocab_size,
        banks=config.banks,
        width=global_width,
        bits=config.bits,
        decay_interval=decay_interval,
        epoch_bits=epoch_bits,
    )
    touched = 0
    for token in stream:
        touched += summary.update(int(token))

    explicit_config = DenseContextConfig(
        vocab_size=config.vocab_size,
        banks=config.banks,
        width=global_width,
        bits=config.bits,
        decay_interval=decay_interval,
    )
    exact = exact_decayed_counts(stream, explicit_config).astype(np.int32)
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
    explicit_decay_cells = decay_events * config.banks * global_width
    effective = summary.effective_counters()

    return HcaLazyDecayResult(
        context_length=config.context_length,
        vocab_size=config.vocab_size,
        hot_tokens=config.hot_tokens,
        global_width=global_width,
        bits=config.bits,
        epoch_bits=epoch_bits,
        decay_interval=decay_interval,
        threshold=threshold,
        queries=config.queries,
        state_bytes=summary.state_bytes,
        read_bytes_per_query=summary.read_bytes_per_query,
        avg_update_cells_per_token=touched / config.context_length,
        avg_decay_cells_per_token=0.0,
        saturation_rate=float(np.count_nonzero(effective == summary.max_value) / effective.size),
        clipped_mean_abs_error=float(np.mean(np.abs(estimates - exact))),
        top64_recall=_topk_recall(estimates, exact, 64),
        top256_recall=_topk_recall(estimates, exact, 256),
        threshold_precision=_safe_divide(true_positive, predicted_positive),
        threshold_recall=_safe_divide(true_positive, actual_positive),
        query_route_accuracy=query_correct / config.queries,
        query_false_hca_rate=_safe_divide(query_false_hca, query_actual_cold),
        query_missed_hca_rate=_safe_divide(query_missed_hca, query_actual_hot),
        explicit_decay_cells_per_token=explicit_decay_cells / config.context_length,
    )


def run_hca_lazy_metadata_sweep(
    global_width: int = 2048,
    candidates: Tuple[Tuple[int, int], ...] = (
        (8, 256),
        (8, 512),
        (8, 1024),
        (4, 4096),
        (4, 8192),
        (16, 256),
    ),
    threshold: int = 2,
    context_length: int = 65536,
    queries: int = 4096,
    seed: int = 37,
) -> HcaLazyMetadataSweepResult:
    """Compare lazy epoch metadata sizes that avoid per-counter epoch wrap."""

    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")
    points = []
    for epoch_bits, decay_interval in candidates:
        points.append(
            run_hca_lazy_decay_trial(
                global_width=global_width,
                decay_interval=int(decay_interval),
                threshold=threshold,
                epoch_bits=int(epoch_bits),
                context_length=context_length,
                queries=queries,
                seed=seed,
            )
        )

    config = CompressedBlockIndexConfig(context_length=context_length, queries=queries)
    return HcaLazyMetadataSweepResult(
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


def _validate_epoch_capacity(
    context_length: int,
    decay_interval: int,
    epoch_bits: int,
) -> None:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if decay_interval <= 0:
        raise ValueError("decay_interval must be positive")
    if epoch_bits not in (4, 8, 16):
        raise ValueError("epoch_bits must be one of 4, 8, 16")
    max_stored_epoch = (context_length - 1) // decay_interval
    if max_stored_epoch >= (1 << epoch_bits):
        raise ValueError("epoch_bits cannot represent stored per-counter epochs")


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


def _make_rare_directory_stress_case(
    config: CompressedBlockIndexConfig,
    scenario: str,
    hca_threshold: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    scenario = scenario.lower()
    if scenario == "zipf_reference":
        return (
            _make_zipf_topic_stream(config, seed=seed),
            _make_zipf_topic_stream(config, seed=seed + 1, length=config.queries),
            0,
        )

    valid = {"rare_burst", "split_rare", "repeated_name", "collision_noise"}
    if scenario not in valid:
        raise ValueError(f"unknown rare-directory stress scenario: {scenario}")

    rng = np.random.default_rng(seed)
    stream = _make_zipf_topic_stream(config, seed=seed)
    non_tail_blocks = max(1, config.blocks - config.tail_blocks)
    target_count = min(128, non_tail_blocks, config.vocab_size - config.hot_tokens)
    if target_count <= 0:
        raise ValueError("vocab does not have room for stress tokens")
    target_start = config.hot_tokens
    target_tokens = np.arange(target_start, target_start + target_count, dtype=np.int32)

    accidental = (stream >= target_start) & (stream < target_start + target_count)
    if np.any(accidental):
        replacement_low = target_start + target_count
        if replacement_low < config.vocab_size:
            stream[accidental] = rng.integers(
                replacement_low,
                config.vocab_size,
                size=int(np.count_nonzero(accidental)),
            )
        else:
            stream[accidental] = rng.integers(0, config.hot_tokens, size=int(np.count_nonzero(accidental)))

    def write_token(token: int, block: int, ordinal: int) -> None:
        offset = (int(token) * 17 + ordinal * 23) % config.block_size
        stream[int(block) * config.block_size + offset] = int(token)

    for index, token in enumerate(target_tokens):
        base_block = (index * 37 + 11) % non_tail_blocks
        if scenario == "rare_burst":
            for occurrence in range(min(6, hca_threshold - 1)):
                write_token(int(token), base_block, occurrence)
        elif scenario == "split_rare":
            stride = max(1, non_tail_blocks // 3)
            for split in range(3):
                block = (base_block + split * stride) % non_tail_blocks
                for occurrence in range(2):
                    write_token(int(token), block, split * 2 + occurrence)
        elif scenario == "repeated_name":
            stride = max(1, non_tail_blocks // 6)
            for occurrence in range(min(6, hca_threshold - 1)):
                block = (base_block + occurrence * stride) % non_tail_blocks
                write_token(int(token), block, occurrence)
        else:
            for occurrence in range(min(4, hca_threshold - 1)):
                write_token(int(token), base_block, occurrence)
            distractor_start = target_start + target_count + index * 16
            for distractor in range(16):
                token_id = distractor_start + distractor
                if token_id >= config.vocab_size:
                    token_id = target_start + target_count + (
                        (index * 16 + distractor) % max(1, config.vocab_size - target_start - target_count)
                    )
                write_token(int(token_id), base_block, 10 + distractor)

    query_tokens = np.resize(target_tokens, config.queries).astype(np.int32)
    return stream, query_tokens, target_count


def _make_bloom_collision_retirement_stress_case(
    config: CompressedBlockIndexConfig,
    hca_threshold: int,
    rare_token_count: int,
    rare_occurrences_per_token: int,
    colliders_per_rare: int,
    bits_per_entry: int,
    hash_count: int,
    bank_count: int,
    bank_mode: str,
    sidecar_salt: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    """Build a stream where retiring hot tokens share Bloom slots with rare tokens."""

    if rare_token_count <= 0:
        raise ValueError("rare_token_count must be positive")
    if rare_occurrences_per_token <= 0:
        raise ValueError("rare_occurrences_per_token must be positive")
    if rare_occurrences_per_token >= hca_threshold:
        raise ValueError("rare_occurrences_per_token must stay below hca_threshold")
    if hca_threshold <= 1:
        raise ValueError("hca_threshold must be greater than 1")
    if colliders_per_rare <= 0:
        raise ValueError("colliders_per_rare must be positive")
    if 1 + colliders_per_rare * hca_threshold > config.block_size:
        raise ValueError("colliders_per_rare does not fit in one block")
    if config.hot_tokens + rare_token_count >= config.vocab_size:
        raise ValueError("vocab does not have room for rare collision tokens")

    rng = np.random.default_rng(seed)
    stream = np.arange(config.context_length, dtype=np.int32) % max(1, config.hot_tokens)
    rare_tokens = np.arange(
        config.hot_tokens,
        config.hot_tokens + rare_token_count,
        dtype=np.int32,
    )
    bit_count = max(1, int(max(1, rare_token_count) * bits_per_entry))
    sidecar = LowBitPresenceBloomSidecar(
        bit_count=bit_count,
        hash_count=hash_count,
        bank_count=bank_count,
        bank_mode=bank_mode,
        salt=sidecar_salt,
    )
    candidate_start = config.hot_tokens + rare_token_count
    candidate_span = max(1, config.vocab_size - candidate_start)
    slot_to_candidates = {slot: [] for slot in range(bit_count)}
    candidate_slots: Dict[int, set[int]] = {}
    for candidate in range(candidate_start, config.vocab_size):
        slots = set(sidecar.slots(int(candidate)))
        candidate_slots[int(candidate)] = slots
        for slot in slots:
            slot_to_candidates[int(slot)].append(int(candidate))
    used_colliders = set()
    colliders = []
    overlaps = []
    missing = 0

    for rare_index, rare_token in enumerate(rare_tokens):
        rare_slots = set(sidecar.slots(int(rare_token)))
        rare_colliders = []
        total_overlap = 0
        for collider_index in range(colliders_per_rare):
            best_token = None
            best_overlap = 0
            pool = []
            for slot in sorted(rare_slots):
                pool.extend(slot_to_candidates[int(slot)])
            if pool:
                start = int(rng.integers(0, len(pool)))
            else:
                start = 0
            seen_candidates = set()
            for attempt in range(len(pool)):
                candidate = int(pool[(start + attempt * 37) % len(pool)])
                if candidate in seen_candidates:
                    continue
                seen_candidates.add(candidate)
                if candidate in used_colliders:
                    continue
                overlap = len(rare_slots.intersection(candidate_slots[int(candidate)]))
                if overlap > best_overlap:
                    best_token = int(candidate)
                    best_overlap = int(overlap)
                    if best_overlap >= min(hash_count, 2):
                        break
            if best_token is None:
                missing += 1
                best_token = candidate_start + (
                    (rare_index * 193 + collider_index * 997) % candidate_span
                )
                best_overlap = len(rare_slots.intersection(candidate_slots[int(best_token)]))
            used_colliders.add(int(best_token))
            rare_colliders.append(int(best_token))
            total_overlap += int(best_overlap)
        colliders.append(tuple(rare_colliders))
        overlaps.append(total_overlap)

    non_tail_blocks = max(1, config.blocks - config.tail_blocks)
    if rare_token_count * rare_occurrences_per_token > non_tail_blocks:
        raise ValueError("rare collision stress tokens must fit in non-tail blocks")

    for index, rare_token in enumerate(rare_tokens):
        for occurrence_index in range(rare_occurrences_per_token):
            block = (index * rare_occurrences_per_token + occurrence_index + 7) % non_tail_blocks
            base_position = block * config.block_size
            stream[base_position] = int(rare_token)
            offset = 1
            for collider in colliders[index]:
                for _ in range(hca_threshold):
                    stream[base_position + offset] = int(collider)
                    offset += 1

    query_tokens = np.resize(rare_tokens, config.queries).astype(np.int32)
    found_colliders = rare_token_count * colliders_per_rare - missing
    mean_overlap = float(sum(overlaps) / len(overlaps)) if overlaps else 0.0
    return stream, query_tokens, found_colliders, missing, mean_overlap


def _evaluate_rare_directory_stress_point(
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    directory_state_bytes: float,
    hca_threshold: int,
    directory_guard: bool,
    directory_read_blocks_per_token: int | None,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
    stress_token_count: int,
) -> CsaHcaRareDirectoryStressPoint:
    hca_queries = 0
    csa_queries = 0
    directory_queries = 0
    directory_guard_hits = 0
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    csa_relevant = 0
    base_hits = 0
    repaired_hits = 0
    base_coverage = 0.0
    repaired_coverage = 0.0
    base_csa_hits = 0
    repaired_csa_hits = 0
    base_csa_coverage = 0.0
    repaired_csa_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    read_limit = (
        directory_blocks_per_token
        if directory_read_blocks_per_token is None
        else max(0, int(directory_read_blocks_per_token))
    )
    read_limit = min(directory_blocks_per_token, read_limit)

    for token in query_tokens:
        token = int(token)
        global_estimate = global_summary.estimate(token)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        route_hca = global_estimate >= hca_threshold
        if directory_guard and directory_blocks_per_token > 0:
            directory_read_bytes += directory_entry_bytes
            if directory_hit:
                route_hca = False
                directory_guard_hits += 1
        block_counts = exact_counts.get(token)
        is_relevant = block_counts is not None

        if route_hca:
            base_selected = recent_blocks
            selected = recent_blocks
            hca_queries += 1
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            csa_queries += 1
            if directory_blocks_per_token > 0 and (directory_guard or read_limit > 0):
                if directory_guard:
                    directory_read_bytes += directory_entry_bytes * max(
                        0,
                        len(readable_directory_blocks) - 1,
                    )
                else:
                    directory_read_bytes += directory_entry_bytes * max(
                        1,
                        len(readable_directory_blocks),
                    )
            if len(directory_blocks) > 0:
                directory_queries += 1

        token_reads += len(selected) * config.block_size
        if not is_relevant:
            continue

        relevant_queries += 1
        exact_total = sum(int(value) for value in block_counts.values())
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)

        base_hit = _block_hit(base_selected, block_counts)
        repaired_hit = _block_hit(selected, block_counts)
        base_hits += base_hit
        repaired_hits += repaired_hit
        base_coverage += _occurrence_coverage(base_selected, block_counts)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

        if not route_hca:
            csa_relevant += 1
            base_csa_hits += base_hit
            repaired_csa_hits += repaired_hit
            base_csa_coverage += _occurrence_coverage(base_selected, block_counts)
            repaired_csa_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    csa_denominator = csa_relevant if csa_relevant else 1
    rare_denominator = rare_relevant_queries if rare_relevant_queries else 1
    token_reads_per_query = token_reads / query_denominator
    return CsaHcaRareDirectoryStressPoint(
        scenario=scenario,
        directory_guard=directory_guard,
        directory_blocks_per_token=directory_blocks_per_token,
        directory_read_blocks_per_token=read_limit,
        stress_token_count=stress_token_count,
        directory_entries=sum(len(blocks) for blocks in directory.values()),
        directory_state_bytes=directory_state_bytes,
        block_plus_directory_state_bytes=index.state_bytes + directory_state_bytes,
        hca_query_rate=hca_queries / query_denominator,
        csa_query_rate=csa_queries / query_denominator,
        rare_false_hca_rate=rare_false_hca / rare_denominator,
        directory_guard_hit_rate=directory_guard_hits / query_denominator,
        directory_query_rate=directory_queries / query_denominator,
        base_relevant_hit_rate=base_hits / relevant_denominator,
        repaired_relevant_hit_rate=repaired_hits / relevant_denominator,
        base_relevant_coverage=base_coverage / relevant_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        base_csa_relevant_hit_rate=base_csa_hits / csa_denominator,
        repaired_csa_relevant_hit_rate=repaired_csa_hits / csa_denominator,
        base_csa_relevant_coverage=base_csa_coverage / csa_denominator,
        repaired_csa_relevant_coverage=repaired_csa_coverage / csa_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_reads_per_query=token_reads_per_query,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_adaptive_policy_point(
    policy: str,
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    directory_state_bytes: float,
    hca_threshold: int,
    directory_guard: bool,
    base_read_blocks_per_token: int,
    expanded_read_blocks_per_token: int,
    spread_threshold_blocks: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
) -> CsaHcaRareDirectoryAdaptivePolicyPoint:
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    repaired_hits = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    directory_hit_queries = 0
    directory_entries_seen = 0
    directory_blocks_read = 0
    expanded_read_queries = 0

    max_read_blocks = min(
        max(base_read_blocks_per_token, expanded_read_blocks_per_token),
        directory_blocks_per_token,
    )
    fanout_lut_state_bytes = 4 * 2 / 8
    spread_metadata_state_bytes = len(directory) * 2 / 8

    for token in query_tokens:
        token = int(token)
        global_estimate = global_summary.estimate(token)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        route_hca = global_estimate >= hca_threshold
        if directory_guard and directory_blocks_per_token > 0:
            directory_read_bytes += directory_entry_bytes
            if directory_hit:
                route_hca = False
        block_counts = exact_counts.get(token)
        is_relevant = block_counts is not None

        if route_hca:
            selected = recent_blocks
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = _adaptive_directory_read_limit(
                directory_blocks=directory_blocks,
                directory_blocks_per_token=directory_blocks_per_token,
                base_read_blocks_per_token=base_read_blocks_per_token,
                expanded_read_blocks_per_token=expanded_read_blocks_per_token,
                spread_threshold_blocks=spread_threshold_blocks,
            )
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            if directory_blocks_per_token > 0 and (directory_guard or max_read_blocks > 0):
                if directory_guard:
                    directory_read_bytes += directory_entry_bytes * max(
                        0,
                        len(readable_directory_blocks) - 1,
                    )
                else:
                    directory_read_bytes += directory_entry_bytes * max(
                        1,
                        len(readable_directory_blocks),
                    )
            if directory_hit:
                directory_hit_queries += 1
                directory_entries_seen += len(directory_blocks)
                directory_blocks_read += len(readable_directory_blocks)
                expanded_read_queries += int(
                    len(readable_directory_blocks)
                    > min(base_read_blocks_per_token, directory_blocks_per_token)
                )

        token_reads += len(selected) * config.block_size
        if not is_relevant:
            continue

        relevant_queries += 1
        exact_total = sum(int(value) for value in block_counts.values())
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)
        repaired_hits += _block_hit(selected, block_counts)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    rare_denominator = rare_relevant_queries if rare_relevant_queries else 1
    directory_hit_denominator = directory_hit_queries if directory_hit_queries else 1
    token_reads_per_query = token_reads / query_denominator

    return CsaHcaRareDirectoryAdaptivePolicyPoint(
        policy=policy,
        scenario=scenario,
        hca_threshold=hca_threshold,
        directory_guard=directory_guard,
        directory_blocks_per_token=directory_blocks_per_token,
        base_read_blocks_per_token=min(base_read_blocks_per_token, directory_blocks_per_token),
        expanded_read_blocks_per_token=max_read_blocks,
        spread_threshold_blocks=spread_threshold_blocks,
        fanout_metadata_state_bytes=fanout_lut_state_bytes + spread_metadata_state_bytes,
        directory_state_bytes=directory_state_bytes,
        avg_directory_entries_per_hit=directory_entries_seen / directory_hit_denominator,
        avg_directory_read_blocks_per_hit=directory_blocks_read / directory_hit_denominator,
        expanded_read_rate=expanded_read_queries / directory_hit_denominator,
        rare_false_hca_rate=rare_false_hca / rare_denominator,
        repaired_relevant_hit_rate=repaired_hits / relevant_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_reads_per_query=token_reads_per_query,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_lut_fanout_point(
    policy: str,
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    directory_state_bytes: float,
    hca_threshold: int,
    directory_guard: bool,
    lut: LowBitRareDirectoryFanoutLUT,
    min_read_blocks_per_token: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
) -> CsaHcaRareDirectoryAdaptivePolicyPoint:
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    repaired_hits = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    directory_hit_queries = 0
    directory_entries_seen = 0
    directory_blocks_read = 0
    expanded_read_queries = 0
    min_read_blocks_per_token = min(max(0, int(min_read_blocks_per_token)), directory_blocks_per_token)

    for token in query_tokens:
        token = int(token)
        global_estimate = global_summary.estimate(token)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        route_hca = global_estimate >= hca_threshold
        if directory_guard and directory_blocks_per_token > 0:
            directory_read_bytes += directory_entry_bytes
            if directory_hit:
                route_hca = False
        block_counts = exact_counts.get(token)
        is_relevant = block_counts is not None

        if route_hca:
            selected = recent_blocks
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = lut.predict(directory_blocks, base_selected)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            if directory_blocks_per_token > 0 and (directory_guard or lut.max_entries > 0):
                if directory_guard:
                    directory_read_bytes += directory_entry_bytes * max(
                        0,
                        len(readable_directory_blocks) - 1,
                    )
                else:
                    directory_read_bytes += directory_entry_bytes * max(
                        1,
                        len(readable_directory_blocks),
                    )
            if directory_hit:
                directory_hit_queries += 1
                directory_entries_seen += len(directory_blocks)
                directory_blocks_read += len(readable_directory_blocks)
                expanded_read_queries += int(len(readable_directory_blocks) > min_read_blocks_per_token)

        token_reads += len(selected) * config.block_size
        if not is_relevant:
            continue

        relevant_queries += 1
        exact_total = sum(int(value) for value in block_counts.values())
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)
        repaired_hits += _block_hit(selected, block_counts)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    rare_denominator = rare_relevant_queries if rare_relevant_queries else 1
    directory_hit_denominator = directory_hit_queries if directory_hit_queries else 1
    token_reads_per_query = token_reads / query_denominator
    spread_metadata_state_bytes = len(directory) * 2 / 8

    return CsaHcaRareDirectoryAdaptivePolicyPoint(
        policy=policy,
        scenario=scenario,
        hca_threshold=hca_threshold,
        directory_guard=directory_guard,
        directory_blocks_per_token=directory_blocks_per_token,
        base_read_blocks_per_token=min_read_blocks_per_token,
        expanded_read_blocks_per_token=lut.max_entries,
        spread_threshold_blocks=0,
        fanout_metadata_state_bytes=lut.state_bytes + spread_metadata_state_bytes,
        directory_state_bytes=directory_state_bytes,
        avg_directory_entries_per_hit=directory_entries_seen / directory_hit_denominator,
        avg_directory_read_blocks_per_hit=directory_blocks_read / directory_hit_denominator,
        expanded_read_rate=expanded_read_queries / directory_hit_denominator,
        rare_false_hca_rate=rare_false_hca / rare_denominator,
        repaired_relevant_hit_rate=repaired_hits / relevant_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_reads_per_query=token_reads_per_query,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_joint_policy_point(
    policy: str,
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    directory_state_bytes: float,
    hca_threshold: int,
    probe_mode: str,
    lut: LowBitRareDirectoryFanoutLUT,
    probe_lut: LowBitRareDirectoryProbeLUT,
    min_read_blocks_per_token: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
) -> CsaHcaRareDirectoryJointPolicyPoint:
    probe_mode = probe_mode.lower()
    if probe_mode not in {"never", "hca_only", "confidence", "always"}:
        raise ValueError("probe_mode must be one of never, hca_only, confidence, always")

    hca_queries = 0
    csa_queries = 0
    directory_probes = 0
    directory_hits = 0
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    repaired_hits = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    directory_entries_seen = 0
    directory_blocks_read = 0
    expanded_read_queries = 0
    min_read_blocks_per_token = min(max(0, int(min_read_blocks_per_token)), directory_blocks_per_token)

    for token in query_tokens:
        token = int(token)
        counter_values = _dense_counter_values(global_summary, token)
        global_estimate = min(counter_values)
        route_hca = global_estimate >= hca_threshold
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        should_probe = (
            directory_blocks_per_token > 0
            and (
                probe_mode == "always"
                or (probe_mode == "hca_only" and route_hca)
                or (probe_mode == "confidence" and route_hca and probe_lut.probe(counter_values))
            )
        )
        if should_probe:
            directory_probes += 1
            directory_read_bytes += directory_entry_bytes
            if directory_hit:
                route_hca = False

        block_counts = exact_counts.get(token)
        is_relevant = block_counts is not None

        if route_hca:
            selected = recent_blocks
            hca_queries += 1
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = lut.predict(directory_blocks, base_selected)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            csa_queries += 1
            if directory_blocks_per_token > 0 and lut.max_entries > 0:
                if should_probe:
                    directory_read_bytes += directory_entry_bytes * max(
                        0,
                        len(readable_directory_blocks) - 1,
                    )
                else:
                    directory_read_bytes += directory_entry_bytes * max(
                        1,
                        len(readable_directory_blocks),
                    )
            if directory_hit:
                directory_hits += 1
                directory_entries_seen += len(directory_blocks)
                directory_blocks_read += len(readable_directory_blocks)
                expanded_read_queries += int(len(readable_directory_blocks) > min_read_blocks_per_token)

        token_reads += len(selected) * config.block_size
        if not is_relevant:
            continue

        relevant_queries += 1
        exact_total = sum(int(value) for value in block_counts.values())
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)
        repaired_hits += _block_hit(selected, block_counts)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    rare_denominator = rare_relevant_queries if rare_relevant_queries else 1
    directory_hit_denominator = directory_hits if directory_hits else 1
    token_reads_per_query = token_reads / query_denominator
    spread_metadata_state_bytes = len(directory) * 2 / 8

    return CsaHcaRareDirectoryJointPolicyPoint(
        policy=policy,
        scenario=scenario,
        hca_threshold=hca_threshold,
        probe_mode=probe_mode,
        directory_blocks_per_token=directory_blocks_per_token,
        probe_lut_state_bytes=probe_lut.state_bytes,
        fanout_lut_state_bytes=lut.state_bytes,
        fanout_metadata_state_bytes=lut.state_bytes + spread_metadata_state_bytes,
        directory_state_bytes=directory_state_bytes,
        directory_probe_rate=directory_probes / query_denominator,
        directory_hit_rate=directory_hits / query_denominator,
        avg_directory_entries_per_hit=directory_entries_seen / directory_hit_denominator,
        avg_directory_read_blocks_per_hit=directory_blocks_read / directory_hit_denominator,
        expanded_read_rate=expanded_read_queries / directory_hit_denominator,
        hca_query_rate=hca_queries / query_denominator,
        csa_query_rate=csa_queries / query_denominator,
        rare_false_hca_rate=rare_false_hca / rare_denominator,
        repaired_relevant_hit_rate=repaired_hits / relevant_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_reads_per_query=token_reads_per_query,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_hca_route_lut_point(
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    directory_state_bytes: float,
    hca_threshold: int,
    route_lut: LowBitHcaRouteLUT,
    fanout_lut: LowBitRareDirectoryFanoutLUT,
    min_read_blocks_per_token: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
) -> CsaHcaRareDirectoryJointPolicyPoint:
    hca_queries = 0
    csa_queries = 0
    directory_hits = 0
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    repaired_hits = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    directory_entries_seen = 0
    directory_blocks_read = 0
    expanded_read_queries = 0
    min_read_blocks_per_token = min(max(0, int(min_read_blocks_per_token)), directory_blocks_per_token)

    for token in query_tokens:
        token = int(token)
        counter_values = _dense_counter_values(global_summary, token)
        route_hca = route_lut.route_hca(counter_values)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        block_counts = exact_counts.get(token)
        is_relevant = block_counts is not None

        if route_hca:
            selected = recent_blocks
            hca_queries += 1
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = fanout_lut.predict(directory_blocks, base_selected)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            csa_queries += 1
            if directory_blocks_per_token > 0 and fanout_lut.max_entries > 0:
                directory_read_bytes += directory_entry_bytes * max(
                    1,
                    len(readable_directory_blocks),
                )
            if directory_hit:
                directory_hits += 1
                directory_entries_seen += len(directory_blocks)
                directory_blocks_read += len(readable_directory_blocks)
                expanded_read_queries += int(len(readable_directory_blocks) > min_read_blocks_per_token)

        token_reads += len(selected) * config.block_size
        if not is_relevant:
            continue

        relevant_queries += 1
        exact_total = sum(int(value) for value in block_counts.values())
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)
        repaired_hits += _block_hit(selected, block_counts)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    rare_denominator = rare_relevant_queries if rare_relevant_queries else 1
    directory_hit_denominator = directory_hits if directory_hits else 1
    token_reads_per_query = token_reads / query_denominator
    spread_metadata_state_bytes = len(directory) * 2 / 8

    return CsaHcaRareDirectoryJointPolicyPoint(
        policy="hca_route_lut",
        scenario=scenario,
        hca_threshold=hca_threshold,
        probe_mode="route_lut",
        directory_blocks_per_token=directory_blocks_per_token,
        probe_lut_state_bytes=route_lut.state_bytes,
        fanout_lut_state_bytes=fanout_lut.state_bytes,
        fanout_metadata_state_bytes=fanout_lut.state_bytes + spread_metadata_state_bytes,
        directory_state_bytes=directory_state_bytes,
        directory_probe_rate=0.0,
        directory_hit_rate=directory_hits / query_denominator,
        avg_directory_entries_per_hit=directory_entries_seen / directory_hit_denominator,
        avg_directory_read_blocks_per_hit=directory_blocks_read / directory_hit_denominator,
        expanded_read_rate=expanded_read_queries / directory_hit_denominator,
        hca_query_rate=hca_queries / query_denominator,
        csa_query_rate=csa_queries / query_denominator,
        rare_false_hca_rate=rare_false_hca / rare_denominator,
        repaired_relevant_hit_rate=repaired_hits / relevant_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_reads_per_query=token_reads_per_query,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_aware_hca_route_lut_point(
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    directory_state_bytes: float,
    hca_threshold: int,
    route_lut: LowBitDirectoryAwareHcaRouteLUT,
    fanout_lut: LowBitRareDirectoryFanoutLUT,
    min_read_blocks_per_token: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
    route_feature_read_bytes: float,
) -> CsaHcaRareDirectoryJointPolicyPoint:
    hca_queries = 0
    csa_queries = 0
    directory_hits = 0
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    repaired_hits = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    directory_entries_seen = 0
    directory_blocks_read = 0
    expanded_read_queries = 0
    min_read_blocks_per_token = min(max(0, int(min_read_blocks_per_token)), directory_blocks_per_token)

    for token in query_tokens:
        token = int(token)
        counter_values = _dense_counter_values(global_summary, token)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        directory_read_bytes += route_feature_read_bytes
        directory_hits += int(directory_hit)
        directory_entries_seen += len(directory_blocks)

        route_hca = route_lut.route_hca(counter_values, directory_blocks)
        block_counts = exact_counts.get(token)
        is_relevant = block_counts is not None

        if route_hca:
            selected = recent_blocks
            hca_queries += 1
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = fanout_lut.predict(directory_blocks, base_selected)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            csa_queries += 1
            if directory_hit:
                directory_read_bytes += directory_entry_bytes * len(readable_directory_blocks)
                directory_blocks_read += len(readable_directory_blocks)
                expanded_read_queries += int(len(readable_directory_blocks) > min_read_blocks_per_token)

        token_reads += len(selected) * config.block_size
        if not is_relevant:
            continue

        relevant_queries += 1
        exact_total = sum(int(value) for value in block_counts.values())
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)
        repaired_hits += _block_hit(selected, block_counts)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    rare_denominator = rare_relevant_queries if rare_relevant_queries else 1
    directory_hit_denominator = directory_hits if directory_hits else 1
    token_reads_per_query = token_reads / query_denominator
    spread_metadata_state_bytes = len(directory) * 2 / 8

    return CsaHcaRareDirectoryJointPolicyPoint(
        policy="dir_aware_route_lut",
        scenario=scenario,
        hca_threshold=hca_threshold,
        probe_mode="presence_bit_route_lut",
        directory_blocks_per_token=directory_blocks_per_token,
        probe_lut_state_bytes=route_lut.state_bytes,
        fanout_lut_state_bytes=fanout_lut.state_bytes,
        fanout_metadata_state_bytes=fanout_lut.state_bytes + spread_metadata_state_bytes,
        directory_state_bytes=directory_state_bytes,
        directory_probe_rate=1.0,
        directory_hit_rate=directory_hits / query_denominator,
        avg_directory_entries_per_hit=directory_entries_seen / directory_hit_denominator,
        avg_directory_read_blocks_per_hit=directory_blocks_read / directory_hit_denominator,
        expanded_read_rate=expanded_read_queries / directory_hit_denominator,
        hca_query_rate=hca_queries / query_denominator,
        csa_query_rate=csa_queries / query_denominator,
        rare_false_hca_rate=rare_false_hca / rare_denominator,
        repaired_relevant_hit_rate=repaired_hits / relevant_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_reads_per_query=token_reads_per_query,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_presence_sidecar_point(
    false_positive_rate: float,
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    directory_state_bytes: float,
    sidecar_state_bytes: float,
    hca_threshold: int,
    route_lut: LowBitDirectoryAwareHcaRouteLUT,
    fanout_lut: LowBitRareDirectoryFanoutLUT,
    min_read_blocks_per_token: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
    route_feature_read_bytes: float,
    sidecar_salt: int,
) -> CsaHcaRareDirectoryPresenceSidecarPoint:
    hca_queries = 0
    csa_queries = 0
    directory_hits = 0
    sidecar_false_positive_queries = 0
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    repaired_hits = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    min_read_blocks_per_token = min(max(0, int(min_read_blocks_per_token)), directory_blocks_per_token)

    for token in query_tokens:
        token = int(token)
        counter_values = _dense_counter_values(global_summary, token)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        sidecar_false_positive = (not directory_hit) and _presence_sidecar_false_positive(
            token=token,
            false_positive_rate=false_positive_rate,
            salt=sidecar_salt,
        )
        if directory_hit:
            visible_directory_blocks = directory_blocks
        elif sidecar_false_positive:
            visible_directory_blocks = np.array([-1], dtype=np.int32)
        else:
            visible_directory_blocks = np.empty(0, dtype=np.int32)
        directory_read_bytes += route_feature_read_bytes
        directory_hits += int(directory_hit)
        sidecar_false_positive_queries += int(sidecar_false_positive)

        route_hca = route_lut.route_hca(counter_values, visible_directory_blocks)
        block_counts = exact_counts.get(token)
        is_relevant = block_counts is not None

        if route_hca:
            selected = recent_blocks
            hca_queries += 1
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = fanout_lut.predict(directory_blocks, base_selected)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            csa_queries += 1
            if directory_hit:
                directory_read_bytes += directory_entry_bytes * len(readable_directory_blocks)

        token_reads += len(selected) * config.block_size
        if not is_relevant:
            continue

        relevant_queries += 1
        exact_total = sum(int(value) for value in block_counts.values())
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)
        repaired_hits += _block_hit(selected, block_counts)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    rare_denominator = rare_relevant_queries if rare_relevant_queries else 1
    token_reads_per_query = token_reads / query_denominator

    return CsaHcaRareDirectoryPresenceSidecarPoint(
        false_positive_rate=false_positive_rate,
        scenario=scenario,
        hca_threshold=hca_threshold,
        route_lut_state_bytes=route_lut.state_bytes,
        sidecar_state_bytes=sidecar_state_bytes,
        fanout_lut_state_bytes=fanout_lut.state_bytes,
        directory_state_bytes=directory_state_bytes,
        route_feature_read_bytes=route_feature_read_bytes,
        sidecar_false_positive_query_rate=sidecar_false_positive_queries / query_denominator,
        directory_hit_rate=directory_hits / query_denominator,
        hca_query_rate=hca_queries / query_denominator,
        csa_query_rate=csa_queries / query_denominator,
        rare_false_hca_rate=rare_false_hca / rare_denominator,
        repaired_relevant_hit_rate=repaired_hits / relevant_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_reads_per_query=token_reads_per_query,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_bloom_sidecar_point(
    bits_per_entry: int,
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    directory_state_bytes: float,
    hca_threshold: int,
    route_lut: LowBitDirectoryAwareHcaRouteLUT,
    fanout_lut: LowBitRareDirectoryFanoutLUT,
    sidecar: LowBitPresenceBloomSidecar,
    min_read_blocks_per_token: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
    directory_tokens: Tuple[int, ...],
) -> CsaHcaRareDirectoryBloomSidecarPoint:
    hca_queries = 0
    csa_queries = 0
    directory_hits = 0
    sidecar_false_positive_queries = 0
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    repaired_hits = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    query_bank_conflicts = 0
    query_unique_banks = 0
    min_read_blocks_per_token = min(max(0, int(min_read_blocks_per_token)), directory_blocks_per_token)

    for token in query_tokens:
        token = int(token)
        counter_values = _dense_counter_values(global_summary, token)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        slots = sidecar.slots(token)
        banks = sidecar.banks(slots)
        unique_banks = len(set(banks))
        query_unique_banks += unique_banks
        query_bank_conflicts += int(unique_banks < len(banks))
        sidecar_visible = all(bool(sidecar.bits[slot]) for slot in slots)
        sidecar_false_positive = sidecar_visible and not directory_hit
        if directory_hit:
            visible_directory_blocks = directory_blocks
        elif sidecar_false_positive:
            visible_directory_blocks = np.array([-1], dtype=np.int32)
        else:
            visible_directory_blocks = np.empty(0, dtype=np.int32)
        directory_read_bytes += sidecar.read_bytes_per_query
        directory_hits += int(directory_hit)
        sidecar_false_positive_queries += int(sidecar_false_positive)

        route_hca = route_lut.route_hca(counter_values, visible_directory_blocks)
        block_counts = exact_counts.get(token)
        is_relevant = block_counts is not None

        if route_hca:
            selected = recent_blocks
            hca_queries += 1
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = fanout_lut.predict(directory_blocks, base_selected)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            csa_queries += 1
            if directory_hit:
                directory_read_bytes += directory_entry_bytes * len(readable_directory_blocks)

        token_reads += len(selected) * config.block_size
        if not is_relevant:
            continue

        relevant_queries += 1
        exact_total = sum(int(value) for value in block_counts.values())
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)
        repaired_hits += _block_hit(selected, block_counts)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    update_conflicts = 0
    update_unique_banks = 0
    for token in directory_tokens:
        slots = sidecar.slots(int(token))
        banks = sidecar.banks(slots)
        unique_banks = len(set(banks))
        update_unique_banks += unique_banks
        update_conflicts += int(unique_banks < len(banks))

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    update_denominator = len(directory_tokens) if len(directory_tokens) else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    rare_denominator = rare_relevant_queries if rare_relevant_queries else 1
    token_reads_per_query = token_reads / query_denominator

    return CsaHcaRareDirectoryBloomSidecarPoint(
        bits_per_entry=bits_per_entry,
        hash_count=sidecar.hash_count,
        bank_count=sidecar.bank_count,
        bank_mode=sidecar.bank_mode,
        scenario=scenario,
        hca_threshold=hca_threshold,
        route_lut_state_bytes=route_lut.state_bytes,
        sidecar_state_bytes=sidecar.state_bytes,
        fanout_lut_state_bytes=fanout_lut.state_bytes,
        directory_state_bytes=directory_state_bytes,
        sidecar_entries=len(directory_tokens),
        read_bytes_per_query=sidecar.read_bytes_per_query,
        write_bytes_per_insert=sidecar.write_bytes_per_insert,
        update_bytes_per_context_token=(
            len(directory_tokens) * sidecar.write_bytes_per_insert / config.context_length
        ),
        sidecar_false_positive_query_rate=sidecar_false_positive_queries / query_denominator,
        query_bank_conflict_rate=query_bank_conflicts / query_denominator,
        update_bank_conflict_rate=update_conflicts / update_denominator,
        avg_query_unique_banks=query_unique_banks / query_denominator,
        avg_update_unique_banks=update_unique_banks / update_denominator,
        directory_hit_rate=directory_hits / query_denominator,
        hca_query_rate=hca_queries / query_denominator,
        csa_query_rate=csa_queries / query_denominator,
        rare_false_hca_rate=rare_false_hca / rare_denominator,
        repaired_relevant_hit_rate=repaired_hits / relevant_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_reads_per_query=token_reads_per_query,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_bloom_salt_point(
    salt_index: int,
    sidecar_salt: int,
    bits_per_entry: int,
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    hca_threshold: int,
    route_lut: LowBitDirectoryAwareHcaRouteLUT,
    fanout_lut: LowBitRareDirectoryFanoutLUT,
    sidecar: LowBitPresenceBloomSidecar,
    min_read_blocks_per_token: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
) -> CsaHcaRareDirectoryBloomSaltPoint:
    hca_queries = 0
    csa_queries = 0
    sidecar_false_positive_queries = 0
    hot_queries = 0
    hot_sidecar_false_positive_queries = 0
    relevant_queries = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    query_bank_conflicts = 0
    min_read_blocks_per_token = min(max(0, int(min_read_blocks_per_token)), directory_blocks_per_token)

    for token in query_tokens:
        token = int(token)
        counter_values = _dense_counter_values(global_summary, token)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        slots = sidecar.slots(token)
        banks = sidecar.banks(slots)
        query_bank_conflicts += int(len(set(banks)) < len(banks))
        sidecar_visible = all(bool(sidecar.bits[slot]) for slot in slots)
        sidecar_false_positive = sidecar_visible and not directory_hit
        if directory_hit:
            visible_directory_blocks = directory_blocks
        elif sidecar_false_positive:
            visible_directory_blocks = np.array([-1], dtype=np.int32)
        else:
            visible_directory_blocks = np.empty(0, dtype=np.int32)
        directory_read_bytes += sidecar.read_bytes_per_query
        sidecar_false_positive_queries += int(sidecar_false_positive)

        block_counts = exact_counts.get(token)
        exact_total = 0 if block_counts is None else sum(int(value) for value in block_counts.values())
        is_hot = exact_total >= hca_threshold
        hot_queries += int(is_hot)
        hot_sidecar_false_positive_queries += int(is_hot and sidecar_false_positive)

        route_hca = route_lut.route_hca(counter_values, visible_directory_blocks)
        if route_hca:
            selected = recent_blocks
            hca_queries += 1
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = fanout_lut.predict(directory_blocks, base_selected)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            csa_queries += 1
            if directory_hit:
                directory_read_bytes += directory_entry_bytes * len(readable_directory_blocks)

        token_reads += len(selected) * config.block_size
        if block_counts is None:
            continue
        relevant_queries += 1
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    hot_denominator = hot_queries if hot_queries else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    token_reads_per_query = token_reads / query_denominator

    return CsaHcaRareDirectoryBloomSaltPoint(
        salt_index=salt_index,
        sidecar_salt=sidecar_salt,
        bits_per_entry=bits_per_entry,
        hash_count=sidecar.hash_count,
        bank_count=sidecar.bank_count,
        bank_mode=sidecar.bank_mode,
        scenario=scenario,
        sidecar_state_bytes=sidecar.state_bytes,
        read_bytes_per_query=sidecar.read_bytes_per_query,
        write_bytes_per_insert=sidecar.write_bytes_per_insert,
        sidecar_false_positive_query_rate=sidecar_false_positive_queries / query_denominator,
        hot_query_rate=hot_queries / query_denominator,
        hot_sidecar_false_positive_rate=hot_sidecar_false_positive_queries / hot_denominator,
        query_bank_conflict_rate=query_bank_conflicts / query_denominator,
        hca_query_rate=hca_queries / query_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_bloom_streaming_point(
    policy: int,
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    hca_threshold: int,
    route_lut: LowBitDirectoryAwareHcaRouteLUT,
    fanout_lut: LowBitRareDirectoryFanoutLUT,
    min_read_blocks_per_token: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
    stream: np.ndarray,
    bits_per_entry: int,
    hash_count: int,
    bank_count: int,
    bank_mode: str,
    sidecar_salt: int,
) -> CsaHcaRareDirectoryBloomStreamingPoint:
    final_rare_tokens = tuple(sorted(int(token) for token in directory))
    bit_count = max(1, int(max(1, len(final_rare_tokens)) * bits_per_entry))
    sidecar = LowBitPresenceBloomSidecar(
        bit_count=bit_count,
        hash_count=hash_count,
        bank_count=bank_count,
        bank_mode=bank_mode,
        salt=sidecar_salt,
    )

    final_totals = {
        int(token): sum(int(value) for value in block_counts.values())
        for token, block_counts in exact_counts.items()
    }
    final_hot_tokens = {
        int(token)
        for token, total in final_totals.items()
        if int(total) >= hca_threshold
    }
    inserted_tokens = set()
    bank_writes = np.zeros(bank_count, dtype=np.int64)
    update_conflicts = 0
    update_unique_banks = 0

    def insert_token(token: int) -> None:
        nonlocal update_conflicts, update_unique_banks
        if int(token) in inserted_tokens:
            return
        slots = sidecar.slots(int(token))
        banks = sidecar.banks(slots)
        unique_banks = len(set(banks))
        for bank in banks:
            bank_writes[int(bank)] += 1
        update_unique_banks += unique_banks
        update_conflicts += int(unique_banks < len(banks))
        sidecar.insert(int(token))
        inserted_tokens.add(int(token))

    if policy == 0:
        for token in final_rare_tokens:
            insert_token(int(token))
        policy_name = "final_oracle"
    else:
        counts: Dict[int, int] = {}
        for token in stream:
            token = int(token)
            count = counts.get(token, 0) + 1
            counts[token] = count
            if count == policy and count < hca_threshold:
                insert_token(token)
        policy_name = f"count{policy}"

    inserted_final_rare = sum(1 for token in inserted_tokens if token in directory)
    inserted_hot_tokens = sum(1 for token in inserted_tokens if token in final_hot_tokens)
    insert_denominator = len(inserted_tokens) if inserted_tokens else 1
    hot_denominator = len(final_hot_tokens) if final_hot_tokens else 1
    rare_denominator_for_insert = len(final_rare_tokens) if final_rare_tokens else 1

    hca_queries = 0
    csa_queries = 0
    sidecar_false_positive_queries = 0
    hot_queries = 0
    hot_sidecar_false_positive_queries = 0
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    min_read_blocks_per_token = min(max(0, int(min_read_blocks_per_token)), directory_blocks_per_token)

    for token in query_tokens:
        token = int(token)
        counter_values = _dense_counter_values(global_summary, token)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        sidecar_visible = sidecar.query(token)
        sidecar_false_positive = sidecar_visible and not directory_hit
        if directory_hit:
            visible_directory_blocks = directory_blocks
        elif sidecar_false_positive:
            visible_directory_blocks = np.array([-1], dtype=np.int32)
        else:
            visible_directory_blocks = np.empty(0, dtype=np.int32)
        directory_read_bytes += sidecar.read_bytes_per_query
        sidecar_false_positive_queries += int(sidecar_false_positive)

        block_counts = exact_counts.get(token)
        exact_total = 0 if block_counts is None else sum(int(value) for value in block_counts.values())
        is_hot = exact_total >= hca_threshold
        hot_queries += int(is_hot)
        hot_sidecar_false_positive_queries += int(is_hot and sidecar_false_positive)

        route_hca = route_lut.route_hca(counter_values, visible_directory_blocks)
        if route_hca:
            selected = recent_blocks
            hca_queries += 1
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = fanout_lut.predict(directory_blocks, base_selected)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            csa_queries += 1
            if directory_hit:
                directory_read_bytes += directory_entry_bytes * len(readable_directory_blocks)

        token_reads += len(selected) * config.block_size
        if block_counts is None:
            continue
        relevant_queries += 1
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    hot_query_denominator = hot_queries if hot_queries else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    rare_query_denominator = rare_relevant_queries if rare_relevant_queries else 1
    token_reads_per_query = token_reads / query_denominator
    insert_events = len(inserted_tokens)
    bank_write_bytes = bank_writes * sidecar.write_bytes_per_insert / max(1, hash_count)

    return CsaHcaRareDirectoryBloomStreamingPoint(
        policy=policy_name,
        insert_count_threshold=policy,
        scenario=scenario,
        bits_per_entry=bits_per_entry,
        hash_count=hash_count,
        bank_count=bank_count,
        bank_mode=bank_mode,
        sidecar_salt=sidecar_salt,
        sidecar_state_bytes=sidecar.state_bytes,
        sidecar_fill_rate=float(np.count_nonzero(sidecar.bits) / sidecar.bit_count),
        inserted_tokens=insert_events,
        final_rare_tokens=len(final_rare_tokens),
        final_hot_tokens=len(final_hot_tokens),
        inserted_final_rare_rate=inserted_final_rare / rare_denominator_for_insert,
        hot_polluted_token_rate=inserted_hot_tokens / hot_denominator,
        update_bytes_per_context_token=insert_events * sidecar.write_bytes_per_insert / config.context_length,
        max_bank_update_bytes_per_context_token=float(np.max(bank_write_bytes) / config.context_length),
        update_bank_conflict_rate=update_conflicts / insert_denominator,
        avg_update_unique_banks=update_unique_banks / insert_denominator,
        sidecar_false_positive_query_rate=sidecar_false_positive_queries / query_denominator,
        hot_sidecar_false_positive_rate=hot_sidecar_false_positive_queries / hot_query_denominator,
        hca_query_rate=hca_queries / query_denominator,
        rare_false_hca_rate=rare_false_hca / rare_query_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _evaluate_rare_directory_bloom_retirement_point(
    insert_count_threshold: int,
    retire_count_threshold: int,
    scenario: str,
    config: CompressedBlockIndexConfig,
    index: LowBitCompressedBlockIndex,
    global_summary: LowBitDenseContext,
    exact_counts: Dict[int, Dict[int, int]],
    directory: Dict[int, np.ndarray],
    directory_blocks_per_token: int,
    directory_entry_bytes: float,
    hca_threshold: int,
    route_lut: LowBitDirectoryAwareHcaRouteLUT,
    fanout_lut: LowBitRareDirectoryFanoutLUT,
    min_read_blocks_per_token: int,
    recent_blocks: np.ndarray,
    query_tokens: np.ndarray,
    stream: np.ndarray,
    bits_per_entry: int,
    hash_count: int,
    counter_bits: int,
    bank_count: int,
    bank_mode: str,
    sidecar_salt: int,
) -> CsaHcaRareDirectoryBloomRetirementPoint:
    final_rare_tokens = tuple(sorted(int(token) for token in directory))
    bit_count = max(1, int(max(1, len(final_rare_tokens)) * bits_per_entry))
    sidecar = LowBitCountingBloomSidecar(
        bit_count=bit_count,
        hash_count=hash_count,
        counter_bits=counter_bits,
        bank_count=bank_count,
        bank_mode=bank_mode,
        salt=sidecar_salt,
    )

    final_totals = {
        int(token): sum(int(value) for value in block_counts.values())
        for token, block_counts in exact_counts.items()
    }
    final_hot_tokens = {
        int(token)
        for token, total in final_totals.items()
        if int(total) >= hca_threshold
    }
    ever_inserted_tokens = set()
    active_tokens = set()
    deleted_tokens = set()
    bank_writes = np.zeros(bank_count, dtype=np.int64)
    update_conflicts = 0
    update_unique_banks = 0
    update_events = 0

    def record_update(slots: Tuple[int, ...]) -> None:
        nonlocal update_conflicts, update_unique_banks, update_events
        banks = sidecar.banks(slots)
        unique_banks = len(set(banks))
        for bank in banks:
            bank_writes[int(bank)] += 1
        update_unique_banks += unique_banks
        update_conflicts += int(unique_banks < len(banks))
        update_events += 1

    def insert_token(token: int) -> None:
        if int(token) in ever_inserted_tokens:
            return
        slots = sidecar.slots(int(token))
        record_update(slots)
        sidecar.insert(int(token))
        ever_inserted_tokens.add(int(token))
        active_tokens.add(int(token))

    def delete_token(token: int) -> None:
        if int(token) not in active_tokens:
            return
        slots = sidecar.slots(int(token))
        record_update(slots)
        sidecar.delete(int(token))
        active_tokens.remove(int(token))
        deleted_tokens.add(int(token))

    counts: Dict[int, int] = {}
    for token in stream:
        token = int(token)
        count = counts.get(token, 0) + 1
        counts[token] = count
        if count == insert_count_threshold and count < retire_count_threshold:
            insert_token(token)
        if count == retire_count_threshold:
            delete_token(token)

    inserted_final_rare = sum(1 for token in ever_inserted_tokens if token in directory)
    active_final_rare_tokens = [int(token) for token in active_tokens if token in directory]
    active_final_rare = len(active_final_rare_tokens)
    visible_active_rare = sum(1 for token in active_final_rare_tokens if sidecar.query(token))
    active_hot_tokens = sum(1 for token in active_tokens if token in final_hot_tokens)
    retired_hot_tokens = sum(1 for token in deleted_tokens if token in final_hot_tokens)
    update_denominator = update_events if update_events else 1
    hot_denominator = len(final_hot_tokens) if final_hot_tokens else 1
    rare_denominator_for_insert = len(final_rare_tokens) if final_rare_tokens else 1

    hca_queries = 0
    csa_queries = 0
    sidecar_false_positive_queries = 0
    hot_queries = 0
    hot_sidecar_false_positive_queries = 0
    relevant_queries = 0
    rare_relevant_queries = 0
    rare_false_hca = 0
    repaired_coverage = 0.0
    token_reads = 0.0
    directory_read_bytes = 0.0
    min_read_blocks_per_token = min(max(0, int(min_read_blocks_per_token)), directory_blocks_per_token)

    for token in query_tokens:
        token = int(token)
        counter_values = _dense_counter_values(global_summary, token)
        directory_blocks = directory.get(token, np.empty(0, dtype=np.int32))
        directory_hit = len(directory_blocks) > 0
        sidecar_visible = sidecar.query(token)
        sidecar_false_positive = sidecar_visible and not directory_hit
        if directory_hit and sidecar_visible:
            visible_directory_blocks = directory_blocks
        elif sidecar_false_positive:
            visible_directory_blocks = np.array([-1], dtype=np.int32)
        else:
            visible_directory_blocks = np.empty(0, dtype=np.int32)
        directory_read_bytes += sidecar.read_bytes_per_query
        sidecar_false_positive_queries += int(sidecar_false_positive)

        block_counts = exact_counts.get(token)
        exact_total = 0 if block_counts is None else sum(int(value) for value in block_counts.values())
        is_hot = exact_total >= hca_threshold
        hot_queries += int(is_hot)
        hot_sidecar_false_positive_queries += int(is_hot and sidecar_false_positive)

        route_hca = route_lut.route_hca(counter_values, visible_directory_blocks)
        if route_hca:
            selected = recent_blocks
            hca_queries += 1
        else:
            scores = index.estimate_blocks(token)
            base_selected = np.union1d(_top_blocks(scores, config.selected_blocks), recent_blocks)
            read_limit = fanout_lut.predict(directory_blocks, base_selected)
            readable_directory_blocks = directory_blocks[:read_limit]
            selected = np.union1d(base_selected, readable_directory_blocks)
            csa_queries += 1
            if directory_hit and sidecar_visible:
                directory_read_bytes += directory_entry_bytes * len(readable_directory_blocks)

        token_reads += len(selected) * config.block_size
        if block_counts is None:
            continue
        relevant_queries += 1
        is_exact_rare = exact_total < hca_threshold
        rare_relevant_queries += int(is_exact_rare)
        rare_false_hca += int(is_exact_rare and route_hca)
        repaired_coverage += _occurrence_coverage(selected, block_counts)

    query_denominator = len(query_tokens) if len(query_tokens) else 1
    hot_query_denominator = hot_queries if hot_queries else 1
    relevant_denominator = relevant_queries if relevant_queries else 1
    rare_query_denominator = rare_relevant_queries if rare_relevant_queries else 1
    token_reads_per_query = token_reads / query_denominator
    bank_write_bytes = bank_writes * sidecar.write_bytes_per_insert / max(1, hash_count)

    return CsaHcaRareDirectoryBloomRetirementPoint(
        policy=f"count{insert_count_threshold}_retire{retire_count_threshold}",
        insert_count_threshold=insert_count_threshold,
        retire_count_threshold=retire_count_threshold,
        scenario=scenario,
        bits_per_entry=bits_per_entry,
        hash_count=hash_count,
        counter_bits=counter_bits,
        bank_count=bank_count,
        bank_mode=bank_mode,
        sidecar_salt=sidecar_salt,
        sidecar_state_bytes=sidecar.state_bytes,
        sidecar_fill_rate=float(np.count_nonzero(sidecar.bits) / sidecar.bit_count),
        inserted_tokens=len(ever_inserted_tokens),
        active_tokens=len(active_tokens),
        deleted_tokens=len(deleted_tokens),
        final_rare_tokens=len(final_rare_tokens),
        final_hot_tokens=len(final_hot_tokens),
        inserted_final_rare_rate=inserted_final_rare / rare_denominator_for_insert,
        active_final_rare_rate=active_final_rare / rare_denominator_for_insert,
        visible_active_rare_rate=visible_active_rare / rare_denominator_for_insert,
        hot_retired_token_rate=retired_hot_tokens / hot_denominator,
        hot_polluted_token_rate=active_hot_tokens / hot_denominator,
        update_bytes_per_context_token=update_events * sidecar.write_bytes_per_insert / config.context_length,
        max_bank_update_bytes_per_context_token=float(np.max(bank_write_bytes) / config.context_length),
        update_bank_conflict_rate=update_conflicts / update_denominator,
        avg_update_unique_banks=update_unique_banks / update_denominator,
        sidecar_false_positive_query_rate=sidecar_false_positive_queries / query_denominator,
        hot_sidecar_false_positive_rate=hot_sidecar_false_positive_queries / hot_query_denominator,
        hca_query_rate=hca_queries / query_denominator,
        rare_false_hca_rate=rare_false_hca / rare_query_denominator,
        repaired_relevant_coverage=repaired_coverage / relevant_denominator,
        directory_read_bytes_per_query=directory_read_bytes / query_denominator,
        token_read_reduction=_safe_divide(config.context_length, token_reads_per_query),
    )


def _rare_fanout_lut_index(
    directory_blocks: np.ndarray,
    base_selected: np.ndarray,
    max_entries: int,
    span_thresholds: Tuple[int, ...],
    max_overlap_bucket: int,
) -> int:
    entry_count = min(len(directory_blocks), max(0, int(max_entries)))
    span_bucket = _rare_directory_span_bucket(directory_blocks[:entry_count], span_thresholds)
    if entry_count == 0:
        overlap_bucket = 0
    else:
        base_set = {int(block) for block in base_selected}
        overlap = sum(1 for block in directory_blocks[:entry_count] if int(block) in base_set)
        overlap_bucket = min(overlap, max(0, int(max_overlap_bucket)))
    span_bucket_count = len(span_thresholds) + 1
    overlap_bucket_count = max(0, int(max_overlap_bucket)) + 1
    return (entry_count * span_bucket_count + span_bucket) * overlap_bucket_count + overlap_bucket


def _hca_probe_lut_index(
    counter_values: Tuple[int, ...],
    max_counter: int,
    spread_thresholds: Tuple[int, ...],
    max_saturation_bucket: int,
) -> int:
    return _hca_control_lut_index(
        counter_values=counter_values,
        max_counter=max_counter,
        spread_thresholds=spread_thresholds,
        max_saturation_bucket=max_saturation_bucket,
    )


def _hca_control_lut_index(
    counter_values: Tuple[int, ...],
    max_counter: int,
    spread_thresholds: Tuple[int, ...],
    max_saturation_bucket: int,
) -> int:
    if len(counter_values) == 0:
        estimate = 0
        spread = 0
        saturation = 0
    else:
        clipped = tuple(min(max(0, int(value)), int(max_counter)) for value in counter_values)
        estimate = min(clipped)
        spread = max(clipped) - estimate
        saturation = sum(value == int(max_counter) for value in clipped)
    spread_bucket = 0
    for threshold in sorted(int(value) for value in spread_thresholds):
        if spread >= threshold:
            spread_bucket += 1
    saturation_bucket = min(saturation, max(0, int(max_saturation_bucket)))
    spread_bucket_count = len(spread_thresholds) + 1
    saturation_bucket_count = max(0, int(max_saturation_bucket)) + 1
    return (estimate * spread_bucket_count + spread_bucket) * saturation_bucket_count + saturation_bucket


def _directory_aware_hca_route_lut_index(
    counter_values: Tuple[int, ...],
    directory_blocks: np.ndarray,
    max_counter: int,
    spread_thresholds: Tuple[int, ...],
    max_saturation_bucket: int,
) -> int:
    hca_index = _hca_control_lut_index(
        counter_values=counter_values,
        max_counter=max_counter,
        spread_thresholds=spread_thresholds,
        max_saturation_bucket=max_saturation_bucket,
    )
    directory_hit_bucket = int(len(directory_blocks) > 0)
    return hca_index * 2 + directory_hit_bucket


def _presence_sidecar_false_positive(
    token: int,
    false_positive_rate: float,
    salt: int,
) -> bool:
    if false_positive_rate <= 0.0:
        return False
    threshold = int(float(false_positive_rate) * (1 << 64))
    return keyed_hash(int(token), int(salt)) < threshold


def _presence_sidecar_state_bytes(entries: int, false_positive_rate: float) -> float:
    entries = max(0, int(entries))
    if entries == 0:
        return 0.0
    if false_positive_rate <= 0.0:
        return entries / 8
    bits_per_entry = -float(np.log(false_positive_rate)) / float(np.log(2.0) ** 2)
    return entries * bits_per_entry / 8


def _dense_counter_values(summary: LowBitDenseContext, token: int) -> Tuple[int, ...]:
    slots = summary._slots(int(token))
    return tuple(
        int(summary.counters[bank, slot])
        for bank, slot in enumerate(slots)
    )


def _decode_rare_fanout_lut_index(
    index: int,
    span_bucket_count: int,
    overlap_bucket_count: int,
) -> Tuple[int, int, int]:
    entry_count = int(index) // (span_bucket_count * overlap_bucket_count)
    remainder = int(index) % (span_bucket_count * overlap_bucket_count)
    span_bucket = remainder // overlap_bucket_count
    overlap_bucket = remainder % overlap_bucket_count
    return entry_count, span_bucket, overlap_bucket


def _rare_directory_span_bucket(
    directory_blocks: np.ndarray,
    span_thresholds: Tuple[int, ...],
) -> int:
    if len(directory_blocks) <= 1:
        return 0
    span = int(np.max(directory_blocks)) - int(np.min(directory_blocks))
    bucket = 0
    for threshold in sorted(int(value) for value in span_thresholds):
        if span >= threshold:
            bucket += 1
    return bucket


def _adaptive_directory_read_limit(
    directory_blocks: np.ndarray,
    directory_blocks_per_token: int,
    base_read_blocks_per_token: int,
    expanded_read_blocks_per_token: int,
    spread_threshold_blocks: int,
) -> int:
    entry_count = min(len(directory_blocks), max(0, int(directory_blocks_per_token)))
    if entry_count == 0:
        return 0

    base_read = min(max(0, int(base_read_blocks_per_token)), entry_count)
    expanded_read = min(max(base_read, int(expanded_read_blocks_per_token)), entry_count)
    if entry_count <= base_read:
        return entry_count
    if spread_threshold_blocks <= 0:
        return expanded_read

    block_span = int(np.max(directory_blocks[:entry_count])) - int(np.min(directory_blocks[:entry_count]))
    if block_span >= spread_threshold_blocks:
        return expanded_read
    return base_read


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


def _build_rare_block_directory(
    exact_counts: Dict[int, Dict[int, int]],
    hca_threshold: int,
    max_blocks_per_token: int,
) -> Dict[int, np.ndarray]:
    if max_blocks_per_token == 0:
        return {}
    directory: Dict[int, np.ndarray] = {}
    for token, block_counts in exact_counts.items():
        total = sum(int(value) for value in block_counts.values())
        if total <= 0 or total >= hca_threshold:
            continue
        ordered = sorted(
            block_counts.items(),
            key=lambda item: (-int(item[1]), -int(item[0])),
        )
        blocks = [int(block) for block, _ in ordered[:max_blocks_per_token]]
        directory[int(token)] = np.array(blocks, dtype=np.int32)
    return directory


def _rare_directory_entry_bytes(vocab_size: int, blocks: int) -> float:
    token_bits = max(1, (vocab_size - 1).bit_length())
    block_bits = max(1, (blocks - 1).bit_length())
    valid_bits = 1
    return (token_bits + block_bits + valid_bits) / 8


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
