"""Unified hardware proxy accounting for HARC-CA prototypes.

The formulas here are deliberately simple. They are not a silicon power model;
they are a first-pass accounting layer for local CA traffic, sparse rule-bank
work, and Transformer KV-cache read volume.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cellular_moe import CellularMoEConfig, MoERolloutResult
from .hardware import TransformerKVMetrics, estimate_transformer_kv
from .synthetic_lm import SyntheticLMConfig, SyntheticLMResult


@dataclass(frozen=True)
class HarcEventEfficiency:
    """Per-event proxy metrics for a HARC-CA decode schedule."""

    context_length: int
    events: int
    moe_ticks_per_event: int
    exact_local_bytes_per_event: float
    dense_local_bytes_per_event: float
    candidate_local_bytes_per_event: float
    moe_local_bytes_per_event: float
    total_local_bytes_per_event: float
    onchip_state_bytes: float
    exact_memory_bytes: float
    dense_memory_bytes: float
    candidate_memory_bytes: float
    moe_state_bytes: float
    exact_query_fraction: float
    exact_avg_visited_cells: float
    overflow_query_rate: float
    dense_update_cells_per_event: float
    candidate_update_cells_per_event: float
    candidate_gate_cells_per_event: float
    moe_sparse_rule_updates_per_event: float
    moe_dense_equivalent_rule_updates_per_event: float
    moe_update_reduction: float


@dataclass(frozen=True)
class EfficiencyComparison:
    """HARC-CA event proxy versus Transformer KV-cache proxy."""

    harc: HarcEventEfficiency
    transformer: TransformerKVMetrics
    local_vs_kv_byte_ratio: float

    @property
    def kv_vs_local_byte_ratio(self) -> float:
        if self.harc.total_local_bytes_per_event == 0.0:
            return 0.0
        return self.transformer.kv_read_bytes_per_token / self.harc.total_local_bytes_per_event


def estimate_harc_event_efficiency(
    synthetic: SyntheticLMResult,
    synthetic_config: SyntheticLMConfig,
    moe: MoERolloutResult,
    moe_config: CellularMoEConfig,
    moe_ticks_per_event: int = 4,
) -> HarcEventEfficiency:
    """Estimate local byte movement for one mixed synthetic decode event."""

    if moe_ticks_per_event <= 0:
        raise ValueError("moe_ticks_per_event must be positive")

    events = synthetic.topic_events + synthetic.query_events
    query_fraction = synthetic.query_events / events

    exact_entry_bytes = (synthetic_config.tag_bits + 32 + 1) / 8
    exact_local_bytes = synthetic.exact_avg_visited_cells * exact_entry_bytes * query_fraction

    exact_cells_per_event = synthetic.exact_avg_visited_cells * query_fraction
    candidate_cells_per_event = synthetic.candidate_update_cells_per_event
    candidate_gate_cells_per_event = synthetic.candidate_gate_cells_per_event
    dense_cells_per_event = max(
        0.0,
        synthetic.avg_cells_per_event
        - exact_cells_per_event
        - candidate_cells_per_event
        - candidate_gate_cells_per_event,
    )
    dense_counter_bytes = synthetic_config.dense_bits / 8
    dense_local_bytes = dense_cells_per_event * dense_counter_bytes * 2
    candidate_token_bits = max(1, (synthetic_config.vocab_size - 1).bit_length())
    candidate_entry_bytes = (
        candidate_token_bits + synthetic_config.candidate_cache_score_bits + 1
    ) / 8
    candidate_local_bytes = (
        candidate_cells_per_event * candidate_entry_bytes * 2
        + candidate_gate_cells_per_event * dense_counter_bytes
    )

    sparse_rule_updates_per_tick = moe_config.length * moe.avg_active_fraction * moe_config.top_k
    sparse_rule_updates_per_event = sparse_rule_updates_per_tick * moe_ticks_per_event
    dense_equiv_updates_per_event = moe_config.length * moe_config.rule_banks * moe_ticks_per_event
    cell_state_bytes = moe_config.channels * moe_config.bits / 8
    moe_local_bytes = sparse_rule_updates_per_event * cell_state_bytes * 4

    moe_state_bytes = moe_config.length * cell_state_bytes
    total_local_bytes = exact_local_bytes + dense_local_bytes + candidate_local_bytes + moe_local_bytes
    onchip_state_bytes = synthetic.total_memory_bytes + moe_state_bytes

    return HarcEventEfficiency(
        context_length=synthetic.fact_count,
        events=events,
        moe_ticks_per_event=moe_ticks_per_event,
        exact_local_bytes_per_event=exact_local_bytes,
        dense_local_bytes_per_event=dense_local_bytes,
        candidate_local_bytes_per_event=candidate_local_bytes,
        moe_local_bytes_per_event=moe_local_bytes,
        total_local_bytes_per_event=total_local_bytes,
        onchip_state_bytes=onchip_state_bytes,
        exact_memory_bytes=synthetic.exact_memory_bytes,
        dense_memory_bytes=synthetic.dense_memory_bytes,
        candidate_memory_bytes=synthetic.candidate_memory_bytes,
        moe_state_bytes=moe_state_bytes,
        exact_query_fraction=query_fraction,
        exact_avg_visited_cells=synthetic.exact_avg_visited_cells,
        overflow_query_rate=synthetic.overflow_query_rate,
        dense_update_cells_per_event=dense_cells_per_event,
        candidate_update_cells_per_event=candidate_cells_per_event,
        candidate_gate_cells_per_event=candidate_gate_cells_per_event,
        moe_sparse_rule_updates_per_event=sparse_rule_updates_per_event,
        moe_dense_equivalent_rule_updates_per_event=dense_equiv_updates_per_event,
        moe_update_reduction=moe.avg_update_reduction,
    )


def compare_to_transformer_kv(
    synthetic: SyntheticLMResult,
    synthetic_config: SyntheticLMConfig,
    moe: MoERolloutResult,
    moe_config: CellularMoEConfig,
    moe_ticks_per_event: int = 4,
    layers: int = 12,
    heads: int = 8,
    head_dim: int = 64,
    kv_bits: int = 16,
) -> EfficiencyComparison:
    """Build a HARC event profile and a Transformer KV-cache reference."""

    harc = estimate_harc_event_efficiency(
        synthetic=synthetic,
        synthetic_config=synthetic_config,
        moe=moe,
        moe_config=moe_config,
        moe_ticks_per_event=moe_ticks_per_event,
    )
    transformer = estimate_transformer_kv(
        context_length=synthetic.fact_count,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        kv_bits=kv_bits,
    )
    local_vs_kv = harc.total_local_bytes_per_event / transformer.kv_read_bytes_per_token
    return EfficiencyComparison(harc=harc, transformer=transformer, local_vs_kv_byte_ratio=local_vs_kv)
