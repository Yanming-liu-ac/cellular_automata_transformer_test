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
class ContextSummaryBudget:
    """CSA/HCA context-summary state and per-event local traffic budget."""

    block_summary_state_bytes: float = 0.0
    csa_directory_state_bytes: float = 0.0
    control_lut_state_bytes: float = 0.0
    sidecar_state_bytes: float = 0.0
    hca_summary_state_bytes: float = 0.0
    hca_summary_read_bytes_per_event: float = 0.0
    hca_summary_update_bytes_per_event: float = 0.0
    control_lut_read_bytes_per_event: float = 0.0
    sidecar_read_bytes_per_event: float = 0.0
    sidecar_update_bytes_per_event: float = 0.0
    csa_block_score_bytes_per_event: float = 0.0
    csa_directory_read_bytes_per_event: float = 0.0
    csa_token_read_bytes_per_event: float = 0.0

    @property
    def state_bytes(self) -> float:
        return (
            self.block_summary_state_bytes
            + self.csa_directory_state_bytes
            + self.control_lut_state_bytes
            + self.sidecar_state_bytes
            + self.hca_summary_state_bytes
        )

    @property
    def local_bytes_per_event(self) -> float:
        return (
            self.hca_summary_read_bytes_per_event
            + self.hca_summary_update_bytes_per_event
            + self.control_lut_read_bytes_per_event
            + self.sidecar_read_bytes_per_event
            + self.sidecar_update_bytes_per_event
            + self.csa_block_score_bytes_per_event
            + self.csa_directory_read_bytes_per_event
            + self.csa_token_read_bytes_per_event
        )


@dataclass(frozen=True)
class HarcEventEfficiency:
    """Per-event proxy metrics for a HARC-CA decode schedule."""

    context_length: int
    events: int
    moe_ticks_per_event: int
    exact_local_bytes_per_event: float
    dense_local_bytes_per_event: float
    candidate_local_bytes_per_event: float
    context_summary_local_bytes_per_event: float
    moe_local_bytes_per_event: float
    total_local_bytes_per_event: float
    onchip_state_bytes: float
    exact_memory_bytes: float
    dense_memory_bytes: float
    candidate_score_memory_bytes: float
    candidate_memory_bytes: float
    context_summary_state_bytes: float
    hca_summary_state_bytes: float
    csa_block_summary_state_bytes: float
    csa_directory_state_bytes: float
    control_lut_state_bytes: float
    sidecar_state_bytes: float
    moe_state_bytes: float
    exact_query_fraction: float
    exact_avg_visited_cells: float
    overflow_query_rate: float
    dense_update_cells_per_event: float
    candidate_update_cells_per_event: float
    candidate_gate_cells_per_event: float
    candidate_score_cells_per_event: float
    candidate_score_update_cells_per_event: float
    hca_summary_read_bytes_per_event: float
    hca_summary_update_bytes_per_event: float
    control_lut_read_bytes_per_event: float
    sidecar_read_bytes_per_event: float
    sidecar_update_bytes_per_event: float
    csa_block_score_bytes_per_event: float
    csa_directory_read_bytes_per_event: float
    csa_token_read_bytes_per_event: float
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
    context_summary: ContextSummaryBudget | None = None,
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
    candidate_score_cells_per_event = synthetic.candidate_score_cells_per_event
    dense_cells_per_event = max(
        0.0,
        synthetic.avg_cells_per_event
        - exact_cells_per_event
        - candidate_cells_per_event
        - candidate_gate_cells_per_event
        - candidate_score_cells_per_event,
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
        + candidate_score_cells_per_event * dense_counter_bytes
    )

    sparse_rule_updates_per_tick = moe_config.length * moe.avg_active_fraction * moe_config.top_k
    sparse_rule_updates_per_event = sparse_rule_updates_per_tick * moe_ticks_per_event
    dense_equiv_updates_per_event = moe_config.length * moe_config.rule_banks * moe_ticks_per_event
    cell_state_bytes = moe_config.channels * moe_config.bits / 8
    moe_local_bytes = sparse_rule_updates_per_event * cell_state_bytes * 4

    moe_state_bytes = moe_config.length * cell_state_bytes
    context_budget = context_summary or ContextSummaryBudget()
    total_local_bytes = (
        exact_local_bytes
        + dense_local_bytes
        + candidate_local_bytes
        + context_budget.local_bytes_per_event
        + moe_local_bytes
    )
    onchip_state_bytes = synthetic.total_memory_bytes + context_budget.state_bytes + moe_state_bytes

    return HarcEventEfficiency(
        context_length=synthetic.fact_count,
        events=events,
        moe_ticks_per_event=moe_ticks_per_event,
        exact_local_bytes_per_event=exact_local_bytes,
        dense_local_bytes_per_event=dense_local_bytes,
        candidate_local_bytes_per_event=candidate_local_bytes,
        context_summary_local_bytes_per_event=context_budget.local_bytes_per_event,
        moe_local_bytes_per_event=moe_local_bytes,
        total_local_bytes_per_event=total_local_bytes,
        onchip_state_bytes=onchip_state_bytes,
        exact_memory_bytes=synthetic.exact_memory_bytes,
        dense_memory_bytes=synthetic.dense_memory_bytes,
        candidate_score_memory_bytes=synthetic.candidate_score_memory_bytes,
        candidate_memory_bytes=synthetic.candidate_memory_bytes,
        context_summary_state_bytes=context_budget.state_bytes,
        hca_summary_state_bytes=context_budget.hca_summary_state_bytes,
        csa_block_summary_state_bytes=context_budget.block_summary_state_bytes,
        csa_directory_state_bytes=context_budget.csa_directory_state_bytes,
        control_lut_state_bytes=context_budget.control_lut_state_bytes,
        sidecar_state_bytes=context_budget.sidecar_state_bytes,
        moe_state_bytes=moe_state_bytes,
        exact_query_fraction=query_fraction,
        exact_avg_visited_cells=synthetic.exact_avg_visited_cells,
        overflow_query_rate=synthetic.overflow_query_rate,
        dense_update_cells_per_event=dense_cells_per_event,
        candidate_update_cells_per_event=candidate_cells_per_event,
        candidate_gate_cells_per_event=candidate_gate_cells_per_event,
        candidate_score_cells_per_event=candidate_score_cells_per_event,
        candidate_score_update_cells_per_event=synthetic.candidate_score_update_cells_per_event,
        hca_summary_read_bytes_per_event=context_budget.hca_summary_read_bytes_per_event,
        hca_summary_update_bytes_per_event=context_budget.hca_summary_update_bytes_per_event,
        control_lut_read_bytes_per_event=context_budget.control_lut_read_bytes_per_event,
        sidecar_read_bytes_per_event=context_budget.sidecar_read_bytes_per_event,
        sidecar_update_bytes_per_event=context_budget.sidecar_update_bytes_per_event,
        csa_block_score_bytes_per_event=context_budget.csa_block_score_bytes_per_event,
        csa_directory_read_bytes_per_event=context_budget.csa_directory_read_bytes_per_event,
        csa_token_read_bytes_per_event=context_budget.csa_token_read_bytes_per_event,
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
    context_summary: ContextSummaryBudget | None = None,
) -> EfficiencyComparison:
    """Build a HARC event profile and a Transformer KV-cache reference."""

    harc = estimate_harc_event_efficiency(
        synthetic=synthetic,
        synthetic_config=synthetic_config,
        moe=moe,
        moe_config=moe_config,
        moe_ticks_per_event=moe_ticks_per_event,
        context_summary=context_summary,
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


def wide_csa_hca_context_budget() -> ContextSummaryBudget:
    """Earlier wide CSA/HCA context-summary budget for comparison.

    This uses 64-token blocks, 1024 block summaries, 4-bit counters,
    ``summary_width=256``, a 4-bit counter plus 8-bit lazy epoch HCA summary,
    threshold-routed CSA/HCA reads, and 16-bit token-cell reads for selected
    context cells.
    """

    block_summary_state = 1024 * 4 * 256 * 4 / 8
    hca_state = 4 * 2048 * (4 + 8) / 8
    hca_read = 4 * (4 + 8) / 8
    hca_update = 4 * (4 + 8) / 8 * 2
    csa_block_score = 300.0
    csa_token_read = 165.0 * 16 / 8
    return ContextSummaryBudget(
        block_summary_state_bytes=block_summary_state,
        hca_summary_state_bytes=hca_state,
        hca_summary_read_bytes_per_event=hca_read,
        hca_summary_update_bytes_per_event=hca_update,
        csa_block_score_bytes_per_event=csa_block_score,
        csa_token_read_bytes_per_event=csa_token_read,
    )


def compact_csa_hca_context_budget() -> ContextSummaryBudget:
    """Compact block-only CSA/HCA budget for comparison.

    This uses the block-state sweep's measured compact setting: 128-token
    blocks, 512 block summaries, 4-bit counters, ``summary_width=256``, the
    8-bit lazy-epoch HCA summary, and threshold-routed CSA/HCA reads.
    """

    block_summary_state = 512 * 4 * 256 * 4 / 8
    hca_state = 4 * 2048 * (4 + 8) / 8
    hca_read = 4 * (4 + 8) / 8
    hca_update = 4 * (4 + 8) / 8 * 2
    csa_block_score = 150.0
    csa_token_read = 330.9 * 16 / 8
    return ContextSummaryBudget(
        block_summary_state_bytes=block_summary_state,
        hca_summary_state_bytes=hca_state,
        hca_summary_read_bytes_per_event=hca_read,
        hca_summary_update_bytes_per_event=hca_update,
        csa_block_score_bytes_per_event=csa_block_score,
        csa_token_read_bytes_per_event=csa_token_read,
    )


def rare_directory_csa_hca_context_budget() -> ContextSummaryBudget:
    """Current rare-directory CSA/HCA context-summary budget.

    This uses the measured low-state point from ``run_csa_hca_rare_directory_sweep``:
    128-token blocks, ``summary_width=128``, six exact rare-token directory
    blocks per token, an 8-bit lazy-epoch HCA summary, and a threshold-15
    CSA/HCA route gate.
    """

    block_summary_state = 512 * 4 * 128 * 4 / 8
    directory_entry_bytes = (16 + 9 + 1) / 8
    directory_state = 9694 * directory_entry_bytes
    hca_state = 4 * 2048 * (4 + 8) / 8
    hca_read = 4 * (4 + 8) / 8
    hca_update = 4 * (4 + 8) / 8 * 2
    csa_block_score = 150.0
    csa_directory_read = 0.48
    csa_token_read = 331.6 * 16 / 8
    return ContextSummaryBudget(
        block_summary_state_bytes=block_summary_state,
        csa_directory_state_bytes=directory_state,
        hca_summary_state_bytes=hca_state,
        hca_summary_read_bytes_per_event=hca_read,
        hca_summary_update_bytes_per_event=hca_update,
        csa_block_score_bytes_per_event=csa_block_score,
        csa_directory_read_bytes_per_event=csa_directory_read,
        csa_token_read_bytes_per_event=csa_token_read,
    )


def joint_control_csa_hca_context_budget() -> ContextSummaryBudget:
    """Current joint probe/fanout CSA/HCA context-summary budget.

    This keeps the rare128 storage geometry but adds the measured low-bit control
    state from the joint probe/fanout sweep: a 42B fanout LUT, a 40B HCA
    confidence probe LUT, and about 2.25KB of per-row spread metadata. The
    threshold-15 ``confidence_probe`` point keeps rare-directory reads near the
    old average path while making the exact-memory guard and fanout trainable.
    """

    block_summary_state = 512 * 4 * 128 * 4 / 8
    directory_entry_bytes = (16 + 9 + 1) / 8
    directory_state = 9694 * directory_entry_bytes
    fanout_lut_state = 42.0
    probe_lut_state = 40.0
    spread_metadata_state = 2252.0
    hca_state = 4 * 2048 * (4 + 8) / 8
    hca_read = 4 * (4 + 8) / 8
    hca_update = 4 * (4 + 8) / 8 * 2
    csa_block_score = 150.0
    control_lut_read = 0.17
    csa_directory_read = 0.50
    csa_token_read = 331.6 * 16 / 8
    return ContextSummaryBudget(
        block_summary_state_bytes=block_summary_state,
        csa_directory_state_bytes=directory_state,
        control_lut_state_bytes=fanout_lut_state + probe_lut_state + spread_metadata_state,
        hca_summary_state_bytes=hca_state,
        hca_summary_read_bytes_per_event=hca_read,
        hca_summary_update_bytes_per_event=hca_update,
        control_lut_read_bytes_per_event=control_lut_read,
        csa_block_score_bytes_per_event=csa_block_score,
        csa_directory_read_bytes_per_event=csa_directory_read,
        csa_token_read_bytes_per_event=csa_token_read,
    )


def retiring_sidecar_csa_hca_context_budget() -> ContextSummaryBudget:
    """Joint CSA/HCA budget with the conservative online counting sidecar.

    This keeps ``joint128`` and adds the measured ``count1_retire15`` counting
    Bloom sidecar: about 44-45KB of local state, 3 presence-bit reads per event,
    and the worst measured update pressure from the streaming stress cases.
    """

    block_summary_state = 512 * 4 * 128 * 4 / 8
    directory_entry_bytes = (16 + 9 + 1) / 8
    directory_state = 9694 * directory_entry_bytes
    fanout_lut_state = 42.0
    probe_lut_state = 40.0
    spread_metadata_state = 2252.0
    hca_state = 4 * 2048 * (4 + 8) / 8
    hca_read = 4 * (4 + 8) / 8
    hca_update = 4 * (4 + 8) / 8 * 2
    csa_block_score = 150.0
    control_lut_read = 0.17
    csa_directory_read = 0.50
    csa_token_read = 331.6 * 16 / 8
    sidecar_state = 44.9 * 1024
    sidecar_read = 3 / 8
    sidecar_update = 0.278
    return ContextSummaryBudget(
        block_summary_state_bytes=block_summary_state,
        csa_directory_state_bytes=directory_state,
        control_lut_state_bytes=fanout_lut_state + probe_lut_state + spread_metadata_state,
        sidecar_state_bytes=sidecar_state,
        hca_summary_state_bytes=hca_state,
        hca_summary_read_bytes_per_event=hca_read,
        hca_summary_update_bytes_per_event=hca_update,
        control_lut_read_bytes_per_event=control_lut_read,
        sidecar_read_bytes_per_event=sidecar_read,
        sidecar_update_bytes_per_event=sidecar_update,
        csa_block_score_bytes_per_event=csa_block_score,
        csa_directory_read_bytes_per_event=csa_directory_read,
        csa_token_read_bytes_per_event=csa_token_read,
    )


def compressed_retiring_sidecar_csa_hca_context_budget() -> ContextSummaryBudget:
    """Joint CSA/HCA budget with the compressed 2-bit retirement sidecar."""

    block_summary_state = 512 * 4 * 128 * 4 / 8
    directory_entry_bytes = (16 + 9 + 1) / 8
    directory_state = 9694 * directory_entry_bytes
    fanout_lut_state = 42.0
    probe_lut_state = 40.0
    spread_metadata_state = 2252.0
    hca_state = 4 * 2048 * (4 + 8) / 8
    hca_read = 4 * (4 + 8) / 8
    hca_update = 4 * (4 + 8) / 8 * 2
    csa_block_score = 150.0
    control_lut_read = 0.17
    csa_directory_read = 0.50
    csa_token_read = 331.6 * 16 / 8
    sidecar_state = 26.9 * 1024
    sidecar_read = 3 / 8
    sidecar_update = 0.167
    return ContextSummaryBudget(
        block_summary_state_bytes=block_summary_state,
        csa_directory_state_bytes=directory_state,
        control_lut_state_bytes=fanout_lut_state + probe_lut_state + spread_metadata_state,
        sidecar_state_bytes=sidecar_state,
        hca_summary_state_bytes=hca_state,
        hca_summary_read_bytes_per_event=hca_read,
        hca_summary_update_bytes_per_event=hca_update,
        control_lut_read_bytes_per_event=control_lut_read,
        sidecar_read_bytes_per_event=sidecar_read,
        sidecar_update_bytes_per_event=sidecar_update,
        csa_block_score_bytes_per_event=csa_block_score,
        csa_directory_read_bytes_per_event=csa_directory_read,
        csa_token_read_bytes_per_event=csa_token_read,
    )


def current_csa_hca_context_budget() -> ContextSummaryBudget:
    """Current recommended CSA/HCA context-summary budget."""

    return compressed_retiring_sidecar_csa_hca_context_budget()
