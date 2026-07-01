"""Unified hardware proxy profile for the current HARC-CA prototype."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.cellular_moe import CellularMoE, CellularMoEConfig
from cellular_transformer.efficiency import (
    compact_csa_hca_context_budget,
    compressed_retiring_sidecar_csa_hca_context_budget,
    compare_to_transformer_kv,
    joint_control_csa_hca_context_budget,
    rare_directory_csa_hca_context_budget,
    retiring_sidecar_csa_hca_context_budget,
    robust_retiring_sidecar_csa_hca_context_budget,
    wide_csa_hca_context_budget,
)
from cellular_transformer.hardware import format_bytes
from cellular_transformer.synthetic_lm import DualPathSyntheticLM, SyntheticLMConfig


def main() -> None:
    synthetic_config = SyntheticLMConfig(
        dense_width=2048,
        candidate_strategy="online_cache",
        candidate_admission_threshold=1,
    )
    synthetic = DualPathSyntheticLM(synthetic_config, seed=31).run()

    moe_config = CellularMoEConfig(
        length=2048,
        channels=16,
        bits=4,
        rule_banks=6,
        top_k=1,
        active_budget_fraction=0.20,
        balance_rate=0.18,
    )
    moe = CellularMoE(moe_config, seed=13)
    moe.randomize_sparse(density=0.06)
    moe.inject_patch(256, 96)
    moe.inject_patch(1400, 64, value=10)
    moe_result = moe.rollout(128)

    print("Unified HARC-CA hardware proxy profile")
    print("context=16k facts, synthetic mixed decode events, tiny Transformer KV reference")
    print()
    wide_context_budget = wide_csa_hca_context_budget()
    compact_context_budget = compact_csa_hca_context_budget()
    rare_context_budget = rare_directory_csa_hca_context_budget()
    joint_context_budget = joint_control_csa_hca_context_budget()
    retiring_context_budget = retiring_sidecar_csa_hca_context_budget()
    compressed_context_budget = compressed_retiring_sidecar_csa_hca_context_budget()
    context_budget = robust_retiring_sidecar_csa_hca_context_budget()
    headers = [
        "profile",
        "moe_ticks",
        "exact/event",
        "dense/event",
        "cand/event",
        "ctx/event",
        "moe/event",
        "total/event",
        "KV/token",
        "KV/local",
        "state",
    ]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 148)
    rows = []
    for label, maybe_context in (
        ("legacy", None),
        ("wide64", wide_context_budget),
        ("compact128", compact_context_budget),
        ("rare128", rare_context_budget),
        ("joint128", joint_context_budget),
        ("retire128c4", retiring_context_budget),
        ("retire128c2", compressed_context_budget),
        ("retire128c3", context_budget),
    ):
        for moe_ticks in (1, 2, 4, 8):
            rows.append((label, moe_ticks, maybe_context))
    for label, moe_ticks, maybe_context in rows:
        comparison = compare_to_transformer_kv(
            synthetic=synthetic,
            synthetic_config=synthetic_config,
            moe=moe_result,
            moe_config=moe_config,
            moe_ticks_per_event=moe_ticks,
            context_summary=maybe_context,
        )
        harc = comparison.harc
        row = [
            label,
            f"{moe_ticks}",
            format_bytes(harc.exact_local_bytes_per_event),
            format_bytes(harc.dense_local_bytes_per_event),
            format_bytes(harc.candidate_local_bytes_per_event),
            format_bytes(harc.context_summary_local_bytes_per_event),
            format_bytes(harc.moe_local_bytes_per_event),
            format_bytes(harc.total_local_bytes_per_event),
            format_bytes(comparison.transformer.kv_read_bytes_per_token),
            f"{comparison.kv_vs_local_byte_ratio:0.1f}x",
            format_bytes(harc.onchip_state_bytes),
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))

    print()
    comparison = compare_to_transformer_kv(
        synthetic=synthetic,
        synthetic_config=synthetic_config,
        moe=moe_result,
        moe_config=moe_config,
        moe_ticks_per_event=4,
        context_summary=context_budget,
    )
    harc = comparison.harc
    print("Detailed robust retirement-sidecar CSA/HCA-aware profile:")
    print(f"  exact_query_fraction={harc.exact_query_fraction:0.3f}")
    print(f"  exact_avg_visited_cells={harc.exact_avg_visited_cells:0.1f}")
    print(f"  overflow_query_rate={harc.overflow_query_rate:0.3f}")
    print(f"  dense_update_cells_per_event={harc.dense_update_cells_per_event:0.1f}")
    print(f"  candidate_update_cells_per_event={harc.candidate_update_cells_per_event:0.1f}")
    print(f"  candidate_gate_cells_per_event={harc.candidate_gate_cells_per_event:0.1f}")
    print(f"  candidate_score_cells_per_event={harc.candidate_score_cells_per_event:0.1f}")
    print(f"  candidate_score_update_cells_per_event={harc.candidate_score_update_cells_per_event:0.1f}")
    print(f"  hca_summary_state={format_bytes(harc.hca_summary_state_bytes)}")
    print(f"  csa_block_summary_state={format_bytes(harc.csa_block_summary_state_bytes)}")
    print(f"  csa_directory_state={format_bytes(harc.csa_directory_state_bytes)}")
    print(f"  control_lut_state={format_bytes(harc.control_lut_state_bytes)}")
    print(f"  sidecar_state={format_bytes(harc.sidecar_state_bytes)}")
    print(f"  hca_summary_read_bytes/event={format_bytes(harc.hca_summary_read_bytes_per_event)}")
    print(f"  hca_summary_update_bytes/event={format_bytes(harc.hca_summary_update_bytes_per_event)}")
    print(f"  control_lut_read_bytes/event={format_bytes(harc.control_lut_read_bytes_per_event)}")
    print(f"  sidecar_read_bytes/event={format_bytes(harc.sidecar_read_bytes_per_event)}")
    print(f"  sidecar_update_bytes/event={format_bytes(harc.sidecar_update_bytes_per_event)}")
    print(f"  csa_block_score_bytes/event={format_bytes(harc.csa_block_score_bytes_per_event)}")
    print(f"  csa_directory_read_bytes/event={format_bytes(harc.csa_directory_read_bytes_per_event)}")
    print(f"  csa_token_read_bytes/event={format_bytes(harc.csa_token_read_bytes_per_event)}")
    print(f"  moe_sparse_rule_updates/event={harc.moe_sparse_rule_updates_per_event:0.1f}")
    print(f"  moe_dense_equiv_rule_updates/event={harc.moe_dense_equivalent_rule_updates_per_event:0.1f}")
    print(f"  moe_update_reduction={harc.moe_update_reduction:0.1f}x")
    print()
    print("Interpretation:")
    print("- HARC numbers are local on-chip byte movement proxies.")
    print("- wide64 is the earlier 512KB block-summary baseline.")
    print("- compact128 uses 256KB block summaries plus a 12KB lazy-epoch HCA summary.")
    print("- rare128 uses 128KB block summaries plus a small exact rare-token directory.")
    print("- joint128 adds learned probe/fanout control metadata to rare128.")
    print("- retire128c4 adds the original 4-bit counting Bloom sidecar.")
    print("- retire128c2 is the normal-stress compressed 2-bit sidecar.")
    print("- retire128c3 is the adversarial-collision robust sidecar used by the current budget.")
    print("- Transformer KV is KV-cache read volume, not full model traffic.")
    print("- The ratio is a design target indicator, not a measured energy claim.")


if __name__ == "__main__":
    main()
