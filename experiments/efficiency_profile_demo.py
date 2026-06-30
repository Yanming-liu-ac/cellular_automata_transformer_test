"""Unified hardware proxy profile for the current HARC-CA prototype."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.cellular_moe import CellularMoE, CellularMoEConfig
from cellular_transformer.efficiency import compare_to_transformer_kv
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
    headers = [
        "moe_ticks",
        "exact/event",
        "dense/event",
        "cand/event",
        "moe/event",
        "total/event",
        "KV/token",
        "KV/local",
        "state",
    ]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 118)
    for moe_ticks in (1, 2, 4, 8):
        comparison = compare_to_transformer_kv(
            synthetic=synthetic,
            synthetic_config=synthetic_config,
            moe=moe_result,
            moe_config=moe_config,
            moe_ticks_per_event=moe_ticks,
        )
        harc = comparison.harc
        row = [
            f"{moe_ticks}",
            format_bytes(harc.exact_local_bytes_per_event),
            format_bytes(harc.dense_local_bytes_per_event),
            format_bytes(harc.candidate_local_bytes_per_event),
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
    )
    harc = comparison.harc
    print("Detailed current profile:")
    print(f"  exact_query_fraction={harc.exact_query_fraction:0.3f}")
    print(f"  exact_avg_visited_cells={harc.exact_avg_visited_cells:0.1f}")
    print(f"  overflow_query_rate={harc.overflow_query_rate:0.3f}")
    print(f"  dense_update_cells_per_event={harc.dense_update_cells_per_event:0.1f}")
    print(f"  candidate_update_cells_per_event={harc.candidate_update_cells_per_event:0.1f}")
    print(f"  candidate_gate_cells_per_event={harc.candidate_gate_cells_per_event:0.1f}")
    print(f"  candidate_score_cells_per_event={harc.candidate_score_cells_per_event:0.1f}")
    print(f"  candidate_score_update_cells_per_event={harc.candidate_score_update_cells_per_event:0.1f}")
    print(f"  moe_sparse_rule_updates/event={harc.moe_sparse_rule_updates_per_event:0.1f}")
    print(f"  moe_dense_equiv_rule_updates/event={harc.moe_dense_equivalent_rule_updates_per_event:0.1f}")
    print(f"  moe_update_reduction={harc.moe_update_reduction:0.1f}x")
    print()
    print("Interpretation:")
    print("- HARC numbers are local on-chip byte movement proxies.")
    print("- Transformer KV is KV-cache read volume, not full model traffic.")
    print("- The ratio is a design target indicator, not a measured energy claim.")


if __name__ == "__main__":
    main()
