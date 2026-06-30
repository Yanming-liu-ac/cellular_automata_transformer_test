"""Tile-level floorplan proxy for the current HARC-CA prototype."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.cellular_moe import CellularMoE, CellularMoEConfig
from cellular_transformer.chip_model import ChipConfig, TileConfig, profile_chip
from cellular_transformer.efficiency import compare_to_transformer_kv, current_csa_hca_context_budget
from cellular_transformer.hardware import format_bytes
from cellular_transformer.synthetic_lm import DualPathSyntheticLM, SyntheticLMConfig


def fmt_rate(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:0.2f}B/s"
    if value >= 1_000_000:
        return f"{value / 1_000_000:0.2f}M/s"
    if value >= 1_000:
        return f"{value / 1_000:0.2f}K/s"
    return f"{value:0.2f}/s"


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

    comparison = compare_to_transformer_kv(
        synthetic=synthetic,
        synthetic_config=synthetic_config,
        moe=moe_result,
        moe_config=moe_config,
        moe_ticks_per_event=4,
        context_summary=current_csa_hca_context_budget(),
    )

    tile = TileConfig(cells_per_tile=64, local_sram_bytes=16 * 1024, local_bytes_per_cycle=32)
    print("HARC-CA tile/floorplan proxy")
    print("tile: 64 cells, 16KB local SRAM, 32 local bytes/cycle")
    print("profile: rare-directory CSA/HCA-aware context summaries")
    print("target: 1M synthetic decode events/s")
    print()

    headers = [
        "tiles",
        "cells",
        "SRAM",
        "state",
        "state_util",
        "state_tiles",
        "local/event",
        "bw_util",
        "max_events",
    ]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 126)
    for tiles in (32, 64, 128, 256, 512):
        profile = profile_chip(
            comparison,
            tile=tile,
            chip=ChipConfig(tiles=tiles, frequency_mhz=1000.0, target_events_per_second=1_000_000.0),
        )
        row = [
            f"{profile.tiles}",
            f"{profile.cells}",
            format_bytes(profile.total_local_sram_bytes),
            format_bytes(profile.state_bytes),
            f"{100.0 * profile.state_utilization:0.2f}%",
            f"{profile.state_tiles_required}",
            format_bytes(profile.local_bytes_per_event),
            f"{100.0 * profile.bandwidth_utilization:0.4f}%",
            fmt_rate(profile.max_events_per_second_proxy),
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))

    print()
    print("Reference:")
    print(f"  HARC local bytes/event: {format_bytes(comparison.harc.total_local_bytes_per_event)}")
    print(f"  Transformer KV read/token reference: {format_bytes(comparison.transformer.kv_read_bytes_per_token)}")
    print()
    print("Interpretation:")
    print("- This is a floorplan proxy, not area/timing closure.")
    print("- SRAM and bandwidth are local on-chip budgets.")
    print("- rare128 tests whether a small exact directory can buy back SRAM headroom.")
    print("- The table helps track whether future learned rules break the local budget.")


if __name__ == "__main__":
    main()
