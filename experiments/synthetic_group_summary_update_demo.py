"""Group-summary maintenance cost diagnostic for synthetic candidate reduction."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.synthetic_lm import (
    SyntheticLMGroupSummaryUpdateResult,
    run_synthetic_lm_group_summary_update_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_summary(result: SyntheticLMGroupSummaryUpdateResult) -> None:
    print("Synthetic group-summary update cost")
    print(
        f"facts={result.fact_count}, candidate_pool={result.candidate_pool_size}, "
        f"topic_events={result.topic_events}, dense={result.dense_banks}x{result.dense_width}, "
        f"bits={result.bits}, decay={result.decay_interval}, source={result.candidate_score_source}"
    )
    headers = [
        "grp",
        "groups",
        "state_B",
        "imp_rows",
        "imp_grp",
        "recomp",
        "writes",
        "decay",
        "maint",
        "t16_score",
        "t16_total",
        "t16_cut",
        "t32_score",
        "t32_total",
        "t32_cut",
    ]
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 180)
    for point in result.points:
        row = [
            f"{point.group_size}",
            f"{point.total_groups}",
            f"{point.summary_state_bytes:0.1f}",
            f"{point.mean_impacted_candidate_rows:0.2f}",
            f"{point.mean_impacted_groups:0.2f}",
            f"{point.recompute_read_cells_per_topic_event:0.1f}",
            f"{point.summary_write_cells_per_topic_event:0.1f}",
            f"{point.summary_decay_cells_per_topic_event:0.1f}",
            f"{point.maintenance_cells_per_topic_event:0.1f}",
            f"{point.top16_score_cells_per_topic_event:0.1f}",
            f"{point.top16_total_cells_per_topic_event:0.1f}",
            fmt_pct(point.top16_total_reduction_rate),
            f"{point.top32_score_cells_per_topic_event:0.1f}",
            f"{point.top32_total_cells_per_topic_event:0.1f}",
            fmt_pct(point.top32_total_reduction_rate),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- Maint is exact group-summary recompute plus summary writes and decay shifts.")
    print("- T16/T32 totals add maintenance to hierarchical score reads.")
    print("- Cut is total reduction versus scoring all 512 candidates every topic event.")


def main() -> None:
    print_summary(run_synthetic_lm_group_summary_update_sweep())


if __name__ == "__main__":
    main()
