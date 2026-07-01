"""Triggered group-summary reducer diagnostic for the synthetic HARC-CA LM."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.synthetic_lm import (
    SyntheticLMTriggeredGroupSummaryReducerResult,
    run_synthetic_lm_triggered_group_summary_reducer_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_triggered(result: SyntheticLMTriggeredGroupSummaryReducerResult) -> None:
    print("Synthetic triggered group-summary reducer + phase/rank content gate")
    print(
        f"facts={result.fact_count}, candidate_pool={result.candidate_pool_size}, "
        f"base_top_k={result.base_top_k}, events={result.total_events}, "
        f"topic_events={result.topic_events}, query_events={result.query_events}, "
        f"bits={result.bits}, cost={result.write_cost:0.2f}"
    )
    headers = [
        "rows",
        "policy",
        "dirty",
        "age",
        "top64_hit",
        "rows_hit",
        "retain",
        "maint",
        "score",
        "cut",
        "refresh",
        "r_groups",
        "gate_wr/e",
        "exact",
    ]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 190)
    for point in result.points:
        row = [
            f"{point.reducer_rows}",
            point.trigger_policy,
            f"{point.dirty_threshold}",
            f"{point.max_age}",
            fmt_pct(point.base_topic_hit_rate),
            fmt_pct(point.reduced_topic_hit_rate),
            fmt_pct(point.hit_retention_rate),
            f"{point.maintenance_cells_per_topic_event:0.1f}",
            f"{point.score_cells_per_topic_event:0.1f}",
            fmt_pct(point.score_cell_reduction_rate),
            f"{point.refresh_events}",
            f"{point.mean_refreshed_groups:0.1f}",
            f"{point.phase_channel_writes_per_event:0.1f}",
            fmt_pct(point.phase_demand_exact_rate),
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))

    print()
    print("Interpretation:")
    print("- Policy uses only dirty count, top dirty groups, and max summary age.")
    print("- Score includes group scan, fine score, and triggered maintenance.")
    print("- The target is refresh-4 quality with refresh-16-like score work.")


def main() -> None:
    print_triggered(run_synthetic_lm_triggered_group_summary_reducer_sweep())


if __name__ == "__main__":
    main()
