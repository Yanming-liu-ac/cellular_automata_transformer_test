"""Lazy group-summary reducer diagnostic for the synthetic HARC-CA LM."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.synthetic_lm import (
    SyntheticLMLazyGroupSummaryReducerResult,
    run_synthetic_lm_lazy_group_summary_reducer_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_lazy(result: SyntheticLMLazyGroupSummaryReducerResult) -> None:
    print("Synthetic lazy group-summary reducer + phase/rank content gate")
    print(
        f"facts={result.fact_count}, candidate_pool={result.candidate_pool_size}, "
        f"base_top_k={result.base_top_k}, events={result.total_events}, "
        f"topic_events={result.topic_events}, query_events={result.query_events}, "
        f"bits={result.bits}, cost={result.write_cost:0.2f}"
    )
    headers = [
        "rows",
        "grp",
        "sel_g",
        "refresh",
        "top64_hit",
        "rows_hit",
        "retain",
        "scan",
        "fine",
        "maint",
        "score",
        "cut",
        "dirty",
        "gate_wr/e",
        "exact",
    ]
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 180)
    for point in result.points:
        row = [
            f"{point.reducer_rows}",
            f"{point.group_size}",
            f"{point.selected_groups}",
            f"{point.refresh_interval}",
            fmt_pct(point.base_topic_hit_rate),
            fmt_pct(point.reduced_topic_hit_rate),
            fmt_pct(point.hit_retention_rate),
            f"{point.group_scan_cells_per_topic_event:0.1f}",
            f"{point.fine_score_cells_per_topic_event:0.1f}",
            f"{point.maintenance_cells_per_topic_event:0.1f}",
            f"{point.score_cells_per_topic_event:0.1f}",
            fmt_pct(point.score_cell_reduction_rate),
            f"{point.mean_dirty_groups_per_refresh:0.1f}",
            f"{point.phase_channel_writes_per_event:0.1f}",
            fmt_pct(point.phase_demand_exact_rate),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- Refresh is topic steps between dirty-summary recomputes.")
    print("- Score includes group scan, fine score, and lazy maintenance.")
    print("- Dirty is the average dirty groups recomputed on refresh events.")


def main() -> None:
    print_lazy(run_synthetic_lm_lazy_group_summary_reducer_sweep())


if __name__ == "__main__":
    main()
