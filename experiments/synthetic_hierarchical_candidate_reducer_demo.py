"""Hierarchical candidate reducer diagnostic for the synthetic HARC-CA LM."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.synthetic_lm import (
    SyntheticLMHierarchicalCandidateReducerResult,
    run_synthetic_lm_hierarchical_candidate_reducer_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_reducer(result: SyntheticLMHierarchicalCandidateReducerResult) -> None:
    print("Synthetic hierarchical candidate reducer + phase/rank content gate")
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
        "scored",
        "top64_hit",
        "rows_hit",
        "retain",
        "grp/tpc",
        "fine/tpc",
        "score/tpc",
        "score_cut",
        "gate_wr/e",
        "exact",
        "wstates",
    ]
    print(" | ".join(f"{h:>11}" for h in headers))
    print("-" * 178)
    for point in result.points:
        row = [
            f"{point.reducer_rows}",
            f"{point.group_size}",
            f"{point.selected_groups}",
            f"{point.candidate_rows_scored}",
            fmt_pct(point.base_topic_hit_rate),
            fmt_pct(point.reduced_topic_hit_rate),
            fmt_pct(point.hit_retention_rate),
            f"{point.group_score_cells_per_topic_event:0.1f}",
            f"{point.fine_score_cells_per_topic_event:0.1f}",
            f"{point.score_cells_per_topic_event:0.1f}",
            fmt_pct(point.score_cell_reduction_rate),
            f"{point.phase_channel_writes_per_event:0.1f}",
            fmt_pct(point.phase_demand_exact_rate),
            f"{point.phase_lut_write_state_count}",
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))

    print()
    print("Interpretation:")
    print("- Group summaries are modeled as local max-score reduction cells.")
    print("- Score_cut is reduction versus scoring all 512 candidates every topic event.")
    print("- Gate_wr/e is exact content-exposure traffic after hierarchical reduction.")


def main() -> None:
    print_reducer(run_synthetic_lm_hierarchical_candidate_reducer_sweep())


if __name__ == "__main__":
    main()
