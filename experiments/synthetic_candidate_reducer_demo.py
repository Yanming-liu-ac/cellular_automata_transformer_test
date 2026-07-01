"""Low-bit candidate reducer plus exact content exposure diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.synthetic_lm import (
    SyntheticLMCandidateReducerResult,
    run_synthetic_lm_candidate_reducer_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_reducer(result: SyntheticLMCandidateReducerResult) -> None:
    print("Synthetic low-bit candidate reducer + phase/rank content gate")
    print(
        f"facts={result.fact_count}, candidate_pool={result.candidate_pool_size}, "
        f"base_top_k={result.base_top_k}, events={result.total_events}, "
        f"topic_events={result.topic_events}, query_events={result.query_events}, "
        f"bits={result.bits}, cost={result.write_cost:0.2f}"
    )
    headers = [
        "rows",
        "score_src",
        "top64_hit",
        "rows_hit",
        "retain",
        "demand",
        "score/tpc",
        "score/evt",
        "gate_wr/t",
        "gate_wr/e",
        "exact",
        "err",
        "lut_B",
        "wstates",
    ]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 194)
    for point in result.points:
        row = [
            f"{point.reducer_rows}",
            point.candidate_score_source,
            fmt_pct(point.base_topic_hit_rate),
            fmt_pct(point.reduced_topic_hit_rate),
            fmt_pct(point.hit_retention_rate),
            fmt_pct(point.mean_demand_fraction),
            f"{point.score_cells_per_topic_event:0.1f}",
            f"{point.score_cells_per_event:0.1f}",
            f"{point.phase_writes_per_token_tick:0.4f}",
            f"{point.phase_channel_writes_per_event:0.1f}",
            fmt_pct(point.phase_demand_exact_rate),
            fmt_pct(point.phase_demand_mean_abs_error),
            f"{point.phase_lut_state_bytes:0.1f}",
            f"{point.phase_lut_write_state_count}",
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))

    print()
    print("Interpretation:")
    print("- The reducer ranks the 512-row static candidate pool with low-bit topic scores.")
    print("- Rows_hit is the topic hit rate after reducing top-64 to top-M content demand.")
    print("- Gate_wr/e is channel writes per mixed event after the 9-byte exact exposure gate.")


def main() -> None:
    print_reducer(run_synthetic_lm_candidate_reducer_sweep())


if __name__ == "__main__":
    main()
