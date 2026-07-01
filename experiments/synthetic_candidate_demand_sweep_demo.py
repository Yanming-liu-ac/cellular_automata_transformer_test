"""Candidate-output demand sparsity sweep for the synthetic HARC-CA LM."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.synthetic_lm import (
    SyntheticLMCandidateDemandSweepResult,
    run_synthetic_lm_candidate_demand_sparsity_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_sweep(result: SyntheticLMCandidateDemandSweepResult) -> None:
    base_lut_bytes = result.points[0].lut_state_bytes if result.points else 0.0
    phase_lut_bytes = result.points[0].phase_lut_state_bytes if result.points else 0.0
    print("Synthetic candidate-output demand sparsity sweep")
    print(
        f"facts={result.fact_count}, events={result.total_events}, "
        f"topic_events={result.topic_events}, query_events={result.query_events}, "
        f"bits={result.bits}, cost={result.write_cost:0.2f}, "
        f"base_lut={base_lut_bytes:0.1f}B, phase_lut={phase_lut_bytes:0.1f}B"
    )
    headers = [
        "cand_rows",
        "rows",
        "demand",
        "fixed_wr",
        "fixed_exact",
        "dm1_wr",
        "dm1_exact",
        "learn_wr",
        "learn_exact",
        "learn_err",
        "phase_wr",
        "phase_exact",
        "phase_err",
        "lut_writes",
        "phase_wstates",
    ]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 208)
    for point in result.points:
        row = [
            f"{point.candidate_rows}",
            f"{point.content_rows}",
            fmt_pct(point.mean_demand_fraction),
            f"{point.fixed_refresh_writes_per_token_tick:0.4f}",
            fmt_pct(point.fixed_refresh_demand_exact_rate),
            f"{point.demand_mismatch_writes_per_token_tick:0.4f}",
            fmt_pct(point.demand_mismatch_demand_exact_rate),
            f"{point.learned_writes_per_token_tick:0.4f}",
            fmt_pct(point.learned_demand_exact_rate),
            fmt_pct(point.learned_demand_mean_abs_error),
            f"{point.phase_writes_per_token_tick:0.4f}",
            fmt_pct(point.phase_demand_exact_rate),
            fmt_pct(point.phase_demand_mean_abs_error),
            f"{point.lut_write_state_count}",
            f"{point.phase_lut_write_state_count}",
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))

    print()
    print("Interpretation:")
    print("- Candidate rows approximate how many output rows assert content demand.")
    print("- Low candidate rows model output-side pruning before content exposure.")
    print("- Phase rows use exact/candidate/rank features and an exactness-oriented objective.")
    print("- The target is high demanded exactness with writes far below fixed refresh16.")


def main() -> None:
    print_sweep(run_synthetic_lm_candidate_demand_sparsity_sweep())


if __name__ == "__main__":
    main()
