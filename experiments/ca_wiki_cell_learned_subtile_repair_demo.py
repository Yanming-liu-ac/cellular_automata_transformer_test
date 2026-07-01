"""Learned source-subtile provenance controller for CA Wiki Cell."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    CAWikiCellLearnedSubtileResult,
    run_ca_wiki_cell_learned_subtile_repair_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(result: CAWikiCellLearnedSubtileResult) -> None:
    cfg = result.config
    print("CA Wiki Cell learned source-subtile repair")
    print(
        f"claims={cfg.claim_count}, sources/claim={cfg.sources_per_claim}, "
        f"updates={cfg.update_events}, candidates={result.candidate_count}, "
        f"lut={format_bytes(result.lut_state_bytes)}"
    )
    print()
    print("Learned importance LUT")
    headers = [
        "importance",
        "stale_max",
        "policy",
        "scope",
        "tile",
        "probe",
        "u_tick",
        "u_per",
        "e_tick",
        "cost",
    ]
    print(" | ".join(f"{header:>12}" for header in headers))
    print("-" * 132)
    for entry in result.entries:
        row = [
            entry.importance,
            fmt_pct(entry.max_stale_source_rate),
            entry.chosen_policy,
            entry.chosen_repair_scope,
            f"{entry.chosen_subtile_size}",
            f"{entry.chosen_read_sources}",
            f"{entry.chosen_update_repair_ticks}",
            f"{entry.chosen_update_repair_period}",
            f"{entry.chosen_error_repair_ticks}",
            f"{entry.training_cost:0.2f}",
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))
    print()
    print("Evaluation")
    headers = [
        "importance",
        "seed",
        "policy",
        "stale",
        "fresh_tile",
        "consistent",
        "q_read",
        "touch/e",
        "pass",
    ]
    print(" | ".join(f"{header:>12}" for header in headers))
    print("-" * 120)
    for point in result.points:
        row = [
            point.importance,
            f"{point.eval_seed}",
            point.chosen_policy,
            fmt_pct(point.stale_source_rate),
            fmt_pct(point.fresh_subtile_rate),
            fmt_pct(point.consistent_claim_rate),
            f"{point.cells_read_per_query:0.2f}",
            f"{point.cells_touched_per_event:0.2f}",
            "yes" if point.target_met else "no",
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))
    failures = sum(1 for point in result.points if not point.target_met)
    print()
    print("Interpretation:")
    print("- The LUT maps page-importance mode to a provenance repair policy id.")
    print("- Loose mode favors low touch; strict mode chooses whole-claim repair.")
    print("- Answer recall stays at the summary-lane ceiling; target checks source freshness.")
    print(f"- Target failures: {failures}/{len(result.points)} evaluation rows.")


def main() -> None:
    print_result(run_ca_wiki_cell_learned_subtile_repair_sweep())


if __name__ == "__main__":
    main()
