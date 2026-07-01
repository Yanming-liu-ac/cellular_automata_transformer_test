"""Learned repair-schedule LUT for CA Wiki Cell v0."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    CAWikiCellLearnedRepairResult,
    run_ca_wiki_cell_learned_repair_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(result: CAWikiCellLearnedRepairResult) -> None:
    print("CA Wiki Cell learned repair-schedule LUT")
    print(
        f"claims={result.claim_count}, queries={result.query_events}, "
        f"sources={result.source_options}, updates={result.update_event_options}, "
        f"target=recall>={fmt_pct(result.target_recall)}, "
        f"recent>={fmt_pct(result.target_recent_recall)}, "
        f"stale<={fmt_pct(result.max_stale_source_rate)}, "
        f"candidates={result.candidate_count}, lut={format_bytes(result.lut_state_bytes)}"
    )
    print()
    print("Learned LUT entries")
    headers = [
        "src",
        "updates",
        "policy",
        "read",
        "rad",
        "u_tick",
        "u_per",
        "e_tick",
        "cost",
    ]
    print(" | ".join(f"{header:>11}" for header in headers))
    print("-" * 113)
    for entry in result.entries:
        row = [
            f"{entry.sources_per_claim}",
            f"{entry.update_events}",
            entry.chosen_policy,
            f"{entry.chosen_read_sources}",
            f"{entry.chosen_local_radius}",
            f"{entry.chosen_update_repair_ticks}",
            f"{entry.chosen_update_repair_period}",
            f"{entry.chosen_error_repair_ticks}",
            f"{entry.training_cost:0.2f}",
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))
    print()
    print("Evaluation")
    headers = [
        "seed",
        "src",
        "updates",
        "policy",
        "recall",
        "recent",
        "stale",
        "q_read",
        "touch/e",
        "pass",
    ]
    print(" | ".join(f"{header:>11}" for header in headers))
    print("-" * 128)
    for point in result.points:
        row = [
            f"{point.eval_seed}",
            f"{point.sources_per_claim}",
            f"{point.update_events}",
            point.chosen_policy,
            fmt_pct(point.recall),
            fmt_pct(point.recent_recall),
            fmt_pct(point.stale_source_rate),
            f"{point.cells_read_per_query:0.2f}",
            f"{point.cells_touched_per_event:0.2f}",
            "yes" if point.target_met else "no",
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))
    failures = sum(1 for point in result.points if not point.target_met)
    print()
    print("Interpretation:")
    print("- The LUT chooses a repair schedule from low-bit local policy fields.")
    print("- For 8-source claims it learns periodic repair instead of repairing every update.")
    print("- Failures mark fan-in/update regimes where fixed-radius repair is not enough.")
    print(f"- Target failures: {failures}/{len(result.points)} evaluation rows.")


def main() -> None:
    print_result(run_ca_wiki_cell_learned_repair_sweep())


if __name__ == "__main__":
    main()
