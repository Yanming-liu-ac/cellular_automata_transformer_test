"""Source-subtile repair behind the CA Wiki Cell summary lane."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    CAWikiCellSubtileSweepResult,
    run_ca_wiki_cell_subtile_repair_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(result: CAWikiCellSubtileSweepResult) -> None:
    cfg = result.config
    print("CA Wiki Cell source-subtile repair")
    print(
        f"claims={cfg.claim_count}, sources/claim={cfg.sources_per_claim}, "
        f"queries={cfg.query_events}, updates={cfg.update_events}, "
        f"base_state={format_bytes(cfg.state_bytes)}, "
        f"summary_state={format_bytes(result.summary_state_bytes)}, seed={result.seed}"
    )
    print()
    headers = [
        "policy",
        "scope",
        "tile",
        "probe",
        "recall",
        "recent",
        "q_read",
        "touch/e",
        "stale_src",
        "fresh_tile",
        "consistent",
        "ticks",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 180)
    for point in result.points:
        row = [
            point.policy,
            point.repair_scope,
            f"{point.subtile_size}",
            f"{point.read_sources}",
            fmt_pct(point.recall),
            fmt_pct(point.recent_recall),
            f"{point.cells_read_per_query:0.2f}",
            f"{point.cells_touched_per_event:0.2f}",
            fmt_pct(point.stale_source_rate),
            fmt_pct(point.fresh_subtile_rate),
            fmt_pct(point.consistent_claim_rate),
            f"{point.repair_ticks}",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))
    print()
    print("Interpretation:")
    print("- All rows answer through the claim summary, so recall stays at the answer-path ceiling.")
    print("- claim_error_repair refreshes every source in a claim when a probe finds staleness.")
    print("- subtile repair refreshes only the local source tile touched by the probe.")
    print("- More probes trade query reads for lower source staleness without whole-claim repair.")


def main() -> None:
    print_result(run_ca_wiki_cell_subtile_repair_sweep())


if __name__ == "__main__":
    main()
