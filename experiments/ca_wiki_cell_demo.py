"""Cell-level CA wiki-memory update diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import CAWikiCellSweepResult, run_ca_wiki_cell_sweep


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(result: CAWikiCellSweepResult) -> None:
    cfg = result.config
    print("CA Wiki Cell v0: local mutable claim storage")
    print(
        f"claims={cfg.claim_count}, sources/claim={cfg.sources_per_claim}, "
        f"queries={cfg.query_events}, updates={cfg.update_events}, "
        f"sparse_reads={cfg.read_sources}, recent_query={fmt_pct(cfg.recent_query_rate)}, "
        f"state={format_bytes(cfg.state_bytes)}, seed={result.seed}"
    )
    print()
    headers = [
        "policy",
        "read",
        "scan",
        "u_tick",
        "e_tick",
        "recall",
        "recent",
        "stale_a",
        "disagr",
        "err_trig",
        "consistent",
        "stale_src",
        "q_read",
        "repair_r/e",
        "write/u",
        "touch/e",
        "ticks",
        "p_writes",
        "c_writes",
    ]
    print(" | ".join(f"{header:>11}" for header in headers))
    print("-" * 245)
    for point in result.points:
        row = [
            point.policy,
            f"{point.read_sources}",
            "yes" if point.scan_all_sources else "no",
            f"{point.update_repair_ticks}",
            f"{point.error_repair_ticks}",
            fmt_pct(point.recall),
            fmt_pct(point.recent_recall),
            fmt_pct(point.stale_answer_rate),
            fmt_pct(point.disagreement_rate),
            fmt_pct(point.error_book_trigger_rate),
            fmt_pct(point.consistent_claim_rate),
            fmt_pct(point.stale_source_rate),
            f"{point.cells_read_per_query:0.2f}",
            f"{point.repair_cells_read_per_event:0.2f}",
            f"{point.cells_written_per_update:0.2f}",
            f"{point.cells_touched_per_event:0.2f}",
            f"{point.repair_ticks}",
            f"{point.page_writes}",
            f"{point.counter_writes}",
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))
    print()
    print("Interpretation:")
    print("- sample_no_repair is the cheap sparse-read baseline with no local update propagation.")
    print("- flat_scan is the upper-bound RAG-like read path over every source page for a claim.")
    print("- tile_update_ca spends one tile-local pulse after writes and keeps sparse query reads.")
    print("- error_book_ca is lazy: failed or disagreeing sparse queries trigger local repair.")
    print("- hybrid_error_book_ca mixes a narrow write pulse with query-triggered repair.")


def main() -> None:
    print_result(run_ca_wiki_cell_sweep())


if __name__ == "__main__":
    main()
