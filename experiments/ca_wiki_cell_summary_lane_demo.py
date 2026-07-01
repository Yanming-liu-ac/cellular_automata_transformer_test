"""Claim-summary lane for high fan-in CA Wiki Cell."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    CAWikiCellConfig,
    CAWikiCellPolicy,
    CAWikiCellSummarySweepResult,
    run_ca_wiki_cell_summary_sweep,
    run_ca_wiki_cell_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(summary: CAWikiCellSummarySweepResult) -> None:
    cfg = summary.config
    source_baseline = run_ca_wiki_cell_sweep(
        cfg,
        (
            CAWikiCellPolicy(
                name="flat_scan",
                read_sources=cfg.sources_per_claim,
                scan_all_sources=True,
            ),
            CAWikiCellPolicy(
                name="strict_source_repair",
                read_sources=cfg.read_sources,
                update_repair_ticks=2,
                local_radius=min(4, cfg.sources_per_claim - 1),
            ),
        ),
        seed=summary.seed,
    )
    print("CA Wiki Cell claim-summary lane")
    print(
        f"claims={cfg.claim_count}, sources/claim={cfg.sources_per_claim}, "
        f"queries={cfg.query_events}, updates={cfg.update_events}, "
        f"base_state={format_bytes(cfg.state_bytes)}, "
        f"summary_state={format_bytes(summary.summary_state_bytes)}, seed={summary.seed}"
    )
    print()
    headers = [
        "policy",
        "kind",
        "recall",
        "recent",
        "q_read",
        "touch/e",
        "sum_stale",
        "src_stale",
        "consistent",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 146)
    for point in source_baseline.points:
        row = [
            point.policy,
            "source",
            fmt_pct(point.recall),
            fmt_pct(point.recent_recall),
            f"{point.cells_read_per_query:0.2f}",
            f"{point.cells_touched_per_event:0.2f}",
            "n/a",
            fmt_pct(point.stale_source_rate),
            fmt_pct(point.consistent_claim_rate),
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))
    for point in summary.points:
        row = [
            point.policy,
            "summary",
            fmt_pct(point.recall),
            fmt_pct(point.recent_recall),
            f"{point.cells_read_per_query:0.2f}",
            f"{point.cells_touched_per_event:0.2f}",
            fmt_pct(point.summary_stale_rate),
            fmt_pct(point.stale_source_rate),
            fmt_pct(point.consistent_claim_rate),
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))
    print()
    print("Interpretation:")
    print("- The summary cell is a low-bit second-level answer path per claim.")
    print("- summary_only makes answers cheap but leaves source pages stale.")
    print("- summary_error_repair spends local repair traffic to recover provenance freshness.")
    print("- This separates answer latency from background source-cell consistency.")


def main() -> None:
    print_result(run_ca_wiki_cell_summary_sweep())


if __name__ == "__main__":
    main()
