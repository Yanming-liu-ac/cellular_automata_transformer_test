"""Facts-per-page and summary-width pressure diagnostic for CA wiki-memory."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    WikiMemoryDensityResult,
    run_wiki_memory_density_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_density(result: WikiMemoryDensityResult) -> None:
    print("CA wiki-memory density: facts/page and summary-width pressure")
    print(
        f"policy={result.policy}, pages={result.page_count}, "
        f"events={result.query_events + result.update_events}, "
        f"summary_banks={result.summary_banks}, bits={result.summary_bits}"
    )
    headers = [
        "width",
        "facts/p",
        "state",
        "ca_acc",
        "flat_acc",
        "ca_ok",
        "flat_ok",
        "ca_rd/q",
        "flat_rd/q",
        "exact/q",
        "ca_cut_f",
        "ca_cut_x",
        "ca_wr/u",
        "flat_wr/u",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 162)
    for point in result.points:
        row = [
            f"{point.summary_width}",
            f"{point.facts_per_page}",
            format_bytes(point.state_bytes),
            fmt_pct(point.ca_overall_recall),
            fmt_pct(point.flat_overall_recall),
            fmt_pct(point.ca_cluster_consistency_rate),
            fmt_pct(point.flat_cluster_consistency_rate),
            f"{point.ca_cells_read_per_query:0.1f}",
            f"{point.flat_cells_read_per_query:0.1f}",
            f"{point.exact_scan_cells_per_query:0.1f}",
            fmt_pct(point.ca_read_reduction_vs_flat),
            fmt_pct(point.ca_read_reduction_vs_exact_scan),
            f"{point.ca_cells_written_per_update:0.1f}",
            f"{point.flat_cells_written_per_update:0.1f}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- width is the per-bank low-bit page-summary width.")
    print("- facts/p increases exact payload density inside each page.")
    print("- this tests summary collision pressure, not only page-count scaling.")


def main() -> None:
    print_density(run_wiki_memory_density_sweep())


if __name__ == "__main__":
    main()
