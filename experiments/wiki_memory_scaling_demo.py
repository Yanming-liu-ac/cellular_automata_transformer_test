"""Scaling diagnostic for CA wiki-memory versus flat page-summary scans."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    WikiMemoryScalingResult,
    run_wiki_memory_scaling_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_scaling(result: WikiMemoryScalingResult) -> None:
    print("CA wiki-memory scaling: hierarchical route versus flat page-summary scan")
    print(
        f"policy={result.policy}, events={result.query_events + result.update_events}, "
        f"summary={result.summary_banks}x{result.summary_width}x{result.summary_bits}-bit"
    )
    headers = [
        "pages",
        "facts/p",
        "clusters",
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
    print("-" * 174)
    for point in result.points:
        row = [
            f"{point.page_count}",
            f"{point.facts_per_page}",
            f"{point.contradiction_clusters}",
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
    print("- ca_rd/q uses group summaries plus selected page summaries.")
    print("- flat_rd/q scans every page summary before reading selected exact facts.")
    print("- exact/q is a full exact fact scan, included as a simple lower-bound baseline.")


def main() -> None:
    print_scaling(run_wiki_memory_scaling_sweep())


if __name__ == "__main__":
    main()
