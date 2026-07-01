"""Density-aware routing-tile diagnostic for mixed CA wiki-memory regions."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    WikiMemoryDensityAwareTileResult,
    run_wiki_memory_density_aware_tile_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_density_aware(result: WikiMemoryDensityAwareTileResult) -> None:
    print("CA wiki-memory density-aware routing tiles")
    print(
        f"policy={result.policy}, target={result.target_route_coverage:0.2f}, "
        f"events={result.query_events + result.update_events}, "
        f"summary={result.summary_banks}x{result.summary_width}x{result.summary_bits}-bit, "
        f"region_dir={result.region_directory_cells_per_query} cells/query"
    )
    headers = [
        "dense%",
        "sparse",
        "dense",
        "dense_on",
        "base_acc",
        "aware_acc",
        "all4_acc",
        "flat_acc",
        "base_rd",
        "aware_rd",
        "all4_rd",
        "flat_rd",
        "cut_f",
        "cut_b",
        "base_B",
        "aware_B",
        "all4_B",
        "save4",
        "extra_B",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 220)
    for point in result.points:
        row = [
            fmt_pct(point.dense_page_fraction),
            f"{point.sparse_pages}",
            f"{point.dense_pages}",
            "yes" if point.dense_tile_enabled else "no",
            fmt_pct(point.baseline_overall_recall),
            fmt_pct(point.aware_overall_recall),
            fmt_pct(point.all_dense_overall_recall),
            fmt_pct(point.flat_overall_recall),
            f"{point.baseline_cells_read_per_query:0.1f}",
            f"{point.aware_cells_read_per_query:0.1f}",
            f"{point.all_dense_cells_read_per_query:0.1f}",
            f"{point.flat_cells_read_per_query:0.1f}",
            fmt_pct(point.aware_read_reduction_vs_flat),
            fmt_pct(point.aware_read_reduction_vs_baseline),
            format_bytes(point.baseline_state_bytes),
            format_bytes(point.aware_state_bytes),
            format_bytes(point.all_dense_state_bytes),
            fmt_pct(point.aware_state_saving_vs_all_dense),
            format_bytes(point.aware_state_increase_vs_baseline),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- baseline uses learned fanout with 16-page tiles in every region.")
    print("- aware keeps sparse regions at 16-page tiles and uses 4-page dense tiles only where needed.")
    print("- all4 uses 4-page dense tiles everywhere.")
    print("- save4 is aware state saved versus all4.")


def main() -> None:
    print_density_aware(run_wiki_memory_density_aware_tile_sweep())


if __name__ == "__main__":
    main()
