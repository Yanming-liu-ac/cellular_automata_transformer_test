"""Dense routing-tile diagnostic for CA wiki-memory."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    WikiMemoryDenseTileResult,
    run_wiki_memory_dense_tile_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_dense_tile(result: WikiMemoryDenseTileResult) -> None:
    print("CA wiki-memory dense routing tiles")
    print(
        f"policy={result.policy}, target={result.target_route_coverage:0.2f}, "
        f"events={result.query_events + result.update_events}, "
        f"summary={result.summary_banks}x{result.summary_width}x{result.summary_bits}-bit"
    )
    headers = [
        "pages",
        "facts/p",
        "base_gs",
        "dense_gs",
        "base_acc",
        "dense_acc",
        "flat_acc",
        "base_rd",
        "dense_rd",
        "flat_rd",
        "cut_f",
        "cut_b",
        "dense_B",
        "extra_B",
        "lut_B",
        "train",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 195)
    for point in result.points:
        row = [
            f"{point.page_count}",
            f"{point.facts_per_page}",
            f"{point.baseline_group_size}",
            f"{point.dense_group_size}",
            fmt_pct(point.baseline_overall_recall),
            fmt_pct(point.dense_overall_recall),
            fmt_pct(point.flat_overall_recall),
            f"{point.baseline_cells_read_per_query:0.1f}",
            f"{point.dense_cells_read_per_query:0.1f}",
            f"{point.flat_cells_read_per_query:0.1f}",
            fmt_pct(point.dense_read_reduction_vs_flat),
            fmt_pct(point.dense_read_reduction_vs_baseline),
            format_bytes(point.dense_state_bytes),
            format_bytes(point.dense_state_increase_bytes),
            format_bytes(point.dense_lut_state_bytes),
            f"{point.dense_training_examples}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- base_gs is the previous 16-page routing tile with learned max32 fanout.")
    print("- dense_gs is a 4-page routing tile with learned max48 fanout.")
    print("- cut_f compares dense-tile reads with flat page-summary scan.")
    print("- cut_b compares dense-tile reads with the previous learned baseline.")


def main() -> None:
    print_dense_tile(run_wiki_memory_dense_tile_sweep())


if __name__ == "__main__":
    main()
