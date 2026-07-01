"""Grid diagnostic for learned CA wiki-memory fanout."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    WikiMemoryLearnedFanoutGridResult,
    run_wiki_memory_learned_fanout_grid_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_grid(result: WikiMemoryLearnedFanoutGridResult) -> None:
    print("CA wiki-memory learned fanout grid")
    print(
        f"policy={result.policy}, target={result.target_route_coverage:0.2f}, "
        f"events={result.query_events + result.update_events}, "
        f"summary={result.summary_banks}x{result.summary_width}x{result.summary_bits}-bit"
    )
    headers = [
        "pages",
        "facts/p",
        "fixed",
        "adapt",
        "learned",
        "flat",
        "fix_rd",
        "ad_rd",
        "lut_rd",
        "flat_rd",
        "cut_f",
        "cut_a",
        "lut_B",
        "train",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 171)
    for point in result.points:
        row = [
            f"{point.page_count}",
            f"{point.facts_per_page}",
            fmt_pct(point.fixed_overall_recall),
            fmt_pct(point.adaptive_overall_recall),
            fmt_pct(point.learned_overall_recall),
            fmt_pct(point.flat_overall_recall),
            f"{point.fixed_cells_read_per_query:0.1f}",
            f"{point.adaptive_cells_read_per_query:0.1f}",
            f"{point.learned_cells_read_per_query:0.1f}",
            f"{point.flat_cells_read_per_query:0.1f}",
            fmt_pct(point.learned_read_reduction_vs_flat),
            fmt_pct(point.learned_read_reduction_vs_adaptive),
            format_bytes(point.fanout_lut_state_bytes),
            f"{point.fanout_training_examples}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- fixed is the original four-group CA route.")
    print("- adapt is hand adaptive g4_max32_margin1 on the same workload.")
    print("- learned is the conservative local fanout LUT trained per geometry.")
    print("- cut_f and cut_a compare learned reads with flat scan and hand adaptive.")


def main() -> None:
    print_grid(run_wiki_memory_learned_fanout_grid_sweep())


if __name__ == "__main__":
    main()
