"""Refresh-derived density-tag diagnostic for mixed CA wiki-memory regions."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    WikiMemoryDensityTagResult,
    run_wiki_memory_density_tag_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_density_tags(result: WikiMemoryDensityTagResult) -> None:
    print("CA wiki-memory refresh-derived density tags")
    print(
        f"policy={result.policy}, target={result.target_route_coverage:0.2f}, "
        f"events={result.query_events + result.update_events}, "
        f"summary={result.summary_banks}x{result.summary_width}x{result.summary_bits}-bit, "
        f"tag_bits={result.density_tag_bits}, "
        f"region_dir={result.region_directory_cells_per_query} cells/query, "
        f"probe={result.quality_probe_queries}q/{result.quality_probe_updates}u/"
        f"{100.0 * result.quality_probe_min_gain:0.1f}%, "
        f"guard_counter={result.guard_counter_bits}b/"
        f"{result.guard_counter_block_pages}p/"
        f"need{result.guard_required_win_count}, "
        f"counter_state={format_bytes(result.guard_counter_state_bytes)}"
    )
    headers = [
        "dense%",
        "thr",
        "s_tag",
        "d_tag",
        "tag_on",
        "guard_on",
        "p_base",
        "p_dense",
        "p_win",
        "p_loss",
        "c_win",
        "c_loss",
        "base",
        "tag_acc",
        "guard",
        "flat",
        "tag_rd",
        "guard_rd",
        "flat_rd",
        "tag_cut",
        "guard_cut",
        "tag_B",
        "guard_B",
    ]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for point in result.points:
        row = [
            fmt_pct(point.dense_page_fraction),
            f"{point.tag_threshold}",
            f"{point.sparse_density_tag}",
            f"{point.dense_density_tag}",
            "yes" if point.tag_dense_enabled else "no",
            "yes" if point.guard_dense_enabled else "no",
            fmt_pct(point.dense_probe_baseline_recall),
            fmt_pct(point.dense_probe_dense_recall),
            f"{point.dense_probe_dense_wins}",
            f"{point.dense_probe_dense_losses}",
            f"{point.dense_probe_win_counter}",
            f"{point.dense_probe_loss_counter}",
            fmt_pct(point.baseline_overall_recall),
            fmt_pct(point.tag_only_overall_recall),
            fmt_pct(point.guarded_overall_recall),
            fmt_pct(point.flat_overall_recall),
            f"{point.tag_only_cells_read_per_query:0.1f}",
            f"{point.guarded_cells_read_per_query:0.1f}",
            f"{point.flat_cells_read_per_query:0.1f}",
            fmt_pct(point.tag_only_read_reduction_vs_flat),
            fmt_pct(point.guarded_read_reduction_vs_flat),
            format_bytes(point.tag_only_state_bytes),
            format_bytes(point.guarded_state_bytes),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- s_tag and d_tag are low-bit tags from refresh-visible fact density.")
    print("- tag_on applies dense tiles from the density threshold alone.")
    print("- guard_on uses low-bit saturated counters: c_win >= need and c_loss == 0.")
    print("- p_base/p_dense and p_win/p_loss show raw dense-region paired-probe evidence.")
    print("- tag_cut and guard_cut compare read traffic with flat scan.")


def main() -> None:
    print_density_tags(run_wiki_memory_density_tag_sweep())


if __name__ == "__main__":
    main()
