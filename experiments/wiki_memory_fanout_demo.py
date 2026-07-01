"""Adaptive group-fanout diagnostic for dense CA wiki-memory pages."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.wiki_memory import (
    WikiMemoryFanoutResult,
    run_wiki_memory_fanout_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_fanout(result: WikiMemoryFanoutResult) -> None:
    print("CA wiki-memory fanout: fixed versus adaptive group reads")
    print(
        f"policy={result.policy}, pages={result.page_count}, "
        f"facts/page={result.facts_per_page}, width={result.summary_width}, "
        f"events={result.query_events + result.update_events}"
    )
    headers = [
        "route",
        "base_g",
        "max_g",
        "margin",
        "ca_acc",
        "flat_acc",
        "ca_ok",
        "ca_rd/q",
        "flat_rd/q",
        "exact/q",
        "cut_f",
        "cut_x",
        "ca_wr/u",
    ]
    print(" | ".join(f"{header:>15}" for header in headers))
    print("-" * 215)
    for point in result.points:
        row = [
            point.route_label,
            f"{point.selected_groups}",
            f"{point.adaptive_max_groups}",
            f"{point.adaptive_score_margin}",
            fmt_pct(point.ca_overall_recall),
            fmt_pct(point.flat_overall_recall),
            fmt_pct(point.ca_cluster_consistency_rate),
            f"{point.ca_cells_read_per_query:0.1f}",
            f"{point.flat_cells_read_per_query:0.1f}",
            f"{point.exact_scan_cells_per_query:0.1f}",
            fmt_pct(point.ca_read_reduction_vs_flat),
            fmt_pct(point.ca_read_reduction_vs_exact_scan),
            f"{point.ca_cells_written_per_update:0.1f}",
        ]
        print(" | ".join(f"{cell:>15}" for cell in row))

    print()
    print("Interpretation:")
    print("- adaptive routes start from base_g groups and expand on low-margin summary ties.")
    print("- max_g caps local fanout so dense pages do not degrade into flat page scans.")
    print("- cut_f compares CA reads with the flat page-summary scan on the same workload.")


def main() -> None:
    print_fanout(run_wiki_memory_fanout_sweep())


if __name__ == "__main__":
    main()
