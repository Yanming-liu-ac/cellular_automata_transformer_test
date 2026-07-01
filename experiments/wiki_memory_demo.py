"""Mutable CA wiki-memory diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import WikiMemorySweepResult, run_wiki_memory_sweep


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_wiki(result: WikiMemorySweepResult) -> None:
    print("CA wiki-memory: mutable facts, page links, and triggered summaries")
    print(
        f"pages={result.page_count}, facts/page={result.facts_per_page}, "
        f"links/page={result.links_per_page}, group={result.group_size}, "
        f"select_groups={result.selected_groups}, select_pages={result.selected_pages}, "
        f"summary={result.summary_banks}x{result.summary_width}x{result.summary_bits}-bit, "
        f"events={result.query_events + result.update_events}, "
        f"state={format_bytes(result.state_bytes)}"
    )
    headers = [
        "policy",
        "dirty",
        "age",
        "single",
        "multi",
        "overall",
        "recent",
        "stale",
        "read/q",
        "flat/q",
        "cut",
        "write/u",
        "refresh",
        "pages/r",
        "groups/r",
        "errbook",
        "errfix",
        "prov",
    ]
    print(" | ".join(f"{header:>12}" for header in headers))
    print("-" * 230)
    for point in result.points:
        row = [
            point.policy,
            f"{point.dirty_threshold}",
            f"{point.max_age}",
            fmt_pct(point.single_hop_recall),
            fmt_pct(point.multihop_recall),
            fmt_pct(point.overall_recall),
            fmt_pct(point.recent_update_recall),
            fmt_pct(point.stale_miss_rate),
            f"{point.cells_read_per_query:0.1f}",
            f"{point.flat_cells_read_per_query:0.1f}",
            fmt_pct(point.read_reduction_rate),
            f"{point.cells_written_per_update:0.1f}",
            f"{point.refresh_events}",
            f"{point.mean_pages_refreshed:0.1f}",
            f"{point.mean_groups_refreshed:0.1f}",
            f"{point.error_book_repairs}",
            f"{point.error_book_recoveries}",
            fmt_pct(point.provenance_precision),
        ]
        print(" | ".join(f"{cell:>12}" for cell in row))

    print()
    print("Interpretation:")
    print("- flat/q is a full exact scan over all page facts.")
    print("- read/q is group summary routing, page summary routing, then exact fact reads.")
    print("- stale counts misses where the queried page was dirty when routing failed.")
    print("- errfix counts failed probes that would succeed immediately after repair.")


def main() -> None:
    print_wiki(run_wiki_memory_sweep())


if __name__ == "__main__":
    main()
