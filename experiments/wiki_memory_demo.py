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


def print_points(title: str, points) -> None:
    print(title)
    headers = [
        "policy",
        "dirty",
        "age",
        "single",
        "multi",
        "overall",
        "recent",
        "stale",
        "route_m",
        "value_m",
        "read/q",
        "flat/q",
        "cut",
        "write/u",
        "key_u",
        "rev_u",
        "refresh",
        "pages/r",
        "groups/r",
        "errbook",
        "errfix",
        "probe_q",
        "probe_r",
        "clu_q",
        "clu_r",
        "clu_ok",
        "clu_u",
        "clu_fix",
        "prov",
    ]
    print(" | ".join(f"{header:>10}" for header in headers))
    print("-" * 292)
    for point in points:
        row = [
            point.policy,
            f"{point.dirty_threshold}",
            f"{point.max_age}",
            fmt_pct(point.single_hop_recall),
            fmt_pct(point.multihop_recall),
            fmt_pct(point.overall_recall),
            fmt_pct(point.recent_update_recall),
            fmt_pct(point.stale_miss_rate),
            fmt_pct(point.route_miss_rate),
            fmt_pct(point.value_miss_rate),
            f"{point.cells_read_per_query:0.1f}",
            f"{point.flat_cells_read_per_query:0.1f}",
            fmt_pct(point.read_reduction_rate),
            f"{point.cells_written_per_update:0.1f}",
            f"{point.key_updates}",
            f"{point.revision_updates}",
            f"{point.refresh_events}",
            f"{point.mean_pages_refreshed:0.1f}",
            f"{point.mean_groups_refreshed:0.1f}",
            f"{point.error_book_repairs}",
            f"{point.error_book_recoveries}",
            f"{point.error_probe_queries}",
            fmt_pct(point.error_probe_recall),
            f"{point.cluster_queries}",
            fmt_pct(point.cluster_recall),
            fmt_pct(point.cluster_consistency_rate),
            f"{point.cluster_updates}",
            f"{point.cluster_repair_events}",
            fmt_pct(point.provenance_precision),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))
    print()


def print_wiki(result: WikiMemorySweepResult) -> None:
    print("CA wiki-memory: mutable facts, page links, and triggered summaries")
    print(
        f"pages={result.page_count}, facts/page={result.facts_per_page}, "
        f"links/page={result.links_per_page}, group={result.group_size}, "
        f"select_groups={result.selected_groups}, select_pages={result.selected_pages}, "
        f"summary={result.summary_banks}x{result.summary_width}x{result.summary_bits}-bit, "
        f"events={result.query_events + result.update_events}, "
        f"revision_rate={result.revision_update_rate:0.2f}, "
        f"probe_rate={result.error_probe_query_rate:0.2f}, "
        f"clusters={result.contradiction_clusters}x{result.cluster_sources}, "
        f"cluster_update={result.cluster_update_rate:0.2f}, "
        f"cluster_query={result.cluster_query_rate:0.2f}, "
        f"state={format_bytes(result.state_bytes)}"
    )
    print()
    print_points("CA hierarchical group-route policies", result.points)
    print_points("Flat/RAG page-summary scan baselines", result.flat_points)
    print("Interpretation:")
    print("- flat/q is a full exact scan over all page facts.")
    print("- read/q is group summary routing, page summary routing, then exact fact reads.")
    print("- flat/RAG baselines scan every page summary before exact reads from selected pages.")
    print("- stale counts misses where the queried page was dirty when routing failed.")
    print("- value_m catches routed pages whose stored fact value is stale.")
    print("- errfix counts failed probes that would succeed immediately after repair.")
    print("- probe_r is recall on repeated failed-query probes drawn from the error book.")
    print("- clu_r measures replicated-claim queries across multi-source contradiction clusters.")
    print("- clu_ok requires every source page in the claim cluster to hold the current value.")


def main() -> None:
    print_wiki(run_wiki_memory_sweep())


if __name__ == "__main__":
    main()
