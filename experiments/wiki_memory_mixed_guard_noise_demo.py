"""Update-noise stress test for mixed CA wiki-memory guard counters."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import run_wiki_memory_mixed_guard_counter_sweep


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def main() -> None:
    scenarios = (
        ("base", 0.50, 0.30),
        ("rev80", 0.80, 0.30),
        ("clu60", 0.50, 0.60),
        ("both", 0.80, 0.60),
    )
    print("CA wiki-memory mixed guard counter update-noise stress")
    headers = [
        "scenario",
        "rev",
        "cluster",
        "q_s",
        "q_d",
        "local",
        "shared",
        "s_false",
        "sh_false",
        "d_w/l",
        "d_cmax",
        "c_B",
    ]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for label, revision_rate, cluster_rate in scenarios:
        result = run_wiki_memory_mixed_guard_counter_sweep(
            dense_page_fractions=(0.50,),
            tag_thresholds=(2,),
            guard_counter_block_page_options=(512,),
            guard_share_radius_options=(1,),
            quality_probe_event_options=((512, 256),),
            revision_update_rate=revision_rate,
            cluster_update_rate=cluster_rate,
        )
        point = result.points[0]
        row = [
            label,
            fmt_pct(revision_rate),
            fmt_pct(cluster_rate),
            f"{point.sparse_probe_queries}",
            f"{point.dense_probe_queries}",
            f"{point.dense_enabled_blocks}/{point.dense_guard_blocks}",
            f"{point.dense_shared_enabled_blocks}/{point.dense_guard_blocks}",
            fmt_pct(point.sparse_false_enable_rate),
            fmt_pct(point.sparse_shared_false_enable_rate),
            f"{point.dense_raw_wins}/{point.dense_raw_losses}",
            f"{point.dense_max_win_counter}/{point.dense_max_loss_counter}",
            format_bytes(point.guard_counter_state_bytes),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- This isolates 50% dense, 512-page blocks, same-tag radius-1 sharing.")
    print("- Revision and cluster rates stress dirty summaries and repair pressure.")
    print("- shared should remain full while sparse false-enable stays zero.")


if __name__ == "__main__":
    main()
