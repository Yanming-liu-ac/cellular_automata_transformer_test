"""Observation-window stress test for mixed CA wiki-memory guard counters."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    WikiMemoryMixedGuardCounterResult,
    run_wiki_memory_mixed_guard_counter_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_stress(result: WikiMemoryMixedGuardCounterResult) -> None:
    print("CA wiki-memory mixed guard counter observation-window stress")
    print(
        f"policy={result.policy}, target={result.target_route_coverage:0.2f}, "
        f"summary={result.summary_banks}x{result.summary_width}x"
        f"{result.summary_bits}-bit, tag_bits={result.density_tag_bits}"
    )
    headers = [
        "events",
        "q_s",
        "q_d",
        "need",
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
    for point in result.points:
        row = [
            f"{point.probe_queries}/{point.probe_updates}",
            f"{point.sparse_probe_queries}",
            f"{point.dense_probe_queries}",
            f"{point.guard_required_win_count}",
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
    print("- This isolates the 50% dense, 512-page block, same-tag radius-1 setting.")
    print("- shared should reach all dense blocks without raising sparse false-enable.")
    print("- Short windows expose whether the counter rule has enough evidence.")


def main() -> None:
    print_stress(
        run_wiki_memory_mixed_guard_counter_sweep(
            dense_page_fractions=(0.50,),
            tag_thresholds=(2,),
            guard_counter_block_page_options=(512,),
            guard_share_radius_options=(1,),
            quality_probe_event_options=((128, 64), (256, 128), (512, 256), (1024, 512)),
        )
    )


if __name__ == "__main__":
    main()
