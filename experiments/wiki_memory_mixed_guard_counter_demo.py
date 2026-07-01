"""Mixed-stream low-bit guard counters for CA wiki-memory density tags."""

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


def print_mixed_guard_counters(result: WikiMemoryMixedGuardCounterResult) -> None:
    print("CA wiki-memory mixed-stream low-bit density guard counters")
    if len(result.points) == 0:
        print("no points")
        return
    first = result.points[0]
    print(
        f"policy={result.policy}, target={result.target_route_coverage:0.2f}, "
        f"probe={result.quality_probe_queries}q/{result.quality_probe_updates}u/"
        f"{100.0 * result.quality_probe_min_gain:0.1f}%, "
        f"summary={result.summary_banks}x{result.summary_width}x"
        f"{result.summary_bits}-bit, tag_bits={result.density_tag_bits}, "
        f"counter={first.guard_counter_bits}b, locality=rows"
    )
    headers = [
        "dense%",
        "thr",
        "blk_pg",
        "sh_r",
        "loss",
        "need",
        "s_tag",
        "d_tag",
        "q_s",
        "q_d",
        "blk_s",
        "blk_d",
        "en_s",
        "en_d",
        "sh_s",
        "sh_d",
        "s_false",
        "d_on",
        "sh_false",
        "sh_on",
        "s_w/l",
        "d_w/l",
        "s_cmax",
        "d_cmax",
        "c_B",
    ]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for point in result.points:
        row = [
            fmt_pct(point.dense_page_fraction),
            f"{point.tag_threshold}",
            f"{point.guard_counter_block_pages}",
            f"{point.guard_share_radius_blocks}",
            f"{point.guard_allowed_loss_count}",
            f"{point.guard_required_win_count}",
            f"{point.sparse_density_tag}",
            f"{point.dense_density_tag}",
            f"{point.sparse_probe_queries}",
            f"{point.dense_probe_queries}",
            f"{point.sparse_guard_blocks}",
            f"{point.dense_guard_blocks}",
            f"{point.sparse_enabled_blocks}",
            f"{point.dense_enabled_blocks}",
            f"{point.sparse_shared_enabled_blocks}",
            f"{point.dense_shared_enabled_blocks}",
            fmt_pct(point.sparse_false_enable_rate),
            fmt_pct(point.dense_enable_rate),
            fmt_pct(point.sparse_shared_false_enable_rate),
            fmt_pct(point.dense_shared_enable_rate),
            f"{point.sparse_raw_wins}/{point.sparse_raw_losses}",
            f"{point.dense_raw_wins}/{point.dense_raw_losses}",
            f"{point.sparse_max_win_counter}/{point.sparse_max_loss_counter}",
            f"{point.dense_max_win_counter}/{point.dense_max_loss_counter}",
            format_bytes(point.guard_counter_state_bytes),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- One mixed event stream feeds sparse and dense guard blocks together.")
    print("- en_s/en_d are local counters; sh_s/sh_d add same-tag neighbor sharing.")
    print("- loss is the allowed low-bit loss-counter value before suppressing a block.")
    print("- false columns should stay near zero; on columns show dense block coverage.")
    print("- cmax columns are max saturated win/loss counters within sparse/dense blocks.")


def main() -> None:
    print_mixed_guard_counters(
        run_wiki_memory_mixed_guard_counter_sweep(
            tag_thresholds=(2,),
            guard_counter_block_page_options=(256, 512, 1024),
            guard_share_radius_options=(0, 1, 2),
        )
    )


if __name__ == "__main__":
    main()
