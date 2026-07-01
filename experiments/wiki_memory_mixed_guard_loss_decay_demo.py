"""Event-driven loss decay for mixed CA wiki-memory guard counters."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.wiki_memory import (  # noqa: E402
    WikiMemoryMixedGuardCounterResult,
    run_wiki_memory_mixed_guard_counter_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_loss_decay(result: WikiMemoryMixedGuardCounterResult) -> None:
    print("CA wiki-memory mixed guard loss-decay modes")
    print(
        f"policy={result.policy}, strict_loss_gate=0, "
        f"probe={result.quality_probe_queries}q/{result.quality_probe_updates}u"
    )
    headers = [
        "dense%",
        "decay",
        "shared",
        "sh_false",
        "d_w/l",
        "d_cmax",
        "s_w/l",
    ]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for point in result.points:
        row = [
            fmt_pct(point.dense_page_fraction),
            point.guard_loss_decay_mode,
            f"{point.dense_shared_enabled_blocks}/{point.dense_guard_blocks}",
            fmt_pct(point.sparse_shared_false_enable_rate),
            f"{point.dense_raw_wins}/{point.dense_raw_losses}",
            f"{point.dense_max_win_counter}/{point.dense_max_loss_counter}",
            f"{point.sparse_raw_wins}/{point.sparse_raw_losses}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- none is the previous strict zero-loss gate.")
    print("- win decrements the local loss counter when the dense route later wins.")
    print("- nonloss decrements the local loss counter on later non-loss queries.")
    print("- These modes use the same 4-bit win/loss counters and only local updates.")


def main() -> None:
    print_loss_decay(
        run_wiki_memory_mixed_guard_counter_sweep(
            dense_page_fractions=(0.25, 0.75),
            tag_thresholds=(2,),
            guard_counter_block_page_options=(512,),
            guard_share_radius_options=(1,),
            guard_loss_decay_options=("none", "win", "nonloss"),
            guard_allowed_loss_options=(0,),
            quality_probe_event_options=((512, 256),),
            quality_probe_seed=1501,
        )
    )


if __name__ == "__main__":
    main()
