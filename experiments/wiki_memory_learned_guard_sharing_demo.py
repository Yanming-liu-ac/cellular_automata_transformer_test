"""Learn a tiny sharing-radius LUT for mixed CA wiki-memory guard counters."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    WikiMemoryLearnedGuardSharingResult,
    run_wiki_memory_learned_guard_sharing_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_learned_guard(result: WikiMemoryLearnedGuardSharingResult) -> None:
    print("CA wiki-memory learned guard-counter sharing radius")
    print(
        f"policy={result.policy}, radius_options={result.radius_options}, "
        f"enable_dense_at>={fmt_pct(result.min_dense_fraction_to_enable)}, "
        f"lut_state={format_bytes(result.radius_lut_state_bytes)}"
    )
    print()
    print("Learned radius LUT")
    headers = ["blk_pg", "radius", "train_n", "cost"]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for entry in result.entries:
        row = [
            f"{entry.guard_counter_block_pages}",
            f"{entry.chosen_share_radius_blocks}",
            f"{entry.training_points}",
            f"{entry.training_cost:0.3f}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Evaluation")
    headers = [
        "dense%",
        "blk_pg",
        "radius",
        "target",
        "local",
        "learned",
        "s_false",
        "sh_false",
        "d_w/l",
    ]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for point in result.points:
        row = [
            fmt_pct(point.dense_page_fraction),
            f"{point.guard_counter_block_pages}",
            f"{point.chosen_share_radius_blocks}",
            fmt_pct(point.target_dense_enable_rate),
            fmt_pct(point.local_dense_enable_rate),
            fmt_pct(point.learned_dense_enable_rate),
            fmt_pct(point.local_sparse_false_enable_rate),
            fmt_pct(point.learned_sparse_false_enable_rate),
            f"{point.dense_raw_wins}/{point.dense_raw_losses}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- The LUT maps guard block size to a same-tag sharing radius.")
    print("- The training objective enables dense blocks at 50%+ dense while keeping sparse false-enable at zero.")
    print("- lut_state is sub-byte here because only three block-size entries are learned.")


def main() -> None:
    print_learned_guard(
        run_wiki_memory_learned_guard_sharing_sweep(
            dense_page_fractions=(0.25, 0.50, 0.75),
            guard_counter_block_page_options=(256, 512, 1024),
            guard_share_radius_options=(0, 1, 2),
            quality_probe_event_options=((512, 256),),
        )
    )


if __name__ == "__main__":
    main()
