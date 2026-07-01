"""Learn a tiny LUT over wiki-memory guard radius, loss decay, and tolerance."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes  # noqa: E402
from cellular_transformer.wiki_memory import (  # noqa: E402
    WikiMemoryLearnedGuardSharingResult,
    run_wiki_memory_learned_guard_sharing_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(result: WikiMemoryLearnedGuardSharingResult) -> None:
    print("CA wiki-memory learned guard decay/tolerance LUT")
    print(
        f"policy={result.policy}, radius_options={result.radius_options}, "
        f"decay_options={result.loss_decay_options}, "
        f"loss_options={result.allowed_loss_options}, "
        f"lut_state={format_bytes(result.radius_lut_state_bytes)}"
    )
    print()
    print("Learned LUT")
    headers = ["blk_pg", "radius", "decay", "loss", "train_n", "cost"]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for entry in result.entries:
        row = [
            f"{entry.guard_counter_block_pages}",
            f"{entry.chosen_share_radius_blocks}",
            entry.chosen_loss_decay_mode,
            f"{entry.chosen_allowed_loss_count}",
            f"{entry.training_points}",
            f"{entry.training_cost:0.3f}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Evaluation")
    headers = ["seed", "dense%", "decay", "loss", "target", "local", "learned", "false", "d_w/l"]
    header_line = " | ".join(f"{header:>10}" for header in headers)
    print(header_line)
    print("-" * len(header_line))
    for point in result.points:
        row = [
            f"{point.eval_seed}",
            fmt_pct(point.dense_page_fraction),
            point.chosen_loss_decay_mode,
            f"{point.chosen_allowed_loss_count}",
            fmt_pct(point.target_dense_enable_rate),
            fmt_pct(point.local_dense_enable_rate),
            fmt_pct(point.learned_dense_enable_rate),
            fmt_pct(point.learned_sparse_false_enable_rate),
            f"{point.dense_raw_wins}/{point.dense_raw_losses}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- This narrow audit learns the 512-page/radius-1 guard controller.")
    print("- Ties prefer strict loss=0, then event-driven decay over permanent tolerance.")
    print("- The held-out seed1501 row checks whether decay repairs the old 99/1 failure.")


def main() -> None:
    print_result(
        run_wiki_memory_learned_guard_sharing_sweep(
            dense_page_fractions=(0.25, 0.75),
            guard_counter_block_page_options=(512,),
            guard_share_radius_options=(1,),
            guard_loss_decay_options=("none", "win", "nonloss"),
            guard_allowed_loss_options=(0, 1),
            quality_probe_event_options=((512, 256),),
            eval_seeds=(1201, 1501),
        )
    )


if __name__ == "__main__":
    main()
