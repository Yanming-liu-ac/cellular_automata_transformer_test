"""Strict versus budget repair modes for CA Wiki Cell v0."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    CAWikiCellLearnedRepairPoint,
    CAWikiCellLearnedRepairResult,
    run_ca_wiki_cell_learned_repair_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def bucket_stats(
    result: CAWikiCellLearnedRepairResult,
) -> dict[tuple[int, int], tuple[float, float, float, float, int, int]]:
    buckets: dict[tuple[int, int], list[CAWikiCellLearnedRepairPoint]] = defaultdict(list)
    for point in result.points:
        buckets[(point.sources_per_claim, point.update_events)].append(point)
    stats = {}
    for key, points in buckets.items():
        count = float(len(points))
        stats[key] = (
            sum(point.recall for point in points) / count,
            sum(point.recent_recall for point in points) / count,
            sum(point.stale_source_rate for point in points) / count,
            sum(point.cells_touched_per_event for point in points) / count,
            sum(1 for point in points if not point.target_met),
            len(points),
        )
    return stats


def entry_map(result: CAWikiCellLearnedRepairResult):
    return {
        (entry.sources_per_claim, entry.update_events): entry
        for entry in result.entries
    }


def print_result(
    *,
    strict: CAWikiCellLearnedRepairResult,
    budget: CAWikiCellLearnedRepairResult,
) -> None:
    strict_stats = bucket_stats(strict)
    budget_stats = bucket_stats(budget)
    strict_entries = entry_map(strict)
    budget_entries = entry_map(budget)
    dual_lut_bytes = strict.lut_state_bytes + budget.lut_state_bytes
    print("CA Wiki Cell dual repair modes")
    print(
        f"strict target=recall>={fmt_pct(strict.target_recall)}, "
        f"recent>={fmt_pct(strict.target_recent_recall)}, "
        f"stale<={fmt_pct(strict.max_stale_source_rate)}"
    )
    print(
        f"budget target=recall>={fmt_pct(budget.target_recall)}, "
        f"recent>={fmt_pct(budget.target_recent_recall)}, "
        f"stale<={fmt_pct(budget.max_stale_source_rate)}"
    )
    print(
        f"single-mode lut={format_bytes(strict.lut_state_bytes)}, "
        f"dual-mode lut={format_bytes(dual_lut_bytes)}, "
        f"candidates={strict.candidate_count}"
    )
    print()

    headers = [
        "src",
        "updates",
        "strict",
        "s_touch",
        "s_fail",
        "budget",
        "b_touch",
        "b_fail",
        "saved",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 146)
    for key in sorted(strict_stats):
        strict_entry = strict_entries[key]
        budget_entry = budget_entries[key]
        strict_touch = strict_stats[key][3]
        budget_touch = budget_stats[key][3]
        saved = max(0.0, strict_touch - budget_touch)
        saved_rate = saved / strict_touch if strict_touch else 0.0
        row = [
            f"{key[0]}",
            f"{key[1]}",
            strict_entry.chosen_policy,
            f"{strict_touch:0.2f}",
            f"{strict_stats[key][4]}/{strict_stats[key][5]}",
            budget_entry.chosen_policy,
            f"{budget_touch:0.2f}",
            f"{budget_stats[key][4]}/{budget_stats[key][5]}",
            fmt_pct(saved_rate),
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))

    strict_failures = sum(1 for point in strict.points if not point.target_met)
    budget_failures = sum(1 for point in budget.points if not point.target_met)
    strict_touch = sum(point.cells_touched_per_event for point in strict.points) / len(strict.points)
    budget_touch = sum(point.cells_touched_per_event for point in budget.points) / len(budget.points)
    print()
    print("Aggregate:")
    print(
        f"- strict: failures={strict_failures}/{len(strict.points)}, "
        f"mean_touch={strict_touch:0.2f}"
    )
    print(
        f"- budget: failures={budget_failures}/{len(budget.points)}, "
        f"mean_touch={budget_touch:0.2f}, "
        f"saved={fmt_pct((strict_touch - budget_touch) / strict_touch)}"
    )
    print("Interpretation:")
    print("- Strict mode buys zero target failures by repairing every update in hard buckets.")
    print("- Budget mode uses periodic or query-triggered repair and saves maintenance traffic.")
    print("- Storing both policy ids is still only a few bytes in this diagnostic.")


def main() -> None:
    budget = run_ca_wiki_cell_learned_repair_sweep()
    strict = run_ca_wiki_cell_learned_repair_sweep(
        target_recall=0.99,
        target_recent_recall=0.98,
        max_stale_source_rate=0.01,
        miss_penalty_weight=1000.0,
        stale_penalty_weight=1000.0,
    )
    print_result(strict=strict, budget=budget)


if __name__ == "__main__":
    main()
