"""Paragraph field-coverage tradeoff for CA Wiki Cell importance."""

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
    CAWikiCellParagraphCoveragePoint,
    CAWikiCellParagraphCoverageResult,
    run_ca_wiki_cell_paragraph_coverage_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def mean(points: list[CAWikiCellParagraphCoveragePoint], field: str) -> float:
    return sum(float(getattr(point, field)) for point in points) / float(len(points))


def print_result(result: CAWikiCellParagraphCoverageResult) -> None:
    print("CA Wiki Cell paragraph coverage tradeoff")
    print(
        f"claims/eval={result.claim_count}, train_claims={result.train_claim_count}, "
        f"query_events={result.query_events}, update_events={result.update_events}, "
        f"compile_events={result.compile_events}, "
        f"source_omit(core/weak)={result.source_core_omit_rate:0.2f}/"
        f"{result.source_weak_omit_rate:0.2f}, "
        f"summary_omit(core/weak)={result.summary_core_omit_rate:0.2f}/"
        f"{result.summary_weak_omit_rate:0.2f}, "
        f"distractor={result.distractor_rate:0.2f}, "
        f"misread={result.parser_misread_rate:0.2f}, drop={result.parser_drop_rate:0.2f}"
    )
    print()
    headers = [
        "variant",
        "seed",
        "signal",
        "lut",
        "guard",
        "accuracy",
        "strict_p",
        "strict_r",
        "under",
        "over",
        "strict",
        "gap",
        "touch/e",
        "down",
        "pass",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 226)
    buckets: dict[str, list[CAWikiCellParagraphCoveragePoint]] = defaultdict(list)
    for point in result.points:
        buckets[point.variant].append(point)
        row = [
            point.variant,
            f"{point.eval_seed}",
            f"{point.local_signal_bits_per_claim}b",
            format_bytes(point.classifier_lut_bytes),
            format_bytes(point.guard_lut_bytes),
            fmt_pct(point.accuracy),
            fmt_pct(point.strict_precision),
            fmt_pct(point.strict_recall),
            fmt_pct(point.under_strict_rate),
            fmt_pct(point.over_strict_rate),
            fmt_pct(point.strict_rate),
            f"{point.mean_observed_coverage_gap_per_claim:0.2f}",
            f"{point.estimated_touch_per_event:0.2f}",
            f"{point.downgraded_strict_claims}",
            "yes" if point.target_met else "no",
        ]
        print(" | ".join(f"{cell:>14}" for cell in row))
    print()
    print("Mean by variant:")
    for variant in result.variants:
        points = buckets[variant]
        failures = sum(1 for point in points if not point.target_met)
        total_lut = points[0].classifier_lut_bytes + points[0].guard_lut_bytes
        print(
            f"- {variant}: total_lut={format_bytes(total_lut)}, "
            f"accuracy={fmt_pct(mean(points, 'accuracy'))}, "
            f"strict_r={fmt_pct(mean(points, 'strict_recall'))}, "
            f"under={fmt_pct(mean(points, 'under_strict_rate'))}, "
            f"over={fmt_pct(mean(points, 'over_strict_rate'))}, "
            f"strict={fmt_pct(mean(points, 'strict_rate'))}, "
            f"touch/e={mean(points, 'estimated_touch_per_event'):0.2f}, "
            f"failures={failures}/{len(points)}"
        )
    print()
    print("Interpretation:")
    print("- baseline_4d is the existing paragraph controller: error, conflict, stale, parser-miss.")
    print("- coverage_guard adds a 1-bit safe downgrade table indexed by weighted field-coverage gap.")
    print("- coverage_lut5d makes field-coverage gap a fifth 2-bit LUT input.")


def main() -> None:
    print_result(run_ca_wiki_cell_paragraph_coverage_sweep())


if __name__ == "__main__":
    main()
