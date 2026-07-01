"""Split paragraph-confidence tradeoff for CA Wiki Cell importance."""

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
    CAWikiCellParagraphSplitConfidencePoint,
    CAWikiCellParagraphSplitConfidenceResult,
    run_ca_wiki_cell_paragraph_split_confidence_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def mean(points: list[CAWikiCellParagraphSplitConfidencePoint], field: str) -> float:
    return sum(float(getattr(point, field)) for point in points) / float(len(points))


def print_result(result: CAWikiCellParagraphSplitConfidenceResult) -> None:
    print("CA Wiki Cell paragraph split-confidence tradeoff")
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
        "sum_gap",
        "src_gap",
        "agree",
        "touch/e",
        "down",
        "pass",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 257)
    buckets: dict[str, list[CAWikiCellParagraphSplitConfidencePoint]] = defaultdict(list)
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
            f"{point.mean_summary_core_gap_per_claim:0.2f}",
            f"{point.mean_source_core_gap_per_claim:0.2f}",
            f"{point.mean_agreement_gap_per_claim:0.2f}",
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
    print("- baseline_4d uses observed error, core conflict, stale weight, and parser misses.")
    print("- coverage_lut5d is the previous combined weighted field-coverage bucket.")
    print("- split_guard7d adds a conservative guard over summary-core, source-core, and agreement gaps.")
    print("- factor_vote56b/80b compress the split guard into small projected vote tables.")
    print("- covsafe/shiftguard are hand-coded coverage-shift diagnostics.")
    print("- learned_shift_selector is a multi-distribution selector over base mode, votes, core gap, and parser misses.")
    print("- two-branch variants test parser-tolerant factor routing plus learned coverage repair.")
    print("- regime_counter_selector uses a tile-level regime LUT to choose factor or coverage-repair routing.")
    print("- traffic_regime_selector is a higher-cost traffic diagnostic and is not promoted as baseline.")
    print("- split_lut7d is the direct 7D classifier; it is more aggressive and can miss the safety gate.")


def main() -> None:
    print_result(run_ca_wiki_cell_paragraph_split_confidence_sweep())


if __name__ == "__main__":
    main()
