"""Parser-miss guard tradeoff for text-source CA Wiki Cell importance."""

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
    CAWikiCellTextTraceGuardPoint,
    CAWikiCellTextTraceGuardResult,
    run_ca_wiki_cell_text_trace_guard_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def mean(points: list[CAWikiCellTextTraceGuardPoint], field: str) -> float:
    return sum(float(getattr(point, field)) for point in points) / float(len(points))


def print_result(result: CAWikiCellTextTraceGuardResult) -> None:
    print("CA Wiki Cell parser-miss guard tradeoff")
    print(
        f"claims/eval={result.claim_count}, train_claims={result.train_claim_count}, "
        f"query_events={result.query_events}, update_events={result.update_events}, "
        f"compile_events={result.compile_events}, "
        f"misread={result.parser_misread_rate:0.2f}, drop={result.parser_drop_rate:0.2f}"
    )
    print()
    headers = [
        "variant",
        "seed",
        "lut",
        "guard",
        "accuracy",
        "strict_p",
        "strict_r",
        "under",
        "over",
        "strict",
        "touch/e",
        "down",
        "pass",
    ]
    print(" | ".join(f"{header:>14}" for header in headers))
    print("-" * 211)
    buckets: dict[str, list[CAWikiCellTextTraceGuardPoint]] = defaultdict(list)
    for point in result.points:
        buckets[point.variant].append(point)
        row = [
            point.variant,
            f"{point.eval_seed}",
            format_bytes(point.classifier_lut_bytes),
            format_bytes(point.guard_lut_bytes),
            fmt_pct(point.accuracy),
            fmt_pct(point.strict_precision),
            fmt_pct(point.strict_recall),
            fmt_pct(point.under_strict_rate),
            fmt_pct(point.over_strict_rate),
            fmt_pct(point.strict_rate),
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
        print(
            f"- {variant}: accuracy={fmt_pct(mean(points, 'accuracy'))}, "
            f"strict_r={fmt_pct(mean(points, 'strict_recall'))}, "
            f"under={fmt_pct(mean(points, 'under_strict_rate'))}, "
            f"over={fmt_pct(mean(points, 'over_strict_rate'))}, "
            f"strict={fmt_pct(mean(points, 'strict_rate'))}, "
            f"touch/e={mean(points, 'estimated_touch_per_event'):0.2f}, "
            f"failures={failures}/{len(points)}"
        )
    print()
    print("Interpretation:")
    print("- baseline_3d is the existing 16B parser-noise controller.")
    print("- safe_miss_guard is a conservative 1-bit 4D downgrade table after the 3D LUT.")
    print("- miss_lut4d uses parser-miss as a fourth LUT input; it cuts over-strict traffic more, at 64B.")


def main() -> None:
    print_result(run_ca_wiki_cell_text_trace_guard_sweep())


if __name__ == "__main__":
    main()
