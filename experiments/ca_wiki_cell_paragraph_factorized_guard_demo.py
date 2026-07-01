"""Factorized paragraph-confidence guards for CA Wiki Cell importance."""

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
    CAWikiCellParagraphFactorizedGuardPoint,
    CAWikiCellParagraphFactorizedGuardResult,
    run_ca_wiki_cell_paragraph_factorized_guard_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def mean(points: list[CAWikiCellParagraphFactorizedGuardPoint], field: str) -> float:
    return sum(float(getattr(point, field)) for point in points) / float(len(points))


def print_result(result: CAWikiCellParagraphFactorizedGuardResult) -> None:
    print("CA Wiki Cell paragraph factorized guard tradeoff")
    print(
        f"claims/eval={result.claim_count}, train_claims={result.train_claim_count}, "
        f"query_events={result.query_events}, update_events={result.update_events}, "
        f"compile_events={result.compile_events}"
    )
    print()
    print(
        "variant                    total_lut  guard     signal  accuracy  strict_r  "
        "under     over      strict    touch/e  down    pass"
    )
    print("-" * 126)
    buckets: dict[str, list[CAWikiCellParagraphFactorizedGuardPoint]] = defaultdict(list)
    for point in result.points:
        buckets[point.variant].append(point)
    for variant in result.variants:
        points = buckets[variant]
        failures = sum(1 for point in points if not point.target_met)
        total_lut = points[0].classifier_lut_bytes + points[0].guard_lut_bytes
        print(
            f"{variant:<26} "
            f"{format_bytes(total_lut):>8} "
            f"{format_bytes(points[0].guard_lut_bytes):>8} "
            f"{points[0].local_signal_bits_per_claim:>5}b "
            f"{fmt_pct(mean(points, 'accuracy')):>9} "
            f"{fmt_pct(mean(points, 'strict_recall')):>9} "
            f"{fmt_pct(mean(points, 'under_strict_rate')):>8} "
            f"{fmt_pct(mean(points, 'over_strict_rate')):>8} "
            f"{fmt_pct(mean(points, 'strict_rate')):>9} "
            f"{mean(points, 'estimated_touch_per_event'):>7.2f} "
            f"{mean(points, 'downgraded_strict_claims'):>6.1f} "
            f"{len(points) - failures}/{len(points)}"
        )
    print()
    print("Interpretation:")
    print("- factor_vote56b uses four small projections, 56B of one-bit guard state.")
    print("- factor_vote80b uses two 4D and two 3D projections, 80B of guard state.")
    print("- covsafe/shiftguard add hand-coded coverage-shift protection diagnostics.")
    print("- all factorized variants reuse the 64B baseline classifier.")


def main() -> None:
    print_result(run_ca_wiki_cell_paragraph_factorized_guard_sweep())


if __name__ == "__main__":
    main()
