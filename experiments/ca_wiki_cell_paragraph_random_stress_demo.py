"""Randomized held-out stress test for paragraph regime counters."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    CAWikiCellParagraphFactorizedGuardStressResult,
    run_ca_wiki_cell_paragraph_factorized_guard_random_stress_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(result: CAWikiCellParagraphFactorizedGuardStressResult) -> None:
    print("CA Wiki Cell paragraph randomized stress")
    print(f"train_seeds={result.train_seeds}, eval_seeds={result.eval_seeds}")
    print()
    print(
        "scenario                 variant                    total_lut  claims  "
        "misread  src_omit  dist   accuracy  strict_r  under     over      pass"
    )
    print("-" * 142)
    for point in result.points:
        total_lut = point.classifier_lut_bytes + point.guard_lut_bytes
        print(
            f"{point.scenario:<24} "
            f"{point.variant:<26} "
            f"{format_bytes(total_lut):>8} "
            f"{point.claim_count:>7} "
            f"{point.parser_misread_rate:>7.3f} "
            f"{point.source_core_omit_rate:>8.3f} "
            f"{point.distractor_rate:>6.3f} "
            f"{fmt_pct(point.accuracy):>9} "
            f"{fmt_pct(point.strict_recall):>9} "
            f"{fmt_pct(point.under_strict_rate):>8} "
            f"{fmt_pct(point.over_strict_rate):>8} "
            f"{point.eval_seed_count - point.failures}/{point.eval_seed_count}"
        )
    print()
    print("Interpretation:")
    print(
        "- Scenarios are deterministic from the random seed but mix parser, "
        "omission, distractor, and scale shifts."
    )
    print("- factor_vote80b is the parser-tolerant branch.")
    print("- two_branch_factor_selector is the coverage-repair branch.")
    print(
        "- regime_counter_selector chooses between them from aggregate low-bit counters."
    )
    print(
        "- subtile_regime_selector keeps the parent tile safety decision but "
        "allows local subtiles to relax repair when coverage risk is not maximal."
    )
    print(
        "- volatility_subtile_selector adds one conservative short-window dynamics bit."
    )
    print(
        "- directional_subtile_selector separates parser-rising from coverage-rising subtiles."
    )
    print(
        "- traffic_regime_selector is a safety-tuned traffic diagnostic; under "
        "random stress it currently collapses back to the conservative branch."
    )


def main() -> None:
    print_result(run_ca_wiki_cell_paragraph_factorized_guard_random_stress_sweep())


if __name__ == "__main__":
    main()
