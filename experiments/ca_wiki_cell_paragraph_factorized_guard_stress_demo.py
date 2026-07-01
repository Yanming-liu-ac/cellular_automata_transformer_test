"""Distribution-shift stress test for paragraph factorized guards."""

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
    run_ca_wiki_cell_paragraph_factorized_guard_stress_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(result: CAWikiCellParagraphFactorizedGuardStressResult) -> None:
    print("CA Wiki Cell paragraph factorized guard stress")
    print(
        "train_noise="
        f"misread={result.train_parser_misread_rate:0.2f}, "
        f"drop={result.train_parser_drop_rate:0.2f}, "
        f"source_omit={result.train_source_core_omit_rate:0.2f}/"
        f"{result.train_source_weak_omit_rate:0.2f}, "
        f"summary_omit={result.train_summary_core_omit_rate:0.2f}/"
        f"{result.train_summary_weak_omit_rate:0.2f}, "
        f"distractor={result.train_distractor_rate:0.2f}"
    )
    print(f"train_seeds={result.train_seeds}, eval_seeds={result.eval_seeds}")
    print()
    print(
        "scenario       variant          total_lut  accuracy  strict_r  "
        "under     over      strict    touch/e  down    pass"
    )
    print("-" * 116)
    for point in result.points:
        total_lut = point.classifier_lut_bytes + point.guard_lut_bytes
        print(
            f"{point.scenario:<14} "
            f"{point.variant:<16} "
            f"{format_bytes(total_lut):>8} "
            f"{fmt_pct(point.accuracy):>9} "
            f"{fmt_pct(point.strict_recall):>9} "
            f"{fmt_pct(point.under_strict_rate):>8} "
            f"{fmt_pct(point.over_strict_rate):>8} "
            f"{fmt_pct(point.strict_rate):>9} "
            f"{point.estimated_touch_per_event:>7.2f} "
            f"{point.downgraded_strict_claims:>6.1f} "
            f"{point.eval_seed_count - point.failures}/{point.eval_seed_count}"
        )
    print()
    print("Interpretation:")
    print("- Training stays on the default paragraph distribution.")
    print("- Eval shifts parser noise, omitted fields, distractors, and claim count.")
    print("- A pass means every eval seed met accuracy, strict-recall, and under-strict gates.")


def main() -> None:
    print_result(run_ca_wiki_cell_paragraph_factorized_guard_stress_sweep())


if __name__ == "__main__":
    main()
