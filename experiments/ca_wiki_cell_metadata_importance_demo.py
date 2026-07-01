"""Metadata-derived importance for CA Wiki Cell provenance repair."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.wiki_memory import (
    CAWikiCellMetadataImportanceResult,
    run_ca_wiki_cell_metadata_importance_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(result: CAWikiCellMetadataImportanceResult) -> None:
    total_lut = result.classifier_lut_bytes + result.repair_lut_bytes
    mode_counts = Counter(entry.chosen_importance for entry in result.entries)
    print("CA Wiki Cell metadata-derived importance")
    print(
        f"claims/eval={result.claim_count}, metadata={result.metadata_bits_per_claim}b/claim, "
        f"classifier_lut={format_bytes(result.classifier_lut_bytes)}, "
        f"repair_lut={format_bytes(result.repair_lut_bytes)}, total_lut={format_bytes(total_lut)}"
    )
    print(
        "metadata buckets: trust/citation/recency/query_freq, "
        "each 2-bit in this synthetic proxy"
    )
    print(
        "bucket outputs: "
        + ", ".join(f"{mode}={mode_counts[mode]}" for mode in result.importance_modes)
    )
    print()
    headers = [
        "seed",
        "accuracy",
        "strict_p",
        "strict_r",
        "under",
        "loose",
        "normal",
        "strict",
        "touch/e",
        "pass",
    ]
    print(" | ".join(f"{header:>11}" for header in headers))
    print("-" * 120)
    for point in result.points:
        row = [
            f"{point.eval_seed}",
            fmt_pct(point.accuracy),
            fmt_pct(point.strict_precision),
            fmt_pct(point.strict_recall),
            fmt_pct(point.under_strict_rate),
            fmt_pct(point.loose_rate),
            fmt_pct(point.normal_rate),
            fmt_pct(point.strict_rate),
            f"{point.estimated_touch_per_event:0.2f}",
            "yes" if point.target_met else "no",
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))
    failures = sum(1 for point in result.points if not point.target_met)
    print()
    print("Interpretation:")
    print("- This is a synthetic local-metadata proxy, not a real wiki dataset result.")
    print("- The classifier maps 2-bit trust/citation/recency/query metadata to importance.")
    print("- Importance then selects the learned provenance repair policy from the previous LUT.")
    print(f"- Target failures: {failures}/{len(result.points)} evaluation rows.")


def main() -> None:
    print_result(run_ca_wiki_cell_metadata_importance_sweep())


if __name__ == "__main__":
    main()
