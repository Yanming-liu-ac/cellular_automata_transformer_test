"""Trace-derived metadata importance audit for CA Wiki Cell repair."""

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
    CAWikiCellTraceMetadataImportanceResult,
    run_ca_wiki_cell_trace_metadata_importance_sweep,
)


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def print_result(result: CAWikiCellTraceMetadataImportanceResult) -> None:
    total_lut = result.classifier_lut_bytes + result.repair_lut_bytes
    mode_counts = Counter(entry.chosen_importance for entry in result.entries)
    print("CA Wiki Cell trace metadata importance audit")
    print(
        f"claims/eval={result.claim_count}, train_claims={result.train_claim_count}, "
        f"query_events={result.query_events}, update_events={result.update_events}, "
        f"metadata_bits/claim={result.metadata_bits_per_claim}, "
        f"pressure_bits/claim={result.local_pressure_bits_per_claim}, "
        f"classifier_lut={format_bytes(result.classifier_lut_bytes)}, "
        f"repair_lut={format_bytes(result.repair_lut_bytes)}, total_lut={format_bytes(total_lut)}"
    )
    print(
        f"loss weights: under={result.under_importance_weight:0.2f}, "
        f"over={result.over_importance_weight:0.2f}"
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
        "over",
        "loose",
        "normal",
        "strict",
        "q/claim",
        "u/claim",
        "stale/claim",
        "touch/e",
        "pass",
    ]
    print(" | ".join(f"{header:>11}" for header in headers))
    print("-" * 172)
    for point in result.points:
        row = [
            f"{point.eval_seed}",
            fmt_pct(point.accuracy),
            fmt_pct(point.strict_precision),
            fmt_pct(point.strict_recall),
            fmt_pct(point.under_strict_rate),
            fmt_pct(point.over_strict_rate),
            fmt_pct(point.loose_rate),
            fmt_pct(point.normal_rate),
            fmt_pct(point.strict_rate),
            f"{point.mean_queries_per_claim:0.2f}",
            f"{point.mean_updates_per_claim:0.2f}",
            f"{point.mean_stale_probes_per_claim:0.2f}",
            f"{point.estimated_touch_per_event:0.2f}",
            "yes" if point.target_met else "no",
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))
    failures = sum(1 for point in result.points if not point.target_met)
    print()
    print("Interpretation:")
    print("- Labels come from a query/update/stale-probe trace, not a hand-written teacher.")
    print("- The LUT reads a 2-bit local pressure bucket derived from CA-maintained counters.")
    print("- The asymmetric loss protects pages whose trace pressure calls for strict repair.")
    print(f"- Target failures: {failures}/{len(result.points)} evaluation rows.")


def main() -> None:
    print_result(run_ca_wiki_cell_trace_metadata_importance_sweep())


if __name__ == "__main__":
    main()
