"""Trainable multi-feature candidate indexer benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.candidate_indexer import run_candidate_indexer_trial
from cellular_transformer.hardware import format_bytes


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def main() -> None:
    print("Trainable low-bit candidate indexer")
    print("features: dense, topic-phase, candidate-cache, contamination")
    print()
    headers = [
        "case",
        "weights",
        "state",
        "resident",
        "ceiling",
        "unique",
        "bucket",
        "dense",
        "topic",
        "topic_cache",
        "additive",
        "learned",
        "add_state",
        "score_rd",
        "score_wr",
    ]
    print(" | ".join(f"{h:>13}" for h in headers))
    print("-" * 142)
    for label, admission_threshold in (("online", 0), ("gated", 1)):
        result = run_candidate_indexer_trial(admission_threshold=admission_threshold)
        row = [
            label,
            str(result.weights),
            format_bytes(result.state_bytes),
            fmt_pct(result.resident_hit_rate),
            fmt_pct(result.feature_ceiling_hit_rate),
            fmt_pct(result.positive_unique_rate),
            f"{result.mean_positive_bucket_size:0.1f}",
            fmt_pct(result.dense_hit_rate),
            fmt_pct(result.topic_hit_rate),
            fmt_pct(result.topic_cache_hit_rate),
            fmt_pct(result.additive_hit_rate),
            fmt_pct(result.learned_hit_rate),
            format_bytes(result.additive_state_bytes),
            f"{result.learned_score_cells_per_event:0.1f}",
            f"{result.topic_score_update_cells_per_event:0.1f}",
        ]
        print(" | ".join(f"{cell:>13}" for cell in row))

    print()
    print("Interpretation:")
    print("- The learned rules are tiny, but they do not beat the best hand formula yet.")
    print("- Topic/cache/source features are useful; the learner and objective need work.")
    print("- Ceiling estimates the best possible top-k hit from these features if learned perfectly.")
    print("- The current strongest synthetic path remains admission-gated dense scoring.")


if __name__ == "__main__":
    main()
