"""Learned low-bit candidate scorer benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.admission_policy import make_topic_stream
from cellular_transformer.candidate_scorer import (
    run_candidate_scorer_trial,
    train_future_window_candidate_scorer_lut,
    train_repeat_candidate_scorer_lut,
)
from cellular_transformer.dense_context import DenseContextConfig
from cellular_transformer.hardware import format_bytes
from cellular_transformer.synthetic_lm import SyntheticLMConfig, run_synthetic_lm_trial


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def main() -> None:
    current_trial = run_candidate_scorer_trial(
        training_target="current_token_repeat",
        scoring_dense_weight=0,
    )
    window_trial = run_candidate_scorer_trial(
        training_target="future_window",
        scoring_dense_weight=1,
    )
    dense_config = DenseContextConfig(
        vocab_size=65536,
        banks=4,
        width=2048,
        bits=4,
        decay_interval=256,
    )
    train_stream = make_topic_stream(
        length=8192,
        vocab_size=65536,
        hot_tokens=256,
        topic_probability=0.85,
        zipf_exponent=1.15,
        seed=100,
    )
    current_scorer = train_repeat_candidate_scorer_lut(train_stream, dense_config)
    window_scorer = train_future_window_candidate_scorer_lut(train_stream, dense_config)

    baseline_lm = run_synthetic_lm_trial(
        seed=31,
        config=SyntheticLMConfig(
            dense_width=2048,
            candidate_strategy="online_cache",
            candidate_admission_threshold=1,
        ),
    )
    current_lm = run_synthetic_lm_trial(
        seed=31,
        config=SyntheticLMConfig(
            dense_width=2048,
            candidate_strategy="online_cache",
            candidate_admission_threshold=1,
            candidate_scorer_lut=current_scorer.scores,
        ),
    )
    window_lm = run_synthetic_lm_trial(
        seed=31,
        config=SyntheticLMConfig(
            dense_width=2048,
            candidate_strategy="online_cache",
            candidate_admission_threshold=1,
            candidate_scorer_lut=window_scorer.scores,
            candidate_scorer_dense_weight=1,
        ),
    )

    print("Learned low-bit candidate scorer")
    print("training labels: current-token repeat vs future-window teacher")
    print("features: dense-sketch estimate + candidate-cache score")
    print(f"LUT state={format_bytes(window_scorer.state_bytes)} each")
    print()

    headers = [
        "target",
        "baseline",
        "learned",
        "dense_w",
        "cache_w",
        "admit_r",
        "upd_hit",
        "score_cells",
        "replace",
        "scan",
    ]
    print("Standalone candidate scoring")
    print(" | ".join(f"{h:>11}" for h in headers))
    print("-" * 112)
    for trial in (current_trial, window_trial):
        row = [
            trial.training_target,
            fmt_pct(trial.baseline_topk_hit_rate),
            fmt_pct(trial.learned_topk_hit_rate),
            f"{trial.scoring_dense_weight}",
            f"{trial.scoring_cache_weight}",
            fmt_pct(trial.admission_rate),
            fmt_pct(trial.cache_update_hit_rate),
            f"{trial.avg_score_cells:0.1f}",
            f"{trial.replacements}",
            f"{trial.full_vocab_scan_tokens}",
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))

    print()
    print("Synthetic LM scoring")
    headers = ["policy", "scorer", "topic@k", "score_cells", "avg_cells"]
    print(" | ".join(f"{h:>11}" for h in headers))
    print("-" * 72)
    for label, result in (
        ("baseline", baseline_lm),
        ("current", current_lm),
        ("window", window_lm),
    ):
        row = [
            label,
            result.candidate_scorer_mode,
            fmt_pct(result.topic_topk_hit_rate),
            f"{result.candidate_score_cells_per_event:0.1f}",
            f"{result.avg_cells_per_event:0.1f}",
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))

    print()
    print("Interpretation:")
    print("- The current-token LUT remains a negative synthetic-LM result.")
    print("- The future-window residual helps standalone scoring but still fails in mixed context.")
    print("- Two local scalar features are not enough for a CSA-like indexer yet.")
    print("- Candidate scoring reads are now explicitly counted instead of treated as free.")


if __name__ == "__main__":
    main()
