"""Learned low-bit admission policy for candidate-cache writes."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.admission_policy import run_learned_admission_trial
from cellular_transformer.candidate_cache import run_candidate_cache_trial
from cellular_transformer.hardware import format_bytes
from cellular_transformer.synthetic_lm import SyntheticLMConfig, run_synthetic_lm_trial


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def main() -> None:
    learned = run_learned_admission_trial(
        train_length=8192,
        eval_length=8192,
        warmup_events=1024,
        future_horizon=256,
    )

    threshold = run_candidate_cache_trial(
        context_length=8192,
        warmup_events=1024,
        capacity=512,
        admission_threshold=1,
        seed=17,
    )

    static_lm = run_synthetic_lm_trial(seed=31, config=SyntheticLMConfig(dense_width=2048))
    threshold_lm = run_synthetic_lm_trial(
        seed=31,
        config=SyntheticLMConfig(
            dense_width=2048,
            candidate_strategy="online_cache",
            candidate_admission_threshold=1,
        ),
    )
    learned_lm = run_synthetic_lm_trial(
        seed=31,
        config=SyntheticLMConfig(
            dense_width=2048,
            candidate_strategy="online_cache",
            candidate_admission_lut=learned.scores,
        ),
    )

    print("Learned low-bit admission LUT")
    print("training label: token repeats within future_horizon=256")
    print("inference features: dense-sketch estimate only")
    print(f"LUT scores={learned.scores}")
    print(f"LUT state={format_bytes(learned.lut_state_bytes)}")
    print()

    headers = [
        "policy",
        "topk_hit",
        "admit_r",
        "precision",
        "recall",
        "upd_hit",
        "gate",
        "cache",
        "replace",
        "scan",
    ]
    print("Standalone candidate-cache admission")
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 126)
    rows = [
        (
            "threshold",
            threshold.topk_hit_rate,
            threshold.admission_rate,
            0.0,
            0.0,
            threshold.cache_update_hit_rate,
            threshold.avg_gate_cells,
            threshold.avg_local_update_cells + threshold.avg_decay_cells,
            threshold.replacements,
            threshold.full_vocab_scan_tokens,
        ),
        (
            "learned",
            learned.topk_hit_rate,
            learned.admission_rate,
            learned.admission_precision,
            learned.admission_recall,
            learned.cache_update_hit_rate,
            learned.avg_gate_cells,
            learned.avg_cache_cells,
            learned.replacements,
            learned.full_vocab_scan_tokens,
        ),
    ]
    for row in rows:
        label, topk, admit, precision, recall, hit, gate, cache, replacements, scans = row
        cells = [
            label,
            fmt_pct(topk),
            fmt_pct(admit),
            "-" if precision == 0.0 and recall == 0.0 and label == "threshold" else fmt_pct(precision),
            "-" if precision == 0.0 and recall == 0.0 and label == "threshold" else fmt_pct(recall),
            fmt_pct(hit),
            f"{gate:0.1f}",
            f"{cache:0.1f}",
            f"{replacements}",
            f"{scans}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in cells))

    print()
    print("Synthetic LM candidate admission")
    headers = ["policy", "mode", "topic@k", "admit_r", "cand_hit", "cand_upd", "gate_upd"]
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 94)
    for label, result in (
        ("static", static_lm),
        ("threshold", threshold_lm),
        ("learned", learned_lm),
    ):
        row = [
            label,
            result.candidate_admission_mode,
            fmt_pct(result.topic_topk_hit_rate),
            "-" if label == "static" else fmt_pct(result.candidate_admission_rate),
            "-" if label == "static" else fmt_pct(result.candidate_cache_hit_rate),
            f"{result.candidate_update_cells_per_event:0.1f}",
            f"{result.candidate_gate_cells_per_event:0.1f}",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- The learned LUT recovers the threshold gate from self-supervised repeat labels.")
    print("- It is still a tiny synthetic policy, not a trained language model router.")
    print("- The deployment path is a few signed low-bit LUT entries plus dense-sketch reads.")


if __name__ == "__main__":
    main()
