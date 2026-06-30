"""Online candidate-cache benchmark for HARC-CA output shortlists."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.candidate_cache import run_candidate_cache_trial
from cellular_transformer.hardware import format_bytes


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def main() -> None:
    print("Online low-bit candidate cache")
    print("stream: vocab=65536, hot_tokens=256, top_k=64, topic_probability=0.85")
    print("cache: set-associative, routes=2, ways=4, score_bits=4, decay_interval=256")
    print()

    headers = [
        "capacity",
        "state",
        "topk_hit",
        "upd_hit",
        "local_upd",
        "decay",
        "total_upd",
        "resident",
        "replace",
        "vocab_scan",
    ]
    print(" | ".join(f"{h:>11}" for h in headers))
    print("-" * 132)
    for capacity in (128, 256, 512, 1024):
        result = run_candidate_cache_trial(
            context_length=8192,
            warmup_events=1024,
            capacity=capacity,
            seed=17,
        )
        row = [
            f"{result.capacity}",
            format_bytes(result.state_bytes),
            fmt_pct(result.topk_hit_rate),
            fmt_pct(result.cache_update_hit_rate),
            f"{result.avg_local_update_cells:0.1f}",
            f"{result.avg_decay_cells:0.1f}",
            f"{result.avg_total_update_cells:0.1f}",
            f"{result.resident_tokens}",
            f"{result.replacements}",
            f"{result.full_vocab_scan_tokens}",
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))

    print()
    print("Interpretation:")
    print("- The candidate list is formed online from observed tokens; no hot-token oracle is used.")
    print("- Top-k prediction still depends on the stream distribution and is not LLM quality.")
    print("- The chip-facing metric is that updates touch a few local cache cells, not 65k vocab IDs.")


if __name__ == "__main__":
    main()
