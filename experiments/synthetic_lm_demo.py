"""Synthetic next-token benchmark for the HARC-CA dual-path prototype."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.synthetic_lm import SyntheticLMConfig, run_synthetic_lm_trial


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:6.2f}%"


def fmt_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    for unit in units:
        if size < 1024:
            return f"{size:6.2f} {unit}"
        size /= 1024
    return f"{size:6.2f} TB"


def main() -> None:
    configs = [
        SyntheticLMConfig(dense_width=1024),
        SyntheticLMConfig(dense_width=2048),
        SyntheticLMConfig(dense_width=4096),
    ]

    headers = [
        "dense_w",
        "induct",
        "topic@k",
        "exact_vis",
        "overflow_q",
        "dense_upd",
        "avg_cells",
        "memory",
    ]
    print("HARC-CA synthetic next-token benchmark")
    print("exact task: key -> next value; dense task: topic token in top-k candidate shortlist")
    print(" | ".join(f"{h:>11}" for h in headers))
    print("-" * 112)
    for config in configs:
        result = run_synthetic_lm_trial(seed=31, config=config)
        row = [
            f"{config.dense_width}",
            fmt_pct(result.induction_accuracy),
            fmt_pct(result.topic_topk_hit_rate),
            f"{result.exact_avg_visited_cells:0.1f}",
            fmt_pct(result.overflow_query_rate),
            f"{result.dense_update_cells_per_event:0.1f}",
            f"{result.avg_cells_per_event:0.1f}",
            fmt_bytes(result.total_memory_bytes),
        ]
        print(" | ".join(f"{cell:>11}" for cell in row))

    print()
    print("Interpretation:")
    print("- Induction uses the exact sparse associative lane.")
    print("- Topic@k uses the compressed dense-context sketch over a candidate pool.")
    print("- This is a non-trained inference skeleton, not an LLM quality result.")


if __name__ == "__main__":
    main()
