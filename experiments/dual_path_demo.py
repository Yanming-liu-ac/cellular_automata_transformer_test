"""Combined exact sparse memory plus compressed dense-context demo."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.benchmarks import run_memory_task
from cellular_transformer.dense_context import run_dense_context_trial


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
    exact = run_memory_task(
        task="induction",
        context_length=16384,
        buckets=4096,
        ways=4,
        routes=2,
        tag_bits=32,
        overflow_bucket_multiplier=1 / 16,
        overflow_ways=4,
        overflow_routes=2,
        query_count=None,
        seed=23,
    )
    dense = run_dense_context_trial(
        context_length=65536,
        vocab_size=65536,
        hot_tokens=256,
        top_k=64,
        width=2048,
        bits=4,
        seed=5,
    )
    total_memory = exact.memory_bytes + dense.state_bytes

    print("HARC-CA dual-path memory prototype")
    print()
    print("Exact sparse path:")
    print(
        f"  task=induction recall={fmt_pct(exact.correct_rate)} "
        f"visited={exact.avg_visited_cells:0.1f} overflow_q={fmt_pct(exact.overflow_query_rate)} "
        f"memory={fmt_bytes(exact.memory_bytes)}"
    )
    print("Compressed dense path:")
    print(
        f"  topk_recall={fmt_pct(dense.topk_recall)} update_cells={dense.avg_update_cells:0.1f} "
        f"state={fmt_bytes(dense.state_bytes)} exact_counter_table={fmt_bytes(dense.dense_exact_bytes)}"
    )
    print("Combined:")
    print(f"  memory={fmt_bytes(total_memory)}")
    print()
    print("Interpretation:")
    print("- Sparse lane handles exact rare facts and induction-style recall.")
    print("- Dense sketch handles coarse topic/recency context with low-bit counters.")
    print("- This is the CA analog of DeepSeek-V4's sparse/dense memory split.")


if __name__ == "__main__":
    main()
