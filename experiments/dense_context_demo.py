"""Compressed dense-context sketch benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
    trials = [
        run_dense_context_trial(
            context_length=65536,
            vocab_size=65536,
            hot_tokens=256,
            top_k=64,
            width=512,
            seed=5,
        ),
        run_dense_context_trial(
            context_length=65536,
            vocab_size=65536,
            hot_tokens=256,
            top_k=64,
            width=1024,
            seed=5,
        ),
        run_dense_context_trial(
            context_length=65536,
            vocab_size=65536,
            hot_tokens=256,
            top_k=64,
            width=2048,
            seed=5,
        ),
        run_dense_context_trial(
            context_length=65536,
            vocab_size=65536,
            hot_tokens=256,
            top_k=64,
            width=4096,
            seed=5,
        ),
    ]

    headers = [
        "ctx",
        "vocab",
        "width",
        "bits",
        "topk",
        "recall",
        "mae",
        "update",
        "state",
        "exact",
        "compress",
    ]
    print("Low-bit compressed dense-context sketch")
    print("topic stream: vocab=65536, hot_tokens=256, top_k=64, context=65536")
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 132)
    for result in trials:
        row = [
            f"{result.context_length}",
            f"{result.vocab_size}",
            f"{result.width}",
            f"{result.bits}",
            f"{result.top_k}",
            fmt_pct(result.topk_recall),
            f"{result.mean_abs_error:0.3f}",
            f"{result.avg_update_cells:0.1f}",
            fmt_bytes(result.state_bytes),
            fmt_bytes(result.dense_exact_bytes),
            f"{result.compression_ratio:0.2f}x",
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- This is the compressed dense-context path, not exact recall memory.")
    print("- It tracks coarse topic/recency distribution with low-bit local counters.")
    print("- Exact facts should still go through the associative lane.")


if __name__ == "__main__":
    main()
