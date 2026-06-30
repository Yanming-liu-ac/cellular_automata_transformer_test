"""Copy, induction, and key/value benchmarks for associative CA retrieval."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.benchmarks import sweep_memory_tasks


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
    results = sweep_memory_tasks(
        tasks=("copy", "induction", "key_value"),
        lengths=(1024, 4096, 16384),
        routes_options=(1, 2),
        bucket_multiplier=0.25,
        ways=4,
        tag_bits=24,
        query_count=1000,
        seed=23,
    )

    headers = [
        "task",
        "ctx",
        "routes",
        "load",
        "evict",
        "correct",
        "false+",
        "visited",
        "scan_x",
        "mem",
    ]
    print("Sequence-memory benchmarks at fixed capacity: buckets=context/4, ways=4")
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 122)
    for result in results:
        row = [
            result.task,
            f"{result.context_length}",
            f"{result.routes}",
            f"{result.load_factor:0.2f}",
            f"{result.evictions}",
            fmt_pct(result.correct_rate),
            fmt_pct(result.false_positive_rate),
            f"{result.avg_visited_cells:0.1f}",
            f"{result.scan_avoidance_ratio:0.1f}x",
            fmt_bytes(result.memory_bytes),
        ]
        print(" | ".join(f"{cell:>10}" for cell in row))

    print()
    print("Interpretation:")
    print("- routes=2 keeps the same stored capacity but checks two hash-routed buckets.")
    print("- scan_x is full-context scan cells divided by visited route/bucket cells.")
    print("- These are retrieval primitives, not trained language-model results.")


if __name__ == "__main__":
    main()
