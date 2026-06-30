"""Compare single-lane and tiered overflow associative CA memory."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.benchmarks import run_memory_task


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
    context_length = 16384
    primary_buckets = context_length // 4
    tag_bits = 32
    tasks = ("copy", "induction", "key_value")

    rows = []
    for task in tasks:
        rows.append(
            run_memory_task(
                task=task,
                context_length=context_length,
                buckets=primary_buckets,
                ways=4,
                routes=2,
                tag_bits=tag_bits,
                query_count=None,
                seed=23,
            )
        )
        rows.append(
            run_memory_task(
                task=task,
                context_length=context_length,
                buckets=primary_buckets,
                ways=4,
                routes=2,
                tag_bits=tag_bits,
                overflow_bucket_multiplier=1 / 16,
                overflow_ways=4,
                overflow_routes=2,
                query_count=None,
                seed=23,
            )
        )

    headers = [
        "task",
        "memory",
        "correct",
        "visited",
        "overflow_q",
        "scan_x",
        "primary_evict",
        "overflow_evict",
        "mem",
    ]
    print("Overflow-tier benchmark at 16k context")
    print("primary: buckets=context/4, ways=4, routes=2")
    print("overflow: buckets=context/16, ways=4, routes=2")
    print(f"tag_bits={tag_bits}, queries=full context")
    print(" | ".join(f"{h:>14}" for h in headers))
    print("-" * 135)
    for row in rows:
        cells = [
            row.task,
            row.memory,
            fmt_pct(row.correct_rate),
            f"{row.avg_visited_cells:0.1f}",
            fmt_pct(row.overflow_query_rate),
            f"{row.scan_avoidance_ratio:0.1f}x",
            f"{row.evictions}",
            f"{row.overflow_evictions}",
            fmt_bytes(row.memory_bytes),
        ]
        print(" | ".join(f"{cell:>14}" for cell in cells))

    print()
    print("Interpretation:")
    print("- Tiered memory keeps the same primary lane and adds a smaller overflow lane.")
    print("- Queries only touch overflow after a primary miss or tag collision.")
    print("- This tests a CA-native cache hierarchy, not a full-context scan fallback.")


if __name__ == "__main__":
    main()
