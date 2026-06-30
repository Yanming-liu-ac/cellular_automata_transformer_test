"""Induction-style recall experiment for the HARC-CA associative lane."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.retrieval import sweep_recall_trials


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
    results = sweep_recall_trials(
        lengths=[1024, 4096, 16384],
        bucket_multipliers=[0.25, 0.5, 1.0],
        ways=4,
        tag_bits=24,
        query_count=1000,
        seed=11,
    )

    headers = [
        "ctx",
        "buckets",
        "load",
        "evict",
        "correct",
        "false+",
        "visited",
        "scan",
        "mem",
    ]
    print("Hash-routed associative CA lane: induction-style key/value recall")
    print(" | ".join(f"{h:>9}" for h in headers))
    print("-" * 104)
    for r in results:
        row = [
            f"{r.context_length}",
            f"{r.buckets}",
            f"{r.load_factor:0.2f}",
            f"{r.evictions}",
            fmt_pct(r.correct_rate),
            fmt_pct(r.false_positive_rate),
            f"{r.avg_visited_cells:0.1f}",
            f"{r.full_scan_cells}",
            fmt_bytes(r.memory_bytes),
        ]
        print(" | ".join(f"{cell:>9}" for cell in row))

    print()
    print("Interpretation:")
    print("- correct is exact key->value recall for random induction pairs.")
    print("- visited is route depth plus bucket ways, not a full context scan.")
    print("- evictions show when set-associative capacity is too tight.")


if __name__ == "__main__":
    main()
