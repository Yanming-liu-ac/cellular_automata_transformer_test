"""Compare local CA and HARC-CA propagation depth."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.propagation import summarize_lengths


def main() -> None:
    rows = summarize_lengths([8, 16, 32, 64, 128, 256, 512, 1024])
    headers = [
        "length",
        "line_steps",
        "harc_steps",
        "line_nodes",
        "harc_nodes",
        "line_edges",
        "harc_edges",
    ]

    print("Propagation from newest token to farthest previous token")
    print(" | ".join(f"{h:>10}" for h in headers))
    print("-" * 88)
    for row in rows:
        print(" | ".join(f"{row[h]:>10}" for h in headers))

    print()
    print("Interpretation:")
    print("- line_steps is the lower-bound pain point of a plain radius-1 CA.")
    print("- harc_steps is the multiscale routing depth that HARC-CA targets in hardware.")


if __name__ == "__main__":
    main()
