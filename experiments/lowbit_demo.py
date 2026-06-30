"""Smoke-test the integer-only low-bit CA core."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.lowbit_ca import LowBitCA, LowBitConfig


def main() -> None:
    ca = LowBitCA(LowBitConfig(length=32, channels=16, bits=4), seed=7)
    ca.inject_byte(0, ord("C"))
    ca.inject_byte(31, ord("A"), offset=8)

    print("Integer-only low-bit CA smoke test")
    print(f"initial active_fraction={ca.active_fraction():.4f} checksum={ca.checksum()}")
    for step in range(1, 17):
        ca.step()
        if step in (1, 2, 4, 8, 16):
            left = ca.read_byte(0)
            right = ca.read_byte(31, offset=8)
            print(
                f"step={step:02d} active_fraction={ca.active_fraction():.4f} "
                f"left_byte={left:03d} right_byte={right:03d} checksum={ca.checksum()}"
            )


if __name__ == "__main__":
    main()
