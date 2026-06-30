"""Compare HARC-CA local traffic with Transformer KV-cache traffic."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import (
    estimate_harc_ca,
    estimate_transformer_kv,
    format_bytes,
)


def main() -> None:
    print("Hardware proxy comparison")
    print()

    for length in (1024, 4096, 16384):
        full = estimate_harc_ca(length, channels=128, bits_per_channel=4, active_fraction=1.0)
        sparse = estimate_harc_ca(length, channels=128, bits_per_channel=4, active_fraction=0.25)
        transformer = estimate_transformer_kv(length, layers=12, heads=8, head_dim=64, kv_bits=16)

        print(f"context_length={length}")
        print(
            "  HARC-CA: "
            f"nodes={full.nodes}, edges={full.edges}, ticks={full.ticks_per_token}, "
            f"state={format_bytes(full.state_bytes)}"
        )
        print(
            "  HARC-CA local messages/token: "
            f"full_active={format_bytes(full.local_message_bytes_per_token)}, "
            f"25pct_active={format_bytes(sparse.local_message_bytes_per_token)}"
        )
        print(
            "  Tiny Transformer KV: "
            f"cache={format_bytes(transformer.kv_cache_bytes)}, "
            f"read/token={format_bytes(transformer.kv_read_bytes_per_token)}"
        )
        print()

    print("Note: HARC-CA traffic is local on-chip message movement.")
    print("Transformer KV traffic is cache read volume before accounting for memory hierarchy.")


if __name__ == "__main__":
    main()
