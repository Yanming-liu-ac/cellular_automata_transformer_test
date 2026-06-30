"""Output-head budget comparison for HARC-CA."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellular_transformer.hardware import format_bytes
from cellular_transformer.output_head import OutputHeadConfig, compare_output_heads
from cellular_transformer.synthetic_lm import SyntheticLMConfig


def fmt_count(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:0.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:0.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:0.2f}K"
    return f"{value:0.1f}"


def main() -> None:
    lm_config = SyntheticLMConfig()
    topic_fraction = lm_config.topic_events / (lm_config.topic_events + lm_config.query_events)
    output_config = OutputHeadConfig(
        vocab_size=lm_config.vocab_size,
        hidden_channels=128,
        activation_bits=4,
        weight_bits=4,
        logit_bits=16,
    )

    print("Output-head budget proxy")
    print("vocab=65536, hidden=128, activation=4b, weights=4b, logits=16b")
    print(f"synthetic exact-query bypass fraction={1.0 - topic_fraction:0.3f}")
    print()

    headers = [
        "candidates",
        "mode",
        "bytes/event",
        "MACs/event",
        "resident_w",
        "byte_reduct",
        "mac_reduct",
    ]
    print(" | ".join(f"{h:>14}" for h in headers))
    print("-" * 120)
    for candidate_count in (128, 512, 2048, 8192):
        comparison = compare_output_heads(
            output_config,
            candidate_count=candidate_count,
            candidate_event_fraction=topic_fraction,
        )
        rows = [
            (comparison.full, 1.0, 1.0),
            (
                comparison.candidate,
                comparison.candidate_byte_reduction,
                comparison.candidate_mac_reduction,
            ),
            (
                comparison.mixed_candidate_bypass,
                comparison.mixed_byte_reduction,
                comparison.mixed_mac_reduction,
            ),
        ]
        for metrics, byte_reduction, mac_reduction in rows:
            row = [
                f"{candidate_count}",
                metrics.mode,
                format_bytes(metrics.total_bytes_per_event),
                fmt_count(metrics.macs_per_event),
                format_bytes(metrics.resident_weight_bytes),
                f"{byte_reduction:0.1f}x",
                f"{mac_reduction:0.1f}x",
            ]
            print(" | ".join(f"{cell:>14}" for cell in row))
        print("-" * 120)

    print()
    print("Interpretation:")
    print("- Full-vocab projection is a separate bottleneck from KV cache.")
    print("- Candidate heads are only useful if candidate generation is cheap and accurate.")
    print("- Exact associative hits can bypass dense output scoring entirely.")


if __name__ == "__main__":
    main()
