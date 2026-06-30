"""Output-head cost proxies for HARC-CA.

A CA-first memory system can still lose if every generated token requires a
full-vocabulary dense projection. This module compares full-vocab scoring with
candidate-shortlist scoring and exact-recall bypass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputHeadConfig:
    """Configuration for an output-head proxy."""

    vocab_size: int = 65536
    hidden_channels: int = 128
    activation_bits: int = 4
    weight_bits: int = 4
    logit_bits: int = 16

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if self.activation_bits <= 0:
            raise ValueError("activation_bits must be positive")
        if self.weight_bits <= 0:
            raise ValueError("weight_bits must be positive")
        if self.logit_bits <= 0:
            raise ValueError("logit_bits must be positive")


@dataclass(frozen=True)
class OutputHeadMetrics:
    """Per-event output-head proxy metrics."""

    mode: str
    scored_tokens: int
    event_fraction: float
    activation_bytes_per_event: float
    weight_bytes_per_event: float
    logit_bytes_per_event: float
    total_bytes_per_event: float
    dot_products_per_event: float
    macs_per_event: float
    resident_weight_bytes: float


@dataclass(frozen=True)
class OutputHeadComparison:
    """Full output head versus candidate output head."""

    full: OutputHeadMetrics
    candidate: OutputHeadMetrics
    mixed_candidate_bypass: OutputHeadMetrics

    @property
    def candidate_byte_reduction(self) -> float:
        return self.full.total_bytes_per_event / self.candidate.total_bytes_per_event

    @property
    def mixed_byte_reduction(self) -> float:
        return self.full.total_bytes_per_event / self.mixed_candidate_bypass.total_bytes_per_event

    @property
    def candidate_mac_reduction(self) -> float:
        return self.full.macs_per_event / self.candidate.macs_per_event

    @property
    def mixed_mac_reduction(self) -> float:
        return self.full.macs_per_event / self.mixed_candidate_bypass.macs_per_event


def estimate_output_head(
    config: OutputHeadConfig,
    scored_tokens: int,
    mode: str,
    event_fraction: float = 1.0,
) -> OutputHeadMetrics:
    """Estimate output-head traffic and MACs for a scored token set."""

    if not 0 <= scored_tokens <= config.vocab_size:
        raise ValueError("scored_tokens must be in [0, vocab_size]")
    if not 0.0 <= event_fraction <= 1.0:
        raise ValueError("event_fraction must be in [0, 1]")

    activation_bytes = config.hidden_channels * config.activation_bits / 8
    weight_bytes = scored_tokens * config.hidden_channels * config.weight_bits / 8
    logit_bytes = scored_tokens * config.logit_bits / 8
    resident_weight_bytes = config.vocab_size * config.hidden_channels * config.weight_bits / 8

    total = (activation_bytes + weight_bytes + logit_bytes) * event_fraction
    dots = scored_tokens * event_fraction
    macs = scored_tokens * config.hidden_channels * event_fraction

    return OutputHeadMetrics(
        mode=mode,
        scored_tokens=scored_tokens,
        event_fraction=event_fraction,
        activation_bytes_per_event=activation_bytes * event_fraction,
        weight_bytes_per_event=weight_bytes * event_fraction,
        logit_bytes_per_event=logit_bytes * event_fraction,
        total_bytes_per_event=total,
        dot_products_per_event=dots,
        macs_per_event=macs,
        resident_weight_bytes=resident_weight_bytes,
    )


def compare_output_heads(
    config: OutputHeadConfig,
    candidate_count: int = 512,
    candidate_event_fraction: float = 1.0,
) -> OutputHeadComparison:
    """Compare full-vocabulary, candidate, and mixed candidate+bypass heads."""

    if not 0 <= candidate_count <= config.vocab_size:
        raise ValueError("candidate_count must be in [0, vocab_size]")

    full = estimate_output_head(
        config,
        scored_tokens=config.vocab_size,
        mode="full_vocab",
        event_fraction=1.0,
    )
    candidate = estimate_output_head(
        config,
        scored_tokens=candidate_count,
        mode="candidate_every_event",
        event_fraction=1.0,
    )
    mixed = estimate_output_head(
        config,
        scored_tokens=candidate_count,
        mode="candidate_with_exact_bypass",
        event_fraction=candidate_event_fraction,
    )
    return OutputHeadComparison(full=full, candidate=candidate, mixed_candidate_bypass=mixed)
