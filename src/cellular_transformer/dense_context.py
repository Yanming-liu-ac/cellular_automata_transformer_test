"""Low-bit compressed dense-context sketches.

DeepSeek-V4's HCA path suggests a second memory path besides exact sparse
retrieval: a compressed recurrent state that tracks dense causal context. This
module implements a small hardware-shaped prototype using low-bit decayed
count-sketch counters.

The sketch is not exact memory. It is meant to preserve coarse context
distribution, topic, and recency while the associative lane stores exact facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .retrieval import keyed_hash


@dataclass(frozen=True)
class DenseContextConfig:
    """Configuration for a low-bit dense-context sketch."""

    vocab_size: int
    banks: int = 4
    width: int = 2048
    bits: int = 4
    decay_interval: int = 256
    decay_shift: int = 1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.banks <= 0:
            raise ValueError("banks must be positive")
        if self.width <= 0:
            raise ValueError("width must be positive")
        if self.bits not in (2, 4, 8):
            raise ValueError("bits must be one of 2, 4, 8")
        if self.decay_interval <= 0:
            raise ValueError("decay_interval must be positive")
        if self.decay_shift <= 0:
            raise ValueError("decay_shift must be positive")

    @property
    def max_value(self) -> int:
        return (1 << self.bits) - 1

    @property
    def state_bytes(self) -> float:
        return self.banks * self.width * self.bits / 8


@dataclass(frozen=True)
class DenseContextResult:
    """Aggregate dense-context benchmark metrics."""

    context_length: int
    vocab_size: int
    hot_tokens: int
    top_k: int
    banks: int
    width: int
    bits: int
    state_bytes: float
    topk_recall: float
    topk_precision: float
    mean_abs_error: float
    avg_update_cells: float
    dense_exact_bytes: float

    @property
    def compression_ratio(self) -> float:
        if self.state_bytes == 0.0:
            return 0.0
        return self.dense_exact_bytes / self.state_bytes


class LowBitDenseContext:
    """Decayed low-bit count-sketch for compressed context state."""

    def __init__(self, config: DenseContextConfig) -> None:
        self.config = config
        self.counters = np.zeros((config.banks, config.width), dtype=np.uint8)
        self.steps = 0

    def _slots(self, token: int) -> List[int]:
        return [
            keyed_hash(int(token), 1000 + bank) % self.config.width
            for bank in range(self.config.banks)
        ]

    def update(self, token: int) -> int:
        """Update sketch with one token and return touched counter count."""

        if not 0 <= int(token) < self.config.vocab_size:
            raise ValueError("token outside vocab")

        touched = 0
        for bank, slot in enumerate(self._slots(token)):
            value = int(self.counters[bank, slot])
            if value < self.config.max_value:
                self.counters[bank, slot] = value + 1
            touched += 1

        self.steps += 1
        if self.steps % self.config.decay_interval == 0:
            self.decay()
        return touched

    def decay(self) -> None:
        """Decay all counters in place with an integer right shift."""

        self.counters >>= self.config.decay_shift

    def estimate(self, token: int) -> int:
        """Estimate one token's decayed count using a count-min readout."""

        slots = self._slots(token)
        return min(int(self.counters[bank, slot]) for bank, slot in enumerate(slots))

    def estimate_all(self) -> np.ndarray:
        """Estimate all vocabulary tokens."""

        estimates = np.zeros(self.config.vocab_size, dtype=np.uint16)
        for token in range(self.config.vocab_size):
            estimates[token] = self.estimate(token)
        return estimates

    def memory_bytes(self) -> float:
        return self.config.state_bytes


def make_topic_stream(
    context_length: int,
    vocab_size: int,
    hot_tokens: int,
    hot_probability: float = 0.75,
    seed: int = 0,
) -> np.ndarray:
    """Generate a stream with dense topic bias plus background noise."""

    if not 0 < hot_tokens <= vocab_size:
        raise ValueError("hot_tokens must be in (0, vocab_size]")
    if not 0.0 <= hot_probability <= 1.0:
        raise ValueError("hot_probability must be in [0, 1]")

    rng = np.random.default_rng(seed)
    stream = np.empty(context_length, dtype=np.int32)
    hot_set = np.arange(hot_tokens, dtype=np.int32)
    cold_set = np.arange(hot_tokens, vocab_size, dtype=np.int32)
    for i in range(context_length):
        if rng.random() < hot_probability or len(cold_set) == 0:
            stream[i] = int(rng.choice(hot_set))
        else:
            stream[i] = int(rng.choice(cold_set))
    return stream


def exact_decayed_counts(stream: Iterable[int], config: DenseContextConfig) -> np.ndarray:
    """Exact dense decayed counts for comparison."""

    counts = np.zeros(config.vocab_size, dtype=np.uint16)
    for step, token in enumerate(stream, start=1):
        if counts[int(token)] < config.max_value:
            counts[int(token)] += 1
        if step % config.decay_interval == 0:
            counts >>= config.decay_shift
    return counts


def run_dense_context_trial(
    context_length: int = 16384,
    vocab_size: int = 4096,
    hot_tokens: int = 64,
    top_k: int = 32,
    banks: int = 4,
    width: int = 2048,
    bits: int = 4,
    decay_interval: int = 256,
    seed: int = 0,
) -> DenseContextResult:
    """Evaluate a low-bit dense-context sketch on a topic-biased stream."""

    config = DenseContextConfig(
        vocab_size=vocab_size,
        banks=banks,
        width=width,
        bits=bits,
        decay_interval=decay_interval,
    )
    stream = make_topic_stream(context_length, vocab_size, hot_tokens, seed=seed)
    sketch = LowBitDenseContext(config)

    touched = 0
    for token in stream:
        touched += sketch.update(int(token))

    exact = exact_decayed_counts(stream, config)
    estimated = sketch.estimate_all()

    exact_top = set(np.argsort(exact)[-top_k:].tolist())
    estimated_top = set(np.argsort(estimated)[-top_k:].tolist())
    overlap = len(exact_top & estimated_top)

    mean_abs_error = float(np.mean(np.abs(exact.astype(np.int32) - estimated.astype(np.int32))))
    dense_exact_bytes = vocab_size * bits / 8

    return DenseContextResult(
        context_length=context_length,
        vocab_size=vocab_size,
        hot_tokens=hot_tokens,
        top_k=top_k,
        banks=banks,
        width=width,
        bits=bits,
        state_bytes=sketch.memory_bytes(),
        topk_recall=overlap / top_k,
        topk_precision=overlap / top_k,
        mean_abs_error=mean_abs_error,
        avg_update_cells=touched / context_length if context_length else 0.0,
        dense_exact_bytes=dense_exact_bytes,
    )
