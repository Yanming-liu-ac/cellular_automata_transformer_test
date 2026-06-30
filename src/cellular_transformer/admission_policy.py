"""Trainable low-bit admission policies for candidate-cache writes.

The fixed threshold gate is a useful baseline, but a CA chip needs admission
rules that can be learned from sequence statistics. This module trains a tiny
integer LUT from a self-supervised repeat label:

```
admit token if it is likely to reappear within a future horizon
```

At inference the LUT only reads the local dense-sketch estimate. It does not use
the future label or token identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .candidate_cache import CandidateCacheConfig, LowBitCandidateCache, sample_zipf_topic_token
from .dense_context import DenseContextConfig, LowBitDenseContext


@dataclass(frozen=True)
class LowBitAdmissionLUT:
    """Signed low-bit LUT indexed by dense-context estimate."""

    scores: Tuple[int, ...]
    score_bits: int = 4

    def __post_init__(self) -> None:
        if len(self.scores) == 0:
            raise ValueError("scores must not be empty")
        if self.score_bits not in (2, 4, 8):
            raise ValueError("score_bits must be one of 2, 4, 8")
        min_score = -(1 << (self.score_bits - 1))
        max_score = (1 << (self.score_bits - 1)) - 1
        for score in self.scores:
            if not min_score <= int(score) <= max_score:
                raise ValueError("score outside signed score_bits range")

    def score(self, estimate: int) -> int:
        index = min(max(int(estimate), 0), len(self.scores) - 1)
        return int(self.scores[index])

    def admit(self, estimate: int) -> bool:
        return self.score(estimate) >= 0

    @property
    def state_bytes(self) -> float:
        return len(self.scores) * self.score_bits / 8


@dataclass(frozen=True)
class LearnedAdmissionResult:
    """Aggregate metrics for a learned admission LUT trial."""

    train_length: int
    eval_length: int
    warmup_events: int
    vocab_size: int
    hot_tokens: int
    top_k: int
    future_horizon: int
    scores: Tuple[int, ...]
    lut_state_bytes: float
    topk_hit_rate: float
    admission_rate: float
    admission_precision: float
    admission_recall: float
    cache_update_hit_rate: float
    avg_gate_cells: float
    avg_cache_cells: float
    avg_total_cells: float
    replacements: int
    resident_tokens: int
    full_vocab_scan_tokens: int = 0


def make_topic_stream(
    length: int,
    vocab_size: int,
    hot_tokens: int,
    topic_probability: float,
    zipf_exponent: float,
    seed: int,
) -> np.ndarray:
    """Generate a deterministic topic/noise stream."""

    if length <= 0:
        raise ValueError("length must be positive")

    rng = np.random.default_rng(seed)
    return np.array(
        [
            sample_zipf_topic_token(
                vocab_size=vocab_size,
                hot_tokens=hot_tokens,
                topic_probability=topic_probability,
                zipf_exponent=zipf_exponent,
                rng=rng,
            )
            for _ in range(length)
        ],
        dtype=np.int32,
    )


def future_repeat_labels(stream: Iterable[int], horizon: int) -> np.ndarray:
    """Label each position by whether the same token reappears soon."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")

    tokens = [int(token) for token in stream]
    labels = np.zeros(len(tokens), dtype=bool)
    next_position: dict[int, int] = {}
    for index in range(len(tokens) - 1, -1, -1):
        token = tokens[index]
        if token in next_position and next_position[token] - index <= horizon:
            labels[index] = True
        next_position[token] = index
    return labels


def train_repeat_admission_lut(
    stream: Iterable[int],
    dense_config: DenseContextConfig,
    future_horizon: int = 256,
    score_bits: int = 4,
) -> LowBitAdmissionLUT:
    """Train a low-bit LUT from future-repeat labels."""

    tokens = [int(token) for token in stream]
    labels = future_repeat_labels(tokens, future_horizon)
    bins = 1 << dense_config.bits
    positive = np.zeros(bins, dtype=np.int32)
    negative = np.zeros(bins, dtype=np.int32)
    sketch = LowBitDenseContext(dense_config)

    for token, label in zip(tokens, labels):
        estimate = min(sketch.estimate(token), bins - 1)
        if bool(label):
            positive[estimate] += 1
        else:
            negative[estimate] += 1
        sketch.update(token)

    min_score = -(1 << (score_bits - 1))
    max_score = (1 << (score_bits - 1)) - 1
    scores = np.clip(positive - negative, min_score, max_score)
    return LowBitAdmissionLUT(tuple(int(score) for score in scores), score_bits=score_bits)


def run_learned_admission_trial(
    train_length: int = 8192,
    eval_length: int = 8192,
    warmup_events: int = 1024,
    vocab_size: int = 65536,
    hot_tokens: int = 256,
    top_k: int = 64,
    future_horizon: int = 256,
    capacity: int = 512,
    ways: int = 4,
    routes: int = 2,
    score_bits: int = 4,
    dense_banks: int = 4,
    dense_width: int = 2048,
    dense_bits: int = 4,
    dense_decay_interval: int = 256,
    topic_probability: float = 0.85,
    zipf_exponent: float = 1.15,
    train_seed: int = 100,
    eval_seed: int = 17,
) -> LearnedAdmissionResult:
    """Train a repeat-prediction LUT and evaluate candidate-cache quality."""

    if not 0 <= warmup_events < eval_length:
        raise ValueError("warmup_events must be in [0, eval_length)")

    dense_config = DenseContextConfig(
        vocab_size=vocab_size,
        banks=dense_banks,
        width=dense_width,
        bits=dense_bits,
        decay_interval=dense_decay_interval,
    )
    train_stream = make_topic_stream(
        length=train_length,
        vocab_size=vocab_size,
        hot_tokens=hot_tokens,
        topic_probability=topic_probability,
        zipf_exponent=zipf_exponent,
        seed=train_seed,
    )
    lut = train_repeat_admission_lut(
        train_stream,
        dense_config=dense_config,
        future_horizon=future_horizon,
        score_bits=score_bits,
    )

    eval_stream = make_topic_stream(
        length=eval_length,
        vocab_size=vocab_size,
        hot_tokens=hot_tokens,
        topic_probability=topic_probability,
        zipf_exponent=zipf_exponent,
        seed=eval_seed,
    )
    repeat_labels = future_repeat_labels(eval_stream, future_horizon)

    cache = LowBitCandidateCache(
        CandidateCacheConfig(
            vocab_size=vocab_size,
            capacity=capacity,
            ways=ways,
            routes=routes,
            score_bits=score_bits,
        )
    )
    sketch = LowBitDenseContext(dense_config)

    hits = 0
    eval_events = 0
    admitted = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    gate_cells = 0

    for step, token in enumerate(eval_stream):
        token = int(token)
        gate_cells += dense_config.banks
        admit = lut.admit(sketch.estimate(token))
        label = bool(repeat_labels[step])

        if step >= warmup_events:
            eval_events += 1
            hits += int(token in cache.topk_set(top_k))
            true_positive += int(admit and label)
            false_positive += int(admit and not label)
            false_negative += int((not admit) and label)

        if admit:
            admitted += 1
            cache.observe(token)
        gate_cells += sketch.update(token)

    cache_cells = cache.local_touched_cells + cache.decay_touched_cells
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    return LearnedAdmissionResult(
        train_length=train_length,
        eval_length=eval_length,
        warmup_events=warmup_events,
        vocab_size=vocab_size,
        hot_tokens=hot_tokens,
        top_k=top_k,
        future_horizon=future_horizon,
        scores=lut.scores,
        lut_state_bytes=lut.state_bytes,
        topk_hit_rate=hits / eval_events if eval_events else 0.0,
        admission_rate=admitted / eval_length if eval_length else 0.0,
        admission_precision=(
            true_positive / precision_denominator if precision_denominator else 0.0
        ),
        admission_recall=true_positive / recall_denominator if recall_denominator else 0.0,
        cache_update_hit_rate=cache.cache_update_hit_rate(),
        avg_gate_cells=gate_cells / eval_length,
        avg_cache_cells=cache_cells / eval_length,
        avg_total_cells=(gate_cells + cache_cells) / eval_length,
        replacements=cache.replacements,
        resident_tokens=cache.resident_count(),
    )
