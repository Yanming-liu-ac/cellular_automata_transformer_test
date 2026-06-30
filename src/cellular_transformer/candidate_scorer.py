"""Trainable low-bit candidate scoring for shortlist ranking.

Candidate generation has two parts:

1. admission decides which tokens enter the small cache;
2. scoring ranks resident candidates before the output head sees them.

This module trains a tiny 2D integer LUT from self-supervised future-repeat
labels. At inference, the scorer only reads local dense-context estimates and
candidate-cache scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .admission_policy import future_repeat_labels, make_topic_stream
from .candidate_cache import CandidateCacheConfig, LowBitCandidateCache
from .dense_context import DenseContextConfig, LowBitDenseContext


@dataclass(frozen=True)
class LowBitCandidateScorerLUT:
    """Signed low-bit 2D LUT indexed by dense estimate and cache score."""

    scores: Tuple[int, ...]
    dense_bins: int = 16
    cache_bins: int = 16
    score_bits: int = 4

    def __post_init__(self) -> None:
        if self.dense_bins <= 0:
            raise ValueError("dense_bins must be positive")
        if self.cache_bins <= 0:
            raise ValueError("cache_bins must be positive")
        if len(self.scores) != self.dense_bins * self.cache_bins:
            raise ValueError("scores length must equal dense_bins * cache_bins")
        if self.score_bits not in (2, 4, 8):
            raise ValueError("score_bits must be one of 2, 4, 8")
        min_score = -(1 << (self.score_bits - 1))
        max_score = (1 << (self.score_bits - 1)) - 1
        for score in self.scores:
            if not min_score <= int(score) <= max_score:
                raise ValueError("score outside signed score_bits range")

    def score(self, dense_estimate: int, cache_score: int) -> int:
        dense_index = min(max(int(dense_estimate), 0), self.dense_bins - 1)
        cache_index = min(max(int(cache_score), 0), self.cache_bins - 1)
        return int(self.scores[dense_index * self.cache_bins + cache_index])

    @property
    def state_bytes(self) -> float:
        return len(self.scores) * self.score_bits / 8


@dataclass(frozen=True)
class CandidateScorerTrialResult:
    """Aggregate learned candidate-scorer benchmark metrics."""

    training_target: str
    train_length: int
    eval_length: int
    warmup_events: int
    vocab_size: int
    hot_tokens: int
    top_k: int
    future_horizon: int
    scoring_dense_weight: int
    scoring_cache_weight: int
    dense_bins: int
    cache_bins: int
    lut_state_bytes: float
    baseline_topk_hit_rate: float
    learned_topk_hit_rate: float
    admission_rate: float
    cache_update_hit_rate: float
    avg_gate_cells: float
    avg_cache_cells: float
    avg_score_cells: float
    replacements: int
    resident_tokens: int
    full_vocab_scan_tokens: int = 0


def train_repeat_candidate_scorer_lut(
    stream: Iterable[int],
    dense_config: DenseContextConfig,
    future_horizon: int = 256,
    admission_threshold: int = 1,
    score_bits: int = 4,
    capacity: int = 512,
    ways: int = 4,
    routes: int = 2,
) -> LowBitCandidateScorerLUT:
    """Train a 2D candidate scorer from future-repeat labels."""

    tokens = [int(token) for token in stream]
    labels = future_repeat_labels(tokens, future_horizon)
    dense_bins = 1 << dense_config.bits
    cache_bins = 1 << score_bits
    positive = np.zeros((dense_bins, cache_bins), dtype=np.int32)
    negative = np.zeros((dense_bins, cache_bins), dtype=np.int32)

    cache = LowBitCandidateCache(
        CandidateCacheConfig(
            vocab_size=dense_config.vocab_size,
            capacity=capacity,
            ways=ways,
            routes=routes,
            score_bits=score_bits,
        )
    )
    sketch = LowBitDenseContext(dense_config)

    for token, label in zip(tokens, labels):
        for candidate, cache_score in cache.resident_entries():
            dense_estimate = min(sketch.estimate(candidate), dense_bins - 1)
            cache_bin = min(int(cache_score), cache_bins - 1)
            if candidate == token and bool(label):
                positive[dense_estimate, cache_bin] += 1
            else:
                negative[dense_estimate, cache_bin] += 1

        estimate = sketch.estimate(token)
        sketch.update(token)
        if estimate >= admission_threshold:
            cache.observe(token)

    min_score = -(1 << (score_bits - 1))
    max_score = (1 << (score_bits - 1)) - 1
    scores = np.clip(positive - negative, min_score, max_score)
    return LowBitCandidateScorerLUT(
        tuple(int(score) for score in scores.reshape(-1)),
        dense_bins=dense_bins,
        cache_bins=cache_bins,
        score_bits=score_bits,
    )


def train_future_window_candidate_scorer_lut(
    stream: Iterable[int],
    dense_config: DenseContextConfig,
    future_horizon: int = 256,
    admission_threshold: int = 1,
    score_bits: int = 4,
    capacity: int = 512,
    ways: int = 4,
    routes: int = 2,
    include_current_target: bool = True,
    target: str = "hit_rate",
) -> LowBitCandidateScorerLUT:
    """Train a candidate indexer from future-window teacher labels.

    The deployed scorer still sees only local features:

    ``dense sketch estimate`` and ``candidate-cache score``.

    The teacher label is less myopic than the earlier current-token classifier:
    it asks whether a resident candidate appears in the next local target
    window, optionally including the current next token.
    """

    if target not in ("hit_rate", "expected_count"):
        raise ValueError("target must be 'hit_rate' or 'expected_count'")

    tokens = [int(token) for token in stream]
    dense_bins = 1 << dense_config.bits
    cache_bins = 1 << score_bits
    max_score = (1 << (score_bits - 1)) - 1
    observations = np.zeros((dense_bins, cache_bins), dtype=np.int32)
    positive = np.zeros((dense_bins, cache_bins), dtype=np.int32)
    reward_sum = np.zeros((dense_bins, cache_bins), dtype=np.int32)

    future_counts = np.zeros(dense_config.vocab_size, dtype=np.uint16)
    start = 0 if include_current_target else 1
    stop = min(len(tokens), start + future_horizon)
    for token in tokens[start:stop]:
        future_counts[token] += 1

    cache = LowBitCandidateCache(
        CandidateCacheConfig(
            vocab_size=dense_config.vocab_size,
            capacity=capacity,
            ways=ways,
            routes=routes,
            score_bits=score_bits,
        )
    )
    sketch = LowBitDenseContext(dense_config)

    for index, token in enumerate(tokens):
        for candidate, cache_score in cache.resident_entries():
            dense_estimate = min(sketch.estimate(candidate), dense_bins - 1)
            cache_bin = min(int(cache_score), cache_bins - 1)
            future_count = int(future_counts[candidate])
            observations[dense_estimate, cache_bin] += 1
            positive[dense_estimate, cache_bin] += int(future_count > 0)
            reward_sum[dense_estimate, cache_bin] += min(future_count, max_score)

        estimate = sketch.estimate(token)
        sketch.update(token)
        if estimate >= admission_threshold:
            cache.observe(token)

        remove_index = index if include_current_target else index + 1
        if remove_index < len(tokens):
            future_counts[tokens[remove_index]] -= 1
        add_index = index + future_horizon
        if not include_current_target:
            add_index += 1
        if add_index < len(tokens):
            future_counts[tokens[add_index]] += 1

    scores = np.zeros((dense_bins, cache_bins), dtype=np.int32)
    observed = observations > 0
    if target == "hit_rate":
        scores[observed] = np.rint(
            positive[observed] / observations[observed] * max_score
        ).astype(np.int32)
    else:
        scores[observed] = np.rint(
            reward_sum[observed] / observations[observed]
        ).astype(np.int32)

    return LowBitCandidateScorerLUT(
        tuple(int(score) for score in np.clip(scores, 0, max_score).reshape(-1)),
        dense_bins=dense_bins,
        cache_bins=cache_bins,
        score_bits=score_bits,
    )


def run_candidate_scorer_trial(
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
    admission_threshold: int = 1,
    topic_probability: float = 0.85,
    zipf_exponent: float = 1.15,
    train_seed: int = 100,
    eval_seed: int = 17,
    training_target: str = "future_window",
    future_window_target: str = "hit_rate",
    scoring_dense_weight: int = 1,
    scoring_cache_weight: int = 0,
) -> CandidateScorerTrialResult:
    """Train and evaluate low-bit candidate scoring on a topic stream."""

    if not 0 <= warmup_events < eval_length:
        raise ValueError("warmup_events must be in [0, eval_length)")
    if training_target not in ("current_token_repeat", "future_window"):
        raise ValueError("training_target must be 'current_token_repeat' or 'future_window'")
    if scoring_dense_weight < 0:
        raise ValueError("scoring_dense_weight must be non-negative")
    if scoring_cache_weight < 0:
        raise ValueError("scoring_cache_weight must be non-negative")

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
    if training_target == "current_token_repeat":
        scorer = train_repeat_candidate_scorer_lut(
            train_stream,
            dense_config=dense_config,
            future_horizon=future_horizon,
            admission_threshold=admission_threshold,
            score_bits=score_bits,
            capacity=capacity,
            ways=ways,
            routes=routes,
        )
    else:
        scorer = train_future_window_candidate_scorer_lut(
            train_stream,
            dense_config=dense_config,
            future_horizon=future_horizon,
            admission_threshold=admission_threshold,
            score_bits=score_bits,
            capacity=capacity,
            ways=ways,
            routes=routes,
            target=future_window_target,
        )

    eval_stream = make_topic_stream(
        length=eval_length,
        vocab_size=vocab_size,
        hot_tokens=hot_tokens,
        topic_probability=topic_probability,
        zipf_exponent=zipf_exponent,
        seed=eval_seed,
    )
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

    baseline_hits = 0
    learned_hits = 0
    eval_events = 0
    admitted = 0
    gate_cells = 0
    score_cells = 0

    for step, token in enumerate(eval_stream):
        token = int(token)
        entries = cache.resident_entries()
        if step >= warmup_events:
            eval_events += 1
            baseline_ranked: list[tuple[int, int]] = []
            learned_ranked: list[tuple[int, int, int]] = []
            for candidate, cache_score in entries:
                dense_estimate = sketch.estimate(candidate)
                score_cells += dense_config.banks
                baseline_ranked.append((dense_estimate, candidate))
                learned_score = (
                    scoring_dense_weight * dense_estimate
                    + scoring_cache_weight * cache_score
                    + scorer.score(dense_estimate, cache_score)
                )
                learned_ranked.append((learned_score, dense_estimate, candidate))

            baseline_top = {candidate for _, candidate in sorted(baseline_ranked)[-top_k:]}
            learned_top = {
                candidate for _, _, candidate in sorted(learned_ranked)[-top_k:]
            }
            baseline_hits += int(token in baseline_top)
            learned_hits += int(token in learned_top)

        gate_cells += dense_config.banks
        estimate = sketch.estimate(token)
        sketch.update(token)
        if estimate >= admission_threshold:
            admitted += 1
            cache.observe(token)

    cache_cells = cache.local_touched_cells + cache.decay_touched_cells
    return CandidateScorerTrialResult(
        training_target=training_target,
        train_length=train_length,
        eval_length=eval_length,
        warmup_events=warmup_events,
        vocab_size=vocab_size,
        hot_tokens=hot_tokens,
        top_k=top_k,
        future_horizon=future_horizon,
        scoring_dense_weight=scoring_dense_weight,
        scoring_cache_weight=scoring_cache_weight,
        dense_bins=scorer.dense_bins,
        cache_bins=scorer.cache_bins,
        lut_state_bytes=scorer.state_bytes,
        baseline_topk_hit_rate=baseline_hits / eval_events if eval_events else 0.0,
        learned_topk_hit_rate=learned_hits / eval_events if eval_events else 0.0,
        admission_rate=admitted / eval_length if eval_length else 0.0,
        cache_update_hit_rate=cache.cache_update_hit_rate(),
        avg_gate_cells=gate_cells / eval_length,
        avg_cache_cells=cache_cells / eval_length,
        avg_score_cells=score_cells / eval_events if eval_events else 0.0,
        replacements=cache.replacements,
        resident_tokens=cache.resident_count(),
    )
