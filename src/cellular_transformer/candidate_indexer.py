"""Trainable multi-feature candidate indexer experiments.

The earlier candidate scorers tested either a 2D LUT or hand-written formulas.
This module tests the next hardware-shaped step: tiny signed low-bit learned
rules over local candidate features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .candidate_cache import CandidateCacheConfig, LowBitCandidateCache
from .synthetic_lm import DualPathSyntheticLM, SyntheticLMConfig, sample_topic_token


FEATURE_NAMES: Tuple[str, ...] = ("dense", "topic", "cache", "contamination", "age")


@dataclass(frozen=True)
class LowBitLinearCandidateIndexer:
    """Signed low-bit linear candidate ranker."""

    weights: Tuple[int, ...]
    bias: int = 0
    score_bits: int = 4
    feature_names: Tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        if len(self.weights) != len(self.feature_names):
            raise ValueError("weights length must match feature_names")
        if self.score_bits not in (2, 4, 8):
            raise ValueError("score_bits must be one of 2, 4, 8")
        min_score = -(1 << (self.score_bits - 1))
        max_score = (1 << (self.score_bits - 1)) - 1
        for value in (*self.weights, self.bias):
            if not min_score <= int(value) <= max_score:
                raise ValueError("weight outside signed score_bits range")

    @property
    def state_bytes(self) -> float:
        return (len(self.weights) + 1) * self.score_bits / 8

    def scores(self, features: np.ndarray) -> np.ndarray:
        weights = np.asarray(self.weights, dtype=np.int32)
        return features.astype(np.int32) @ weights + int(self.bias)


@dataclass(frozen=True)
class LowBitAdditiveCandidateIndexer:
    """Signed low-bit additive LUT ranker over per-feature bins."""

    tables: Tuple[Tuple[int, ...], ...]
    bias: int = 0
    score_bits: int = 4
    feature_names: Tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        if len(self.tables) != len(self.feature_names):
            raise ValueError("tables length must match feature_names")
        if self.score_bits not in (2, 4, 8):
            raise ValueError("score_bits must be one of 2, 4, 8")
        min_score = -(1 << (self.score_bits - 1))
        max_score = (1 << (self.score_bits - 1)) - 1
        for table in self.tables:
            if len(table) != 16:
                raise ValueError("each additive table must have 16 bins")
            for value in table:
                if not min_score <= int(value) <= max_score:
                    raise ValueError("table value outside signed score_bits range")
        if not min_score <= int(self.bias) <= max_score:
            raise ValueError("bias outside signed score_bits range")

    @property
    def state_bytes(self) -> float:
        values = sum(len(table) for table in self.tables) + 1
        return values * self.score_bits / 8

    def scores(self, features: np.ndarray) -> np.ndarray:
        total = np.full(len(features), int(self.bias), dtype=np.int32)
        for index, table in enumerate(self.tables):
            lookup = np.asarray(table, dtype=np.int32)
            total += lookup[np.clip(features[:, index], 0, 15)]
        return total


@dataclass(frozen=True)
class CandidateIndexerTrialResult:
    """Aggregate metrics for the trainable local indexer."""

    train_seed: int
    eval_seed: int
    admission_threshold: int
    epochs: int
    feature_names: Tuple[str, ...]
    weights: Tuple[int, ...]
    additive_state_bytes: float
    state_bytes: float
    resident_hit_rate: float
    feature_ceiling_hit_rate: float
    positive_unique_rate: float
    mean_positive_bucket_size: float
    dense_hit_rate: float
    topic_hit_rate: float
    topic_cache_hit_rate: float
    additive_hit_rate: float
    learned_hit_rate: float
    learned_score_cells_per_event: float
    topic_score_update_cells_per_event: float


def train_linear_candidate_indexer(
    train_seed: int = 100,
    admission_threshold: int = 0,
    epochs: int = 3,
    score_bits: int = 4,
) -> LowBitLinearCandidateIndexer:
    """Train a low-bit linear scorer with a top-k perceptron update."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    accumulator = np.zeros(len(FEATURE_NAMES), dtype=np.int64)
    for _ in range(epochs):
        _replay_candidate_stream(
            seed=train_seed,
            admission_threshold=admission_threshold,
            update_accumulator=accumulator,
        )

    max_abs = int(np.max(np.abs(accumulator)))
    max_weight = (1 << (score_bits - 1)) - 1
    min_weight = -(1 << (score_bits - 1))
    if max_abs == 0:
        weights = np.zeros(len(FEATURE_NAMES), dtype=np.int32)
    else:
        weights = np.rint(accumulator / max_abs * max_weight).astype(np.int32)
    weights = np.clip(weights, min_weight, max_weight)
    return LowBitLinearCandidateIndexer(
        weights=tuple(int(value) for value in weights),
        bias=0,
        score_bits=score_bits,
    )


def train_additive_candidate_indexer(
    train_seed: int = 100,
    admission_threshold: int = 0,
    score_bits: int = 4,
) -> LowBitAdditiveCandidateIndexer:
    """Train per-feature LUTs from smoothed log-odds labels."""

    positive, observed = _collect_additive_statistics(
        seed=train_seed,
        admission_threshold=admission_threshold,
    )
    negative = observed - positive
    raw = np.log2((positive + 1) / (negative + 1))
    max_abs = float(np.max(np.abs(raw)))
    max_score = (1 << (score_bits - 1)) - 1
    min_score = -(1 << (score_bits - 1))
    if max_abs == 0.0:
        tables = np.zeros_like(raw, dtype=np.int32)
    else:
        tables = np.rint(raw / max_abs * max_score).astype(np.int32)
    tables = np.clip(tables, min_score, max_score)
    return LowBitAdditiveCandidateIndexer(
        tables=tuple(tuple(int(value) for value in table) for table in tables),
        bias=0,
        score_bits=score_bits,
    )


def run_candidate_indexer_trial(
    train_seed: int = 100,
    eval_seed: int = 31,
    admission_threshold: int = 0,
    epochs: int = 3,
) -> CandidateIndexerTrialResult:
    """Train and evaluate a multi-feature local candidate indexer."""

    indexer = train_linear_candidate_indexer(
        train_seed=train_seed,
        admission_threshold=admission_threshold,
        epochs=epochs,
    )
    additive = train_additive_candidate_indexer(
        train_seed=train_seed,
        admission_threshold=admission_threshold,
    )
    metrics = _replay_candidate_stream(
        seed=eval_seed,
        admission_threshold=admission_threshold,
        indexer=indexer,
        additive_indexer=additive,
    )
    return CandidateIndexerTrialResult(
        train_seed=train_seed,
        eval_seed=eval_seed,
        admission_threshold=admission_threshold,
        epochs=epochs,
        feature_names=indexer.feature_names,
        weights=indexer.weights,
        additive_state_bytes=additive.state_bytes,
        state_bytes=indexer.state_bytes,
        resident_hit_rate=metrics["resident"],
        feature_ceiling_hit_rate=metrics["feature_ceiling"],
        positive_unique_rate=metrics["positive_unique"],
        mean_positive_bucket_size=metrics["positive_bucket"],
        dense_hit_rate=metrics["dense"],
        topic_hit_rate=metrics["topic"],
        topic_cache_hit_rate=metrics["topic_cache"],
        additive_hit_rate=metrics["additive"],
        learned_hit_rate=metrics["learned"],
        learned_score_cells_per_event=metrics["score_cells"],
        topic_score_update_cells_per_event=metrics["score_updates"],
    )


def _replay_candidate_stream(
    seed: int,
    admission_threshold: int,
    indexer: LowBitLinearCandidateIndexer | None = None,
    additive_indexer: LowBitAdditiveCandidateIndexer | None = None,
    update_accumulator: np.ndarray | None = None,
) -> dict[str, float]:
    if admission_threshold < 0:
        raise ValueError("admission_threshold must be non-negative")

    config = SyntheticLMConfig(
        dense_width=2048,
        candidate_strategy="online_cache",
        candidate_admission_threshold=admission_threshold,
        candidate_score_source="topic_phase",
    )
    lm = DualPathSyntheticLM(config, seed=seed)
    lm.prefill()
    _install_age_cache(lm)

    query_indices = lm.rng.choice(
        len(lm.facts),
        size=lm.config.query_events,
        replace=True,
    )
    event_types = np.array(
        ["topic"] * lm.config.topic_events + ["query"] * lm.config.query_events
    )
    lm.rng.shuffle(event_types)

    hits = {
        name: 0
        for name in ("resident", "dense", "topic", "topic_cache", "additive", "learned")
    }
    score_cells = 0
    score_updates = 0
    topic_events = 0
    feature_ceiling = 0.0
    positive_unique = 0
    positive_bucket_total = 0
    resident_events = 0
    query_cursor = 0

    for event_type in event_types:
        if event_type == "topic":
            token = sample_topic_token(lm.config, lm.rng)
            topic_events += 1
            entries = lm.candidate_cache.resident_feature_entries()
            entries = sorted(entries, key=lambda item: (-item[1], item[0]))[
                : lm.config.candidate_pool_size
            ]
            if entries:
                candidates = np.array([candidate for candidate, _, _ in entries], dtype=np.int32)
                cache_scores = np.array([score for _, score, _ in entries], dtype=np.int32)
                age_scores = np.array([age for _, _, age in entries], dtype=np.int32)
                features = _candidate_features(lm, candidates, cache_scores, age_scores)
                top_k = min(lm.config.topic_top_k, len(candidates))
                token_matches = candidates == int(token)
                hits["resident"] += int(bool(token_matches.any()))
                if bool(token_matches.any()):
                    positive_index = int(np.flatnonzero(token_matches)[0])
                    bucket_size = _feature_bucket_size(features, positive_index)
                    resident_events += 1
                    positive_bucket_total += bucket_size
                    positive_unique += int(bucket_size == 1)
                    feature_ceiling += min(1.0, top_k / bucket_size)
                hits["dense"] += _topk_hit(features[:, 0], candidates, token, top_k)
                hits["topic"] += _topk_hit(features[:, 1], candidates, token, top_k)
                topic_cache_scores = 2 * features[:, 1] + features[:, 2]
                hits["topic_cache"] += _topk_hit(topic_cache_scores, candidates, token, top_k)

                if indexer is not None:
                    hits["learned"] += _topk_hit(indexer.scores(features), candidates, token, top_k)
                if additive_indexer is not None:
                    hits["additive"] += _topk_hit(
                        additive_indexer.scores(features),
                        candidates,
                        token,
                        top_k,
                    )
                if update_accumulator is not None and bool(token_matches.any()):
                    _perceptron_update(update_accumulator, features, token_matches, top_k)
                score_cells += len(candidates) * lm.config.dense_banks * 2

            admit = True
            if admission_threshold > 0:
                admit = lm.dense.estimate(token) >= admission_threshold
            lm.dense.update(token)
            if lm.candidate_score_dense is None:
                raise RuntimeError("topic-phase score state is not initialized")
            score_updates += lm.candidate_score_dense.update(token)
            if admit:
                lm.candidate_cache.observe(token)
            continue

        key, expected = lm.facts[int(query_indices[query_cursor])]
        query_cursor += 1
        lm.exact.lookup(key)
        lm.dense.update(key)
        lm.dense.update(expected)

    denominator = topic_events if topic_events else 1
    return {
        "resident": hits["resident"] / denominator,
        "feature_ceiling": feature_ceiling / denominator,
        "positive_unique": positive_unique / denominator,
        "positive_bucket": (
            positive_bucket_total / resident_events if resident_events else 0.0
        ),
        "dense": hits["dense"] / denominator,
        "topic": hits["topic"] / denominator,
        "topic_cache": hits["topic_cache"] / denominator,
        "additive": hits["additive"] / denominator,
        "learned": hits["learned"] / denominator,
        "score_cells": score_cells / (lm.config.topic_events + lm.config.query_events),
        "score_updates": score_updates / (lm.config.topic_events + lm.config.query_events),
    }


def _collect_additive_statistics(
    seed: int,
    admission_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    if admission_threshold < 0:
        raise ValueError("admission_threshold must be non-negative")

    positive = np.zeros((len(FEATURE_NAMES), 16), dtype=np.int64)
    observed = np.zeros((len(FEATURE_NAMES), 16), dtype=np.int64)
    config = SyntheticLMConfig(
        dense_width=2048,
        candidate_strategy="online_cache",
        candidate_admission_threshold=admission_threshold,
        candidate_score_source="topic_phase",
    )
    lm = DualPathSyntheticLM(config, seed=seed)
    lm.prefill()
    _install_age_cache(lm)
    query_indices = lm.rng.choice(
        len(lm.facts),
        size=lm.config.query_events,
        replace=True,
    )
    event_types = np.array(
        ["topic"] * lm.config.topic_events + ["query"] * lm.config.query_events
    )
    lm.rng.shuffle(event_types)

    query_cursor = 0
    for event_type in event_types:
        if event_type == "topic":
            token = sample_topic_token(lm.config, lm.rng)
            entries = lm.candidate_cache.resident_feature_entries()
            entries = sorted(entries, key=lambda item: (-item[1], item[0]))[
                : lm.config.candidate_pool_size
            ]
            if entries:
                candidates = np.array([candidate for candidate, _, _ in entries], dtype=np.int32)
                cache_scores = np.array([score for _, score, _ in entries], dtype=np.int32)
                age_scores = np.array([age for _, _, age in entries], dtype=np.int32)
                features = np.clip(_candidate_features(lm, candidates, cache_scores, age_scores), 0, 15)
                labels = (candidates == int(token)).astype(np.int64)
                for feature_index in range(len(FEATURE_NAMES)):
                    np.add.at(observed[feature_index], features[:, feature_index], 1)
                    np.add.at(positive[feature_index], features[:, feature_index], labels)

            admit = True
            if admission_threshold > 0:
                admit = lm.dense.estimate(token) >= admission_threshold
            lm.dense.update(token)
            if lm.candidate_score_dense is None:
                raise RuntimeError("topic-phase score state is not initialized")
            lm.candidate_score_dense.update(token)
            if admit:
                lm.candidate_cache.observe(token)
            continue

        key, expected = lm.facts[int(query_indices[query_cursor])]
        query_cursor += 1
        lm.exact.lookup(key)
        lm.dense.update(key)
        lm.dense.update(expected)

    return positive, observed


def _candidate_features(
    lm: DualPathSyntheticLM,
    candidates: np.ndarray,
    cache_scores: np.ndarray,
    age_scores: np.ndarray,
) -> np.ndarray:
    if lm.candidate_score_dense is None:
        raise RuntimeError("topic-phase score state is not initialized")
    candidate_slots = lm._candidate_slots(candidates)
    bank_indices = np.arange(lm.config.dense_banks)[:, None]
    dense_scores = lm.dense.counters[bank_indices, candidate_slots].min(axis=0).astype(np.int32)
    topic_scores = lm.candidate_score_dense.counters[bank_indices, candidate_slots].min(axis=0).astype(np.int32)
    contamination = np.maximum(dense_scores - topic_scores, 0)
    return np.stack(
        [
            dense_scores,
            topic_scores,
            cache_scores.astype(np.int32),
            contamination,
            age_scores.astype(np.int32),
        ],
        axis=1,
    )


def _install_age_cache(lm: DualPathSyntheticLM) -> None:
    lm.candidate_cache = LowBitCandidateCache(
        CandidateCacheConfig(
            vocab_size=lm.config.vocab_size,
            capacity=lm.config.candidate_pool_size,
            ways=lm.config.candidate_cache_ways,
            routes=lm.config.candidate_cache_routes,
            score_bits=lm.config.candidate_cache_score_bits,
            token_bits=max(1, (lm.config.vocab_size - 1).bit_length()),
            decay_interval=lm.config.candidate_cache_decay_interval,
            decay_shift=lm.config.candidate_cache_decay_shift,
            age_bits=4,
            age_bucket_interval=16,
        )
    )


def _topk_hit(scores: np.ndarray, candidates: np.ndarray, token: int, top_k: int) -> int:
    chosen = np.argsort(scores)[-top_k:]
    return int(int(token) in {int(candidates[index]) for index in chosen})


def _feature_bucket_size(features: np.ndarray, index: int) -> int:
    key = features[index]
    return int(np.count_nonzero(np.all(features == key, axis=1)))


def _perceptron_update(
    accumulator: np.ndarray,
    features: np.ndarray,
    token_matches: np.ndarray,
    top_k: int,
) -> None:
    scores = features @ accumulator
    positive_indices = np.flatnonzero(token_matches)
    if len(positive_indices) == 0:
        return
    positive_index = int(positive_indices[np.argmax(scores[positive_indices])])
    top_indices = np.argsort(scores)[-top_k:]
    if positive_index in set(int(index) for index in top_indices):
        return
    negative_mask = ~token_matches
    negative_indices = np.flatnonzero(negative_mask)
    if len(negative_indices) == 0:
        return
    negative_index = int(negative_indices[np.argmax(scores[negative_indices])])
    accumulator += features[positive_index] - features[negative_index]
