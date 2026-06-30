"""Online low-bit candidate cache for output shortlist generation.

The output head only stays cheap if the candidate shortlist is produced without
scanning the whole vocabulary. This module models a hardware-shaped cache:
fixed capacity, set-associative hash routes, low-bit saturating scores, and
periodic integer decay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .dense_context import DenseContextConfig, LowBitDenseContext
from .retrieval import keyed_hash


@dataclass(frozen=True)
class CandidateCacheConfig:
    """Configuration for a set-associative low-bit candidate cache."""

    vocab_size: int = 65536
    capacity: int = 512
    ways: int = 4
    routes: int = 2
    score_bits: int = 4
    token_bits: int = 16
    decay_interval: int = 256
    decay_shift: int = 1
    insertion_score: int = 1
    evict_zero_scores: bool = False
    age_bits: int = 0
    age_bucket_interval: int = 16

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.ways <= 0:
            raise ValueError("ways must be positive")
        if self.capacity % self.ways != 0:
            raise ValueError("capacity must be divisible by ways")
        if self.routes <= 0:
            raise ValueError("routes must be positive")
        if self.score_bits not in (2, 4, 8):
            raise ValueError("score_bits must be one of 2, 4, 8")
        if self.age_bits not in (0, 2, 4, 8):
            raise ValueError("age_bits must be one of 0, 2, 4, 8")
        if self.age_bucket_interval <= 0:
            raise ValueError("age_bucket_interval must be positive")
        if self.token_bits <= 0:
            raise ValueError("token_bits must be positive")
        if self.decay_interval <= 0:
            raise ValueError("decay_interval must be positive")
        if self.decay_shift <= 0:
            raise ValueError("decay_shift must be positive")
        if not 0 < self.insertion_score <= self.max_score:
            raise ValueError("insertion_score must be in (0, max_score]")
        if self.vocab_size > (1 << self.token_bits):
            raise ValueError("token_bits cannot represent vocab_size")

    @property
    def buckets(self) -> int:
        return self.capacity // self.ways

    @property
    def max_score(self) -> int:
        return (1 << self.score_bits) - 1

    @property
    def state_bytes(self) -> float:
        valid_bits = 1
        return self.capacity * (self.token_bits + self.score_bits + self.age_bits + valid_bits) / 8


@dataclass(frozen=True)
class CandidateCacheUpdate:
    """Per-token cache update statistics."""

    hit: bool
    replacement: bool
    local_touched_cells: int
    decay_touched_cells: int

    @property
    def total_touched_cells(self) -> int:
        return self.local_touched_cells + self.decay_touched_cells


@dataclass(frozen=True)
class CandidateCacheTrialResult:
    """Aggregate online candidate-cache benchmark metrics."""

    context_length: int
    eval_events: int
    vocab_size: int
    hot_tokens: int
    top_k: int
    capacity: int
    ways: int
    routes: int
    score_bits: int
    admission_threshold: int
    state_bytes: float
    gate_state_bytes: float
    total_state_bytes: float
    topk_hit_rate: float
    admission_rate: float
    cache_update_hit_rate: float
    avg_gate_cells: float
    avg_local_update_cells: float
    avg_decay_cells: float
    avg_total_update_cells: float
    admission_skips: int
    replacements: int
    resident_tokens: int
    full_vocab_scan_tokens: int = 0


class LowBitCandidateCache:
    """Fixed-capacity candidate generator with low-bit decayed scores."""

    def __init__(self, config: CandidateCacheConfig) -> None:
        self.config = config
        shape = (config.buckets, config.ways)
        self.tokens = np.zeros(shape, dtype=np.uint32)
        self.scores = np.zeros(shape, dtype=np.uint8)
        self.last_seen = np.zeros(shape, dtype=np.uint32)
        self.valid = np.zeros(shape, dtype=bool)
        self.steps = 0
        self.update_hits = 0
        self.updates = 0
        self.replacements = 0
        self.local_touched_cells = 0
        self.decay_touched_cells = 0

    def _route_buckets(self, token: int) -> List[int]:
        buckets = []
        seen: set[int] = set()
        for route in range(self.config.routes):
            bucket = keyed_hash(int(token), 9000 + route) % self.config.buckets
            if bucket not in seen:
                seen.add(bucket)
                buckets.append(bucket)
        return buckets

    def observe(self, token: int) -> CandidateCacheUpdate:
        """Update the cache with one observed token."""

        token = int(token)
        if not 0 <= token < self.config.vocab_size:
            raise ValueError("token outside vocab")

        route_buckets = self._route_buckets(token)
        local_touched = len(route_buckets) * self.config.ways
        self.local_touched_cells += local_touched
        self.updates += 1

        for bucket in route_buckets:
            matches = self.valid[bucket] & (self.tokens[bucket] == token)
            if bool(matches.any()):
                way = int(np.flatnonzero(matches)[0])
                score = int(self.scores[bucket, way])
                if score < self.config.max_score:
                    self.scores[bucket, way] = score + 1
                self.last_seen[bucket, way] = self.steps
                self.update_hits += 1
                decay_touched = self._maybe_decay()
                return CandidateCacheUpdate(
                    hit=True,
                    replacement=False,
                    local_touched_cells=local_touched,
                    decay_touched_cells=decay_touched,
                )

        target_bucket = route_buckets[0]
        target_way = 0
        replacement = True
        for bucket in route_buckets:
            empty = np.flatnonzero(~self.valid[bucket])
            if len(empty) > 0:
                target_bucket = bucket
                target_way = int(empty[0])
                replacement = False
                break

        if replacement:
            best_score = self.config.max_score + 1
            for bucket in route_buckets:
                way = int(np.argmin(self.scores[bucket]))
                score = int(self.scores[bucket, way])
                if score < best_score:
                    best_score = score
                    target_bucket = bucket
                    target_way = way
            self.replacements += 1

        self.tokens[target_bucket, target_way] = token
        self.scores[target_bucket, target_way] = self.config.insertion_score
        self.last_seen[target_bucket, target_way] = self.steps
        self.valid[target_bucket, target_way] = True
        decay_touched = self._maybe_decay()
        return CandidateCacheUpdate(
            hit=False,
            replacement=replacement,
            local_touched_cells=local_touched,
            decay_touched_cells=decay_touched,
        )

    def observe_many(self, tokens: Iterable[int]) -> int:
        """Update many tokens and return total touched cells."""

        touched = 0
        for token in tokens:
            touched += self.observe(int(token)).total_touched_cells
        return touched

    def _maybe_decay(self) -> int:
        self.steps += 1
        if self.steps % self.config.decay_interval != 0:
            return 0
        return self.decay()

    def decay(self) -> int:
        """Decay all cache scores and optionally invalidate zero-score entries."""

        self.scores >>= self.config.decay_shift
        if self.config.evict_zero_scores:
            self.valid &= self.scores > 0
        touched = self.config.capacity
        self.decay_touched_cells += touched
        return touched

    def topk(self, k: int) -> List[int]:
        """Return up to k resident token IDs with the highest cache scores."""

        if k <= 0:
            raise ValueError("k must be positive")

        valid_indices = np.argwhere(self.valid)
        if len(valid_indices) == 0:
            return []

        scores = np.array(
            [self.scores[bucket, way] for bucket, way in valid_indices],
            dtype=np.int32,
        )
        tokens = np.array([self.tokens[bucket, way] for bucket, way in valid_indices])
        order = np.lexsort((tokens, -scores))
        chosen = order[: min(k, len(order))]
        return [int(tokens[index]) for index in chosen]

    def topk_set(self, k: int) -> set[int]:
        return set(self.topk(k))

    def resident_entries(self) -> List[Tuple[int, int]]:
        """Return resident ``(token, score)`` pairs for local candidate scoring."""

        entries: List[Tuple[int, int]] = []
        for bucket, way in np.argwhere(self.valid):
            entries.append((int(self.tokens[bucket, way]), int(self.scores[bucket, way])))
        return entries

    def resident_feature_entries(self) -> List[Tuple[int, int, int]]:
        """Return resident ``(token, score, age_bucket)`` triples."""

        entries: List[Tuple[int, int, int]] = []
        for bucket, way in np.argwhere(self.valid):
            entries.append(
                (
                    int(self.tokens[bucket, way]),
                    int(self.scores[bucket, way]),
                    self._age_bucket(int(self.last_seen[bucket, way])),
                )
            )
        return entries

    def _age_bucket(self, last_seen: int) -> int:
        if self.config.age_bits == 0:
            return 0
        max_age = (1 << self.config.age_bits) - 1
        raw_age = max(0, self.steps - int(last_seen))
        return min(raw_age // self.config.age_bucket_interval, max_age)

    def resident_count(self) -> int:
        return int(self.valid.sum())

    def memory_bytes(self) -> float:
        return self.config.state_bytes

    def cache_update_hit_rate(self) -> float:
        if self.updates == 0:
            return 0.0
        return self.update_hits / self.updates


def sample_zipf_topic_token(
    vocab_size: int,
    hot_tokens: int,
    topic_probability: float,
    zipf_exponent: float,
    rng: np.random.Generator,
) -> int:
    """Sample from a hot Zipf topic mixed with uniform background noise."""

    if rng.random() > topic_probability:
        return int(rng.integers(hot_tokens, vocab_size, dtype=np.uint32))

    ranks = np.arange(1, hot_tokens + 1, dtype=np.float64)
    probabilities = 1.0 / np.power(ranks, zipf_exponent)
    probabilities /= probabilities.sum()
    return int(rng.choice(hot_tokens, p=probabilities))


def run_candidate_cache_trial(
    context_length: int = 8192,
    warmup_events: int = 1024,
    vocab_size: int = 65536,
    hot_tokens: int = 256,
    top_k: int = 64,
    capacity: int = 512,
    ways: int = 4,
    routes: int = 2,
    score_bits: int = 4,
    decay_interval: int = 256,
    admission_threshold: int = 0,
    gate_banks: int = 4,
    gate_width: int = 2048,
    gate_bits: int = 4,
    gate_decay_interval: int = 256,
    topic_probability: float = 0.85,
    zipf_exponent: float = 1.15,
    seed: int = 0,
) -> CandidateCacheTrialResult:
    """Evaluate online candidate generation on a topic/noise stream."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if not 0 <= warmup_events < context_length:
        raise ValueError("warmup_events must be in [0, context_length)")
    if not 0 < hot_tokens < vocab_size:
        raise ValueError("hot_tokens must be in (0, vocab_size)")
    if not 0 < top_k <= capacity:
        raise ValueError("top_k must be in (0, capacity]")
    if not 0.0 <= topic_probability <= 1.0:
        raise ValueError("topic_probability must be in [0, 1]")
    if admission_threshold < 0:
        raise ValueError("admission_threshold must be non-negative")

    config = CandidateCacheConfig(
        vocab_size=vocab_size,
        capacity=capacity,
        ways=ways,
        routes=routes,
        score_bits=score_bits,
        decay_interval=decay_interval,
    )
    cache = LowBitCandidateCache(config)
    gate: LowBitDenseContext | None = None
    if admission_threshold > 0:
        gate = LowBitDenseContext(
            DenseContextConfig(
                vocab_size=vocab_size,
                banks=gate_banks,
                width=gate_width,
                bits=gate_bits,
                decay_interval=gate_decay_interval,
            )
        )
    rng = np.random.default_rng(seed)

    hits = 0
    eval_events = 0
    admitted = 0
    skipped = 0
    gate_touched = 0
    for step in range(context_length):
        token = sample_zipf_topic_token(
            vocab_size=vocab_size,
            hot_tokens=hot_tokens,
            topic_probability=topic_probability,
            zipf_exponent=zipf_exponent,
            rng=rng,
        )
        if step >= warmup_events:
            eval_events += 1
            hits += int(token in cache.topk_set(top_k))

        admit = True
        if gate is not None:
            gate_touched += gate.config.banks
            admit = gate.estimate(token) >= admission_threshold
            gate_touched += gate.update(token)

        if admit:
            admitted += 1
            cache.observe(token)
        else:
            skipped += 1

    gate_state_bytes = gate.memory_bytes() if gate is not None else 0.0
    total_updates = admitted + skipped

    return CandidateCacheTrialResult(
        context_length=context_length,
        eval_events=eval_events,
        vocab_size=vocab_size,
        hot_tokens=hot_tokens,
        top_k=top_k,
        capacity=capacity,
        ways=ways,
        routes=routes,
        score_bits=score_bits,
        admission_threshold=admission_threshold,
        state_bytes=cache.memory_bytes(),
        gate_state_bytes=gate_state_bytes,
        total_state_bytes=cache.memory_bytes() + gate_state_bytes,
        topk_hit_rate=hits / eval_events if eval_events else 0.0,
        admission_rate=admitted / total_updates if total_updates else 0.0,
        cache_update_hit_rate=cache.cache_update_hit_rate(),
        avg_gate_cells=gate_touched / context_length,
        avg_local_update_cells=cache.local_touched_cells / context_length,
        avg_decay_cells=cache.decay_touched_cells / context_length,
        avg_total_update_cells=(
            gate_touched + cache.local_touched_cells + cache.decay_touched_cells
        )
        / context_length,
        admission_skips=skipped,
        replacements=cache.replacements,
        resident_tokens=cache.resident_count(),
    )
