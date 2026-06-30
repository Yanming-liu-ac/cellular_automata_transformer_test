"""Cellular-MoE rule-bank prototype.

DeepSeekMoE's useful lesson for HARC-CA is sparse activation and careful
load-balanced routing. This module implements a low-bit CA fabric where active
cells route to a small number of local rule banks instead of running every rule
every tick.

The rules are hand-written integer kernels for now. A future trainable model can
replace the router and rule banks while preserving this execution shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class CellularMoEConfig:
    """Configuration for the low-bit Cellular-MoE prototype."""

    length: int = 1024
    channels: int = 16
    bits: int = 4
    rule_banks: int = 6
    top_k: int = 1
    active_budget_fraction: float = 0.25
    balance_rate: float = 0.15
    bias_clip: float = 2.0

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.bits not in (2, 4, 8):
            raise ValueError("bits must be one of 2, 4, 8")
        if self.rule_banks != 6:
            raise ValueError("this prototype currently expects exactly 6 rule banks")
        if not 1 <= self.top_k <= self.rule_banks:
            raise ValueError("top_k must be in [1, rule_banks]")
        if not 0.0 < self.active_budget_fraction <= 1.0:
            raise ValueError("active_budget_fraction must be in (0, 1]")
        if self.balance_rate < 0.0:
            raise ValueError("balance_rate must be non-negative")
        if self.bias_clip <= 0.0:
            raise ValueError("bias_clip must be positive")

    @property
    def max_value(self) -> int:
        return (1 << self.bits) - 1


@dataclass(frozen=True)
class MoEStepStats:
    """Metrics for one Cellular-MoE tick."""

    tick: int
    active_cells: int
    active_fraction: float
    sparse_rule_updates: int
    dense_rule_updates: int
    update_reduction: float
    load_cv: float
    saturation_fraction: float
    checksum: int
    rule_loads: tuple[int, ...]


@dataclass(frozen=True)
class MoERolloutResult:
    """Aggregate rollout metrics."""

    ticks: int
    avg_active_fraction: float
    avg_update_reduction: float
    avg_load_cv: float
    final_saturation_fraction: float
    final_checksum: int
    total_sparse_rule_updates: int
    total_dense_rule_updates: int
    final_rule_loads: tuple[int, ...]


class CellularMoE:
    """Low-bit cellular fabric with sparse rule-bank routing."""

    def __init__(self, config: CellularMoEConfig, seed: int = 0) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((config.length, config.channels), dtype=np.uint8)
        self.bias = np.zeros(config.rule_banks, dtype=np.float32)
        self.tick = 0

        self.router_weights = np.array(
            [
                [-0.6, -0.5, -0.2, -0.2, -0.1],  # hold / preserve
                [0.7, -0.2, 0.1, 0.1, 0.9],      # decay saturated regions
                [0.5, 1.2, 0.0, 0.0, 0.1],       # diffuse high-gradient regions
                [0.8, 0.8, 0.2, 0.2, 0.3],       # sharpen contrast
                [0.5, 0.4, 1.3, -0.2, 0.1],      # copy from left
                [0.5, 0.4, -0.2, 1.3, 0.1],      # copy from right
            ],
            dtype=np.float32,
        )

    def randomize_sparse(self, density: float = 0.08) -> None:
        """Initialize sparse low-bit activity."""

        if not 0.0 <= density <= 1.0:
            raise ValueError("density must be in [0, 1]")
        active = self.rng.random(self.state.shape) < density
        values = self.rng.integers(1, self.config.max_value + 1, self.state.shape, dtype=np.uint8)
        self.state[:] = np.where(active, values, 0).astype(np.uint8)

    def inject_patch(self, start: int, width: int, value: int | None = None) -> None:
        """Inject a contiguous activity patch."""

        end = min(self.config.length, start + width)
        if not 0 <= start < self.config.length or end <= start:
            raise ValueError("invalid patch range")
        patch_value = self.config.max_value if value is None else int(value)
        if not 0 <= patch_value <= self.config.max_value:
            raise ValueError("patch value outside low-bit range")
        self.state[start:end, :] = patch_value

    def _neighbors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        current = self.state.astype(np.int16)
        left = np.empty_like(current)
        right = np.empty_like(current)
        left[0] = current[0]
        left[1:] = current[:-1]
        right[-1] = current[-1]
        right[:-1] = current[1:]
        return current, left, right

    def _features(self) -> tuple[np.ndarray, np.ndarray]:
        current, left, right = self._neighbors()
        max_value = float(self.config.max_value)
        activity = current.mean(axis=1) / max_value
        gradient = np.abs(right - left).mean(axis=1) / max_value
        left_greater = np.maximum(left - current, 0).mean(axis=1) / max_value
        right_greater = np.maximum(right - current, 0).mean(axis=1) / max_value
        saturation = (current >= self.config.max_value).mean(axis=1)
        features = np.stack([activity, gradient, left_greater, right_greater, saturation], axis=1)

        activity_score = activity + 0.5 * gradient + 0.25 * saturation
        return features.astype(np.float32), activity_score.astype(np.float32)

    def _active_cells(self, activity_score: np.ndarray) -> np.ndarray:
        positive = np.flatnonzero(activity_score > 0.0)
        if len(positive) == 0:
            return positive

        budget = max(1, int(round(self.config.length * self.config.active_budget_fraction)))
        if len(positive) <= budget:
            return positive

        positive_scores = activity_score[positive]
        selected = np.argpartition(positive_scores, -budget)[-budget:]
        return np.sort(positive[selected])

    def _rule_delta(self, rule: int, cells: np.ndarray) -> np.ndarray:
        current, left, right = self._neighbors()
        cur = current[cells]
        lft = left[cells]
        rgt = right[cells]
        avg = (lft + rgt) // 2

        if rule == 0:
            return np.zeros_like(cur)
        if rule == 1:
            return -np.sign(cur)
        if rule == 2:
            return np.sign(avg - cur)
        if rule == 3:
            return np.sign(cur - avg)
        if rule == 4:
            return np.sign(lft - cur)
        if rule == 5:
            return np.sign(rgt - cur)
        raise ValueError(f"unknown rule bank: {rule}")

    def step(self) -> MoEStepStats:
        """Run one sparse Cellular-MoE tick."""

        features, activity_score = self._features()
        active = self._active_cells(activity_score)
        rule_loads = np.zeros(self.config.rule_banks, dtype=np.int64)

        if len(active) > 0:
            scores = features[active] @ self.router_weights.T + self.bias
            selected = np.argpartition(scores, -self.config.top_k, axis=1)[:, -self.config.top_k :]

            delta = np.zeros((len(active), self.config.channels), dtype=np.int16)
            for rule in range(self.config.rule_banks):
                rows = np.flatnonzero(selected == rule)
                if len(rows) == 0:
                    continue
                unique_rows = np.unique(rows // self.config.top_k)
                rule_loads[rule] = len(unique_rows)
                delta[unique_rows] += self._rule_delta(rule, active[unique_rows])

            delta = np.clip(delta, -1, 1)
            updated = np.clip(
                self.state[active].astype(np.int16) + delta,
                0,
                self.config.max_value,
            )
            self.state[active] = updated.astype(np.uint8)

            if self.config.balance_rate > 0.0:
                total = max(1, int(rule_loads.sum()))
                observed = rule_loads.astype(np.float32) / total
                target = np.full(self.config.rule_banks, 1.0 / self.config.rule_banks, dtype=np.float32)
                self.bias += self.config.balance_rate * (target - observed)
                self.bias = np.clip(self.bias, -self.config.bias_clip, self.config.bias_clip)

        self.tick += 1
        sparse_updates = int(len(active) * self.config.top_k)
        dense_updates = int(self.config.length * self.config.rule_banks)
        update_reduction = dense_updates / max(1, sparse_updates)
        mean_load = float(np.mean(rule_loads))
        load_cv = float(np.std(rule_loads) / mean_load) if mean_load > 0.0 else 0.0
        saturation_fraction = float(np.count_nonzero(self.state == self.config.max_value)) / float(self.state.size)
        checksum = int(np.sum(self.state.astype(np.uint64) * np.arange(1, self.config.channels + 1, dtype=np.uint64)))

        return MoEStepStats(
            tick=self.tick,
            active_cells=len(active),
            active_fraction=len(active) / self.config.length,
            sparse_rule_updates=sparse_updates,
            dense_rule_updates=dense_updates,
            update_reduction=update_reduction,
            load_cv=load_cv,
            saturation_fraction=saturation_fraction,
            checksum=checksum,
            rule_loads=tuple(int(x) for x in rule_loads.tolist()),
        )

    def rollout(self, ticks: int) -> MoERolloutResult:
        """Run several ticks and aggregate metrics."""

        if ticks <= 0:
            raise ValueError("ticks must be positive")

        stats: List[MoEStepStats] = []
        for _ in range(ticks):
            stats.append(self.step())

        total_sparse = sum(s.sparse_rule_updates for s in stats)
        total_dense = sum(s.dense_rule_updates for s in stats)
        return MoERolloutResult(
            ticks=ticks,
            avg_active_fraction=float(np.mean([s.active_fraction for s in stats])),
            avg_update_reduction=total_dense / max(1, total_sparse),
            avg_load_cv=float(np.mean([s.load_cv for s in stats])),
            final_saturation_fraction=stats[-1].saturation_fraction,
            final_checksum=stats[-1].checksum,
            total_sparse_rule_updates=total_sparse,
            total_dense_rule_updates=total_dense,
            final_rule_loads=stats[-1].rule_loads,
        )
