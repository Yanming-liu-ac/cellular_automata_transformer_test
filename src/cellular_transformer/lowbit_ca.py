"""Low-bit cellular automaton primitives.

This is not the final language model. It is a hardware-shaped integer simulator
for the kind of cell state HARC-CA should eventually train and compile into.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LowBitConfig:
    """Configuration for a low-bit 1D CA tile."""

    length: int
    channels: int = 32
    bits: int = 4
    radius: int = 1

    @property
    def levels(self) -> int:
        return 1 << self.bits


class LowBitCA:
    """A small saturating integer CA.

    The update is deliberately simple and hardware-shaped:

    - local neighbor reads only;
    - unsigned low-bit state;
    - one-step saturating residual movement toward a local average;
    - no floating point and no dense matrix multiply.

    Later training code should replace the hand-written update with a learned
    quantized rule while preserving this execution shape.
    """

    def __init__(self, config: LowBitConfig, seed: int = 0) -> None:
        if config.length <= 0:
            raise ValueError("length must be positive")
        if config.channels <= 0:
            raise ValueError("channels must be positive")
        if config.bits not in (1, 2, 4, 8):
            raise ValueError("bits must be one of 1, 2, 4, 8")
        if config.radius <= 0:
            raise ValueError("radius must be positive")

        self.config = config
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros((config.length, config.channels), dtype=np.uint8)

    def randomize(self, density: float = 0.5) -> None:
        """Initialize state with sparse low-bit activity."""

        if not 0.0 <= density <= 1.0:
            raise ValueError("density must be in [0, 1]")
        active = self.rng.random(self.state.shape) < density
        values = self.rng.integers(1, self.config.levels, self.state.shape, dtype=np.uint8)
        self.state[:] = np.where(active, values, 0).astype(np.uint8)

    def inject_byte(self, index: int, value: int, offset: int = 0) -> None:
        """Inject one byte as eight low-bit channels at a cell index."""

        if not 0 <= index < self.config.length:
            raise IndexError("index out of range")
        if not 0 <= value <= 255:
            raise ValueError("value must be a byte")
        if offset < 0 or offset + 8 > self.config.channels:
            raise ValueError("not enough channels for byte injection")

        high = self.config.levels - 1
        for bit in range(8):
            self.state[index, offset + bit] = high if (value >> bit) & 1 else 0

    def read_byte(self, index: int, offset: int = 0) -> int:
        """Read eight channels as a thresholded byte."""

        if not 0 <= index < self.config.length:
            raise IndexError("index out of range")
        if offset < 0 or offset + 8 > self.config.channels:
            raise ValueError("not enough channels for byte read")

        threshold = self.config.levels // 2
        value = 0
        for bit in range(8):
            if int(self.state[index, offset + bit]) >= threshold:
                value |= 1 << bit
        return value

    def step(self) -> None:
        """Run one local, saturating, integer CA update."""

        current = self.state.astype(np.int16)
        total = current.copy()
        count = np.ones((self.config.length, 1), dtype=np.int16)

        for delta in range(1, self.config.radius + 1):
            left = np.empty_like(current)
            right = np.empty_like(current)
            left[0] = current[0]
            left[1:] = current[:-1]
            right[-1] = current[-1]
            right[:-1] = current[1:]
            total += left + right
            count += 2

        average = total // count
        direction = np.sign(average - current).astype(np.int16)
        updated = np.clip(current + direction, 0, self.config.levels - 1)
        self.state[:] = updated.astype(np.uint8)

    def active_fraction(self) -> float:
        """Fraction of nonzero cell-channel entries."""

        return float(np.count_nonzero(self.state)) / float(self.state.size)

    def checksum(self) -> int:
        """Stable integer checksum for quick regression checks."""

        weights = np.arange(1, self.config.channels + 1, dtype=np.uint64)
        return int(np.sum(self.state.astype(np.uint64) * weights) % (2**63 - 1))
