"""Tile-level chip mapping proxies for HARC-CA.

This is a floorplan accounting model, not a PDK area or power model. It maps the
current HARC-CA event profile onto a configurable grid of local-SRAM tiles and
reports capacity, local bandwidth demand, and utilization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .efficiency import EfficiencyComparison


@dataclass(frozen=True)
class TileConfig:
    """One repeated HARC-CA tile."""

    cells_per_tile: int = 64
    local_sram_bytes: int = 16 * 1024
    local_bytes_per_cycle: int = 32
    route_lanes: int = 4
    rule_banks: int = 6

    def __post_init__(self) -> None:
        if self.cells_per_tile <= 0:
            raise ValueError("cells_per_tile must be positive")
        if self.local_sram_bytes <= 0:
            raise ValueError("local_sram_bytes must be positive")
        if self.local_bytes_per_cycle <= 0:
            raise ValueError("local_bytes_per_cycle must be positive")
        if self.route_lanes <= 0:
            raise ValueError("route_lanes must be positive")
        if self.rule_banks <= 0:
            raise ValueError("rule_banks must be positive")


@dataclass(frozen=True)
class ChipConfig:
    """Configurable HARC-CA tile fabric."""

    tiles: int = 256
    frequency_mhz: float = 1000.0
    target_events_per_second: float = 1_000_000.0

    def __post_init__(self) -> None:
        if self.tiles <= 0:
            raise ValueError("tiles must be positive")
        if self.frequency_mhz <= 0.0:
            raise ValueError("frequency_mhz must be positive")
        if self.target_events_per_second <= 0.0:
            raise ValueError("target_events_per_second must be positive")


@dataclass(frozen=True)
class ChipProfile:
    """Tile-level proxy profile for one HARC-CA configuration."""

    tiles: int
    cells: int
    frequency_mhz: float
    total_local_sram_bytes: float
    state_bytes: float
    state_utilization: float
    state_tiles_required: int
    local_bytes_per_event: float
    target_events_per_second: float
    required_local_bandwidth_bytes_per_second: float
    peak_local_bandwidth_bytes_per_second: float
    bandwidth_utilization: float
    max_events_per_second_proxy: float
    local_cycles_per_event_proxy: float
    local_cycles_per_event_per_tile_proxy: float


def profile_chip(
    comparison: EfficiencyComparison,
    tile: TileConfig | None = None,
    chip: ChipConfig | None = None,
) -> ChipProfile:
    """Map a HARC event profile onto a tile fabric."""

    tile_config = tile or TileConfig()
    chip_config = chip or ChipConfig()
    harc = comparison.harc

    total_sram = chip_config.tiles * tile_config.local_sram_bytes
    state_tiles_required = ceil(harc.onchip_state_bytes / tile_config.local_sram_bytes)
    peak_bw = (
        chip_config.tiles
        * tile_config.local_bytes_per_cycle
        * chip_config.frequency_mhz
        * 1_000_000.0
    )
    required_bw = harc.total_local_bytes_per_event * chip_config.target_events_per_second
    max_events = peak_bw / harc.total_local_bytes_per_event if harc.total_local_bytes_per_event else 0.0
    local_cycles = harc.total_local_bytes_per_event / tile_config.local_bytes_per_cycle
    local_cycles_per_tile = local_cycles / chip_config.tiles

    return ChipProfile(
        tiles=chip_config.tiles,
        cells=chip_config.tiles * tile_config.cells_per_tile,
        frequency_mhz=chip_config.frequency_mhz,
        total_local_sram_bytes=float(total_sram),
        state_bytes=harc.onchip_state_bytes,
        state_utilization=harc.onchip_state_bytes / total_sram,
        state_tiles_required=state_tiles_required,
        local_bytes_per_event=harc.total_local_bytes_per_event,
        target_events_per_second=chip_config.target_events_per_second,
        required_local_bandwidth_bytes_per_second=required_bw,
        peak_local_bandwidth_bytes_per_second=peak_bw,
        bandwidth_utilization=required_bw / peak_bw,
        max_events_per_second_proxy=max_events,
        local_cycles_per_event_proxy=local_cycles,
        local_cycles_per_event_per_tile_proxy=local_cycles_per_tile,
    )
