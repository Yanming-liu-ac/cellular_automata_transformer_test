"""Hardware-facing proxy metrics for HARC-CA experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .propagation import edge_count, harc_ca_edges, max_distance_from_tail


@dataclass(frozen=True)
class HarcCAMetrics:
    """Proxy metrics for one HARC-CA context lattice."""

    context_length: int
    channels: int
    bits_per_channel: int
    ticks_per_token: int
    active_fraction: float
    nodes: int
    edges: int
    state_bytes: float
    local_message_bytes_per_token: float


@dataclass(frozen=True)
class TransformerKVMetrics:
    """KV-cache traffic proxy for autoregressive Transformer inference."""

    context_length: int
    layers: int
    heads: int
    head_dim: int
    kv_bits: int
    kv_cache_bytes: float
    kv_read_bytes_per_token: float


def estimate_harc_ca(
    context_length: int,
    channels: int = 128,
    bits_per_channel: int = 4,
    active_fraction: float = 1.0,
) -> HarcCAMetrics:
    """Estimate HARC-CA state and local message movement."""

    if not 0.0 < active_fraction <= 1.0:
        raise ValueError("active_fraction must be in (0, 1]")

    edges = harc_ca_edges(context_length)
    ticks = max_distance_from_tail(edges, context_length)
    nodes = len(edges)
    undirected_edges = edge_count(edges)
    bits_per_cell = channels * bits_per_channel
    state_bytes = nodes * bits_per_cell / 8

    directed_edge_messages = 2 * undirected_edges
    message_bits_per_tick = directed_edge_messages * bits_per_cell * active_fraction
    local_message_bytes_per_token = message_bits_per_tick * ticks / 8

    return HarcCAMetrics(
        context_length=context_length,
        channels=channels,
        bits_per_channel=bits_per_channel,
        ticks_per_token=ticks,
        active_fraction=active_fraction,
        nodes=nodes,
        edges=undirected_edges,
        state_bytes=state_bytes,
        local_message_bytes_per_token=local_message_bytes_per_token,
    )


def estimate_transformer_kv(
    context_length: int,
    layers: int = 12,
    heads: int = 8,
    head_dim: int = 64,
    kv_bits: int = 16,
) -> TransformerKVMetrics:
    """Estimate KV-cache storage and reads per generated token."""

    values_per_token = layers * 2 * heads * head_dim
    kv_cache_bytes = context_length * values_per_token * kv_bits / 8

    return TransformerKVMetrics(
        context_length=context_length,
        layers=layers,
        heads=heads,
        head_dim=head_dim,
        kv_bits=kv_bits,
        kv_cache_bytes=kv_cache_bytes,
        kv_read_bytes_per_token=kv_cache_bytes,
    )


def format_bytes(value: float) -> str:
    """Human-readable byte count."""

    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"
