"""Cellular Transformer research prototypes."""

from .benchmarks import MemoryTaskResult, run_memory_task, sweep_memory_tasks
from .dense_context import (
    DenseContextConfig,
    DenseContextResult,
    LowBitDenseContext,
    run_dense_context_trial,
)
from .propagation import (
    PropagationResult,
    harc_ca_edges,
    line_edges,
    propagation_distances,
    shortest_propagation_steps,
)
from .retrieval import (
    HashRouteCAM,
    HashRouteCAMConfig,
    InsertResult,
    LookupResult,
    RecallTrialResult,
    TieredHashRouteCAM,
    TieredHashRouteCAMConfig,
    TieredLookupResult,
    run_recall_trial,
)

__all__ = [
    "HashRouteCAM",
    "HashRouteCAMConfig",
    "DenseContextConfig",
    "DenseContextResult",
    "InsertResult",
    "LookupResult",
    "LowBitDenseContext",
    "MemoryTaskResult",
    "PropagationResult",
    "RecallTrialResult",
    "TieredHashRouteCAM",
    "TieredHashRouteCAMConfig",
    "TieredLookupResult",
    "harc_ca_edges",
    "line_edges",
    "propagation_distances",
    "run_dense_context_trial",
    "run_memory_task",
    "run_recall_trial",
    "shortest_propagation_steps",
    "sweep_memory_tasks",
]
