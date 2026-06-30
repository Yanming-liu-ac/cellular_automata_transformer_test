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
from .synthetic_lm import DualPathSyntheticLM, SyntheticLMConfig, SyntheticLMResult, run_synthetic_lm_trial

__all__ = [
    "DualPathSyntheticLM",
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
    "SyntheticLMConfig",
    "SyntheticLMResult",
    "TieredHashRouteCAM",
    "TieredHashRouteCAMConfig",
    "TieredLookupResult",
    "harc_ca_edges",
    "line_edges",
    "propagation_distances",
    "run_dense_context_trial",
    "run_memory_task",
    "run_recall_trial",
    "run_synthetic_lm_trial",
    "shortest_propagation_steps",
    "sweep_memory_tasks",
]
