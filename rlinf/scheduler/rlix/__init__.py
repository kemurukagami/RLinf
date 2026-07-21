"""RLix integration helpers owned by RLinf."""

from typing import Any

from .placement import (
    ResolvedGPUWorker,
    ResolvedPlacementStrategy,
    ResolvedRLixPlacements,
    RLixPlacementPlan,
    build_rlix_placement_plan,
    resolve_rlix_placements,
)
from .progress import ElasticPipelineProgress, ElasticProgressTracker
from .protocol import CoordinatorStatus, ElasticCollectionContext, PolicySyncLease

__all__ = [
    "CoordinatorStatus",
    "ElasticCollectionContext",
    "ElasticPipelineProgress",
    "ElasticProgressTracker",
    "PolicySyncLease",
    "RLixPlacementPlan",
    "RLixResizeCoordinator",
    "RLixStageController",
    "RegisteredRLixPipeline",
    "ResolvedGPUWorker",
    "ResolvedPlacementStrategy",
    "ResolvedRLixPlacements",
    "bootstrap_registered_rlix_pipeline",
    "build_rlix_placement_plan",
    "resolve_rlix_placements",
]


def __getattr__(name: str) -> Any:
    """Load Ray/rlix-core-backed integration surfaces only when requested."""
    if name == "RLixResizeCoordinator":
        from .coordinator import RLixResizeCoordinator

        return RLixResizeCoordinator
    if name == "RLixStageController":
        from .controller import RLixStageController

        return RLixStageController
    if name in {"RegisteredRLixPipeline", "bootstrap_registered_rlix_pipeline"}:
        from .runtime import (
            RegisteredRLixPipeline,
            bootstrap_registered_rlix_pipeline,
        )

        return {
            "RegisteredRLixPipeline": RegisteredRLixPipeline,
            "bootstrap_registered_rlix_pipeline": bootstrap_registered_rlix_pipeline,
        }[name]
    raise AttributeError(name)
