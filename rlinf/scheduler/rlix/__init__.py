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
from .protocol import (
    CoordinatorStatus,
    ElasticBatchReceipt,
    ElasticCollectionContext,
    ElasticRankObservation,
    FixedStageResidencyReceipt,
    FixedWorkerResidency,
    PolicySyncLease,
    RunnerStageState,
)

__all__ = [
    "CoordinatorStatus",
    "ElasticBatchReceipt",
    "ElasticCollectionSession",
    "ElasticCollectionContext",
    "ElasticPipelineProgress",
    "ElasticProgressTracker",
    "ElasticRankObservation",
    "FixedStageResidencyReceipt",
    "FixedWorkerResidency",
    "PolicySyncLease",
    "RunnerStageState",
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
    if name in {
        "ElasticCollectionSession",
        "RegisteredRLixPipeline",
        "bootstrap_registered_rlix_pipeline",
    }:
        from .runtime import (
            ElasticCollectionSession,
            RegisteredRLixPipeline,
            bootstrap_registered_rlix_pipeline,
        )

        return {
            "ElasticCollectionSession": ElasticCollectionSession,
            "RegisteredRLixPipeline": RegisteredRLixPipeline,
            "bootstrap_registered_rlix_pipeline": bootstrap_registered_rlix_pipeline,
        }[name]
    raise AttributeError(name)
