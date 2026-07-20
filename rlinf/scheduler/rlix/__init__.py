"""RLix integration helpers owned by RLinf."""

from typing import Any

from .progress import ElasticPipelineProgress, ElasticProgressTracker
from .protocol import CoordinatorStatus, ElasticCollectionContext, PolicySyncLease

__all__ = [
    "CoordinatorStatus",
    "ElasticCollectionContext",
    "ElasticPipelineProgress",
    "ElasticProgressTracker",
    "PolicySyncLease",
    "RLixResizeCoordinator",
    "RLixStageController",
]


def __getattr__(name: str) -> Any:
    """Load Ray/rlix-core-backed integration surfaces only when requested."""
    if name == "RLixResizeCoordinator":
        from .coordinator import RLixResizeCoordinator

        return RLixResizeCoordinator
    if name == "RLixStageController":
        from .controller import RLixStageController

        return RLixStageController
    raise AttributeError(name)
