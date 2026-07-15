"""Shared protocol types for local elastic embodied-rollout lifecycles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from rlinf.data.embodied_io_struct import RolloutTransitionIdentity


class ElasticRankState(str, Enum):
    """Local residency and execution state for one elastic DP shard."""

    INACTIVE_COLD = "inactive_cold"
    EXPANDING = "expanding"
    ACTIVE = "active"
    DRAIN_REQUESTED = "drain_requested"
    SNAPSHOTTING = "snapshotting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED_RESIDENT = "failed_resident"


_ALLOWED_STATE_TRANSITIONS = {
    ElasticRankState.INACTIVE_COLD: frozenset({ElasticRankState.EXPANDING}),
    ElasticRankState.EXPANDING: frozenset(
        {ElasticRankState.ACTIVE, ElasticRankState.FAILED_RESIDENT}
    ),
    ElasticRankState.ACTIVE: frozenset(
        {
            ElasticRankState.DRAIN_REQUESTED,
            ElasticRankState.COMPLETED,
            ElasticRankState.FAILED_RESIDENT,
        }
    ),
    ElasticRankState.DRAIN_REQUESTED: frozenset(
        {
            ElasticRankState.SNAPSHOTTING,
            ElasticRankState.COMPLETED,
            ElasticRankState.FAILED_RESIDENT,
        }
    ),
    ElasticRankState.SNAPSHOTTING: frozenset(
        {ElasticRankState.PAUSED, ElasticRankState.FAILED_RESIDENT}
    ),
    ElasticRankState.PAUSED: frozenset({ElasticRankState.EXPANDING}),
    ElasticRankState.COMPLETED: frozenset({ElasticRankState.EXPANDING}),
    ElasticRankState.FAILED_RESIDENT: frozenset(),
}


def validate_elastic_state_transition(
    current: ElasticRankState, target: ElasticRankState
) -> None:
    """Raise if a local elastic shard cannot make the requested transition."""

    if not isinstance(current, ElasticRankState):
        raise TypeError("current must be an ElasticRankState")
    if not isinstance(target, ElasticRankState):
        raise TypeError("target must be an ElasticRankState")
    if target not in _ALLOWED_STATE_TRANSITIONS[current]:
        raise ValueError(
            f"Invalid elastic rank state transition: {current.value} -> {target.value}"
        )


@dataclass(frozen=True, slots=True)
class DrainRequest:
    """Correlated request to drain one active local DP shard."""

    request_id: str
    worker_rank: int
    lifecycle_generation: int
    expected_policy_version: int

    def __post_init__(self) -> None:
        _validate_request_identity(
            request_id=self.request_id,
            worker_rank=self.worker_rank,
            lifecycle_generation=self.lifecycle_generation,
            policy_version=self.expected_policy_version,
        )


@dataclass(frozen=True, slots=True)
class SafePointToken:
    """Identity of a closed channel boundary that is safe to snapshot."""

    request_id: str
    worker_rank: int
    lifecycle_generation: int
    policy_version: int
    next_transition_id: RolloutTransitionIdentity

    def __post_init__(self) -> None:
        _validate_request_identity(
            request_id=self.request_id,
            worker_rank=self.worker_rank,
            lifecycle_generation=self.lifecycle_generation,
            policy_version=self.policy_version,
        )
        if not isinstance(self.next_transition_id, RolloutTransitionIdentity):
            raise TypeError("next_transition_id must be a RolloutTransitionIdentity")
        if self.next_transition_id.env_worker_rank != self.worker_rank:
            raise ValueError("safe-point token worker rank does not match transition")
        if self.next_transition_id.lifecycle_generation != self.lifecycle_generation:
            raise ValueError("safe-point token lifecycle does not match transition")


class ElasticRunOutcome(str, Enum):
    """Terminal outcome of a local elastic worker run invocation."""

    PAUSE_READY = "pause_ready"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ElasticRankStatus:
    """Inspectable local state for one elastic worker rank."""

    state: ElasticRankState
    worker_rank: int
    lifecycle_generation: int | None
    policy_version: int | None
    expected_transition_id: RolloutTransitionIdentity | None
    drain_request_id: str | None
    snapshot_ready: bool
    model_resident: bool | None
    cuda_graph_captured: bool | None
    failure: str | None


@dataclass(frozen=True, slots=True)
class ElasticRunResult:
    """Result returned when a worker pauses at a boundary or completes."""

    outcome: ElasticRunOutcome
    token: SafePointToken | None
    metrics: dict[str, torch.Tensor] | None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ElasticRunOutcome):
            raise TypeError("outcome must be an ElasticRunOutcome")
        if self.outcome is ElasticRunOutcome.PAUSE_READY and self.token is None:
            raise ValueError("PAUSE_READY requires a safe-point token")
        if self.outcome is ElasticRunOutcome.COMPLETED and self.token is not None:
            raise ValueError("COMPLETED cannot contain a safe-point token")


@dataclass(frozen=True, slots=True)
class ResidencyReceipt:
    """Verified model residency after a pause or resume operation."""

    token: SafePointToken
    state: ElasticRankState
    model_resident: bool
    cuda_graph_captured: bool

    def __post_init__(self) -> None:
        if self.state is ElasticRankState.PAUSED:
            if self.model_resident or self.cuda_graph_captured:
                raise ValueError("PAUSED residency must be CPU-only with no CUDA graph")
        elif self.state is ElasticRankState.EXPANDING:
            if not self.model_resident:
                raise ValueError("EXPANDING residency must have a resident model")
        else:
            raise ValueError("Residency receipts require PAUSED or EXPANDING state")


def _validate_request_identity(
    *,
    request_id: str,
    worker_rank: int,
    lifecycle_generation: int,
    policy_version: int,
) -> None:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a non-empty string")
    values = {
        "worker_rank": worker_rank,
        "lifecycle_generation": lifecycle_generation,
        "policy_version": policy_version,
    }
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if worker_rank < 0:
        raise ValueError("worker_rank must be non-negative")
    if lifecycle_generation <= 0:
        raise ValueError("lifecycle_generation must be positive")
    if policy_version < 0:
        raise ValueError("policy_version must be non-negative")
