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


class ElasticValidationMode(str, Enum):
    """Cost/safety policy for elastic snapshot and residency validation."""

    DEEP = "deep"
    RECEIPT = "receipt"
    OFF = "off"

    @classmethod
    def parse(cls, value: object, *, allow_off: bool = True) -> ElasticValidationMode:
        """Parse a configured mode and reject unsafe/unknown values explicitly."""

        if isinstance(value, cls):
            mode = value
        elif isinstance(value, str):
            try:
                mode = cls(value.lower())
            except ValueError as exc:
                choices = "deep, receipt, off" if allow_off else "deep, receipt"
                raise ValueError(f"validation mode must be one of: {choices}") from exc
        else:
            raise TypeError("validation mode must be a string")
        if mode is cls.OFF and not allow_off:
            raise ValueError("snapshot validation mode does not support off")
        return mode


@dataclass(frozen=True, slots=True)
class ResidencyOperationReceipt:
    """Identity-bound evidence produced after one synchronized residency move.

    This receipt deliberately records what the owning worker actually completed;
    it is not a replacement for the movement implementation reporting byte-level
    accounting. ``moved_bytes`` remains optional until every model backend exposes
    that information.
    """

    worker_rank: int
    lifecycle_generation: int
    policy_version: int
    operation_generation: int
    resident: bool
    destination_device: str
    synchronized: bool
    moved_bytes: int | None = None

    def __post_init__(self) -> None:
        _validate_request_identity(
            request_id="residency-operation",
            worker_rank=self.worker_rank,
            lifecycle_generation=self.lifecycle_generation,
            policy_version=self.policy_version,
        )
        if not isinstance(self.operation_generation, int) or isinstance(
            self.operation_generation, bool
        ):
            raise TypeError("operation_generation must be an integer")
        if self.operation_generation <= 0:
            raise ValueError("operation_generation must be positive")
        expected_device = "accelerator" if self.resident else "cpu"
        if self.destination_device != expected_device:
            raise ValueError("residency receipt destination does not match state")
        if not self.synchronized:
            raise ValueError("residency receipt requires synchronized movement")
        if self.moved_bytes is not None and self.moved_bytes < 0:
            raise ValueError("moved_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class SnapshotValidationReceipt:
    """Opaque proof that one immutable private snapshot passed deep validation."""

    receipt_id: str
    worker_rank: int
    worker_world_size: int
    lifecycle_generation: int
    policy_version: int
    next_transition_id: RolloutTransitionIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_id, str) or not self.receipt_id:
            raise ValueError("receipt_id must be a non-empty string")
        _validate_request_identity(
            request_id=self.receipt_id,
            worker_rank=self.worker_rank,
            lifecycle_generation=self.lifecycle_generation,
            policy_version=self.policy_version,
        )
        if not isinstance(self.worker_world_size, int) or isinstance(
            self.worker_world_size, bool
        ):
            raise TypeError("worker_world_size must be an integer")
        if self.worker_world_size <= 0:
            raise ValueError("worker_world_size must be positive")
        if not isinstance(self.next_transition_id, RolloutTransitionIdentity):
            raise TypeError("next_transition_id must be a RolloutTransitionIdentity")


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
    ElasticRankState.COMPLETED: frozenset(
        {ElasticRankState.EXPANDING, ElasticRankState.FAILED_RESIDENT}
    ),
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
class ElasticRankProgress:
    """Durable progress reported by an activated environment rank."""

    dp_rank: int
    lifecycle_generation: int
    state: ElasticRankState
    assigned_trajectories: int
    completed_trajectories: int
    snapshot_ready: bool
    failed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.dp_rank, int) or isinstance(self.dp_rank, bool):
            raise TypeError("dp_rank must be an integer")
        if self.dp_rank < 0:
            raise ValueError("dp_rank must be non-negative")
        if not isinstance(self.lifecycle_generation, int) or isinstance(
            self.lifecycle_generation, bool
        ):
            raise TypeError("lifecycle_generation must be an integer")
        if self.lifecycle_generation <= 0:
            raise ValueError("lifecycle_generation must be positive")
        if not isinstance(self.state, ElasticRankState):
            raise TypeError("state must be an ElasticRankState")
        for name, value in (
            ("assigned_trajectories", self.assigned_trajectories),
            ("completed_trajectories", self.completed_trajectories),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.assigned_trajectories <= 0:
            raise ValueError("assigned_trajectories must be positive")
        if not 0 <= self.completed_trajectories <= self.assigned_trajectories:
            raise ValueError(
                "completed_trajectories must be between zero and assigned_trajectories"
            )
        if self.state is ElasticRankState.COMPLETED and (
            self.completed_trajectories != self.assigned_trajectories
        ):
            raise ValueError(
                "COMPLETED progress must include every assigned trajectory"
            )
        if self.failed != (self.state is ElasticRankState.FAILED_RESIDENT):
            raise ValueError("failed must match FAILED_RESIDENT state")


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
    validation_mode: ElasticValidationMode = ElasticValidationMode.DEEP
    operation_receipt: ResidencyOperationReceipt | None = None

    def __post_init__(self) -> None:
        if self.state is ElasticRankState.PAUSED:
            if self.model_resident or self.cuda_graph_captured:
                raise ValueError("PAUSED residency must be CPU-only with no CUDA graph")
        elif self.state is ElasticRankState.EXPANDING:
            if not self.model_resident:
                raise ValueError("EXPANDING residency must have a resident model")
        else:
            raise ValueError("Residency receipts require PAUSED or EXPANDING state")
        if not isinstance(self.validation_mode, ElasticValidationMode):
            raise TypeError("validation_mode must be an ElasticValidationMode")
        if self.validation_mode is ElasticValidationMode.RECEIPT:
            if self.operation_receipt is None:
                raise ValueError("receipt validation requires an operation receipt")
            if self.operation_receipt.resident != self.model_resident:
                raise ValueError("operation receipt residency does not match result")
        elif self.operation_receipt is not None:
            raise ValueError("operation receipt is only valid in receipt mode")


@dataclass(frozen=True, slots=True)
class CompletedResidencyReceipt:
    """Verified non-residency for terminal work without a resumable token."""

    worker_rank: int
    lifecycle_generation: int
    policy_version: int
    state: ElasticRankState
    model_resident: bool
    cuda_graph_captured: bool
    validation_mode: ElasticValidationMode = ElasticValidationMode.DEEP
    operation_receipt: ResidencyOperationReceipt | None = None

    def __post_init__(self) -> None:
        _validate_request_identity(
            request_id="completed-residency",
            worker_rank=self.worker_rank,
            lifecycle_generation=self.lifecycle_generation,
            policy_version=self.policy_version,
        )
        if self.state is not ElasticRankState.COMPLETED:
            raise ValueError("Completed residency receipts require COMPLETED state")
        if self.model_resident or self.cuda_graph_captured:
            raise ValueError(
                "Completed residency receipts require CPU-only state with no CUDA graph"
            )
        if not isinstance(self.validation_mode, ElasticValidationMode):
            raise TypeError("validation_mode must be an ElasticValidationMode")
        if self.validation_mode is ElasticValidationMode.RECEIPT:
            if self.operation_receipt is None or self.operation_receipt.resident:
                raise ValueError(
                    "receipt validation requires a non-resident operation receipt"
                )
        elif self.operation_receipt is not None:
            raise ValueError("operation receipt is only valid in receipt mode")


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
