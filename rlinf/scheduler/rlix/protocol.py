"""Serializable protocol types for the RLinf resize coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rlix_core.protocol.validation import validate_pipeline_id


def _validate_non_negative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


class RunnerStageState(str, Enum):
    """Exclusive runner-visible allocation state for one registered pipeline."""

    INACTIVE = "inactive"
    FIXED_INITIALIZATION = "fixed_initialization"
    FIXED_POLICY_SYNC = "fixed_policy_sync"
    ELASTIC_COLLECTION = "elastic_collection"
    SEALED_COLLECTION = "sealed_collection"
    FIXED_ACTOR_TRAIN = "fixed_actor_train"
    FIXED_EVALUATION = "fixed_evaluation"
    FAILED_UNCERTAIN = "failed_uncertain"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class FixedStageResidencyReceipt:
    """Proof that every device in a fixed stage is safe to release."""

    cluster_name: str
    verified_devices: tuple[int, ...]
    all_workers_offloaded: bool
    policy_version: int | None = None

    def __post_init__(self) -> None:
        """Validate immutable release evidence before runtime consumption."""
        if not isinstance(self.cluster_name, str) or not self.cluster_name:
            raise ValueError("cluster_name must be a non-empty string")
        if not isinstance(self.verified_devices, tuple):
            raise TypeError("verified_devices must be a tuple")
        for device in self.verified_devices:
            _validate_non_negative_integer("verified device", device)
        if tuple(sorted(set(self.verified_devices))) != self.verified_devices:
            raise ValueError("verified_devices must be sorted and unique")
        if not isinstance(self.all_workers_offloaded, bool):
            raise TypeError("all_workers_offloaded must be a boolean")
        if self.policy_version is not None:
            _validate_non_negative_integer("policy_version", self.policy_version)


@dataclass(frozen=True, slots=True)
class FixedWorkerResidency:
    """Framework-neutral physical residency reported by one worker rank."""

    component: str
    rank: int
    model_resident: bool
    optimizer_resident: bool
    cuda_graph_captured: bool
    policy_version: int | None = None

    def __post_init__(self) -> None:
        """Validate one public worker residency observation."""
        if self.component not in {"actor", "rollout", "environment"}:
            raise ValueError(f"unsupported residency component {self.component!r}")
        _validate_non_negative_integer("worker rank", self.rank)
        for name in (
            "model_resident",
            "optimizer_resident",
            "cuda_graph_captured",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if self.policy_version is not None:
            _validate_non_negative_integer("policy_version", self.policy_version)

    @property
    def safe_to_release(self) -> bool:
        """Whether this worker proves it owns no live accelerator state."""
        return not (
            self.model_resident or self.optimizer_resident or self.cuda_graph_captured
        )


@dataclass(frozen=True, slots=True)
class ElasticBatchReceipt:
    """Immutable identity and completeness proof for one actor training batch."""

    lifecycle_generation: int
    policy_version: int
    contributing_dp_ranks: tuple[int, ...]
    expected_trajectories: int
    received_trajectories: int
    transition_count: int

    def __post_init__(self) -> None:
        """Validate batch identity and exact trajectory completeness."""
        _validate_non_negative_integer(
            "lifecycle_generation", self.lifecycle_generation
        )
        if self.lifecycle_generation == 0:
            raise ValueError("lifecycle_generation must be positive")
        _validate_non_negative_integer("policy_version", self.policy_version)
        if not isinstance(self.contributing_dp_ranks, tuple):
            raise TypeError("contributing_dp_ranks must be a tuple")
        if not self.contributing_dp_ranks:
            raise ValueError("contributing_dp_ranks must be non-empty")
        for rank in self.contributing_dp_ranks:
            _validate_non_negative_integer("contributing dp rank", rank)
        if tuple(sorted(set(self.contributing_dp_ranks))) != (
            self.contributing_dp_ranks
        ):
            raise ValueError("contributing_dp_ranks must be sorted and unique")
        _validate_non_negative_integer(
            "expected_trajectories", self.expected_trajectories
        )
        if self.expected_trajectories == 0:
            raise ValueError("expected_trajectories must be positive")
        _validate_non_negative_integer(
            "received_trajectories", self.received_trajectories
        )
        if self.received_trajectories != self.expected_trajectories:
            raise ValueError("received_trajectories must equal expected_trajectories")
        _validate_non_negative_integer("transition_count", self.transition_count)


@dataclass(frozen=True, slots=True)
class ElasticRankObservation:
    """One durable coordinator view used by the runner progress monitor."""

    dp_rank: int
    env_status: object
    rollout_status: object
    progress: object | None
    paired_results: tuple[object, object] | None
    callback_applied_active: bool
    failure: str | None

    def __post_init__(self) -> None:
        """Validate framework-neutral observation structure."""
        _validate_non_negative_integer("dp_rank", self.dp_rank)
        if not isinstance(self.callback_applied_active, bool):
            raise TypeError("callback_applied_active must be a boolean")
        if self.paired_results is not None and (
            not isinstance(self.paired_results, tuple) or len(self.paired_results) != 2
        ):
            raise TypeError("paired_results must be a two-item tuple or None")
        if self.failure is not None and (
            not isinstance(self.failure, str) or not self.failure
        ):
            raise ValueError("failure must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class ElasticCollectionContext:
    """Immutable lifecycle and rank identity for one elastic collection."""

    lifecycle_generation: int
    policy_version: int
    dp_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate lifecycle, version, and canonical rank identity."""
        _validate_non_negative_integer(
            "lifecycle_generation", self.lifecycle_generation
        )
        if self.lifecycle_generation == 0:
            raise ValueError("lifecycle_generation must be positive")
        _validate_non_negative_integer("policy_version", self.policy_version)
        if not isinstance(self.dp_ranks, tuple):
            raise TypeError("dp_ranks must be a tuple")
        if not self.dp_ranks:
            raise ValueError("dp_ranks must be non-empty")
        for rank in self.dp_ranks:
            _validate_non_negative_integer("dp rank", rank)
        if len(set(self.dp_ranks)) != len(self.dp_ranks):
            raise ValueError("dp_ranks must be unique")
        if self.dp_ranks != tuple(range(len(self.dp_ranks))):
            raise ValueError("dp_ranks must be the contiguous canonical rank set")


@dataclass(frozen=True, slots=True)
class PolicySyncLease:
    """Opaque exclusive lease around an all-rank policy synchronization."""

    lease_id: str
    expected_policy_version: int

    def __post_init__(self) -> None:
        """Validate the opaque identity and policy version."""
        if not isinstance(self.lease_id, str) or not self.lease_id:
            raise ValueError("lease_id must be a non-empty string")
        _validate_non_negative_integer(
            "expected_policy_version", self.expected_policy_version
        )


@dataclass(frozen=True, slots=True)
class CoordinatorStatus:
    """Read-only diagnostic state for one pipeline coordinator."""

    pipeline_id: str
    collection: ElasticCollectionContext | None
    callback_applied_active_ranks: tuple[int, ...]
    paused_ranks: tuple[int, ...]
    completed_ranks: tuple[int, ...]
    failed_ranks: tuple[int, ...]
    resize_in_progress: bool
    policy_sync_lease: PolicySyncLease | None

    def __post_init__(self) -> None:
        """Validate the diagnostic read model."""
        validate_pipeline_id(self.pipeline_id)
        rank_fields = (
            self.callback_applied_active_ranks,
            self.paused_ranks,
            self.completed_ranks,
            self.failed_ranks,
        )
        for ranks in rank_fields:
            if not isinstance(ranks, tuple):
                raise TypeError("coordinator rank sets must be tuples")
            if tuple(sorted(set(ranks))) != ranks:
                raise ValueError("coordinator rank sets must be sorted and unique")
            for rank in ranks:
                _validate_non_negative_integer("dp rank", rank)
        if not isinstance(self.resize_in_progress, bool):
            raise TypeError("resize_in_progress must be a boolean")
