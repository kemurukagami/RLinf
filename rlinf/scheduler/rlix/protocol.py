"""Serializable protocol types for the RLinf resize coordinator."""

from __future__ import annotations

from dataclasses import dataclass


def _validate_non_negative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


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
        if not isinstance(self.pipeline_id, str) or not self.pipeline_id:
            raise ValueError("pipeline_id must be a non-empty string")
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
