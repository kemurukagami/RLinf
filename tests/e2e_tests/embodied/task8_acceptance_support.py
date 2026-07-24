"""Dependency-light evidence and analysis helpers for Task 8 acceptance."""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch

ACCEPTANCE_SCHEMA_VERSION = 1

_KNOWN_EVENTS = frozenset(
    {
        "connected",
        "registered",
        "admitted",
        "initialized",
        "stage_acquired",
        "stage_released",
        "generation_requested",
        "generation_granted",
        "progress_published",
        "batch_sealed",
        "reward_started",
        "reward_completed",
        "policy_synchronized",
        "advantages_computed",
        "training_started",
        "training_completed",
        "runtime_closed",
        "unregistered",
        "policy_request_started",
        "policy_request_completed",
        "chunk_started",
        "chunk_committed",
        "pending_bootstrap_retained",
        "drain_requested",
        "drain_observed",
        "barrier_consumed",
        "snapshot_started",
        "snapshot_completed",
        "environment_offload_started",
        "environment_offload_verified",
        "rollout_offload_started",
        "rollout_offload_verified",
        "environment_onload_started",
        "environment_onload_verified",
        "rollout_onload_started",
        "rollout_onload_verified",
        "restore_validated",
        "restore_committed",
        "resumed_bootstrap_dispatched",
        "rank_completed",
        "request_enqueued",
        "resize_plan_created",
        "callback_started",
        "callback_completed",
        "callback_failed",
        "release_committed",
        "allocation_committed",
        "useful_cuda_work_started",
        "trace_flushed",
    }
)

_RANK_EVENTS = frozenset(
    {
        event
        for event in _KNOWN_EVENTS
        if event
        not in {
            "connected",
            "registered",
            "admitted",
            "initialized",
            "stage_acquired",
            "stage_released",
            "generation_requested",
            "generation_granted",
            "progress_published",
            "batch_sealed",
            "reward_started",
            "reward_completed",
            "policy_synchronized",
            "advantages_computed",
            "training_started",
            "training_completed",
            "runtime_closed",
            "unregistered",
            "trace_flushed",
        }
    }
)

_TRANSITION_EVENTS = frozenset(
    {
        "policy_request_started",
        "policy_request_completed",
        "chunk_started",
        "chunk_committed",
        "pending_bootstrap_retained",
        "drain_requested",
        "snapshot_started",
        "snapshot_completed",
        "restore_validated",
        "restore_committed",
        "resumed_bootstrap_dispatched",
    }
)

_BUNDLE_EVENTS = frozenset(
    {
        "generation_granted",
        "environment_offload_started",
        "environment_offload_verified",
        "rollout_offload_started",
        "rollout_offload_verified",
        "environment_onload_started",
        "environment_onload_verified",
        "rollout_onload_started",
        "rollout_onload_verified",
        "release_committed",
        "allocation_committed",
        "useful_cuda_work_started",
    }
)

_UNIQUE_EVENTS = frozenset(
    {
        "chunk_committed",
        "snapshot_completed",
        "environment_offload_verified",
        "rollout_offload_verified",
        "environment_onload_verified",
        "rollout_onload_verified",
        "restore_committed",
        "resumed_bootstrap_dispatched",
        "rank_completed",
        "batch_sealed",
        "training_completed",
        "runtime_closed",
        "unregistered",
    }
)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Immutable inputs and hard thresholds for one acceptance run."""

    run_id: str
    environment: str
    mode: str
    scenario: str
    expected_bundles: tuple[tuple[int, ...], ...]
    checkpoint_digests: Mapping[str, str]
    algorithm: str = "grpo"
    training_iterations: int = 2
    repetitions: int = 5
    warmup_repetitions: int = 1
    minimum_throughput_improvement: float = 0.05
    minimum_idle_reduction: float = 0.05
    sample_interval_ms: int = 100
    schema_version: int = ACCEPTANCE_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate topology, inputs, and predeclared measurement gates."""
        if self.schema_version != ACCEPTANCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported run schema version {self.schema_version}")
        if not self.run_id or any(char.isspace() for char in self.run_id):
            raise ValueError("run_id must be non-empty and contain no whitespace")
        if self.environment not in {"wan", "opensora"}:
            raise ValueError(f"unsupported environment {self.environment!r}")
        if self.mode not in {"disaggregated", "collocated"}:
            raise ValueError(f"unsupported placement mode {self.mode!r}")
        if self.scenario not in {"reference", "recovery", "utilization", "all"}:
            raise ValueError(f"unsupported scenario {self.scenario!r}")
        if self.algorithm != "grpo":
            raise ValueError("Task 8 acceptance requires algorithm='grpo'")
        if self.training_iterations < 2:
            raise ValueError(
                "Task 8 requires at least two GRPO iterations to prove policy reuse"
            )
        if len(self.expected_bundles) < 2:
            raise ValueError("Task 8 requires at least two actor-infer bundles")
        widths = {len(bundle) for bundle in self.expected_bundles}
        expected_width = 2 if self.mode == "disaggregated" else 1
        if widths != {expected_width}:
            raise ValueError(
                f"{self.mode} mode requires uniform bundle width {expected_width}"
            )
        flattened = [gpu_id for bundle in self.expected_bundles for gpu_id in bundle]
        if any(
            not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0
            for gpu_id in flattened
        ) or len(flattened) != len(set(flattened)):
            raise ValueError(
                "expected bundles must contain disjoint non-negative GPU IDs"
            )
        if set(self.checkpoint_digests) != {"vla", self.environment} or any(
            not digest for digest in self.checkpoint_digests.values()
        ):
            raise ValueError(
                "checkpoint_digests must contain non-empty VLA and environment digests"
            )
        if self.repetitions < 5 or self.warmup_repetitions < 1:
            raise ValueError("Task 8 requires at least one warmup and five repetitions")
        if not 0.0 <= self.minimum_throughput_improvement <= 1.0:
            raise ValueError("minimum_throughput_improvement must be within [0, 1]")
        if not 0.0 <= self.minimum_idle_reduction <= 1.0:
            raise ValueError("minimum_idle_reduction must be within [0, 1]")
        maximum_interval = 250 if self.scenario == "utilization" else 100
        if not 1 <= self.sample_interval_ms <= maximum_interval:
            raise ValueError(
                f"sample_interval_ms must be within [1, {maximum_interval}]"
            )


@dataclass(frozen=True, slots=True)
class Task8RoleNames:
    """Collision-free RLinf-owned names for one acceptance driver role."""

    prefix: str
    actor_group: str
    rollout_group: str
    env_group: str
    env_input_channel: str
    rollout_request_channel: str
    actor_channel: str
    event_producer: str

    def validate(self) -> None:
        """Require one shared prefix and no duplicate role-owned identity."""
        values = tuple(asdict(self).values())
        if any(not value or any(char.isspace() for char in value) for value in values):
            raise ValueError("Task 8 role names must be non-empty without whitespace")
        owned = values[1:]
        if len(set(owned)) != len(owned):
            raise ValueError("Task 8 role-owned names must be unique")
        if any(not value.startswith(f"{self.prefix}_") for value in owned):
            raise ValueError("Task 8 role-owned names must use the role prefix")

    def runner_channel_names(self) -> dict[str, str]:
        """Project role identities into the synchronous runner channel contract."""
        self.validate()
        return {
            "env": self.env_input_channel,
            "rollout": self.rollout_request_channel,
            "actor": self.actor_channel,
        }


def derive_role_names(*, run_id: str, role: str) -> Task8RoleNames:
    """Derive deterministic collision-free worker and channel identities."""
    if not run_id or any(char.isspace() for char in run_id):
        raise ValueError("run_id must be non-empty and contain no whitespace")
    if role not in {"a", "b"}:
        raise ValueError("Task 8 driver role must be 'a' or 'b'")
    prefix = f"t8_{run_id}_{role}"
    names = Task8RoleNames(
        prefix=prefix,
        actor_group=f"{prefix}_ActorGroup",
        rollout_group=f"{prefix}_RolloutGroup",
        env_group=f"{prefix}_EnvGroup",
        env_input_channel=f"{prefix}_env_input",
        rollout_request_channel=f"{prefix}_rollout_request",
        actor_channel=f"{prefix}_actor_batch",
        event_producer=f"{prefix}_event_producer",
    )
    names.validate()
    return names


@dataclass(frozen=True, slots=True)
class TransitionIdentity:
    """JSON-safe transition identity used by acceptance evidence."""

    worker_rank: int
    lifecycle_generation: int
    episode_generation: int
    transition_id: int

    def validate(self) -> None:
        """Reject malformed or negative identity fields."""
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"transition {name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class AcceptanceEvent:
    """One producer event before or after central sink sequencing."""

    run_id: str
    producer_sequence: int
    producer_time_ns: int
    producer_pid: int
    driver_role: str
    component: str
    event: str
    pipeline_id: str | None = None
    lifecycle_generation: int | None = None
    policy_version: int | None = None
    dp_rank: int | None = None
    transition_identity: TransitionIdentity | None = None
    gpu_ids: tuple[int, ...] = ()
    details: Mapping[str, Any] | None = None
    schema_version: int = ACCEPTANCE_SCHEMA_VERSION
    sink_sequence: int | None = None
    sink_time_ns: int | None = None

    def validate(self) -> None:
        """Validate the frozen wire contract and event-specific identity."""
        if self.schema_version != ACCEPTANCE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported acceptance schema version {self.schema_version}"
            )
        if not self.run_id or any(char.isspace() for char in self.run_id):
            raise ValueError("run_id must be non-empty and contain no whitespace")
        if self.driver_role not in {"a", "b", "core", "orchestrator"}:
            raise ValueError(f"unknown driver role {self.driver_role!r}")
        if not self.component:
            raise ValueError("component must not be empty")
        if self.event not in _KNOWN_EVENTS:
            raise ValueError(f"unknown acceptance event {self.event!r}")
        for name in ("producer_sequence", "producer_time_ns", "producer_pid"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.sink_sequence is not None and self.sink_sequence < 0:
            raise ValueError("sink_sequence must be non-negative")
        if self.sink_time_ns is not None and self.sink_time_ns < 0:
            raise ValueError("sink_time_ns must be non-negative")
        if len(set(self.gpu_ids)) != len(self.gpu_ids) or any(
            not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0
            for gpu_id in self.gpu_ids
        ):
            raise ValueError("gpu_ids must be unique non-negative integers")
        if self.event not in {"connected", "trace_flushed"} and not self.pipeline_id:
            raise ValueError(f"{self.event} requires pipeline_id")
        if self.event in _RANK_EVENTS:
            if self.dp_rank is None or self.dp_rank < 0:
                raise ValueError(f"{self.event} requires a non-negative dp_rank")
            if self.lifecycle_generation is None or self.lifecycle_generation < 0:
                raise ValueError(
                    f"{self.event} requires a non-negative lifecycle_generation"
                )
        if self.transition_identity is not None:
            self.transition_identity.validate()
            if (
                self.dp_rank is not None
                and self.transition_identity.worker_rank != self.dp_rank
            ):
                raise ValueError("transition worker rank does not match event dp_rank")
            if (
                self.lifecycle_generation is not None
                and self.transition_identity.lifecycle_generation
                != self.lifecycle_generation
            ):
                raise ValueError("transition lifecycle generation does not match event")
        if self.event in _TRANSITION_EVENTS and self.transition_identity is None:
            raise ValueError(f"{self.event} requires transition_identity")
        if self.event in _TRANSITION_EVENTS and self.policy_version is None:
            raise ValueError(f"{self.event} requires policy_version")
        if self.event in _BUNDLE_EVENTS and not self.gpu_ids:
            raise ValueError(f"{self.event} requires a non-empty GPU bundle")
        if self.details is not None:
            if not isinstance(self.details, Mapping):
                raise ValueError("event details must be a mapping")
            normalize_manifest(self.details)


class AcceptanceEventSink:
    """Assign one total order while enforcing producer and terminal uniqueness."""

    def __init__(self, *, run_id: str, clock_ns: Callable[[], int] = time.time_ns):
        self._run_id = run_id
        self._clock_ns = clock_ns
        self._events: list[AcceptanceEvent] = []
        self._producer_sequences: dict[tuple[str, int, str, int | None], int] = {}
        self._unique_keys: set[tuple[Any, ...]] = set()

    @property
    def events(self) -> tuple[AcceptanceEvent, ...]:
        """Return an immutable snapshot of accepted events."""
        return tuple(self._events)

    def record(self, event: AcceptanceEvent) -> AcceptanceEvent:
        """Validate, centrally stamp, and append one event."""
        event.validate()
        if event.run_id != self._run_id:
            raise ValueError(
                f"event run_id {event.run_id!r} does not match sink {self._run_id!r}"
            )
        if event.sink_sequence is not None or event.sink_time_ns is not None:
            raise ValueError("producer event must not preassign sink fields")
        producer_key = (
            event.driver_role,
            event.producer_pid,
            event.component,
            event.dp_rank,
        )
        previous = self._producer_sequences.get(producer_key)
        if previous is not None and event.producer_sequence <= previous:
            raise ValueError(
                "producer_sequence must increase strictly for each producer"
            )
        unique_key = (
            event.pipeline_id,
            event.lifecycle_generation,
            event.component,
            event.dp_rank,
            event.event,
            event.transition_identity,
        )
        if event.event in _UNIQUE_EVENTS and unique_key in self._unique_keys:
            raise ValueError(f"duplicate terminal event {event.event}")
        sink_time_ns = self._clock_ns()
        if self._events and sink_time_ns < self._events[-1].sink_time_ns:
            raise ValueError("sink clock must not regress")
        stamped = replace(
            event,
            sink_sequence=len(self._events),
            sink_time_ns=sink_time_ns,
        )
        self._producer_sequences[producer_key] = event.producer_sequence
        if event.event in _UNIQUE_EVENTS:
            self._unique_keys.add(unique_key)
        self._events.append(stamped)
        return stamped


@dataclass(frozen=True, slots=True)
class AcceptanceProducerContext:
    """Mutable-lifecycle identity sampled for one worker or runner event."""

    pipeline_id: str
    lifecycle_generation: int | None
    policy_version: int | None
    transition_identity: TransitionIdentity | None = None
    gpu_ids: tuple[int, ...] = ()


class AcceptanceEventProducer:
    """Enrich acceptance hooks and submit fail-closed monotonic events."""

    def __init__(
        self,
        *,
        run_id: str,
        driver_role: str,
        component: str,
        dp_rank: int | None,
        context_provider: Callable[[], AcceptanceProducerContext],
        sink_record: Callable[[AcceptanceEvent], AcceptanceEvent],
        clock_ns: Callable[[], int] = time.time_ns,
        producer_pid: int | None = None,
    ) -> None:
        if not run_id or not component:
            raise ValueError("acceptance producer requires run_id and component")
        if driver_role not in {"a", "b", "core", "orchestrator"}:
            raise ValueError("acceptance producer has an invalid driver role")
        if dp_rank is not None and dp_rank < 0:
            raise ValueError("acceptance producer rank must be non-negative")
        self._run_id = run_id
        self._driver_role = driver_role
        self._component = component
        self._dp_rank = dp_rank
        self._context_provider = context_provider
        self._sink_record = sink_record
        self._clock_ns = clock_ns
        self._producer_pid = os.getpid() if producer_pid is None else producer_pid
        self._next_sequence = 0

    def record(self, observation: Any) -> AcceptanceEvent:
        """Build, validate, submit, and only then advance producer sequence."""
        event_name = getattr(observation, "event", None)
        details = getattr(observation, "details", None)
        if not isinstance(event_name, str) or not isinstance(details, Mapping):
            raise TypeError("worker observation must provide event and details")
        context = self._context_provider()
        if not isinstance(context, AcceptanceProducerContext):
            raise TypeError("context provider must return AcceptanceProducerContext")
        event = AcceptanceEvent(
            run_id=self._run_id,
            producer_sequence=self._next_sequence,
            producer_time_ns=self._clock_ns(),
            producer_pid=self._producer_pid,
            driver_role=self._driver_role,
            pipeline_id=context.pipeline_id,
            lifecycle_generation=context.lifecycle_generation,
            policy_version=context.policy_version,
            component=self._component,
            dp_rank=self._dp_rank,
            event=event_name,
            transition_identity=context.transition_identity,
            gpu_ids=context.gpu_ids,
            details=details,
        )
        stamped = self._sink_record(event)
        self._next_sequence += 1
        return stamped


def _canonical_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("acceptance manifests require CPU tensors")
        return np.ascontiguousarray(value.detach().numpy())
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    raise TypeError(f"expected tensor or ndarray, got {type(value).__name__}")


def normalize_manifest(value: Any, *, inline_value_limit: int = 4096) -> Any:
    """Convert nested evidence to deterministic JSON-safe typed records."""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        array = _canonical_array(value)
        if (
            np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, np.complexfloating)
        ) and not np.isfinite(array).all():
            raise ValueError("acceptance manifests reject non-finite tensor values")
        raw = array.tobytes(order="C")
        record: dict[str, Any] = {
            "kind": "tensor",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "logical_bytes": len(raw),
        }
        if array.size <= inline_value_limit:
            record["values"] = array.reshape(-1).tolist()
        return record
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_manifest(asdict(value), inline_value_limit=inline_value_limit)
    if isinstance(value, Enum):
        return normalize_manifest(value.value, inline_value_limit=inline_value_limit)
    if isinstance(value, Mapping):
        return {
            str(key): normalize_manifest(item, inline_value_limit=inline_value_limit)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            normalize_manifest(item, inline_value_limit=inline_value_limit)
            for item in value
        ]
    if isinstance(value, np.generic):
        return normalize_manifest(value.item(), inline_value_limit=inline_value_limit)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("acceptance manifests reject non-finite scalar values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported manifest value {type(value).__name__}")


def compare_manifests(
    expected: Any,
    actual: Any,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> tuple[str, ...]:
    """Return stable paths for exact or tolerance-aware manifest differences."""
    differences: list[str] = []

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            if left.get("kind") == "tensor" or right.get("kind") == "tensor":
                if left.get("kind") != right.get("kind"):
                    differences.append(path)
                    return
                for key in ("shape", "dtype", "logical_bytes"):
                    if left.get(key) != right.get(key):
                        differences.append(f"{path}.{key}")
                left_values = left.get("values")
                right_values = right.get("values")
                if left_values is None or right_values is None:
                    if left.get("sha256") != right.get("sha256"):
                        differences.append(f"{path}.sha256")
                else:
                    compare(left_values, right_values, f"{path}.values")
                return
            keys = set(left) | set(right)
            for key in sorted(keys):
                if key not in left or key not in right:
                    differences.append(f"{path}.{key}")
                else:
                    compare(left[key], right[key], f"{path}.{key}")
            return
        if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
            if not isinstance(right, Sequence) or isinstance(right, (str, bytes)):
                differences.append(path)
                return
            if len(left) != len(right):
                differences.append(f"{path}.length")
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left, float) or isinstance(right, float):
            try:
                equal = math.isclose(
                    float(left), float(right), rel_tol=rtol, abs_tol=atol
                )
            except (TypeError, ValueError):
                equal = False
            if not equal:
                differences.append(path)
            return
        if left != right:
            differences.append(path)

    compare(expected, actual, "$")
    return tuple(differences)


def logical_tensor_bytes(value: Any) -> int:
    """Count recursively reachable tensor and ndarray payload bytes."""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return int(_canonical_array(value).nbytes)
    if is_dataclass(value) and not isinstance(value, type):
        return logical_tensor_bytes(asdict(value))
    if isinstance(value, Mapping):
        return sum(logical_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(logical_tensor_bytes(item) for item in value)
    return 0


@dataclass(frozen=True, slots=True)
class AllocationSlice:
    """One half-open scheduler ownership interval for a canonical bundle."""

    pipeline_id: str
    dp_rank: int
    gpu_ids: tuple[int, ...]
    start_ns: int
    end_ns: int

    def validate(self) -> None:
        """Validate interval and canonical bundle fields."""
        if not self.pipeline_id:
            raise ValueError("allocation pipeline_id must not be empty")
        if self.dp_rank < 0:
            raise ValueError("allocation dp_rank must be non-negative")
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError("allocation must have a positive half-open interval")
        if not self.gpu_ids or len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("allocation bundle must be non-empty and duplicate-free")


def validate_exclusive_ownership(slices: Iterable[AllocationSlice]) -> None:
    """Reject physical GPU ownership overlap across scheduler slices."""
    by_gpu: dict[int, list[AllocationSlice]] = {}
    for allocation in slices:
        allocation.validate()
        for gpu_id in allocation.gpu_ids:
            by_gpu.setdefault(gpu_id, []).append(allocation)
    for gpu_id, allocations in by_gpu.items():
        ordered = sorted(allocations, key=lambda item: (item.start_ns, item.end_ns))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_ns < previous.end_ns:
                raise ValueError(
                    "GPU ownership overlap for GPU "
                    f"{gpu_id}: {previous.pipeline_id} and {current.pipeline_id}"
                )


@dataclass(frozen=True, slots=True)
class SchedulerCommitRecord:
    """One per-rank ownership mutation extracted from a core commit marker."""

    cycle_counter: int
    commit_time_ns: int
    operation: str
    cluster_id: str
    pipeline_id: str
    dp_rank: int
    gpu_ids: tuple[int, ...]

    def validate(self) -> None:
        """Reject malformed, non-canonical scheduler commit evidence."""
        if self.cycle_counter < 0 or self.commit_time_ns < 0:
            raise ValueError(
                "scheduler commit cycle and timestamp must be non-negative"
            )
        if self.operation not in {"release", "allocation"}:
            raise ValueError(f"unknown scheduler commit operation {self.operation!r}")
        if not self.cluster_id or not self.pipeline_id:
            raise ValueError("scheduler commit cluster and pipeline must not be empty")
        if self.dp_rank < 0:
            raise ValueError("scheduler commit dp_rank must be non-negative")
        if (
            not self.gpu_ids
            or tuple(sorted(self.gpu_ids)) != self.gpu_ids
            or len(set(self.gpu_ids)) != len(self.gpu_ids)
            or any(gpu_id < 0 for gpu_id in self.gpu_ids)
        ):
            raise ValueError(
                "scheduler commit bundle must be non-empty, sorted, and duplicate-free"
            )


_COMMIT_MARKER_PATTERN = re.compile(r"^Commit C(?P<cycle>[0-9]+)$")


def normalize_scheduler_commit_marker(
    *,
    marker_name: str,
    timestamp_ns: int,
    payload: Mapping[str, Any],
    tracked_clusters: Mapping[str, str],
    canonical_bundles: Mapping[str, Mapping[int, Sequence[int]]],
) -> tuple[SchedulerCommitRecord, ...]:
    """Normalize one extracted Perfetto commit marker into trusted rank records.

    Only explicitly registered acceptance clusters are retained. An ``Exec``
    planning marker is deliberately rejected so pre-callback plans cannot be
    promoted to commit evidence.
    """
    marker_match = _COMMIT_MARKER_PATTERN.fullmatch(marker_name)
    if marker_match is None:
        raise ValueError(f"not a scheduler commit marker: {marker_name!r}")
    if (
        not isinstance(timestamp_ns, int)
        or isinstance(timestamp_ns, bool)
        or timestamp_ns < 0
    ):
        raise ValueError("scheduler commit timestamp must be a non-negative integer")
    if set(payload) != {"shrinks", "removes", "allocates", "expands"}:
        raise ValueError("scheduler commit payload has unexpected operation groups")

    cycle_counter = int(marker_match.group("cycle"))
    records: list[SchedulerCommitRecord] = []

    def add_mapping(
        *,
        operation: str,
        cluster_id: str,
        mapping: Mapping[Any, Any],
    ) -> None:
        pipeline_id = tracked_clusters.get(cluster_id)
        if pipeline_id is None:
            return
        expected_by_rank = canonical_bundles.get(pipeline_id)
        if expected_by_rank is None:
            raise ValueError(f"missing canonical bundles for pipeline {pipeline_id!r}")
        for raw_rank, raw_bundle in mapping.items():
            if isinstance(raw_rank, bool):
                raise ValueError("scheduler commit rank must be an integer")
            try:
                dp_rank = int(raw_rank)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid scheduler commit rank {raw_rank!r}") from exc
            if isinstance(raw_bundle, (str, bytes)) or not isinstance(
                raw_bundle, Sequence
            ):
                raise ValueError("scheduler commit bundle must be a sequence")
            if any(
                not isinstance(gpu_id, int) or isinstance(gpu_id, bool)
                for gpu_id in raw_bundle
            ):
                raise ValueError("scheduler commit GPU ids must be integers")
            gpu_ids = tuple(sorted(raw_bundle))
            expected = expected_by_rank.get(dp_rank)
            if expected is None:
                raise ValueError(
                    f"scheduler commit references unknown rank {pipeline_id}:{dp_rank}"
                )
            if gpu_ids != tuple(sorted(expected)):
                raise ValueError(
                    f"scheduler commit bundle mismatch for {pipeline_id}:{dp_rank}: "
                    f"got {gpu_ids}, expected {tuple(sorted(expected))}"
                )
            record = SchedulerCommitRecord(
                cycle_counter=cycle_counter,
                commit_time_ns=timestamp_ns,
                operation=operation,
                cluster_id=cluster_id,
                pipeline_id=pipeline_id,
                dp_rank=dp_rank,
                gpu_ids=gpu_ids,
            )
            record.validate()
            records.append(record)

    operation_groups = (
        ("shrinks", "release"),
        ("removes", "release"),
        ("allocates", "allocation"),
        ("expands", "allocation"),
    )
    for group_name, operation in operation_groups:
        group = payload[group_name]
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            raise ValueError(f"scheduler commit {group_name} must be a sequence")
        for detail in group:
            if not isinstance(detail, Mapping):
                raise ValueError(
                    f"scheduler commit {group_name} entries must be mappings"
                )
            cluster_id = detail.get("cluster_id")
            if not isinstance(cluster_id, str) or not cluster_id:
                raise ValueError(
                    "scheduler commit cluster_id must be a non-empty string"
                )
            if cluster_id not in tracked_clusters:
                continue
            if group_name == "shrinks":
                mapping = {detail.get("dp_rank"): detail.get("gpus_freed")}
            else:
                mapping = detail.get("dp_rank_to_gpus")
                if not isinstance(mapping, Mapping):
                    raise ValueError(
                        f"tracked {group_name} commit lacks dp_rank_to_gpus"
                    )
            add_mapping(operation=operation, cluster_id=cluster_id, mapping=mapping)

    identities = [
        (record.operation, record.cluster_id, record.dp_rank) for record in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("scheduler commit marker contains duplicate rank operations")
    return tuple(
        sorted(
            records,
            key=lambda record: (
                record.operation,
                record.pipeline_id,
                record.dp_rank,
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class GpuSample:
    """One direct physical GPU sample."""

    timestamp_ns: int
    gpu_id: int
    sm_utilization: float
    memory_utilization: float
    memory_used_bytes: int
    process_pids: tuple[int, ...] = ()
    error: str | None = None

    def validate(self) -> None:
        """Reject malformed or degraded gating samples."""
        if self.timestamp_ns < 0 or self.gpu_id < 0:
            raise ValueError("GPU sample timestamp and gpu_id must be non-negative")
        if not 0.0 <= self.sm_utilization <= 100.0:
            raise ValueError("SM utilization must be within [0, 100]")
        if not 0.0 <= self.memory_utilization <= 100.0:
            raise ValueError("memory utilization must be within [0, 100]")
        if self.memory_used_bytes < 0:
            raise ValueError("memory_used_bytes must be non-negative")
        if self.error:
            raise ValueError(f"GPU sampler reported an error: {self.error}")


@dataclass(frozen=True, slots=True)
class GpuUtilizationSummary:
    """Time-weighted utilization and idle duration for one GPU."""

    gpu_id: int
    duration_ns: int
    mean_sm_utilization: float
    idle_ns: int


def summarize_gpu_samples(
    samples: Iterable[GpuSample],
    *,
    max_gap_ns: int,
    idle_threshold: float,
) -> dict[int, GpuUtilizationSummary]:
    """Integrate irregular direct samples using left-continuous intervals."""
    if max_gap_ns <= 0:
        raise ValueError("max_gap_ns must be positive")
    by_gpu: dict[int, list[GpuSample]] = {}
    for sample in samples:
        sample.validate()
        by_gpu.setdefault(sample.gpu_id, []).append(sample)
    if not by_gpu:
        raise ValueError("at least one GPU sample is required")
    summaries: dict[int, GpuUtilizationSummary] = {}
    for gpu_id, gpu_samples in by_gpu.items():
        ordered = sorted(gpu_samples, key=lambda item: item.timestamp_ns)
        if len(ordered) < 2:
            raise ValueError(f"GPU {gpu_id} requires at least two samples")
        weighted_utilization = 0.0
        idle_ns = 0
        duration_ns = 0
        for left, right in zip(ordered, ordered[1:]):
            gap_ns = right.timestamp_ns - left.timestamp_ns
            if gap_ns <= 0:
                raise ValueError(f"GPU {gpu_id} sample timestamps must increase")
            if gap_ns > max_gap_ns:
                raise ValueError(
                    f"GPU {gpu_id} sample gap {gap_ns} exceeds {max_gap_ns}"
                )
            duration_ns += gap_ns
            weighted_utilization += left.sm_utilization * gap_ns
            if left.sm_utilization <= idle_threshold:
                idle_ns += gap_ns
        summaries[gpu_id] = GpuUtilizationSummary(
            gpu_id=gpu_id,
            duration_ns=duration_ns,
            mean_sm_utilization=weighted_utilization / duration_ns,
            idle_ns=idle_ns,
        )
    return summaries


@dataclass(frozen=True, slots=True)
class UtilizationTrial:
    """One matched static or dynamic acceptance repetition."""

    repetition: int
    mode: str
    completed_transitions: int
    physical_gpu_count: int
    wall_time_s: float
    idle_fraction: float | None = None
    complete_batch: bool = True
    correctness_passed: bool = True

    @property
    def useful_throughput_per_gpu(self) -> float:
        """Compute accepted transitions per physical GPU-second."""
        if (
            not isinstance(self.physical_gpu_count, int)
            or isinstance(self.physical_gpu_count, bool)
            or self.physical_gpu_count <= 0
            or not math.isfinite(self.wall_time_s)
            or self.wall_time_s <= 0
        ):
            raise ValueError("physical_gpu_count and wall_time_s must be positive")
        if (
            not isinstance(self.completed_transitions, int)
            or isinstance(self.completed_transitions, bool)
            or self.completed_transitions < 0
        ):
            raise ValueError("completed_transitions must be non-negative")
        if self.idle_fraction is not None and (
            isinstance(self.idle_fraction, bool)
            or not math.isfinite(self.idle_fraction)
            or not 0.0 <= self.idle_fraction <= 1.0
        ):
            raise ValueError("idle_fraction must be within [0, 1]")
        if not self.complete_batch or not self.correctness_passed:
            return 0.0
        return self.completed_transitions / (self.physical_gpu_count * self.wall_time_s)


@dataclass(frozen=True, slots=True)
class UtilizationAcceptance:
    """Paired throughput threshold result."""

    passed: bool
    paired_improvements: tuple[float, ...]
    median_improvement: float
    improved_repetitions: int
    paired_idle_reductions: tuple[float, ...] = ()
    median_idle_reduction: float | None = None


def evaluate_utilization_trials(
    trials: Iterable[UtilizationTrial],
    *,
    minimum_pairs: int = 5,
    minimum_median_improvement: float = 0.05,
    minimum_improved_pairs: int = 4,
    minimum_median_idle_reduction: float | None = None,
) -> UtilizationAcceptance:
    """Evaluate paired static/dynamic useful-throughput hard gates."""
    paired: dict[int, dict[str, UtilizationTrial]] = {}
    for trial in trials:
        if trial.mode not in {"static", "dynamic"}:
            raise ValueError(f"unknown utilization trial mode {trial.mode!r}")
        modes = paired.setdefault(trial.repetition, {})
        if trial.mode in modes:
            raise ValueError(
                f"duplicate {trial.mode} trial for repetition {trial.repetition}"
            )
        modes[trial.mode] = trial
    complete_pairs = [
        modes
        for _, modes in sorted(paired.items())
        if set(modes) == {"static", "dynamic"}
    ]
    if len(complete_pairs) != len(paired):
        raise ValueError(
            "every utilization repetition requires static and dynamic trials"
        )
    if len(complete_pairs) < minimum_pairs:
        raise ValueError(
            f"requires at least {minimum_pairs} paired repetitions, got {len(complete_pairs)}"
        )
    improvements: list[float] = []
    idle_reductions: list[float] = []
    all_dynamic_correct = True
    for modes in complete_pairs:
        static_throughput = modes["static"].useful_throughput_per_gpu
        dynamic_throughput = modes["dynamic"].useful_throughput_per_gpu
        if static_throughput <= 0:
            raise ValueError("static trial throughput must be positive")
        improvements.append(dynamic_throughput / static_throughput - 1.0)
        if minimum_median_idle_reduction is not None:
            static_idle = modes["static"].idle_fraction
            dynamic_idle = modes["dynamic"].idle_fraction
            if static_idle is None or dynamic_idle is None:
                raise ValueError(
                    "idle_fraction is required when enforcing idle reduction"
                )
            if static_idle <= 0.0:
                raise ValueError(
                    "static idle_fraction must be positive when enforcing idle reduction"
                )
            idle_reductions.append((static_idle - dynamic_idle) / static_idle)
        all_dynamic_correct &= (
            modes["dynamic"].complete_batch and modes["dynamic"].correctness_passed
        )
    median = float(np.median(np.asarray(improvements, dtype=np.float64)))
    median_idle_reduction = (
        float(np.median(np.asarray(idle_reductions, dtype=np.float64)))
        if idle_reductions
        else None
    )
    improved = sum(improvement > 0.0 for improvement in improvements)
    return UtilizationAcceptance(
        passed=(
            all_dynamic_correct
            and median >= minimum_median_improvement
            and improved >= minimum_improved_pairs
            and (
                minimum_median_idle_reduction is None
                or median_idle_reduction >= minimum_median_idle_reduction
            )
        ),
        paired_improvements=tuple(improvements),
        median_improvement=median,
        improved_repetitions=improved,
        paired_idle_reductions=tuple(idle_reductions),
        median_idle_reduction=median_idle_reduction,
    )
