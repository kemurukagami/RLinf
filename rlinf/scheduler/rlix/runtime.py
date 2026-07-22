"""Owner-scoped registration bootstrap for an inactive RLix pipeline."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

import ray
from rlix_core.control_plane import ControlPlane
from rlix_core.protocol.types import (
    ACTOR_TRAIN_CLUSTER_NAME,
    EVALUATION_CLUSTER_NAME,
    GENERATION_CLUSTER_NAME,
    INITIALIZATION_CLUSTER_NAME,
    POLICY_SYNC_CLUSTER_NAME,
    Priority,
    ProgressReport,
    get_pipeline_namespace,
)

from rlinf.workers.elastic_rollout_lifecycle import (
    ElasticRankProgress,
    ElasticRankState,
    ElasticRunOutcome,
)

from .controller import RLixStageController
from .placement import RLixPlacementPlan
from .progress import ElasticProgressTracker
from .protocol import (
    ElasticBatchReceipt,
    ElasticCollectionContext,
    FixedStageResidencyReceipt,
    FixedWorkerResidency,
    RunnerStageState,
)

_T = TypeVar("_T")


async def _await_value(awaitable: Awaitable[_T]) -> _T:
    return await awaitable


def _run_sync(value: _T | Awaitable[_T]) -> _T:
    """Resolve one async driver operation from the synchronous embodied runner."""
    if isinstance(value, ray.ObjectRef):
        return ray.get(value)
    if not inspect.isawaitable(value):
        return value
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_value(value))
    if inspect.iscoroutine(value):
        value.close()
    raise RuntimeError(
        "RLix synchronous runner runtime cannot run inside an active event loop"
    )


def _call_sync(target: Any, method_name: str, **kwargs: Any) -> Any:
    """Call either a direct test double method or a Ray actor method."""
    method = getattr(target, method_name)
    remote = getattr(method, "remote", None)
    if callable(remote):
        return _run_sync(remote(**kwargs))
    return _run_sync(method(**kwargs))


def _add_cleanup_note(primary: BaseException, operation: str, error: Exception) -> None:
    primary.add_note(
        f"RLix {operation} cleanup also failed: {type(error).__name__}: {error}"
    )


def close_worker_groups_after_bootstrap_failure(
    worker_groups: list[Any], primary_error: BaseException
) -> None:
    """Best-effort close partially launched groups in reverse construction order."""
    for group in reversed(worker_groups):
        try:
            group._close()
        except Exception as cleanup_error:
            _add_cleanup_note(primary_error, "worker-group", cleanup_error)


@dataclass(slots=True)
class ElasticCollectionSession:
    """Runtime-owned identity and mutable observations for one collection."""

    context: ElasticCollectionContext
    assignments: tuple[tuple[int, int], ...]
    tracker: ElasticProgressTracker
    cluster_id: str
    active_dp_ranks: set[int] = field(default_factory=set)
    released_dp_ranks: set[int] = field(default_factory=set)
    completed_dp_ranks: set[int] = field(default_factory=set)
    actor_receiver_handle: Any = None
    generation_owned: bool = False
    batch_receipt: ElasticBatchReceipt | None = None
    final_env_metrics: dict[int, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class RegisteredRLixPipeline:
    """Own registration and exclusive runner-stage allocation transactions."""

    control_plane: Any
    scheduler: Any
    controller: Any
    pipeline_id: str
    ray_namespace: str
    placement_plan: RLixPlacementPlan
    operation_timeout_s: float = 300.0
    _closed: bool = field(default=False, init=False, repr=False)
    _stage_state: RunnerStageState = field(
        default=RunnerStageState.INACTIVE, init=False, repr=False
    )
    _fixed_cluster_id: str | None = field(default=None, init=False, repr=False)
    _lifecycle_generation: int = field(default=0, init=False, repr=False)
    _collection: ElasticCollectionSession | None = field(
        default=None, init=False, repr=False
    )

    @property
    def stage_state(self) -> RunnerStageState:
        """Return the current runner-facing allocation state."""
        return self._stage_state

    def fixed_stage(
        self,
        *,
        cluster_name: str,
        priority: Priority,
        global_step: int,
    ) -> _FixedStage:
        """Create one synchronous acquire/verify/release fixed transaction."""
        return _FixedStage(
            runtime=self,
            cluster_name=cluster_name,
            priority=priority,
            global_step=global_step,
        )

    def policy_sync_stage(self, *, expected_policy_version: int) -> _PolicySyncStage:
        """Acquire fixed sync ownership and the coordinator mutation lease."""
        return _PolicySyncStage(
            runtime=self,
            expected_policy_version=expected_policy_version,
        )

    def begin_collection(
        self,
        *,
        policy_version: int,
        assigned_trajectories_by_rank: dict[int, int],
        env_input_channel: Any,
        rollout_request_channel: Any,
        actor_channel: Any,
        actor_receiver_start: Callable[[], Any],
        reward_channel: Any | None = None,
    ) -> ElasticCollectionSession:
        """Configure progress and receiver before requesting generation GPUs."""
        if self._closed or self._stage_state != RunnerStageState.INACTIVE:
            raise RuntimeError(
                f"cannot begin collection while runtime is {self._stage_state.value}"
            )
        canonical_ranks = tuple(
            rank for rank, _ in self.placement_plan.actor_infer_bundles
        )
        if tuple(sorted(assigned_trajectories_by_rank)) != canonical_ranks:
            raise ValueError("collection assignments must cover canonical DP ranks")
        self._lifecycle_generation += 1
        context = ElasticCollectionContext(
            lifecycle_generation=self._lifecycle_generation,
            policy_version=policy_version,
            dp_ranks=canonical_ranks,
        )
        tracker = ElasticProgressTracker(
            lifecycle_generation=context.lifecycle_generation,
            assigned_trajectories_by_rank=assigned_trajectories_by_rank,
        )
        cluster_id = f"{self.pipeline_id}_{GENERATION_CLUSTER_NAME}"
        session = ElasticCollectionSession(
            context=context,
            assignments=tuple(sorted(assigned_trajectories_by_rank.items())),
            tracker=tracker,
            cluster_id=cluster_id,
        )
        self._stage_state = RunnerStageState.ELASTIC_COLLECTION
        self._collection = session
        try:
            _run_sync(
                self.controller.configure_collection(
                    context,
                    env_input_channel=env_input_channel,
                    rollout_request_channel=rollout_request_channel,
                    reward_channel=reward_channel,
                    actor_channel=actor_channel,
                )
            )
            self._publish_collection_progress(session)
            session.actor_receiver_handle = actor_receiver_start()
            granted = _call_sync(
                self.scheduler,
                "request_gpus",
                cluster_id=cluster_id,
                priority=Priority.GENERATION,
                global_step=policy_version,
                step_target_estimate=tracker.step_target_trajectories,
            )
            session.generation_owned = True
            session.active_dp_ranks = self._project_generation_grant(granted)
            self._publish_collection_progress(session)
            return session
        except BaseException as primary_error:
            if not session.generation_owned:
                try:
                    _call_sync(
                        self.scheduler,
                        "clear_progress",
                        pipeline_id=self.pipeline_id,
                    )
                except Exception as cleanup_error:
                    _add_cleanup_note(
                        primary_error, "collection progress", cleanup_error
                    )
                self._collection = None
                self._stage_state = RunnerStageState.INACTIVE
            else:
                self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise

    def _publish_collection_progress(self, session: ElasticCollectionSession) -> None:
        snapshot = session.tracker.snapshot(
            active_dp_ranks=set(session.active_dp_ranks)
        )
        _call_sync(
            self.scheduler,
            "report_progress",
            report=ProgressReport(
                pipeline_id=self.pipeline_id,
                step_target_trajectories=snapshot.step_target_trajectories,
                metrics=snapshot.metrics,
            ),
        )

    def _project_generation_grant(self, granted: list[int]) -> set[int]:
        if not isinstance(granted, list):
            raise TypeError("generation grant must be a list")
        granted_set = set(granted)
        if len(granted_set) != len(granted):
            raise RuntimeError("generation grant contains duplicate GPU devices")
        unknown = granted_set - set(self.placement_plan.actor_infer_devices)
        if unknown:
            raise RuntimeError(
                f"generation grant contains unknown GPUs {sorted(unknown)}"
            )
        active: set[int] = set()
        for rank, bundle in self.placement_plan.actor_infer_bundles:
            overlap = granted_set & set(bundle)
            if overlap and overlap != set(bundle):
                raise RuntimeError(f"generation grant splits canonical bundle {rank}")
            if overlap:
                active.add(rank)
        if not active:
            raise RuntimeError("generation request returned no canonical DP rank")
        return active

    def monitor_collection_once(self, session: ElasticCollectionSession) -> bool:
        """Publish one monotonic observation pass and release completed ranks."""
        if session is not self._collection or (
            self._stage_state != RunnerStageState.ELASTIC_COLLECTION
        ):
            raise RuntimeError("collection session is not the active runtime session")
        try:
            return self._monitor_collection_once(session)
        except BaseException:
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise

    def wait_for_collection(
        self,
        session: ElasticCollectionSession,
        *,
        poll_interval_s: float = 0.01,
        timeout_s: float | None = None,
    ) -> None:
        """Poll without cancellation until every assigned rank completes."""
        if not isinstance(poll_interval_s, (int, float)) or isinstance(
            poll_interval_s, bool
        ):
            raise TypeError("poll_interval_s must be numeric")
        if poll_interval_s <= 0 or poll_interval_s > 1:
            raise ValueError("poll_interval_s must be in (0, 1]")
        timeout = self.operation_timeout_s if timeout_s is None else timeout_s
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
            raise TypeError("timeout_s must be numeric")
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + float(timeout)
        while True:
            if self.monitor_collection_once(session):
                return
            if time.monotonic() >= deadline:
                self._stage_state = RunnerStageState.FAILED_UNCERTAIN
                raise TimeoutError("timed out monitoring elastic collection")
            time.sleep(float(poll_interval_s))

    def _monitor_collection_once(self, session: ElasticCollectionSession) -> bool:
        """Implement one collection monitor pass after active-session validation."""
        observations = [
            _run_sync(self.controller.get_rank_observation(rank))
            for rank, _ in session.assignments
        ]
        callback_active: set[int] = set()
        completed: set[int] = set()
        for observation in observations:
            if observation.failure is not None:
                self._stage_state = RunnerStageState.FAILED_UNCERTAIN
                raise RuntimeError(
                    f"elastic rank {observation.dp_rank} failed: {observation.failure}"
                )
            if observation.callback_applied_active:
                callback_active.add(observation.dp_rank)
            progress = observation.progress
            if progress is not None:
                if not isinstance(progress, ElasticRankProgress):
                    raise TypeError("rank observation contains invalid progress")
                session.tracker.update_worker_progress(progress)
                if progress.state is ElasticRankState.COMPLETED:
                    completed.add(observation.dp_rank)
                    results = observation.paired_results
                    if results is None or any(
                        result.outcome is not ElasticRunOutcome.COMPLETED
                        for result in results
                    ):
                        raise RuntimeError(
                            f"elastic rank {observation.dp_rank} has incomplete paired results"
                        )
                    env_metrics = results[0].metrics
                    if env_metrics is not None:
                        session.final_env_metrics[observation.dp_rank] = env_metrics
        session.active_dp_ranks = callback_active
        self._publish_collection_progress(session)
        session.released_dp_ranks.update(completed - callback_active)
        to_release = tuple(
            sorted((completed & session.active_dp_ranks) - session.released_dp_ranks)
        )
        if to_release:
            _call_sync(
                self.scheduler,
                "await_release_dp_ranks",
                cluster_id=session.cluster_id,
                ranks=to_release,
                global_step=session.context.policy_version,
                timeout_s=self.operation_timeout_s,
            )
            session.active_dp_ranks.difference_update(to_release)
            session.released_dp_ranks.update(to_release)
            self._publish_collection_progress(session)
        session.completed_dp_ranks.update(completed)
        if session.released_dp_ranks == {rank for rank, _ in session.assignments}:
            session.generation_owned = False
        return completed == {rank for rank, _ in session.assignments}

    def seal_collection(
        self,
        session: ElasticCollectionSession,
        *,
        actor_seal_start: Callable[[int], Any],
    ) -> ElasticBatchReceipt:
        """Wait for actor receive, validate all actor shards, and clear progress."""
        expected_ranks = tuple(rank for rank, _ in session.assignments)
        if session is not self._collection or (
            self._stage_state != RunnerStageState.ELASTIC_COLLECTION
        ):
            raise RuntimeError("collection session is not active")
        if session.completed_dp_ranks != set(expected_ranks):
            raise RuntimeError("collection cannot seal before every rank completes")
        if session.active_dp_ranks or session.generation_owned:
            raise RuntimeError("collection cannot seal while generation is owned")
        if session.batch_receipt is not None:
            raise RuntimeError("collection is already sealed")
        try:
            receiver = session.actor_receiver_handle
            if receiver is None or not callable(getattr(receiver, "wait", None)):
                raise TypeError("actor receiver handle must expose wait()")
            receiver.wait()
            actor_count = len(self.placement_plan.actor_workers)
            target = session.tracker.step_target_trajectories
            if target % actor_count != 0:
                raise ValueError(
                    "trajectory target is not divisible by actor world size"
                )
            seal_handle = actor_seal_start(target // actor_count)
            receipts = (
                seal_handle.wait()
                if callable(getattr(seal_handle, "wait", None))
                else seal_handle
            )
            if not isinstance(receipts, list) or len(receipts) != actor_count:
                raise ValueError("actor seal must return one receipt per actor rank")
            for receipt in receipts:
                if not isinstance(receipt, ElasticBatchReceipt):
                    raise TypeError("actor returned an invalid batch receipt")
                if (
                    receipt.lifecycle_generation != session.context.lifecycle_generation
                    or receipt.policy_version != session.context.policy_version
                    or receipt.contributing_dp_ranks != expected_ranks
                    or receipt.expected_trajectories != target // actor_count
                ):
                    raise ValueError("actor batch receipt does not match collection")
            aggregate = ElasticBatchReceipt(
                lifecycle_generation=session.context.lifecycle_generation,
                policy_version=session.context.policy_version,
                contributing_dp_ranks=expected_ranks,
                expected_trajectories=target,
                received_trajectories=sum(
                    receipt.received_trajectories for receipt in receipts
                ),
                transition_count=sum(receipt.transition_count for receipt in receipts),
            )
            _call_sync(
                self.scheduler,
                "clear_progress",
                pipeline_id=self.pipeline_id,
            )
        except BaseException:
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise
        session.batch_receipt = aggregate
        self._collection = None
        self._stage_state = RunnerStageState.INACTIVE
        return aggregate

    def fixed_residency_receipt(
        self,
        *,
        cluster_name: str,
        worker_residencies: list[FixedWorkerResidency],
        policy_version: int | None = None,
    ) -> FixedStageResidencyReceipt:
        """Validate exact worker coverage and construct fixed release evidence."""
        if not isinstance(worker_residencies, list):
            raise TypeError("worker_residencies must be a list")
        expected_by_cluster = {
            INITIALIZATION_CLUSTER_NAME: {
                "actor": tuple(
                    worker.rank for worker in self.placement_plan.actor_workers
                ),
                "rollout": tuple(
                    worker.rank for worker in self.placement_plan.rollout_workers
                ),
                "environment": tuple(
                    worker.rank for worker in self.placement_plan.env_workers
                ),
            },
            POLICY_SYNC_CLUSTER_NAME: {
                "actor": tuple(
                    worker.rank for worker in self.placement_plan.actor_workers
                ),
                "rollout": tuple(
                    worker.rank for worker in self.placement_plan.rollout_workers
                ),
            },
            ACTOR_TRAIN_CLUSTER_NAME: {
                "actor": tuple(
                    worker.rank for worker in self.placement_plan.actor_workers
                ),
            },
            EVALUATION_CLUSTER_NAME: {
                "rollout": tuple(
                    worker.rank for worker in self.placement_plan.rollout_workers
                ),
                "environment": tuple(
                    worker.rank for worker in self.placement_plan.env_workers
                ),
            },
        }
        try:
            expected = expected_by_cluster[cluster_name]
        except KeyError as exc:
            raise ValueError(f"unsupported fixed cluster {cluster_name!r}") from exc
        observed: dict[str, list[int]] = {}
        for status in worker_residencies:
            if not isinstance(status, FixedWorkerResidency):
                raise TypeError("worker residency entries have an invalid type")
            if status.component not in expected:
                raise ValueError(
                    f"unexpected {status.component} residency for {cluster_name!r}"
                )
            observed.setdefault(status.component, []).append(status.rank)
            if not status.safe_to_release:
                raise RuntimeError(
                    f"{status.component} rank {status.rank} remains GPU-resident"
                )
            if (
                cluster_name == POLICY_SYNC_CLUSTER_NAME
                and status.policy_version != policy_version
            ):
                raise ValueError(
                    f"{status.component} rank {status.rank} applied policy version "
                    f"{status.policy_version}, expected {policy_version}"
                )
        normalized = {
            component: tuple(sorted(ranks)) for component, ranks in observed.items()
        }
        if normalized != expected:
            raise ValueError(
                f"fixed cluster {cluster_name!r} residency coverage {normalized} "
                f"does not match expected {expected}"
            )
        _, devices, _ = self._fixed_stage_spec(cluster_name)
        return FixedStageResidencyReceipt(
            cluster_name=cluster_name,
            verified_devices=devices,
            all_workers_offloaded=True,
            policy_version=policy_version,
        )

    def _fixed_stage_spec(
        self, cluster_name: str
    ) -> tuple[RunnerStageState, tuple[int, ...], Priority]:
        specs = {
            INITIALIZATION_CLUSTER_NAME: (
                RunnerStageState.FIXED_INITIALIZATION,
                self.placement_plan.initialization_devices,
                Priority.INITIALIZATION,
            ),
            POLICY_SYNC_CLUSTER_NAME: (
                RunnerStageState.FIXED_POLICY_SYNC,
                self.placement_plan.policy_sync_devices,
                Priority.GENERATION,
            ),
            ACTOR_TRAIN_CLUSTER_NAME: (
                RunnerStageState.FIXED_ACTOR_TRAIN,
                self.placement_plan.actor_train_devices,
                Priority.ACTOR_TRAINING,
            ),
            EVALUATION_CLUSTER_NAME: (
                RunnerStageState.FIXED_EVALUATION,
                self.placement_plan.evaluation_devices,
                Priority.INITIALIZATION,
            ),
        }
        try:
            return specs[cluster_name]
        except KeyError as exc:
            raise ValueError(f"unsupported fixed cluster {cluster_name!r}") from exc

    def _begin_fixed_stage(
        self, *, cluster_name: str, priority: Priority, global_step: int
    ) -> tuple[int, ...]:
        if self._closed or self._stage_state == RunnerStageState.CLOSED:
            raise RuntimeError("RLix runtime is closed")
        if self._stage_state != RunnerStageState.INACTIVE:
            raise RuntimeError(
                f"cannot begin fixed stage while runtime is {self._stage_state.value}"
            )
        if not isinstance(global_step, int) or isinstance(global_step, bool):
            raise TypeError("global_step must be an integer")
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        state, expected_devices, expected_priority = self._fixed_stage_spec(
            cluster_name
        )
        if priority != expected_priority:
            raise ValueError(
                f"fixed cluster {cluster_name!r} requires priority "
                f"{expected_priority.name}, got {priority!r}"
            )
        cluster_id = f"{self.pipeline_id}_{cluster_name}"
        self._stage_state = state
        try:
            granted = _call_sync(
                self.scheduler,
                "request_gpus",
                cluster_id=cluster_id,
                priority=priority,
                global_step=global_step,
            )
        except BaseException:
            self._stage_state = RunnerStageState.INACTIVE
            raise
        granted_tuple = tuple(sorted(granted))
        if len(granted_tuple) != len(set(granted_tuple)):
            self._fixed_cluster_id = cluster_id
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise RuntimeError(
                f"fixed cluster {cluster_name!r} returned duplicate GPU devices"
            )
        if granted_tuple != expected_devices:
            self._fixed_cluster_id = cluster_id
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise RuntimeError(
                f"fixed cluster {cluster_name!r} granted devices {granted_tuple}, "
                f"expected {expected_devices}"
            )
        self._fixed_cluster_id = cluster_id
        return expected_devices

    def _finish_fixed_stage(
        self,
        *,
        cluster_name: str,
        global_step: int,
        receipt: FixedStageResidencyReceipt | None,
        body_error: BaseException | None,
    ) -> None:
        if receipt is None:
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            if body_error is None:
                raise RuntimeError(
                    f"fixed cluster {cluster_name!r} exited without a residency receipt"
                )
            return
        _, expected_devices, _ = self._fixed_stage_spec(cluster_name)
        if receipt.cluster_name != cluster_name:
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise ValueError("fixed-stage receipt cluster does not match active stage")
        if receipt.verified_devices != expected_devices:
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise ValueError(
                "fixed-stage receipt devices do not match registered union"
            )
        if not receipt.all_workers_offloaded:
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise RuntimeError("fixed-stage workers are not verified offloaded")
        cluster_id = self._fixed_cluster_id
        if cluster_id is None:
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise RuntimeError("fixed-stage allocation identity is missing")
        try:
            _call_sync(
                self.scheduler,
                "notify_release_gpus",
                cluster_id=cluster_id,
                global_step=global_step,
            )
        except BaseException:
            self._stage_state = RunnerStageState.FAILED_UNCERTAIN
            raise
        self._fixed_cluster_id = None
        self._stage_state = RunnerStageState.INACTIVE

    async def close(self) -> None:
        """Unregister before closing the coordinator; repeated calls are harmless."""
        if self._closed:
            return
        if self._stage_state != RunnerStageState.INACTIVE:
            raise RuntimeError(
                f"cannot close RLix runtime while it is {self._stage_state.value}"
            )
        primary_error: Exception | None = None
        try:
            self.control_plane.unregister_pipeline(pipeline_id=self.pipeline_id)
        except Exception as exc:
            primary_error = exc
        try:
            await self.controller.close()
        except Exception as exc:
            if primary_error is None:
                primary_error = exc
            else:
                _add_cleanup_note(primary_error, "coordinator", exc)
        if primary_error is not None:
            raise primary_error
        self._closed = True
        self._stage_state = RunnerStageState.CLOSED

    def close_sync(self) -> None:
        """Close an inactive registered pipeline from the synchronous runner."""
        _run_sync(self.close())


@dataclass(slots=True)
class _FixedStage:
    """One-use synchronous fixed-stage context with explicit completion proof."""

    runtime: RegisteredRLixPipeline
    cluster_name: str
    priority: Priority
    global_step: int
    _expected_devices: tuple[int, ...] | None = field(
        default=None, init=False, repr=False
    )
    _receipt: FixedStageResidencyReceipt | None = field(
        default=None, init=False, repr=False
    )

    def __enter__(self) -> _FixedStage:
        if self._expected_devices is not None:
            raise RuntimeError("fixed-stage context cannot be entered twice")
        self._expected_devices = self.runtime._begin_fixed_stage(
            cluster_name=self.cluster_name,
            priority=self.priority,
            global_step=self.global_step,
        )
        return self

    def complete(self, receipt: FixedStageResidencyReceipt) -> None:
        """Record positive offload evidence required for logical release."""
        if self._expected_devices is None:
            raise RuntimeError("fixed-stage context has not been entered")
        if self._receipt is not None:
            raise RuntimeError("fixed-stage context is already complete")
        if not isinstance(receipt, FixedStageResidencyReceipt):
            raise TypeError("receipt must be a FixedStageResidencyReceipt")
        self._receipt = receipt

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, traceback
        try:
            self.runtime._finish_fixed_stage(
                cluster_name=self.cluster_name,
                global_step=self.global_step,
                receipt=self._receipt,
                body_error=exc,
            )
        except Exception as cleanup_error:
            if exc is None:
                raise
            _add_cleanup_note(exc, f"{self.cluster_name} stage", cleanup_error)
        return False


@dataclass(slots=True)
class _PolicySyncStage:
    """Combined fixed allocation and T5 policy-sync lease transaction."""

    runtime: RegisteredRLixPipeline
    expected_policy_version: int
    _fixed_stage: _FixedStage = field(init=False, repr=False)
    _lease: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._fixed_stage = self.runtime.fixed_stage(
            cluster_name=POLICY_SYNC_CLUSTER_NAME,
            priority=Priority.GENERATION,
            global_step=self.expected_policy_version,
        )

    def __enter__(self) -> _PolicySyncStage:
        self._fixed_stage.__enter__()
        try:
            self._lease = _run_sync(
                self.runtime.controller.begin_policy_sync(
                    expected_policy_version=self.expected_policy_version
                )
            )
        except BaseException as lease_error:
            self._fixed_stage.__exit__(
                type(lease_error), lease_error, lease_error.__traceback__
            )
            raise
        return self

    def complete(self, receipt: FixedStageResidencyReceipt) -> None:
        """Record all-rank version and residency evidence."""
        if receipt.policy_version != self.expected_policy_version:
            raise ValueError("policy-sync receipt has the wrong policy version")
        self._fixed_stage.complete(receipt)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        primary_error = exc
        try:
            _run_sync(self.runtime.controller.end_policy_sync(self._lease))
        except Exception as lease_error:
            if primary_error is None:
                primary_error = lease_error
            else:
                _add_cleanup_note(primary_error, "policy-sync lease", lease_error)
            # The lease must end before scheduler ownership can be returned.
            self._fixed_stage._receipt = None
        self._fixed_stage.__exit__(
            type(primary_error) if primary_error is not None else None,
            primary_error,
            primary_error.__traceback__ if primary_error is not None else None,
        )
        if exc is None and primary_error is not None:
            raise primary_error
        return False


async def bootstrap_registered_rlix_pipeline(
    *,
    env_worker_group: Any,
    rollout_worker_group: Any,
    placement_plan: RLixPlacementPlan,
    worker_max_concurrency: int,
    operation_timeout_s: float,
    enable_gpu_tracing: bool = False,
    control_plane_factory: Callable[..., Any] = ControlPlane,
    controller_factory: Callable[..., Any] = RLixStageController,
) -> RegisteredRLixPipeline:
    """Create, register, and admit one pipeline without requesting allocation."""
    if not isinstance(enable_gpu_tracing, bool):
        raise TypeError("enable_gpu_tracing must be a boolean")
    control_plane = control_plane_factory(
        env_vars={"RLIX_ENABLE_GPU_TRACING": "1"} if enable_gpu_tracing else {}
    )
    pipeline_id = control_plane.allocate_pipeline_id(pipeline_type="rlinf")
    ray_namespace = get_pipeline_namespace(pipeline_id)
    controller = controller_factory(
        pipeline_id=pipeline_id,
        ray_namespace=ray_namespace,
        env_worker_group=env_worker_group,
        rollout_worker_group=rollout_worker_group,
        operation_timeout_s=operation_timeout_s,
        worker_max_concurrency=worker_max_concurrency,
    )
    registration_attempted = False
    try:
        registration_attempted = True
        control_plane.register_pipeline(
            pipeline_id=pipeline_id,
            ray_namespace=ray_namespace,
            **placement_plan.registration_payload(),
        )
        admission = control_plane.admit_pipeline(pipeline_id=pipeline_id)
    except BaseException as primary_error:
        if registration_attempted:
            try:
                control_plane.unregister_pipeline(pipeline_id=pipeline_id)
            except Exception as cleanup_error:
                _add_cleanup_note(primary_error, "registration", cleanup_error)
        try:
            await controller.close()
        except Exception as cleanup_error:
            _add_cleanup_note(primary_error, "coordinator", cleanup_error)
        raise

    return RegisteredRLixPipeline(
        control_plane=control_plane,
        scheduler=admission.scheduler,
        controller=controller,
        pipeline_id=pipeline_id,
        ray_namespace=ray_namespace,
        placement_plan=placement_plan,
        operation_timeout_s=operation_timeout_s,
    )


__all__ = [
    "RegisteredRLixPipeline",
    "ElasticCollectionSession",
    "bootstrap_registered_rlix_pipeline",
    "close_worker_groups_after_bootstrap_failure",
]
