"""Fail-closed resize transactions for paired RLinf elastic workers."""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass
from typing import Any

from rlix_core.protocol.types import ActionResponse
from rlix_core.protocol.validation import validate_pipeline_id

from rlinf.workers.elastic_rollout_lifecycle import (
    CompletedResidencyReceipt,
    DrainRequest,
    ElasticRankProgress,
    ElasticRankState,
    ElasticRankStatus,
    ElasticRunOutcome,
    ElasticRunResult,
    ResidencyReceipt,
    SafePointToken,
)

from .protocol import (
    CoordinatorStatus,
    ElasticCollectionContext,
    ElasticRankObservation,
    PolicySyncLease,
)


class ResizeCoordinatorError(RuntimeError):
    """A paired worker transition could not be verified safely."""


@dataclass(frozen=True, slots=True)
class _ChannelBinding:
    env_input_channel: Any
    rollout_request_channel: Any
    reward_channel: Any | None
    actor_channel: Any | None


@dataclass(slots=True)
class _RankRecord:
    token: SafePointToken | None = None
    env_run_task: asyncio.Future[Any] | None = None
    rollout_run_task: asyncio.Future[Any] | None = None
    last_env_result: ElasticRunResult | None = None
    last_rollout_result: ElasticRunResult | None = None
    callback_applied_active: bool = False
    failure: str | None = None


class RLixResizeCoordinator:
    """Coordinate exact same-ranked environment and rollout worker pairs."""

    def __init__(
        self,
        *,
        pipeline_id: str,
        env_workers: dict[int, Any],
        rollout_workers: dict[int, Any],
        operation_timeout_s: float = 300.0,
        activation_poll_interval_s: float = 0.01,
    ) -> None:
        """Validate and retain exact ranked worker handles."""
        validate_pipeline_id(pipeline_id)
        if not isinstance(env_workers, dict) or not isinstance(rollout_workers, dict):
            raise TypeError("worker mappings must be dictionaries")
        if set(env_workers) != set(rollout_workers):
            raise ValueError("environment and rollout worker rank mappings must match")
        ranks = tuple(sorted(env_workers))
        if not ranks or ranks != tuple(range(len(ranks))):
            raise ValueError("worker mappings must use a non-empty contiguous rank set")
        for rank in ranks:
            if not isinstance(rank, int) or isinstance(rank, bool):
                raise TypeError("worker ranks must be integers")
        if not isinstance(operation_timeout_s, (int, float)) or isinstance(
            operation_timeout_s, bool
        ):
            raise TypeError("operation_timeout_s must be numeric")
        if operation_timeout_s <= 0:
            raise ValueError("operation_timeout_s must be positive")
        if activation_poll_interval_s <= 0:
            raise ValueError("activation_poll_interval_s must be positive")

        self._pipeline_id = pipeline_id
        self._env_workers = dict(env_workers)
        self._rollout_workers = dict(rollout_workers)
        self._ranks = ranks
        self._operation_timeout_s = float(operation_timeout_s)
        self._activation_poll_interval_s = float(activation_poll_interval_s)
        self._collection: ElasticCollectionContext | None = None
        self._channels: _ChannelBinding | None = None
        self._records = {rank: _RankRecord() for rank in ranks}
        self._gate = asyncio.Condition()
        self._configuration_in_progress = False
        self._resize_in_progress = False
        self._policy_sync_lease: PolicySyncLease | None = None
        self._failure: str | None = None
        self._closed = False

    @staticmethod
    def _launch(
        worker: Any, method_name: str, *args: Any, **kwargs: Any
    ) -> asyncio.Future[Any]:
        method = getattr(worker, method_name)
        remote = getattr(method, "remote", None)
        result = (
            remote(*args, **kwargs) if callable(remote) else method(*args, **kwargs)
        )
        if inspect.isawaitable(result):
            return asyncio.ensure_future(result)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.set_result(result)
        return future

    async def _wait_tasks(
        self,
        tasks: tuple[asyncio.Future[Any], ...],
        *,
        operation: str,
        timeout_s: float | None = None,
    ) -> tuple[Any, ...]:
        timeout = self._operation_timeout_s if timeout_s is None else timeout_s
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout,
            return_when=asyncio.FIRST_EXCEPTION,
        )
        # A paired lifecycle may legitimately finish one peer before the other,
        # so normal completion still waits for both. A real exception cannot be
        # repaired by waiting for its peer, which may be blocked on a message the
        # failed task was responsible for sending.
        for task in tasks:
            if not task.done():
                continue
            if task.cancelled():
                raise asyncio.CancelledError(
                    f"Task was cancelled while waiting for {operation}"
                )
            exception = task.exception()
            if exception is not None:
                raise exception
        if pending:
            raise TimeoutError(f"Timed out waiting for {operation}")
        return tuple(task.result() for task in tasks)

    async def _wait_for_gate(self, predicate: Any, *, operation: str) -> None:
        try:
            await asyncio.wait_for(
                self._gate.wait_for(predicate), timeout=self._operation_timeout_s
            )
        except TimeoutError as exc:
            raise TimeoutError(f"Timed out waiting to begin {operation}") from exc

    async def _call_pair(
        self,
        rank: int,
        env_method: str,
        rollout_method: str,
        *,
        operation: str,
        env_args: tuple[Any, ...] = (),
        rollout_args: tuple[Any, ...] = (),
        env_kwargs: dict[str, Any] | None = None,
        rollout_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        env_task = self._launch(
            self._env_workers[rank], env_method, *env_args, **(env_kwargs or {})
        )
        rollout_task = self._launch(
            self._rollout_workers[rank],
            rollout_method,
            *rollout_args,
            **(rollout_kwargs or {}),
        )
        env_result, rollout_result = await self._wait_tasks(
            (env_task, rollout_task), operation=f"rank {rank} {operation}"
        )
        return env_result, rollout_result

    async def _get_pair_status(
        self, rank: int
    ) -> tuple[ElasticRankStatus, ElasticRankStatus]:
        env_status, rollout_status = await self._call_pair(
            rank,
            "get_elastic_status",
            "get_elastic_status",
            operation="status query",
        )
        if not isinstance(env_status, ElasticRankStatus) or not isinstance(
            rollout_status, ElasticRankStatus
        ):
            raise ResizeCoordinatorError(f"rank {rank} returned an invalid status")
        return env_status, rollout_status

    @staticmethod
    def _format_peer_state_mismatch(
        rank: int,
        *,
        env_status: ElasticRankStatus,
        rollout_status: ElasticRankStatus,
        env_run_task: asyncio.Future[Any] | None = None,
        rollout_run_task: asyncio.Future[Any] | None = None,
        final: bool = False,
    ) -> str:
        """Describe a paired-state mismatch without dumping full worker payloads."""

        def summarize(value: str | None) -> str:
            if value is None:
                return "None"
            return repr(value[:256])

        def task_state(task: asyncio.Future[Any] | None) -> str:
            if task is None:
                return "missing"
            if task.cancelled():
                return "cancelled"
            if not task.done():
                return "pending"
            try:
                exception = task.exception()
            except Exception as exc:
                return (
                    f"done_exception_unavailable:{type(exc).__name__}:{str(exc)[:128]}"
                )
            if exception is None:
                return "done"
            return (
                f"done_with_exception:{type(exception).__name__}:{str(exception)[:256]}"
            )

        label = "final peer states" if final else "peer states"
        return (
            f"rank {rank} {label} do not match: "
            f"env_state={env_status.state.value} "
            f"rollout_state={rollout_status.state.value} "
            f"env_lifecycle={env_status.lifecycle_generation} "
            f"rollout_lifecycle={rollout_status.lifecycle_generation} "
            f"env_policy={env_status.policy_version} "
            f"rollout_policy={rollout_status.policy_version} "
            f"env_resident={env_status.model_resident} "
            f"rollout_resident={rollout_status.model_resident} "
            f"env_snapshot_ready={env_status.snapshot_ready} "
            f"rollout_snapshot_ready={rollout_status.snapshot_ready} "
            f"env_drain_request_id={env_status.drain_request_id} "
            f"rollout_drain_request_id={rollout_status.drain_request_id} "
            f"env_failure={summarize(env_status.failure)} "
            f"rollout_failure={summarize(rollout_status.failure)} "
            f"env_run_task={task_state(env_run_task)} "
            f"rollout_run_task={task_state(rollout_run_task)}"
        )

    @staticmethod
    def _validate_status_identity(
        rank: int,
        status: ElasticRankStatus,
        context: ElasticCollectionContext,
        *,
        allow_cold: bool = False,
    ) -> None:
        if status.worker_rank != rank:
            raise ResizeCoordinatorError(
                f"rank {rank} worker reported rank {status.worker_rank}"
            )
        if allow_cold and status.state is ElasticRankState.INACTIVE_COLD:
            if (
                status.lifecycle_generation is not None
                or status.policy_version is not None
            ):
                raise ResizeCoordinatorError(
                    f"cold rank {rank} retained lifecycle identity"
                )
            return
        if status.lifecycle_generation != context.lifecycle_generation:
            raise ResizeCoordinatorError(
                f"rank {rank} lifecycle does not match collection: "
                f"component_state={status.state.value} "
                f"reported_lifecycle={status.lifecycle_generation} "
                f"expected_lifecycle={context.lifecycle_generation} "
                f"reported_policy={status.policy_version} "
                f"expected_policy={context.policy_version}"
            )
        if status.policy_version != context.policy_version:
            raise ResizeCoordinatorError(
                f"rank {rank} policy version does not match collection"
            )

    @staticmethod
    def _is_dormant_completed_pair(
        rank: int,
        env_status: ElasticRankStatus,
        rollout_status: ElasticRankStatus,
        context: ElasticCollectionContext,
        record: _RankRecord,
    ) -> bool:
        """Recognize an offloaded rank not yet admitted to this collection."""
        tasks_resolved = all(
            task is None or task.done()
            for task in (record.env_run_task, record.rollout_run_task)
        )
        return (
            not record.callback_applied_active
            and tasks_resolved
            and record.token is None
            and record.last_env_result is None
            and record.last_rollout_result is None
            and env_status.worker_rank == rank
            and rollout_status.worker_rank == rank
            and env_status.state is ElasticRankState.COMPLETED
            and rollout_status.state is ElasticRankState.COMPLETED
            and env_status.lifecycle_generation is not None
            and env_status.lifecycle_generation == rollout_status.lifecycle_generation
            and env_status.lifecycle_generation < context.lifecycle_generation
            and env_status.policy_version == rollout_status.policy_version
            and env_status.model_resident is False
            and rollout_status.model_resident is False
            and env_status.cuda_graph_captured is False
            and rollout_status.cuda_graph_captured is False
            and env_status.failure is None
            and rollout_status.failure is None
        )

    async def configure_collection(
        self,
        context: ElasticCollectionContext,
        *,
        env_input_channel: Any,
        rollout_request_channel: Any,
        reward_channel: Any | None = None,
        actor_channel: Any | None = None,
    ) -> None:
        """Bind one immutable collection identity and its runtime channels."""
        if not isinstance(context, ElasticCollectionContext):
            raise TypeError("context must be an ElasticCollectionContext")
        if context.dp_ranks != self._ranks:
            raise ValueError("collection ranks must match the registered worker ranks")
        if env_input_channel is None or rollout_request_channel is None:
            raise ValueError(
                "environment input and rollout request channels are required"
            )
        async with self._gate:
            if self._closed:
                raise RuntimeError("coordinator is closed")
            if self._failure is not None:
                raise ResizeCoordinatorError("coordinator has failed closed")
            if (
                self._configuration_in_progress
                or self._resize_in_progress
                or self._policy_sync_lease is not None
            ):
                raise RuntimeError(
                    "cannot configure collection during resize or policy sync"
                )
            self._configuration_in_progress = True

        try:
            await self._apply_collection_configuration(
                context,
                channels=_ChannelBinding(
                    env_input_channel=env_input_channel,
                    rollout_request_channel=rollout_request_channel,
                    reward_channel=reward_channel,
                    actor_channel=actor_channel,
                ),
            )
        finally:
            async with self._gate:
                self._configuration_in_progress = False
                self._gate.notify_all()

    async def _apply_collection_configuration(
        self, context: ElasticCollectionContext, *, channels: _ChannelBinding
    ) -> None:
        statuses = await asyncio.gather(
            *(self._get_pair_status(rank) for rank in self._ranks)
        )
        for rank, (env_status, rollout_status) in zip(
            self._ranks, statuses, strict=True
        ):
            record = self._records[rank]
            if record.env_run_task is not None and not record.env_run_task.done():
                raise RuntimeError(f"rank {rank} environment call is unresolved")
            if (
                record.rollout_run_task is not None
                and not record.rollout_run_task.done()
            ):
                raise RuntimeError(f"rank {rank} rollout call is unresolved")
            allowed = {ElasticRankState.INACTIVE_COLD, ElasticRankState.COMPLETED}
            if env_status.state not in allowed or rollout_status.state not in allowed:
                raise RuntimeError(f"rank {rank} is not between collection lifecycles")
            if env_status.state is not rollout_status.state:
                raise ResizeCoordinatorError(f"rank {rank} peer states do not match")
            if env_status.state is ElasticRankState.COMPLETED and (
                env_status.model_resident is not False
                or rollout_status.model_resident is not False
                or env_status.cuda_graph_captured
                or rollout_status.cuda_graph_captured
            ):
                raise RuntimeError(
                    f"completed rank {rank} must be offloaded before reconfiguration"
                )

        if self._collection is not None and (
            context.lifecycle_generation <= self._collection.lifecycle_generation
        ):
            raise ValueError("new collection lifecycle_generation must increase")
        self._collection = context
        self._channels = channels
        for record in self._records.values():
            record.token = None
            record.last_env_result = None
            record.last_rollout_result = None
            record.callback_applied_active = False
            record.failure = None

    async def _wait_for_active(self, rank: int) -> None:
        context = self._require_collection()
        record = self._records[rank]
        deadline = time.monotonic() + self._operation_timeout_s
        while True:
            env_status, rollout_status = await self._get_pair_status(rank)
            self._validate_status_identity(rank, env_status, context)
            self._validate_status_identity(rank, rollout_status, context)
            if (
                env_status.state is ElasticRankState.ACTIVE
                and rollout_status.state is ElasticRankState.ACTIVE
            ):
                return
            if env_status.state is ElasticRankState.FAILED_RESIDENT or (
                rollout_status.state is ElasticRankState.FAILED_RESIDENT
            ):
                raise ResizeCoordinatorError(f"rank {rank} failed during activation")
            if record.env_run_task is not None and record.env_run_task.done():
                raise ResizeCoordinatorError(
                    f"rank {rank} environment call ended before activation"
                )
            if record.rollout_run_task is not None and record.rollout_run_task.done():
                raise ResizeCoordinatorError(
                    f"rank {rank} rollout call ended before activation"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for rank {rank} activation")
            await asyncio.sleep(self._activation_poll_interval_s)

    def _require_collection(self) -> ElasticCollectionContext:
        if self._collection is None or self._channels is None:
            raise RuntimeError("elastic collection is not configured")
        return self._collection

    async def _expand_rank(self, rank: int) -> None:
        context = self._require_collection()
        channels = self._channels
        assert channels is not None
        record = self._records[rank]
        if record.callback_applied_active:
            raise RuntimeError(f"rank {rank} is already locally active")
        if (record.env_run_task is not None and not record.env_run_task.done()) or (
            record.rollout_run_task is not None and not record.rollout_run_task.done()
        ):
            raise RuntimeError(f"rank {rank} has unresolved run calls")

        env_status, rollout_status = await self._get_pair_status(rank)
        if env_status.state is not rollout_status.state:
            raise ResizeCoordinatorError(f"rank {rank} peer states do not match")
        state = env_status.state
        if state in {ElasticRankState.INACTIVE_COLD, ElasticRankState.COMPLETED}:
            if env_status.worker_rank != rank or rollout_status.worker_rank != rank:
                raise ResizeCoordinatorError(f"rank {rank} worker identity mismatch")
            if state is ElasticRankState.COMPLETED:
                if (
                    env_status.lifecycle_generation is None
                    or rollout_status.lifecycle_generation
                    != env_status.lifecycle_generation
                    or env_status.lifecycle_generation >= context.lifecycle_generation
                    or rollout_status.policy_version != env_status.policy_version
                ):
                    raise RuntimeError(
                        f"rank {rank} is not completed in an older matching collection"
                    )
            else:
                self._validate_status_identity(
                    rank, env_status, context, allow_cold=True
                )
                self._validate_status_identity(
                    rank, rollout_status, context, allow_cold=True
                )
            env_receipt, rollout_receipt = await self._call_pair(
                rank,
                "prepare_elastic_collection",
                "prepare_elastic_collection",
                operation="cold preparation",
                env_kwargs={
                    "lifecycle_generation": context.lifecycle_generation,
                    "expected_policy_version": context.policy_version,
                },
                rollout_kwargs={
                    "lifecycle_generation": context.lifecycle_generation,
                    "expected_policy_version": context.policy_version,
                },
            )
            for receipt in (env_receipt, rollout_receipt):
                if (
                    not isinstance(receipt, ElasticRankStatus)
                    or receipt.state is not ElasticRankState.EXPANDING
                ):
                    raise ResizeCoordinatorError(
                        f"rank {rank} returned an invalid preparation receipt"
                    )
        elif state is ElasticRankState.PAUSED:
            self._validate_status_identity(rank, env_status, context)
            self._validate_status_identity(rank, rollout_status, context)
            token = record.token
            if token is None:
                raise ResizeCoordinatorError(f"rank {rank} has no stored pause token")
            if env_status.expected_transition_id != token.next_transition_id or (
                rollout_status.expected_transition_id != token.next_transition_id
            ):
                raise ResizeCoordinatorError(
                    f"rank {rank} pause transition does not match token"
                )
            env_receipt, rollout_receipt = await self._call_pair(
                rank,
                "prepare_elastic_resume",
                "prepare_elastic_resume",
                operation="resume preparation",
                env_args=(token,),
                rollout_args=(token,),
            )
            for receipt in (env_receipt, rollout_receipt):
                if (
                    not isinstance(receipt, ResidencyReceipt)
                    or receipt.token != token
                    or (receipt.state is not ElasticRankState.EXPANDING)
                ):
                    raise ResizeCoordinatorError(
                        f"rank {rank} returned an invalid resume receipt"
                    )
        else:
            raise RuntimeError(f"cannot expand rank {rank} from {state.value}")

        record.env_run_task = self._launch(
            self._env_workers[rank],
            "interact_until_pause_or_complete",
            channels.env_input_channel,
            channels.rollout_request_channel,
            channels.reward_channel,
            channels.actor_channel,
        )
        record.rollout_run_task = self._launch(
            self._rollout_workers[rank],
            "generate_until_pause_or_complete",
            channels.rollout_request_channel,
            channels.env_input_channel,
        )
        await self._wait_for_active(rank)
        record.token = None
        record.callback_applied_active = True

    @staticmethod
    def _validate_pause_results(
        rank: int, env_result: ElasticRunResult, rollout_result: ElasticRunResult
    ) -> SafePointToken:
        if env_result.outcome is not ElasticRunOutcome.PAUSE_READY or (
            rollout_result.outcome is not ElasticRunOutcome.PAUSE_READY
        ):
            raise ResizeCoordinatorError(
                f"rank {rank} peer outcomes do not match a pause"
            )
        if env_result.token is None or env_result.token != rollout_result.token:
            raise ResizeCoordinatorError(f"rank {rank} peer pause tokens do not match")
        return env_result.token

    async def _offload_completed_rank(
        self, rank: int, context: ElasticCollectionContext
    ) -> None:
        env_receipt, rollout_receipt = await self._call_pair(
            rank,
            "offload_completed_elastic_environment",
            "offload_completed_elastic_rollout",
            operation="completed offload",
        )
        for receipt in (env_receipt, rollout_receipt):
            if not isinstance(receipt, CompletedResidencyReceipt):
                raise ResizeCoordinatorError(
                    f"rank {rank} returned an invalid completed receipt"
                )
            if (
                receipt.worker_rank != rank
                or receipt.lifecycle_generation != context.lifecycle_generation
                or receipt.policy_version != context.policy_version
            ):
                raise ResizeCoordinatorError(
                    f"rank {rank} completed receipt identity mismatch"
                )

    async def _shrink_rank(self, rank: int) -> None:
        context = self._require_collection()
        record = self._records[rank]
        if not record.callback_applied_active:
            raise RuntimeError(f"rank {rank} is not locally active")
        if record.env_run_task is None or record.rollout_run_task is None:
            raise ResizeCoordinatorError(f"rank {rank} has no stored run calls")
        env_status, rollout_status = await self._get_pair_status(rank)
        self._validate_status_identity(rank, env_status, context)
        self._validate_status_identity(rank, rollout_status, context)
        states = {env_status.state, rollout_status.state}
        if env_status.state is not rollout_status.state:
            if not (
                ElasticRankState.COMPLETED in states
                and states
                <= {
                    ElasticRankState.ACTIVE,
                    ElasticRankState.DRAIN_REQUESTED,
                    ElasticRankState.COMPLETED,
                }
            ):
                raise ResizeCoordinatorError(
                    self._format_peer_state_mismatch(
                        rank,
                        env_status=env_status,
                        rollout_status=rollout_status,
                        env_run_task=record.env_run_task,
                        rollout_run_task=record.rollout_run_task,
                    )
                )

        if (
            env_status.state is not rollout_status.state
            and ElasticRankState.COMPLETED in states
        ):
            env_result, rollout_result = await self._wait_tasks(
                (record.env_run_task, record.rollout_run_task),
                operation=f"rank {rank} completion race",
            )
            if not isinstance(env_result, ElasticRunResult) or not isinstance(
                rollout_result, ElasticRunResult
            ):
                raise ResizeCoordinatorError(
                    f"rank {rank} returned invalid run results"
                )
            if env_result.outcome is ElasticRunOutcome.COMPLETED and (
                rollout_result.outcome is ElasticRunOutcome.COMPLETED
            ):
                await self._offload_completed_rank(rank, context)
                record.token = None
            else:
                token = self._validate_pause_results(rank, env_result, rollout_result)
                if (
                    token.worker_rank != rank
                    or token.lifecycle_generation != context.lifecycle_generation
                    or token.policy_version != context.policy_version
                ):
                    raise ResizeCoordinatorError(
                        f"rank {rank} pause token identity mismatch"
                    )
                env_receipt, rollout_receipt = await self._call_pair(
                    rank,
                    "offload_elastic_environment",
                    "offload_elastic_rollout",
                    operation="pause offload",
                    env_args=(token,),
                    rollout_args=(token,),
                )
                for receipt in (env_receipt, rollout_receipt):
                    if (
                        not isinstance(receipt, ResidencyReceipt)
                        or receipt.token != token
                        or (receipt.state is not ElasticRankState.PAUSED)
                    ):
                        raise ResizeCoordinatorError(
                            f"rank {rank} returned an invalid pause receipt"
                        )
                record.token = token
            record.last_env_result = env_result
            record.last_rollout_result = rollout_result
        elif env_status.state is ElasticRankState.COMPLETED:
            env_result, rollout_result = await self._wait_tasks(
                (record.env_run_task, record.rollout_run_task),
                operation=f"rank {rank} completed calls",
            )
            if not isinstance(env_result, ElasticRunResult) or not isinstance(
                rollout_result, ElasticRunResult
            ):
                raise ResizeCoordinatorError(
                    f"rank {rank} returned invalid run results"
                )
            if env_result.outcome is ElasticRunOutcome.COMPLETED and (
                rollout_result.outcome is ElasticRunOutcome.COMPLETED
            ):
                await self._offload_completed_rank(rank, context)
                record.token = None
            else:
                token = self._validate_pause_results(rank, env_result, rollout_result)
                if (
                    token.worker_rank != rank
                    or token.lifecycle_generation != context.lifecycle_generation
                    or token.policy_version != context.policy_version
                ):
                    raise ResizeCoordinatorError(
                        f"rank {rank} pause token identity mismatch"
                    )
                env_receipt, rollout_receipt = await self._call_pair(
                    rank,
                    "offload_elastic_environment",
                    "offload_elastic_rollout",
                    operation="pause offload",
                    env_args=(token,),
                    rollout_args=(token,),
                )
                for receipt in (env_receipt, rollout_receipt):
                    if (
                        not isinstance(receipt, ResidencyReceipt)
                        or receipt.token != token
                        or (receipt.state is not ElasticRankState.PAUSED)
                    ):
                        raise ResizeCoordinatorError(
                            f"rank {rank} returned an invalid pause receipt"
                        )
                record.token = token
            record.last_env_result = env_result
            record.last_rollout_result = rollout_result
        elif env_status.state is ElasticRankState.ACTIVE:
            request = DrainRequest(
                request_id=uuid.uuid4().hex,
                worker_rank=rank,
                lifecycle_generation=context.lifecycle_generation,
                expected_policy_version=context.policy_version,
            )
            try:
                await self._call_pair(
                    rank,
                    "request_elastic_drain",
                    "request_elastic_drain",
                    operation="drain request",
                    env_args=(request,),
                    rollout_args=(request,),
                )
            except Exception:
                raced_env, raced_rollout = await self._get_pair_status(rank)
                if not (
                    raced_env.state is ElasticRankState.COMPLETED
                    and raced_rollout.state is ElasticRankState.COMPLETED
                ):
                    raise
            env_result, rollout_result = await self._wait_tasks(
                (record.env_run_task, record.rollout_run_task),
                operation=f"rank {rank} safe point",
            )
            if not isinstance(env_result, ElasticRunResult) or not isinstance(
                rollout_result, ElasticRunResult
            ):
                raise ResizeCoordinatorError(
                    f"rank {rank} returned invalid run results"
                )
            if env_result.outcome is ElasticRunOutcome.COMPLETED and (
                rollout_result.outcome is ElasticRunOutcome.COMPLETED
            ):
                await self._offload_completed_rank(rank, context)
                record.token = None
            else:
                token = self._validate_pause_results(rank, env_result, rollout_result)
                if (
                    token.worker_rank != rank
                    or token.lifecycle_generation != context.lifecycle_generation
                    or token.policy_version != context.policy_version
                ):
                    raise ResizeCoordinatorError(
                        f"rank {rank} pause token identity mismatch"
                    )
                env_receipt, rollout_receipt = await self._call_pair(
                    rank,
                    "offload_elastic_environment",
                    "offload_elastic_rollout",
                    operation="pause offload",
                    env_args=(token,),
                    rollout_args=(token,),
                )
                for receipt in (env_receipt, rollout_receipt):
                    if (
                        not isinstance(receipt, ResidencyReceipt)
                        or receipt.token != token
                        or (receipt.state is not ElasticRankState.PAUSED)
                    ):
                        raise ResizeCoordinatorError(
                            f"rank {rank} returned an invalid pause receipt"
                        )
                record.token = token
            record.last_env_result = env_result
            record.last_rollout_result = rollout_result
        else:
            raise RuntimeError(
                f"cannot shrink rank {rank} from {env_status.state.value}"
            )

        final_env, final_rollout = await self._get_pair_status(rank)
        if final_env.state is not final_rollout.state:
            raise ResizeCoordinatorError(
                self._format_peer_state_mismatch(
                    rank,
                    env_status=final_env,
                    rollout_status=final_rollout,
                    env_run_task=record.env_run_task,
                    rollout_run_task=record.rollout_run_task,
                    final=True,
                )
            )
        if final_env.state not in {ElasticRankState.PAUSED, ElasticRankState.COMPLETED}:
            raise ResizeCoordinatorError(
                f"rank {rank} did not reach a releasable state"
            )
        if (
            final_env.model_resident is not False
            or final_rollout.model_resident is not False
            or final_env.cuda_graph_captured
            or final_rollout.cuda_graph_captured
        ):
            raise ResizeCoordinatorError(f"rank {rank} remained resident after shrink")
        record.callback_applied_active = False

    async def _mark_pair_failed(self, rank: int, reason: str) -> None:
        record = self._records[rank]
        record.failure = reason[:512]
        tasks = (
            self._launch(
                self._env_workers[rank], "fail_elastic_lifecycle", reason=record.failure
            ),
            self._launch(
                self._rollout_workers[rank],
                "fail_elastic_lifecycle",
                reason=record.failure,
            ),
        )
        try:
            await self._wait_tasks(tasks, operation=f"rank {rank} failure recording")
        except Exception:
            pass

    async def _run_rank_batch(self, ranks: tuple[int, ...], operation: str) -> None:
        coroutines = [
            self._shrink_rank(rank)
            if operation == "shrink"
            else self._expand_rank(rank)
            for rank in ranks
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if not failures:
            return
        reason = f"{operation} failed: {type(failures[0]).__name__}: {failures[0]}"
        await asyncio.gather(
            *(self._mark_pair_failed(rank, reason) for rank in ranks),
            return_exceptions=True,
        )
        raise ResizeCoordinatorError(reason) from failures[0]

    def _validate_callback_ranks(
        self, remove: list[int], add: list[int]
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not isinstance(remove, list) or not isinstance(add, list):
            raise TypeError("resize rank arguments must be lists")

        def validate(values: list[int], name: str) -> tuple[int, ...]:
            for rank in values:
                if not isinstance(rank, int) or isinstance(rank, bool):
                    raise TypeError(f"{name} ranks must be integers")
                if rank not in self._records:
                    raise ValueError(f"unknown {name} rank {rank}")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} ranks must be unique")
            return tuple(sorted(values))

        removes = validate(remove, "remove")
        adds = validate(add, "add")
        if set(removes) & set(adds):
            raise ValueError("remove and add ranks must not overlap")
        return removes, adds

    async def resize_infer(
        self,
        dp_ranks_to_remove: list[int],
        dp_ranks_to_add: list[int],
    ) -> ActionResponse:
        """Apply worker residency changes before the scheduler may commit them."""
        removes, adds = self._validate_callback_ranks(
            dp_ranks_to_remove, dp_ranks_to_add
        )
        self._require_collection()
        if not removes and not adds:
            async with self._gate:
                if self._closed:
                    raise RuntimeError("coordinator is closed")
                if self._failure is not None:
                    raise ResizeCoordinatorError("coordinator has failed closed")
                return ActionResponse(success=True)
        async with self._gate:
            try:
                await self._wait_for_gate(
                    lambda: (
                        not self._configuration_in_progress
                        and not self._resize_in_progress
                        and self._policy_sync_lease is None
                    ),
                    operation="resize",
                )
            except TimeoutError as exc:
                self._failure = f"TimeoutError: {exc}"[:512]
                raise
            if self._closed:
                raise RuntimeError("coordinator is closed")
            if self._failure is not None:
                raise ResizeCoordinatorError("coordinator has failed closed")
            self._resize_in_progress = True
        try:
            if removes:
                await self._run_rank_batch(removes, "shrink")
            if adds:
                await self._run_rank_batch(adds, "expand")
            return ActionResponse(success=True)
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"[:512]
            raise
        finally:
            async with self._gate:
                self._resize_in_progress = False
                self._gate.notify_all()

    async def begin_policy_sync(
        self, *, expected_policy_version: int
    ) -> PolicySyncLease:
        """Acquire exclusive policy-sync access after all ranks are inactive."""
        if not isinstance(expected_policy_version, int) or isinstance(
            expected_policy_version, bool
        ):
            raise TypeError("expected_policy_version must be an integer")
        if expected_policy_version < 0:
            raise ValueError("expected_policy_version must be non-negative")
        async with self._gate:
            await self._wait_for_gate(
                lambda: (
                    not self._configuration_in_progress
                    and not self._resize_in_progress
                    and self._policy_sync_lease is None
                ),
                operation="policy sync",
            )
            if self._closed:
                raise RuntimeError("coordinator is closed")
            if self._failure is not None:
                raise ResizeCoordinatorError("coordinator has failed closed")
            if self._collection is not None:
                collection_version = self._collection.policy_version
                if expected_policy_version not in {
                    collection_version,
                    collection_version + 1,
                }:
                    raise ValueError(
                        "policy sync version must match the current collection "
                        "or its immediate successor"
                    )

            statuses = await asyncio.gather(
                *(self._get_pair_status(rank) for rank in self._ranks)
            )
            forbidden = {
                ElasticRankState.ACTIVE,
                ElasticRankState.DRAIN_REQUESTED,
                ElasticRankState.SNAPSHOTTING,
                ElasticRankState.PAUSED,
                ElasticRankState.EXPANDING,
            }
            for rank, (env_status, rollout_status) in zip(
                self._ranks, statuses, strict=True
            ):
                if env_status.state in forbidden or rollout_status.state in forbidden:
                    raise RuntimeError(f"rank {rank} is not inactive for policy sync")
                if env_status.model_resident or rollout_status.model_resident:
                    raise RuntimeError(f"rank {rank} is resident before policy sync")
            lease = PolicySyncLease(uuid.uuid4().hex, expected_policy_version)
            self._policy_sync_lease = lease
            return lease

    async def end_policy_sync(self, lease: PolicySyncLease) -> None:
        """Release an exact policy-sync lease and wake resize waiters."""
        if not isinstance(lease, PolicySyncLease):
            raise TypeError("lease must be a PolicySyncLease")
        async with self._gate:
            if self._policy_sync_lease != lease:
                raise ValueError("policy sync lease does not match the active lease")
            self._policy_sync_lease = None
            self._gate.notify_all()

    async def get_status(self) -> CoordinatorStatus:
        """Return observational coordinator and worker lifecycle state."""
        statuses = await asyncio.gather(
            *(self._get_pair_status(rank) for rank in self._ranks)
        )
        paused: list[int] = []
        completed: list[int] = []
        failed: list[int] = []
        for rank, (env_status, rollout_status) in zip(
            self._ranks, statuses, strict=True
        ):
            states = {env_status.state, rollout_status.state}
            if (
                ElasticRankState.FAILED_RESIDENT in states
                or self._records[rank].failure
            ):
                failed.append(rank)
            elif states == {ElasticRankState.PAUSED}:
                paused.append(rank)
            elif states == {ElasticRankState.COMPLETED}:
                completed.append(rank)
        return CoordinatorStatus(
            pipeline_id=self._pipeline_id,
            collection=self._collection,
            callback_applied_active_ranks=tuple(
                rank
                for rank in self._ranks
                if self._records[rank].callback_applied_active
            ),
            paused_ranks=tuple(paused),
            completed_ranks=tuple(completed),
            failed_ranks=tuple(failed),
            resize_in_progress=self._resize_in_progress,
            policy_sync_lease=self._policy_sync_lease,
        )

    async def get_rank_results(
        self, rank: int, *, wait: bool = False
    ) -> tuple[ElasticRunResult, ElasticRunResult] | None:
        """Observe one pair's matching terminal run results without consuming them."""
        if rank not in self._records:
            raise ValueError(f"unknown rank {rank}")
        record = self._records[rank]
        if record.env_run_task is None or record.rollout_run_task is None:
            return None
        if not wait and (
            not record.env_run_task.done() or not record.rollout_run_task.done()
        ):
            return None
        try:
            env_result, rollout_result = await self._wait_tasks(
                (record.env_run_task, record.rollout_run_task),
                operation=f"rank {rank} result observation",
            )
            if not isinstance(env_result, ElasticRunResult) or not isinstance(
                rollout_result, ElasticRunResult
            ):
                raise ResizeCoordinatorError(f"rank {rank} returned invalid results")
            if env_result.outcome is not rollout_result.outcome:
                raise ResizeCoordinatorError(f"rank {rank} peer outcomes do not match")
        except Exception as exc:
            reason = f"result observation failed: {type(exc).__name__}: {exc}"
            await self._mark_pair_failed(rank, reason)
            self._failure = reason[:512]
            raise ResizeCoordinatorError(reason) from exc
        record.last_env_result = env_result
        record.last_rollout_result = rollout_result
        return env_result, rollout_result

    async def get_rank_observation(self, rank: int) -> ElasticRankObservation:
        """Return status, progress, and durable results without consuming them."""
        if rank not in self._records:
            raise ValueError(f"unknown rank {rank}")
        context = self._require_collection()
        env_status, rollout_status = await self._get_pair_status(rank)
        record = self._records[rank]
        if self._is_dormant_completed_pair(
            rank, env_status, rollout_status, context, record
        ):
            return ElasticRankObservation(
                dp_rank=rank,
                env_status=env_status,
                rollout_status=rollout_status,
                progress=None,
                paired_results=None,
                callback_applied_active=False,
                failure=None,
            )
        self._validate_status_identity(
            rank,
            env_status,
            context,
            allow_cold=True,
        )
        self._validate_status_identity(
            rank,
            rollout_status,
            context,
            allow_cold=True,
        )
        progress = None
        if env_status.state is not ElasticRankState.INACTIVE_COLD:
            progress_task = self._launch(
                self._env_workers[rank], "get_elastic_progress"
            )
            (progress,) = await self._wait_tasks(
                (progress_task,), operation=f"rank {rank} progress query"
            )
            if not isinstance(progress, ElasticRankProgress):
                raise ResizeCoordinatorError(
                    f"rank {rank} returned invalid elastic progress"
                )
        paired_results = await self.get_rank_results(rank, wait=False)
        return ElasticRankObservation(
            dp_rank=rank,
            env_status=env_status,
            rollout_status=rollout_status,
            progress=progress,
            paired_results=paired_results,
            callback_applied_active=record.callback_applied_active,
            failure=record.failure,
        )

    async def close(self) -> None:
        """Close only after all rank work and synchronization have stopped."""
        async with self._gate:
            if (
                self._configuration_in_progress
                or self._resize_in_progress
                or self._policy_sync_lease is not None
            ):
                raise RuntimeError("cannot close during resize or policy sync")
            if any(record.callback_applied_active for record in self._records.values()):
                raise RuntimeError("cannot close while ranks are locally active")
            if any(
                task is not None and not task.done()
                for record in self._records.values()
                for task in (record.env_run_task, record.rollout_run_task)
            ):
                raise RuntimeError("cannot close with unresolved run calls")
            self._closed = True
