from __future__ import annotations

import asyncio
import os
import uuid
from types import SimpleNamespace

import pytest
import ray
from rlix_core.protocol.types import (
    COORDINATOR_ACTOR_NAME_PREFIX,
    GENERATION_CLUSTER_NAME,
    Priority,
    ProgressReport,
)
from rlix_core.scheduler.scheduler import SchedulerImpl
from rlix_core.scheduler.types import ClusterAllocation

from rlinf.data.embodied_io_struct import RolloutTransitionIdentity
from rlinf.scheduler.rlix.controller import RLixStageController
from rlinf.scheduler.rlix.coordinator import (
    ResizeCoordinatorError,
    RLixResizeCoordinator,
)
from rlinf.scheduler.rlix.protocol import (
    CoordinatorStatus,
    ElasticCollectionContext,
    PolicySyncLease,
)
from rlinf.workers.elastic_rollout_lifecycle import (
    CompletedResidencyReceipt,
    ElasticRankProgress,
    ElasticRankState,
    ElasticRankStatus,
    ElasticRunOutcome,
    ElasticRunResult,
    ResidencyReceipt,
    SafePointToken,
)


class _FakeElasticWorker:
    def __init__(
        self,
        rank: int,
        *,
        token_suffix: str = "",
        event_log: list[str] | None = None,
        label: str = "worker",
    ) -> None:
        self.rank = rank
        self.token_suffix = token_suffix
        self.event_log = event_log
        self.label = label
        self.state = ElasticRankState.INACTIVE_COLD
        self.lifecycle: int | None = None
        self.policy_version: int | None = None
        self.resident = False
        self.failure: str | None = None
        self.transition: RolloutTransitionIdentity | None = None
        self.drain_request = None
        self.drain_event = asyncio.Event()
        self.complete_event = asyncio.Event()
        self.activation_gate = asyncio.Event()
        self.activation_gate.set()
        self.active_event = asyncio.Event()
        self.run_cancelled = False
        self.complete_on_drain_request = False
        self.fail_prepare = False
        self.fail_pause_offload = False
        self.fail_resume = False
        self.prepare_count = 0
        self.resume_count = 0
        self.completed_offload_count = 0

    def _status(self) -> ElasticRankStatus:
        return ElasticRankStatus(
            state=self.state,
            worker_rank=self.rank,
            lifecycle_generation=self.lifecycle,
            policy_version=self.policy_version,
            expected_transition_id=self.transition,
            drain_request_id=(
                self.drain_request.request_id
                if self.drain_request is not None
                else None
            ),
            snapshot_ready=self.state
            in {ElasticRankState.SNAPSHOTTING, ElasticRankState.PAUSED},
            model_resident=self.resident,
            cuda_graph_captured=False,
            failure=self.failure,
        )

    def get_elastic_status(self) -> ElasticRankStatus:
        return self._status()

    def get_elastic_progress(self) -> ElasticRankProgress:
        return ElasticRankProgress(
            dp_rank=self.rank,
            lifecycle_generation=self.lifecycle,
            state=self.state,
            assigned_trajectories=2,
            completed_trajectories=(
                2 if self.state is ElasticRankState.COMPLETED else 0
            ),
            snapshot_ready=self.state
            in {ElasticRankState.SNAPSHOTTING, ElasticRankState.PAUSED},
            failed=self.state is ElasticRankState.FAILED_RESIDENT,
        )

    def prepare_elastic_collection(
        self, *, lifecycle_generation: int, expected_policy_version: int
    ) -> ElasticRankStatus:
        if self.fail_prepare:
            raise RuntimeError("injected preparation failure")
        if self.event_log is not None:
            self.event_log.append(f"prepare:{self.label}")
        self.lifecycle = lifecycle_generation
        self.policy_version = expected_policy_version
        self.transition = RolloutTransitionIdentity(
            lifecycle_generation, self.rank, 0, 0
        )
        self.state = ElasticRankState.EXPANDING
        self.resident = True
        self.drain_request = None
        self.drain_event = asyncio.Event()
        self.complete_event = asyncio.Event()
        self.active_event = asyncio.Event()
        self.prepare_count += 1
        return self._status()

    def prepare_elastic_resume(self, token: SafePointToken) -> ResidencyReceipt:
        if self.fail_resume:
            raise RuntimeError("injected resume failure")
        self.state = ElasticRankState.EXPANDING
        self.resident = True
        self.drain_request = None
        self.drain_event = asyncio.Event()
        self.complete_event = asyncio.Event()
        self.active_event = asyncio.Event()
        self.resume_count += 1
        return ResidencyReceipt(
            token=token,
            state=ElasticRankState.EXPANDING,
            model_resident=True,
            cuda_graph_captured=False,
        )

    async def interact_until_pause_or_complete(self, *_args) -> ElasticRunResult:
        return await self._run()

    async def generate_until_pause_or_complete(self, *_args) -> ElasticRunResult:
        return await self._run()

    async def _run(self) -> ElasticRunResult:
        try:
            await self.activation_gate.wait()
            self.state = ElasticRankState.ACTIVE
            self.active_event.set()
            drain_wait = asyncio.create_task(self.drain_event.wait())
            complete_wait = asyncio.create_task(self.complete_event.wait())
            done, pending = await asyncio.wait(
                (drain_wait, complete_wait), return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if complete_wait in done and self.complete_event.is_set():
                self.state = ElasticRankState.COMPLETED
                return ElasticRunResult(ElasticRunOutcome.COMPLETED, None, None)
            request = self.drain_request
            assert request is not None
            token = SafePointToken(
                request_id=f"{request.request_id}{self.token_suffix}",
                worker_rank=self.rank,
                lifecycle_generation=self.lifecycle,
                policy_version=self.policy_version,
                next_transition_id=self.transition,
            )
            self.state = ElasticRankState.SNAPSHOTTING
            return ElasticRunResult(ElasticRunOutcome.PAUSE_READY, token, None)
        except asyncio.CancelledError:
            self.run_cancelled = True
            raise

    async def request_elastic_drain(self, request) -> ElasticRankStatus:
        if self.complete_on_drain_request:
            self.complete_event.set()
            while self.state is not ElasticRankState.COMPLETED:
                await asyncio.sleep(0)
            raise RuntimeError("rank completed before drain registration")
        self.drain_request = request
        self.state = ElasticRankState.DRAIN_REQUESTED
        self.drain_event.set()
        return self._status()

    def offload_elastic_environment(self, token: SafePointToken) -> ResidencyReceipt:
        return self._offload_pause(token)

    def offload_elastic_rollout(self, token: SafePointToken) -> ResidencyReceipt:
        return self._offload_pause(token)

    def _offload_pause(self, token: SafePointToken) -> ResidencyReceipt:
        if self.fail_pause_offload:
            raise RuntimeError("injected pause offload failure")
        if self.event_log is not None:
            self.event_log.append(f"pause-offload:{self.label}")
        self.state = ElasticRankState.PAUSED
        self.resident = False
        return ResidencyReceipt(
            token=token,
            state=ElasticRankState.PAUSED,
            model_resident=False,
            cuda_graph_captured=False,
        )

    def offload_completed_elastic_environment(self) -> CompletedResidencyReceipt:
        return self._offload_completed()

    def offload_completed_elastic_rollout(self) -> CompletedResidencyReceipt:
        return self._offload_completed()

    def _offload_completed(self) -> CompletedResidencyReceipt:
        self.resident = False
        self.completed_offload_count += 1
        return CompletedResidencyReceipt(
            worker_rank=self.rank,
            lifecycle_generation=self.lifecycle,
            policy_version=self.policy_version,
            state=ElasticRankState.COMPLETED,
            model_resident=False,
            cuda_graph_captured=False,
        )

    def fail_elastic_lifecycle(self, *, reason: str) -> ElasticRankStatus:
        self.failure = reason
        if self.state not in {ElasticRankState.PAUSED, ElasticRankState.INACTIVE_COLD}:
            self.state = ElasticRankState.FAILED_RESIDENT
        return self._status()


def _coordinator(
    *,
    ranks: int = 1,
    timeout: float = 1.0,
    rollout_token_suffix: str = "",
    event_log: list[str] | None = None,
):
    env = {
        rank: _FakeElasticWorker(rank, event_log=event_log, label=f"env-{rank}")
        for rank in range(ranks)
    }
    rollout = {
        rank: _FakeElasticWorker(
            rank,
            token_suffix=rollout_token_suffix,
            event_log=event_log,
            label=f"rollout-{rank}",
        )
        for rank in range(ranks)
    }
    coordinator = RLixResizeCoordinator(
        pipeline_id="embodied_abc123def456",
        env_workers=env,
        rollout_workers=rollout,
        operation_timeout_s=timeout,
        activation_poll_interval_s=0.001,
    )
    return coordinator, env, rollout


async def _configure(coordinator: RLixResizeCoordinator, ranks: int = 1) -> None:
    await coordinator.configure_collection(
        ElasticCollectionContext(1, 3, tuple(range(ranks))),
        env_input_channel=object(),
        rollout_request_channel=object(),
    )


class _DirectResizeRemote:
    def __init__(self, coordinator: RLixResizeCoordinator) -> None:
        self._coordinator = coordinator

    def remote(self, **kwargs):
        return self._coordinator.resize_infer(**kwargs)


class _DirectCoordinatorHandle:
    def __init__(self, coordinator: RLixResizeCoordinator) -> None:
        self.resize_infer = _DirectResizeRemote(coordinator)


def test_collection_context_validates_canonical_rank_identity() -> None:
    context = ElasticCollectionContext(
        lifecycle_generation=2,
        policy_version=3,
        dp_ranks=(0, 1),
    )

    assert context.dp_ranks == (0, 1)
    with pytest.raises(ValueError, match="positive"):
        ElasticCollectionContext(0, 3, (0,))
    with pytest.raises(ValueError, match="contiguous"):
        ElasticCollectionContext(2, 3, (1,))
    with pytest.raises(ValueError, match="unique"):
        ElasticCollectionContext(2, 3, (0, 0))
    with pytest.raises(TypeError, match="tuple"):
        ElasticCollectionContext(2, 3, [0])  # type: ignore[arg-type]


def test_policy_sync_lease_validates_identity_and_version() -> None:
    assert PolicySyncLease("lease-1", 4).expected_policy_version == 4

    with pytest.raises(ValueError, match="non-empty"):
        PolicySyncLease("", 4)
    with pytest.raises(ValueError, match="non-negative"):
        PolicySyncLease("lease-1", -1)


def test_coordinator_status_requires_sorted_unique_rank_sets() -> None:
    context = ElasticCollectionContext(2, 3, (0, 1))
    status = CoordinatorStatus(
        pipeline_id="embodied_abc123def456",
        collection=context,
        callback_applied_active_ranks=(0,),
        paused_ranks=(1,),
        completed_ranks=(),
        failed_ranks=(),
        resize_in_progress=False,
        policy_sync_lease=None,
    )

    assert status.collection == context
    with pytest.raises(ValueError, match="sorted and unique"):
        CoordinatorStatus(
            pipeline_id="embodied_abc123def456",
            collection=context,
            callback_applied_active_ranks=(1, 0),
            paused_ranks=(),
            completed_ranks=(),
            failed_ranks=(),
            resize_in_progress=False,
            policy_sync_lease=None,
        )


def test_rlinf_adapters_use_core_pipeline_identity_validation() -> None:
    context = ElasticCollectionContext(2, 3, (0,))
    with pytest.raises(ValueError, match="must not contain"):
        CoordinatorStatus(
            pipeline_id="invalid:pipeline",
            collection=context,
            callback_applied_active_ranks=(),
            paused_ranks=(),
            completed_ranks=(),
            failed_ranks=(),
            resize_in_progress=False,
            policy_sync_lease=None,
        )
    with pytest.raises(ValueError, match="must not contain"):
        RLixResizeCoordinator(
            pipeline_id="invalid:pipeline",
            env_workers={0: object()},
            rollout_workers={0: object()},
        )
    with pytest.raises(ValueError, match="must not contain"):
        RLixStageController(
            pipeline_id="invalid:pipeline",
            ray_namespace="pipeline-namespace",
            env_worker_group=object(),
            rollout_worker_group=object(),
        )


def test_paired_wait_surfaces_first_exception_without_waiting_for_peer() -> None:
    async def run() -> None:
        coordinator, _env, _rollout = _coordinator(timeout=1.0)
        peer_release = asyncio.Event()

        async def fail() -> None:
            await asyncio.sleep(0)
            raise RuntimeError("injected paired task failure")

        async def block() -> None:
            await peer_release.wait()

        failed_task = asyncio.create_task(fail())
        blocked_task = asyncio.create_task(block())
        try:
            with pytest.raises(RuntimeError, match="injected paired task failure"):
                await asyncio.wait_for(
                    coordinator._wait_tasks(
                        (failed_task, blocked_task), operation="test paired tasks"
                    ),
                    timeout=0.1,
                )
            assert not blocked_task.done()
        finally:
            peer_release.set()
            await asyncio.gather(blocked_task, return_exceptions=True)

    asyncio.run(run())


def test_cold_expand_pause_shrink_and_exact_token_resume() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)

        response = await coordinator.resize_infer([], [0])
        assert response.success
        assert (await coordinator.get_status()).callback_applied_active_ranks == (0,)

        response = await coordinator.resize_infer([0], [])
        assert response.success
        paused = await coordinator.get_status()
        assert paused.paused_ranks == (0,)
        assert not env[0].resident
        assert not rollout[0].resident

        response = await coordinator.resize_infer([], [0])
        assert response.success
        assert env[0].resume_count == 1
        assert rollout[0].resume_count == 1
        assert (await coordinator.get_status()).callback_applied_active_ranks == (0,)

    asyncio.run(run())


def test_completed_rank_result_and_token_free_release() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)
        await coordinator.resize_infer([], [0])
        env[0].complete_event.set()
        rollout[0].complete_event.set()

        results = await coordinator.get_rank_results(0, wait=True)
        assert results is not None
        assert results[0].outcome is ElasticRunOutcome.COMPLETED
        assert results[1].outcome is ElasticRunOutcome.COMPLETED
        before = await coordinator.get_status()
        assert before.completed_ranks == (0,)
        assert before.callback_applied_active_ranks == (0,)

        await coordinator.resize_infer([0], [])
        after = await coordinator.get_status()
        assert after.completed_ranks == (0,)
        assert after.callback_applied_active_ranks == ()
        assert env[0].completed_offload_count == 1
        assert rollout[0].completed_offload_count == 1

    asyncio.run(run())


def test_rank_observation_is_repeatable_and_includes_durable_completion() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)
        cold = await coordinator.get_rank_observation(0)
        assert cold.progress is None
        assert not cold.callback_applied_active

        await coordinator.resize_infer([], [0])
        env[0].complete_event.set()
        rollout[0].complete_event.set()
        await coordinator.get_rank_results(0, wait=True)

        first = await coordinator.get_rank_observation(0)
        second = await coordinator.get_rank_observation(0)
        assert first == second
        assert first.progress.completed_trajectories == 2
        assert first.paired_results is not None
        assert first.callback_applied_active

    asyncio.run(run())


def test_final_completion_race_uses_completed_release_without_pause_token() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)
        await coordinator.resize_infer([], [0])
        env[0].complete_on_drain_request = True
        rollout[0].complete_on_drain_request = True

        response = await coordinator.resize_infer([0], [])

        assert response.success
        assert (await coordinator.get_status()).completed_ranks == (0,)
        assert env[0].completed_offload_count == 1
        assert rollout[0].completed_offload_count == 1

    asyncio.run(run())


def test_completed_pair_can_activate_only_in_strictly_newer_collection() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)
        await coordinator.resize_infer([], [0])
        env[0].complete_event.set()
        rollout[0].complete_event.set()
        await coordinator.get_rank_results(0, wait=True)
        await coordinator.resize_infer([0], [])
        await coordinator.configure_collection(
            ElasticCollectionContext(2, 3, (0,)),
            env_input_channel=object(),
            rollout_request_channel=object(),
        )

        await coordinator.resize_infer([], [0])

        assert env[0].prepare_count == 2
        assert rollout[0].prepare_count == 2
        assert env[0].lifecycle == 2
        assert rollout[0].lifecycle == 2

    asyncio.run(run())


def test_selected_rank_resize_leaves_sibling_active() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator(ranks=2)
        await _configure(coordinator, ranks=2)
        await coordinator.resize_infer([], [1, 0])

        await coordinator.resize_infer([0], [])

        status = await coordinator.get_status()
        assert status.paused_ranks == (0,)
        assert status.callback_applied_active_ranks == (1,)
        assert env[1].state is ElasticRankState.ACTIVE
        assert rollout[1].state is ElasticRankState.ACTIVE

    asyncio.run(run())


def test_direct_mixed_callback_finishes_all_shrinks_before_expansion() -> None:
    async def run() -> None:
        events: list[str] = []
        coordinator, _env, _rollout = _coordinator(ranks=2, event_log=events)
        await _configure(coordinator, ranks=2)
        await coordinator.resize_infer([], [0])
        events.clear()
        await coordinator.resize_infer([0], [1])

        offload_indices = [
            index
            for index, event in enumerate(events)
            if event in {"pause-offload:env-0", "pause-offload:rollout-0"}
        ]
        prepare_indices = [
            index
            for index, event in enumerate(events)
            if event in {"prepare:env-1", "prepare:rollout-1"}
        ]
        assert len(offload_indices) == 2
        assert len(prepare_indices) == 2
        assert max(offload_indices) < min(prepare_indices)

    asyncio.run(run())


def test_shrink_tolerates_transient_completed_active_peer_race() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)
        await coordinator.resize_infer([], [0])
        env[0].complete_event.set()
        while env[0].state is not ElasticRankState.COMPLETED:
            await asyncio.sleep(0)
        assert rollout[0].state is ElasticRankState.ACTIVE

        async def complete_rollout_after_shrink_samples_status() -> None:
            await asyncio.sleep(0.01)
            rollout[0].complete_event.set()

        completion = asyncio.create_task(complete_rollout_after_shrink_samples_status())
        await coordinator.resize_infer([0], [])
        await completion

        assert env[0].state is ElasticRankState.COMPLETED
        assert rollout[0].state is ElasticRankState.COMPLETED
        assert not env[0].resident
        assert not rollout[0].resident
        assert env[0].completed_offload_count == 1
        assert rollout[0].completed_offload_count == 1
        status = await coordinator.get_status()
        assert status.callback_applied_active_ranks == ()
        assert status.completed_ranks == (0,)

    asyncio.run(run())


def test_callback_validation_rejects_malformed_ranks_before_worker_calls() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)

        with pytest.raises(TypeError, match="lists"):
            await coordinator.resize_infer((0,), [])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="integers"):
            await coordinator.resize_infer([], [True])
        with pytest.raises(ValueError, match="unique"):
            await coordinator.resize_infer([], [0, 0])
        with pytest.raises(ValueError, match="unknown"):
            await coordinator.resize_infer([], [1])
        with pytest.raises(ValueError, match="overlap"):
            await coordinator.resize_infer([0], [0])
        assert env[0].prepare_count == 0
        assert rollout[0].prepare_count == 0

    asyncio.run(run())


def test_shrink_peer_state_mismatch_reports_both_worker_states() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)
        await coordinator.resize_infer([], [0])
        env[0].state = ElasticRankState.SNAPSHOTTING
        env[0].failure = "env diagnostic"
        rollout[0].failure = "rollout diagnostic"

        with pytest.raises(ResizeCoordinatorError) as exc_info:
            await coordinator.resize_infer([0], [])

        message = str(exc_info.value)
        assert "env_state=snapshotting" in message
        assert "rollout_state=active" in message
        assert "env_run_task=pending" in message
        assert "rollout_run_task=pending" in message
        assert "env_failure='env diagnostic'" in message
        assert "rollout_failure='rollout diagnostic'" in message

    asyncio.run(run())


def test_token_mismatch_fails_closed_without_clearing_active_marker() -> None:
    async def run() -> None:
        coordinator, _env, _rollout = _coordinator(rollout_token_suffix="-mismatch")
        await _configure(coordinator)
        await coordinator.resize_infer([], [0])

        with pytest.raises(ResizeCoordinatorError, match="tokens do not match"):
            await coordinator.resize_infer([0], [])

        status = await coordinator.get_status()
        assert status.callback_applied_active_ranks == (0,)
        assert status.failed_ranks == (0,)
        with pytest.raises(ResizeCoordinatorError, match="failed closed"):
            await coordinator.resize_infer([], [])

    asyncio.run(run())


def test_multi_rank_expansion_failure_fails_batch_without_rolling_back_sibling() -> (
    None
):
    async def run() -> None:
        coordinator, env, rollout = _coordinator(ranks=2)
        rollout[1].fail_prepare = True
        await _configure(coordinator, ranks=2)

        with pytest.raises(ResizeCoordinatorError, match="preparation failure"):
            await coordinator.resize_infer([], [0, 1])

        status = await coordinator.get_status()
        assert status.callback_applied_active_ranks == (0,)
        assert status.failed_ranks == (0, 1)
        assert env[0].resident
        assert rollout[0].resident
        assert env[0].state is ElasticRankState.FAILED_RESIDENT
        assert rollout[0].state is ElasticRankState.FAILED_RESIDENT

    asyncio.run(run())


def test_partial_pause_offload_failure_does_not_onload_successful_peer() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        rollout[0].fail_pause_offload = True
        await _configure(coordinator)
        await coordinator.resize_infer([], [0])

        with pytest.raises(ResizeCoordinatorError, match="pause offload failure"):
            await coordinator.resize_infer([0], [])

        assert env[0].state is ElasticRankState.PAUSED
        assert not env[0].resident
        assert rollout[0].state is ElasticRankState.FAILED_RESIDENT
        assert rollout[0].resident
        assert (await coordinator.get_status()).callback_applied_active_ranks == (0,)

    asyncio.run(run())


def test_partial_resume_failure_does_not_offload_successful_peer() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)
        await coordinator.resize_infer([], [0])
        await coordinator.resize_infer([0], [])
        rollout[0].fail_resume = True

        with pytest.raises(ResizeCoordinatorError, match="resume failure"):
            await coordinator.resize_infer([], [0])

        assert env[0].resident
        assert env[0].state is ElasticRankState.FAILED_RESIDENT
        assert not rollout[0].resident
        assert rollout[0].state is ElasticRankState.PAUSED

    asyncio.run(run())


def test_core_release_keeps_ownership_when_rlinf_completed_offload_fails() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        await _configure(coordinator)
        await coordinator.resize_infer([], [0])
        env[0].complete_event.set()
        rollout[0].complete_event.set()
        await coordinator.get_rank_results(0, wait=True)
        rollout[0].fail_pause_offload = True

        def fail_completed_offload() -> CompletedResidencyReceipt:
            raise RuntimeError("injected completed release failure")

        rollout[0].offload_completed_elastic_rollout = fail_completed_offload

        scheduler = SchedulerImpl()
        scheduler._topology_ready.set()
        scheduler._num_gpus = 1
        scheduler._required_gpus_per_node = 1
        scheduler._state.idle_gpus = set()
        pipeline_id = "embodied_abc123def456"
        cluster_id = f"{pipeline_id}_{GENERATION_CLUSTER_NAME}"
        await scheduler.register_pipeline_topology(
            pipeline_id=pipeline_id,
            ray_namespace="pipeline-namespace",
            cluster_tp_configs={GENERATION_CLUSTER_NAME: 1},
            cluster_device_mappings={GENERATION_CLUSTER_NAME: [0]},
            cluster_dp_device_mappings={GENERATION_CLUSTER_NAME: {0: [0]}},
        )
        scheduler._state.pipeline_registry[pipeline_id]["admitted"] = True
        scheduler._state.active_allocations[cluster_id] = ClusterAllocation(
            cluster_id=cluster_id,
            gpu_ids=[0],
            priority=Priority.GENERATION,
            active_dp_ranks={0},
            dp_rank_to_gpus={0: [0]},
        )
        scheduler._state.latest_progress_by_pipeline[pipeline_id] = {
            "train": {
                "default": ProgressReport(
                    pipeline_id=pipeline_id,
                    step_target_trajectories=1,
                    metrics={
                        "completed": 1,
                        "active_dp_ranks": [0],
                        "completed_dp_ranks": [0],
                        "resumable_dp_ranks": [],
                    },
                )
            }
        }
        scheduler._coordinator_handle_cache[pipeline_id] = (
            "pipeline-namespace",
            _DirectCoordinatorHandle(coordinator),
        )
        release = asyncio.create_task(
            scheduler.await_release_dp_ranks(cluster_id=cluster_id, ranks=[0])
        )
        await asyncio.sleep(0)

        with pytest.raises(ResizeCoordinatorError, match="completed release failure"):
            await scheduler.scheduling_cycle()

        allocation = scheduler._state.active_allocations[cluster_id]
        assert allocation.active_dp_ranks == {0}
        assert allocation.gpu_ids == [0]
        assert scheduler._state.idle_gpus == set()
        assert not release.done()
        release.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release

    asyncio.run(run())


def test_activation_deadline_does_not_cancel_worker_calls() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator(timeout=0.02)
        env[0].activation_gate.clear()
        rollout[0].activation_gate.clear()
        await _configure(coordinator)

        with pytest.raises(ResizeCoordinatorError, match="Timed out"):
            await coordinator.resize_infer([], [0])

        assert not env[0].run_cancelled
        assert not rollout[0].run_cancelled
        env[0].activation_gate.set()
        rollout[0].activation_gate.set()
        await asyncio.gather(env[0].active_event.wait(), rollout[0].active_event.wait())

    asyncio.run(run())


def test_policy_sync_lease_blocks_resize_and_requires_exact_end_token() -> None:
    async def run() -> None:
        coordinator, _env, _rollout = _coordinator()
        await _configure(coordinator)
        lease = await coordinator.begin_policy_sync(expected_policy_version=3)
        resize = asyncio.create_task(coordinator.resize_infer([], [0]))
        await asyncio.sleep(0)
        assert not resize.done()

        with pytest.raises(ValueError, match="does not match"):
            await coordinator.end_policy_sync(PolicySyncLease("wrong", 3))
        await asyncio.sleep(0)
        assert not resize.done()
        await coordinator.end_policy_sync(lease)
        response = await resize
        assert response.success

    asyncio.run(run())


def test_policy_sync_rejects_noncurrent_nonnext_policy_version() -> None:
    async def run() -> None:
        coordinator, _env, _rollout = _coordinator()
        await _configure(coordinator)

        with pytest.raises(ValueError, match="immediate successor"):
            await coordinator.begin_policy_sync(expected_policy_version=5)

        lease = await coordinator.begin_policy_sync(expected_policy_version=4)
        await coordinator.end_policy_sync(lease)

    asyncio.run(run())


def test_policy_sync_lease_never_expires_when_waiting_resize_times_out() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator(timeout=0.02)
        await _configure(coordinator)
        lease = await coordinator.begin_policy_sync(expected_policy_version=3)

        with pytest.raises(TimeoutError, match="begin resize"):
            await coordinator.resize_infer([], [0])

        status = await coordinator.get_status()
        assert status.policy_sync_lease == lease
        assert env[0].prepare_count == 0
        assert rollout[0].prepare_count == 0
        await coordinator.end_policy_sync(lease)
        with pytest.raises(ResizeCoordinatorError, match="failed closed"):
            await coordinator.resize_infer([], [0])

    asyncio.run(run())


def test_policy_sync_waits_for_resize_then_rejects_active_rank() -> None:
    async def run() -> None:
        coordinator, env, rollout = _coordinator()
        env[0].activation_gate.clear()
        rollout[0].activation_gate.clear()
        await _configure(coordinator)
        resize = asyncio.create_task(coordinator.resize_infer([], [0]))
        await asyncio.sleep(0.01)
        sync = asyncio.create_task(
            coordinator.begin_policy_sync(expected_policy_version=3)
        )
        await asyncio.sleep(0)
        assert not sync.done()

        env[0].activation_gate.set()
        rollout[0].activation_gate.set()
        await resize
        with pytest.raises(RuntimeError, match="not inactive"):
            await sync

    asyncio.run(run())


def test_stage_controller_validates_public_ranked_handle_maps(monkeypatch) -> None:
    class _RemoteClass:
        options_kwargs = None
        constructor_kwargs = None

        def options(self, **kwargs):
            self.options_kwargs = kwargs
            return self

        def remote(self, **kwargs):
            self.constructor_kwargs = kwargs
            return object()

    remote_class = _RemoteClass()
    monkeypatch.setattr(ray, "remote", lambda _class: remote_class)
    valid = SimpleNamespace(worker_info_list=[SimpleNamespace(rank=0, worker=object())])
    two_ranks = SimpleNamespace(
        worker_info_list=[
            SimpleNamespace(rank=0, worker=object()),
            SimpleNamespace(rank=1, worker=object()),
        ]
    )
    mismatched = SimpleNamespace(
        worker_info_list=[SimpleNamespace(rank=0, worker=object())]
    )

    with pytest.raises(ValueError, match="must match"):
        RLixStageController(
            pipeline_id="embodied_abc123def456",
            ray_namespace="pipeline-namespace",
            env_worker_group=two_ranks,
            rollout_worker_group=mismatched,
            worker_max_concurrency=2,
        )
    with pytest.raises(ValueError, match="max_concurrency"):
        RLixStageController(
            pipeline_id="embodied_abc123def456",
            ray_namespace="pipeline-namespace",
            env_worker_group=valid,
            rollout_worker_group=valid,
            worker_max_concurrency=1,
        )

    controller = RLixStageController(
        pipeline_id="embodied_abc123def456",
        ray_namespace="pipeline-namespace",
        env_worker_group=valid,
        rollout_worker_group=valid,
        worker_max_concurrency=2,
    )
    assert controller.actor_name == (
        f"{COORDINATOR_ACTOR_NAME_PREFIX}embodied_abc123def456"
    )
    assert remote_class.options_kwargs == {
        "name": controller.actor_name,
        "namespace": "pipeline-namespace",
        "max_concurrency": 1000,
        "max_restarts": 0,
        "max_task_retries": 0,
    }


@pytest.mark.skipif(
    os.environ.get("RLINF_RUN_LOCAL_RAY_TEST") != "1",
    reason="local Ray startup is an opt-in integration check",
)
def test_named_ray_coordinator_uses_exact_registered_namespace() -> None:
    from ray._private import ray_constants

    class ColdWorker:
        def __init__(self, rank: int) -> None:
            self.rank = rank

        def get_elastic_status(self) -> ElasticRankStatus:
            return ElasticRankStatus(
                state=ElasticRankState.INACTIVE_COLD,
                worker_rank=self.rank,
                lifecycle_generation=None,
                policy_version=None,
                expected_transition_id=None,
                drain_request_id=None,
                snapshot_ready=False,
                model_resident=False,
                cuda_graph_captured=False,
                failure=None,
            )

    started_ray = not ray.is_initialized()
    uv_hook_enabled = ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV
    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = False
    if started_ray:
        ray.init(
            num_cpus=3,
            include_dashboard=False,
            log_to_driver=False,
            _skip_env_hook=True,
        )
    namespace = f"rlix_test_{uuid.uuid4().hex}"
    pipeline_id = f"embodied_{uuid.uuid4().hex[:12]}"
    worker_class = ray.remote(ColdWorker)
    env_worker = worker_class.options(max_concurrency=2).remote(0)
    rollout_worker = worker_class.options(max_concurrency=2).remote(0)
    group_env = SimpleNamespace(
        worker_info_list=[SimpleNamespace(rank=0, worker=env_worker)]
    )
    group_rollout = SimpleNamespace(
        worker_info_list=[SimpleNamespace(rank=0, worker=rollout_worker)]
    )

    async def run() -> None:
        controller = RLixStageController(
            pipeline_id=pipeline_id,
            ray_namespace=namespace,
            env_worker_group=group_env,
            rollout_worker_group=group_rollout,
            worker_max_concurrency=2,
        )
        looked_up = ray.get_actor(
            f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}", namespace=namespace
        )
        assert looked_up._actor_id == controller.coordinator._actor_id
        status = await controller.get_status()
        assert status.pipeline_id == pipeline_id
        await controller.close()

    try:
        asyncio.run(run())
    finally:
        ray.kill(env_worker, no_restart=True)
        ray.kill(rollout_worker, no_restart=True)
        if started_ray:
            ray.shutdown()
        ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = uv_hook_enabled
