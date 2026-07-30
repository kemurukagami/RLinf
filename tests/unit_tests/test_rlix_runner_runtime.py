"""CPU-only tests for the Task 7 runner-facing RLix runtime."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rlix_core.protocol.types import (
    ACTOR_TRAIN_CLUSTER_NAME,
    INITIALIZATION_CLUSTER_NAME,
    POLICY_SYNC_CLUSTER_NAME,
    Priority,
)

from rlinf.scheduler.rlix.protocol import (
    ElasticBatchReceipt,
    FixedStageResidencyReceipt,
    FixedWorkerResidency,
    RunnerStageState,
)
from rlinf.scheduler.rlix.runtime import RegisteredRLixPipeline
from rlinf.workers.elastic_rollout_lifecycle import (
    ElasticRankProgress,
    ElasticRankState,
    ElasticRunOutcome,
    ElasticRunResult,
)


class _Scheduler:
    def __init__(
        self,
        events: list[str],
        *,
        grants: dict[str, list[int]] | None = None,
        fail_request: bool = False,
        fail_release: bool = False,
    ) -> None:
        self.events = events
        self.grants = grants or {}
        self.fail_request = fail_request
        self.fail_release = fail_release
        self.reports = []

    async def request_gpus(self, **kwargs):
        self.events.append(
            f"request:{kwargs['cluster_id']}:{kwargs['priority'].name}:"
            f"{kwargs['global_step']}"
        )
        if self.fail_request:
            raise RuntimeError("request failed")
        return self.grants.get(kwargs["cluster_id"], [0, 1, 2])

    async def notify_release_gpus(self, **kwargs):
        self.events.append(f"release:{kwargs['cluster_id']}:{kwargs['global_step']}")
        if self.fail_release:
            raise RuntimeError("release failed")

    async def report_progress(self, report) -> None:
        self.reports.append(report)
        self.events.append(
            f"progress:{report.metrics['completed']}:"
            f"{report.metrics['active_dp_ranks']}"
        )

    async def clear_progress(self, *, pipeline_id: str) -> None:
        self.events.append(f"clear_progress:{pipeline_id}")

    async def await_release_dp_ranks(self, **kwargs) -> None:
        self.events.append(f"await_release:{tuple(kwargs['ranks'])}")

    async def release_dp_ranks_then_request_gpus(self, **kwargs):
        self.events.append(
            f"transition:{tuple(kwargs['release_ranks'])}:"
            f"{kwargs['request_cluster_id']}:{kwargs['request_priority'].name}"
        )
        return self.grants.get(kwargs["request_cluster_id"], [2])


class _ControlPlane:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def unregister_pipeline(self, *, pipeline_id: str) -> None:
        self.events.append(f"unregister:{pipeline_id}")


class _Controller:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.fail_end_policy_sync = False
        self.fail_configure = False
        self.observations = {}

    async def close(self) -> None:
        self.events.append("close_controller")

    async def begin_policy_sync(self, *, expected_policy_version: int):
        self.events.append(f"begin_lease:{expected_policy_version}")
        return expected_policy_version

    async def end_policy_sync(self, lease: int) -> None:
        self.events.append(f"end_lease:{lease}")
        if self.fail_end_policy_sync:
            raise RuntimeError("lease end failed")

    async def configure_collection(self, context, **kwargs) -> None:
        del kwargs
        self.events.append(f"configure:{context.lifecycle_generation}")
        if self.fail_configure:
            raise RuntimeError("configure failed")

    async def get_rank_observation(self, rank: int):
        self.events.append(f"observe:{rank}")
        return self.observations[rank]


def _runtime(
    events: list[str],
    *,
    scheduler: _Scheduler | None = None,
    actor_train_devices: tuple[int, ...] = (2,),
    retain_training_overlap: bool = False,
) -> RegisteredRLixPipeline:
    return RegisteredRLixPipeline(
        control_plane=_ControlPlane(events),
        scheduler=scheduler or _Scheduler(events),
        controller=_Controller(events),
        pipeline_id="rlinf_123456789abc",
        ray_namespace="namespace",
        placement_plan=SimpleNamespace(
            actor_workers=(SimpleNamespace(rank=0),),
            rollout_workers=(SimpleNamespace(rank=0),),
            env_workers=(SimpleNamespace(rank=0),),
            actor_infer_devices=(0, 1, 3, 4),
            actor_infer_bundles=((0, (0, 1)), (1, (3, 4))),
            initialization_devices=(0, 1, 2),
            actor_train_devices=actor_train_devices,
            policy_sync_devices=(0, 2),
            evaluation_devices=(0, 1),
        ),
        retain_training_overlap=retain_training_overlap,
    )


def _receipt(cluster_name: str, devices: tuple[int, ...]) -> FixedStageResidencyReceipt:
    return FixedStageResidencyReceipt(
        cluster_name=cluster_name,
        verified_devices=devices,
        all_workers_offloaded=True,
    )


def test_fixed_initialization_acquires_exact_union_before_release() -> None:
    events: list[str] = []
    runtime = _runtime(events)

    with runtime.fixed_stage(
        cluster_name=INITIALIZATION_CLUSTER_NAME,
        priority=Priority.INITIALIZATION,
        global_step=4,
    ) as stage:
        events.append("worker_use")
        assert runtime.stage_state == RunnerStageState.FIXED_INITIALIZATION
        stage.complete(_receipt(INITIALIZATION_CLUSTER_NAME, (0, 1, 2)))

    assert events == [
        "request:rlinf_123456789abc_initialization:INITIALIZATION:4",
        "worker_use",
        "release:rlinf_123456789abc_initialization:4",
    ]
    assert runtime.stage_state == RunnerStageState.INACTIVE


def test_policy_sync_requires_generation_priority_and_exact_union() -> None:
    events: list[str] = []
    scheduler = _Scheduler(
        events,
        grants={"rlinf_123456789abc_policy_sync": [2, 0]},
    )
    runtime = _runtime(events, scheduler=scheduler)

    with runtime.fixed_stage(
        cluster_name=POLICY_SYNC_CLUSTER_NAME,
        priority=Priority.GENERATION,
        global_step=7,
    ) as stage:
        stage.complete(_receipt(POLICY_SYNC_CLUSTER_NAME, (0, 2)))

    with pytest.raises(ValueError, match="requires priority GENERATION"):
        with runtime.fixed_stage(
            cluster_name=POLICY_SYNC_CLUSTER_NAME,
            priority=Priority.INITIALIZATION,
            global_step=7,
        ):
            pass


def test_policy_sync_holds_lease_inside_fixed_allocation() -> None:
    events: list[str] = []
    runtime = _runtime(
        events,
        scheduler=_Scheduler(
            events,
            grants={"rlinf_123456789abc_policy_sync": [0, 2]},
        ),
    )
    statuses = [
        FixedWorkerResidency("actor", 0, False, False, False, 5),
        FixedWorkerResidency("rollout", 0, False, False, False, 5),
    ]

    with runtime.policy_sync_stage(expected_policy_version=5) as stage:
        events.append("collective")
        stage.complete(
            runtime.fixed_residency_receipt(
                cluster_name=POLICY_SYNC_CLUSTER_NAME,
                worker_residencies=statuses,
                policy_version=5,
            )
        )

    assert events == [
        "request:rlinf_123456789abc_policy_sync:GENERATION:5",
        "begin_lease:5",
        "collective",
        "end_lease:5",
        "release:rlinf_123456789abc_policy_sync:5",
    ]


def test_policy_sync_lease_failure_prevents_fixed_release() -> None:
    events: list[str] = []
    runtime = _runtime(
        events,
        scheduler=_Scheduler(
            events,
            grants={"rlinf_123456789abc_policy_sync": [0, 2]},
        ),
    )
    runtime.controller.fail_end_policy_sync = True
    statuses = [
        FixedWorkerResidency("actor", 0, False, False, False, 5),
        FixedWorkerResidency("rollout", 0, False, False, False, 5),
    ]

    with pytest.raises(RuntimeError, match="lease end failed"):
        with runtime.policy_sync_stage(expected_policy_version=5) as stage:
            stage.complete(
                runtime.fixed_residency_receipt(
                    cluster_name=POLICY_SYNC_CLUSTER_NAME,
                    worker_residencies=statuses,
                    policy_version=5,
                )
            )

    assert runtime.stage_state == RunnerStageState.FAILED_UNCERTAIN
    assert not any(event.startswith("release:") for event in events)


def test_missing_receipt_retains_ownership_and_fails_runtime() -> None:
    runtime = _runtime([])

    with pytest.raises(RuntimeError, match="without a residency receipt"):
        with runtime.fixed_stage(
            cluster_name=INITIALIZATION_CLUSTER_NAME,
            priority=Priority.INITIALIZATION,
            global_step=0,
        ):
            pass

    assert runtime.stage_state == RunnerStageState.FAILED_UNCERTAIN


def test_wrong_grant_fails_before_worker_body_and_retains_allocation() -> None:
    events: list[str] = []
    scheduler = _Scheduler(
        events,
        grants={"rlinf_123456789abc_actor_train": [0]},
    )
    runtime = _runtime(events, scheduler=scheduler)

    with pytest.raises(RuntimeError, match="granted devices"):
        with runtime.fixed_stage(
            cluster_name=ACTOR_TRAIN_CLUSTER_NAME,
            priority=Priority.ACTOR_TRAINING,
            global_step=1,
        ):
            events.append("must_not_run")

    assert "must_not_run" not in events
    assert runtime.stage_state == RunnerStageState.FAILED_UNCERTAIN
    assert not any(event.startswith("release:") for event in events)


def test_request_failure_returns_runtime_to_inactive() -> None:
    events: list[str] = []
    runtime = _runtime(events, scheduler=_Scheduler(events, fail_request=True))

    with pytest.raises(RuntimeError, match="request failed"):
        with runtime.fixed_stage(
            cluster_name=INITIALIZATION_CLUSTER_NAME,
            priority=Priority.INITIALIZATION,
            global_step=0,
        ):
            pass

    assert runtime.stage_state == RunnerStageState.INACTIVE


def test_primary_worker_error_survives_release_cleanup_error() -> None:
    events: list[str] = []
    runtime = _runtime(events, scheduler=_Scheduler(events, fail_release=True))

    with pytest.raises(ValueError, match="worker failed") as exc_info:
        with runtime.fixed_stage(
            cluster_name=INITIALIZATION_CLUSTER_NAME,
            priority=Priority.INITIALIZATION,
            global_step=0,
        ) as stage:
            stage.complete(_receipt(INITIALIZATION_CLUSTER_NAME, (0, 1, 2)))
            raise ValueError("worker failed")

    assert runtime.stage_state == RunnerStageState.FAILED_UNCERTAIN
    assert any("release failed" in note for note in exc_info.value.__notes__)


def test_close_requires_inactive_runtime_and_then_is_ordered() -> None:
    events: list[str] = []
    runtime = _runtime(events)
    runtime._stage_state = RunnerStageState.FAILED_UNCERTAIN

    with pytest.raises(RuntimeError, match="failed_uncertain"):
        runtime.close_sync()
    assert events == []

    runtime._stage_state = RunnerStageState.INACTIVE
    runtime.close_sync()
    runtime.close_sync()
    assert events == [
        "unregister:rlinf_123456789abc",
        "close_controller",
    ]
    assert runtime.stage_state == RunnerStageState.CLOSED


def test_elastic_batch_receipt_requires_complete_batch() -> None:
    receipt = ElasticBatchReceipt(
        lifecycle_generation=1,
        policy_version=3,
        contributing_dp_ranks=(0, 1),
        expected_trajectories=8,
        received_trajectories=8,
        transition_count=32,
    )
    assert receipt.policy_version == 3

    with pytest.raises(ValueError, match="must equal"):
        ElasticBatchReceipt(
            lifecycle_generation=1,
            policy_version=3,
            contributing_dp_ranks=(0, 1),
            expected_trajectories=8,
            received_trajectories=7,
            transition_count=28,
        )


def test_collection_publishes_cold_progress_and_starts_receiver_before_request() -> (
    None
):
    events: list[str] = []
    scheduler = _Scheduler(
        events,
        grants={"rlinf_123456789abc_actor_infer": [0, 1]},
    )
    runtime = _runtime(events, scheduler=scheduler)

    session = runtime.begin_collection(
        policy_version=8,
        assigned_trajectories_by_rank={0: 2, 1: 2},
        env_input_channel="env",
        rollout_request_channel="rollout",
        actor_channel="actor",
        actor_receiver_start=lambda: events.append("start_receiver") or "handle",
    )

    assert events == [
        "configure:1",
        "progress:0:[]",
        "start_receiver",
        "request:rlinf_123456789abc_actor_infer:GENERATION:8",
        "progress:0:[0]",
    ]
    assert session.active_dp_ranks == {0}
    assert session.actor_receiver_handle == "handle"
    assert runtime.stage_state == RunnerStageState.ELASTIC_COLLECTION
    assert scheduler.reports[0].metrics["resumable_dp_ranks"] == [0, 1]


def test_collection_rejects_partial_bundle_without_false_cleanup() -> None:
    events: list[str] = []
    scheduler = _Scheduler(
        events,
        grants={"rlinf_123456789abc_actor_infer": [0]},
    )
    runtime = _runtime(events, scheduler=scheduler)

    with pytest.raises(RuntimeError, match="splits canonical bundle"):
        runtime.begin_collection(
            policy_version=1,
            assigned_trajectories_by_rank={0: 2, 1: 2},
            env_input_channel="env",
            rollout_request_channel="rollout",
            actor_channel="actor",
            actor_receiver_start=lambda: "handle",
        )

    assert runtime.stage_state == RunnerStageState.FAILED_UNCERTAIN
    assert not any(event.startswith("clear_progress") for event in events)


def test_collection_reports_completion_before_exact_release() -> None:
    events: list[str] = []
    scheduler = _Scheduler(
        events,
        grants={"rlinf_123456789abc_actor_infer": [0, 1, 3, 4]},
    )
    runtime = _runtime(events, scheduler=scheduler)
    receiver = SimpleNamespace(wait=lambda: events.append("wait_receiver"))
    session = runtime.begin_collection(
        policy_version=2,
        assigned_trajectories_by_rank={0: 2, 1: 2},
        env_input_channel="env",
        rollout_request_channel="rollout",
        actor_channel="actor",
        actor_receiver_start=lambda: receiver,
    )
    events.clear()
    result = ElasticRunResult(ElasticRunOutcome.COMPLETED, None, None)
    for rank in (0, 1):
        runtime.controller.observations[rank] = SimpleNamespace(
            dp_rank=rank,
            failure=None,
            callback_applied_active=True,
            progress=ElasticRankProgress(
                dp_rank=rank,
                lifecycle_generation=1,
                state=ElasticRankState.COMPLETED,
                assigned_trajectories=2,
                completed_trajectories=2,
                snapshot_ready=False,
                failed=False,
            ),
            paired_results=(result, result),
        )

    assert runtime.monitor_collection_once(session)
    assert events == [
        "observe:0",
        "observe:1",
        "progress:4:[0, 1]",
        "await_release:(0, 1)",
        "progress:4:[]",
    ]
    assert session.released_dp_ranks == {0, 1}

    aggregate = runtime.seal_collection(
        session,
        actor_seal_start=lambda expected: SimpleNamespace(
            wait=lambda: [
                ElasticBatchReceipt(
                    lifecycle_generation=1,
                    policy_version=2,
                    contributing_dp_ranks=(0, 1),
                    expected_trajectories=expected,
                    received_trajectories=expected,
                    transition_count=12,
                )
            ]
        ),
    )
    assert aggregate.expected_trajectories == 4
    assert events[-2:] == [
        "wait_receiver",
        "clear_progress:rlinf_123456789abc",
    ]


def test_collection_retains_training_bundle_and_transitions_atomically() -> None:
    events: list[str] = []
    scheduler = _Scheduler(
        events,
        grants={
            "rlinf_123456789abc_actor_infer": [0, 1, 3, 4],
            "rlinf_123456789abc_actor_train": [0],
        },
    )
    runtime = _runtime(
        events,
        scheduler=scheduler,
        actor_train_devices=(0,),
        retain_training_overlap=True,
    )
    receiver = SimpleNamespace(wait=lambda: events.append("wait_receiver"))
    session = runtime.begin_collection(
        policy_version=2,
        assigned_trajectories_by_rank={0: 2, 1: 2},
        env_input_channel="env",
        rollout_request_channel="rollout",
        actor_channel="actor",
        actor_receiver_start=lambda: receiver,
    )
    result = ElasticRunResult(ElasticRunOutcome.COMPLETED, None, None)
    for rank in (0, 1):
        runtime.controller.observations[rank] = SimpleNamespace(
            dp_rank=rank,
            failure=None,
            callback_applied_active=True,
            progress=ElasticRankProgress(
                dp_rank=rank,
                lifecycle_generation=1,
                state=ElasticRankState.COMPLETED,
                assigned_trajectories=2,
                completed_trajectories=2,
                snapshot_ready=False,
                failed=False,
            ),
            paired_results=(result, result),
        )

    assert runtime.monitor_collection_once(session)
    assert session.active_dp_ranks == {0}
    assert session.released_dp_ranks == {1}
    assert scheduler.reports[-1].metrics["reserved_dp_ranks"] == [0]
    runtime.seal_collection(
        session,
        actor_seal_start=lambda expected: SimpleNamespace(
            wait=lambda: [ElasticBatchReceipt(1, 2, (0, 1), expected, expected, 12)]
        ),
    )
    assert runtime.stage_state == RunnerStageState.SEALED_COLLECTION

    with runtime.fixed_stage(
        cluster_name=ACTOR_TRAIN_CLUSTER_NAME,
        priority=Priority.ACTOR_TRAINING,
        global_step=2,
    ) as stage:
        stage.complete(_receipt(ACTOR_TRAIN_CLUSTER_NAME, (0,)))

    assert "transition:(0,):rlinf_123456789abc_actor_train:ACTOR_TRAINING" in events
    assert events.index(
        "transition:(0,):rlinf_123456789abc_actor_train:ACTOR_TRAINING"
    ) < events.index("clear_progress:rlinf_123456789abc")
    assert runtime.stage_state == RunnerStageState.INACTIVE
    assert runtime.stage_state == RunnerStageState.INACTIVE


def test_scheduler_driven_completed_rank_shrink_is_not_released_twice() -> None:
    events: list[str] = []
    scheduler = _Scheduler(
        events,
        grants={"rlinf_123456789abc_actor_infer": [0, 1, 3, 4]},
    )
    runtime = _runtime(events, scheduler=scheduler)
    session = runtime.begin_collection(
        policy_version=2,
        assigned_trajectories_by_rank={0: 2, 1: 2},
        env_input_channel="env",
        rollout_request_channel="rollout",
        actor_channel="actor",
        actor_receiver_start=lambda: SimpleNamespace(wait=lambda: None),
    )
    events.clear()
    completed = ElasticRunResult(ElasticRunOutcome.COMPLETED, None, None)
    runtime.controller.observations[0] = SimpleNamespace(
        dp_rank=0,
        failure=None,
        callback_applied_active=False,
        progress=ElasticRankProgress(
            dp_rank=0,
            lifecycle_generation=1,
            state=ElasticRankState.COMPLETED,
            assigned_trajectories=2,
            completed_trajectories=2,
            snapshot_ready=False,
            failed=False,
        ),
        paired_results=(completed, completed),
    )
    runtime.controller.observations[1] = SimpleNamespace(
        dp_rank=1,
        failure=None,
        callback_applied_active=True,
        progress=ElasticRankProgress(
            dp_rank=1,
            lifecycle_generation=1,
            state=ElasticRankState.ACTIVE,
            assigned_trajectories=2,
            completed_trajectories=1,
            snapshot_ready=False,
            failed=False,
        ),
        paired_results=None,
    )

    assert not runtime.monitor_collection_once(session)

    assert session.released_dp_ranks == {0}
    assert session.active_dp_ranks == {1}
    assert not any(event.startswith("await_release") for event in events)
    assert events[-1] == "progress:3:[1]"


def test_collection_monitor_timeout_fails_closed(monkeypatch) -> None:
    events: list[str] = []
    scheduler = _Scheduler(
        events,
        grants={"rlinf_123456789abc_actor_infer": [0, 1]},
    )
    runtime = _runtime(events, scheduler=scheduler)
    session = runtime.begin_collection(
        policy_version=2,
        assigned_trajectories_by_rank={0: 2, 1: 2},
        env_input_channel="env",
        rollout_request_channel="rollout",
        actor_channel="actor",
        actor_receiver_start=lambda: SimpleNamespace(wait=lambda: None),
    )
    monkeypatch.setattr(
        RegisteredRLixPipeline, "monitor_collection_once", lambda self, active: False
    )
    monotonic_values = iter((0.0, 1.0))
    monkeypatch.setattr(
        "rlinf.scheduler.rlix.runtime.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr("rlinf.scheduler.rlix.runtime.time.sleep", lambda _: None)

    with pytest.raises(TimeoutError, match="timed out"):
        runtime.wait_for_collection(session, poll_interval_s=0.01, timeout_s=0.5)

    assert runtime.stage_state == RunnerStageState.FAILED_UNCERTAIN
    assert session.generation_owned
    assert not any(event.startswith("clear_progress") for event in events)


def test_fixed_receipt_requires_exact_safe_worker_coverage() -> None:
    runtime = _runtime([])
    statuses = [
        FixedWorkerResidency("actor", 0, False, False, False),
        FixedWorkerResidency("rollout", 0, False, False, False),
        FixedWorkerResidency("environment", 0, False, False, False),
    ]

    receipt = runtime.fixed_residency_receipt(
        cluster_name=INITIALIZATION_CLUSTER_NAME,
        worker_residencies=statuses,
    )
    assert receipt.verified_devices == (0, 1, 2)

    with pytest.raises(RuntimeError, match="remains GPU-resident"):
        runtime.fixed_residency_receipt(
            cluster_name=INITIALIZATION_CLUSTER_NAME,
            worker_residencies=[
                *statuses[:2],
                FixedWorkerResidency("environment", 0, True, False, False),
            ],
        )


def test_actor_training_receipt_requires_published_successor_version() -> None:
    runtime = _runtime([])
    published = [FixedWorkerResidency("actor", 0, False, False, False, 10)]

    receipt = runtime.fixed_residency_receipt(
        cluster_name=ACTOR_TRAIN_CLUSTER_NAME,
        worker_residencies=published,
        policy_version=10,
    )
    assert receipt.policy_version == 10

    with pytest.raises(ValueError, match="produced policy version 9, expected 10"):
        runtime.fixed_residency_receipt(
            cluster_name=ACTOR_TRAIN_CLUSTER_NAME,
            worker_residencies=[
                FixedWorkerResidency("actor", 0, False, False, False, 9)
            ],
            policy_version=10,
        )

    with pytest.raises(ValueError, match="requires a policy version"):
        runtime.fixed_residency_receipt(
            cluster_name=ACTOR_TRAIN_CLUSTER_NAME,
            worker_residencies=published,
        )
