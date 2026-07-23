"""Task 8 composed multipipeline tests at production scheduler/runtime boundaries."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Coroutine

import pytest
from rlix_core.protocol.types import GENERATION_CLUSTER_NAME
from rlix_core.scheduler.scheduler import SchedulerImpl

from rlinf.data.embodied_io_struct import RolloutTransitionIdentity
from rlinf.scheduler.rlix.coordinator import RLixResizeCoordinator
from rlinf.scheduler.rlix.protocol import ElasticBatchReceipt
from rlinf.scheduler.rlix.runtime import RegisteredRLixPipeline
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


class _LoopThread:
    """Own one persistent event loop, matching a scheduler/Ray actor lifetime."""

    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=2):
            raise RuntimeError("persistent test loop did not start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self.loop = loop
        asyncio.set_event_loop(loop)
        self.ready.set()
        while not self.stopping.is_set():
            loop.run_until_complete(asyncio.sleep(0.001))

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> Future[Any]:
        """Submit one coroutine to the persistent loop."""
        assert self.loop is not None
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def call(self, callback, *args) -> None:
        """Schedule a thread-safe synchronous callback."""
        assert self.loop is not None
        self.loop.call_soon_threadsafe(callback, *args)

    def close(self) -> None:
        """Stop and join the loop thread."""
        assert self.loop is not None

        async def cancel_pending() -> None:
            current = asyncio.current_task()
            pending = [
                task
                for task in asyncio.all_tasks()
                if task is not current
                and not task.done()
                and task.get_coro().__qualname__ != "sleep"
            ]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        self.submit(cancel_pending()).result(timeout=2)
        self.stopping.set()
        self.thread.join(timeout=2)
        self.loop.close()


async def _bridge(future: Future[Any]) -> Any:
    # Some restricted CI kernels do not reliably wake an asyncio self-pipe
    # across threads. Polling preserves the actor-shaped boundary without
    # introducing timing into the production transaction itself.
    while not future.done():
        await asyncio.sleep(0.001)
    return future.result()


class _CoordinatorProxy:
    """Expose a production coordinator to the synchronous runtime."""

    def __init__(self, loop: _LoopThread, coordinator: RLixResizeCoordinator):
        self._loop = loop
        self._coordinator = coordinator

    def configure_collection(self, context, **kwargs):
        return _bridge(
            self._loop.submit(self._coordinator.configure_collection(context, **kwargs))
        )

    def get_rank_observation(self, rank: int):
        return _bridge(self._loop.submit(self._coordinator.get_rank_observation(rank)))

    def begin_policy_sync(self, *, expected_policy_version: int):
        return _bridge(
            self._loop.submit(
                self._coordinator.begin_policy_sync(
                    expected_policy_version=expected_policy_version
                )
            )
        )

    def end_policy_sync(self, lease):
        return _bridge(self._loop.submit(self._coordinator.end_policy_sync(lease)))

    def close(self):
        return _bridge(self._loop.submit(self._coordinator.close()))


class _ResizeRemote:
    def __init__(self, coordinator: RLixResizeCoordinator) -> None:
        self._coordinator = coordinator

    def remote(self, **kwargs):
        return self._coordinator.resize_infer(**kwargs)


class _CoordinatorHandle:
    def __init__(self, coordinator: RLixResizeCoordinator) -> None:
        self.resize_infer = _ResizeRemote(coordinator)


class _DrivingScheduler:
    """Drive one production scheduling cycle for each blocking runtime call."""

    def __init__(self, loop: _LoopThread, scheduler: SchedulerImpl) -> None:
        self._loop = loop
        self._scheduler = scheduler

    async def _request_and_cycle(self, kwargs: dict[str, Any]) -> list[int]:
        request = asyncio.create_task(self._scheduler.request_gpus(**kwargs))
        await asyncio.sleep(0)
        await self._scheduler.scheduling_cycle()
        return await request

    def request_gpus(self, **kwargs):
        return _bridge(self._loop.submit(self._request_and_cycle(kwargs)))

    async def _release_and_cycle(self, kwargs: dict[str, Any]) -> None:
        release = asyncio.create_task(self._scheduler.await_release_dp_ranks(**kwargs))
        await asyncio.sleep(0)
        await self._scheduler.scheduling_cycle()
        await release

    def await_release_dp_ranks(self, **kwargs):
        return _bridge(self._loop.submit(self._release_and_cycle(kwargs)))

    def report_progress(self, *, report):
        return _bridge(self._loop.submit(self._scheduler.report_progress(report)))

    def clear_progress(self, *, pipeline_id: str):
        return _bridge(
            self._loop.submit(self._scheduler.clear_progress(pipeline_id=pipeline_id))
        )

    def notify_release_gpus(self, **kwargs):
        return _bridge(self._loop.submit(self._scheduler.notify_release_gpus(**kwargs)))


class _ControlPlaneProxy:
    def __init__(self, loop: _LoopThread, scheduler: SchedulerImpl) -> None:
        self._loop = loop
        self._scheduler = scheduler
        self.unregistered: list[str] = []

    def unregister_pipeline(self, *, pipeline_id: str) -> None:
        self._loop.submit(
            self._scheduler.unregister_pipeline(pipeline_id=pipeline_id)
        ).result(timeout=2)
        self.unregistered.append(pipeline_id)


class _Worker:
    """Deterministic worker protocol double preserving production transitions."""

    def __init__(self, rank: int, *, token_suffix: str = "") -> None:
        self.rank = rank
        self.token_suffix = token_suffix
        self.state = ElasticRankState.INACTIVE_COLD
        self.lifecycle: int | None = None
        self.policy_version: int | None = None
        self.resident = False
        self.failure: str | None = None
        self.transition: RolloutTransitionIdentity | None = None
        self.drain_request = None
        self.drain_event = asyncio.Event()
        self.complete_event = asyncio.Event()
        self.resume_count = 0
        self.fail_pause_offload = False

    def _status(self) -> ElasticRankStatus:
        return ElasticRankStatus(
            state=self.state,
            worker_rank=self.rank,
            lifecycle_generation=self.lifecycle,
            policy_version=self.policy_version,
            expected_transition_id=self.transition,
            drain_request_id=(
                None if self.drain_request is None else self.drain_request.request_id
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
        return self._status()

    def prepare_elastic_resume(self, token: SafePointToken) -> ResidencyReceipt:
        self.state = ElasticRankState.EXPANDING
        self.resident = True
        self.drain_request = None
        self.drain_event = asyncio.Event()
        self.complete_event = asyncio.Event()
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
        self.state = ElasticRankState.ACTIVE
        drain = asyncio.create_task(self.drain_event.wait())
        complete = asyncio.create_task(self.complete_event.wait())
        done, pending = await asyncio.wait(
            (drain, complete), return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if complete in done:
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

    def request_elastic_drain(self, request) -> ElasticRankStatus:
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


class _Receiver:
    def __init__(self) -> None:
        self.waited = False

    def wait(self) -> None:
        self.waited = True


class _TestScheduler(SchedulerImpl):
    """Scheduler boundary that suppresses cluster shutdown in injected failures."""

    async def _fail_fast_shutdown(self, *, reason: str) -> None:
        del reason


class _Harness:
    def __init__(
        self,
        *,
        mismatched_a_token: bool = False,
        fail_a_environment_offload: bool = False,
    ) -> None:
        self.loop = _LoopThread()
        self.scheduler = _TestScheduler()
        self.scheduler._topology_ready.set()
        self.scheduler._num_gpus = 4
        self.scheduler._required_gpus_per_node = 4
        self.scheduler._state.idle_gpus = {0, 1, 2, 3}
        self.pipeline_a = "rlinf_aaaaaaaaaaaa"
        self.pipeline_b = "rlinf_bbbbbbbbbbbb"
        self.env_a = {rank: _Worker(rank) for rank in range(2)}
        self.env_a[0].fail_pause_offload = fail_a_environment_offload
        self.rollout_a = {
            rank: _Worker(
                rank,
                token_suffix="-mismatch" if mismatched_a_token and rank == 0 else "",
            )
            for rank in range(2)
        }
        self.env_b = {rank: _Worker(rank) for rank in range(2)}
        self.rollout_b = {rank: _Worker(rank) for rank in range(2)}
        self.coordinator_a = RLixResizeCoordinator(
            pipeline_id=self.pipeline_a,
            env_workers=self.env_a,
            rollout_workers=self.rollout_a,
            operation_timeout_s=1,
            activation_poll_interval_s=0.001,
        )
        self.coordinator_b = RLixResizeCoordinator(
            pipeline_id=self.pipeline_b,
            env_workers=self.env_b,
            rollout_workers=self.rollout_b,
            operation_timeout_s=1,
            activation_poll_interval_s=0.001,
        )
        self._register()
        scheduler_proxy = _DrivingScheduler(self.loop, self.scheduler)
        control_plane = _ControlPlaneProxy(self.loop, self.scheduler)
        self.runtime_a = self._runtime(
            pipeline_id=self.pipeline_a,
            scheduler=scheduler_proxy,
            control_plane=control_plane,
            coordinator=self.coordinator_a,
        )
        self.runtime_b = self._runtime(
            pipeline_id=self.pipeline_b,
            scheduler=scheduler_proxy,
            control_plane=control_plane,
            coordinator=self.coordinator_b,
        )

    def _register(self) -> None:
        async def register() -> None:
            for pipeline_id, coordinator in (
                (self.pipeline_a, self.coordinator_a),
                (self.pipeline_b, self.coordinator_b),
            ):
                await self.scheduler.register_pipeline_topology(
                    pipeline_id=pipeline_id,
                    ray_namespace=f"namespace_{pipeline_id}",
                    cluster_tp_configs={GENERATION_CLUSTER_NAME: 1},
                    cluster_device_mappings={GENERATION_CLUSTER_NAME: [0, 1, 2, 3]},
                    cluster_dp_device_mappings={
                        GENERATION_CLUSTER_NAME: {0: [0, 2], 1: [1, 3]}
                    },
                )
                self.scheduler._state.pipeline_registry[pipeline_id]["admitted"] = True
                self.scheduler._coordinator_handle_cache[pipeline_id] = (
                    f"namespace_{pipeline_id}",
                    _CoordinatorHandle(coordinator),
                )

        self.loop.submit(register()).result(timeout=2)

    def _runtime(
        self,
        *,
        pipeline_id: str,
        scheduler: _DrivingScheduler,
        control_plane: _ControlPlaneProxy,
        coordinator: RLixResizeCoordinator,
    ) -> RegisteredRLixPipeline:
        workers = tuple(SimpleNamespace(rank=rank) for rank in range(2))
        return RegisteredRLixPipeline(
            control_plane=control_plane,
            scheduler=scheduler,
            controller=_CoordinatorProxy(self.loop, coordinator),
            pipeline_id=pipeline_id,
            ray_namespace=f"namespace_{pipeline_id}",
            placement_plan=SimpleNamespace(
                actor_workers=(SimpleNamespace(rank=0),),
                rollout_workers=workers,
                env_workers=workers,
                actor_infer_devices=(0, 1, 2, 3),
                actor_infer_bundles=((0, (0, 2)), (1, (1, 3))),
                initialization_devices=(0, 1, 2, 3),
                actor_train_devices=(0,),
                policy_sync_devices=(0, 1, 2, 3),
                evaluation_devices=(0, 1, 2, 3),
            ),
            operation_timeout_s=1,
        )

    def complete(
        self, workers: tuple[dict[int, _Worker], dict[int, _Worker]], rank: int
    ) -> None:
        for group in workers:
            self.loop.call(group[rank].complete_event.set)

    def cycle(self) -> None:
        self.loop.submit(self.scheduler.scheduling_cycle()).result(timeout=2)

    def close(self) -> None:
        self.loop.close()


def _begin(runtime: RegisteredRLixPipeline, receiver: _Receiver):
    return runtime.begin_collection(
        policy_version=3,
        assigned_trajectories_by_rank={0: 2, 1: 2},
        env_input_channel=object(),
        rollout_request_channel=object(),
        actor_channel=object(),
        actor_receiver_start=lambda: receiver,
    )


def _seal(runtime: RegisteredRLixPipeline, session, receiver: _Receiver):
    receipt = runtime.seal_collection(
        session,
        actor_seal_start=lambda expected: [
            ElasticBatchReceipt(
                lifecycle_generation=session.context.lifecycle_generation,
                policy_version=session.context.policy_version,
                contributing_dp_ranks=(0, 1),
                expected_trajectories=expected,
                received_trajectories=expected,
                transition_count=8,
            )
        ],
    )
    assert receiver.waited
    return receipt


def test_two_registered_runtimes_transfer_complete_and_seal_batches() -> None:
    harness = _Harness()
    receiver_a = _Receiver()
    receiver_b = _Receiver()
    try:
        session_a = _begin(harness.runtime_a, receiver_a)
        assert session_a.active_dp_ranks == {0, 1}

        session_b = _begin(harness.runtime_b, receiver_b)
        assert session_b.active_dp_ranks == {0}
        assert harness.env_a[0].state is ElasticRankState.PAUSED
        assert harness.rollout_a[0].state is ElasticRankState.PAUSED
        assert harness.env_a[1].state is ElasticRankState.ACTIVE

        harness.complete((harness.env_a, harness.rollout_a), 1)
        assert not harness.runtime_a.monitor_collection_once(session_a)

        harness.complete((harness.env_b, harness.rollout_b), 0)
        assert not harness.runtime_b.monitor_collection_once(session_b)
        harness.cycle()
        assert harness.env_a[0].resume_count == 1
        assert harness.rollout_a[0].resume_count == 1
        assert harness.env_a[0].state is ElasticRankState.ACTIVE
        assert harness.env_b[1].state is ElasticRankState.ACTIVE

        harness.complete((harness.env_a, harness.rollout_a), 0)
        harness.complete((harness.env_b, harness.rollout_b), 1)
        assert harness.runtime_a.monitor_collection_once(session_a)
        assert harness.runtime_b.monitor_collection_once(session_b)

        receipt_a = _seal(harness.runtime_a, session_a, receiver_a)
        receipt_b = _seal(harness.runtime_b, session_b, receiver_b)
        assert receipt_a.received_trajectories == 4
        assert receipt_b.received_trajectories == 4
        assert harness.scheduler._state.idle_gpus == {0, 1, 2, 3}

        harness.runtime_a.close_sync()
        harness.runtime_b.close_sync()
    finally:
        harness.close()


@pytest.mark.parametrize(
    ("harness_kwargs", "message"),
    [
        ({"mismatched_a_token": True}, "peer pause tokens do not match"),
        (
            {"fail_a_environment_offload": True},
            "injected pause offload failure",
        ),
    ],
)
def test_callback_failure_prevents_scheduler_transfer_and_batch_seal(
    harness_kwargs, message
) -> None:
    harness = _Harness(**harness_kwargs)
    receiver_a = _Receiver()
    receiver_b = _Receiver()
    try:
        session_a = _begin(harness.runtime_a, receiver_a)
        with pytest.raises(RuntimeError, match=message):
            _begin(harness.runtime_b, receiver_b)

        cluster_a = f"{harness.pipeline_a}_{GENERATION_CLUSTER_NAME}"
        cluster_b = f"{harness.pipeline_b}_{GENERATION_CLUSTER_NAME}"
        assert harness.scheduler._state.active_allocations[
            cluster_a
        ].active_dp_ranks == {0, 1}
        assert cluster_b not in harness.scheduler._state.active_allocations
        assert session_a.batch_receipt is None
        assert harness.runtime_b.stage_state.value == "inactive"
    finally:
        harness.close()


def _wait_for_json(
    path: Path, processes: tuple[subprocess.Popen[str], ...], timeout_s: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        failures = [process for process in processes if process.poll() is not None]
        if failures:
            process = failures[0]
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"Task 8 subprocess exited with {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.02)
    return json.loads(path.read_text(encoding="utf-8"))


def _signal(path: Path) -> None:
    path.touch()


@pytest.mark.skipif(
    os.environ.get("RLINF_RUN_LOCAL_RAY_TEST") != "1",
    reason="set RLINF_RUN_LOCAL_RAY_TEST=1 to exercise subprocess Ray isolation",
)
def test_two_subprocess_clients_share_detached_core_and_isolate_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove two OS drivers share core singletons without RLinf name collisions."""
    import ray
    from ray._private import ray_constants
    from rlix_core.client import connect
    from rlix_core.protocol.types import RLIX_NAMESPACE

    # Ray's uv ancestry probe can fail in PID-restricted containers before it
    # determines that this explicitly invoked interpreter is not `uv run`.
    monkeypatch.setattr(ray_constants, "RAY_ENABLE_UV_RUN_RUNTIME_ENV", False)
    monkeypatch.setenv("RLIX_CORE_REQUIRED_GPUS_PER_NODE", "2")
    ray_context = ray.init(
        num_cpus=4,
        num_gpus=2,
        namespace=RLIX_NAMESPACE,
        include_dashboard=True,
        log_to_driver=False,
    )
    control_plane = connect(create_if_missing=True)
    expected_control_plane_id = control_plane._actor_id.hex()
    expected_scheduler = ray.get(control_plane.get_scheduler.remote())
    expected_scheduler_id = expected_scheduler._actor_id.hex()
    address = ray_context.address_info["address"]
    run_id = uuid.uuid4().hex[:12]
    helper = (
        Path(__file__).parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_local_ray_client.py"
    )
    pythonpath = os.pathsep.join(
        (
            str(Path(__file__).parents[2]),
            str(Path(__file__).parents[3] / "rlix-core" / "src"),
            os.environ.get("PYTHONPATH", ""),
        )
    )
    environment = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
    }
    paths = {
        name: tmp_path / name
        for name in (
            "a_ready",
            "b_ready",
            "a_disconnect",
            "b_disconnect",
            "b_recheck",
            "b_rechecked",
        )
    }

    def launch(role: str) -> subprocess.Popen[str]:
        command = [
            sys.executable,
            str(helper),
            "--address",
            address,
            "--run-id",
            run_id,
            "--role",
            role,
            "--ready",
            str(paths[f"{role}_ready"]),
            "--disconnect",
            str(paths[f"{role}_disconnect"]),
        ]
        if role == "b":
            command.extend(
                (
                    "--recheck",
                    str(paths["b_recheck"]),
                    "--rechecked",
                    str(paths["b_rechecked"]),
                )
            )
        return subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    processes: tuple[subprocess.Popen[str], ...] = ()
    pipeline_ids: list[str] = []
    try:
        processes = (launch("a"), launch("b"))
        result_a = _wait_for_json(paths["a_ready"], processes, 45.0)
        result_b = _wait_for_json(paths["b_ready"], processes, 45.0)
        pipeline_ids.extend((result_a["pipeline_id"], result_b["pipeline_id"]))

        assert result_a["pid"] != result_b["pid"]
        assert (
            result_a["control_plane_actor_id"]
            == result_b["control_plane_actor_id"]
            == expected_control_plane_id
        )
        assert (
            result_a["scheduler_actor_id"]
            == result_b["scheduler_actor_id"]
            == expected_scheduler_id
        )
        assert result_a["pipeline_id"] != result_b["pipeline_id"]
        assert result_a["pipeline_namespace"] != result_b["pipeline_namespace"]
        assert result_a["candidate_mapping"] == result_b["candidate_mapping"]
        assert result_a["candidate_dp_mapping"] == result_b["candidate_dp_mapping"]
        assert set(result_a["role_names"].values()).isdisjoint(
            result_b["role_names"].values()
        )

        _signal(paths["a_disconnect"])
        stdout_a, stderr_a = processes[0].communicate(timeout=15)
        assert processes[0].returncode == 0, f"stdout:\n{stdout_a}\nstderr:\n{stderr_a}"

        _signal(paths["b_recheck"])
        rechecked_b = _wait_for_json(paths["b_rechecked"], (processes[1],), 15.0)
        assert rechecked_b == {
            "control_plane_actor_id": expected_control_plane_id,
            "pipeline_id": result_b["pipeline_id"],
            "scheduler_actor_id": expected_scheduler_id,
        }
        _signal(paths["b_disconnect"])
        stdout_b, stderr_b = processes[1].communicate(timeout=15)
        assert processes[1].returncode == 0, f"stdout:\n{stdout_b}\nstderr:\n{stderr_b}"
    finally:
        for path in (paths["a_disconnect"], paths["b_recheck"], paths["b_disconnect"]):
            _signal(path)
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        for pipeline_id in pipeline_ids:
            ray.get(control_plane.unregister_pipeline.remote(pipeline_id=pipeline_id))
        ray.shutdown()
