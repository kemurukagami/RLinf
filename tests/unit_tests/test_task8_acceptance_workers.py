"""Transparency tests for Task 8 acceptance-only worker instrumentation."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_workers():
    embodied = Path(__file__).resolve().parents[1] / "e2e_tests" / "embodied"
    sys.path.insert(0, str(embodied))
    try:
        module_path = embodied / "task8_acceptance_workers.py"
        spec = importlib.util.spec_from_file_location(
            "task8_acceptance_workers", module_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"cannot load Task 8 acceptance workers from {module_path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(embodied))


_workers = _load_workers()
RecordingEnvWorkerMixin = _workers.RecordingEnvWorkerMixin
RecordingEmbodiedFSDPActorMixin = _workers.RecordingEmbodiedFSDPActorMixin
RecordingEmbodiedRunnerMixin = _workers.RecordingEmbodiedRunnerMixin
RecordingMultiStepRolloutWorkerMixin = _workers.RecordingMultiStepRolloutWorkerMixin


class _FakeEnvWorker:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def env_interact_step(self, chunk_actions, stage_id):
        self.calls.append(("chunk", chunk_actions.clone(), stage_id))
        return {"obs": chunk_actions + 1, "stage_id": stage_id}

    def offload_elastic_environment(self, token):
        self.calls.append(("offload", token))
        return {"token": token, "resident": False}

    def snapshot_rollout_stage(self):
        self.calls.append(("snapshot",))
        return {"cursor": torch.tensor([3, 4])}

    def restore_rollout_stage(self, state, **kwargs):
        self.calls.append(("restore", state, kwargs))
        return None

    def prepare_elastic_resume(self, token):
        self.calls.append(("resume", token))
        self.restore_rollout_stage({"cursor": 3}, policy_version=7)
        return {"token": token, "resident": True}

    async def request_elastic_drain(self, request):
        self.calls.append(("drain", request))
        return {"state": "drain_requested"}

    async def _send_elastic_barrier(self, rollout_channel, token):
        self.calls.append(("barrier", rollout_channel, token))

    async def _send_elastic_observation(self, rollout_channel, env_output):
        self.calls.append(("observation", rollout_channel, env_output))


class _RecordingFakeEnv(RecordingEnvWorkerMixin, _FakeEnvWorker):
    pass


class _FailingBarrierEnvWorker(_FakeEnvWorker):
    async def _send_elastic_barrier(self, rollout_channel, token):
        self.calls.append(("barrier", rollout_channel, token))
        raise ValueError("injected barrier routing failure")


class _RecordingFailingBarrierEnv(RecordingEnvWorkerMixin, _FailingBarrierEnvWorker):
    pass


class _FakeRolloutWorker:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def _build_train_rollout_result(self, env_output, **kwargs):
        self.calls.append(("policy", env_output, kwargs))
        return {"actions": env_output["obs"] * 2}

    def offload_elastic_rollout(self, token):
        self.calls.append(("offload", token))
        return {"token": token, "resident": False}

    def prepare_elastic_resume(self, token):
        self.calls.append(("resume", token))
        return {"token": token, "resident": True}

    async def request_elastic_drain(self, request):
        self.calls.append(("drain", request))
        return {"state": "drain_requested"}

    async def generate_until_pause_or_complete(self, outcome):
        self.calls.append(("generate", outcome))
        return _RunResult(_Outcome(outcome))


class _RecordingFakeRollout(RecordingMultiStepRolloutWorkerMixin, _FakeRolloutWorker):
    pass


class _Outcome(str, Enum):
    PAUSE_READY = "pause_ready"
    COMPLETED = "completed"


@dataclass(frozen=True)
class _RunResult:
    outcome: _Outcome


class _FakeActor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.cfg = SimpleNamespace(algorithm=SimpleNamespace(adv_type="grpo"))
        self.rollout_batch = {
            "versions": torch.tensor([[3.0, 3.0]]),
            "rewards": torch.tensor([[1.0, 2.0]]),
        }
        self._rlix_received_transition_ids = ("t0", "t1")

    def seal_rlix_batch(self, **kwargs):
        self.calls.append(("seal", kwargs))
        return {"policy_version": kwargs["policy_version"]}

    def compute_advantages_and_returns(self):
        self.calls.append(("advantages",))
        return {"reward_mean": 1.5}

    def run_rlix_training(self, batch_receipt):
        self.calls.append(("train", batch_receipt))
        return {"loss": 0.25}


class _RecordingFakeActor(RecordingEmbodiedFSDPActorMixin, _FakeActor):
    pass


class _FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.global_step = 3
        self.cfg = SimpleNamespace(algorithm=SimpleNamespace(adv_type="grpo"))
        self.rlix_runtime = None

    def update_rollout_weights(self):
        self.calls.append(("sync", self.global_step))
        hook = getattr(self, "_on_rlix_policy_sync_stage_acquired", None)
        if hook is not None:
            hook()
        return "synced"

    def _collect_rlix_rollouts(self):
        self.calls.append(("collect", self.global_step))
        return {"policy_version": self.global_step, "received_trajectories": 8}

    def _train_rlix_batch(self, batch_receipt):
        self.calls.append(("train", batch_receipt))
        self.global_step += 1
        return ("rollout_metrics", "training_metrics", "handle")


class _RecordingFakeRunner(RecordingEmbodiedRunnerMixin, _FakeRunner):
    pass


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_recording_env_worker_preserves_outputs_calls_and_rng() -> None:
    plain = _FakeEnvWorker()
    recorded = _RecordingFakeEnv()
    events = []
    recorded.configure_acceptance_observer(events.append)
    actions = torch.tensor([[1.0, 2.0]])

    torch.manual_seed(41)
    expected = plain.env_interact_step(actions.clone(), 2)
    expected_next_rng = torch.rand(3)
    torch.manual_seed(41)
    actual = recorded.env_interact_step(actions.clone(), 2)
    actual_next_rng = torch.rand(3)
    _assert_nested_equal(expected, actual)
    _assert_nested_equal(plain.calls, recorded.calls)
    assert torch.equal(expected_next_rng, actual_next_rng)
    assert [event.event for event in events] == ["chunk_started", "chunk_committed"]
    assert events[-1].details == {"stage_id": 2}

    _assert_nested_equal(
        recorded.snapshot_rollout_stage(), plain.snapshot_rollout_stage()
    )
    assert recorded.offload_elastic_environment(
        "token"
    ) == plain.offload_elastic_environment("token")
    assert recorded.prepare_elastic_resume("token") == plain.prepare_elastic_resume(
        "token"
    )
    _assert_nested_equal(plain.calls, recorded.calls)
    assert [event.event for event in events[2:]] == [
        "snapshot_started",
        "snapshot_completed",
        "environment_offload_started",
        "environment_offload_verified",
        "environment_onload_started",
        "restore_validated",
        "restore_committed",
        "environment_onload_verified",
    ]
    assert events[3].details["encoded_tensor_bytes"] == 16

    asyncio.run(recorded.request_elastic_drain("request"))
    asyncio.run(recorded._send_elastic_barrier("channel", "token"))
    asyncio.run(recorded._send_elastic_observation("channel", {"transition_id": "t1"}))
    assert [event.event for event in events[-4:]] == [
        "drain_requested",
        "drain_observed",
        "bootstrap_dispatched",
        "resumed_bootstrap_dispatched",
    ]


def test_recording_rollout_worker_preserves_outputs_and_call_order() -> None:
    plain = _FakeRolloutWorker()
    recorded = _RecordingFakeRollout()
    events = []
    recorded.configure_acceptance_observer(events.append)
    observation = {"obs": torch.tensor([3.0])}

    expected = plain._build_train_rollout_result(observation, final_bootstrap=False)
    actual = recorded._build_train_rollout_result(observation, final_bootstrap=False)
    _assert_nested_equal(expected, actual)
    assert recorded.offload_elastic_rollout("token") == plain.offload_elastic_rollout(
        "token"
    )
    assert recorded.prepare_elastic_resume("token") == plain.prepare_elastic_resume(
        "token"
    )
    _assert_nested_equal(plain.calls, recorded.calls)
    assert [event.event for event in events] == [
        "policy_request_started",
        "policy_request_completed",
        "rollout_offload_started",
        "rollout_offload_verified",
        "rollout_onload_started",
        "rollout_onload_verified",
    ]

    asyncio.run(recorded.request_elastic_drain("request"))
    paused = asyncio.run(recorded.generate_until_pause_or_complete("pause_ready"))
    assert paused.outcome.value == "pause_ready"
    assert [event.event for event in events[-2:]] == [
        "drain_requested",
        "barrier_consumed",
    ]


def test_barrier_phase_diagnostics_record_success_and_failure() -> None:
    successful = _RecordingFakeEnv()
    success_events = []
    successful.configure_acceptance_observer(success_events.append)
    successful.configure_task8_phase_diagnostics(True)

    asyncio.run(successful._send_elastic_barrier("channel", "token"))

    assert [event.event for event in success_events] == [
        "barrier_send_started",
        "barrier_send_completed",
        "drain_observed",
    ]

    failing = _RecordingFailingBarrierEnv()
    failure_events = []
    failing.configure_acceptance_observer(failure_events.append)
    failing.configure_task8_phase_diagnostics(True)

    with pytest.raises(ValueError, match="injected barrier routing failure"):
        asyncio.run(failing._send_elastic_barrier("channel", "token"))

    assert [event.event for event in failure_events] == [
        "barrier_send_started",
        "barrier_send_failed",
    ]
    assert failure_events[-1].details["error_type"] == "ValueError"
    assert failure_events[-1].details["error"] == "injected barrier routing failure"


def test_observer_failure_propagates_before_worker_call() -> None:
    worker = _RecordingFakeEnv()

    def fail(_event) -> None:
        raise RuntimeError("evidence sink unavailable")

    worker.configure_acceptance_observer(fail)
    with pytest.raises(RuntimeError, match="evidence sink unavailable"):
        worker.env_interact_step(torch.tensor([1]), 0)
    assert worker.calls == []


def test_recording_runner_preserves_grpo_stage_outputs_and_policy_advance() -> None:
    plain = _FakeRunner()
    recorded = _RecordingFakeRunner()
    events = []
    recorded.configure_acceptance_observer(events.append)

    assert recorded.update_rollout_weights() == plain.update_rollout_weights()
    plain_receipt = plain._collect_rlix_rollouts()
    recorded_receipt = recorded._collect_rlix_rollouts()
    assert recorded_receipt == plain_receipt
    assert recorded._train_rlix_batch(recorded_receipt) == plain._train_rlix_batch(
        plain_receipt
    )
    assert recorded.calls == plain.calls
    assert recorded.global_step == plain.global_step == 4
    assert [event.event for event in events] == [
        "stage_requested",
        "stage_acquired",
        "policy_synchronized",
        "stage_released",
        "reward_started",
        "generation_requested",
        "reward_completed",
        "batch_sealed",
    ]


def test_recording_actor_captures_sealed_batch_advantages_and_training() -> None:
    plain = _FakeActor()
    recorded = _RecordingFakeActor()
    events = []
    recorded.configure_acceptance_observer(events.append)
    seal_kwargs = {
        "lifecycle_generation": 2,
        "policy_version": 3,
        "contributing_dp_ranks": (0, 1),
        "expected_trajectories": 2,
    }

    plain_receipt = plain.seal_rlix_batch(**seal_kwargs)
    recorded_receipt = recorded.seal_rlix_batch(**seal_kwargs)
    assert recorded_receipt == plain_receipt
    assert (
        recorded.compute_advantages_and_returns()
        == plain.compute_advantages_and_returns()
    )
    assert recorded.run_rlix_training(recorded_receipt) == plain.run_rlix_training(
        plain_receipt
    )
    assert recorded.calls == plain.calls
    assert [event.event for event in events] == [
        "batch_sealed",
        "advantages_computed",
        "training_started",
        "training_completed",
    ]
    assert events[0].details["batch_tensor_bytes"] == 16
    assert events[0].details["batch"]["versions"]["kind"] == "tensor"
