"""CPU fakes for Task 7 synchronous embodied-runner stage ordering."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.embodied_io_struct import RolloutTransitionIdentity
from rlinf.runners.embodied_runner import EmbodiedRunner, _resolve_channel_names
from rlinf.scheduler.rlix.protocol import ElasticBatchReceipt, FixedWorkerResidency
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def test_runner_channel_names_preserve_defaults_and_allow_isolation() -> None:
    assert _resolve_channel_names(None, use_reward=False) == {
        "env": "Env",
        "rollout": "Rollout",
        "actor": "Actor",
    }
    assert _resolve_channel_names(None, use_reward=True)["reward"] == "Reward"
    isolated = {
        "env": "t8_a_env",
        "rollout": "t8_a_rollout",
        "actor": "t8_a_actor",
    }
    assert _resolve_channel_names(isolated, use_reward=False) == isolated
    with pytest.raises(ValueError, match="exactly"):
        _resolve_channel_names({"env": "only-one"}, use_reward=False)
    with pytest.raises(ValueError, match="unique"):
        _resolve_channel_names(
            {"env": "same", "rollout": "same", "actor": "actor"},
            use_reward=False,
        )


class _Handle:
    def __init__(self, events: list[str], event: str, result=None) -> None:
        self.events = events
        self.event = event
        self.result = result

    def wait(self):
        self.events.append(self.event)
        return self.result


class _Group:
    def __init__(self, component: str, events: list[str]) -> None:
        self.component = component
        self.events = events
        self.policy_version = 0

    def init_worker(self):
        self.events.append(f"start_init:{self.component}")
        return _Handle(self.events, f"wait_init:{self.component}")

    def load_checkpoint(self, path):
        self.events.append(f"start_legacy_restore:{path}")
        return _Handle(self.events, "wait_legacy_restore")

    def load_rlix_checkpoint(self, path):
        self.events.append(f"start_rlix_restore:{path}")
        return _Handle(self.events, "wait_rlix_restore")

    def get_rlix_fixed_residency(self):
        self.events.append(f"start_residency:{self.component}")
        return _Handle(
            self.events,
            f"wait_residency:{self.component}",
            [
                FixedWorkerResidency(
                    component=self.component,
                    rank=0,
                    model_resident=False,
                    optimizer_resident=False,
                    cuda_graph_captured=False,
                    policy_version=self.policy_version,
                )
            ],
        )

    def set_global_step(self, global_step):
        self.events.append(f"start_set_global_step:{self.component}:{global_step}")

        class _SetGlobalStepHandle(_Handle):
            def wait(handle_self):
                self.policy_version = global_step
                return super().wait()

        return _SetGlobalStepHandle(
            self.events, f"wait_set_global_step:{self.component}:{global_step}"
        )

    def sync_model_from_actor(self):
        self.events.append(f"start_sync_from:{self.component}")
        return _Handle(self.events, f"wait_sync_from:{self.component}")

    def sync_model_to_rollout(self):
        self.events.append(f"start_sync_to:{self.component}")
        return _Handle(self.events, f"wait_sync_to:{self.component}")

    def recv_rollout_trajectories(self, *, input_channel):
        self.events.append(f"start_receiver:{input_channel}")
        return _Handle(self.events, "wait_receiver")

    def seal_rlix_batch(self, **kwargs):
        self.events.append(f"start_seal:{kwargs['expected_trajectories']}")
        return _Handle(self.events, "wait_seal", ["actor-receipt"])

    def compute_advantages_and_returns(self):
        self.events.append("start_advantages")
        return _Handle(self.events, "wait_advantages", ["rollout-metrics"])

    def run_rlix_training(self, receipt):
        self.events.append(f"start_training:{receipt}")
        return _Handle(self.events, "wait_training", ["training-metrics"])

    def evaluate(self, **kwargs):
        del kwargs
        self.events.append(f"start_evaluate:{self.component}")
        result = (
            [{"success": torch.tensor([1.0])}]
            if self.component == "environment"
            else None
        )
        return _Handle(self.events, f"wait_evaluate:{self.component}", result)


class _Stage:
    def __init__(self, events: list[str], label: str) -> None:
        self.events = events
        self.label = label

    def __enter__(self):
        self.events.append(f"acquire_{self.label}")
        return self

    def complete(self, receipt) -> None:
        self.events.append(f"complete:{receipt}")

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.events.append(f"release_{self.label}")


class _Runtime:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def fixed_stage(self, **kwargs):
        self.events.append(
            f"fixed_stage:{kwargs['cluster_name']}:{kwargs['global_step']}"
        )
        return _Stage(self.events, kwargs["cluster_name"])

    def policy_sync_stage(self, **kwargs):
        self.events.append(f"policy_sync_stage:{kwargs['expected_policy_version']}")
        return _Stage(self.events, "policy_sync")

    def fixed_residency_receipt(self, **kwargs):
        statuses = kwargs["worker_residencies"]
        self.events.append(
            "verify:" + ",".join(status.component for status in statuses)
        )
        return "verified-receipt"


class _CollectionRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.placement_plan = SimpleNamespace(
            actor_infer_bundles=((0, (0,)), (1, (1,)))
        )

    def begin_collection(self, **kwargs):
        self.events.append(
            f"begin:{kwargs['policy_version']}:{kwargs['assigned_trajectories_by_rank']}"
        )
        receiver = kwargs["actor_receiver_start"]()
        return SimpleNamespace(
            context=SimpleNamespace(
                lifecycle_generation=3,
                policy_version=kwargs["policy_version"],
                dp_ranks=(0, 1),
            ),
            actor_receiver_handle=receiver,
        )

    def wait_for_collection(self, session, *, poll_interval_s):
        del session
        self.events.append(f"monitor:{poll_interval_s}")

    def seal_collection(self, session, *, actor_seal_start):
        self.events.append("seal_collection")
        actor_seal_start(4).wait()
        session.actor_receiver_handle.wait()
        return "batch-receipt"


def _runner(events: list[str], *, runtime=None) -> EmbodiedRunner:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(runner={"resume_dir": None})
    runner.actor = _Group("actor", events)
    runner.rollout = _Group("rollout", events)
    runner.env = _Group("environment", events)
    runner.reward = None
    runner.rlix_runtime = runtime
    runner.global_step = 0
    return runner


def test_disabled_initialization_preserves_exact_legacy_order() -> None:
    events: list[str] = []
    runner = _runner(events)

    runner.init_workers()

    assert events == [
        "start_init:rollout",
        "start_init:environment",
        "wait_init:rollout",
        "wait_init:environment",
        "start_init:actor",
        "wait_init:actor",
    ]


def test_enabled_initialization_is_allocated_and_verified() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runner = _runner(events, runtime=runtime)

    runner.init_workers()

    assert events == [
        "fixed_stage:initialization:0",
        "acquire_initialization",
        "start_init:rollout",
        "start_init:environment",
        "wait_init:rollout",
        "wait_init:environment",
        "start_init:actor",
        "wait_init:actor",
        "start_residency:actor",
        "wait_residency:actor",
        "start_residency:rollout",
        "wait_residency:rollout",
        "start_residency:environment",
        "wait_residency:environment",
        "verify:actor,rollout,environment",
        "complete:verified-receipt",
        "release_initialization",
    ]


def test_enabled_checkpoint_restore_reoffloads_before_initialization_release(
    tmp_path,
) -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runner = _runner(events, runtime=runtime)
    resume_dir = tmp_path / "global_step_4"
    (resume_dir / "actor").mkdir(parents=True)
    runner.cfg = SimpleNamespace(runner={"resume_dir": str(resume_dir)})
    runner.logger = SimpleNamespace(info=lambda message: None)

    runner.init_workers()

    restore_start = events.index(f"start_rlix_restore:{resume_dir / 'actor'}")
    restore_wait = events.index("wait_rlix_restore")
    verify = events.index("verify:actor,rollout,environment")
    release = events.index("release_initialization")
    assert restore_start < restore_wait < verify < release
    assert not any(event.startswith("start_legacy_restore") for event in events)
    assert runner.global_step == 4


def test_disabled_checkpoint_restore_keeps_legacy_loader(tmp_path) -> None:
    events: list[str] = []
    runner = _runner(events)
    resume_dir = tmp_path / "global_step_3"
    (resume_dir / "actor").mkdir(parents=True)
    runner.cfg = SimpleNamespace(runner={"resume_dir": str(resume_dir)})
    runner.logger = SimpleNamespace(info=lambda message: None)

    runner.init_workers()

    assert f"start_legacy_restore:{resume_dir / 'actor'}" in events
    assert not any(event.startswith("start_rlix_restore") for event in events)
    assert runner.global_step == 3


def test_disabled_policy_sync_preserves_collective_order() -> None:
    events: list[str] = []
    runner = _runner(events)

    runner.update_rollout_weights()

    assert events == [
        "start_sync_from:rollout",
        "start_sync_to:actor",
        "wait_sync_to:actor",
        "wait_sync_from:rollout",
    ]


def test_enabled_policy_sync_is_leased_versioned_and_verified() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runner = _runner(events, runtime=runtime)
    runner.global_step = 6

    runner.update_rollout_weights()

    assert events == [
        "policy_sync_stage:6",
        "acquire_policy_sync",
        "start_sync_from:rollout",
        "start_sync_to:actor",
        "wait_sync_to:actor",
        "wait_sync_from:rollout",
        "start_residency:actor",
        "wait_residency:actor",
        "start_residency:rollout",
        "wait_residency:rollout",
        "verify:actor,rollout",
        "complete:verified-receipt",
        "release_policy_sync",
    ]


def test_actor_seals_only_complete_single_version_cpu_batch() -> None:
    actor = object.__new__(EmbodiedFSDPActor)
    actor._rlix_batch_receipt = None
    actor.rollout_batch = {"versions": torch.full((3, 4, 1), 6)}
    actor._rlix_received_transition_ids = (
        RolloutTransitionIdentity(2, 0, 0, 0),
        RolloutTransitionIdentity(2, 1, 0, 0),
    )

    receipt = actor.seal_rlix_batch(
        lifecycle_generation=2,
        policy_version=6,
        contributing_dp_ranks=(0, 1),
        expected_trajectories=4,
    )

    assert receipt.received_trajectories == 4
    assert receipt.transition_count == 2
    with pytest.raises(RuntimeError, match="already sealed"):
        actor.seal_rlix_batch(
            lifecycle_generation=2,
            policy_version=6,
            contributing_dp_ranks=(0, 1),
            expected_trajectories=4,
        )


def test_actor_rejects_partial_or_mixed_version_batch() -> None:
    actor = object.__new__(EmbodiedFSDPActor)
    actor._rlix_batch_receipt = None
    actor.rollout_batch = {"versions": torch.tensor([[[6]], [[7]]])}
    actor._rlix_received_transition_ids = (RolloutTransitionIdentity(2, 0, 0, 0),)

    with pytest.raises(ValueError, match="mixed policy versions"):
        actor.seal_rlix_batch(
            lifecycle_generation=2,
            policy_version=6,
            contributing_dp_ranks=(0,),
            expected_trajectories=1,
        )

    actor.rollout_batch = {"versions": torch.full((3, 3, 1), 6)}
    with pytest.raises(ValueError, match="must equal expected"):
        actor.seal_rlix_batch(
            lifecycle_generation=2,
            policy_version=6,
            contributing_dp_ranks=(0,),
            expected_trajectories=4,
        )


def test_actor_rejects_duplicate_transition_identity() -> None:
    actor = object.__new__(EmbodiedFSDPActor)
    actor._rlix_batch_receipt = None
    actor.rollout_batch = {"versions": torch.full((3, 2, 1), 6)}
    identity = RolloutTransitionIdentity(2, 0, 0, 0)
    actor._rlix_received_transition_ids = (identity, identity)

    with pytest.raises(ValueError, match="duplicate transition"):
        actor.seal_rlix_batch(
            lifecycle_generation=2,
            policy_version=6,
            contributing_dp_ranks=(0,),
            expected_trajectories=2,
        )


def test_runner_builds_uniform_assignments_and_uses_collection_runtime() -> None:
    events: list[str] = []
    runtime = _CollectionRuntime(events)
    runner = _runner(events, runtime=runtime)
    runner.global_step = 5
    runner.cfg = SimpleNamespace(
        env=SimpleNamespace(train=SimpleNamespace(total_num_envs=4, rollout_epoch=2)),
        rlix={"monitor_poll_interval_s": 0.02},
    )
    runner.env_channel = "env-channel"
    runner.rollout_channel = "rollout-channel"
    runner.actor_channel = "actor-channel"
    runner.reward_channel = None

    receipt = runner._collect_rlix_rollouts()

    assert receipt == "batch-receipt"
    assert events == [
        "begin:5:{0: 4, 1: 4}",
        "start_receiver:actor-channel",
        "monitor:0.02",
        "seal_collection",
        "start_seal:4",
        "wait_seal",
        "wait_receiver",
    ]


def test_fixed_actor_training_advances_version_only_after_verified_release() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runner = _runner(events, runtime=runtime)
    runner.global_step = 9

    rollout_metrics, training_metrics, _ = runner._train_rlix_batch("batch-9")

    assert rollout_metrics == ["rollout-metrics"]
    assert training_metrics == ["training-metrics"]
    assert runner.global_step == 10
    assert events == [
        "fixed_stage:actor_train:9",
        "acquire_actor_train",
        "start_advantages",
        "wait_advantages",
        "start_training:batch-9",
        "wait_training",
        "start_set_global_step:actor:10",
        "wait_set_global_step:actor:10",
        "start_residency:actor",
        "wait_residency:actor",
        "verify:actor",
        "complete:verified-receipt",
        "release_actor_train",
    ]


def test_failed_actor_training_does_not_advance_policy_version() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runner = _runner(events, runtime=runtime)
    runner.global_step = 9

    class _FailedHandle:
        def wait(self):
            events.append("wait_training")
            raise RuntimeError("optimizer failed")

    runner.actor.run_rlix_training = lambda receipt: (
        events.append(f"start_training:{receipt}") or _FailedHandle()
    )

    with pytest.raises(RuntimeError, match="optimizer failed"):
        runner._train_rlix_batch("batch-9")

    assert runner.global_step == 9
    assert not any("set_global_step" in event for event in events)
    assert "complete:verified-receipt" not in events


def test_failed_actor_version_publication_does_not_release_or_advance() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runner = _runner(events, runtime=runtime)
    runner.global_step = 9

    class _FailedHandle:
        def wait(self):
            events.append("wait_set_global_step:actor:10")
            raise RuntimeError("version publication failed")

    runner.actor.set_global_step = lambda version: (
        events.append(f"start_set_global_step:actor:{version}") or _FailedHandle()
    )

    with pytest.raises(RuntimeError, match="version publication failed"):
        runner._train_rlix_batch("batch-9")

    assert runner.global_step == 9
    assert "complete:verified-receipt" not in events


def test_actor_training_consumes_seal_and_offloads_state() -> None:
    actor = object.__new__(EmbodiedFSDPActor)
    receipt = ElasticBatchReceipt(2, 6, (0,), 4, 4, 12)
    actor._rlix_batch_receipt = ElasticBatchReceipt(2, 6, (0,), 4, 4, 12)
    actor.is_weight_offloaded = False
    actor.is_optimizer_offloaded = False
    events: list[str] = []
    actor.run_training = lambda: events.append("train") or {"loss": 1.0}
    actor.offload_param_and_grad = lambda offload_grad: events.append(
        f"offload_weights:{offload_grad}"
    )
    actor.offload_optimizer = lambda: events.append("offload_optimizer")

    metrics = actor.run_rlix_training(receipt)

    assert metrics == {"loss": 1.0}
    assert events == ["train", "offload_weights:True", "offload_optimizer"]
    assert actor._rlix_batch_receipt is None


def test_actor_rlix_checkpoint_restore_reestablishes_offload() -> None:
    actor = object.__new__(EmbodiedFSDPActor)
    actor.is_weight_offloaded = True
    actor.is_optimizer_offloaded = True
    events: list[str] = []

    def load_checkpoint(path):
        events.append(f"restore:{path}")
        actor.is_weight_offloaded = False
        actor.is_optimizer_offloaded = False

    def offload_weights(offload_grad):
        events.append(f"offload_weights:{offload_grad}")
        actor.is_weight_offloaded = True

    def offload_optimizer():
        events.append("offload_optimizer")
        actor.is_optimizer_offloaded = True

    actor.load_checkpoint = load_checkpoint
    actor.offload_param_and_grad = offload_weights
    actor.offload_optimizer = offload_optimizer

    actor.load_rlix_checkpoint("checkpoint/actor")

    assert events == [
        "restore:checkpoint/actor",
        "offload_weights:True",
        "offload_optimizer",
    ]
    assert actor.is_weight_offloaded
    assert actor.is_optimizer_offloaded


def test_enabled_evaluation_is_fixed_and_residency_verified() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runner = _runner(events, runtime=runtime)
    runner.global_step = 10
    runner.env_channel = "env"
    runner.rollout_channel = "rollout"

    metrics = runner.evaluate()

    assert "success" in metrics
    assert events == [
        "fixed_stage:evaluation:10",
        "acquire_evaluation",
        "start_evaluate:environment",
        "start_evaluate:rollout",
        "wait_evaluate:environment",
        "wait_evaluate:rollout",
        "start_residency:rollout",
        "wait_residency:rollout",
        "start_residency:environment",
        "wait_residency:environment",
        "verify:rollout,environment",
        "complete:verified-receipt",
        "release_evaluation",
    ]


def test_evaluation_offload_failure_prevents_verified_completion() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runner = _runner(events, runtime=runtime)
    runner.env_channel = "env"
    runner.rollout_channel = "rollout"
    runtime.fixed_residency_receipt = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("evaluation remains resident")
    )

    with pytest.raises(RuntimeError, match="evaluation remains resident"):
        runner.evaluate()

    assert "complete:verified-receipt" not in events


def test_enabled_checkpoint_is_fixed_and_verified_before_release() -> None:
    events: list[str] = []
    runtime = _Runtime(events)
    runner = _runner(events, runtime=runtime)
    runner.global_step = 11
    runner._save_checkpoint_workers = lambda: events.append("save_checkpoint")

    runner._save_checkpoint()

    assert events == [
        "fixed_stage:actor_train:11",
        "acquire_actor_train",
        "save_checkpoint",
        "start_residency:actor",
        "wait_residency:actor",
        "verify:actor",
        "complete:verified-receipt",
        "release_actor_train",
    ]


def test_post_training_evaluation_syncs_current_version_first(monkeypatch) -> None:
    events: list[str] = []
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.global_step = 12
    runner.max_steps = 20
    runner.cfg = SimpleNamespace(
        runner=SimpleNamespace(val_check_interval=1, save_interval=-1)
    )
    runner.timer = lambda label: nullcontext()
    runner.update_rollout_weights = lambda: events.append("sync:12")
    runner.evaluate = lambda: events.append("evaluate:12") or {"success": 1.0}
    runner.metric_logger = SimpleNamespace(
        log=lambda **kwargs: events.append(f"log:{kwargs['step']}")
    )
    runner._save_checkpoint = lambda: events.append("checkpoint")
    monkeypatch.setattr(
        "rlinf.runners.embodied_runner.check_progress",
        lambda *args, **kwargs: (True, False, False),
    )

    metrics = runner._maybe_eval_and_checkpoint(step=11)

    assert metrics == {"eval/success": 1.0}
    assert events == ["sync:12", "evaluate:12", "log:11"]


def test_enabled_run_selects_rlix_loop_before_legacy_pipeline() -> None:
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.rlix_runtime = object()
    runner._run_rlix = lambda: "rlix-run"

    assert runner.run() == "rlix-run"


def test_disabled_run_preserves_complete_single_step_call_trace() -> None:
    events: list[str] = []

    class _Timer:
        def __call__(self, label):
            class _Context:
                def __enter__(inner_self):
                    events.append(f"enter:{label}")

                def __exit__(inner_self, exc_type, exc, traceback):
                    del exc_type, exc, traceback
                    events.append(f"exit:{label}")

            return _Context()

    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.rlix_runtime = None
    runner.cfg = SimpleNamespace(runner={"use_training_pipeline": False})
    runner.global_step = 0
    runner.max_steps = 1
    runner.weight_sync_interval = 1
    runner.timer = _Timer()
    runner.reward = None
    runner.overlap_env_bootstrap = False
    runner.actor = SimpleNamespace(
        set_global_step=lambda step: events.append(f"actor_step:{step}"),
        recv_rollout_trajectories=lambda **kwargs: _Handle(events, "wait_receiver"),
        compute_advantages_and_returns=lambda: _Handle(events, "wait_advantages", []),
        run_training=lambda: _Handle(events, "wait_training", []),
    )
    runner.rollout = SimpleNamespace(
        set_global_step=lambda step: events.append(f"rollout_step:{step}"),
        generate=lambda **kwargs: (
            events.append("start_rollout") or _Handle(events, "wait_rollout")
        ),
    )
    runner.env = SimpleNamespace(
        interact=lambda **kwargs: (
            events.append("start_env") or _Handle(events, "wait_env")
        )
    )
    runner.env_channel = "env"
    runner.rollout_channel = "rollout"
    runner.reward_channel = None
    runner.actor_channel = "actor"
    runner._should_profile_step = lambda step: False
    runner.update_rollout_weights = lambda: events.append("sync")
    runner._maybe_eval_and_checkpoint = lambda step: events.append(f"post:{step}") or {}
    runner._log_step_metrics = lambda **kwargs: events.append(f"log:{kwargs['step']}")
    runner._finish_run = lambda: events.append("finish")

    runner.run()

    assert events == [
        "actor_step:0",
        "rollout_step:0",
        "enter:step",
        "enter:sync_weights",
        "sync",
        "exit:sync_weights",
        "enter:generate_rollouts",
        "start_env",
        "start_rollout",
        "wait_receiver",
        "wait_rollout",
        "exit:generate_rollouts",
        "enter:cal_adv_and_returns",
        "wait_advantages",
        "exit:cal_adv_and_returns",
        "wait_training",
        "post:0",
        "exit:step",
        "log:0",
        "finish",
    ]


def test_rlix_loop_orders_sync_collection_training_and_post_step() -> None:
    events: list[str] = []

    class _Timer:
        def __call__(self, label):
            class _Context:
                def __enter__(inner_self):
                    events.append(f"enter:{label}")

                def __exit__(inner_self, exc_type, exc, traceback):
                    del exc_type, exc, traceback
                    events.append(f"exit:{label}")

            return _Context()

    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(runner={"use_training_pipeline": False})
    runner.global_step = 0
    runner.max_steps = 1
    runner.weight_sync_interval = 1
    runner.timer = _Timer()
    runner.actor = SimpleNamespace(
        set_global_step=lambda step: events.append(f"actor_step:{step}")
    )
    runner.rollout = SimpleNamespace(
        set_global_step=lambda step: events.append(f"rollout_step:{step}")
    )
    runner._should_profile_step = lambda step: False
    runner.update_rollout_weights = lambda: events.append("sync")
    runner._collect_rlix_rollouts = lambda: events.append("collect") or "batch"

    def train(receipt):
        events.append(f"train:{receipt}")
        runner.global_step += 1
        return [], [], "handle"

    runner._train_rlix_batch = train
    runner._maybe_eval_and_checkpoint = lambda step: events.append(f"post:{step}") or {}
    runner._log_rlix_step_metrics = lambda **kwargs: events.append(
        f"log:{kwargs['step']}"
    )
    runner._finish_run = lambda: events.append("finish")

    runner._run_rlix()

    assert events == [
        "actor_step:0",
        "rollout_step:0",
        "enter:step",
        "enter:sync_weights",
        "sync",
        "exit:sync_weights",
        "enter:generate_rollouts",
        "collect",
        "exit:generate_rollouts",
        "enter:cal_adv_and_returns",
        "train:batch",
        "exit:cal_adv_and_returns",
        "post:0",
        "exit:step",
        "log:0",
        "finish",
    ]
