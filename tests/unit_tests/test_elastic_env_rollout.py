import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.registry import calculate_adv_and_returns
from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    ElasticRolloutRequest,
    ElasticRolloutRequestKind,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutResult,
    infer_elastic_rollout_request_batch_size,
    merge_elastic_rollout_requests,
    merge_transition_identities,
    split_elastic_rollout_request,
)
from rlinf.scheduler.rlix.coordinator import RLixResizeCoordinator
from rlinf.scheduler.rlix.protocol import ElasticCollectionContext
from rlinf.scheduler.worker.routing import build_send_plan
from rlinf.utils.metric_utils import compute_loss_mask
from rlinf.workers.elastic_rollout_lifecycle import (
    CompletedResidencyReceipt,
    DrainRequest,
    ElasticRankState,
    ElasticRunOutcome,
    ElasticRunResult,
    ElasticValidationMode,
    ResidencyOperationReceipt,
    ResidencyReceipt,
    RolloutTransitionIdentity,
    SafePointToken,
    validate_elastic_state_transition,
)
from rlinf.workers.env.env_worker import (
    ENV_ROLLOUT_RESUME_SCHEMA_VERSION,
    EnvRolloutCursor,
    EnvWorker,
    RolloutCursorPhase,
)
from rlinf.workers.rollout.hf.huggingface_worker import (
    MultiStepRolloutWorker,
    RolloutPeerCursor,
    RolloutPeerPhase,
)


def _identity(*, lifecycle: int = 1, rank: int = 0, sequence: int = 0):
    return RolloutTransitionIdentity(
        lifecycle_generation=lifecycle,
        env_worker_rank=rank,
        stage_id=0,
        sequence=sequence,
    )


def test_receipt_residency_mode_skips_python_tensor_traversal() -> None:
    worker = _elastic_rollout_worker([])
    worker._elastic_residency_validation_mode = ElasticValidationMode.RECEIPT
    worker._elastic_residency_operation_generation = 0
    worker._elastic_residency_operation_receipt = None
    worker._elastic_cursor = RolloutPeerCursor(
        lifecycle_generation=1,
        policy_version=3,
        epoch_index=0,
        committed_chunk_count=0,
        expected_transition_id=_identity(),
        phase=RolloutPeerPhase.IDLE,
    )
    worker._model_resident = True

    receipt = worker._validate_rollout_residency_operation(resident=True)

    assert receipt is not None
    assert receipt.operation_generation == 1
    assert receipt.destination_device == "accelerator"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"destination_device": "cpu"}, "destination"),
        ({"synchronized": False}, "synchronized"),
        ({"moved_bytes": -1}, "non-negative"),
    ],
)
def test_residency_operation_receipt_rejects_incomplete_evidence(updates, message):
    values = {
        "worker_rank": 0,
        "lifecycle_generation": 1,
        "policy_version": 3,
        "operation_generation": 1,
        "resident": True,
        "destination_device": "accelerator",
        "synchronized": True,
        "moved_bytes": None,
    }
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        ResidencyOperationReceipt(**values)


def _env_output(identity: RolloutTransitionIdentity | None) -> EnvOutput:
    return EnvOutput(
        obs={"states": torch.zeros((1, 2))},
        dones=torch.zeros((1, 1), dtype=torch.bool),
        transition_id=identity,
    )


class _ImmediateWork:
    def __init__(self, value=None):
        self.value = value
        self.awaited = False

    async def async_wait(self):
        self.awaited = True
        return self.value


class _QueueGetWork:
    def __init__(self, queue):
        self.queue = queue

    async def async_wait(self):
        return await self.queue.get()


class _QueuePutWork:
    def __init__(self, queue, value):
        self.queue = queue
        self.value = value

    async def async_wait(self):
        await self.queue.put(self.value)


class _FakeResidentModel:
    def __init__(self, device="cuda"):
        self.device = device
        self.moves = []

    def to(self, device):
        self.device = str(device)
        self.moves.append(str(device))
        return self

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())

    def verify_elastic_residency(self, *, resident):
        if resident != (self.device != "cpu"):
            raise RuntimeError("fake residency mismatch")


class _FakePlatform:
    def __init__(self):
        self.synchronize_count = 0
        self.empty_cache_count = 0

    def synchronize(self):
        self.synchronize_count += 1

    def empty_cache(self):
        self.empty_cache_count += 1


class _FakeElasticEnvironment:
    def __init__(self):
        self.resident = False

    def onload(self):
        self.resident = True

    def offload(self):
        self.resident = False

    def verify_elastic_residency(self, *, resident):
        if self.resident != resident:
            raise RuntimeError("fake environment residency mismatch")


class _FakePartialRollout:
    def __init__(self):
        self.steps = []

    def append_step_result(self, result):
        self.steps.append(result)


@dataclass(frozen=True)
class _FakeEnvResumeState:
    resume_bootstraps: tuple[EnvOutput, ...]


def _elastic_rollout_worker(requests, *, rank=0, world_size=1):
    worker = object.__new__(MultiStepRolloutWorker)
    worker._rank = rank
    worker.enable_train = True
    worker.num_pipeline_stages = 1
    worker.env_decoupled_mode = False
    worker.enable_offload = True
    worker.version = 3
    worker._elastic_cursor = None
    worker._elastic_state = ElasticRankState.INACTIVE_COLD
    worker._elastic_expected_policy_version = None
    worker._elastic_drain_request = None
    worker._elastic_safe_point_token = None
    worker._elastic_failure = None
    worker._elastic_dagger_epoch_index = None
    worker._model_resident = True
    worker._cuda_graph_captured = False
    worker.enable_cuda_graph = False
    worker.hf_model = None
    worker.expert_model = None
    worker.rlt_feature_model = None
    worker.rollout_epoch = 1
    worker.n_train_chunk_steps = 1
    worker.train_batch_size = 1
    worker.collect_prev_infos = True
    worker.model_cfg = SimpleNamespace(num_action_chunks=1)
    worker.cfg = SimpleNamespace(env=SimpleNamespace(group_name="env"))
    worker.placement = SimpleNamespace(get_world_size=lambda _group_name: world_size)
    worker.get_bootstrap_values = lambda _final_obs: None
    worker.predict_count = 0
    worker.sent_results = []
    worker.send_works = []

    def predict(_obs, **_kwargs):
        worker.predict_count += 1
        actions = torch.ones((1, 2))
        return actions, {
            "prev_logprobs": torch.zeros((1, 2)),
            "prev_values": torch.zeros((1, 1)),
            "forward_inputs": {"action": actions},
            "expert_label_flag": False,
        }

    request_queue = list(requests)

    def recv_from(**_kwargs):
        return _ImmediateWork(request_queue.pop(0))

    def send_to(**kwargs):
        worker.sent_results.append(kwargs["data"])
        work = _ImmediateWork()
        worker.send_works.append(work)
        return work

    worker._predict_rollout_actions = predict
    worker.recv_from = recv_from
    worker.send_to = send_to
    return worker


def _elastic_env_worker(*, rank=0, world_size=1):
    worker = object.__new__(EnvWorker)
    worker._rank = rank
    worker._world_size = world_size
    worker._accelerator_type = None
    worker._timer_metrics = {}
    worker.stage_num = 1
    worker.enable_train = True
    worker.train_enable_offload = True
    worker.train_batch_size = 1
    worker.train_num_envs_per_stage = 1
    worker.rollout_epoch = 1
    worker.env_decoupled_mode = False
    worker._component_placement = SimpleNamespace(
        get_world_size=lambda _group_name: world_size
    )
    worker.cfg = OmegaConf.create(
        {
            "rollout": {"group_name": "rollout"},
            "env": {"train": {"auto_reset": True, "ignore_terminations": False}},
        }
    )
    worker.env_list = [_FakeElasticEnvironment()]
    worker._rollout_cursor = None
    worker._elastic_state = ElasticRankState.INACTIVE_COLD
    worker._elastic_expected_policy_version = None
    worker._elastic_drain_request = None
    worker._elastic_safe_point_token = None
    worker._elastic_resume_state = None
    worker._elastic_failure = None
    worker._environment_resident = False
    worker._lifecycle_generation = 0
    worker.stop_rank_when_all_done = False
    worker._validate_snapshot_capability = lambda: None
    worker.sent_requests = []
    worker.send_works = []

    def send_to(**kwargs):
        worker.sent_requests.append(kwargs["data"])
        work = _ImmediateWork()
        worker.send_works.append(work)
        return work

    worker.send_to = send_to
    return worker


def _configure_single_chunk_env(worker, *, snapshot_factory=lambda: object()):
    worker.rollout_epoch = 1
    worker.n_train_chunk_steps = 1
    worker.collect_prev_infos = True
    worker.collect_transitions = False
    worker.reward_mode = "per_step"
    worker.history_reward_assign = False
    worker.use_training_pipeline = False
    worker.enable_online_lerobot = False
    worker.enable_rlt = False
    worker.stop_rank_when_all_done = False
    worker.train_prev_done = [
        torch.zeros(worker.train_num_envs_per_stage, dtype=torch.bool)
    ]
    worker.model_cfg = SimpleNamespace(num_action_chunks=1)
    worker._rollout_call_active = False
    worker._policy_request_in_flight = False
    worker._prefetched_train_bootstrap = None
    worker._rlt_pending_obs = [None]
    worker._resume_bootstraps = [None]
    worker._current_env_outputs = None
    worker._rollout_env_metrics = {}
    worker.last_obs_list = [{}]
    worker.last_intervened_info_list = [(None, None)]
    partial_rollout = _FakePartialRollout()
    worker._prepare_rollout_results = lambda _previous: [partial_rollout]
    worker.bootstrap_step = lambda: [_env_output(None)]
    events = []

    def interact_step(_actions, _stage_id):
        events.append("chunk_committed")
        return _env_output(None), {}, {}

    worker.env_interact_step = interact_step
    worker.compute_bootstrap_rewards = lambda *_args: torch.zeros((1, 1))
    worker.snapshot_rollout_stage = lambda: (
        events.append("snapshot") or snapshot_factory()
    )
    worker.store_last_obs_and_intervened_info = lambda _outputs: None
    worker.finish_rollout = lambda: None
    return events, partial_rollout


def test_transition_identity_validates_components():
    assert _identity(sequence=3).sequence == 3

    with pytest.raises(ValueError, match="positive"):
        _identity(lifecycle=0)
    with pytest.raises(ValueError, match="env_worker_rank"):
        _identity(rank=-1)
    with pytest.raises(TypeError, match="sequence"):
        RolloutTransitionIdentity(1, 0, 0, True)


def test_env_output_to_dict_preserves_transition_identity():
    identity = _identity()

    assert _env_output(identity).to_dict()["transition_id"] == identity


@pytest.mark.parametrize(
    "identities",
    [
        [None, _identity()],
        [_identity(), _identity(lifecycle=2)],
        [_identity(), _identity(rank=1)],
        [_identity(), RolloutTransitionIdentity(1, 0, 1, 0)],
        [_identity(), _identity(sequence=1)],
    ],
)
def test_transition_identity_merge_rejects_mixed_or_different_values(identities):
    with pytest.raises(ValueError):
        merge_transition_identities(identities)


def test_transition_identity_merge_preserves_legacy_and_identified_values():
    identity = _identity()

    assert merge_transition_identities([None, None]) is None
    assert merge_transition_identities([identity, identity]) == identity


def test_elastic_rollout_request_validates_final_bootstrap_marker():
    with pytest.raises(TypeError, match="final_bootstrap must be a boolean"):
        ElasticRolloutRequest(
            kind=ElasticRolloutRequestKind.OBSERVATION,
            transition_id=_identity(),
            logical_batch_size=1,
            env_output=_env_output(_identity()).to_dict(),
            final_bootstrap=1,
        )

    with pytest.raises(ValueError, match="drain barrier cannot be a final"):
        ElasticRolloutRequest(
            kind=ElasticRolloutRequestKind.DRAIN_BARRIER,
            transition_id=_identity(),
            logical_batch_size=1,
            env_output=None,
            drain_request_id="drain-1",
            final_bootstrap=True,
        )


def test_elastic_rollout_request_split_and_merge_preserve_final_bootstrap():
    output = EnvOutput(
        obs={"states": torch.zeros((2, 2))},
        dones=torch.zeros((2, 1), dtype=torch.bool),
        transition_id=_identity(),
    )
    request = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.OBSERVATION,
        transition_id=_identity(),
        logical_batch_size=2,
        env_output=output.to_dict(),
        final_bootstrap=True,
    )

    shards = split_elastic_rollout_request(request, [1, 1])
    assert all(shard.final_bootstrap for shard in shards)
    assert merge_elastic_rollout_requests(shards).final_bootstrap

    shards[1].final_bootstrap = False
    with pytest.raises(ValueError, match="different final-bootstrap"):
        merge_elastic_rollout_requests(shards)


def test_rollout_result_merge_preserves_identity():
    identity = _identity()
    results = [
        RolloutResult(actions=torch.ones((1, 2)), transition_id=identity),
        RolloutResult(actions=torch.zeros((1, 2)), transition_id=identity),
    ]

    merged = RolloutResult.merge_rollout_results(results)

    assert merged.transition_id == identity
    assert merged.actions.shape == (2, 2)


def test_observation_request_merge_preserves_identity_and_batch():
    identity = _identity()
    requests = [
        ElasticRolloutRequest(
            kind=ElasticRolloutRequestKind.OBSERVATION,
            transition_id=identity,
            logical_batch_size=1,
            env_output=_env_output(identity).to_dict(),
        ),
        ElasticRolloutRequest(
            kind=ElasticRolloutRequestKind.OBSERVATION,
            transition_id=identity,
            logical_batch_size=1,
            env_output=_env_output(identity).to_dict(),
        ),
    ]

    merged = merge_elastic_rollout_requests(requests)

    assert merged.kind is ElasticRolloutRequestKind.OBSERVATION
    assert merged.transition_id == identity
    assert merged.logical_batch_size == 2
    assert merged.env_output["transition_id"] == identity
    assert merged.env_output["obs"]["states"].shape == (2, 2)


def test_drain_barrier_preserves_routed_batch_size_without_observation():
    identity = _identity(sequence=1)
    request = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.DRAIN_BARRIER,
        transition_id=identity,
        logical_batch_size=4,
        env_output=None,
        drain_request_id="drain-1",
    )

    merged = merge_elastic_rollout_requests([request])

    assert merged.env_output is None
    assert merged.drain_request_id == "drain-1"
    assert infer_elastic_rollout_request_batch_size(merged) == 4


def test_observation_request_split_preserves_identity_and_route_sizes():
    identity = _identity()
    request = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.OBSERVATION,
        transition_id=identity,
        logical_batch_size=2,
        env_output=EnvOutput(
            obs={"states": torch.zeros((2, 2))},
            dones=torch.zeros((2, 1), dtype=torch.bool),
            transition_id=identity,
        ).to_dict(),
    )

    shards = split_elastic_rollout_request(request, [1, 1])

    assert [shard.logical_batch_size for shard in shards] == [1, 1]
    assert all(shard.transition_id == identity for shard in shards)
    assert all(shard.env_output["obs"]["states"].shape == (1, 2) for shard in shards)


def test_drain_barrier_rejects_non_one_to_one_routing():
    request = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.DRAIN_BARRIER,
        transition_id=_identity(),
        logical_batch_size=2,
        env_output=None,
        drain_request_id="drain-1",
    )

    with pytest.raises(ValueError, match="one-to-one"):
        split_elastic_rollout_request(request, [1, 1])


def test_drain_barrier_local_batch_matches_two_rank_route_plan():
    request = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.DRAIN_BARRIER,
        transition_id=_identity(sequence=1),
        logical_batch_size=2,
        env_output=None,
        drain_request_id="drain-1",
    )
    plan = build_send_plan(
        src_group_name="env",
        dst_group_name="rollout",
        src_rank=0,
        src_world_size=2,
        dst_world_size=2,
        tag="train_rollout_results",
        route_key=0,
        batch_size=4,
    )
    split_sizes = [entry.batch_size for entry in plan.entries]

    shards = split_elastic_rollout_request(request, split_sizes)

    assert split_sizes == [2]
    assert shards == [request]


def test_request_split_error_reports_kind_sizes_and_logical_batch():
    request = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.DRAIN_BARRIER,
        transition_id=_identity(sequence=1),
        logical_batch_size=4,
        env_output=None,
        drain_request_id="drain-1",
    )

    with pytest.raises(
        ValueError,
        match=r"drain_barrier request split sizes \[2\] sum to 2.*logical batch size is 4",
    ):
        split_elastic_rollout_request(request, [2])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "kind": ElasticRolloutRequestKind.OBSERVATION,
                "logical_batch_size": 1,
                "env_output": None,
                "drain_request_id": None,
            },
            "must contain env_output",
        ),
        (
            {
                "kind": ElasticRolloutRequestKind.DRAIN_BARRIER,
                "logical_batch_size": 1,
                "env_output": _env_output(_identity()).to_dict(),
                "drain_request_id": "drain-1",
            },
            "cannot contain env_output",
        ),
        (
            {
                "kind": ElasticRolloutRequestKind.DRAIN_BARRIER,
                "logical_batch_size": 1,
                "env_output": None,
                "drain_request_id": None,
            },
            "must contain a drain_request_id",
        ),
    ],
)
def test_rollout_request_rejects_malformed_envelopes(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ElasticRolloutRequest(transition_id=_identity(), **kwargs)


def test_observation_request_rejects_mismatched_payload_identity():
    with pytest.raises(ValueError, match="must match"):
        ElasticRolloutRequest(
            kind=ElasticRolloutRequestKind.OBSERVATION,
            transition_id=_identity(),
            logical_batch_size=1,
            env_output=_env_output(_identity(sequence=1)).to_dict(),
        )


def test_observation_request_rejects_mismatched_logical_batch_size():
    with pytest.raises(ValueError, match="logical batch size"):
        ElasticRolloutRequest(
            kind=ElasticRolloutRequestKind.OBSERVATION,
            transition_id=_identity(),
            logical_batch_size=2,
            env_output=_env_output(_identity()).to_dict(),
        )


def test_elastic_rank_state_transition_table():
    allowed = {
        (ElasticRankState.INACTIVE_COLD, ElasticRankState.EXPANDING),
        (ElasticRankState.EXPANDING, ElasticRankState.ACTIVE),
        (ElasticRankState.EXPANDING, ElasticRankState.FAILED_RESIDENT),
        (ElasticRankState.ACTIVE, ElasticRankState.DRAIN_REQUESTED),
        (ElasticRankState.ACTIVE, ElasticRankState.COMPLETED),
        (ElasticRankState.ACTIVE, ElasticRankState.FAILED_RESIDENT),
        (ElasticRankState.DRAIN_REQUESTED, ElasticRankState.SNAPSHOTTING),
        (ElasticRankState.DRAIN_REQUESTED, ElasticRankState.COMPLETED),
        (ElasticRankState.DRAIN_REQUESTED, ElasticRankState.FAILED_RESIDENT),
        (ElasticRankState.SNAPSHOTTING, ElasticRankState.PAUSED),
        (ElasticRankState.SNAPSHOTTING, ElasticRankState.FAILED_RESIDENT),
        (ElasticRankState.PAUSED, ElasticRankState.EXPANDING),
        (ElasticRankState.COMPLETED, ElasticRankState.EXPANDING),
        (ElasticRankState.COMPLETED, ElasticRankState.FAILED_RESIDENT),
    }

    for current in ElasticRankState:
        for target in ElasticRankState:
            if (current, target) in allowed:
                validate_elastic_state_transition(current, target)
            else:
                with pytest.raises(ValueError, match="Invalid elastic rank state"):
                    validate_elastic_state_transition(current, target)


def test_lifecycle_operation_types_validate_identity_and_outcomes():
    identity = _identity(sequence=2)
    request = DrainRequest("drain-1", 0, 1, 3)
    token = SafePointToken("drain-1", 0, 1, 3, identity)

    assert request.expected_policy_version == token.policy_version
    assert ElasticRunResult(ElasticRunOutcome.PAUSE_READY, token, None).token == token
    assert (
        ResidencyReceipt(token, ElasticRankState.PAUSED, False, False).state
        is ElasticRankState.PAUSED
    )

    with pytest.raises(ValueError, match="requires a safe-point token"):
        ElasticRunResult(ElasticRunOutcome.PAUSE_READY, None, None)
    with pytest.raises(ValueError, match="worker rank"):
        SafePointToken("drain-1", 1, 1, 3, identity)


def test_rollout_peer_cursor_persists_remaining_work_identity():
    identity = _identity(sequence=3)
    cursor = RolloutPeerCursor(
        lifecycle_generation=1,
        policy_version=4,
        epoch_index=2,
        committed_chunk_count=3,
        expected_transition_id=identity,
        phase=RolloutPeerPhase.WAITING_FOR_ENV,
    )

    assert cursor.expected_transition_id == identity
    assert cursor.phase is RolloutPeerPhase.WAITING_FOR_ENV


def test_rollout_activation_and_drain_request_are_versioned_and_idempotent():
    worker = _elastic_rollout_worker([])
    status = worker.prepare_elastic_collection(
        lifecycle_generation=1, expected_policy_version=3
    )

    assert status.state is ElasticRankState.EXPANDING
    assert status.expected_transition_id == _identity()

    worker._elastic_state = ElasticRankState.ACTIVE
    request = DrainRequest("drain-1", 0, 1, 3)
    first = asyncio.run(worker.request_elastic_drain(request))
    second = asyncio.run(worker.request_elastic_drain(request))

    assert first.state is ElasticRankState.DRAIN_REQUESTED
    assert second.drain_request_id == "drain-1"


def test_completed_rollout_worker_requires_a_newer_lifecycle():
    worker = _elastic_rollout_worker([])
    worker.prepare_elastic_collection(lifecycle_generation=2, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.COMPLETED

    with pytest.raises(ValueError, match="newer lifecycle"):
        worker.prepare_elastic_collection(
            lifecycle_generation=2, expected_policy_version=3
        )

    status = worker.prepare_elastic_collection(
        lifecycle_generation=3, expected_policy_version=3
    )
    assert status.state is ElasticRankState.EXPANDING
    assert status.lifecycle_generation == 3


def test_elastic_rollout_finishes_inflight_observation_then_accepts_barrier():
    observation = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.OBSERVATION,
        transition_id=_identity(),
        logical_batch_size=1,
        env_output=_env_output(_identity()).to_dict(),
    )
    barrier = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.DRAIN_BARRIER,
        transition_id=_identity(sequence=1),
        logical_batch_size=1,
        env_output=None,
        drain_request_id="drain-1",
    )
    worker = _elastic_rollout_worker([observation, barrier])
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.ACTIVE
    asyncio.run(worker.request_elastic_drain(DrainRequest("drain-1", 0, 1, 3)))

    result = asyncio.run(worker.generate_until_pause_or_complete(None, None))

    assert result.outcome is ElasticRunOutcome.PAUSE_READY
    assert result.token.next_transition_id == _identity(sequence=1)
    assert worker.predict_count == 1
    assert len(worker.sent_results) == 1
    assert worker.sent_results[0].transition_id == _identity()
    assert all(work.awaited for work in worker.send_works)
    assert worker._elastic_cursor.committed_chunk_count == 1


def test_elastic_rollout_offloads_and_resumes_all_owned_models():
    barrier = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.DRAIN_BARRIER,
        transition_id=_identity(),
        logical_batch_size=1,
        env_output=None,
        drain_request_id="drain-1",
    )
    worker = _elastic_rollout_worker([barrier])
    worker.hf_model = _FakeResidentModel()
    worker.expert_model = _FakeResidentModel()
    worker.rlt_feature_model = _FakeResidentModel()
    worker.device = "cuda"
    worker.torch_platform = _FakePlatform()
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.ACTIVE
    asyncio.run(worker.request_elastic_drain(DrainRequest("drain-1", 0, 1, 3)))
    run_result = asyncio.run(worker.generate_until_pause_or_complete(None, None))

    pause_receipt = worker.offload_elastic_rollout(run_result.token)

    assert pause_receipt.state is ElasticRankState.PAUSED
    assert all(
        model.device == "cpu"
        for model in (worker.hf_model, worker.expert_model, worker.rlt_feature_model)
    )
    assert worker.torch_platform.synchronize_count == 1

    resume_receipt = worker.prepare_elastic_resume(run_result.token)

    assert resume_receipt.state is ElasticRankState.EXPANDING
    assert all(
        model.device == "cuda"
        for model in (worker.hf_model, worker.expert_model, worker.rlt_feature_model)
    )
    assert worker._elastic_drain_request is None
    assert worker._elastic_safe_point_token is None


def test_elastic_rollout_completes_chunk_and_final_bootstrap_once():
    observations = [
        ElasticRolloutRequest(
            kind=ElasticRolloutRequestKind.OBSERVATION,
            transition_id=_identity(sequence=sequence),
            logical_batch_size=1,
            env_output=_env_output(_identity(sequence=sequence)).to_dict(),
            final_bootstrap=sequence == 1,
        )
        for sequence in (0, 1)
    ]
    worker = _elastic_rollout_worker(observations)
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)

    result = asyncio.run(worker.generate_until_pause_or_complete(None, None))

    assert result.outcome is ElasticRunOutcome.COMPLETED
    assert worker.get_elastic_status().state is ElasticRankState.COMPLETED
    assert worker.predict_count == 2
    assert [item.transition_id.sequence for item in worker.sent_results] == [0, 1]
    assert worker.sent_results[0].versions is not None
    assert worker.sent_results[1].versions is None
    assert all(work.awaited for work in worker.send_works)


def test_elastic_rollout_accepts_early_final_bootstrap():
    requests = [
        ElasticRolloutRequest(
            kind=ElasticRolloutRequestKind.OBSERVATION,
            transition_id=_identity(sequence=sequence),
            logical_batch_size=1,
            env_output=_env_output(_identity(sequence=sequence)).to_dict(),
            final_bootstrap=sequence == 1,
        )
        for sequence in (0, 1)
    ]
    worker = _elastic_rollout_worker(requests)
    worker.n_train_chunk_steps = 4
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)

    result = asyncio.run(worker.generate_until_pause_or_complete(None, None))

    assert result.outcome is ElasticRunOutcome.COMPLETED
    assert worker.predict_count == 2
    assert worker.sent_results[0].versions is not None
    assert worker.sent_results[1].versions is None


def test_elastic_rollout_rejects_final_bootstrap_before_committed_chunk():
    request = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.OBSERVATION,
        transition_id=_identity(),
        logical_batch_size=1,
        env_output=_env_output(_identity()).to_dict(),
        final_bootstrap=True,
    )
    worker = _elastic_rollout_worker([request])
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)

    with pytest.raises(ValueError, match="at least one committed chunk"):
        asyncio.run(worker.generate_until_pause_or_complete(None, None))

    assert worker.predict_count == 0


def _completed_early_rollout_result() -> EmbodiedRolloutResult:
    result = EmbodiedRolloutResult(max_episode_length=4)
    batch_size = 2
    for step in range(2):
        done = torch.zeros((batch_size, 1), dtype=torch.bool)
        result.append_step_result(
            ChunkStepResult(
                actions=torch.full((batch_size, 2), float(step + 1)),
                rewards=torch.tensor([[float(step + 1)], [float(2 * (step + 1))]]),
                terminations=torch.zeros_like(done),
                truncations=torch.zeros_like(done),
                dones=torch.zeros_like(done),
                prev_logprobs=torch.full((batch_size, 1), 0.25),
                prev_values=torch.full((batch_size, 1), 0.5),
                versions=torch.full((batch_size, 1), 7.0),
                forward_inputs={"action": torch.full((batch_size, 2), 1.0)},
            )
        )
        result.append_transitions(
            {"states": torch.full((batch_size, 2), float(step))},
            {"states": torch.full((batch_size, 2), float(step + 1))},
        )
    result.append_step_result(
        ChunkStepResult(
            rewards=None,
            terminations=torch.ones((batch_size, 1), dtype=torch.bool),
            truncations=torch.zeros((batch_size, 1), dtype=torch.bool),
            dones=torch.ones((batch_size, 1), dtype=torch.bool),
            prev_values=torch.full((batch_size, 1), 0.5),
        )
    )
    return result


def test_completed_epoch_padding_preserves_fixed_shapes_and_policy_version():
    result = _completed_early_rollout_result()

    padded = result.pad_completed_epoch(
        completed_epoch_index=0,
        target_chunk_steps=4,
        policy_version=7,
    )
    trajectory = result.to_trajectory()

    assert padded == 2
    for name in (
        "actions",
        "intervene_flags",
        "rewards",
        "prev_logprobs",
        "versions",
    ):
        assert getattr(trajectory, name).shape[0] == 4
    for name in ("terminations", "truncations", "dones", "prev_values"):
        assert getattr(trajectory, name).shape[0] == 5
    assert trajectory.curr_obs["states"].shape[0] == 4
    assert trajectory.next_obs["states"].shape[0] == 4
    assert torch.all(trajectory.versions[2:] == 7)
    assert not trajectory.dones[3:].any()
    assert not trajectory.terminations[3:].any()
    assert torch.count_nonzero(trajectory.actions[2:]) == 0
    assert torch.count_nonzero(trajectory.rewards[2:]) == 0


def test_completed_epoch_padding_supports_openvla_results_without_actions():
    result = EmbodiedRolloutResult(max_episode_length=4)
    batch_size = 2
    sequence_length = 5
    for step in range(2):
        done = torch.zeros((batch_size, 1), dtype=torch.bool)
        result.append_step_result(
            ChunkStepResult(
                actions=None,
                rewards=torch.ones((batch_size, 1)),
                terminations=torch.zeros_like(done),
                truncations=torch.zeros_like(done),
                dones=done,
                prev_logprobs=torch.full((batch_size, 1), 0.25),
                prev_values=torch.full((batch_size, 1), 0.5),
                versions=torch.full((batch_size, 1), 7.0),
                forward_inputs={
                    "action_tokens": torch.full(
                        (batch_size, 1, 7), step + 10, dtype=torch.long
                    ),
                    "attention_mask": torch.ones(
                        (batch_size, sequence_length), dtype=torch.long
                    ),
                    "input_ids": torch.tensor(
                        [[1, 2, 3, 4, step + 5]] * batch_size, dtype=torch.long
                    ),
                    "pixel_values": torch.full((batch_size, 3, 2, 2), float(step + 1)),
                },
            )
        )
    result.append_step_result(
        ChunkStepResult(
            rewards=None,
            terminations=torch.ones((batch_size, 1), dtype=torch.bool),
            truncations=torch.zeros((batch_size, 1), dtype=torch.bool),
            dones=torch.ones((batch_size, 1), dtype=torch.bool),
            prev_values=torch.full((batch_size, 1), 0.5),
        )
    )

    padded = result.pad_completed_epoch(
        completed_epoch_index=0,
        target_chunk_steps=4,
        policy_version=7,
    )

    assert padded == 2
    assert result.actions == []
    assert result.intervene_flags == []
    assert len(result.forward_inputs) == 4
    assert (
        result.forward_inputs[2]["input_ids"]
        is not result.forward_inputs[1]["input_ids"]
    )
    assert (
        result.forward_inputs[3]["pixel_values"]
        is not result.forward_inputs[2]["pixel_values"]
    )
    assert (
        result.forward_inputs[2]["attention_mask"].data_ptr()
        != result.forward_inputs[1]["attention_mask"].data_ptr()
    )

    trajectory = result.to_trajectory()
    assert trajectory.actions is None
    assert trajectory.intervene_flags is None
    assert trajectory.forward_inputs["action_tokens"].shape[0] == 4
    assert trajectory.forward_inputs["attention_mask"][2:].all()
    assert trajectory.forward_inputs["input_ids"][2:, :, -1].ne(0).all()
    assert torch.count_nonzero(trajectory.rewards[2:]) == 0
    assert torch.count_nonzero(trajectory.prev_logprobs[2:]) == 0
    assert torch.all(trajectory.versions[2:] == 7)
    assert not compute_loss_mask(trajectory.dones)[0][2:].any()


def test_completed_epoch_padding_matches_regular_post_terminal_masking():
    early = _completed_early_rollout_result()
    early.pad_completed_epoch(
        completed_epoch_index=0,
        target_chunk_steps=4,
        policy_version=7,
    )
    padded = early.to_trajectory()
    regular_dones = padded.dones.clone()
    regular_rewards = padded.rewards.clone()
    regular_logprobs = padded.prev_logprobs.clone()
    regular_rewards[2:] = 99.0
    regular_logprobs[2:] = -37.0

    padded_mask, padded_count = compute_loss_mask(padded.dones)
    regular_mask, regular_count = compute_loss_mask(regular_dones)

    torch.testing.assert_close(padded_mask, regular_mask)
    torch.testing.assert_close(padded_count, regular_count)
    torch.testing.assert_close(
        padded.rewards * padded_mask,
        regular_rewards * regular_mask,
    )
    torch.testing.assert_close(
        padded.prev_logprobs * padded_mask,
        regular_logprobs * regular_mask,
    )
    padded_advantages = calculate_adv_and_returns(
        adv_type="grpo",
        task_type="embodied",
        reward_type="step_level",
        rewards=padded.rewards,
        dones=padded.dones,
        loss_mask=padded_mask,
        loss_mask_sum=padded_count,
        group_size=2,
    )["advantages"]
    regular_advantages = calculate_adv_and_returns(
        adv_type="grpo",
        task_type="embodied",
        reward_type="step_level",
        rewards=regular_rewards,
        dones=regular_dones,
        loss_mask=regular_mask,
        loss_mask_sum=regular_count,
        group_size=2,
    )["advantages"]
    torch.testing.assert_close(padded_advantages, regular_advantages)
    assert not padded_mask[2:].any()


def test_completed_epoch_padding_requires_every_trajectory_to_be_terminal():
    result = _completed_early_rollout_result()
    result.dones[-1][1] = False
    result.terminations[-1][1] = False

    with pytest.raises(ValueError, match="every trajectory"):
        result.pad_completed_epoch(
            completed_epoch_index=0,
            target_chunk_steps=4,
            policy_version=7,
        )


def test_environment_rank_early_completion_drives_final_bootstrap_and_padding():
    env_worker = _elastic_env_worker()
    _configure_single_chunk_env(env_worker)
    env_worker.n_train_chunk_steps = 3
    env_worker.stop_rank_when_all_done = True
    env_worker.cfg.env.train.auto_reset = False
    env_worker._prepare_rollout_results = lambda _previous: [
        EmbodiedRolloutResult(max_episode_length=3)
    ]
    bootstrap = EnvOutput(
        obs={"states": torch.zeros((1, 2))},
        dones=torch.zeros((1, 1), dtype=torch.bool),
        terminations=torch.zeros((1, 1), dtype=torch.bool),
        truncations=torch.zeros((1, 1), dtype=torch.bool),
    )
    env_worker.bootstrap_step = lambda: [bootstrap]
    step_calls = 0

    def successful_step(_actions, _stage_id):
        nonlocal step_calls
        step_calls += 1
        return (
            EnvOutput(
                obs={"states": torch.ones((1, 2))},
                dones=torch.ones((1, 1), dtype=torch.bool),
                terminations=torch.ones((1, 1), dtype=torch.bool),
                truncations=torch.zeros((1, 1), dtype=torch.bool),
                rewards=torch.ones((1, 1)),
            ),
            {},
            {},
        )

    env_worker.env_interact_step = successful_step
    env_worker.compute_bootstrap_rewards = lambda output, *_args: (
        None if output.rewards is None else output.rewards.clone()
    )
    env_worker.record_env_metrics = lambda *_args: None
    env_worker.store_last_obs_and_intervened_info = lambda _outputs: None
    env_worker.finish_rollout = lambda: None
    markers = []
    env_worker.log_info = markers.append

    rollout_worker = _elastic_rollout_worker([])
    rollout_worker.n_train_chunk_steps = 3

    async def run_case():
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        env_worker.send_to = lambda **kwargs: _QueuePutWork(
            env_to_rollout, kwargs["data"]
        )
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _QueueGetWork(env_to_rollout)
        rollout_worker.send_to = lambda **kwargs: _QueuePutWork(
            rollout_to_env, kwargs["data"]
        )
        env_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        return await asyncio.gather(
            rollout_worker.generate_until_pause_or_complete(None, None),
            env_worker.interact_until_pause_or_complete(None, None, None),
        )

    rollout_result, env_result = asyncio.run(run_case())

    assert rollout_result.outcome is ElasticRunOutcome.COMPLETED
    assert env_result.outcome is ElasticRunOutcome.COMPLETED
    assert step_calls == 1
    assert rollout_worker.predict_count == 2
    trajectory = env_worker.rollout_results[0].to_trajectory()
    assert trajectory.actions.shape[0] == 3
    assert trajectory.dones.shape[0] == 4
    assert not compute_loss_mask(trajectory.dones)[0][1:].any()
    assert env_worker._rollout_cursor.synthetic_padding_chunks == 2
    assert env_worker.get_elastic_progress().completed_trajectories == 1
    assert any("RLIX_TRAJECTORY_COMPLETED" in marker for marker in markers)
    assert any("RLIX_RANK_EARLY_FINALIZED" in marker for marker in markers)


def test_rank_completion_is_sticky_until_every_environment_succeeds():
    worker = _elastic_env_worker()
    worker.train_num_envs_per_stage = 2
    worker.train_batch_size = 2
    worker.train_prev_done = [torch.zeros(2, dtype=torch.bool)]
    worker.model_cfg = SimpleNamespace(num_action_chunks=8)
    worker.log_info = lambda message: markers.append(message)
    markers = []
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)

    first = EnvOutput(
        obs={"states": torch.zeros((2, 2))},
        terminations=torch.tensor(
            [
                [False, False, False, False, False, False, False, False],
                [False, False, False, True, False, False, False, False],
            ]
        ),
    )
    second = EnvOutput(
        obs={"states": torch.zeros((2, 2))},
        terminations=torch.tensor(
            [
                [False, True, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False],
            ]
        ),
    )

    assert not worker._record_rank_trajectory_completions(
        first, stage_id=0, epoch=0, chunk_step_idx=2
    )
    assert worker.get_elastic_progress().completed_trajectories == 1
    assert worker._record_rank_trajectory_completions(
        second, stage_id=0, epoch=0, chunk_step_idx=4
    )
    assert worker.train_prev_done[0].tolist() == [True, True]
    assert worker.get_elastic_progress().completed_trajectories == 2
    assert len(markers) == 2
    assert "env_index=1" in markers[0] and "step=20" in markers[0]
    assert "env_index=0" in markers[1] and "step=34" in markers[1]


def test_elastic_rollout_rejects_stale_observation_before_inference():
    stale = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.OBSERVATION,
        transition_id=_identity(sequence=1),
        logical_batch_size=1,
        env_output=_env_output(_identity(sequence=1)).to_dict(),
    )
    worker = _elastic_rollout_worker([stale])
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)

    with pytest.raises(ValueError, match="expected transition"):
        asyncio.run(worker.generate_until_pause_or_complete(None, None))

    assert worker.predict_count == 0
    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT


def test_environment_activation_and_drain_are_versioned_and_idempotent():
    worker = _elastic_env_worker()

    status = worker.prepare_elastic_collection(
        lifecycle_generation=1, expected_policy_version=3
    )

    assert status.state is ElasticRankState.EXPANDING
    assert status.expected_transition_id == _identity()
    assert worker.env_list[0].resident

    worker._elastic_state = ElasticRankState.ACTIVE
    request = DrainRequest("drain-1", 0, 1, 3)
    first = asyncio.run(worker.request_elastic_drain(request))
    second = asyncio.run(worker.request_elastic_drain(request))

    assert first.state is ElasticRankState.DRAIN_REQUESTED
    assert second.drain_request_id == "drain-1"


def test_environment_sends_identified_observation_with_awaited_route_work():
    worker = _elastic_env_worker()
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)
    env_output = _env_output(None)
    worker._assign_elastic_transition_identity(env_output, 0)

    asyncio.run(worker._send_elastic_observation(None, env_output))

    assert len(worker.sent_requests) == 1
    request = worker.sent_requests[0]
    assert request.kind is ElasticRolloutRequestKind.OBSERVATION
    assert request.transition_id == _identity()
    assert request.logical_batch_size == 1
    assert all(work.awaited for work in worker.send_works)


def test_environment_sends_barrier_with_local_not_global_batch_size():
    worker = _elastic_env_worker(world_size=2)
    worker.train_batch_size = 4
    worker.train_num_envs_per_stage = 2
    token = SafePointToken("drain-1", 0, 1, 3, _identity(sequence=1))

    asyncio.run(worker._send_elastic_barrier(None, token))

    assert len(worker.sent_requests) == 1
    request = worker.sent_requests[0]
    assert request.kind is ElasticRolloutRequestKind.DRAIN_BARRIER
    assert request.logical_batch_size == 2
    assert all(work.awaited for work in worker.send_works)


def test_environment_rejects_stale_rollout_result_before_mutation():
    worker = _elastic_env_worker()
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)
    current_output = _env_output(_identity())
    worker._current_env_outputs = [current_output]
    stale_result = RolloutResult(
        actions=torch.ones((1, 2)), transition_id=_identity(sequence=1)
    )
    worker.recv_from = lambda **_kwargs: _ImmediateWork(stale_result)

    with pytest.raises(ValueError, match="in-flight transition"):
        asyncio.run(worker._recv_elastic_rollout_result(None, 0))


def test_environment_drains_only_after_committing_current_chunk():
    worker = _elastic_env_worker()
    worker.rollout_epoch = 1
    worker.n_train_chunk_steps = 1
    worker.collect_prev_infos = True
    worker.collect_transitions = False
    worker.reward_mode = "per_step"
    worker.history_reward_assign = False
    worker.use_training_pipeline = False
    worker.enable_online_lerobot = False
    worker.enable_rlt = False
    worker.model_cfg = SimpleNamespace(num_action_chunks=1)
    worker._rollout_call_active = False
    worker._policy_request_in_flight = False
    worker._prefetched_train_bootstrap = None
    worker._rlt_pending_obs = [None]
    worker._resume_bootstraps = [None]
    worker._current_env_outputs = None
    worker._rollout_env_metrics = {}
    worker.last_obs_list = [{}]
    worker.last_intervened_info_list = [(None, None)]
    worker._prepare_rollout_results = lambda _previous: [_FakePartialRollout()]
    worker.bootstrap_step = lambda: [_env_output(None)]
    events = []

    def interact_step(_actions, _stage_id):
        events.append("chunk_committed")
        return _env_output(None), {}, {}

    worker.env_interact_step = interact_step
    worker.compute_bootstrap_rewards = lambda *_args: torch.zeros((1, 1))
    worker.snapshot_rollout_stage = lambda: events.append("snapshot") or object()
    rollout_result = RolloutResult(
        actions=torch.ones((1, 2)),
        prev_logprobs=torch.zeros((1, 2)),
        prev_values=torch.zeros((1, 1)),
        forward_inputs={"action": torch.ones((1, 2))},
        versions=torch.full((1, 2), 3.0),
        transition_id=_identity(),
    )
    worker.recv_from = lambda **_kwargs: _ImmediateWork(rollout_result)
    original_send = worker.send_to

    def send_to(**kwargs):
        request = kwargs["data"]
        events.append(request.kind.value)
        return original_send(**kwargs)

    worker.send_to = send_to
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.ACTIVE
    asyncio.run(worker.request_elastic_drain(DrainRequest("drain-1", 0, 1, 3)))

    result = asyncio.run(worker.interact_until_pause_or_complete(None, None, None))

    assert result.outcome is ElasticRunOutcome.PAUSE_READY
    assert result.token.next_transition_id == _identity(sequence=1)
    assert events == [
        "observation",
        "chunk_committed",
        "drain_barrier",
        "snapshot",
    ]
    assert worker._resume_bootstraps[0] is not None
    assert worker._resume_bootstraps[0].transition_id == _identity(sequence=1)


def test_environment_offload_and_resume_preserve_pending_bootstrap():
    worker = _elastic_env_worker()
    worker.torch_platform = _FakePlatform()
    worker.env_list[0].resident = True
    worker._environment_resident = True
    worker._elastic_state = ElasticRankState.SNAPSHOTTING
    worker._rollout_call_active = False
    worker._policy_request_in_flight = False
    pending = _env_output(_identity(sequence=1))
    worker._rollout_cursor = EnvRolloutCursor(
        schema_version=ENV_ROLLOUT_RESUME_SCHEMA_VERSION,
        lifecycle_generation=1,
        policy_version=3,
        epoch_index=0,
        chunk_index=1,
        stage_id=0,
        next_transition_ids=(1,),
        phase=RolloutCursorPhase.BOOTSTRAP_PENDING,
    )
    worker._resume_bootstraps = [pending]
    token = SafePointToken("drain-1", 0, 1, 3, _identity(sequence=1))
    worker._elastic_safe_point_token = token
    worker._elastic_drain_request = DrainRequest("drain-1", 0, 1, 3)
    worker._elastic_resume_state = _FakeEnvResumeState((pending,))
    worker.validate_rollout_resume_state = lambda *_args, **_kwargs: None
    worker.restore_rollout_stage = lambda *_args, **_kwargs: None

    pause_receipt = worker.offload_elastic_environment(token)

    assert pause_receipt.state is ElasticRankState.PAUSED
    assert not worker.env_list[0].resident

    resume_receipt = worker.prepare_elastic_resume(token)

    assert resume_receipt.state is ElasticRankState.EXPANDING
    assert worker.env_list[0].resident
    assert worker._resume_bootstraps[0].transition_id == token.next_transition_id
    assert worker._elastic_drain_request is None
    assert worker._elastic_safe_point_token is None


def test_paired_workers_produce_the_same_safe_point_token():
    env_worker = _elastic_env_worker()
    env_worker.rollout_epoch = 1
    env_worker.n_train_chunk_steps = 1
    env_worker.collect_prev_infos = True
    env_worker.collect_transitions = False
    env_worker.reward_mode = "per_step"
    env_worker.history_reward_assign = False
    env_worker.use_training_pipeline = False
    env_worker.enable_online_lerobot = False
    env_worker.enable_rlt = False
    env_worker.model_cfg = SimpleNamespace(num_action_chunks=1)
    env_worker._rollout_call_active = False
    env_worker._policy_request_in_flight = False
    env_worker._prefetched_train_bootstrap = None
    env_worker._rlt_pending_obs = [None]
    env_worker._resume_bootstraps = [None]
    env_worker._current_env_outputs = None
    env_worker._rollout_env_metrics = {}
    env_worker.last_obs_list = [{}]
    env_worker.last_intervened_info_list = [(None, None)]
    env_worker._prepare_rollout_results = lambda _previous: [_FakePartialRollout()]
    env_worker.bootstrap_step = lambda: [_env_output(None)]
    env_worker.env_interact_step = lambda _actions, _stage_id: (
        _env_output(None),
        {},
        {},
    )
    env_worker.compute_bootstrap_rewards = lambda *_args: torch.zeros((1, 1))
    env_worker.snapshot_rollout_stage = lambda: object()
    env_worker.store_last_obs_and_intervened_info = lambda _outputs: None
    env_worker.finish_rollout = lambda: None
    env_worker.torch_platform = _FakePlatform()
    rollout_worker = _elastic_rollout_worker([])
    rollout_worker.hf_model = _FakeResidentModel()
    rollout_worker.expert_model = None
    rollout_worker.rlt_feature_model = None
    rollout_worker.device = "cuda"
    rollout_worker.torch_platform = _FakePlatform()

    async def run_pair():
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        env_worker.send_to = lambda **kwargs: _QueuePutWork(
            env_to_rollout, kwargs["data"]
        )
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _QueueGetWork(env_to_rollout)
        rollout_worker.send_to = lambda **kwargs: _QueuePutWork(
            rollout_to_env, kwargs["data"]
        )
        env_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        env_worker._elastic_state = ElasticRankState.ACTIVE
        rollout_worker._elastic_state = ElasticRankState.ACTIVE
        drain = DrainRequest("drain-1", 0, 1, 3)
        await asyncio.gather(
            env_worker.request_elastic_drain(drain),
            rollout_worker.request_elastic_drain(drain),
        )
        rollout_run, env_run = await asyncio.gather(
            rollout_worker.generate_until_pause_or_complete(None, None),
            env_worker.interact_until_pause_or_complete(None, None, None),
        )
        token = env_run.token
        env_worker._elastic_resume_state = _FakeEnvResumeState(
            (env_worker._resume_bootstraps[0],)
        )
        env_worker.validate_rollout_resume_state = lambda *_args, **_kwargs: None
        env_worker.restore_rollout_stage = lambda *_args, **_kwargs: None
        env_worker.offload_elastic_environment(token)
        rollout_worker.offload_elastic_rollout(token)
        env_worker.prepare_elastic_resume(token)
        rollout_worker.prepare_elastic_resume(token)
        rollout_complete, env_complete = await asyncio.gather(
            rollout_worker.generate_until_pause_or_complete(None, None),
            env_worker.interact_until_pause_or_complete(None, None, None),
        )
        return (
            rollout_run,
            env_run,
            rollout_complete,
            env_complete,
            env_to_rollout,
            rollout_to_env,
        )

    (
        rollout_run,
        env_run,
        rollout_complete,
        env_complete,
        env_to_rollout,
        rollout_to_env,
    ) = asyncio.run(run_pair())

    assert rollout_run.outcome is ElasticRunOutcome.PAUSE_READY
    assert env_run.outcome is ElasticRunOutcome.PAUSE_READY
    assert rollout_run.token == env_run.token
    assert rollout_run.token.next_transition_id == _identity(sequence=1)
    assert rollout_complete.outcome is ElasticRunOutcome.COMPLETED
    assert env_complete.outcome is ElasticRunOutcome.COMPLETED
    progress = env_worker.get_elastic_progress()
    assert progress.completed_trajectories == progress.assigned_trajectories == 1
    assert progress.state is ElasticRankState.COMPLETED
    assert rollout_worker.predict_count == 2
    assert env_to_rollout.empty()
    assert rollout_to_env.empty()


def test_snapshot_failure_after_barrier_fails_environment_resident():
    worker = _elastic_env_worker()

    def fail_snapshot():
        raise RuntimeError("snapshot failed")

    events, _ = _configure_single_chunk_env(worker, snapshot_factory=fail_snapshot)
    rollout_result = RolloutResult(
        actions=torch.ones((1, 2)),
        prev_logprobs=torch.zeros((1, 2)),
        prev_values=torch.zeros((1, 1)),
        forward_inputs={"action": torch.ones((1, 2))},
        versions=torch.full((1, 2), 3.0),
        transition_id=_identity(),
    )
    worker.recv_from = lambda **_kwargs: _ImmediateWork(rollout_result)
    original_send = worker.send_to

    def send_to(**kwargs):
        events.append(kwargs["data"].kind.value)
        return original_send(**kwargs)

    worker.send_to = send_to
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.ACTIVE
    asyncio.run(worker.request_elastic_drain(DrainRequest("drain-1", 0, 1, 3)))

    with pytest.raises(RuntimeError, match="snapshot failed"):
        asyncio.run(worker.interact_until_pause_or_complete(None, None, None))

    assert events == ["observation", "chunk_committed", "drain_barrier", "snapshot"]
    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT
    assert worker.get_elastic_status().model_resident


def test_partial_pair_offload_failure_does_not_rollback_paused_peer():
    identity = _identity(sequence=1)
    token = SafePointToken("drain-1", 0, 1, 3, identity)
    env_worker = _elastic_env_worker()
    env_worker.torch_platform = _FakePlatform()
    env_worker.env_list[0].resident = True
    env_worker._environment_resident = True
    env_worker._elastic_state = ElasticRankState.SNAPSHOTTING
    env_worker._rollout_cursor = EnvRolloutCursor(
        schema_version=ENV_ROLLOUT_RESUME_SCHEMA_VERSION,
        lifecycle_generation=1,
        policy_version=3,
        epoch_index=0,
        chunk_index=1,
        stage_id=0,
        next_transition_ids=(1,),
        phase=RolloutCursorPhase.BOOTSTRAP_PENDING,
    )
    pending = _env_output(identity)
    env_worker._resume_bootstraps = [pending]
    env_worker._elastic_safe_point_token = token
    env_worker._elastic_resume_state = _FakeEnvResumeState((pending,))
    env_worker.validate_rollout_resume_state = lambda *_args, **_kwargs: None
    env_worker.offload_elastic_environment(token)

    rollout_worker = _elastic_rollout_worker([])
    rollout_worker._elastic_state = ElasticRankState.SNAPSHOTTING
    rollout_worker._elastic_safe_point_token = token
    rollout_worker._elastic_cursor = RolloutPeerCursor(
        lifecycle_generation=1,
        policy_version=3,
        epoch_index=0,
        committed_chunk_count=1,
        expected_transition_id=identity,
        phase=RolloutPeerPhase.PAUSE_READY,
    )
    rollout_worker.hf_model = _FakeResidentModel()
    rollout_worker.expert_model = None
    rollout_worker.rlt_feature_model = None
    rollout_worker.torch_platform = _FakePlatform()
    rollout_worker.hf_model.to = lambda _device: (_ for _ in ()).throw(
        RuntimeError("rollout offload failed")
    )

    with pytest.raises(RuntimeError, match="rollout offload failed"):
        rollout_worker.offload_elastic_rollout(token)

    assert env_worker.get_elastic_status().state is ElasticRankState.PAUSED
    assert not env_worker.get_elastic_status().model_resident
    assert rollout_worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT


def test_environment_offload_failure_is_fail_closed():
    identity = _identity(sequence=1)
    token = SafePointToken("drain-1", 0, 1, 3, identity)
    worker = _elastic_env_worker()
    worker._elastic_state = ElasticRankState.SNAPSHOTTING
    worker._environment_resident = True
    worker.env_list[0].resident = True
    worker._rollout_cursor = EnvRolloutCursor(
        schema_version=ENV_ROLLOUT_RESUME_SCHEMA_VERSION,
        lifecycle_generation=1,
        policy_version=3,
        epoch_index=0,
        chunk_index=1,
        stage_id=0,
        next_transition_ids=(1,),
        phase=RolloutCursorPhase.BOOTSTRAP_PENDING,
    )
    pending = _env_output(identity)
    worker._resume_bootstraps = [pending]
    worker._elastic_safe_point_token = token
    worker._elastic_resume_state = _FakeEnvResumeState((pending,))
    worker.validate_rollout_resume_state = lambda *_args, **_kwargs: None
    worker.env_list[0].offload = lambda: (_ for _ in ()).throw(
        RuntimeError("environment offload failed")
    )

    with pytest.raises(RuntimeError, match="environment offload failed"):
        worker.offload_elastic_environment(token)

    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT
    assert worker.get_elastic_status().model_resident


def test_rollout_resume_failure_is_fail_closed():
    identity = _identity(sequence=1)
    token = SafePointToken("drain-1", 0, 1, 3, identity)
    worker = _elastic_rollout_worker([])
    worker._elastic_state = ElasticRankState.PAUSED
    worker._elastic_safe_point_token = token
    worker._elastic_cursor = RolloutPeerCursor(
        lifecycle_generation=1,
        policy_version=3,
        epoch_index=0,
        committed_chunk_count=1,
        expected_transition_id=identity,
        phase=RolloutPeerPhase.PAUSE_READY,
    )
    worker.hf_model = _FakeResidentModel(device="cpu")
    worker.expert_model = None
    worker.rlt_feature_model = None
    worker.device = "cuda"
    worker.torch_platform = _FakePlatform()
    worker.hf_model.to = lambda _device: (_ for _ in ()).throw(
        RuntimeError("rollout onload failed")
    )

    with pytest.raises(RuntimeError, match="rollout onload failed"):
        worker.prepare_elastic_resume(token)

    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT


def test_environment_restore_failure_after_onload_is_fail_closed():
    identity = _identity(sequence=1)
    token = SafePointToken("drain-1", 0, 1, 3, identity)
    worker = _elastic_env_worker()
    worker._elastic_state = ElasticRankState.PAUSED
    worker._elastic_safe_point_token = token
    worker._environment_resident = False
    worker.env_list[0].resident = False
    worker._rollout_cursor = EnvRolloutCursor(
        schema_version=ENV_ROLLOUT_RESUME_SCHEMA_VERSION,
        lifecycle_generation=1,
        policy_version=3,
        epoch_index=0,
        chunk_index=1,
        stage_id=0,
        next_transition_ids=(1,),
        phase=RolloutCursorPhase.BOOTSTRAP_PENDING,
    )
    pending = _env_output(identity)
    worker._resume_bootstraps = [pending]
    worker._elastic_resume_state = _FakeEnvResumeState((pending,))
    worker.validate_rollout_resume_state = lambda *_args, **_kwargs: None
    worker.restore_rollout_stage = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("environment restore failed")
    )

    with pytest.raises(RuntimeError, match="environment restore failed"):
        worker.prepare_elastic_resume(token)

    assert worker.env_list[0].resident
    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT


def test_completed_environment_requires_a_newer_lifecycle():
    worker = _elastic_env_worker()
    worker.prepare_elastic_collection(lifecycle_generation=2, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.COMPLETED

    with pytest.raises(ValueError, match="newer lifecycle"):
        worker.prepare_elastic_collection(
            lifecycle_generation=2, expected_policy_version=3
        )

    status = worker.prepare_elastic_collection(
        lifecycle_generation=3, expected_policy_version=3
    )
    assert status.state is ElasticRankState.EXPANDING
    assert status.lifecycle_generation == 3


def test_selected_rank_drains_while_sibling_pair_completes():
    env_workers = [_elastic_env_worker(rank=rank, world_size=2) for rank in range(2)]
    rollout_workers = [
        _elastic_rollout_worker([], rank=rank, world_size=2) for rank in range(2)
    ]
    for worker in env_workers:
        _configure_single_chunk_env(worker)
        worker.torch_platform = _FakePlatform()
    rollout_workers[0].hf_model = _FakeResidentModel()
    rollout_workers[0].expert_model = None
    rollout_workers[0].rlt_feature_model = None
    rollout_workers[0].device = "cuda"
    rollout_workers[0].torch_platform = _FakePlatform()

    async def run_pairs():
        env_to_rollout = [asyncio.Queue(), asyncio.Queue()]
        rollout_to_env = [asyncio.Queue(), asyncio.Queue()]
        request_kinds = [[], []]
        for rank in range(2):
            env_workers[rank].send_to = (
                lambda rank: (
                    lambda **kwargs: (
                        request_kinds[rank].append(kwargs["data"].kind),
                        _QueuePutWork(env_to_rollout[rank], kwargs["data"]),
                    )[1]
                )
            )(rank)
            env_workers[rank].recv_from = (
                lambda rank: lambda **_kwargs: _QueueGetWork(rollout_to_env[rank])
            )(rank)
            rollout_workers[rank].recv_from = (
                lambda rank: lambda **_kwargs: _QueueGetWork(env_to_rollout[rank])
            )(rank)
            rollout_workers[rank].send_to = (
                lambda rank: (
                    lambda **kwargs: _QueuePutWork(rollout_to_env[rank], kwargs["data"])
                )
            )(rank)
            env_workers[rank].prepare_elastic_collection(
                lifecycle_generation=1, expected_policy_version=3
            )
            rollout_workers[rank].prepare_elastic_collection(
                lifecycle_generation=1, expected_policy_version=3
            )

        env_workers[0]._elastic_state = ElasticRankState.ACTIVE
        rollout_workers[0]._elastic_state = ElasticRankState.ACTIVE
        drain = DrainRequest("rank-0-drain", 0, 1, 3)
        await asyncio.gather(
            env_workers[0].request_elastic_drain(drain),
            rollout_workers[0].request_elastic_drain(drain),
        )
        results = await asyncio.gather(
            rollout_workers[0].generate_until_pause_or_complete(None, None),
            env_workers[0].interact_until_pause_or_complete(None, None, None),
            rollout_workers[1].generate_until_pause_or_complete(None, None),
            env_workers[1].interact_until_pause_or_complete(None, None, None),
        )
        token = results[1].token
        env_workers[0]._elastic_resume_state = _FakeEnvResumeState(
            (env_workers[0]._resume_bootstraps[0],)
        )
        env_workers[0].validate_rollout_resume_state = lambda *_args, **_kwargs: None
        env_workers[0].restore_rollout_stage = lambda *_args, **_kwargs: None
        env_workers[0].offload_elastic_environment(token)
        rollout_workers[0].offload_elastic_rollout(token)
        env_workers[0].prepare_elastic_resume(token)
        rollout_workers[0].prepare_elastic_resume(token)
        rank_zero_completion = await asyncio.gather(
            rollout_workers[0].generate_until_pause_or_complete(None, None),
            env_workers[0].interact_until_pause_or_complete(None, None, None),
        )
        return (
            results,
            rank_zero_completion,
            request_kinds,
            env_to_rollout,
            rollout_to_env,
        )

    results, rank_zero_completion, request_kinds, env_to_rollout, rollout_to_env = (
        asyncio.run(run_pairs())
    )

    assert results[0].outcome is ElasticRunOutcome.PAUSE_READY
    assert results[1].outcome is ElasticRunOutcome.PAUSE_READY
    assert results[2].outcome is ElasticRunOutcome.COMPLETED
    assert results[3].outcome is ElasticRunOutcome.COMPLETED
    assert all(
        result.outcome is ElasticRunOutcome.COMPLETED for result in rank_zero_completion
    )
    assert ElasticRolloutRequestKind.DRAIN_BARRIER in request_kinds[0]
    assert ElasticRolloutRequestKind.DRAIN_BARRIER not in request_kinds[1]
    assert rollout_workers[1].predict_count == 2
    assert rollout_workers[0].predict_count == 2
    assert all(queue.empty() for queue in (*env_to_rollout, *rollout_to_env))


def test_pause_resume_matches_uninterrupted_transition_and_trajectory_order():
    async def execute(*, pause):
        env_worker = _elastic_env_worker()
        _, partial_rollout = _configure_single_chunk_env(env_worker)
        env_worker.torch_platform = _FakePlatform()
        rollout_worker = _elastic_rollout_worker([])
        rollout_worker.hf_model = _FakeResidentModel()
        rollout_worker.expert_model = None
        rollout_worker.rlt_feature_model = None
        rollout_worker.device = "cuda"
        rollout_worker.torch_platform = _FakePlatform()
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        result_identities = []
        env_worker.send_to = lambda **kwargs: _QueuePutWork(
            env_to_rollout, kwargs["data"]
        )
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _QueueGetWork(env_to_rollout)

        def send_result(**kwargs):
            result_identities.append(kwargs["data"].transition_id)
            return _QueuePutWork(rollout_to_env, kwargs["data"])

        rollout_worker.send_to = send_result
        env_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        if pause:
            env_worker._elastic_state = ElasticRankState.ACTIVE
            rollout_worker._elastic_state = ElasticRankState.ACTIVE
            drain = DrainRequest("drain-1", 0, 1, 3)
            await asyncio.gather(
                env_worker.request_elastic_drain(drain),
                rollout_worker.request_elastic_drain(drain),
            )
            rollout_pause, env_pause = await asyncio.gather(
                rollout_worker.generate_until_pause_or_complete(None, None),
                env_worker.interact_until_pause_or_complete(None, None, None),
            )
            assert rollout_pause.token == env_pause.token
            token = env_pause.token
            env_worker._elastic_resume_state = _FakeEnvResumeState(
                (env_worker._resume_bootstraps[0],)
            )
            env_worker.validate_rollout_resume_state = lambda *_args, **_kwargs: None
            env_worker.restore_rollout_stage = lambda *_args, **_kwargs: None
            env_worker.offload_elastic_environment(token)
            rollout_worker.offload_elastic_rollout(token)
            env_worker.prepare_elastic_resume(token)
            rollout_worker.prepare_elastic_resume(token)

        rollout_complete, env_complete = await asyncio.gather(
            rollout_worker.generate_until_pause_or_complete(None, None),
            env_worker.interact_until_pause_or_complete(None, None, None),
        )
        assert rollout_complete.outcome is ElasticRunOutcome.COMPLETED
        assert env_complete.outcome is ElasticRunOutcome.COMPLETED
        return rollout_worker, partial_rollout, result_identities

    uninterrupted = asyncio.run(execute(pause=False))
    resumed = asyncio.run(execute(pause=True))

    uninterrupted_worker, uninterrupted_rollout, uninterrupted_ids = uninterrupted
    resumed_worker, resumed_rollout, resumed_ids = resumed
    assert uninterrupted_ids == resumed_ids == [_identity(), _identity(sequence=1)]
    assert uninterrupted_worker.predict_count == resumed_worker.predict_count == 2
    assert len(uninterrupted_rollout.steps) == len(resumed_rollout.steps) == 2
    for uninterrupted_step, resumed_step in zip(
        uninterrupted_rollout.steps, resumed_rollout.steps
    ):
        for field_name in (
            "actions",
            "prev_logprobs",
            "prev_values",
            "rewards",
            "dones",
            "terminations",
            "truncations",
            "versions",
        ):
            uninterrupted_value = getattr(uninterrupted_step, field_name)
            resumed_value = getattr(resumed_step, field_name)
            if uninterrupted_value is None:
                assert resumed_value is None
            else:
                torch.testing.assert_close(uninterrupted_value, resumed_value)


def test_drain_after_next_observation_send_finishes_that_exchange():
    env_worker = _elastic_env_worker()
    _configure_single_chunk_env(env_worker)
    env_worker.n_train_chunk_steps = 2
    rollout_worker = _elastic_rollout_worker([])
    rollout_worker.n_train_chunk_steps = 2

    async def run_timing_case():
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        second_receive_started = asyncio.Event()
        release_second_receive = asyncio.Event()
        receive_count = 0

        class _GatedRolloutReceive:
            async def async_wait(self):
                nonlocal receive_count
                receive_count += 1
                if receive_count == 2:
                    second_receive_started.set()
                    await release_second_receive.wait()
                return await env_to_rollout.get()

        env_worker.send_to = lambda **kwargs: _QueuePutWork(
            env_to_rollout, kwargs["data"]
        )
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _GatedRolloutReceive()
        rollout_worker.send_to = lambda **kwargs: _QueuePutWork(
            rollout_to_env, kwargs["data"]
        )
        env_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_task = asyncio.create_task(
            rollout_worker.generate_until_pause_or_complete(None, None)
        )
        env_task = asyncio.create_task(
            env_worker.interact_until_pause_or_complete(None, None, None)
        )
        await second_receive_started.wait()
        assert not env_to_rollout.empty()
        drain = DrainRequest("late-drain", 0, 1, 3)
        await asyncio.gather(
            env_worker.request_elastic_drain(drain),
            rollout_worker.request_elastic_drain(drain),
        )
        release_second_receive.set()
        return await asyncio.gather(rollout_task, env_task)

    rollout_result, env_result = asyncio.run(run_timing_case())

    assert rollout_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert env_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert rollout_result.token == env_result.token
    assert rollout_result.token.next_transition_id == _identity(sequence=2)
    assert rollout_worker.predict_count == 2


def test_drain_queued_during_prediction_pauses_after_current_exchange():
    env_worker = _elastic_env_worker()
    _configure_single_chunk_env(env_worker)
    rollout_worker = _elastic_rollout_worker([])
    original_predict = rollout_worker._predict_rollout_actions

    async def run_timing_case():
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        drain_tasks = []

        def predict(obs, **kwargs):
            drain = DrainRequest("during-prediction", 0, 1, 3)
            drain_tasks.extend(
                (
                    asyncio.create_task(env_worker.request_elastic_drain(drain)),
                    asyncio.create_task(rollout_worker.request_elastic_drain(drain)),
                )
            )
            assert env_worker._elastic_state is ElasticRankState.ACTIVE
            assert rollout_worker._elastic_state is ElasticRankState.ACTIVE
            return original_predict(obs, **kwargs)

        env_worker.send_to = lambda **kwargs: _QueuePutWork(
            env_to_rollout, kwargs["data"]
        )
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _QueueGetWork(env_to_rollout)
        rollout_worker.send_to = lambda **kwargs: _QueuePutWork(
            rollout_to_env, kwargs["data"]
        )
        rollout_worker._predict_rollout_actions = predict
        env_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        results = await asyncio.gather(
            rollout_worker.generate_until_pause_or_complete(None, None),
            env_worker.interact_until_pause_or_complete(None, None, None),
        )
        await asyncio.gather(*drain_tasks)
        return results

    rollout_result, env_result = asyncio.run(run_timing_case())

    assert rollout_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert env_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert rollout_result.token == env_result.token
    assert rollout_result.token.next_transition_id == _identity(sequence=1)
    assert rollout_worker.predict_count == 1


def test_drain_queued_during_chunk_step_pauses_at_that_commit_boundary():
    env_worker = _elastic_env_worker()
    _configure_single_chunk_env(env_worker)
    rollout_worker = _elastic_rollout_worker([])
    original_interact_step = env_worker.env_interact_step

    async def run_timing_case():
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        drain_tasks = []

        def interact_step(actions, stage_id):
            drain = DrainRequest("during-chunk", 0, 1, 3)
            drain_tasks.extend(
                (
                    asyncio.create_task(env_worker.request_elastic_drain(drain)),
                    asyncio.create_task(rollout_worker.request_elastic_drain(drain)),
                )
            )
            assert env_worker._elastic_state is ElasticRankState.ACTIVE
            assert rollout_worker._elastic_state is ElasticRankState.ACTIVE
            return original_interact_step(actions, stage_id)

        env_worker.send_to = lambda **kwargs: _QueuePutWork(
            env_to_rollout, kwargs["data"]
        )
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _QueueGetWork(env_to_rollout)
        rollout_worker.send_to = lambda **kwargs: _QueuePutWork(
            rollout_to_env, kwargs["data"]
        )
        env_worker.env_interact_step = interact_step
        env_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        results = await asyncio.gather(
            rollout_worker.generate_until_pause_or_complete(None, None),
            env_worker.interact_until_pause_or_complete(None, None, None),
        )
        await asyncio.gather(*drain_tasks)
        return results

    rollout_result, env_result = asyncio.run(run_timing_case())

    assert rollout_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert env_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert rollout_result.token == env_result.token
    assert rollout_result.token.next_transition_id == _identity(sequence=1)
    assert rollout_worker.predict_count == 1


def test_drain_while_rollout_waits_for_observation_pauses_after_exchange():
    env_worker = _elastic_env_worker()
    _configure_single_chunk_env(env_worker)
    rollout_worker = _elastic_rollout_worker([])

    async def run_timing_case():
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        rollout_receive_started = asyncio.Event()
        env_send_started = asyncio.Event()
        release_env_send = asyncio.Event()

        class _ObservedRolloutReceive:
            async def async_wait(self):
                rollout_receive_started.set()
                return await env_to_rollout.get()

        class _GatedEnvSend:
            def __init__(self, value):
                self.value = value

            async def async_wait(self):
                env_send_started.set()
                await release_env_send.wait()
                await env_to_rollout.put(self.value)

        env_worker.send_to = lambda **kwargs: _GatedEnvSend(kwargs["data"])
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _ObservedRolloutReceive()
        rollout_worker.send_to = lambda **kwargs: _QueuePutWork(
            rollout_to_env, kwargs["data"]
        )
        env_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_task = asyncio.create_task(
            rollout_worker.generate_until_pause_or_complete(None, None)
        )
        env_task = asyncio.create_task(
            env_worker.interact_until_pause_or_complete(None, None, None)
        )
        await asyncio.gather(rollout_receive_started.wait(), env_send_started.wait())
        drain = DrainRequest("waiting-observation", 0, 1, 3)
        await asyncio.gather(
            env_worker.request_elastic_drain(drain),
            rollout_worker.request_elastic_drain(drain),
        )
        release_env_send.set()
        return await asyncio.gather(rollout_task, env_task)

    rollout_result, env_result = asyncio.run(run_timing_case())

    assert rollout_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert env_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert rollout_result.token == env_result.token
    assert rollout_result.token.next_transition_id == _identity(sequence=1)
    assert rollout_worker.predict_count == 1


def test_drain_while_environment_waits_for_result_pauses_after_exchange():
    env_worker = _elastic_env_worker()
    _configure_single_chunk_env(env_worker)
    rollout_worker = _elastic_rollout_worker([])

    async def run_timing_case():
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        rollout_received_observation = asyncio.Event()
        release_rollout_receive = asyncio.Event()

        class _GatedRolloutReceive:
            async def async_wait(self):
                request = await env_to_rollout.get()
                rollout_received_observation.set()
                await release_rollout_receive.wait()
                return request

        env_worker.send_to = lambda **kwargs: _QueuePutWork(
            env_to_rollout, kwargs["data"]
        )
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _GatedRolloutReceive()
        rollout_worker.send_to = lambda **kwargs: _QueuePutWork(
            rollout_to_env, kwargs["data"]
        )
        env_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_task = asyncio.create_task(
            rollout_worker.generate_until_pause_or_complete(None, None)
        )
        env_task = asyncio.create_task(
            env_worker.interact_until_pause_or_complete(None, None, None)
        )
        await rollout_received_observation.wait()
        assert env_worker._policy_request_in_flight
        drain = DrainRequest("waiting-result", 0, 1, 3)
        await asyncio.gather(
            env_worker.request_elastic_drain(drain),
            rollout_worker.request_elastic_drain(drain),
        )
        release_rollout_receive.set()
        return await asyncio.gather(rollout_task, env_task)

    rollout_result, env_result = asyncio.run(run_timing_case())

    assert rollout_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert env_result.outcome is ElasticRunOutcome.PAUSE_READY
    assert rollout_result.token == env_result.token
    assert rollout_result.token.next_transition_id == _identity(sequence=1)
    assert rollout_worker.predict_count == 1


def test_drain_during_final_bootstrap_completes_instead_of_pausing():
    env_worker = _elastic_env_worker()
    _configure_single_chunk_env(env_worker)
    rollout_worker = _elastic_rollout_worker([])

    async def run_timing_case():
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        final_receive_started = asyncio.Event()
        release_final_receive = asyncio.Event()
        receive_count = 0

        class _GatedFinalRolloutReceive:
            async def async_wait(self):
                nonlocal receive_count
                receive_count += 1
                if receive_count == 2:
                    final_receive_started.set()
                    await release_final_receive.wait()
                return await env_to_rollout.get()

        env_worker.send_to = lambda **kwargs: _QueuePutWork(
            env_to_rollout, kwargs["data"]
        )
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _GatedFinalRolloutReceive()
        rollout_worker.send_to = lambda **kwargs: _QueuePutWork(
            rollout_to_env, kwargs["data"]
        )
        env_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_worker.prepare_elastic_collection(
            lifecycle_generation=1, expected_policy_version=3
        )
        rollout_task = asyncio.create_task(
            rollout_worker.generate_until_pause_or_complete(None, None)
        )
        env_task = asyncio.create_task(
            env_worker.interact_until_pause_or_complete(None, None, None)
        )
        await final_receive_started.wait()
        assert not env_to_rollout.empty()
        drain = DrainRequest("final-bootstrap", 0, 1, 3)
        await asyncio.gather(
            env_worker.request_elastic_drain(drain),
            rollout_worker.request_elastic_drain(drain),
        )
        release_final_receive.set()
        results = await asyncio.gather(rollout_task, env_task)
        return results, env_to_rollout, rollout_to_env

    (rollout_result, env_result), env_to_rollout, rollout_to_env = asyncio.run(
        run_timing_case()
    )

    assert rollout_result.outcome is ElasticRunOutcome.COMPLETED
    assert env_result.outcome is ElasticRunOutcome.COMPLETED
    assert rollout_result.token is None
    assert env_result.token is None
    assert rollout_worker.predict_count == 2
    assert env_to_rollout.empty()
    assert rollout_to_env.empty()


def test_rollout_rejects_barrier_with_wrong_drain_request_id():
    barrier = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.DRAIN_BARRIER,
        transition_id=_identity(),
        logical_batch_size=1,
        env_output=None,
        drain_request_id="wrong-drain",
    )
    worker = _elastic_rollout_worker([barrier])
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.ACTIVE
    asyncio.run(worker.request_elastic_drain(DrainRequest("drain-1", 0, 1, 3)))

    with pytest.raises(ValueError, match="request ID"):
        asyncio.run(worker.generate_until_pause_or_complete(None, None))

    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT
    assert worker.predict_count == 0


def test_rollout_rejects_policy_version_change_before_inference():
    observation = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.OBSERVATION,
        transition_id=_identity(),
        logical_batch_size=1,
        env_output=_env_output(_identity()).to_dict(),
    )
    worker = _elastic_rollout_worker([observation])
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)
    worker.version = 4

    with pytest.raises(ValueError, match="policy version changed"):
        asyncio.run(worker.generate_until_pause_or_complete(None, None))

    assert worker.predict_count == 0
    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT


def test_duplicate_observation_fails_after_one_committed_inference():
    duplicate = ElasticRolloutRequest(
        kind=ElasticRolloutRequestKind.OBSERVATION,
        transition_id=_identity(),
        logical_batch_size=1,
        env_output=_env_output(_identity()).to_dict(),
    )
    worker = _elastic_rollout_worker([duplicate, duplicate])
    worker.n_train_chunk_steps = 2
    worker.prepare_elastic_collection(lifecycle_generation=1, expected_policy_version=3)

    with pytest.raises(ValueError, match="expected transition"):
        asyncio.run(worker.generate_until_pause_or_complete(None, None))

    assert worker.predict_count == 1
    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT


def test_completed_environment_offload_is_verified_idempotent_and_preserves_progress():
    worker = _elastic_env_worker()
    worker.torch_platform = _FakePlatform()
    worker._rollout_call_active = False
    worker._policy_request_in_flight = False
    worker.prepare_elastic_collection(lifecycle_generation=2, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.COMPLETED
    worker._elastic_completed_trajectories = 1

    first = worker.offload_completed_elastic_environment()
    second = worker.offload_completed_elastic_environment()

    assert isinstance(first, CompletedResidencyReceipt)
    assert second == first
    assert worker.get_elastic_status().state is ElasticRankState.COMPLETED
    assert worker.get_elastic_status().model_resident is False
    assert worker.get_elastic_progress().completed_trajectories == 1


def test_completed_rollout_offload_clears_all_model_residency_and_cuda_graphs():
    worker = _elastic_rollout_worker([])
    worker.hf_model = _FakeResidentModel()
    worker.torch_platform = _FakePlatform()
    worker._elastic_run_call_active = False
    worker.enable_cuda_graph = False
    worker.prepare_elastic_collection(lifecycle_generation=2, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.COMPLETED

    first = worker.offload_completed_elastic_rollout()
    second = worker.offload_completed_elastic_rollout()

    assert isinstance(first, CompletedResidencyReceipt)
    assert second == first
    assert worker.hf_model.device == "cpu"
    assert worker.get_elastic_status().state is ElasticRankState.COMPLETED
    assert worker.get_elastic_status().model_resident is False
    assert worker.get_elastic_status().cuda_graph_captured is False


def test_completed_environment_offload_failure_enters_failed_resident():
    worker = _elastic_env_worker()
    worker.torch_platform = _FakePlatform()
    worker._rollout_call_active = False
    worker._policy_request_in_flight = False
    worker.prepare_elastic_collection(lifecycle_generation=2, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.COMPLETED
    worker.env_list[0].offload = lambda: (_ for _ in ()).throw(
        RuntimeError("offload failed")
    )

    with pytest.raises(RuntimeError, match="offload failed"):
        worker.offload_completed_elastic_environment()

    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT


def test_completed_rollout_offload_failure_enters_failed_resident():
    worker = _elastic_rollout_worker([])
    worker.hf_model = _FakeResidentModel()
    worker.torch_platform = _FakePlatform()
    worker._elastic_run_call_active = False
    worker.prepare_elastic_collection(lifecycle_generation=2, expected_policy_version=3)
    worker._elastic_state = ElasticRankState.COMPLETED
    worker.offload_model = lambda: (_ for _ in ()).throw(
        RuntimeError("rollout offload failed")
    )

    with pytest.raises(RuntimeError, match="rollout offload failed"):
        worker.offload_completed_elastic_rollout()

    assert worker.get_elastic_status().state is ElasticRankState.FAILED_RESIDENT


@pytest.mark.parametrize(
    "worker_factory", [_elastic_env_worker, lambda: _elastic_rollout_worker([])]
)
def test_public_pair_failure_surface_is_idempotent(worker_factory):
    worker = worker_factory()
    worker.prepare_elastic_collection(lifecycle_generation=2, expected_policy_version=3)

    first = worker.fail_elastic_lifecycle(reason="peer token mismatch")
    second = worker.fail_elastic_lifecycle(reason="ignored replacement")

    assert first.state is ElasticRankState.FAILED_RESIDENT
    assert second == first
    assert second.failure == "peer token mismatch"


def test_coordinator_drives_stubbed_backend_pair_through_pause_and_resume():
    env_worker = _elastic_env_worker()
    _configure_single_chunk_env(
        env_worker,
        snapshot_factory=lambda: _FakeEnvResumeState(
            (env_worker._resume_bootstraps[0],)
        ),
    )
    env_worker.validate_rollout_resume_state = lambda *_args, **_kwargs: None
    env_worker.restore_rollout_stage = lambda *_args, **_kwargs: None
    env_worker.torch_platform = _FakePlatform()
    rollout_worker = _elastic_rollout_worker([])
    rollout_worker.hf_model = _FakeResidentModel()
    rollout_worker.device = "cuda"
    rollout_worker.torch_platform = _FakePlatform()

    async def run_pair():
        env_to_rollout = asyncio.Queue()
        rollout_to_env = asyncio.Queue()
        allow_rollout_receive = asyncio.Event()

        class _GatedQueueGetWork:
            def __init__(self, queue):
                self.queue = queue

            async def async_wait(self):
                await allow_rollout_receive.wait()
                return await self.queue.get()

        env_worker.send_to = lambda **kwargs: _QueuePutWork(
            env_to_rollout, kwargs["data"]
        )
        env_worker.recv_from = lambda **_kwargs: _QueueGetWork(rollout_to_env)
        rollout_worker.recv_from = lambda **_kwargs: _GatedQueueGetWork(env_to_rollout)
        rollout_worker.send_to = lambda **kwargs: _QueuePutWork(
            rollout_to_env, kwargs["data"]
        )
        coordinator = RLixResizeCoordinator(
            pipeline_id="embodied_abc123def456",
            env_workers={0: env_worker},
            rollout_workers={0: rollout_worker},
            operation_timeout_s=1.0,
            activation_poll_interval_s=0.001,
        )
        await coordinator.configure_collection(
            ElasticCollectionContext(1, 3, (0,)),
            env_input_channel=object(),
            rollout_request_channel=object(),
        )
        await coordinator.resize_infer([], [0])
        shrink = asyncio.create_task(coordinator.resize_infer([0], []))
        for _ in range(100):
            if (
                env_worker._elastic_state is ElasticRankState.DRAIN_REQUESTED
                and rollout_worker._elastic_state is ElasticRankState.DRAIN_REQUESTED
            ):
                break
            await asyncio.sleep(0)
        allow_rollout_receive.set()
        await shrink
        paused = await coordinator.get_status()
        assert paused.paused_ranks == (0,)
        assert not env_worker._environment_resident
        assert not rollout_worker._model_resident

        allow_rollout_receive.clear()
        await coordinator.resize_infer([], [0])
        allow_rollout_receive.set()
        results = await coordinator.get_rank_results(0, wait=True)
        completed = await coordinator.get_status()
        return results, completed, env_to_rollout, rollout_to_env

    results, completed, env_to_rollout, rollout_to_env = asyncio.run(run_pair())

    assert results[0].outcome is ElasticRunOutcome.COMPLETED
    assert results[1].outcome is ElasticRunOutcome.COMPLETED
    assert completed.completed_ranks == (0,)
    assert env_to_rollout.empty()
    assert rollout_to_env.empty()
