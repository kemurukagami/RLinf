from __future__ import annotations

import importlib
import sys
from collections import deque
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    EnvOutput,
    RolloutTransitionIdentity,
)
from rlinf.envs.world_model.base_world_env import (
    WORLD_ENV_RESUME_SCHEMA_VERSION,
    BaseWorldEnv,
    WorldEnvSnapshotContext,
)
from rlinf.workers.env.env_worker import (
    ENV_ROLLOUT_RESUME_SCHEMA_VERSION,
    EnvRolloutCursor,
    EnvWorker,
    RolloutCursorPhase,
)


class _FakeWorldEnv(BaseWorldEnv):
    def _build_dataset(self, cfg):
        return [0, 1, 2, 3]

    def _continuation_config(self):
        config = super()._continuation_config()
        config.update(
            {
                "chunk": self.cfg.chunk,
                "condition_frame_length": self.cfg.condition_frame_length,
            }
        )
        return config

    def chunk_step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError


def _make_env(*, chunk: int = 2) -> _FakeWorldEnv:
    cfg = OmegaConf.create(
        {
            "seed": 11,
            "video_cfg": {"save_video": False},
            "auto_reset": True,
            "ignore_terminations": False,
            "use_rel_reward": False,
            "enable_kir": False,
            "chunk": chunk,
            "condition_frame_length": 3,
        }
    )
    env = _FakeWorldEnv(
        cfg=cfg,
        num_envs=2,
        seed_offset=5,
        total_num_processes=2,
        worker_info=SimpleNamespace(rank=1, group_world_size=2),
    )
    env._generator = torch.Generator()
    env._generator.manual_seed(env.seed)
    env.reset_state_ids = torch.tensor([7, 8], dtype=torch.int64)
    env.current_obs = torch.arange(12, dtype=torch.float32).reshape(2, 3, 1, 2)
    env.image_queue = [
        deque([torch.full((1, 2), 1.0), torch.full((1, 2), 2.0)]),
        deque([torch.full((1, 2), 3.0), torch.full((1, 2), 4.0)]),
    ]
    env.task_descriptions = ["open drawer", "close drawer"]
    env.init_ee_poses = [torch.tensor([0.1]), {"pose": torch.tensor([0.2])}]
    env.prev_step_reward = torch.tensor([0.5, 0.75], dtype=torch.float32)
    env.success_once = torch.tensor([True, False], dtype=torch.bool)
    env.returns = torch.tensor([1.5, 2.5], dtype=torch.float32)
    env.elapsed_steps = 4
    env.is_start = False
    env._commit_episode_reset()
    return env


def _context(env: _FakeWorldEnv) -> WorldEnvSnapshotContext:
    return WorldEnvSnapshotContext(
        worker_rank=env.worker_info.rank,
        worker_world_size=env.worker_info.group_world_size,
        stage_id=0,
        lifecycle_generation=3,
        chunk_index=1,
        next_transition_id=2,
        episode_generations=env.episode_generations.detach().cpu().clone(),
        reset_state_ids=env.reset_state_ids.detach().cpu().clone(),
    )


def test_base_world_env_snapshot_has_identity_and_cpu_copies():
    env = _make_env()
    state = env.snapshot_resume_state(_context(env))

    assert state.schema_version == WORLD_ENV_RESUME_SCHEMA_VERSION
    assert state.environment_type == "_FakeWorldEnv"
    assert state.worker_rank == 1
    assert state.worker_world_size == 2
    assert state.stage_id == 0
    assert state.lifecycle_generation == 3
    assert state.chunk_index == 1
    assert state.next_transition_id == 2
    BaseWorldEnv.assert_cpu_only(state)

    env.current_obs.add_(100)
    env.image_queue[0][0].add_(100)
    env.prev_step_reward.add_(100)

    torch.testing.assert_close(
        state.current_obs, torch.arange(12, dtype=torch.float32).reshape(2, 3, 1, 2)
    )
    torch.testing.assert_close(state.image_queue[0][0], torch.full((1, 2), 1.0))
    torch.testing.assert_close(
        state.prev_step_reward, torch.tensor([0.5, 0.75], dtype=torch.float32)
    )


def test_base_world_env_prepare_and_commit_round_trip():
    source = _make_env()
    state = source.snapshot_resume_state(_context(source))

    target = _make_env()
    target.current_obs.zero_()
    target.image_queue[0][0].zero_()
    target.prev_step_reward.zero_()

    prepared = target.prepare_resume_state(state, _context(source))
    target.commit_resume_state(prepared)

    torch.testing.assert_close(target.current_obs, state.current_obs)
    torch.testing.assert_close(target.image_queue[0][0], state.image_queue[0][0])
    torch.testing.assert_close(target.prev_step_reward, state.prev_step_reward)
    torch.testing.assert_close(
        target.episode_generations.cpu(), state.episode_generations
    )
    torch.testing.assert_close(target.reset_state_ids.cpu(), state.reset_state_ids)
    assert target.task_descriptions == list(state.task_descriptions)
    assert target.elapsed_steps == state.elapsed_steps
    assert target.is_start == state.is_start


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("schema_version", 999, "schema_version mismatch"),
        ("environment_type", "OtherEnv", "environment_type mismatch"),
        ("worker_rank", 0, "worker_rank mismatch"),
        ("stage_id", 4, "stage_id mismatch"),
        ("next_transition_id", 99, "next_transition_id mismatch"),
        (
            "episode_generations",
            torch.tensor([10, 20], dtype=torch.int64),
            "episode_generations do not match",
        ),
        (
            "reset_state_ids",
            torch.tensor([10, 20], dtype=torch.int64),
            "reset_state_ids do not match",
        ),
        (
            "prev_step_reward",
            torch.ones(2, dtype=torch.float64),
            "prev_step_reward must have dtype",
        ),
    ],
)
def test_base_world_env_validation_rejects_corrupt_state_before_mutation(
    field, value, match
):
    env = _make_env()
    context = _context(env)
    state = env.snapshot_resume_state(context)
    bad_state = replace(state, **{field: value})

    before = env.current_obs.clone()
    with pytest.raises((ValueError, AssertionError), match=match):
        env.prepare_resume_state(bad_state, context)

    torch.testing.assert_close(env.current_obs, before)


def test_base_world_env_config_fingerprint_changes_for_relevant_config():
    env = _make_env(chunk=2)
    changed = _make_env(chunk=3)

    state = env.snapshot_resume_state(_context(env))

    assert state.config_fingerprint != changed._config_fingerprint()
    with pytest.raises(ValueError, match="config_fingerprint mismatch"):
        changed.prepare_resume_state(state, _context(changed))


def test_base_world_env_context_must_match_live_identity():
    env = _make_env()
    context = replace(_context(env), worker_rank=0)

    with pytest.raises(ValueError, match="context.worker_rank mismatch"):
        env.snapshot_resume_state(context)


def _load_opensora_env_class(monkeypatch):
    registry = ModuleType("opensora.registry")
    registry.MODELS = object()
    registry.SCHEDULERS = object()
    registry.build_module = lambda *args, **kwargs: None

    inference_utils = ModuleType("opensora.utils.inference_utils")
    inference_utils.prepare_multi_resolution_info = lambda *args, **kwargs: {}
    misc = ModuleType("opensora.utils.misc")
    misc.to_torch_dtype = lambda value: torch.float32

    opensora = ModuleType("opensora")
    utils = ModuleType("opensora.utils")
    datasets = ModuleType("rlinf.data.datasets")
    datasets.__path__ = []
    world_model_dataset = ModuleType("rlinf.data.datasets.world_model")
    world_model_dataset.NpyTrajectoryDatasetWrapper = object
    monkeypatch.setitem(sys.modules, "opensora", opensora)
    monkeypatch.setitem(sys.modules, "opensora.registry", registry)
    monkeypatch.setitem(sys.modules, "opensora.utils", utils)
    monkeypatch.setitem(sys.modules, "opensora.utils.inference_utils", inference_utils)
    monkeypatch.setitem(sys.modules, "opensora.utils.misc", misc)
    monkeypatch.setitem(sys.modules, "rlinf.data.datasets", datasets)
    monkeypatch.setitem(
        sys.modules, "rlinf.data.datasets.world_model", world_model_dataset
    )

    module = importlib.import_module("rlinf.envs.world_model.world_model_opensora_env")
    return module.OpenSoraEnv


def _make_opensora_env(monkeypatch):
    opensora_env_class = _load_opensora_env_class(monkeypatch)
    env = object.__new__(opensora_env_class)
    env.cfg = OmegaConf.create({})
    env.device = torch.device("cpu")
    env.seed = 17
    env.num_envs = 2
    env.worker_info = SimpleNamespace(rank=0, group_world_size=1)
    env.record_metrics = True
    env.auto_reset = True
    env.ignore_terminations = False
    env.use_rel_reward = False
    env.chunk = 2
    env.condition_frame_length = 3
    env.image_size = (4, 5)
    env.inference_dtype = torch.float32
    env.z_condition_frame_length = 2
    env.z_mask_frame_num = 1
    env.world_model_cfg = OmegaConf.create(
        {
            "model": {"type": "FakeOpenSora"},
            "reward_model": {"type": "FakeReward"},
            "vae": {"type": "FakeVAE"},
        }
    )

    # Concrete environments own generator initialization; the base class does not.
    env._generator = torch.Generator()
    env._generator.manual_seed(env.seed)
    env._diffusion_generator = env._new_diffusion_generator()
    env.reset_state_ids = torch.tensor([2, 3], dtype=torch.int64)
    env.episode_generations = torch.tensor([4, 5], dtype=torch.int64)
    env.current_obs = torch.arange(2 * 3 * 1 * 5 * 4 * 5, dtype=torch.float32).reshape(
        2, 3, 1, 5, 4, 5
    )
    env.image_queue = [
        deque(
            [
                torch.full((1, 4, 1, 2, 3), env_idx * 2 + frame_idx + 1.0)
                for frame_idx in range(2)
            ],
            maxlen=2,
        )
        for env_idx in range(2)
    ]
    env.task_descriptions = ["task zero", "task one"]
    env.init_ee_poses = [torch.tensor([0.1]), torch.tensor([0.2])]
    env.elapsed_steps = 2
    env.prev_step_reward = torch.tensor([0.25, 0.5], dtype=torch.float32)
    env.success_once = torch.tensor([False, True], dtype=torch.bool)
    env.returns = torch.tensor([1.0, 2.0], dtype=torch.float32)
    env._is_start = False
    return env


def _opensora_context(env):
    return WorldEnvSnapshotContext(
        worker_rank=0,
        worker_world_size=1,
        stage_id=0,
        lifecycle_generation=2,
        chunk_index=1,
        next_transition_id=2,
        episode_generations=env.episode_generations.clone(),
        reset_state_ids=env.reset_state_ids.clone(),
    )


def test_opensora_round_trip_restores_dedicated_diffusion_generator(monkeypatch):
    source = _make_opensora_env(monkeypatch)
    context = _opensora_context(source)
    state = source.snapshot_resume_state(context)
    expected_noise = torch.randn(8, generator=source._diffusion_generator)

    torch.manual_seed(999)
    torch.randn(128)

    target = _make_opensora_env(monkeypatch)
    target.current_obs.zero_()
    target.image_queue[0][0].zero_()
    target.commit_resume_state(target.prepare_resume_state(state, context))
    actual_noise = torch.randn(8, generator=target._diffusion_generator)

    torch.testing.assert_close(actual_noise, expected_noise)
    torch.testing.assert_close(target.current_obs, state.current_obs)
    torch.testing.assert_close(target.image_queue[0][0], state.image_queue[0][0])
    assert target.task_descriptions == list(state.task_descriptions)


@pytest.mark.parametrize(
    "replacement,match",
    [
        ({"condition_action": torch.zeros(2, 3, 7)}, "condition_action must be None"),
        ({"diffusion_generator_state": None}, "requires diffusion_generator_state"),
        (
            {"current_obs": torch.zeros(2, 3, 1, 8, 4, 5)},
            "time dimension is out of bounds",
        ),
    ],
)
def test_opensora_rejects_invalid_model_state_before_mutation(
    monkeypatch, replacement, match
):
    env = _make_opensora_env(monkeypatch)
    context = _opensora_context(env)
    state = env.snapshot_resume_state(context)
    before = env.current_obs.clone()

    with pytest.raises(ValueError, match=match):
        env.prepare_resume_state(replace(state, **replacement), context)

    torch.testing.assert_close(env.current_obs, before)


def test_opensora_rejects_inconsistent_latent_queue(monkeypatch):
    env = _make_opensora_env(monkeypatch)
    context = _opensora_context(env)
    state = env.snapshot_resume_state(context)
    bad_queue = [list(queue) for queue in state.image_queue]
    bad_queue[1][1] = torch.zeros((1, 4, 1, 3, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="latent queue frame shapes must match"):
        env.prepare_resume_state(
            replace(state, image_queue=tuple(tuple(queue) for queue in bad_queue)),
            context,
        )


def test_opensora_rejects_latent_queue_dtype(monkeypatch):
    env = _make_opensora_env(monkeypatch)
    context = _opensora_context(env)
    state = env.snapshot_resume_state(context)
    bad_queue = [list(queue) for queue in state.image_queue]
    bad_queue[0][0] = bad_queue[0][0].to(torch.float64)

    with pytest.raises(ValueError, match="must have dtype torch.float32"):
        env.prepare_resume_state(
            replace(state, image_queue=tuple(tuple(queue) for queue in bad_queue)),
            context,
        )


def test_opensora_offload_onload_is_idempotent_with_cpu_fakes(monkeypatch):
    class _FakeModel:
        def __init__(self):
            self.to_calls = []

        def to(self, *args):
            self.to_calls.append(args)
            return self

    env = _make_opensora_env(monkeypatch)
    env.vae = _FakeModel()
    env.model = _FakeModel()
    env.reward_model = _FakeModel()
    env._is_offloaded = False
    cache_clear_calls = []
    env._clear_accelerator_cache = lambda: cache_clear_calls.append(True)

    env.offload()
    env.offload()
    assert env._is_offloaded
    assert len(env.vae.to_calls) == 1
    assert len(cache_clear_calls) == 1

    env.onload()
    env.onload()
    assert not env._is_offloaded
    assert len(env.vae.to_calls) == 2


def _load_wan_env_class(monkeypatch):
    reward_model = ModuleType("diffsynth.models.reward_model")
    reward_model.ResnetRewModel = object
    reward_model.TaskEmbedResnetRewModel = object
    wan_pipeline = ModuleType("diffsynth.pipelines.wan_video_new")
    wan_pipeline.ModelConfig = object
    wan_pipeline.WanVideoPipeline = object
    diffsynth = ModuleType("diffsynth")
    models = ModuleType("diffsynth.models")
    pipelines = ModuleType("diffsynth.pipelines")
    datasets = ModuleType("rlinf.data.datasets")
    datasets.__path__ = []
    world_model_dataset = ModuleType("rlinf.data.datasets.world_model")
    world_model_dataset.NpyTrajectoryDatasetWrapper = object

    monkeypatch.setitem(sys.modules, "diffsynth", diffsynth)
    monkeypatch.setitem(sys.modules, "diffsynth.models", models)
    monkeypatch.setitem(sys.modules, "diffsynth.models.reward_model", reward_model)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, "diffsynth.pipelines.wan_video_new", wan_pipeline)
    monkeypatch.setitem(sys.modules, "rlinf.data.datasets", datasets)
    monkeypatch.setitem(
        sys.modules, "rlinf.data.datasets.world_model", world_model_dataset
    )

    module = importlib.import_module("rlinf.envs.world_model.world_model_wan_env")
    return module.WanEnv


def _make_wan_env(monkeypatch):
    wan_env_class = _load_wan_env_class(monkeypatch)
    env = object.__new__(wan_env_class)
    env.cfg = OmegaConf.create({"reward_model": {"type": "FakeReward"}})
    env.device = torch.device("cpu")
    env.seed = 23
    env.num_envs = 2
    env.worker_info = SimpleNamespace(rank=0, group_world_size=1)
    env.record_metrics = True
    env.auto_reset = True
    env.ignore_terminations = False
    env.use_rel_reward = False
    env.chunk = 8
    env.condition_frame_length = 5
    env.num_frames = 13
    env.image_size = (4, 5)
    env.num_inference_steps = 4
    env.retain_action = True
    env.reset_gripper_open = True
    env.is_libero_env = True
    env._diffusion_seed = 0

    env._generator = torch.Generator()
    env._generator.manual_seed(env.seed)
    env.reset_state_ids = torch.tensor([6, 7], dtype=torch.int64)
    env.episode_generations = torch.tensor([8, 9], dtype=torch.int64)
    env.current_obs = torch.linspace(
        -1, 1, 2 * 3 * 1 * 5 * 4 * 5, dtype=torch.float32
    ).reshape(2, 3, 1, 5, 4, 5)
    env.image_queue = [
        [
            torch.full((3, 1, 4, 5), -0.9 + 0.1 * (env_idx * 5 + frame_idx))
            for frame_idx in range(5)
        ]
        for env_idx in range(2)
    ]
    env.condition_action = torch.arange(70, dtype=torch.float32).reshape(2, 5, 7)
    env.task_descriptions = ["wan zero", "wan one"]
    env.init_ee_poses = [torch.tensor([0.3]), torch.tensor([0.4])]
    env.elapsed_steps = 2
    env.prev_step_reward = torch.tensor([0.1, 0.2], dtype=torch.float32)
    env.success_once = torch.tensor([True, False], dtype=torch.bool)
    env.returns = torch.tensor([3.0, 4.0], dtype=torch.float32)
    env._is_start = False
    env._is_offloaded = False
    return env


def test_wan_round_trip_restores_queue_actions_and_seed(monkeypatch):
    source = _make_wan_env(monkeypatch)
    context = _opensora_context(source)
    state = source.snapshot_resume_state(context)

    target = _make_wan_env(monkeypatch)
    target.image_queue[0][0].zero_()
    target.condition_action.zero_()
    target.commit_resume_state(target.prepare_resume_state(state, context))

    assert state.diffusion_seed == 0
    assert target._diffusion_seed == 0
    assert isinstance(target.image_queue[0], list)
    torch.testing.assert_close(target.image_queue[0][0], state.image_queue[0][0])
    torch.testing.assert_close(target.condition_action, state.condition_action)


@pytest.mark.parametrize(
    "replacement,match",
    [
        ({"diffusion_seed": 9}, "diffusion_seed must equal 0"),
        (
            {"condition_action": torch.zeros(2, 2, 7)},
            "condition_action must have shape",
        ),
        ({"condition_action": None}, "condition_action must be a torch.Tensor"),
    ],
)
def test_wan_rejects_invalid_state_before_mutation(monkeypatch, replacement, match):
    env = _make_wan_env(monkeypatch)
    context = _opensora_context(env)
    state = env.snapshot_resume_state(context)
    before = env.current_obs.clone()

    with pytest.raises((TypeError, ValueError), match=match):
        env.prepare_resume_state(replace(state, **replacement), context)

    torch.testing.assert_close(env.current_obs, before)


def test_wan_rejects_short_queue(monkeypatch):
    env = _make_wan_env(monkeypatch)
    context = _opensora_context(env)
    state = env.snapshot_resume_state(context)
    short_queue = (state.image_queue[0][:-1], state.image_queue[1])

    with pytest.raises(ValueError, match="length must equal condition_frame_length"):
        env.prepare_resume_state(replace(state, image_queue=short_queue), context)


def test_wan_resumed_next_chunk_matches_queue_and_action_conditioning(monkeypatch):
    from PIL import Image

    class _FakePipeline:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(
                {
                    "seed": kwargs["seed"],
                    "action": kwargs["action"].clone(),
                    "input_image": [
                        np.asarray(image).copy() for image in kwargs["input_image"]
                    ],
                    "input_image4": [
                        [np.asarray(image).copy() for image in images]
                        for images in kwargs["input_image4"]
                    ],
                }
            )
            return [
                [
                    Image.fromarray(
                        np.full((4, 5, 3), env_idx * 20 + frame_idx, dtype=np.uint8)
                    )
                    for frame_idx in range(13)
                ]
                for env_idx in range(2)
            ]

    source = _make_wan_env(monkeypatch)
    context = _opensora_context(source)
    state = source.snapshot_resume_state(context)
    actions = torch.linspace(-1, 1, 2 * 8 * 7).reshape(2, 8, 7)
    source.pipe = _FakePipeline()
    source._infer_next_chunk_frames(actions)

    target = _make_wan_env(monkeypatch)
    target.commit_resume_state(target.prepare_resume_state(state, context))
    target.pipe = _FakePipeline()
    target._infer_next_chunk_frames(actions)

    torch.testing.assert_close(target.condition_action, source.condition_action)
    torch.testing.assert_close(target.current_obs, source.current_obs)
    for target_queue, source_queue in zip(target.image_queue, source.image_queue):
        for target_frame, source_frame in zip(target_queue, source_queue):
            torch.testing.assert_close(target_frame, source_frame)
    assert target.pipe.calls[0]["seed"] == source.pipe.calls[0]["seed"] == 0
    torch.testing.assert_close(
        target.pipe.calls[0]["action"], source.pipe.calls[0]["action"]
    )
    for target_images, source_images in zip(
        target.pipe.calls[0]["input_image4"], source.pipe.calls[0]["input_image4"]
    ):
        for target_image, source_image in zip(target_images, source_images):
            np.testing.assert_array_equal(target_image, source_image)


def test_wan_offload_onload_moves_actions_and_keeps_queue_on_cpu(monkeypatch):
    class _FakeModel:
        def to(self, *args):
            return self

    env = _make_wan_env(monkeypatch)
    env.pipe = SimpleNamespace(vae=_FakeModel(), dit=_FakeModel())
    env.reward_model = _FakeModel()
    env._clear_accelerator_cache = lambda: None

    env.offload()
    env.offload()
    assert env.condition_action.device.type == "cpu"
    assert all(
        frame.device.type == "cpu" for queue in env.image_queue for frame in queue
    )

    env.onload()
    env.onload()
    assert env.condition_action.device == env.device


def test_wan_offload_preserves_uninitialized_image_queue(monkeypatch):
    class _FakeModel:
        def to(self, *args):
            return self

    env = _make_wan_env(monkeypatch)
    env.pipe = SimpleNamespace(vae=_FakeModel(), dit=_FakeModel())
    env.reward_model = _FakeModel()
    env.current_obs = None
    env.image_queue = [[None] * env.condition_frame_length for _ in range(env.num_envs)]
    env._clear_accelerator_cache = lambda: None

    env.offload()

    assert env._is_offloaded
    assert env.current_obs is None
    assert env.image_queue == [
        [None] * env.condition_frame_length for _ in range(env.num_envs)
    ]


def _make_wan_worker(monkeypatch):
    env = _make_wan_env(monkeypatch)
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "use_fixed_reset_state_ids": True,
                    "data_collection": None,
                }
            },
            "rollout": {"group_name": "RolloutGroup"},
        }
    )
    worker._rank = 0
    worker._world_size = 1
    worker.stage_num = 1
    worker.env_list = [env]
    worker.env_decoupled_mode = False
    worker.enable_online_lerobot = False
    worker.enable_rlt = False
    worker.reward_mode = "per_step"
    worker.use_training_pipeline = False
    worker.train_num_envs_per_stage = 2
    worker._rollout_call_active = False
    worker._policy_request_in_flight = False
    worker._lifecycle_generation = 2
    worker._rollout_cursor = EnvRolloutCursor(
        schema_version=ENV_ROLLOUT_RESUME_SCHEMA_VERSION,
        lifecycle_generation=2,
        policy_version=7,
        epoch_index=0,
        chunk_index=1,
        stage_id=0,
        next_transition_ids=(1,),
        phase=RolloutCursorPhase.BOOTSTRAP_PENDING,
    )
    worker.rollout_results = [
        EmbodiedRolloutResult(
            max_episode_length=240,
            actions=[torch.ones(2, 56)],
            intervene_flags=[torch.zeros(2, 56, dtype=torch.bool)],
            rewards=[torch.ones(2, 8)],
            terminations=[torch.zeros(2, 8, dtype=torch.bool)],
            truncations=[torch.zeros(2, 8, dtype=torch.bool)],
            dones=[torch.zeros(2, 8, dtype=torch.bool)],
            prev_logprobs=[torch.full((2, 8), 0.25)],
            prev_values=[torch.full((2, 1), 0.5)],
            versions=[torch.full((2, 8), 7.0)],
            forward_inputs=[{"action": torch.ones(2, 56)}],
        )
    ]
    env_output = EnvOutput(
        obs={
            "main_images": torch.zeros(2, 4, 5, 3, dtype=torch.uint8),
            "wrist_images": None,
            "states": torch.zeros(2, 16),
            "task_descriptions": ["wan zero", "wan one"],
        },
        rewards=torch.ones(2, 8),
        dones=torch.zeros(2, 8, dtype=torch.bool),
        terminations=torch.zeros(2, 8, dtype=torch.bool),
        truncations=torch.zeros(2, 8, dtype=torch.bool),
    )
    worker._current_env_outputs = [env_output]
    worker._resume_bootstraps = [env_output]
    worker.last_obs_list = [env_output.obs]
    worker.last_intervened_info_list = [(None, None)]
    worker.train_prev_done = [torch.zeros(2, dtype=torch.bool)]
    worker._rollout_env_metrics = {"return": [torch.tensor([1.0, 2.0])]}
    worker._rlt_pending_obs = [None]
    worker._prefetched_train_bootstrap = None
    return worker


def test_wan_worker_snapshot_owns_cpu_state_and_restores(monkeypatch):
    source = _make_wan_worker(monkeypatch)
    state = source.snapshot_rollout_stage()
    BaseWorldEnv.assert_cpu_only(state)

    source.rollout_results[0].actions[0].add_(100)
    source.env_list[0].condition_action.zero_()
    torch.testing.assert_close(state.rollout_results[0].actions[0], torch.ones(2, 56))
    assert state.world_states[0].condition_action.count_nonzero() > 0

    target = _make_wan_worker(monkeypatch)
    target.rollout_results[0].actions[0].zero_()
    target.restore_rollout_stage(
        state,
        expected_lifecycle_generation=2,
        expected_policy_version=7,
    )

    assert target._rollout_cursor.phase is RolloutCursorPhase.BOOTSTRAP_PENDING
    assert target._resume_bootstraps[0] is not None
    torch.testing.assert_close(
        target.rollout_results[0].actions[0], state.rollout_results[0].actions[0]
    )
    torch.testing.assert_close(
        target.env_list[0].condition_action, state.world_states[0].condition_action
    )


def test_elastic_worker_snapshot_preserves_pending_transition_identity(monkeypatch):
    worker = _make_wan_worker(monkeypatch)
    identity = RolloutTransitionIdentity(
        lifecycle_generation=2,
        env_worker_rank=0,
        stage_id=0,
        sequence=1,
    )
    worker._current_env_outputs[0].transition_id = identity
    worker._resume_bootstraps[0].transition_id = identity

    state = worker.snapshot_rollout_stage()

    assert state.current_env_outputs[0].transition_id == identity
    assert state.resume_bootstraps[0].transition_id == identity
    assert state.cursor.next_transition_ids[0] == identity.sequence
    assert state.world_states[0].next_transition_id == identity.sequence


def test_wan_worker_restore_rejects_policy_mismatch_before_mutation(monkeypatch):
    worker = _make_wan_worker(monkeypatch)
    state = worker.snapshot_rollout_stage()
    before = worker.rollout_results[0].actions[0].clone()

    with pytest.raises(ValueError, match="policy_version mismatch"):
        worker.restore_rollout_stage(
            state,
            expected_lifecycle_generation=2,
            expected_policy_version=8,
        )

    torch.testing.assert_close(worker.rollout_results[0].actions[0], before)


def test_wan_worker_snapshot_requires_fixed_reset_ids(monkeypatch):
    worker = _make_wan_worker(monkeypatch)
    worker.cfg.env.train.use_fixed_reset_state_ids = False

    with pytest.raises(NotImplementedError, match="fixed reset-state IDs"):
        worker.snapshot_rollout_stage()


def test_wan_worker_pending_bootstrap_is_consumed_once(monkeypatch):
    worker = _make_wan_worker(monkeypatch)
    sends = []
    worker.send_to = lambda **kwargs: sends.append(kwargs)

    worker._send_pending_bootstrap(object(), 0)

    assert len(sends) == 1
    assert worker._resume_bootstraps == [None]
    assert worker._rollout_cursor.phase is RolloutCursorPhase.WAITING_FOR_POLICY
    with pytest.raises(RuntimeError, match="no pending bootstrap"):
        worker._send_pending_bootstrap(object(), 0)


def test_wan_worker_two_chunk_resume_matches_uninterrupted(monkeypatch):
    from PIL import Image

    class _DeterministicWanPipeline:
        def __call__(self, **kwargs):
            assert kwargs["seed"] == 0
            return [
                [
                    Image.fromarray(
                        np.full((4, 5, 3), env_idx * 20 + frame_idx, dtype=np.uint8)
                    )
                    for frame_idx in range(13)
                ]
                for env_idx in range(2)
            ]

    uninterrupted = _make_wan_worker(monkeypatch)
    state = uninterrupted.snapshot_rollout_stage()
    resumed = _make_wan_worker(monkeypatch)
    resumed.restore_rollout_stage(
        state,
        expected_lifecycle_generation=2,
        expected_policy_version=7,
    )

    for worker in (uninterrupted, resumed):
        worker.send_to = lambda **kwargs: None
        worker._send_pending_bootstrap(object(), 0)
        worker.env_list[0].pipe = _DeterministicWanPipeline()

    actions = torch.linspace(-1, 1, 2 * 8 * 7).reshape(2, 8, 7)
    uninterrupted.env_list[0]._infer_next_chunk_frames(actions)
    resumed.env_list[0]._infer_next_chunk_frames(actions)

    for worker in (uninterrupted, resumed):
        worker.rollout_results[0].append_step_result(
            ChunkStepResult(
                actions=actions.reshape(2, -1),
                rewards=torch.full((2, 8), 0.5),
                terminations=torch.zeros(2, 8, dtype=torch.bool),
                truncations=torch.zeros(2, 8, dtype=torch.bool),
                dones=torch.zeros(2, 8, dtype=torch.bool),
                prev_logprobs=torch.full((2, 8), 0.25),
                prev_values=torch.full((2, 1), 0.5),
                versions=torch.full((2, 8), 7.0),
                forward_inputs={"action": actions.reshape(2, -1)},
            )
        )

    torch.testing.assert_close(
        resumed.env_list[0].current_obs, uninterrupted.env_list[0].current_obs
    )
    torch.testing.assert_close(
        resumed.env_list[0].condition_action,
        uninterrupted.env_list[0].condition_action,
    )
    for resumed_queue, uninterrupted_queue in zip(
        resumed.env_list[0].image_queue, uninterrupted.env_list[0].image_queue
    ):
        for resumed_frame, uninterrupted_frame in zip(
            resumed_queue, uninterrupted_queue
        ):
            torch.testing.assert_close(resumed_frame, uninterrupted_frame)
    for field_name in (
        "actions",
        "rewards",
        "terminations",
        "truncations",
        "dones",
        "prev_logprobs",
        "prev_values",
        "versions",
    ):
        resumed_values = getattr(resumed.rollout_results[0], field_name)
        uninterrupted_values = getattr(uninterrupted.rollout_results[0], field_name)
        assert len(resumed_values) == len(uninterrupted_values)
        for resumed_value, uninterrupted_value in zip(
            resumed_values, uninterrupted_values
        ):
            torch.testing.assert_close(resumed_value, uninterrupted_value)
    for field_name in (
        "prev_step_reward",
        "success_once",
        "returns",
        "episode_generations",
        "reset_state_ids",
    ):
        torch.testing.assert_close(
            getattr(resumed.env_list[0], field_name),
            getattr(uninterrupted.env_list[0], field_name),
        )
    assert resumed.env_list[0].elapsed_steps == uninterrupted.env_list[0].elapsed_steps
    assert resumed.env_list[0].is_start == uninterrupted.env_list[0].is_start
    assert (
        resumed.env_list[0].task_descriptions
        == uninterrupted.env_list[0].task_descriptions
    )
    for resumed_obs, uninterrupted_obs in zip(
        resumed.last_obs_list, uninterrupted.last_obs_list
    ):
        for key in ("main_images", "states"):
            torch.testing.assert_close(resumed_obs[key], uninterrupted_obs[key])
    for resumed_done, uninterrupted_done in zip(
        resumed.train_prev_done, uninterrupted.train_prev_done
    ):
        torch.testing.assert_close(resumed_done, uninterrupted_done)
    assert (
        resumed._rollout_env_metrics.keys() == uninterrupted._rollout_env_metrics.keys()
    )
    for key in resumed._rollout_env_metrics:
        for resumed_metric, uninterrupted_metric in zip(
            resumed._rollout_env_metrics[key], uninterrupted._rollout_env_metrics[key]
        ):
            torch.testing.assert_close(resumed_metric, uninterrupted_metric)
