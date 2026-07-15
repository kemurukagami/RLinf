"""Real-model Task 1 snapshot/resume equivalence test for Wan LIBERO Spatial."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from rlinf.envs.world_model.base_world_env import WorldEnvSnapshotContext
from rlinf.scheduler import Worker


def _require_checkpoint(path: Path, required_files: tuple[str, ...]) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(f"checkpoint {path} is missing: {', '.join(missing)}")


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _assert_nested_close(
    actual: Any,
    expected: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(
            actual.cpu(), expected.cpu(), rtol=rtol, atol=atol, msg=path
        )
        return
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=path)
        return
    if is_dataclass(actual) and is_dataclass(expected):
        if type(actual) is not type(expected):
            raise AssertionError(f"{path}: dataclass types differ")
        for field in fields(actual):
            _assert_nested_close(
                getattr(actual, field.name),
                getattr(expected, field.name),
                path=f"{path}.{field.name}",
                rtol=rtol,
                atol=atol,
            )
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: mapping keys differ")
        for key in actual:
            _assert_nested_close(
                actual[key], expected[key], path=f"{path}.{key}", rtol=rtol, atol=atol
            )
        return
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: sequence lengths differ")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_nested_close(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
            )
        return
    if isinstance(actual, float) and isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _snapshot_context(env, chunk_index: int) -> WorldEnvSnapshotContext:
    return WorldEnvSnapshotContext(
        worker_rank=0,
        worker_world_size=1,
        stage_id=0,
        lifecycle_generation=0,
        chunk_index=chunk_index,
        next_transition_id=chunk_index,
        episode_generations=env.episode_generations.detach().cpu().clone(),
        reset_state_ids=env.reset_state_ids.detach().cpu().clone(),
    )


def _load_configs(repo_path: Path, test_cfg):
    env_cfg = OmegaConf.load(
        repo_path / "examples/embodiment/config/env/wan_libero_spatial.yaml"
    )
    wm_checkpoint = Path(test_cfg.models.world_model.checkpoint)
    with open_dict(env_cfg):
        env_cfg.total_num_envs = 1
        env_cfg.group_size = 1
        env_cfg.auto_reset = False
        env_cfg.max_episode_steps = 24
        env_cfg.max_steps_per_rollout_epoch = 24
        env_cfg.video_cfg.save_video = False
        env_cfg.enable_offload = True
        env_cfg.num_inference_steps = int(
            test_cfg.models.world_model.num_inference_steps
        )
        env_cfg.wan_wm_hf_ckpt_path = str(wm_checkpoint)
        env_cfg.VAE_path = str(wm_checkpoint / "Wan2.2_VAE.pth")
        env_cfg.model_path = str(wm_checkpoint / "model-00001.safetensors")
        env_cfg.initial_image_path = str(wm_checkpoint / "dataset")
        env_cfg.reward_model.from_pretrained = str(wm_checkpoint / "resnet_rm.pth")

    vla_cfg = OmegaConf.load(
        repo_path / "examples/embodiment/config/model/openvla_oft.yaml"
    )
    with open_dict(vla_cfg):
        vla_cfg.model_path = str(test_cfg.models.vla.checkpoint)
        vla_cfg.precision = str(test_cfg.models.vla.precision)
        vla_cfg.unnorm_key = str(test_cfg.models.vla.unnorm_key)
        vla_cfg.max_prompt_length = 128
        vla_cfg.attn_implementation = "eager"
    return env_cfg, vla_cfg


def _predict_action(policy, observation: dict[str, Any]) -> torch.Tensor:
    actions, _ = policy.predict_action_batch(
        env_obs=_to_cpu(observation),
        do_sample=False,
        calculate_logprobs=False,
        calculate_values=False,
    )
    return actions.detach().cpu().contiguous()


def _activate_policy(policy, env, device: torch.device, *, collocated: bool) -> None:
    """Give a collocated GPU to the policy while keeping Wan offloaded."""
    if not collocated:
        return
    env.offload()
    policy.to(device)


def _activate_world_model(policy, env, *, collocated: bool) -> None:
    """Give a collocated GPU to Wan while keeping the policy offloaded."""
    if not collocated:
        return
    policy.to("cpu")
    torch.cuda.empty_cache()
    env.onload()


def run(config_path: Path) -> None:
    repo_path = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(config_path)
    vla_checkpoint = Path(cfg.models.vla.checkpoint)
    wm_checkpoint = Path(cfg.models.world_model.checkpoint)
    if int(cfg.test.action_chunks) != 2:
        raise ValueError("Task 1 equivalence requires exactly two action chunks")

    _require_checkpoint(
        vla_checkpoint,
        ("config.json", "dataset_statistics.json", "model.safetensors.index.json"),
    )
    _require_checkpoint(
        wm_checkpoint,
        ("model-00001.safetensors", "Wan2.2_VAE.pth", "resnet_rm.pth", "dataset"),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Task 1 real-model E2E requires CUDA")
    placement = str(cfg.test.get("placement", "disaggregated"))
    if placement not in {"collocated", "disaggregated"}:
        raise ValueError(
            "test.placement must be either 'collocated' or 'disaggregated'"
        )
    collocated = placement == "collocated"
    required_gpus = 1 if collocated else 2
    if torch.cuda.device_count() < required_gpus:
        raise RuntimeError(
            f"Task 1 {placement} real-model E2E requires {required_gpus} visible GPU(s)"
        )

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.manual_seed(0)
    np.random.seed(0)

    wm_device = torch.device(str(cfg.models.world_model.device))
    vla_device = torch.device(str(cfg.models.vla.device))
    if collocated and wm_device != vla_device:
        raise ValueError("Collocated placement requires VLA and Wan on the same GPU")
    if not collocated and wm_device == vla_device:
        raise ValueError(
            "Disaggregated placement requires VLA and Wan on different GPUs"
        )

    env_cfg, vla_cfg = _load_configs(repo_path, cfg)
    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda

    torch.cuda.set_device(wm_device)
    from rlinf.envs.world_model.world_model_wan_env import WanEnv

    worker_info = SimpleNamespace(rank=0, group_world_size=1)
    env = WanEnv(
        env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        record_metrics=True,
        worker_info=worker_info,
    )
    # BaseWorldEnv normally runs with one visible GPU. Pin the configured device
    # explicitly because this harness tests both paired and shared placement.
    env.device = wm_device

    observation = None
    if collocated:
        # Reset before loading OpenVLA, then move all Wan state to CPU. The two
        # checkpoints cannot be resident together on a 24 GB GPU.
        observation, _ = env.reset(
            episode_indices=torch.tensor([int(cfg.test.episode_index)])
        )
        env.offload()

    torch.cuda.set_device(vla_device)
    from rlinf.models.embodiment.openvla_oft import get_model

    policy = get_model(vla_cfg, torch_dtype=torch.bfloat16).to(vla_device).eval()

    try:
        if observation is None:
            observation, _ = env.reset(
                episode_indices=torch.tensor([int(cfg.test.episode_index)])
            )
        actions = [_predict_action(policy, observation)]
        _activate_world_model(policy, env, collocated=collocated)
        chunk0 = env.chunk_step(actions[0].numpy())
        observation_after_chunk0 = chunk0[0][0]

        snapshot_context = _snapshot_context(env, chunk_index=1)
        snapshot = env.snapshot_resume_state(snapshot_context)
        env.validate_resume_state(snapshot, snapshot_context)
        env.assert_cpu_only(snapshot, "resume_state")

        _activate_policy(policy, env, vla_device, collocated=collocated)
        actions.append(_predict_action(policy, observation_after_chunk0))
        _activate_world_model(policy, env, collocated=collocated)
        reference_output = _to_cpu(env.chunk_step(actions[1].numpy()))
        reference_context = _snapshot_context(env, chunk_index=2)
        reference_state = env.snapshot_resume_state(reference_context)

        env.offload()
        prepared = env.prepare_resume_state(snapshot, snapshot_context)
        env.commit_resume_state(prepared)
        resumed_output = _to_cpu(env.chunk_step(actions[1].numpy()))
        resumed_context = _snapshot_context(env, chunk_index=2)
        resumed_state = env.snapshot_resume_state(resumed_context)

        rtol = float(cfg.test.tensor_rtol)
        atol = float(cfg.test.tensor_atol)
        _assert_nested_close(
            resumed_output,
            reference_output,
            path="chunk_1_output",
            rtol=rtol,
            atol=atol,
        )
        _assert_nested_close(
            resumed_state,
            reference_state,
            path="chunk_1_state",
            rtol=rtol,
            atol=atol,
        )
        print("Task 1 Wan snapshot/resume E2E passed")
        print(f"Placement: {placement} ({vla_device}, {wm_device})")
        print(f"VLA checkpoint: {vla_checkpoint}")
        print(f"Wan checkpoint: {wm_checkpoint}")
        print(f"Snapshot current_obs shape: {tuple(snapshot.current_obs.shape)}")
    finally:
        policy.to("cpu")
        env.offload()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
