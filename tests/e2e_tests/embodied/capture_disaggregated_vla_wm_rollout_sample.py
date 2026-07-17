"""Capture one inspectable disaggregated OpenVLA-OFT + Wan rollout chunk."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from benchmark_disaggregated_vla_wm import (
    _copy_observation_to_device,
    _load_configs,
    _predict_action,
    _require_checkpoint,
)
from capture_collocated_vla_wm_rollout_sample import (
    ACTION_COLUMNS,
    _frame_to_image,
    _uint8_observation_to_image,
    _write_contact_sheet,
)
from omegaconf import OmegaConf, open_dict

from rlinf.scheduler import Worker


def _timed(device: torch.device, operation: Any) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    result = operation()
    torch.cuda.synchronize(device)
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


def _timed_handoff(
    source: torch.device, destination: torch.device, operation: Any
) -> tuple[Any, float]:
    torch.cuda.synchronize(source)
    torch.cuda.synchronize(destination)
    started = time.perf_counter_ns()
    result = operation()
    torch.cuda.synchronize(destination)
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


def capture(config_path: Path, output_dir: Path, episode_index: int) -> None:
    repo_path = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        cfg.benchmark.iterations = 1
        cfg.benchmark.warmup_iterations = 0
        cfg.benchmark.episode_index = episode_index
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    vla_checkpoint = Path(cfg.models.vla.checkpoint)
    wm_checkpoint = Path(cfg.models.world_model.checkpoint)
    _require_checkpoint(
        vla_checkpoint,
        ("config.json", "dataset_statistics.json", "model.safetensors.index.json"),
    )
    _require_checkpoint(
        wm_checkpoint,
        ("model-00001.safetensors", "Wan2.2_VAE.pth", "resnet_rm.pth", "dataset"),
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("capture requires exactly two visible CUDA GPUs")
    wm_device = torch.device(str(cfg.models.world_model.device))
    vla_device = torch.device(str(cfg.models.vla.device))
    if wm_device == vla_device:
        raise ValueError("disaggregated capture requires distinct logical devices")

    torch.manual_seed(int(cfg.benchmark.seed))
    np.random.seed(int(cfg.benchmark.seed))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda
    env_cfg, vla_cfg = _load_configs(repo_path, cfg)

    torch.cuda.set_device(wm_device)
    from rlinf.envs.world_model.world_model_wan_env import WanEnv

    env = WanEnv(
        env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        record_metrics=True,
        worker_info=SimpleNamespace(rank=0, group_world_size=1),
    )
    env.device = wm_device
    observation_wm, _ = env.reset(episode_indices=torch.tensor([episode_index]))
    task_description = str(observation_wm["task_descriptions"][0])
    initial_image = _uint8_observation_to_image(observation_wm["main_images"][0])
    initial_image.save(frames_dir / "frame_00_initial.png")

    torch.cuda.set_device(vla_device)
    from rlinf.models.embodiment.openvla_oft import get_model

    policy = get_model(vla_cfg, torch_dtype=torch.bfloat16).to(vla_device).eval()
    try:
        observation_vla, observation_handoff_ms = _timed_handoff(
            wm_device,
            vla_device,
            lambda: _copy_observation_to_device(observation_wm, vla_device),
        )
        actions, vla_generation_ms = _timed(
            vla_device, lambda: _predict_action(policy, observation_vla)
        )
        if tuple(actions.shape) != (1, int(env.chunk), len(ACTION_COLUMNS)):
            raise ValueError(f"unexpected action shape: {tuple(actions.shape)}")
        if actions.device.type != "cpu":
            raise ValueError(
                f"expected OpenVLA CPU action output, got {actions.device}"
            )
        actions_cpu = actions.detach().float().cpu().contiguous()
        actions_wm, action_handoff_ms = _timed_handoff(
            vla_device, wm_device, lambda: actions_cpu.to(wm_device)
        )
        chunk_result, wm_generation_ms = _timed(
            wm_device, lambda: env.chunk_step(actions_wm)
        )
        generated = env.current_obs[0, :, 0, -int(env.chunk) :].detach().cpu()
        rewards = chunk_result[1][0].detach().float().cpu()
        terminations = chunk_result[2][0].detach().cpu()
        truncations = chunk_result[3][0].detach().cpu()

        images = [("Initial camera frame", initial_image)]
        for index in range(generated.shape[1]):
            image = _frame_to_image(generated[:, index])
            image.save(frames_dir / f"frame_{index + 1:02d}_generated.png")
            images.append((f"Generated frame {index + 1}/8", image))
        _write_contact_sheet(images, output_dir / "camera_frames_contact_sheet.png")

        with (output_dir / "actions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", *ACTION_COLUMNS))
            for step, action in enumerate(actions_cpu[0].tolist()):
                writer.writerow((step, *action))
        with (output_dir / "rewards.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", "relative_reward", "termination", "truncation"))
            for step in range(int(env.chunk)):
                writer.writerow(
                    (
                        step,
                        float(rewards[step]),
                        bool(terminations[step]),
                        bool(truncations[step]),
                    )
                )
        np.savez_compressed(
            output_dir / "rollout_chunk.npz",
            actions=actions_cpu.numpy(),
            generated_frames=generated.numpy(),
            rewards=rewards.numpy(),
            terminations=terminations.numpy(),
            truncations=truncations.numpy(),
        )

        timings = {
            "vla_action_generation_ms": vla_generation_ms,
            "action_handoff_ms": action_handoff_ms,
            "wm_observation_generation_ms": wm_generation_ms,
            "observation_handoff_ms": observation_handoff_ms,
        }
        timings["complete_cycle_ms"] = sum(timings.values())
        metadata = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "wm_logical_device": str(wm_device),
            "wm_gpu": torch.cuda.get_device_name(wm_device),
            "vla_logical_device": str(vla_device),
            "vla_gpu": torch.cuda.get_device_name(vla_device),
            "episode_index": episode_index,
            "task_description": task_description,
            "action_shape": list(actions_cpu.shape),
            "generated_frame_shape": list(generated.shape),
            "action_chunk_length": int(env.chunk),
            "wan_inference_steps": int(env.num_inference_steps),
            "timings": timings,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        readme = f"""# Example disaggregated rollout chunk

Task: **{task_description}**

OpenVLA-OFT remained resident on logical `{vla_device}` and Wan remained
resident on logical `{wm_device}`. This sample contains one 8×7 OpenVLA action
chunk and the corresponding eight camera frames predicted by Wan.

- `frames/frame_00_initial.png`: input observation.
- `frames/frame_01_generated.png` through `frame_08_generated.png`: temporal
  world-model predictions, not frames from a physical simulator.
- `actions.csv`: one 7-DoF action per predicted frame.
- `rewards.csv`: predicted reward deltas and done flags.
- `camera_frames_contact_sheet.png`: all frames in temporal order.
- `rollout_chunk.npz`: machine-readable tensors.
- `metadata.json`: device placement and measured timing.

This is one rollout **chunk**, not a complete LIBERO episode. The standard Wan
configuration has a nominal 240-action horizon, or 30 chunks if no earlier
success occurs. Wan used one inference step for consistency with the timing
benchmark.
"""
        (output_dir / "README.md").write_text(readme, encoding="utf-8")
    finally:
        policy.to("cpu")
        env.offload()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()
    capture(args.config.resolve(), args.output_dir.resolve(), args.episode_index)


if __name__ == "__main__":
    main()
