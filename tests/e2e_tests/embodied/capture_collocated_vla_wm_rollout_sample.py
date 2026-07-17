"""Capture one inspectable OpenVLA-OFT + Wan rollout chunk."""

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
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw

from rlinf.scheduler import Worker

ACTION_COLUMNS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _timed(device: torch.device, operation: Any) -> tuple[Any, float]:
    _sync(device)
    started = time.perf_counter_ns()
    result = operation()
    _sync(device)
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


def _frame_to_image(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().float().cpu().permute(1, 2, 0).numpy()
    array = np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _uint8_observation_to_image(frame: torch.Tensor) -> Image.Image:
    array = frame.detach().cpu().numpy().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _write_contact_sheet(images: list[tuple[str, Image.Image]], path: Path) -> None:
    label_height = 28
    width, height = images[0][1].size
    sheet = Image.new("RGB", (width * 3, (height + label_height) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        x = (index % 3) * width
        y = (index // 3) * (height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 8, y + 7), label, fill="black")
    sheet.save(path)


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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("capture requires exactly one visible CUDA GPU")

    device = torch.device(str(cfg.models.vla.device))
    if str(device) != str(cfg.models.world_model.device):
        raise ValueError("capture requires a collocated logical device")
    torch.manual_seed(int(cfg.benchmark.seed))
    np.random.seed(int(cfg.benchmark.seed))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.cuda.set_device(device)
    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda
    env_cfg, vla_cfg = _load_configs(repo_path, cfg)

    from rlinf.envs.world_model.world_model_wan_env import WanEnv

    env = WanEnv(
        env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        record_metrics=True,
        worker_info=SimpleNamespace(rank=0, group_world_size=1),
    )
    env.device = device
    observation, _ = env.reset(episode_indices=torch.tensor([episode_index]))
    task_description = str(observation["task_descriptions"][0])
    initial_image = _uint8_observation_to_image(observation["main_images"][0])
    initial_image.save(frames_dir / "frame_00_initial.png")
    observation_cpu = _copy_observation_to_device(observation, torch.device("cpu"))
    env.offload()
    torch.cuda.empty_cache()

    from rlinf.models.embodiment.openvla_oft import get_model

    policy = get_model(vla_cfg, torch_dtype=torch.bfloat16).to(device).eval()
    policy.to("cpu")
    torch.cuda.empty_cache()
    try:
        _, vla_onload_ms = _timed(device, lambda: policy.to(device))
        observation_vla, observation_handoff_ms = _timed(
            device, lambda: _copy_observation_to_device(observation_cpu, device)
        )
        actions, vla_generation_ms = _timed(
            device, lambda: _predict_action(policy, observation_vla)
        )
        if tuple(actions.shape) != (1, int(env.chunk), len(ACTION_COLUMNS)):
            raise ValueError(f"unexpected action shape: {tuple(actions.shape)}")
        actions_cpu = actions.detach().float().cpu().contiguous()
        _, vla_offload_ms = _timed(device, lambda: policy.to("cpu"))
        torch.cuda.empty_cache()

        _, wm_onload_ms = _timed(device, env.onload)
        actions_wm, action_handoff_ms = _timed(device, lambda: actions_cpu.to(device))
        chunk_result, wm_generation_ms = _timed(
            device, lambda: env.chunk_step(actions_wm)
        )
        generated = env.current_obs[0, :, 0, -int(env.chunk) :].detach().cpu()
        rewards = chunk_result[1][0].detach().float().cpu()
        terminations = chunk_result[2][0].detach().cpu()
        truncations = chunk_result[3][0].detach().cpu()
        _, wm_offload_ms = _timed(device, env.offload)
        torch.cuda.empty_cache()

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
            "vla_onload_ms": vla_onload_ms,
            "observation_handoff_ms": observation_handoff_ms,
            "vla_action_generation_ms": vla_generation_ms,
            "vla_offload_ms": vla_offload_ms,
            "wm_onload_ms": wm_onload_ms,
            "action_handoff_ms": action_handoff_ms,
            "wm_observation_generation_ms": wm_generation_ms,
            "wm_offload_ms": wm_offload_ms,
        }
        timings["complete_cycle_ms"] = sum(timings.values())
        metadata = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
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
        readme = f"""# Example collocated rollout chunk

Task: **{task_description}**

This sample contains one OpenVLA-OFT action chunk with 8 actions and the eight
camera frames predicted by Wan from that chunk. `frame_00_initial.png` is the
input observation; frames 01–08 are world-model predictions, not images from a
physical simulator. The experiment used one Wan inference step.

- `actions.csv`: one 7-DoF action per generated frame.
- `rewards.csv`: world-model reward deltas and done flags.
- `frames/`: initial and intermediate camera frames.
- `camera_frames_contact_sheet.png`: all frames in temporal order.
- `rollout_chunk.npz`: machine-readable tensors.
- `metadata.json`: configuration and measured timing.

This is one rollout **chunk**, not a complete LIBERO episode. The benchmark
configuration defines a nominal 240-action episode horizon, or 30 such chunks
if it runs to truncation without earlier success.
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
