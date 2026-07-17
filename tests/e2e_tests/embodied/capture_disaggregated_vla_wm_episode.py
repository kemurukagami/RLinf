"""Capture a complete disaggregated OpenVLA-OFT + Wan episode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace

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
from capture_disaggregated_vla_wm_rollout_sample import _timed, _timed_handoff
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw

from rlinf.scheduler import Worker


def _write_episode_overview(images: list[tuple[str, Image.Image]], path: Path) -> None:
    columns = 5
    label_height = 28
    width, height = images[0][1].size
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 8, y + 7), label, fill="black")
    sheet.save(path)


def capture(config_path: Path, output_dir: Path, episode_index: int) -> None:
    repo_path = Path(__file__).resolve().parents[3]
    cfg = OmegaConf.load(config_path)
    nominal_horizon = 240
    chunk_length = 8
    max_chunks = nominal_horizon // chunk_length
    with open_dict(cfg):
        cfg.benchmark.iterations = max_chunks
        cfg.benchmark.warmup_iterations = 0
        cfg.benchmark.episode_index = episode_index
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    sheets_dir = output_dir / "chunk_contact_sheets"
    frames_dir.mkdir(exist_ok=True)
    sheets_dir.mkdir(exist_ok=True)

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
        raise RuntimeError("episode capture requires exactly two visible CUDA GPUs")
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
    with open_dict(env_cfg):
        env_cfg.max_episode_steps = nominal_horizon
        env_cfg.max_steps_per_rollout_epoch = nominal_horizon
        env_cfg.auto_reset = False

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
    initial_image.save(frames_dir / "frame_000_initial.png")
    overview_images = [("Initial", initial_image)]

    torch.cuda.set_device(vla_device)
    from rlinf.models.embodiment.openvla_oft import get_model

    policy = get_model(vla_cfg, torch_dtype=torch.bfloat16).to(vla_device).eval()
    action_chunks: list[np.ndarray] = []
    generated_chunks: list[np.ndarray] = []
    reward_chunks: list[np.ndarray] = []
    termination_chunks: list[np.ndarray] = []
    truncation_chunks: list[np.ndarray] = []
    timing_rows: list[dict[str, float | int | bool]] = []
    termination_reason = "max_chunks"
    try:
        for chunk_index in range(max_chunks):
            input_image = _uint8_observation_to_image(observation_wm["main_images"][0])
            observation_vla, observation_handoff_ms = _timed_handoff(
                wm_device,
                vla_device,
                lambda: _copy_observation_to_device(observation_wm, vla_device),
            )
            actions, vla_generation_ms = _timed(
                vla_device, lambda: _predict_action(policy, observation_vla)
            )
            if tuple(actions.shape) != (1, chunk_length, len(ACTION_COLUMNS)):
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
            generated = env.current_obs[0, :, 0, -chunk_length:].detach().float().cpu()
            rewards = chunk_result[1][0].detach().float().cpu()
            terminations = chunk_result[2][0].detach().cpu()
            truncations = chunk_result[3][0].detach().cpu()

            chunk_images = [(f"Chunk {chunk_index} input", input_image)]
            for step in range(chunk_length):
                global_step = chunk_index * chunk_length + step + 1
                image = _frame_to_image(generated[:, step])
                image.save(frames_dir / f"frame_{global_step:03d}_generated.png")
                chunk_images.append((f"Step {global_step}", image))
            _write_contact_sheet(
                chunk_images, sheets_dir / f"chunk_{chunk_index:02d}.png"
            )
            overview_images.append(
                (f"After step {(chunk_index + 1) * chunk_length}", chunk_images[-1][1])
            )

            action_chunks.append(actions_cpu[0].numpy())
            generated_chunks.append(generated.numpy())
            reward_chunks.append(rewards.numpy())
            termination_chunks.append(terminations.numpy())
            truncation_chunks.append(truncations.numpy())
            cycle_ms = (
                observation_handoff_ms
                + vla_generation_ms
                + action_handoff_ms
                + wm_generation_ms
            )
            timing_rows.append(
                {
                    "chunk": chunk_index,
                    "first_action_step": chunk_index * chunk_length,
                    "last_action_step": (chunk_index + 1) * chunk_length - 1,
                    "vla_action_generation_ms": vla_generation_ms,
                    "action_handoff_ms": action_handoff_ms,
                    "wm_observation_generation_ms": wm_generation_ms,
                    "observation_handoff_ms": observation_handoff_ms,
                    "cycle_ms": cycle_ms,
                    "terminated": bool(terminations.any()),
                    "truncated": bool(truncations.any()),
                }
            )
            observation_wm = chunk_result[0][0]
            print(
                f"Captured chunk {chunk_index + 1}/{max_chunks}, "
                f"elapsed actions={(chunk_index + 1) * chunk_length}, "
                f"cycle={cycle_ms:.1f} ms",
                flush=True,
            )
            if terminations.any():
                termination_reason = "success"
                break
            if truncations.any():
                termination_reason = "horizon_truncation"
                break

        actions_array = np.stack(action_chunks)
        frames_array = np.stack(generated_chunks)
        rewards_array = np.stack(reward_chunks)
        terminations_array = np.stack(termination_chunks)
        truncations_array = np.stack(truncation_chunks)
        completed_chunks = len(action_chunks)
        completed_actions = completed_chunks * chunk_length

        with (output_dir / "actions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(("global_step", "chunk", "step_in_chunk", *ACTION_COLUMNS))
            for chunk_index, chunk in enumerate(actions_array):
                for step, action in enumerate(chunk):
                    writer.writerow(
                        (chunk_index * chunk_length + step, chunk_index, step, *action)
                    )
        with (output_dir / "rewards.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "global_step",
                    "chunk",
                    "step_in_chunk",
                    "relative_reward",
                    "termination",
                    "truncation",
                )
            )
            for chunk_index in range(completed_chunks):
                for step in range(chunk_length):
                    writer.writerow(
                        (
                            chunk_index * chunk_length + step,
                            chunk_index,
                            step,
                            rewards_array[chunk_index, step],
                            terminations_array[chunk_index, step],
                            truncations_array[chunk_index, step],
                        )
                    )
        with (output_dir / "chunk_timings.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(timing_rows[0].keys()))
            writer.writeheader()
            writer.writerows(timing_rows)

        np.savez_compressed(
            output_dir / "episode.npz",
            actions=actions_array,
            generated_frames=frames_array,
            rewards=rewards_array,
            terminations=terminations_array,
            truncations=truncations_array,
        )
        _write_episode_overview(overview_images, output_dir / "episode_overview.png")

        cycle_values = np.asarray([float(row["cycle_ms"]) for row in timing_rows])
        summary = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "wm_logical_device": str(wm_device),
            "wm_gpu": torch.cuda.get_device_name(wm_device),
            "vla_logical_device": str(vla_device),
            "vla_gpu": torch.cuda.get_device_name(vla_device),
            "episode_index": episode_index,
            "task_description": task_description,
            "termination_reason": termination_reason,
            "nominal_horizon_actions": nominal_horizon,
            "completed_chunks": completed_chunks,
            "completed_actions": completed_actions,
            "action_chunk_length": chunk_length,
            "wan_inference_steps": int(env.num_inference_steps),
            "action_tensor_shape": list(actions_array.shape),
            "generated_frame_tensor_shape": list(frames_array.shape),
            "total_measured_rollout_ms": float(cycle_values.sum()),
            "mean_chunk_ms": float(cycle_values.mean()),
            "p95_chunk_ms": float(np.percentile(cycle_values, 95)),
        }
        (output_dir / "episode_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        readme = f"""# Complete disaggregated OpenVLA-OFT + Wan episode

Task: **{task_description}**

This episode contains **{completed_actions} actions in {completed_chunks}
chunks** and ended by **{termination_reason.replace("_", " ")}**. OpenVLA-OFT
remained resident on logical `{vla_device}` and Wan remained resident on
logical `{wm_device}` throughout the rollout.

- `actions.csv`: every 7-DoF action in episode order.
- `rewards.csv`: every predicted reward and done flag.
- `chunk_timings.csv`: disaggregated latency for every 8-action chunk.
- `frames/`: initial frame plus every world-model-predicted camera frame.
- `chunk_contact_sheets/`: input and eight outputs for each chunk.
- `episode_overview.png`: initial frame and every eighth generated frame.
- `episode.npz`: machine-readable actions, frames, rewards, and done flags.
- `episode_summary.json`: task, devices, completion reason, shapes, and timing.

The generated frames are Wan predictions, not observations from a physical
simulator. Wan used one inference step, matching the latency benchmark.
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
