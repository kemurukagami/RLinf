"""Benchmark a resident disaggregated OpenVLA-OFT and Wan rollout pair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from rlinf.scheduler import Worker

ITERATION_FIELDS = (
    "iteration",
    "vla_action_generation_ms",
    "action_handoff_ms",
    "wm_observation_generation_ms",
    "observation_handoff_ms",
    "cycle_ms",
    "action_payload_bytes",
    "observation_payload_bytes",
    "action_source_device",
    "action_destination_device",
    "observation_source_device",
    "observation_destination_device",
    "vla_memory_allocated_mib",
    "wm_memory_allocated_mib",
    "environment_elapsed_steps",
)


def _require_checkpoint(path: Path, required_files: tuple[str, ...]) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    missing = [name for name in required_files if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint {path} is missing required entries: {', '.join(missing)}"
        )


def _payload_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_bytes(item) for item in value)
    return 0


def _scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("expected a scalar tensor")
        return int(value.item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("expected a scalar array")
        return int(value.item())
    return int(value)


def _copy_observation_to_device(
    observation: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            copied[key] = value.to(device=device, non_blocking=True)
        elif isinstance(value, dict):
            copied[key] = _copy_observation_to_device(value, device)
        else:
            copied[key] = value
    return copied


def _timed_observation_handoff(
    observation: dict[str, Any],
    *,
    source_device: torch.device,
    destination_device: torch.device,
) -> tuple[dict[str, Any], float]:
    torch.cuda.synchronize(source_device)
    torch.cuda.synchronize(destination_device)
    started = time.perf_counter_ns()
    copied = _copy_observation_to_device(observation, destination_device)
    torch.cuda.synchronize(destination_device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return copied, elapsed_ms


def _predict_action(policy: Any, observation: dict[str, Any]) -> torch.Tensor:
    actions, _ = policy.predict_action_batch(
        env_obs=observation,
        do_sample=False,
        calculate_logprobs=False,
        calculate_values=False,
    )
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    return actions.detach().contiguous()


def _timed_vla_action_generation(
    policy: Any,
    observation: dict[str, Any],
    *,
    vla_device: torch.device,
) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize(vla_device)
    started = time.perf_counter_ns()
    actions = _predict_action(policy, observation)
    torch.cuda.synchronize(vla_device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return actions, elapsed_ms


def _timed_action_handoff(
    actions: torch.Tensor,
    *,
    wm_device: torch.device,
) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize(wm_device)
    started = time.perf_counter_ns()
    copied = actions.to(device=wm_device, non_blocking=True)
    torch.cuda.synchronize(wm_device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return copied, elapsed_ms


def _timed_world_model_step(
    env: Any,
    actions: torch.Tensor,
    *,
    wm_device: torch.device,
) -> tuple[dict[str, Any], float]:
    torch.cuda.synchronize(wm_device)
    started = time.perf_counter_ns()
    chunk_result = env.chunk_step(actions)
    torch.cuda.synchronize(wm_device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    observation = chunk_result[0][0]
    return observation, elapsed_ms


def _load_configs(repo_path: Path, cfg: Any) -> tuple[Any, Any]:
    env_cfg = OmegaConf.load(
        repo_path / "examples/embodiment/config/env/wan_libero_spatial.yaml"
    )
    wm_checkpoint = Path(cfg.models.world_model.checkpoint)
    total_chunks = int(cfg.benchmark.warmup_iterations) + int(cfg.benchmark.iterations)
    max_steps = (total_chunks + 2) * int(env_cfg.chunk)
    with open_dict(env_cfg):
        env_cfg.total_num_envs = int(cfg.benchmark.num_envs)
        env_cfg.group_size = 1
        env_cfg.auto_reset = False
        env_cfg.ignore_terminations = True
        env_cfg.max_episode_steps = max_steps
        env_cfg.max_steps_per_rollout_epoch = max_steps
        env_cfg.video_cfg.save_video = False
        env_cfg.enable_offload = False
        env_cfg.num_inference_steps = int(cfg.models.world_model.num_inference_steps)
        env_cfg.wan_wm_hf_ckpt_path = str(wm_checkpoint)
        env_cfg.VAE_path = str(wm_checkpoint / "Wan2.2_VAE.pth")
        env_cfg.model_path = str(wm_checkpoint / "model-00001.safetensors")
        env_cfg.initial_image_path = str(wm_checkpoint / "dataset")
        env_cfg.reward_model.from_pretrained = str(wm_checkpoint / "resnet_rm.pth")

    vla_cfg = OmegaConf.load(
        repo_path / "examples/embodiment/config/model/openvla_oft.yaml"
    )
    with open_dict(vla_cfg):
        vla_cfg.model_path = str(cfg.models.vla.checkpoint)
        vla_cfg.precision = str(cfg.models.vla.precision)
        vla_cfg.unnorm_key = str(cfg.models.vla.unnorm_key)
        vla_cfg.max_prompt_length = 128
        vla_cfg.attn_implementation = "eager"
    return env_cfg, vla_cfg


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "mean_ms": statistics.fmean(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "p50_ms": _percentile(values, 50),
        "p90_ms": _percentile(values, 90),
        "p95_ms": _percentile(values, 95),
        "p99_ms": _percentile(values, 99),
        "max_ms": max(values),
    }


def _write_summary_csv(
    output_path: Path, summaries: dict[str, dict[str, float]]
) -> None:
    columns = (
        "metric",
        "count",
        "mean_ms",
        "std_ms",
        "min_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for metric, summary in summaries.items():
            writer.writerow({"metric": metric, **summary})


def _write_plots(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iteration = np.asarray([row["iteration"] for row in rows])
    vla = np.asarray([row["vla_action_generation_ms"] for row in rows])
    action_handoff = np.asarray([row["action_handoff_ms"] for row in rows])
    wm = np.asarray([row["wm_observation_generation_ms"] for row in rows])
    obs_handoff = np.asarray([row["observation_handoff_ms"] for row in rows])

    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(iteration, vla, label="VLA action generation", linewidth=1)
    axes[0].plot(iteration, wm, label="Wan observation generation", linewidth=1)
    axes[0].set_ylabel("Compute latency (ms)")
    axes[0].set_title("Disaggregated OpenVLA-OFT + Wan per-iteration latency")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        iteration, action_handoff, label="Action handoff (CPU→WM GPU)", linewidth=1
    )
    axes[1].plot(
        iteration,
        obs_handoff,
        label="Observation handoff (WM GPU→VLA GPU)",
        linewidth=1,
    )
    axes[1].set_xlabel("Measured iteration")
    axes[1].set_ylabel("Handoff latency (ms)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "latency_timeseries.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].boxplot([vla, wm], tick_labels=["VLA action", "Wan observation"])
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title("Compute latency distribution")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].boxplot(
        [action_handoff, obs_handoff],
        tick_labels=["Action handoff", "Observation handoff"],
    )
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Handoff latency distribution")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "latency_distributions.png", dpi=180)
    plt.close(figure)

    labels = [
        "VLA action generation",
        "Action handoff",
        "Wan observation generation",
        "Observation handoff",
    ]
    means = [
        summaries["vla_action_generation"]["mean_ms"],
        summaries["action_handoff"]["mean_ms"],
        summaries["wm_observation_generation"]["mean_ms"],
        summaries["observation_handoff"]["mean_ms"],
    ]
    figure, axis = plt.subplots(figsize=(12, 5))
    left = 0.0
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B"]
    for label, value, color in zip(labels, means, colors):
        axis.barh(["Mean cycle"], [value], left=left, label=label, color=color)
        if value / sum(means) > 0.03:
            axis.text(left + value / 2, 0, f"{value:.1f} ms", ha="center", va="center")
        left += value
    axis.set_xlabel("Mean latency contribution (ms)")
    axis.set_title("Mean synchronous cycle breakdown")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "mean_cycle_breakdown.png", dpi=180)
    plt.close(figure)


def _write_report(
    output_dir: Path,
    *,
    cfg: Any,
    summaries: dict[str, dict[str, float]],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    vla_mean = summaries["vla_action_generation"]["mean_ms"]
    wm_mean = summaries["wm_observation_generation"]["mean_ms"]
    cycle_mean = summaries["cycle"]["mean_ms"]
    ratio = wm_mean / vla_mean if vla_mean > 0 else math.inf
    chunk = int(metadata["action_chunk_length"])
    chunks_per_second = 1000.0 / cycle_mean
    environment_steps_per_second = chunk * chunks_per_second
    compute_total = vla_mean + wm_mean
    handoff_total = (
        summaries["action_handoff"]["mean_ms"]
        + summaries["observation_handoff"]["mean_ms"]
    )
    vla_compute_share = 100.0 * vla_mean / cycle_mean
    wm_compute_share = 100.0 * wm_mean / cycle_mean
    ideal_overlapped_cycle_ms = max(vla_mean, wm_mean)
    ideal_pipeline_speedup = cycle_mean / ideal_overlapped_cycle_ms
    derived_rows = (
        ("wan_to_vla_mean_latency_ratio", ratio, "ratio"),
        ("handoff_share_of_cycle", 100.0 * handoff_total / cycle_mean, "percent"),
        ("vla_compute_share_of_cycle", vla_compute_share, "percent"),
        ("wm_compute_share_of_cycle", wm_compute_share, "percent"),
        ("implied_vla_idle_share_sequential", 100.0 - vla_compute_share, "percent"),
        ("implied_wm_idle_share_sequential", 100.0 - wm_compute_share, "percent"),
        ("ideal_overlapped_cycle", ideal_overlapped_cycle_ms, "ms"),
        ("ideal_pipeline_speedup_upper_bound", ideal_pipeline_speedup, "ratio"),
        ("action_chunks_per_second", chunks_per_second, "chunks/s"),
        ("environment_actions_per_second", environment_steps_per_second, "actions/s"),
    )
    with (output_dir / "derived_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value", "unit"))
        writer.writerows(derived_rows)
    report = f"""# Disaggregated OpenVLA-OFT + Wan timing experiment

## Configuration

- Measured iterations: {int(cfg.benchmark.iterations)}
- Warmup iterations: {int(cfg.benchmark.warmup_iterations)}
- Environments per model call: {int(cfg.benchmark.num_envs)}
- Action chunk length: {chunk}
- Wan inference steps: {int(cfg.models.world_model.num_inference_steps)}
- VLA logical device: `{cfg.models.vla.device}`
- Wan logical device: `{cfg.models.world_model.device}`
- `CUDA_VISIBLE_DEVICES`: `{metadata["cuda_visible_devices"]}`
- VLA GPU: {metadata["vla_gpu_name"]}
- Wan GPU: {metadata["wm_gpu_name"]}

Both models remained resident throughout warmup and measurement. Timings use
`perf_counter_ns()` with CUDA synchronization before and after each measured
boundary.

OpenVLA-OFT currently decodes action tokens through NumPy and returns the action
chunk as a CPU tensor. Therefore `action_handoff_ms` measures the actual
CPU-to-Wan-GPU handoff in this implementation. `observation_handoff_ms` measures
the generated observation tensor copy from the Wan GPU to the VLA GPU. It does
not include Ray actor or RLinf Channel queue latency.

## Principal results

| Metric | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| VLA action generation | {vla_mean:.3f} | {summaries["vla_action_generation"]["p50_ms"]:.3f} | {summaries["vla_action_generation"]["p95_ms"]:.3f} | {summaries["vla_action_generation"]["p99_ms"]:.3f} |
| Action handoff | {summaries["action_handoff"]["mean_ms"]:.3f} | {summaries["action_handoff"]["p50_ms"]:.3f} | {summaries["action_handoff"]["p95_ms"]:.3f} | {summaries["action_handoff"]["p99_ms"]:.3f} |
| Wan observation generation | {wm_mean:.3f} | {summaries["wm_observation_generation"]["p50_ms"]:.3f} | {summaries["wm_observation_generation"]["p95_ms"]:.3f} | {summaries["wm_observation_generation"]["p99_ms"]:.3f} |
| Observation handoff | {summaries["observation_handoff"]["mean_ms"]:.3f} | {summaries["observation_handoff"]["p50_ms"]:.3f} | {summaries["observation_handoff"]["p95_ms"]:.3f} | {summaries["observation_handoff"]["p99_ms"]:.3f} |
| Complete synchronous cycle | {cycle_mean:.3f} | {summaries["cycle"]["p50_ms"]:.3f} | {summaries["cycle"]["p95_ms"]:.3f} | {summaries["cycle"]["p99_ms"]:.3f} |

- Wan/VLA mean latency ratio: **{ratio:.2f}×**
- Mean handoff overhead: **{handoff_total:.3f} ms**
- Mean model compute: **{compute_total:.3f} ms**
- Handoff share of measured cycle: **{100.0 * handoff_total / cycle_mean:.4f}%**
- VLA compute share of sequential cycle: **{vla_compute_share:.2f}%**
- Wan compute share of sequential cycle: **{wm_compute_share:.2f}%**
- Implied VLA-GPU idle share without overlap: **{100.0 - vla_compute_share:.2f}%**
- Implied Wan-GPU idle share without overlap: **{100.0 - wm_compute_share:.2f}%**
- Ideal two-stage overlap upper bound: **{ideal_pipeline_speedup:.3f}×** throughput
- Sustained measured rate: **{chunks_per_second:.4f} action chunks/s**
- Equivalent environment-action rate: **{environment_steps_per_second:.4f} actions/s**
- Action payload: **{rows[0]["action_payload_bytes"]} bytes**
- Observation tensor payload: **{rows[0]["observation_payload_bytes"]} bytes**

## Artifacts

- `iterations.csv`: spreadsheet-compatible per-iteration measurements.
- `summary.csv`: aggregate distribution statistics.
- `derived_metrics.csv`: ratios, implied utilization, and throughput metrics.
- `metadata.json`: exact runtime, device, model, and timing metadata.
- `latency_timeseries.png`: per-iteration compute and handoff timings.
- `latency_distributions.png`: compute and handoff distributions.
- `mean_cycle_breakdown.png`: mean synchronous critical-path breakdown.

## Interpretation limits

- This is a resident, synchronous, single-process/two-GPU disaggregated model
  benchmark. It isolates model computation and tensor movement.
- It excludes Ray scheduling, serialization, Channel actor queueing, and
  multi-process collective setup.
- Idle shares are critical-path implications, not direct GPU utilization
  samples. They assume the two model stages remain sequential and do not run
  unrelated work concurrently.
- Wan uses the configured reduced diffusion-step count, so results must not be
  generalized to the default five-step quality setting without another run.
- One environment is intentionally used to obtain several hundred real model
  cycles at bounded cost; batching changes both VLA and Wan latency.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def run(
    config_path: Path,
    output_override: Path | None,
    iterations_override: int | None,
    warmup_override: int | None,
) -> None:
    repo_path = Path(__file__).resolve().parents[3]
    workspace_root = repo_path.parent
    cfg = OmegaConf.load(config_path)
    with open_dict(cfg):
        if iterations_override is not None:
            cfg.benchmark.iterations = iterations_override
        if warmup_override is not None:
            cfg.benchmark.warmup_iterations = warmup_override
    output_dir = (
        output_override.resolve()
        if output_override is not None
        else (workspace_root / str(cfg.benchmark.output_dir)).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    iterations = int(cfg.benchmark.iterations)
    warmup_iterations = int(cfg.benchmark.warmup_iterations)
    if iterations < 1:
        raise ValueError("benchmark.iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("benchmark.warmup_iterations must be non-negative")
    if int(cfg.benchmark.num_envs) != 1:
        raise ValueError("this benchmark currently requires benchmark.num_envs=1")

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
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("benchmark requires two visible CUDA GPUs")

    vla_device = torch.device(str(cfg.models.vla.device))
    wm_device = torch.device(str(cfg.models.world_model.device))
    if vla_device == wm_device:
        raise ValueError("VLA and world model must use different logical GPUs")

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.manual_seed(int(cfg.benchmark.seed))
    np.random.seed(int(cfg.benchmark.seed))
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
    observation, _ = env.reset(
        episode_indices=torch.tensor([int(cfg.benchmark.episode_index)])
    )

    torch.cuda.set_device(vla_device)
    from rlinf.models.embodiment.openvla_oft import get_model

    policy = get_model(vla_cfg, torch_dtype=torch.bfloat16).to(vla_device).eval()
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "vla_logical_device": str(vla_device),
        "wm_logical_device": str(wm_device),
        "vla_gpu_name": torch.cuda.get_device_name(vla_device),
        "wm_gpu_name": torch.cuda.get_device_name(wm_device),
        "vla_checkpoint": str(vla_checkpoint),
        "wm_checkpoint": str(wm_checkpoint),
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "num_envs": 1,
        "action_chunk_length": int(env_cfg.chunk),
        "action_dim": int(vla_cfg.action_dim),
        "wan_inference_steps": int(env_cfg.num_inference_steps),
        "timing_clock": "time.perf_counter_ns",
        "timing_synchronization": "CUDA synchronize before and after each boundary",
        "action_handoff_transport": "OpenVLA CPU output tensor to Wan GPU",
        "observation_handoff_transport": "Wan GPU tensor to VLA GPU via torch.Tensor.to",
        "ray_channel_overhead_included": False,
    }

    rows: list[dict[str, Any]] = []
    csv_path = output_dir / "iterations.csv"
    try:
        observation_on_vla, _ = _timed_observation_handoff(
            observation,
            source_device=wm_device,
            destination_device=vla_device,
        )
        for warmup_index in range(warmup_iterations):
            actions, _ = _timed_vla_action_generation(
                policy, observation_on_vla, vla_device=vla_device
            )
            actions_on_wm, _ = _timed_action_handoff(actions, wm_device=wm_device)
            observation, _ = _timed_world_model_step(
                env, actions_on_wm, wm_device=wm_device
            )
            observation_on_vla, _ = _timed_observation_handoff(
                observation,
                source_device=wm_device,
                destination_device=vla_device,
            )
            print(f"Warmup {warmup_index + 1}/{warmup_iterations} complete", flush=True)

        with csv_path.open("w", newline="", encoding="utf-8") as csv_stream:
            writer = csv.DictWriter(csv_stream, fieldnames=ITERATION_FIELDS)
            writer.writeheader()
            for iteration in range(iterations):
                actions, vla_ms = _timed_vla_action_generation(
                    policy, observation_on_vla, vla_device=vla_device
                )
                if actions.device.type != "cpu":
                    raise AssertionError(
                        "OpenVLA action transport metadata expects the current CPU action output"
                    )
                if not torch.isfinite(actions).all():
                    raise AssertionError("VLA produced non-finite actions")
                action_bytes = _payload_bytes(actions)
                actions_on_wm, action_handoff_ms = _timed_action_handoff(
                    actions, wm_device=wm_device
                )
                if actions_on_wm.device != wm_device:
                    raise AssertionError("action handoff did not reach the Wan GPU")

                observation, wm_ms = _timed_world_model_step(
                    env, actions_on_wm, wm_device=wm_device
                )
                observation_bytes = _payload_bytes(observation)
                observation_on_vla, observation_handoff_ms = _timed_observation_handoff(
                    observation,
                    source_device=wm_device,
                    destination_device=vla_device,
                )
                main_images = observation_on_vla.get("main_images")
                if not isinstance(main_images, torch.Tensor):
                    raise AssertionError("Wan observation missing main_images tensor")
                if main_images.device != vla_device:
                    raise AssertionError(
                        "observation handoff did not reach the VLA GPU"
                    )

                cycle_ms = vla_ms + action_handoff_ms + wm_ms + observation_handoff_ms
                row = {
                    "iteration": iteration,
                    "vla_action_generation_ms": vla_ms,
                    "action_handoff_ms": action_handoff_ms,
                    "wm_observation_generation_ms": wm_ms,
                    "observation_handoff_ms": observation_handoff_ms,
                    "cycle_ms": cycle_ms,
                    "action_payload_bytes": action_bytes,
                    "observation_payload_bytes": observation_bytes,
                    "action_source_device": str(actions.device),
                    "action_destination_device": str(actions_on_wm.device),
                    "observation_source_device": str(wm_device),
                    "observation_destination_device": str(main_images.device),
                    "vla_memory_allocated_mib": torch.cuda.memory_allocated(vla_device)
                    / (1024**2),
                    "wm_memory_allocated_mib": torch.cuda.memory_allocated(wm_device)
                    / (1024**2),
                    "environment_elapsed_steps": _scalar_int(env.elapsed_steps),
                }
                rows.append(row)
                writer.writerow(row)
                csv_stream.flush()
                if (iteration + 1) % int(cfg.benchmark.progress_interval) == 0:
                    print(
                        f"Measured {iteration + 1}/{iterations}: "
                        f"VLA={vla_ms:.2f} ms, action={action_handoff_ms:.3f} ms, "
                        f"Wan={wm_ms:.2f} ms, obs={observation_handoff_ms:.3f} ms",
                        flush=True,
                    )

        metric_columns = {
            "vla_action_generation": "vla_action_generation_ms",
            "action_handoff": "action_handoff_ms",
            "wm_observation_generation": "wm_observation_generation_ms",
            "observation_handoff": "observation_handoff_ms",
            "cycle": "cycle_ms",
        }
        summaries = {
            metric: _metric_summary([float(row[column]) for row in rows])
            for metric, column in metric_columns.items()
        }
        _write_summary_csv(output_dir / "summary.csv", summaries)
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
        _write_plots(output_dir, rows, summaries)
        _write_report(
            output_dir,
            cfg=cfg,
            summaries=summaries,
            rows=rows,
            metadata=metadata,
        )
        print(f"Benchmark complete: {output_dir}", flush=True)
    finally:
        policy.to("cpu")
        env.offload()
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    args = parser.parse_args()
    run(
        args.config.resolve(),
        args.output_dir,
        args.iterations,
        args.warmup_iterations,
    )


if __name__ == "__main__":
    main()
