"""Benchmark alternating OpenVLA-OFT and Wan residency on one GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from benchmark_disaggregated_vla_wm import (
    _copy_observation_to_device,
    _load_configs,
    _payload_bytes,
    _predict_action,
    _require_checkpoint,
    _scalar_int,
)
from omegaconf import OmegaConf, open_dict

from rlinf.scheduler import Worker

ITERATION_FIELDS = (
    "iteration",
    "vla_onload_ms",
    "observation_to_vla_ms",
    "vla_action_generation_ms",
    "vla_offload_ms",
    "wm_onload_ms",
    "action_handoff_ms",
    "wm_observation_generation_ms",
    "observation_to_cpu_ms",
    "wm_offload_ms",
    "observation_handoff_ms",
    "model_swap_ms",
    "cycle_ms",
    "action_payload_bytes",
    "observation_payload_bytes",
    "action_source_device",
    "action_destination_device",
    "observation_source_device",
    "observation_destination_device",
    "vla_resident_memory_mib",
    "wm_resident_memory_mib",
    "post_offload_memory_mib",
    "environment_elapsed_steps",
)


METRIC_COLUMNS = {
    "vla_onload": "vla_onload_ms",
    "observation_to_vla": "observation_to_vla_ms",
    "vla_action_generation": "vla_action_generation_ms",
    "vla_offload": "vla_offload_ms",
    "wm_onload": "wm_onload_ms",
    "action_handoff": "action_handoff_ms",
    "wm_observation_generation": "wm_observation_generation_ms",
    "observation_to_cpu": "observation_to_cpu_ms",
    "wm_offload": "wm_offload_ms",
    "observation_handoff": "observation_handoff_ms",
    "model_swap": "model_swap_ms",
    "cycle": "cycle_ms",
}


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


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _timed_policy_move(
    policy: Any,
    destination: torch.device,
    *,
    shared_device: torch.device,
) -> float:
    _sync(shared_device)
    started = time.perf_counter_ns()
    policy.to(destination)
    _sync(shared_device)
    if destination.type == "cpu":
        torch.cuda.empty_cache()
        _sync(shared_device)
    return (time.perf_counter_ns() - started) / 1_000_000.0


def _timed_environment_move(
    env: Any,
    *,
    resident: bool,
    shared_device: torch.device,
) -> float:
    _sync(shared_device)
    started = time.perf_counter_ns()
    if resident:
        env.onload()
    else:
        env.offload()
    _sync(shared_device)
    if not resident:
        torch.cuda.empty_cache()
        _sync(shared_device)
    return (time.perf_counter_ns() - started) / 1_000_000.0


def _timed_observation_copy(
    observation: dict[str, Any],
    destination: torch.device,
    *,
    shared_device: torch.device,
) -> tuple[dict[str, Any], float]:
    _sync(shared_device)
    started = time.perf_counter_ns()
    copied = _copy_observation_to_device(observation, destination)
    _sync(shared_device)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return copied, elapsed_ms


def _timed_vla_action_generation(
    policy: Any,
    observation: dict[str, Any],
    *,
    shared_device: torch.device,
) -> tuple[torch.Tensor, float]:
    _sync(shared_device)
    started = time.perf_counter_ns()
    actions = _predict_action(policy, observation)
    _sync(shared_device)
    return actions, (time.perf_counter_ns() - started) / 1_000_000.0


def _timed_action_handoff(
    actions: torch.Tensor,
    *,
    shared_device: torch.device,
) -> tuple[torch.Tensor, float]:
    _sync(shared_device)
    started = time.perf_counter_ns()
    copied = actions.to(device=shared_device, non_blocking=True)
    _sync(shared_device)
    return copied, (time.perf_counter_ns() - started) / 1_000_000.0


def _timed_world_model_step(
    env: Any,
    actions: torch.Tensor,
    *,
    shared_device: torch.device,
) -> tuple[dict[str, Any], float]:
    _sync(shared_device)
    started = time.perf_counter_ns()
    chunk_result = env.chunk_step(actions)
    _sync(shared_device)
    return chunk_result[0][0], (time.perf_counter_ns() - started) / 1_000_000.0


def _run_cycle(
    *,
    policy: Any,
    env: Any,
    observation_cpu: dict[str, Any],
    shared_device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vla_onload_ms = _timed_policy_move(
        policy, shared_device, shared_device=shared_device
    )
    vla_resident_memory_mib = torch.cuda.memory_allocated(shared_device) / (1024**2)
    observation_vla, observation_to_vla_ms = _timed_observation_copy(
        observation_cpu, shared_device, shared_device=shared_device
    )
    actions, vla_ms = _timed_vla_action_generation(
        policy, observation_vla, shared_device=shared_device
    )
    if actions.device.type != "cpu":
        raise AssertionError(
            "OpenVLA action transport metadata expects the current CPU action output"
        )
    if not torch.isfinite(actions).all():
        raise AssertionError("VLA produced non-finite actions")
    action_bytes = _payload_bytes(actions)
    del observation_vla
    vla_offload_ms = _timed_policy_move(
        policy, torch.device("cpu"), shared_device=shared_device
    )

    wm_onload_ms = _timed_environment_move(
        env, resident=True, shared_device=shared_device
    )
    actions_wm, action_handoff_ms = _timed_action_handoff(
        actions, shared_device=shared_device
    )
    observation_wm, wm_ms = _timed_world_model_step(
        env, actions_wm, shared_device=shared_device
    )
    wm_resident_memory_mib = torch.cuda.memory_allocated(shared_device) / (1024**2)
    observation_bytes = _payload_bytes(observation_wm)
    observation_cpu, observation_to_cpu_ms = _timed_observation_copy(
        observation_wm, torch.device("cpu"), shared_device=shared_device
    )
    del observation_wm, actions_wm
    wm_offload_ms = _timed_environment_move(
        env, resident=False, shared_device=shared_device
    )
    post_offload_memory_mib = torch.cuda.memory_allocated(shared_device) / (1024**2)

    observation_handoff_ms = observation_to_vla_ms + observation_to_cpu_ms
    model_swap_ms = vla_onload_ms + vla_offload_ms + wm_onload_ms + wm_offload_ms
    cycle_ms = (
        model_swap_ms + observation_handoff_ms + vla_ms + action_handoff_ms + wm_ms
    )
    row = {
        "vla_onload_ms": vla_onload_ms,
        "observation_to_vla_ms": observation_to_vla_ms,
        "vla_action_generation_ms": vla_ms,
        "vla_offload_ms": vla_offload_ms,
        "wm_onload_ms": wm_onload_ms,
        "action_handoff_ms": action_handoff_ms,
        "wm_observation_generation_ms": wm_ms,
        "observation_to_cpu_ms": observation_to_cpu_ms,
        "wm_offload_ms": wm_offload_ms,
        "observation_handoff_ms": observation_handoff_ms,
        "model_swap_ms": model_swap_ms,
        "cycle_ms": cycle_ms,
        "action_payload_bytes": action_bytes,
        "observation_payload_bytes": observation_bytes,
        "action_source_device": str(actions.device),
        "action_destination_device": str(shared_device),
        "observation_source_device": str(shared_device),
        "observation_destination_device": "cpu_then_shared_gpu",
        "vla_resident_memory_mib": vla_resident_memory_mib,
        "wm_resident_memory_mib": wm_resident_memory_mib,
        "post_offload_memory_mib": post_offload_memory_mib,
        "environment_elapsed_steps": _scalar_int(env.elapsed_steps),
    }
    return observation_cpu, row


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

    x = np.asarray([row["iteration"] for row in rows])

    figure, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(
        x,
        [row["vla_action_generation_ms"] for row in rows],
        label="VLA action generation",
    )
    axes[0].plot(
        x,
        [row["wm_observation_generation_ms"] for row in rows],
        label="Wan observation generation",
    )
    axes[0].set_ylabel("Compute latency (ms)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    for key, label in (
        ("vla_onload_ms", "VLA onload"),
        ("vla_offload_ms", "VLA offload"),
        ("wm_onload_ms", "Wan onload"),
        ("wm_offload_ms", "Wan offload"),
    ):
        axes[1].plot(x, [row[key] for row in rows], label=label)
    axes[1].set_ylabel("Residency transition (ms)")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.25)
    for key, label in (
        ("action_handoff_ms", "Action CPU→GPU"),
        ("observation_to_cpu_ms", "Observation GPU→CPU"),
        ("observation_to_vla_ms", "Observation CPU→GPU"),
    ):
        axes[2].plot(x, [row[key] for row in rows], label=label)
    axes[2].set_xlabel("Measured iteration")
    axes[2].set_ylabel("Handoff latency (ms)")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    figure.suptitle("Collocated OpenVLA-OFT + Wan alternating-residency timing")
    figure.tight_layout()
    figure.savefig(output_dir / "latency_timeseries.png", dpi=180)
    plt.close(figure)

    compute_ms = (
        summaries["vla_action_generation"]["mean_ms"]
        + summaries["wm_observation_generation"]["mean_ms"]
    )
    swap_ms = summaries["model_swap"]["mean_ms"]
    transfer_ms = (
        summaries["action_handoff"]["mean_ms"]
        + summaries["observation_handoff"]["mean_ms"]
    )
    figure, axis = plt.subplots(figsize=(13, 5))
    left = 0.0
    for label, value, color in (
        ("Model compute", compute_ms, "#4C78A8"),
        ("Model onload/offload", swap_ms, "#F58518"),
        ("Payload handoff", transfer_ms, "#54A24B"),
    ):
        axis.barh(["Mean cycle"], [value], left=left, label=label, color=color)
        if value / (compute_ms + swap_ms + transfer_ms) > 0.025:
            axis.text(left + value / 2, 0, f"{value:.1f} ms", ha="center", va="center")
        left += value
    axis.set_xlabel("Mean latency contribution (ms)")
    axis.set_title("Collocated mean cycle breakdown")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    figure.tight_layout()
    figure.savefig(output_dir / "mean_cycle_breakdown.png", dpi=180)
    plt.close(figure)

    residency_keys = ("vla_onload", "vla_offload", "wm_onload", "wm_offload")
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.boxplot(
        [[row[METRIC_COLUMNS[key]] for row in rows] for key in residency_keys],
        tick_labels=["VLA onload", "VLA offload", "Wan onload", "Wan offload"],
    )
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Model residency-transition latency distributions")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "residency_latency_distributions.png", dpi=180)
    plt.close(figure)


def _write_report(
    output_dir: Path,
    *,
    cfg: Any,
    summaries: dict[str, dict[str, float]],
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    vla_ms = summaries["vla_action_generation"]["mean_ms"]
    wm_ms = summaries["wm_observation_generation"]["mean_ms"]
    swap_ms = summaries["model_swap"]["mean_ms"]
    observation_ms = summaries["observation_handoff"]["mean_ms"]
    action_ms = summaries["action_handoff"]["mean_ms"]
    cycle_ms = summaries["cycle"]["mean_ms"]
    compute_ms = vla_ms + wm_ms
    handoff_ms = observation_ms + action_ms
    chunk = int(metadata["action_chunk_length"])
    chunks_per_second = 1000.0 / cycle_ms
    actions_per_second = chunk * chunks_per_second
    derived_rows = (
        ("wan_to_vla_mean_latency_ratio", wm_ms / vla_ms, "ratio"),
        ("model_swap_share_of_cycle", 100.0 * swap_ms / cycle_ms, "percent"),
        ("compute_share_of_cycle", 100.0 * compute_ms / cycle_ms, "percent"),
        ("handoff_share_of_cycle", 100.0 * handoff_ms / cycle_ms, "percent"),
        ("action_chunks_per_second", chunks_per_second, "chunks/s"),
        ("environment_actions_per_second", actions_per_second, "actions/s"),
    )
    with (output_dir / "derived_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value", "unit"))
        writer.writerows(derived_rows)

    report = f"""# Collocated OpenVLA-OFT + Wan timing experiment

## Configuration

- Measured iterations: {int(cfg.benchmark.iterations)}
- Warmup iterations: {int(cfg.benchmark.warmup_iterations)}
- Environments per model call: 1
- Action chunk length: {chunk}
- Wan inference steps: {int(cfg.models.world_model.num_inference_steps)}
- Shared logical device: `{cfg.models.vla.device}`
- `CUDA_VISIBLE_DEVICES`: `{metadata["cuda_visible_devices"]}`
- GPU: {metadata["gpu_name"]}

OpenVLA and Wan alternated residence on one GPU every iteration. Timings use
`perf_counter_ns()` with CUDA synchronization around every boundary. Model
offload includes CUDA allocator cache release. Observation handoff contains
both GPU-to-CPU staging after Wan generation and CPU-to-GPU transfer after the
next OpenVLA onload.

## Principal results

| Metric | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| VLA onload | {summaries["vla_onload"]["mean_ms"]:.3f} | {summaries["vla_onload"]["p50_ms"]:.3f} | {summaries["vla_onload"]["p95_ms"]:.3f} | {summaries["vla_onload"]["p99_ms"]:.3f} |
| VLA action generation | {vla_ms:.3f} | {summaries["vla_action_generation"]["p50_ms"]:.3f} | {summaries["vla_action_generation"]["p95_ms"]:.3f} | {summaries["vla_action_generation"]["p99_ms"]:.3f} |
| VLA offload | {summaries["vla_offload"]["mean_ms"]:.3f} | {summaries["vla_offload"]["p50_ms"]:.3f} | {summaries["vla_offload"]["p95_ms"]:.3f} | {summaries["vla_offload"]["p99_ms"]:.3f} |
| Wan onload | {summaries["wm_onload"]["mean_ms"]:.3f} | {summaries["wm_onload"]["p50_ms"]:.3f} | {summaries["wm_onload"]["p95_ms"]:.3f} | {summaries["wm_onload"]["p99_ms"]:.3f} |
| Wan observation generation | {wm_ms:.3f} | {summaries["wm_observation_generation"]["p50_ms"]:.3f} | {summaries["wm_observation_generation"]["p95_ms"]:.3f} | {summaries["wm_observation_generation"]["p99_ms"]:.3f} |
| Wan offload | {summaries["wm_offload"]["mean_ms"]:.3f} | {summaries["wm_offload"]["p50_ms"]:.3f} | {summaries["wm_offload"]["p95_ms"]:.3f} | {summaries["wm_offload"]["p99_ms"]:.3f} |
| Action handoff | {action_ms:.3f} | {summaries["action_handoff"]["p50_ms"]:.3f} | {summaries["action_handoff"]["p95_ms"]:.3f} | {summaries["action_handoff"]["p99_ms"]:.3f} |
| Observation handoff, both legs | {observation_ms:.3f} | {summaries["observation_handoff"]["p50_ms"]:.3f} | {summaries["observation_handoff"]["p95_ms"]:.3f} | {summaries["observation_handoff"]["p99_ms"]:.3f} |
| All model onload/offload | {swap_ms:.3f} | {summaries["model_swap"]["p50_ms"]:.3f} | {summaries["model_swap"]["p95_ms"]:.3f} | {summaries["model_swap"]["p99_ms"]:.3f} |
| Complete collocated cycle | {cycle_ms:.3f} | {summaries["cycle"]["p50_ms"]:.3f} | {summaries["cycle"]["p95_ms"]:.3f} | {summaries["cycle"]["p99_ms"]:.3f} |

- Model swap share of cycle: **{100.0 * swap_ms / cycle_ms:.2f}%**
- Model compute share of cycle: **{100.0 * compute_ms / cycle_ms:.2f}%**
- Payload handoff share of cycle: **{100.0 * handoff_ms / cycle_ms:.4f}%**
- Sustained rate: **{chunks_per_second:.4f} action chunks/s**
- Equivalent environment-action rate: **{actions_per_second:.4f} actions/s**
- Action payload: **{rows[0]["action_payload_bytes"]} bytes**
- Observation payload: **{rows[0]["observation_payload_bytes"]} bytes**

## Artifacts

- `iterations.csv`: spreadsheet-compatible per-iteration measurements.
- `summary.csv`: latency distribution statistics.
- `derived_metrics.csv`: overhead shares and throughput.
- `metadata.json`: runtime, model, and device metadata.
- `latency_timeseries.png`: compute, residency, and handoff timing series.
- `residency_latency_distributions.png`: onload/offload distributions.
- `mean_cycle_breakdown.png`: compute/swap/handoff breakdown.

## Interpretation limits

- This is a single-process benchmark of the alternating-residency path used by
  the existing collocated acceptance harness.
- It excludes Ray and RLinf Channel scheduling overhead.
- Wan uses one diffusion step rather than its default five.
- Model transfer latency depends strongly on CPU memory bandwidth, PCIe
  topology, allocator state, and whether host pages are pinned.
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
    if iterations < 1 or warmup_iterations < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    if int(cfg.benchmark.num_envs) != 1:
        raise ValueError("this benchmark currently requires one environment")
    if str(cfg.models.vla.device) != str(cfg.models.world_model.device):
        raise ValueError("collocated benchmark requires the same logical device")

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
        raise RuntimeError("collocated benchmark requires exactly one visible CUDA GPU")

    shared_device = torch.device(str(cfg.models.vla.device))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    torch.manual_seed(int(cfg.benchmark.seed))
    np.random.seed(int(cfg.benchmark.seed))
    Worker.torch_device_type = "cuda"
    Worker.torch_platform = torch.cuda
    env_cfg, vla_cfg = _load_configs(repo_path, cfg)

    torch.cuda.set_device(shared_device)
    from rlinf.envs.world_model.world_model_wan_env import WanEnv

    env = WanEnv(
        env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        record_metrics=True,
        worker_info=SimpleNamespace(rank=0, group_world_size=1),
    )
    env.device = shared_device
    observation, _ = env.reset(
        episode_indices=torch.tensor([int(cfg.benchmark.episode_index)])
    )
    observation_cpu, _ = _timed_observation_copy(
        observation, torch.device("cpu"), shared_device=shared_device
    )
    _timed_environment_move(env, resident=False, shared_device=shared_device)

    from rlinf.models.embodiment.openvla_oft import get_model

    policy = get_model(vla_cfg, torch_dtype=torch.bfloat16).to(shared_device).eval()
    _timed_policy_move(policy, torch.device("cpu"), shared_device=shared_device)

    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "shared_logical_device": str(shared_device),
        "gpu_name": torch.cuda.get_device_name(shared_device),
        "vla_checkpoint": str(vla_checkpoint),
        "wm_checkpoint": str(wm_checkpoint),
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "num_envs": 1,
        "action_chunk_length": int(env_cfg.chunk),
        "action_dim": int(vla_cfg.action_dim),
        "wan_inference_steps": int(env_cfg.num_inference_steps),
        "timing_clock": "time.perf_counter_ns",
        "offload_includes_empty_cache": True,
        "ray_channel_overhead_included": False,
    }

    rows: list[dict[str, Any]] = []
    try:
        for warmup_index in range(warmup_iterations):
            observation_cpu, _ = _run_cycle(
                policy=policy,
                env=env,
                observation_cpu=observation_cpu,
                shared_device=shared_device,
            )
            print(f"Warmup {warmup_index + 1}/{warmup_iterations} complete", flush=True)

        with (output_dir / "iterations.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=ITERATION_FIELDS)
            writer.writeheader()
            for iteration in range(iterations):
                observation_cpu, row = _run_cycle(
                    policy=policy,
                    env=env,
                    observation_cpu=observation_cpu,
                    shared_device=shared_device,
                )
                row = {"iteration": iteration, **row}
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                if (iteration + 1) % int(cfg.benchmark.progress_interval) == 0:
                    print(
                        f"Measured {iteration + 1}/{iterations}: "
                        f"VLA load/run/off={row['vla_onload_ms']:.1f}/"
                        f"{row['vla_action_generation_ms']:.1f}/"
                        f"{row['vla_offload_ms']:.1f} ms, "
                        f"Wan load/run/off={row['wm_onload_ms']:.1f}/"
                        f"{row['wm_observation_generation_ms']:.1f}/"
                        f"{row['wm_offload_ms']:.1f} ms",
                        flush=True,
                    )

        summaries = {
            metric: _metric_summary([float(row[column]) for row in rows])
            for metric, column in METRIC_COLUMNS.items()
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
