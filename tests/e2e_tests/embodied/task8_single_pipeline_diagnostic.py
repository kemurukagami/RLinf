"""Run one production Wan pipeline without cross-pipeline preemption.

This is a diagnostic control experiment, not Task 8 acceptance.  It preserves
the four-GPU Task 8 model, placement, horizon, GRPO, and reward-filter settings,
while registering only one pipeline.  The sealed CPU batch and review videos
are persisted before advantage calculation so a later training failure cannot
destroy the generated evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from task7_real_runner_smoke import (
    _close_launched,
    _parse_gpu_ids,
    _require_checkpoint,
    _wait_for_registered_workers_ready,
)
from task8_acceptance_support import Task8RoleNames, validate_run_id
from task8_two_pipeline_driver import (
    compose_wan_model_driver_config,
    parse_canonical_bundles,
)

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.scheduler.rlix.entrypoint import (
    launch_registered_rlix_workers,
    preflight_rlix_placements,
)
from rlinf.scheduler.rlix.protocol import RunnerStageState
from rlinf.scheduler.rlix.runtime import bootstrap_registered_rlix_pipeline
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def derive_single_pipeline_names(*, run_id: str) -> Task8RoleNames:
    """Derive collision-free names for the standalone control pipeline."""
    validate_run_id(run_id)
    prefix = f"t8_{run_id}_single"
    names = Task8RoleNames(
        prefix=prefix,
        actor_group=f"{prefix}_ActorGroup",
        rollout_group=f"{prefix}_RolloutGroup",
        env_group=f"{prefix}_EnvGroup",
        env_input_channel=f"{prefix}_env_input",
        rollout_request_channel=f"{prefix}_rollout_request",
        actor_channel=f"{prefix}_actor_batch",
        event_producer=f"{prefix}_event_producer",
    )
    names.validate()
    return names


def configure_single_pipeline_artifacts(
    cfg: DictConfig, *, run_dir: str | Path
) -> Path:
    """Enable run-local video and sealed-batch persistence on a composed config."""
    root = Path(run_dir).resolve()
    trajectory_dir = root / "trajectories"
    with open_dict(cfg):
        cfg.env.train.video_cfg.save_video = True
        cfg.env.train.video_cfg.video_base_dir = str(trajectory_dir / "videos")
        cfg.env.train.video_cfg.info_on_video = True
        cfg.env.train.video_cfg.extra_info_on_video = [
            "episode.success_once",
            "episode.return",
            "episode.episode_len",
        ]
        cfg.runner.task8_single_pipeline_artifact_dir = str(trajectory_dir)
    return trajectory_dir


def summarize_sealed_batch(
    batch: Mapping[str, Any],
    *,
    group_size: int,
    filter_rewards: bool,
    rewards_lower_bound: float,
    rewards_upper_bound: float,
) -> dict[str, Any]:
    """Build a JSON-safe reward/filter summary from the exact sealed actor batch."""
    rewards = batch.get("rewards")
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 3:
        raise ValueError("sealed batch rewards must be a rank-3 tensor")
    if rewards.shape[1] % group_size != 0:
        raise ValueError("sealed reward batch must divide evenly into GRPO groups")

    reward_values = rewards.detach().to(dtype=torch.float32, device="cpu")
    trajectory_sums = (
        reward_values.transpose(0, 1).reshape(rewards.shape[1], -1).sum(dim=1)
    )
    group_sums = trajectory_sums.reshape(-1, group_size)
    group_means = group_sums.mean(dim=1)
    accepted_groups = (group_means >= rewards_lower_bound) & (
        group_means <= rewards_upper_bound
    )

    loss_mask = batch.get("loss_mask")
    active_trajectories: list[bool] | None = None
    if isinstance(loss_mask, torch.Tensor):
        mask = loss_mask.detach().to(dtype=torch.bool, device="cpu")
        active_trajectories = (
            mask.transpose(0, 1).reshape(mask.shape[1], -1).any(dim=1).tolist()
        )

    tensor_counts = {}
    for key in ("terminations", "truncations", "dones"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            tensor_counts[key] = int(value.detach().to(dtype=torch.bool).sum().item())

    versions = batch.get("versions")
    unique_versions = (
        [int(value) for value in torch.unique(versions.detach().cpu()).tolist()]
        if isinstance(versions, torch.Tensor)
        else []
    )
    nonzero_rewards = int(torch.count_nonzero(reward_values).item())
    accepted_group_values = [bool(value) for value in accepted_groups.tolist()]
    return {
        "reward_shape": list(rewards.shape),
        "reward_min": float(reward_values.min().item()),
        "reward_max": float(reward_values.max().item()),
        "reward_nonzero_count": nonzero_rewards,
        "reward_all_zero": nonzero_rewards == 0,
        "trajectory_reward_sums": [float(value) for value in trajectory_sums.tolist()],
        "group_size": group_size,
        "group_reward_sums": [
            [float(value) for value in row] for row in group_sums.tolist()
        ],
        "group_mean_rewards": [float(value) for value in group_means.tolist()],
        "reward_filter": {
            "enabled": filter_rewards,
            "lower_bound": float(rewards_lower_bound),
            "upper_bound": float(rewards_upper_bound),
            "accepted_groups": accepted_group_values,
            "accepted_group_count": sum(accepted_group_values),
            "all_groups_filtered": filter_rewards and not any(accepted_group_values),
        },
        "post_filter_active_trajectories": active_trajectories,
        "post_filter_active_trajectory_count": (
            sum(active_trajectories) if active_trajectories is not None else None
        ),
        "terminal_counts": tensor_counts,
        "policy_versions": unique_versions,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class DiagnosticEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Production actor that persists each sealed CPU batch before training."""

    def seal_rlix_batch(self, **kwargs: Any) -> Any:
        receipt = super().seal_rlix_batch(**kwargs)
        iteration_dir = Path(self.cfg.runner.task8_single_pipeline_artifact_dir) / (
            f"iteration_{receipt.policy_version:04d}"
        )
        if iteration_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite trajectory artifacts: {iteration_dir}"
            )
        iteration_dir.mkdir(parents=True)
        _atomic_torch_save(iteration_dir / "sealed_batch.pt", self.rollout_batch)
        summary = summarize_sealed_batch(
            self.rollout_batch,
            group_size=int(self.cfg.algorithm.group_size),
            filter_rewards=bool(self.cfg.algorithm.get("filter_rewards", False)),
            rewards_lower_bound=float(self.cfg.algorithm.rewards_lower_bound),
            rewards_upper_bound=float(self.cfg.algorithm.rewards_upper_bound),
        )
        summary["batch_receipt"] = {
            "lifecycle_generation": receipt.lifecycle_generation,
            "policy_version": receipt.policy_version,
            "contributing_dp_ranks": list(receipt.contributing_dp_ranks),
            "expected_trajectories": receipt.expected_trajectories,
            "received_trajectories": receipt.received_trajectories,
            "transition_count": receipt.transition_count,
        }
        _atomic_write_json(iteration_dir / "reward_summary.json", summary)
        return receipt


def _prepare_run_dir(output_dir: Path, run_id: str) -> Path:
    validate_run_id(run_id)
    root = output_dir.resolve()
    run_dir = root / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _failure_payload(exc: BaseException, *, started: float) -> dict[str, Any]:
    return {
        "status": "failed",
        "diagnostic_only": True,
        "cross_pipeline_preemption": False,
        "elapsed_seconds": time.monotonic() - started,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def run(args: argparse.Namespace) -> None:
    """Launch and run the single-pipeline control experiment."""
    started = time.monotonic()
    run_dir = _prepare_run_dir(args.output_dir, args.run_id)
    names = derive_single_pipeline_names(run_id=args.run_id)
    expected_bundles = parse_canonical_bundles(args.bundles, mode="disaggregated")
    smoke_cfg = OmegaConf.load(args.config)
    vla_checkpoint = Path(smoke_cfg.models.vla.checkpoint).resolve()
    wm_checkpoint = Path(smoke_cfg.models.world_model.checkpoint).resolve()
    _require_checkpoint(
        vla_checkpoint,
        ("config.json", "dataset_statistics.json", "model.safetensors.index.json"),
    )
    _require_checkpoint(
        wm_checkpoint,
        ("model-00001.safetensors", "Wan2.2_VAE.pth", "resnet_rm.pth", "dataset"),
    )
    required_gpu_ids = {
        *_parse_gpu_ids(
            smoke_cfg.smoke.get("actor_gpus", smoke_cfg.smoke.get("actor_gpu"))
        ),
        *_parse_gpu_ids(smoke_cfg.smoke.rollout_gpu),
        *_parse_gpu_ids(smoke_cfg.smoke.env_gpu),
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(
        required_gpu_ids
    ):
        raise RuntimeError(
            "single-pipeline diagnostic requires all configured CUDA GPU IDs; "
            f"configured={sorted(required_gpu_ids)}, visible_count={torch.cuda.device_count()}"
        )

    cfg, channel_names = compose_wan_model_driver_config(
        args.config,
        names=names,
        driver_dir=run_dir,
    )
    trajectory_dir = configure_single_pipeline_artifacts(cfg, run_dir=run_dir)
    (run_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )

    if not ray.is_initialized():
        ray.init(address=args.address, ignore_reinit_error=True)
    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    resolved = preflight_rlix_placements(component_placement, cluster)
    actual_bundles = tuple(bundle for _, bundle in resolved.plan.actor_infer_bundles)
    if actual_bundles != expected_bundles:
        raise ValueError(
            "configured actor-infer bundles do not match --bundles: "
            f"configured={actual_bundles}, expected={expected_bundles}"
        )

    launched = None
    runner = None
    runner_finished = False
    primary_error: BaseException | None = None
    try:
        launched = launch_registered_rlix_workers(
            cluster=cluster,
            actor_group=DiagnosticEmbodiedFSDPActor.create_group(cfg),
            rollout_group=MultiStepRolloutWorker.create_group(cfg),
            env_group=EnvWorker.create_group(cfg),
            actor_name=cfg.actor.group_name,
            rollout_name=cfg.rollout.group_name,
            env_name=cfg.env.group_name,
            resolved=resolved,
            worker_max_concurrency=cfg.rlix.worker_max_concurrency,
            operation_timeout_s=cfg.rlix.operation_timeout_s,
            enable_gpu_tracing=cfg.rlix.enable_gpu_tracing,
            bootstrapper=bootstrap_registered_rlix_pipeline,
        )
        _wait_for_registered_workers_ready(launched)
        runner = EmbodiedRunner(
            cfg=cfg,
            actor=launched.actor,
            rollout=launched.rollout,
            env=launched.env,
            reward=None,
            rlix_runtime=launched.runtime,
            channel_names=channel_names,
        )
        runner.init_workers()
        if launched.runtime.stage_state is not RunnerStageState.INACTIVE:
            raise AssertionError(
                "runtime not inactive after initialization: "
                f"{launched.runtime.stage_state.value}"
            )
        runner.run()
        runner_finished = True
        pipeline_id = launched.runtime.pipeline_id
        launched.runtime.close_sync()
        _close_launched(launched, None)
        launched = None
        result = {
            "status": "passed",
            "diagnostic_only": True,
            "cross_pipeline_preemption": False,
            "pipeline_id": pipeline_id,
            "global_step": runner.global_step,
            "configured_iterations": int(smoke_cfg.smoke.max_train_steps),
            "elapsed_seconds": time.monotonic() - started,
            "hostname": platform.node(),
            "actor_gpus": sorted(_parse_gpu_ids(smoke_cfg.smoke.actor_gpus)),
            "actor_infer_bundles": [list(bundle) for bundle in actual_bundles],
            "trajectory_dir": str(trajectory_dir),
        }
        _atomic_write_json(run_dir / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException as exc:
        primary_error = exc
        _atomic_write_json(
            run_dir / "result.json", _failure_payload(exc, started=started)
        )
        (run_dir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if runner is not None and not runner_finished:
            try:
                runner._finish_run()
            except Exception as cleanup_error:
                if primary_error is not None:
                    primary_error.add_note(
                        "runner logging cleanup also failed: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
        _close_launched(launched, primary_error)
        if ray.is_initialized():
            ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:6379")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundles", default="0,2;1,3")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
