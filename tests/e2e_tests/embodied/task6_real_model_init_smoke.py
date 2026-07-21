"""Real-model Task 6 placement, registration, and initialization smoke test."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.scheduler.rlix.entrypoint import (
    launch_registered_rlix_workers,
    preflight_rlix_placements,
)
from rlinf.scheduler.rlix.runtime import bootstrap_registered_rlix_pipeline
from rlinf.scheduler.rlix.validation import validate_rlix_entrypoint
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.elastic_rollout_lifecycle import ElasticRankState
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _require_checkpoint(path: Path, required_entries: tuple[str, ...]) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {path}")
    missing = [entry for entry in required_entries if not (path / entry).exists()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint {path} is missing required entries: {', '.join(missing)}"
        )


def _compose_runtime_config(repo_path: Path, smoke_cfg: Any) -> Any:
    config_dir = repo_path / "examples" / "embodiment" / "config"
    os.environ.setdefault("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=str(smoke_cfg.smoke.base_config))

    vla_checkpoint = str(Path(smoke_cfg.models.vla.checkpoint).resolve())
    wm_checkpoint = Path(smoke_cfg.models.world_model.checkpoint).resolve()
    actor_gpu = int(smoke_cfg.smoke.actor_gpu)
    rollout_gpu = int(smoke_cfg.smoke.rollout_gpu)
    env_gpu = int(smoke_cfg.smoke.env_gpu)
    with open_dict(cfg):
        cfg.cluster.component_placement = OmegaConf.create(
            {
                "actor": str(actor_gpu),
                "rollout": str(rollout_gpu),
                "env": str(env_gpu),
            }
        )
        cfg.runner.logger.log_path = str(
            Path(smoke_cfg.smoke.output_dir).resolve() / "logs"
        )
        cfg.runner.logger.experiment_name = "task6_real_model_init_smoke"
        cfg.runner.val_check_interval = -1
        cfg.runner.save_interval = -1
        cfg.runner.resume_dir = None
        cfg.algorithm.adv_type = "gae"
        cfg.algorithm.group_size = 1
        cfg.env.train.total_num_envs = 1
        cfg.env.train.group_size = 1
        cfg.env.train.rollout_epoch = 1
        cfg.env.train.video_cfg.save_video = False
        cfg.env.train.wan_wm_hf_ckpt_path = str(wm_checkpoint)
        cfg.env.train.VAE_path = str(wm_checkpoint / "Wan2.2_VAE.pth")
        cfg.env.train.model_path = str(wm_checkpoint / "model-00001.safetensors")
        cfg.env.train.initial_image_path = str(wm_checkpoint / "dataset")
        cfg.env.train.reward_model.from_pretrained = str(
            wm_checkpoint / "resnet_rm.pth"
        )
        cfg.env.train.num_inference_steps = int(
            smoke_cfg.models.world_model.num_inference_steps
        )
        cfg.actor.micro_batch_size = 1
        cfg.actor.global_batch_size = 1
        cfg.actor.model.model_path = vla_checkpoint
        cfg.actor.model.precision = str(smoke_cfg.models.vla.precision)
        cfg.actor.model.unnorm_key = str(smoke_cfg.models.vla.unnorm_key)
        cfg.actor.model.attn_implementation = "eager"
        cfg.rollout.model.model_path = vla_checkpoint
        cfg.rollout.model.precision = str(smoke_cfg.models.vla.precision)
        cfg.rlix.operation_timeout_s = float(smoke_cfg.smoke.operation_timeout_s)
    validate_rlix_entrypoint(cfg, entrypoint="train_embodied_agent")
    return validate_cfg(cfg)


def _assert_cold_offloaded(statuses: list[Any], *, component: str) -> None:
    if not statuses:
        raise AssertionError(f"{component} returned no elastic status records")
    for rank, status in enumerate(statuses):
        if status.worker_rank != rank:
            raise AssertionError(
                f"{component} status rank mismatch: expected {rank}, got {status.worker_rank}"
            )
        if status.state is not ElasticRankState.INACTIVE_COLD:
            raise AssertionError(
                f"{component} rank {rank} must remain inactive after init, got {status.state}"
            )
        if status.model_resident:
            raise AssertionError(f"{component} rank {rank} remained GPU-resident")
        if status.failure is not None:
            raise AssertionError(
                f"{component} rank {rank} reported failure: {status.failure}"
            )


def _cleanup(launched: Any, primary_error: BaseException | None) -> None:
    cleanup_errors: list[Exception] = []
    if launched is not None:
        try:
            asyncio.run(launched.runtime.close())
        except Exception as exc:
            cleanup_errors.append(exc)
        for group in (launched.env, launched.rollout, launched.actor):
            try:
                group._close()
            except Exception as exc:
                cleanup_errors.append(exc)
    if primary_error is not None:
        for error in cleanup_errors:
            primary_error.add_note(
                f"Task 6 smoke cleanup also failed: {type(error).__name__}: {error}"
            )
    elif cleanup_errors:
        raise cleanup_errors[0]


def run(config_path: Path) -> None:
    repo_path = Path(__file__).resolve().parents[3]
    smoke_cfg = OmegaConf.load(config_path)
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
        int(smoke_cfg.smoke.actor_gpu),
        int(smoke_cfg.smoke.rollout_gpu),
        int(smoke_cfg.smoke.env_gpu),
    }
    if len(required_gpu_ids) != 3:
        raise ValueError("Task 6 real-model smoke requires three distinct GPU IDs")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(
        required_gpu_ids
    ):
        raise RuntimeError(
            "Task 6 real-model smoke requires all configured CUDA GPU IDs; "
            f"configured={sorted(required_gpu_ids)}, visible_count={torch.cuda.device_count()}"
        )

    cfg = _compose_runtime_config(repo_path, smoke_cfg)
    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)
    resolved = preflight_rlix_placements(component_placement, cluster)

    output_dir = Path(smoke_cfg.smoke.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    launched = None
    primary_error: BaseException | None = None
    started = time.monotonic()
    try:
        launched = launch_registered_rlix_workers(
            cluster=cluster,
            actor_group=EmbodiedFSDPActor.create_group(cfg),
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
        coordinator_status = asyncio.run(launched.runtime.controller.get_status())
        if coordinator_status.collection is not None:
            raise AssertionError("Task 6 smoke must not configure a collection")
        if coordinator_status.callback_applied_active_ranks:
            raise AssertionError("Task 6 smoke must not activate any DP ranks")

        # T6 owns inactive registration, not actor training initialization. Load
        # only the real inference-side models so this smoke does not construct
        # full gradients and AdamW state on the single actor placement.
        rollout_init = launched.rollout.init_worker()
        env_init = launched.env.init_worker()
        rollout_init.wait()
        env_init.wait()

        rollout_statuses = launched.rollout.get_elastic_status().wait()
        env_statuses = launched.env.get_elastic_status().wait()
        _assert_cold_offloaded(rollout_statuses, component="rollout")
        _assert_cold_offloaded(env_statuses, component="environment")

        result = {
            "status": "passed",
            "pipeline_id": launched.runtime.pipeline_id,
            "ray_namespace": launched.runtime.ray_namespace,
            "elapsed_seconds": time.monotonic() - started,
            "hostname": platform.node(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "actor_initialization": "skipped_task6_bootstrap_only",
            "initialized_model_components": ["rollout", "environment"],
            "gpu_names": [
                torch.cuda.get_device_name(index) for index in sorted(required_gpu_ids)
            ],
            "cluster_device_mappings": resolved.plan.registration_payload()[
                "cluster_device_mappings"
            ],
            "actor_infer_bundles": resolved.plan.registration_payload()[
                "cluster_dp_device_mappings"
            ]["actor_infer"],
            "rollout_states": [status.state.value for status in rollout_statuses],
            "environment_states": [status.state.value for status in env_statuses],
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _cleanup(launched, primary_error)
        if ray.is_initialized():
            ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
