"""Real-model Task 7 registered-runner lifecycle smoke test."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Callable

import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import ListConfig, OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.scheduler.rlix.entrypoint import (
    launch_registered_rlix_workers,
    preflight_rlix_placements,
)
from rlinf.scheduler.rlix.protocol import RunnerStageState
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"expected a boolean value, got {value!r}")


def _parse_gpu_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple, ListConfig)):
        ids = tuple(int(item) for item in value)
    elif isinstance(value, str):
        ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        ids = (int(value),)
    if not ids:
        raise ValueError("GPU list must not be empty")
    if len(set(ids)) != len(ids):
        raise ValueError(f"GPU list contains duplicates: {ids}")
    return ids


def _compose_runtime_config(
    repo_path: Path,
    smoke_cfg: Any,
    *,
    validator: Callable[[Any], Any] = validate_cfg,
) -> Any:
    config_dir = repo_path / "examples" / "embodiment" / "config"
    os.environ.setdefault("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=str(smoke_cfg.smoke.base_config))

    vla_checkpoint = str(Path(smoke_cfg.models.vla.checkpoint).resolve())
    wm_checkpoint = Path(smoke_cfg.models.world_model.checkpoint).resolve()
    actor_gpus = _parse_gpu_ids(
        smoke_cfg.smoke.get("actor_gpus", smoke_cfg.smoke.get("actor_gpu"))
    )
    rollout_gpus = _parse_gpu_ids(smoke_cfg.smoke.rollout_gpu)
    env_gpus = _parse_gpu_ids(smoke_cfg.smoke.env_gpu)
    if len(rollout_gpus) != len(env_gpus):
        raise ValueError("rollout and environment GPU lists must have equal length")
    run_evaluation = _as_bool(smoke_cfg.smoke.run_evaluation)
    save_checkpoint = _as_bool(smoke_cfg.smoke.save_checkpoint)
    output_dir = Path(smoke_cfg.smoke.output_dir).resolve()
    with open_dict(cfg):
        cfg.cluster.component_placement = OmegaConf.create(
            {
                "actor": ",".join(str(gpu) for gpu in actor_gpus),
                "rollout": ",".join(str(gpu) for gpu in rollout_gpus),
                "env": ",".join(str(gpu) for gpu in env_gpus),
            }
        )
        cfg.runner.logger.log_path = str(output_dir / "logs")
        cfg.runner.logger.experiment_name = "task7_real_runner_smoke"
        cfg.runner.logger.logger_backends = []
        max_train_steps = int(smoke_cfg.smoke.max_train_steps)
        cfg.runner.max_epochs = max_train_steps
        cfg.runner.max_steps = max_train_steps
        cfg.runner.weight_sync_interval = 1
        cfg.runner.val_check_interval = 1 if run_evaluation else -1
        cfg.runner.save_interval = 1 if save_checkpoint else -1
        cfg.runner.resume_dir = None
        cfg.runner.per_worker_log = False
        cfg.algorithm.adv_type = str(smoke_cfg.smoke.get("adv_type", "gae"))
        cfg.algorithm.group_size = int(smoke_cfg.smoke.get("group_size", 1))
        total_num_envs = int(smoke_cfg.smoke.total_num_envs)
        if total_num_envs % len(actor_gpus) != 0:
            raise ValueError(
                "Task 7 smoke total_num_envs must divide evenly across actor ranks"
            )
        cfg.env.train.total_num_envs = total_num_envs
        cfg.env.train.group_size = int(smoke_cfg.smoke.get("group_size", 1))
        cfg.env.train.rollout_epoch = int(smoke_cfg.smoke.rollout_epoch)
        stop_rank_when_all_done = smoke_cfg.smoke.get(
            "stop_rank_when_all_done",
            cfg.env.train.get("stop_rank_when_all_done", False),
        )
        if not isinstance(stop_rank_when_all_done, bool):
            raise ValueError("smoke.stop_rank_when_all_done must be a boolean")
        cfg.env.train.stop_rank_when_all_done = stop_rank_when_all_done
        cfg.env.train.max_episode_steps = int(smoke_cfg.smoke.max_episode_steps)
        cfg.env.train.max_steps_per_rollout_epoch = int(
            smoke_cfg.smoke.max_steps_per_rollout_epoch
        )
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
        cfg.env.eval.total_num_envs = int(smoke_cfg.smoke.total_num_envs)
        cfg.env.eval.group_size = int(smoke_cfg.smoke.get("group_size", 1))
        cfg.env.eval.rollout_epoch = 1
        cfg.env.eval.max_episode_steps = int(smoke_cfg.smoke.max_episode_steps)
        cfg.env.eval.max_steps_per_rollout_epoch = int(
            smoke_cfg.smoke.max_steps_per_rollout_epoch
        )
        cfg.env.eval.video_cfg.save_video = False
        cfg.actor.micro_batch_size = 1
        cfg.actor.global_batch_size = total_num_envs
        cfg.actor.model.model_path = vla_checkpoint
        cfg.actor.model.precision = str(smoke_cfg.models.vla.precision)
        cfg.actor.model.unnorm_key = str(smoke_cfg.models.vla.unnorm_key)
        cfg.actor.model.attn_implementation = "eager"
        cfg.rollout.model.model_path = vla_checkpoint
        cfg.rollout.model.precision = str(smoke_cfg.models.vla.precision)
        cfg.rlix.operation_timeout_s = float(smoke_cfg.smoke.operation_timeout_s)
    validate_rlix_entrypoint(cfg, entrypoint="train_embodied_agent")
    return validator(cfg)


def _assert_fixed_residencies_safe(
    statuses: list[Any], *, components: set[str]
) -> None:
    seen: set[str] = set()
    for status in statuses:
        seen.add(status.component)
        if (
            status.model_resident
            or status.optimizer_resident
            or status.cuda_graph_captured
        ):
            raise AssertionError(
                f"{status.component} rank {status.rank} is not offloaded"
            )
    missing = components - seen
    if missing:
        raise AssertionError(f"missing fixed residency reports for {sorted(missing)}")


def _assert_elastic_completed_offloaded(statuses: list[Any], *, component: str) -> None:
    if not statuses:
        raise AssertionError(f"{component} returned no elastic status records")
    for rank, status in enumerate(statuses):
        if status.worker_rank != rank:
            raise AssertionError(
                f"{component} status rank mismatch: expected {rank}, got {status.worker_rank}"
            )
        if status.state is not ElasticRankState.COMPLETED:
            raise AssertionError(
                f"{component} rank {rank} must complete collection, got {status.state}"
            )
        if status.model_resident:
            raise AssertionError(f"{component} rank {rank} remained GPU-resident")
        if status.failure is not None:
            raise AssertionError(
                f"{component} rank {rank} reported failure: {status.failure}"
            )


def _wait_for_registered_workers_ready(launched: Any) -> None:
    for group in (launched.actor, launched.rollout, launched.env):
        group._is_ready().wait()


def _close_launched(launched: Any, primary_error: BaseException | None) -> None:
    cleanup_errors: list[Exception] = []
    if launched is not None:
        runtime = launched.runtime
        if runtime.stage_state not in {
            RunnerStageState.CLOSED,
            RunnerStageState.FAILED_UNCERTAIN,
        }:
            try:
                runtime.close_sync()
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
                f"Task 7 smoke cleanup also failed: {type(error).__name__}: {error}"
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
    actor_gpus = _parse_gpu_ids(
        smoke_cfg.smoke.get("actor_gpus", smoke_cfg.smoke.get("actor_gpu"))
    )
    rollout_gpus = _parse_gpu_ids(smoke_cfg.smoke.rollout_gpu)
    env_gpus = _parse_gpu_ids(smoke_cfg.smoke.env_gpu)
    if len(rollout_gpus) != 1 or len(env_gpus) != 1:
        raise ValueError("Task 7 smoke expects one rollout GPU and one environment GPU")
    required_gpu_ids = {*actor_gpus, *rollout_gpus, *env_gpus}
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(
        required_gpu_ids
    ):
        raise RuntimeError(
            "Task 7 real-runner smoke requires all configured CUDA GPU IDs; "
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
        _wait_for_registered_workers_ready(launched)
        runner = EmbodiedRunner(
            cfg=cfg,
            actor=launched.actor,
            rollout=launched.rollout,
            env=launched.env,
            reward=None,
            rlix_runtime=launched.runtime,
        )
        runner.init_workers()
        if launched.runtime.stage_state is not RunnerStageState.INACTIVE:
            raise AssertionError(
                f"runtime not inactive after initialization: {launched.runtime.stage_state}"
            )

        runner.run()
        if runner.global_step != int(smoke_cfg.smoke.max_train_steps):
            raise AssertionError(
                f"runner global_step {runner.global_step} did not reach expected "
                f"{smoke_cfg.smoke.max_train_steps}"
            )
        if launched.runtime.stage_state is not RunnerStageState.INACTIVE:
            raise AssertionError(
                f"runtime not inactive after runner completion: {launched.runtime.stage_state}"
            )

        collection_session = runner._last_rlix_collection_session
        if collection_session.batch_receipt is None:
            raise AssertionError("runner did not seal an elastic batch")
        if collection_session.active_dp_ranks:
            raise AssertionError(
                f"collection kept active ranks: {sorted(collection_session.active_dp_ranks)}"
            )
        assigned_ranks = {rank for rank, _ in collection_session.assignments}
        if collection_session.released_dp_ranks != assigned_ranks:
            raise AssertionError(
                "collection did not release every completed rank: "
                f"{sorted(collection_session.released_dp_ranks)} vs {sorted(assigned_ranks)}"
            )

        actor_residencies = launched.actor.get_rlix_fixed_residency().wait()
        rollout_residencies = launched.rollout.get_rlix_fixed_residency().wait()
        env_residencies = launched.env.get_rlix_fixed_residency().wait()
        _assert_fixed_residencies_safe(actor_residencies, components={"actor"})
        _assert_fixed_residencies_safe(rollout_residencies, components={"rollout"})
        _assert_fixed_residencies_safe(env_residencies, components={"environment"})

        rollout_statuses = launched.rollout.get_elastic_status().wait()
        env_statuses = launched.env.get_elastic_status().wait()
        _assert_elastic_completed_offloaded(rollout_statuses, component="rollout")
        _assert_elastic_completed_offloaded(env_statuses, component="environment")

        runtime_state_before_close = launched.runtime.stage_state.value
        launched.runtime.close_sync()
        runtime_state_after_close = launched.runtime.stage_state.value
        if runtime_state_after_close != RunnerStageState.CLOSED.value:
            raise AssertionError(f"runtime did not close: {runtime_state_after_close}")

        result = {
            "status": "passed",
            "pipeline_id": launched.runtime.pipeline_id,
            "ray_namespace": launched.runtime.ray_namespace,
            "elapsed_seconds": time.monotonic() - started,
            "hostname": platform.node(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "actor_gpus": list(actor_gpus),
            "rollout_gpu": rollout_gpus[0],
            "env_gpu": env_gpus[0],
            "global_step": runner.global_step,
            "runtime_state_before_close": runtime_state_before_close,
            "runtime_state_after_close": runtime_state_after_close,
            "batch_receipt": {
                "lifecycle_generation": collection_session.batch_receipt.lifecycle_generation,
                "policy_version": collection_session.batch_receipt.policy_version,
                "contributing_dp_ranks": list(
                    collection_session.batch_receipt.contributing_dp_ranks
                ),
                "expected_trajectories": collection_session.batch_receipt.expected_trajectories,
                "received_trajectories": collection_session.batch_receipt.received_trajectories,
                "transition_count": collection_session.batch_receipt.transition_count,
            },
            "released_dp_ranks": sorted(collection_session.released_dp_ranks),
            "rollout_states": [status.state.value for status in rollout_statuses],
            "environment_states": [status.state.value for status in env_statuses],
            "cluster_device_mappings": resolved.plan.registration_payload()[
                "cluster_device_mappings"
            ],
            "actor_infer_bundles": resolved.plan.registration_payload()[
                "cluster_dp_device_mappings"
            ]["actor_infer"],
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        _close_launched(launched, primary_error)
        if ray.is_initialized():
            ray.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
