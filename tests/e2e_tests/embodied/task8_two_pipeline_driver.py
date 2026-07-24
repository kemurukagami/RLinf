"""Task 8 subprocess driver foundation for one pipeline role."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from rlix_core.client import connect
from rlix_core.protocol.types import GENERATION_CLUSTER_NAME, get_pipeline_namespace
from task7_real_runner_smoke import (
    _close_launched,
    _compose_runtime_config,
    _parse_gpu_ids,
    _require_checkpoint,
    _wait_for_registered_workers_ready,
)
from task8_acceptance_artifacts import atomic_write_json
from task8_acceptance_support import Task8RoleNames, derive_role_names

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


def configure_role_owned_surface(
    cfg: DictConfig,
    *,
    names: Task8RoleNames,
    driver_dir: str | Path,
) -> dict[str, str]:
    """Apply collision-free worker, output, and runner-channel identities."""
    names.validate()
    output = Path(driver_dir).resolve()
    with open_dict(cfg):
        cfg.actor.group_name = names.actor_group
        cfg.rollout.group_name = names.rollout_group
        cfg.env.group_name = names.env_group
        cfg.runner.logger.log_path = str(output / "logs")
        cfg.runner.logger.experiment_name = names.prefix
        if cfg.runner.get("per_worker_log_path") is not None:
            cfg.runner.per_worker_log_path = str(output / "worker_logs")
    return names.runner_channel_names()


def compose_wan_model_driver_config(
    config_path: str | Path,
    *,
    names: Task8RoleNames,
    driver_dir: str | Path,
    validator: Callable[[Any], Any] | None = None,
) -> tuple[DictConfig, dict[str, str]]:
    """Compose a validated multi-rank Wan config and apply one role's identities."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Task 8 driver config does not exist: {path}")
    smoke_cfg = OmegaConf.load(path)
    repo_path = Path(__file__).resolve().parents[3]
    compose_kwargs = {} if validator is None else {"validator": validator}
    cfg = _compose_runtime_config(repo_path, smoke_cfg, **compose_kwargs)
    channels = configure_role_owned_surface(
        cfg,
        names=names,
        driver_dir=driver_dir,
    )
    return cfg, channels


def parse_canonical_bundles(raw: str, *, mode: str) -> tuple[tuple[int, ...], ...]:
    """Parse and validate a complete rank-ordered physical GPU bundle mapping."""
    expected_width = 2 if mode == "disaggregated" else 1
    bundles: list[tuple[int, ...]] = []
    try:
        for encoded_bundle in raw.split(";"):
            bundle = tuple(int(gpu_id) for gpu_id in encoded_bundle.split(","))
            bundles.append(bundle)
    except ValueError as exc:
        raise ValueError("canonical bundles must contain integer GPU IDs") from exc
    if len(bundles) < 2 or any(len(bundle) != expected_width for bundle in bundles):
        raise ValueError(
            f"{mode} connectivity requires at least two width-{expected_width} bundles"
        )
    flattened = [gpu_id for bundle in bundles for gpu_id in bundle]
    if any(gpu_id < 0 for gpu_id in flattened) or len(flattened) != len(set(flattened)):
        raise ValueError("canonical bundles require disjoint non-negative GPU IDs")
    return tuple(bundles)


def run_connectivity_driver(args: argparse.Namespace) -> None:
    """Register one candidate topology and prove shared-core process isolation."""
    role = os.environ.get("RLINF_TASK8_ROLE")
    ready_path = _required_environment_path("RLINF_TASK8_READY")
    start_path = _required_environment_path("RLINF_TASK8_START")
    result_path = _required_environment_path("RLINF_TASK8_RESULT")
    if role not in {"a", "b"}:
        raise ValueError("RLINF_TASK8_ROLE must be 'a' or 'b'")
    bundles = parse_canonical_bundles(args.bundles, mode=args.mode)
    names = derive_role_names(run_id=args.run_id, role=role)
    candidate_mapping = {
        GENERATION_CLUSTER_NAME: [gpu_id for bundle in bundles for gpu_id in bundle]
    }
    candidate_dp_mapping = {
        GENERATION_CLUSTER_NAME: {
            rank: list(bundle) for rank, bundle in enumerate(bundles)
        }
    }

    control_plane = connect(address=args.address, create_if_missing=False)
    pipeline_id: str | None = None
    primary_error: BaseException | None = None
    try:
        pipeline_id = ray.get(
            control_plane.allocate_pipeline_id.remote(pipeline_type="rlinf")
        )
        namespace = get_pipeline_namespace(pipeline_id)
        ray.get(
            control_plane.register_pipeline.remote(
                pipeline_id=pipeline_id,
                ray_namespace=namespace,
                cluster_tp_configs={GENERATION_CLUSTER_NAME: 1},
                cluster_device_mappings=candidate_mapping,
                cluster_dp_device_mappings=candidate_dp_mapping,
            )
        )
        admission = ray.get(
            control_plane.admit_pipeline.remote(pipeline_id=pipeline_id)
        )
        atomic_write_json(
            ready_path,
            {
                "scope": "connectivity_only",
                "role": role,
                "pid": os.getpid(),
                "control_plane_actor_id": _actor_id(control_plane),
                "scheduler_actor_id": _actor_id(admission.scheduler),
                "pipeline_id": pipeline_id,
                "pipeline_namespace": namespace,
                "candidate_mapping": candidate_mapping,
                "candidate_dp_mapping": candidate_dp_mapping,
                "role_names": {
                    field: getattr(names, field)
                    for field in names.__dataclass_fields__
                    if field != "prefix"
                },
            },
        )
        _wait_for(start_path, args.timeout_s)
        current_scheduler = ray.get(control_plane.get_scheduler.remote())
        if _actor_id(current_scheduler) != _actor_id(admission.scheduler):
            raise RuntimeError("scheduler identity changed after driver rendezvous")
        ray.get(control_plane.unregister_pipeline.remote(pipeline_id=pipeline_id))
        pipeline_id = None
        atomic_write_json(
            result_path,
            {
                "status": "passed",
                "role": role,
                "scope": "connectivity_only",
                "task8_accepted": False,
                "message": "Shared-core identity passed; no model work was executed.",
            },
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if pipeline_id is not None:
            try:
                ray.get(
                    control_plane.unregister_pipeline.remote(pipeline_id=pipeline_id)
                )
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "pipeline unregister also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        ray.shutdown()


def run_model_init_driver(args: argparse.Namespace) -> None:
    """Cold-initialize one real Wan pipeline and prove safe offload."""
    role = os.environ.get("RLINF_TASK8_ROLE")
    ready_path = _required_environment_path("RLINF_TASK8_READY")
    start_path = _required_environment_path("RLINF_TASK8_START")
    result_path = _required_environment_path("RLINF_TASK8_RESULT")
    if role not in {"a", "b"}:
        raise ValueError("RLINF_TASK8_ROLE must be 'a' or 'b'")
    if args.config is None:
        raise ValueError("--config is required with --model-init-only")

    expected_bundles = parse_canonical_bundles(args.bundles, mode=args.mode)
    names = derive_role_names(run_id=args.run_id, role=role)
    driver_dir = ready_path.parent
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
            "Task 8 model initialization requires all configured CUDA GPU IDs; "
            f"configured={sorted(required_gpu_ids)}, "
            f"visible_count={torch.cuda.device_count()}"
        )

    cfg, channel_names = compose_wan_model_driver_config(
        args.config,
        names=names,
        driver_dir=driver_dir,
    )
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
            channel_names=channel_names,
        )
        runner.init_workers()
        if launched.runtime.stage_state is not RunnerStageState.INACTIVE:
            raise AssertionError(
                "runtime not inactive after model initialization: "
                f"{launched.runtime.stage_state.value}"
            )
        residencies = _collect_fixed_residencies(launched)
        scheduler = ray.get(launched.runtime.control_plane.get_scheduler.remote())
        registration = resolved.plan.registration_payload()
        atomic_write_json(
            ready_path,
            {
                "scope": "model_init_only",
                "role": role,
                "pid": os.getpid(),
                "control_plane_actor_id": _actor_id(launched.runtime.control_plane),
                "scheduler_actor_id": _actor_id(scheduler),
                "pipeline_id": launched.runtime.pipeline_id,
                "pipeline_namespace": launched.runtime.ray_namespace,
                "candidate_mapping": registration["cluster_device_mappings"],
                "candidate_dp_mapping": registration["cluster_dp_device_mappings"],
                "role_names": {
                    field: getattr(names, field)
                    for field in names.__dataclass_fields__
                    if field != "prefix"
                },
                "runtime_state": launched.runtime.stage_state.value,
                "actor_infer_bundles": [list(bundle) for bundle in actual_bundles],
                "residencies": residencies,
            },
        )
        _wait_for(start_path, args.timeout_s)

        runner._finish_run()  # Test-only initialization does not enter runner.run().
        runner = None
        launched.runtime.close_sync()
        _close_launched(launched, None)
        launched = None
        atomic_write_json(
            result_path,
            {
                "status": "passed",
                "role": role,
                "scope": "model_init_only",
                "task8_accepted": False,
                "elapsed_seconds": time.monotonic() - started,
                "message": (
                    "Cold model initialization and offload passed; no generation, "
                    "preemption, or GRPO work was executed."
                ),
            },
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error: Exception | None = None
        if runner is not None:
            try:
                runner._finish_run()
            except Exception as exc:
                if primary_error is None:
                    cleanup_error = exc
                else:
                    primary_error.add_note(
                        "runner logging cleanup also failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        _close_launched(launched, primary_error or cleanup_error)
        if ray.is_initialized():
            ray.shutdown()
        if cleanup_error is not None:
            raise cleanup_error


def _collect_fixed_residencies(launched: Any) -> list[dict[str, Any]]:
    """Collect and validate post-initialization offload evidence."""
    statuses = []
    for group in (launched.actor, launched.rollout, launched.env):
        statuses.extend(group.get_rlix_fixed_residency().wait())
    if not statuses:
        raise AssertionError("model initialization returned no residency evidence")
    for status in statuses:
        if not status.safe_to_release:
            raise AssertionError(
                f"{status.component} rank {status.rank} remained GPU-resident"
            )
    return [
        asdict(status) | {"safe_to_release": status.safe_to_release}
        for status in statuses
    ]


def _actor_id(handle: Any) -> str:
    actor_id = handle._actor_id  # noqa: SLF001 - identity is acceptance evidence.
    to_hex = getattr(actor_id, "hex", None)
    return to_hex() if callable(to_hex) else str(actor_id)


def _required_environment_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise ValueError(f"{name} must be set by the Task 8 orchestrator")
    return Path(raw)


def _wait_for(path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for orchestrator gate {path}")
        time.sleep(0.05)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--connectivity-only", action="store_true")
    mode.add_argument("--model-init-only", action="store_true")
    parser.add_argument("--address", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode", choices=("disaggregated", "collocated"), required=True
    )
    parser.add_argument("--bundles", required=True)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if args.connectivity_only:
        if args.config is not None:
            raise ValueError("--config is only valid with --model-init-only")
        run_connectivity_driver(args)
    else:
        run_model_init_driver(args)


if __name__ == "__main__":
    main()
