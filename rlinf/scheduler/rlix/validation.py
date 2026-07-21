"""Fail-closed validation for the opt-in RLix elastic VLA mode."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from .placement import RLixPlacementPlan, validate_rlix_placement_plan


class RLixConfigurationError(ValueError):
    """An enabled RLix setting is outside the Task 6 supported contract."""


_RLIX_DEFAULTS: dict[str, Any] = {
    "rollout_allocation_policy": "elastic",
    "rollout_safe_point": "world_model_chunk",
    "progress_unit": "trajectories",
    "worker_max_concurrency": 2,
    "operation_timeout_s": 300.0,
    "enable_gpu_tracing": False,
}


def _select(cfg: DictConfig, path: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, path, default=default)


def _reject(path: str, value: Any, requirement: str) -> None:
    raise RLixConfigurationError(f"RLix requires {path} {requirement}; got {value!r}")


def _require_exact(cfg: DictConfig, path: str, expected: Any) -> None:
    value = _select(cfg, path)
    if value != expected or type(value) is not type(expected):
        _reject(path, value, f"to be {expected!r}")


def _require_true(cfg: DictConfig, path: str) -> None:
    _require_exact(cfg, path, True)


def _require_false(cfg: DictConfig, path: str) -> None:
    value = _select(cfg, path, False)
    if value is not False:
        _reject(path, value, "to be false")


def normalize_rlix_config(cfg: DictConfig) -> None:
    """Normalize only the documented RLix schema without importing Ray/core."""
    if not isinstance(cfg, DictConfig):
        raise TypeError("cfg must be an OmegaConf DictConfig")
    with open_dict(cfg):
        if "rlix" not in cfg or cfg.rlix is None:
            cfg.rlix = OmegaConf.create({"enabled": False})
            return
        if not isinstance(cfg.rlix, DictConfig):
            _reject("rlix", cfg.rlix, "to be a mapping")
        enabled = cfg.rlix.get("enabled", False)
        if not isinstance(enabled, bool):
            _reject("rlix.enabled", enabled, "to be a boolean")
        cfg.rlix.enabled = enabled
        if not enabled:
            return
        unknown = sorted(set(cfg.rlix) - ({"enabled"} | set(_RLIX_DEFAULTS)))
        if unknown:
            _reject("rlix", unknown, "to contain only supported Task 6 keys")
        for key, value in _RLIX_DEFAULTS.items():
            if key not in cfg.rlix:
                cfg.rlix[key] = value


def validate_elastic_vla_config(cfg: DictConfig) -> None:
    """Validate the pure configuration half of the Task 6 contract."""
    normalize_rlix_config(cfg)
    if not cfg.rlix.enabled:
        return

    _require_exact(cfg, "rlix.rollout_allocation_policy", "elastic")
    _require_exact(cfg, "rlix.rollout_safe_point", "world_model_chunk")
    _require_exact(cfg, "rlix.progress_unit", "trajectories")
    concurrency = cfg.rlix.worker_max_concurrency
    if (
        not isinstance(concurrency, int)
        or isinstance(concurrency, bool)
        or concurrency < 2
    ):
        _reject("rlix.worker_max_concurrency", concurrency, "to be an integer >= 2")
    timeout = cfg.rlix.operation_timeout_s
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        _reject("rlix.operation_timeout_s", timeout, "to be a number > 0")
    if not isinstance(cfg.rlix.enable_gpu_tracing, bool):
        _reject(
            "rlix.enable_gpu_tracing",
            cfg.rlix.enable_gpu_tracing,
            "to be a boolean",
        )

    _require_exact(cfg, "cluster.num_nodes", 1)
    _require_exact(cfg, "runner.task_type", "embodied")
    _require_false(cfg, "runner.only_eval")
    _require_false(cfg, "runner.enable_decoupled_mode")
    _require_false(cfg, "runner.use_training_pipeline")
    _require_false(cfg, "runner.overlap_env_bootstrap")
    weight_sync_interval = _select(cfg, "runner.weight_sync_interval", 1)
    if (
        not isinstance(weight_sync_interval, int)
        or isinstance(weight_sync_interval, bool)
        or weight_sync_interval != 1
    ):
        _reject("runner.weight_sync_interval", weight_sync_interval, "to be integer 1")
    _require_exact(cfg, "actor.training_backend", "fsdp")
    _require_exact(cfg, "rollout.generation_backend", "huggingface")
    _require_exact(cfg, "rollout.pipeline_stage_num", 1)

    for path in (
        "actor.model.tensor_model_parallel_size",
        "actor.model.pipeline_model_parallel_size",
        "rollout.model.tensor_model_parallel_size",
        "rollout.model.pipeline_model_parallel_size",
    ):
        value = _select(cfg, path, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value != 1:
            _reject(path, value, "to be integer 1")

    _require_true(cfg, "actor.enable_offload")
    _require_true(cfg, "rollout.enable_offload")
    _require_true(cfg, "env.train.enable_offload")
    init_offload = _select(cfg, "env.train.enable_init_offload", True)
    if init_offload is not True:
        _reject("env.train.enable_init_offload", init_offload, "to be true")
    if _select(cfg, "runner.val_check_interval", -1) > 0:
        _require_true(cfg, "env.eval.enable_offload")

    env_type = _select(cfg, "env.train.env_type")
    if env_type not in {"wan_wm", "opensora_wm"}:
        _reject("env.train.env_type", env_type, "to be 'wan_wm' or 'opensora_wm'")
    _require_true(cfg, "env.train.use_fixed_reset_state_ids")
    _require_false(cfg, "env.train.data_collection.enabled")
    _require_false(cfg, "algorithm.dagger.online_lerobot.enabled")
    if _select(cfg, "algorithm.loss_type", "") == "rlt_ac":
        _reject("algorithm.loss_type", "rlt_ac", "to exclude RLT state")
    if _select(cfg, "reward.reward_mode", "per_step") == "history_buffer":
        _reject(
            "reward.reward_mode", "history_buffer", "to exclude history-buffer state"
        )
    _require_false(cfg, "reward.use_reward_model")


def validate_rlix_entrypoint(cfg: DictConfig, *, entrypoint: str) -> None:
    """Reject enabled RLix mode outside the synchronous embodied entrypoint."""
    enabled = _select(cfg, "rlix.enabled", False)
    if enabled is not True:
        return
    if entrypoint != "train_embodied_agent":
        raise RLixConfigurationError(
            "RLix Task 6 supports only examples/embodiment/"
            f"train_embodied_agent.py; got entrypoint {entrypoint!r}"
        )


def validate_elastic_vla_placement(
    plan: RLixPlacementPlan,
    cluster: Any,
    *,
    ray_nodes: list[dict[str, Any]] | None = None,
) -> None:
    """Require one live GPU node and matching RLinf/Ray GPU numbering."""
    validate_rlix_placement_plan(plan)
    if ray_nodes is None:
        import ray

        ray_nodes = ray.nodes()
    gpu_nodes = [
        node
        for node in ray_nodes
        if node.get("Alive", False)
        and float(node.get("Resources", {}).get("GPU", 0)) > 0
    ]
    if len(gpu_nodes) != 1:
        raise RLixConfigurationError(
            "RLix requires exactly one alive Ray GPU-bearing node; "
            f"got {len(gpu_nodes)}"
        )
    ray_gpu_count_raw = gpu_nodes[0]["Resources"]["GPU"]
    ray_gpu_count = int(ray_gpu_count_raw)
    if ray_gpu_count <= 0 or ray_gpu_count != ray_gpu_count_raw:
        raise RLixConfigurationError(
            f"RLix requires a positive integral Ray GPU count; got {ray_gpu_count_raw!r}"
        )
    node_info = cluster.get_node_info(0)
    rlinf_gpu_count = node_info.num_accelerators
    if rlinf_gpu_count != ray_gpu_count:
        raise RLixConfigurationError(
            "RLinf and Ray GPU counts must match for local GPU numbering; "
            f"RLinf={rlinf_gpu_count}, Ray={ray_gpu_count}"
        )
    used_gpus = {
        worker.local_gpu
        for workers in (plan.actor_workers, plan.rollout_workers, plan.env_workers)
        for worker in workers
    }
    invalid = sorted(gpu for gpu in used_gpus if gpu >= ray_gpu_count)
    if invalid:
        raise RLixConfigurationError(
            f"RLix placement uses GPUs outside the live Ray range: {invalid}"
        )


__all__ = [
    "RLixConfigurationError",
    "normalize_rlix_config",
    "validate_elastic_vla_config",
    "validate_rlix_entrypoint",
    "validate_elastic_vla_placement",
]
