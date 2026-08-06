"""Isolated four-rank FSDP variant of the Task 8 generation-proof driver.

The production Task 8 driver remains unchanged. This entrypoint reuses its
recording workers and lifecycle implementation while selecting the explicit
completed-bundle handoff needed when actor training overlaps every generation
bundle.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import task8_two_pipeline_driver as task8_driver
from omegaconf import OmegaConf, open_dict

from rlinf.scheduler.rlix.entrypoint import (
    launch_registered_rlix_workers as _launch_registered_rlix_workers,
)

_compose_wan_model_driver_config = task8_driver.compose_wan_model_driver_config


def _validate_four_rank_config(argv: list[str]) -> None:
    """Reject accidental use with the legacy one-rank training topology."""
    try:
        config_index = argv.index("--config") + 1
        config_path = Path(argv[config_index]).resolve()
    except (ValueError, IndexError) as exc:
        raise ValueError("four-rank FSDP driver requires --config") from exc
    cfg = OmegaConf.load(config_path)
    actor_gpus = tuple(int(gpu_id) for gpu_id in cfg.smoke.actor_gpus)
    if actor_gpus != (0, 1, 2, 3):
        raise ValueError("four-rank FSDP driver requires smoke.actor_gpus=[0,1,2,3]")
    handoff = str(cfg.smoke.get("completed_bundle_handoff", ""))
    if handoff != "release_before_training":
        raise ValueError(
            "four-rank FSDP driver requires "
            "smoke.completed_bundle_handoff=release_before_training"
        )
    if str(cfg.smoke.get("policy_sync_mode", "")) != "async_cpu_prefetch":
        raise ValueError(
            "four-rank FSDP driver requires async_cpu_prefetch policy sync"
        )
    total_num_envs = int(cfg.smoke.total_num_envs)
    group_size = int(cfg.smoke.group_size)
    rollout_epoch = int(cfg.smoke.rollout_epoch)
    env_world_size = len(tuple(cfg.smoke.env_gpu))
    actor_world_size = len(actor_gpus)
    if min(total_num_envs, group_size, rollout_epoch, env_world_size) <= 0:
        raise ValueError("environment, group, epoch, and world sizes must be positive")
    if total_num_envs % env_world_size != 0:
        raise ValueError("total_num_envs must divide evenly across environment ranks")
    env_local_batch = total_num_envs // env_world_size * rollout_epoch
    actor_splits_per_env = actor_world_size // env_world_size
    if actor_world_size % env_world_size != 0:
        raise ValueError(
            "four-rank FSDP proof requires actor world size divisible by env world size"
        )
    if env_local_batch % actor_splits_per_env != 0:
        raise ValueError("environment batch must divide evenly across actor receivers")
    actor_local_batch = env_local_batch // actor_splits_per_env
    if actor_local_batch % group_size != 0:
        raise ValueError(
            "each FSDP actor rank must receive complete GRPO groups: "
            f"local_actor_batch={actor_local_batch}, group_size={group_size}, "
            f"total_num_envs={total_num_envs}, rollout_epoch={rollout_epoch}, "
            f"actor_world_size={actor_world_size}"
        )


def _launch_four_rank_fsdp_workers(**kwargs: Any) -> Any:
    """Launch with completed bundles released before the all-GPU train request."""
    kwargs["completed_bundle_handoff"] = "release_before_training"
    kwargs["policy_sync_mode"] = "async_cpu_prefetch"
    kwargs["policy_sync_max_retries"] = 1
    return _launch_registered_rlix_workers(**kwargs)


def _compose_four_rank_fsdp_config(*args: Any, **kwargs: Any) -> Any:
    """Expose the selected handoff in the composed runtime configuration."""
    config_path = args[0] if args else kwargs["config_path"]
    smoke_cfg = OmegaConf.load(config_path)
    cfg, channels = _compose_wan_model_driver_config(*args, **kwargs)
    with open_dict(cfg):
        cfg.rlix.completed_bundle_handoff = "release_before_training"
        cfg.rlix.policy_sync = {
            "mode": str(smoke_cfg.smoke.policy_sync_mode),
            "bucket_size_mb": int(smoke_cfg.smoke.policy_sync_bucket_size_mb),
            "max_cached_versions": int(smoke_cfg.smoke.policy_sync_max_cached_versions),
            "max_retries": int(smoke_cfg.smoke.policy_sync_max_retries),
        }
        cfg.rlix.residency_validation_mode = os.environ.get(
            "RLINF_TASK8_RESIDENCY_VALIDATION_MODE", "deep"
        )
        cfg.rlix.snapshot_validation_mode = os.environ.get(
            "RLINF_TASK8_SNAPSHOT_VALIDATION_MODE", "deep"
        )
    return cfg, channels


def main() -> None:
    """Validate the isolated topology and run the copied Task 8 lifecycle."""
    _validate_four_rank_config(sys.argv[1:])
    # The shared driver supplies this function as the bootstrap dependency at
    # each launch site. Rebinding only this imported module keeps the original
    # file and its default behavior untouched.
    task8_driver.launch_registered_rlix_workers = _launch_four_rank_fsdp_workers
    task8_driver.compose_wan_model_driver_config = _compose_four_rank_fsdp_config
    task8_driver.main()


if __name__ == "__main__":
    main()
