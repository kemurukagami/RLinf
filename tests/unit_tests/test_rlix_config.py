"""Tests for Task 6 opt-in RLix configuration validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.scheduler.rlix.validation import (
    RLixConfigurationError,
    normalize_rlix_config,
    validate_elastic_vla_config,
    validate_rlix_entrypoint,
)


def _valid_config():
    return OmegaConf.create(
        {
            "rlix": {"enabled": True},
            "cluster": {"num_nodes": 1},
            "runner": {"task_type": "embodied", "val_check_interval": -1},
            "actor": {
                "training_backend": "fsdp",
                "enable_offload": True,
                "model": {},
            },
            "rollout": {
                "generation_backend": "huggingface",
                "pipeline_stage_num": 1,
                "enable_offload": True,
                "model": {},
            },
            "env": {
                "train": {
                    "env_type": "wan_wm",
                    "enable_offload": True,
                    "use_fixed_reset_state_ids": True,
                },
                "eval": {"enable_offload": True},
            },
            "algorithm": {"loss_type": "actor"},
            "reward": {"use_reward_model": False},
        }
    )


def test_absent_rlix_section_normalizes_to_disabled_only() -> None:
    cfg = OmegaConf.create({"runner": {}})

    normalize_rlix_config(cfg)

    assert OmegaConf.to_container(cfg.rlix) == {"enabled": False}


def test_disabled_mode_does_not_enforce_elastic_requirements() -> None:
    cfg = OmegaConf.create({"rlix": {"enabled": False}, "cluster": {"num_nodes": 9}})

    validate_elastic_vla_config(cfg)

    assert OmegaConf.to_container(cfg.rlix) == {"enabled": False}


def test_enabled_mode_inserts_documented_defaults() -> None:
    cfg = _valid_config()

    validate_elastic_vla_config(cfg)

    assert OmegaConf.to_container(cfg.rlix) == {
        "enabled": True,
        "rollout_allocation_policy": "elastic",
        "rollout_safe_point": "world_model_chunk",
        "progress_unit": "trajectories",
        "worker_max_concurrency": 2,
        "operation_timeout_s": 300.0,
        "enable_gpu_tracing": False,
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("rlix.worker_max_concurrency", True, "worker_max_concurrency"),
        ("rlix.worker_max_concurrency", 1, "worker_max_concurrency"),
        ("rlix.operation_timeout_s", False, "operation_timeout_s"),
        ("rlix.operation_timeout_s", 0, "operation_timeout_s"),
        ("rlix.enable_gpu_tracing", 1, "enable_gpu_tracing"),
        ("cluster.num_nodes", 2, "cluster.num_nodes"),
        ("runner.task_type", "embodied_eval", "runner.task_type"),
        ("runner.only_eval", True, "runner.only_eval"),
        ("runner.enable_decoupled_mode", True, "enable_decoupled_mode"),
        ("runner.use_training_pipeline", True, "use_training_pipeline"),
        ("runner.overlap_env_bootstrap", True, "overlap_env_bootstrap"),
        ("runner.weight_sync_interval", 2, "weight_sync_interval"),
        ("actor.training_backend", "megatron", "training_backend"),
        ("rollout.generation_backend", "vllm", "generation_backend"),
        ("rollout.pipeline_stage_num", 2, "pipeline_stage_num"),
        ("actor.model.tensor_model_parallel_size", 2, "tensor_model_parallel_size"),
        (
            "rollout.model.pipeline_model_parallel_size",
            2,
            "pipeline_model_parallel_size",
        ),
        ("actor.enable_offload", False, "actor.enable_offload"),
        ("rollout.enable_offload", False, "rollout.enable_offload"),
        ("env.train.enable_offload", False, "env.train.enable_offload"),
        ("env.train.enable_init_offload", False, "enable_init_offload"),
        ("env.train.env_type", "libero", "env.train.env_type"),
        ("env.train.use_fixed_reset_state_ids", False, "use_fixed_reset_state_ids"),
        ("env.train.data_collection.enabled", True, "data_collection"),
        ("algorithm.dagger.online_lerobot.enabled", True, "online_lerobot"),
        ("algorithm.loss_type", "rlt_ac", "algorithm.loss_type"),
        ("reward.reward_mode", "history_buffer", "reward.reward_mode"),
        ("reward.use_reward_model", True, "use_reward_model"),
    ],
)
def test_enabled_mode_rejects_unsupported_configuration(
    path: str,
    value: object,
    message: str,
) -> None:
    cfg = _valid_config()
    OmegaConf.update(cfg, path, value, force_add=True)

    with pytest.raises(RLixConfigurationError, match=message):
        validate_elastic_vla_config(cfg)


def test_enabled_evaluation_requires_environment_offload() -> None:
    cfg = _valid_config()
    cfg.runner.val_check_interval = 10
    cfg.env.eval.enable_offload = False

    with pytest.raises(RLixConfigurationError, match="env.eval.enable_offload"):
        validate_elastic_vla_config(cfg)


def test_unknown_rlix_key_is_rejected() -> None:
    cfg = _valid_config()
    cfg.rlix.unrecognized = True

    with pytest.raises(RLixConfigurationError, match="supported Task 6 keys"):
        validate_elastic_vla_config(cfg)


def test_enabled_mode_rejects_async_entrypoint() -> None:
    with pytest.raises(RLixConfigurationError, match="train_embodied_agent.py"):
        validate_rlix_entrypoint(_valid_config(), entrypoint="train_async")


def test_disabled_mode_does_not_restrict_entrypoint() -> None:
    cfg = _valid_config()
    cfg.rlix.enabled = False

    validate_rlix_entrypoint(cfg, entrypoint="train_async")


def test_dedicated_wan_example_composes_and_passes_pure_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
    )
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))

    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="wan_libero_spatial_grpo_openvlaoft_rlix")

    validate_rlix_entrypoint(cfg, entrypoint="train_embodied_agent")
    validate_elastic_vla_config(cfg)
    assert cfg.rlix.enabled is True
    assert cfg.env.train.env_type == "wan_wm"
    assert cfg.runner.logger.experiment_name.endswith("_rlix")
