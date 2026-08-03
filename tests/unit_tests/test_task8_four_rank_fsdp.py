"""CPU contract tests for the isolated Task 8 four-rank FSDP harness."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from rlix_core.protocol.types import ACTOR_TRAIN_CLUSTER_NAME

_EMBODIED_TESTS = Path(__file__).resolve().parents[1] / "e2e_tests" / "embodied"
sys.path.insert(0, str(_EMBODIED_TESTS))

from task8_four_rank_fsdp_acceptance import (  # noqa: E402
    _validate_four_rank_lifecycle,
    _validate_four_rank_ready,
)
from task8_four_rank_fsdp_driver import _validate_four_rank_config  # noqa: E402


def test_four_rank_driver_config_contract_accepts_isolated_yaml() -> None:
    config = _EMBODIED_TESTS / "task8_wan_disaggregated_four_rank_fsdp.yaml"

    _validate_four_rank_config(["--config", str(config)])


def test_four_rank_driver_config_rejects_legacy_actor_topology(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "smoke": {
                "actor_gpus": [0],
                "completed_bundle_handoff": "release_before_training",
            }
        }
    )
    path = tmp_path / "legacy.yaml"
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="actor_gpus"):
        _validate_four_rank_config(["--config", str(path)])


def test_four_rank_driver_rejects_partial_local_grpo_group(tmp_path: Path) -> None:
    source = OmegaConf.load(
        _EMBODIED_TESTS / "task8_wan_disaggregated_four_rank_fsdp.yaml"
    )
    source.smoke.rollout_epoch = 1
    path = tmp_path / "partial-group.yaml"
    OmegaConf.save(source, path)

    with pytest.raises(ValueError, match="complete GRPO groups"):
        _validate_four_rank_config(["--config", str(path)])


def test_four_rank_readiness_requires_all_gpu_actor_mapping() -> None:
    ready = {
        role: {
            "candidate_mapping": {ACTOR_TRAIN_CLUSTER_NAME: [0, 1, 2, 3]},
            "actor_infer_bundles": [[0, 2], [1, 3]],
        }
        for role in ("a", "b")
    }

    _validate_four_rank_ready(ready)
    ready["b"]["candidate_mapping"][ACTOR_TRAIN_CLUSTER_NAME] = [0]
    with pytest.raises(ValueError, match="actor_train mapping"):
        _validate_four_rank_ready(ready)


def _event(
    role: str,
    component: str,
    rank: int,
    event: str,
    **details,
) -> SimpleNamespace:
    return SimpleNamespace(
        driver_role=role,
        component=component,
        dp_rank=rank,
        event=event,
        details=details,
    )


def test_four_rank_lifecycle_requires_two_rank_resume_and_four_rank_training() -> None:
    events = []
    for rank in (0, 1):
        for event in (
            "drain_requested",
            "snapshot_completed",
            "environment_offload_verified",
            "environment_onload_verified",
            "restore_validated",
        ):
            events.append(_event("b", "environment", rank, event))
        for event in (
            "drain_requested",
            "rollout_offload_verified",
            "rollout_onload_verified",
        ):
            events.append(_event("b", "rollout", rank, event))
    for rank in range(4):
        events.append(_event("a", "actor", rank, "training_completed"))
    for role in ("a", "b"):
        for version in (1, 2):
            events.append(
                _event(
                    role,
                    "actor",
                    0,
                    "policy_cache_promoted",
                    policy_version=version,
                    promoted=True,
                )
            )
        events.append(
            _event(
                role,
                "rollout",
                0,
                "async_policy_update_committed",
                policy_version=1,
            )
        )

    _validate_four_rank_lifecycle(events, iterations=2)
    events = [event for event in events if event.event != "restore_validated"]
    with pytest.raises(ValueError, match="preemption evidence"):
        _validate_four_rank_lifecycle(events, iterations=2)
