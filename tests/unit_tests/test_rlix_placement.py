"""Tests for immutable RLix placement conversion."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rlinf.scheduler.hardware import AcceleratorType
from rlinf.scheduler.placement.placement import Placement
from rlinf.scheduler.rlix.placement import (
    ResolvedPlacementStrategy,
    build_rlix_placement_plan,
    resolve_rlix_placements,
    validate_rlix_placement_plan,
)
from rlinf.scheduler.rlix.validation import (
    RLixConfigurationError,
    validate_elastic_vla_placement,
)


def _placement(
    rank: int,
    gpu: int,
    *,
    node: int = 0,
    accelerator_type: AcceleratorType = AcceleratorType.NV_GPU,
    hardware_ranks: list[int] | None = None,
) -> Placement:
    return Placement(
        rank=rank,
        cluster_node_rank=node,
        placement_node_rank=0,
        local_accelerator_rank=gpu,
        accelerator_type=accelerator_type,
        local_rank=rank,
        local_world_size=1,
        visible_accelerators=[str(gpu)],
        isolate_accelerator=True,
        local_hardware_ranks=[gpu] if hardware_ranks is None else hardware_ranks,
        node_group_label="default",
    )


def _plan(
    *,
    actor_gpus: tuple[int, ...] = (4, 5),
    rollout_gpus: tuple[int, ...] = (0, 1),
    env_gpus: tuple[int, ...] = (2, 3),
):
    return build_rlix_placement_plan(
        actor_placements=[_placement(rank, gpu) for rank, gpu in enumerate(actor_gpus)],
        rollout_placements=[
            _placement(rank, gpu) for rank, gpu in enumerate(rollout_gpus)
        ],
        env_placements=[_placement(rank, gpu) for rank, gpu in enumerate(env_gpus)],
        node_gpu_count=8,
    )


def test_disaggregated_plan_builds_all_canonical_mappings() -> None:
    plan = _plan()

    assert plan.initialization_devices == (0, 1, 2, 3, 4, 5)
    assert plan.actor_train_devices == (4, 5)
    assert plan.actor_infer_devices == (0, 1, 2, 3)
    assert plan.policy_sync_devices == (0, 1, 4, 5)
    assert plan.evaluation_devices == (0, 1, 2, 3)
    assert plan.actor_infer_bundles == ((0, (0, 2)), (1, (1, 3)))


def test_collocated_plan_deduplicates_each_rank_bundle() -> None:
    plan = _plan(actor_gpus=(0, 1), rollout_gpus=(0, 1), env_gpus=(0, 1))

    assert plan.initialization_devices == (0, 1)
    assert plan.actor_infer_devices == (0, 1)
    assert plan.actor_infer_bundles == ((0, (0,)), (1, (1,)))


def test_registration_payload_is_complete_and_defensively_copied() -> None:
    plan = _plan()
    first = plan.registration_payload()

    assert first["cluster_tp_configs"] == {
        "initialization": 1,
        "actor_train": 1,
        "actor_infer": 1,
        "policy_sync": 1,
        "evaluation": 1,
    }
    assert first["cluster_allocation_policies"] == {
        "initialization": "fixed",
        "actor_train": "fixed",
        "actor_infer": "elastic",
        "policy_sync": "fixed",
        "evaluation": "fixed",
    }
    first["cluster_device_mappings"]["actor_infer"].append(7)
    first["cluster_dp_device_mappings"]["actor_infer"][0].append(7)

    second = plan.registration_payload()
    assert second["cluster_device_mappings"]["actor_infer"] == [0, 1, 2, 3]
    assert second["cluster_dp_device_mappings"]["actor_infer"] == {
        0: [0, 2],
        1: [1, 3],
    }


def test_registration_payload_passes_core_protocol_validation() -> None:
    from rlix_core.protocol.validation import (
        RegisterValidationInput,
        validate_register_pipeline,
    )

    payload = _plan().registration_payload()
    validate_register_pipeline(
        RegisterValidationInput(
            pipeline_id="rlinf_000000000000",
            ray_namespace="rlix_pipeline_rlinf_000000000000",
            **payload,
        )
    )


def test_resolved_strategy_returns_fresh_placement_copies() -> None:
    original = _placement(0, 2)
    strategy = ResolvedPlacementStrategy([original])
    original.local_hardware_ranks.append(3)

    first = strategy.get_placement(object())
    first[0].local_hardware_ranks.append(4)
    second = strategy.get_placement(object())

    assert second[0].local_hardware_ranks == [2]
    assert second[0] is not first[0]


def test_component_strategies_are_resolved_once_then_replayed() -> None:
    calls = {"actor": 0, "rollout": 0, "env": 0}
    placements = {
        "actor": [_placement(0, 4)],
        "rollout": [_placement(0, 0)],
        "env": [_placement(0, 2)],
    }

    class _Strategy:
        def __init__(self, component: str) -> None:
            self.component = component

        def get_placement(self, cluster, isolate_accelerator=True):
            calls[self.component] += 1
            return placements[self.component]

    class _Components:
        def get_strategy(self, component: str):
            return _Strategy(component)

    resolved = resolve_rlix_placements(_Components(), _Cluster(8))
    resolved.actor_strategy.get_placement(_Cluster(8))
    resolved.rollout_strategy.get_placement(_Cluster(8))
    resolved.env_strategy.get_placement(_Cluster(8))

    assert calls == {"actor": 1, "rollout": 1, "env": 1}


@pytest.mark.parametrize(
    ("component", "placements", "message"),
    [
        ("actor", [], "actor placement must be non-empty"),
        ("actor", [_placement(0, 4), _placement(0, 5)], "duplicate rank"),
        ("actor", [_placement(1, 4)], "ranks must be contiguous"),
        ("actor", [_placement(-1, 4)], "rank must be non-negative"),
        ("actor", [_placement(True, 4)], "rank must be an integer"),
        ("actor", [_placement(0, 4, node=1)], "must be on cluster node 0"),
        (
            "actor",
            [_placement(0, 4, accelerator_type=AcceleratorType.NO_ACCEL)],
            "must use NV_GPU",
        ),
        ("actor", [_placement(0, 4, hardware_ranks=[])], "exactly one local GPU"),
        (
            "actor",
            [_placement(0, 4, hardware_ranks=[4, 5])],
            "exactly one local GPU",
        ),
        ("actor", [_placement(0, 8)], "outside node GPU range"),
        (
            "actor",
            [_placement(0, 4), _placement(1, 4)],
            "assigns GPU 4 to multiple workers",
        ),
    ],
)
def test_component_placement_rejections(
    component: str,
    placements: list[Placement],
    message: str,
) -> None:
    kwargs = {
        "actor_placements": [_placement(0, 4)],
        "rollout_placements": [_placement(0, 0)],
        "env_placements": [_placement(0, 2)],
        "node_gpu_count": 8,
    }
    kwargs[f"{component}_placements"] = placements

    with pytest.raises(ValueError, match=message):
        build_rlix_placement_plan(**kwargs)


def test_rollout_and_environment_rank_sets_must_match() -> None:
    with pytest.raises(ValueError, match="ranks must match exactly"):
        build_rlix_placement_plan(
            actor_placements=[_placement(0, 4)],
            rollout_placements=[_placement(0, 0), _placement(1, 1)],
            env_placements=[_placement(0, 2)],
            node_gpu_count=8,
        )


def test_generation_bundles_must_be_uniform() -> None:
    with pytest.raises(ValueError, match="uniform width"):
        _plan(rollout_gpus=(0, 1), env_gpus=(0, 3))


def test_generation_bundles_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="bundles must be disjoint"):
        _plan(rollout_gpus=(0, 1), env_gpus=(2, 0))


def test_plan_self_check_rejects_mutated_flat_generation_union() -> None:
    invalid = replace(_plan(), actor_infer_devices=(0, 1, 2))

    with pytest.raises(ValueError, match="bundle union"):
        validate_rlix_placement_plan(invalid)


class _Cluster:
    def __init__(self, gpu_count: int) -> None:
        self.gpu_count = gpu_count

    def get_node_info(self, rank: int):
        assert rank == 0
        return type("Node", (), {"num_accelerators": self.gpu_count})()


def test_live_topology_accepts_one_matching_gpu_node() -> None:
    validate_elastic_vla_placement(
        _plan(),
        _Cluster(8),
        ray_nodes=[{"Alive": True, "Resources": {"GPU": 8.0}}],
    )


@pytest.mark.parametrize(
    ("nodes", "cluster_gpus", "message"),
    [
        ([], 8, "exactly one alive"),
        (
            [
                {"Alive": True, "Resources": {"GPU": 4}},
                {"Alive": True, "Resources": {"GPU": 4}},
            ],
            8,
            "exactly one alive",
        ),
        ([{"Alive": True, "Resources": {"GPU": 8}}], 7, "GPU counts must match"),
    ],
)
def test_live_topology_rejections(
    nodes: list[dict],
    cluster_gpus: int,
    message: str,
) -> None:
    with pytest.raises(RLixConfigurationError, match=message):
        validate_elastic_vla_placement(
            _plan(),
            _Cluster(cluster_gpus),
            ray_nodes=nodes,
        )
