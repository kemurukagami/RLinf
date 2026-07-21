"""Placement conversion for the opt-in RLix embodied pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from rlinf.scheduler.hardware import AcceleratorType
from rlinf.scheduler.placement.placement import Placement


@dataclass(frozen=True, slots=True)
class ResolvedGPUWorker:
    """Immutable GPU ownership projected from an RLinf worker placement."""

    rank: int
    cluster_node_rank: int
    local_gpu: int
    accelerator_type: AcceleratorType


@dataclass(frozen=True, slots=True)
class RLixPlacementPlan:
    """Canonical RLix registration topology derived from resolved placements."""

    actor_workers: tuple[ResolvedGPUWorker, ...]
    rollout_workers: tuple[ResolvedGPUWorker, ...]
    env_workers: tuple[ResolvedGPUWorker, ...]
    initialization_devices: tuple[int, ...]
    actor_train_devices: tuple[int, ...]
    actor_infer_devices: tuple[int, ...]
    policy_sync_devices: tuple[int, ...]
    evaluation_devices: tuple[int, ...]
    actor_infer_bundles: tuple[tuple[int, tuple[int, ...]], ...]

    def registration_payload(self) -> dict[str, Any]:
        """Return fresh mutable values accepted by ``register_pipeline``."""
        from rlix_core.protocol.types import (
            ACTOR_TRAIN_CLUSTER_NAME,
            EVALUATION_CLUSTER_NAME,
            GENERATION_CLUSTER_NAME,
            INITIALIZATION_CLUSTER_NAME,
            POLICY_SYNC_CLUSTER_NAME,
            AllocationPolicy,
        )

        names_and_devices = (
            (INITIALIZATION_CLUSTER_NAME, self.initialization_devices),
            (ACTOR_TRAIN_CLUSTER_NAME, self.actor_train_devices),
            (GENERATION_CLUSTER_NAME, self.actor_infer_devices),
            (POLICY_SYNC_CLUSTER_NAME, self.policy_sync_devices),
            (EVALUATION_CLUSTER_NAME, self.evaluation_devices),
        )
        return {
            "cluster_tp_configs": {name: 1 for name, _ in names_and_devices},
            "cluster_device_mappings": {
                name: list(devices) for name, devices in names_and_devices
            },
            "cluster_allocation_policies": {
                INITIALIZATION_CLUSTER_NAME: AllocationPolicy.FIXED.value,
                ACTOR_TRAIN_CLUSTER_NAME: AllocationPolicy.FIXED.value,
                GENERATION_CLUSTER_NAME: AllocationPolicy.ELASTIC.value,
                POLICY_SYNC_CLUSTER_NAME: AllocationPolicy.FIXED.value,
                EVALUATION_CLUSTER_NAME: AllocationPolicy.FIXED.value,
            },
            "cluster_dp_device_mappings": {
                GENERATION_CLUSTER_NAME: {
                    rank: list(bundle) for rank, bundle in self.actor_infer_bundles
                }
            },
        }


@dataclass(frozen=True, slots=True)
class ResolvedRLixPlacements:
    """Canonical plan plus the exact replay strategies used for worker launch."""

    plan: RLixPlacementPlan
    actor_strategy: "ResolvedPlacementStrategy"
    rollout_strategy: "ResolvedPlacementStrategy"
    env_strategy: "ResolvedPlacementStrategy"


class ResolvedPlacementStrategy:
    """Placement strategy that replays defensive copies of one resolution."""

    def __init__(self, placements: Sequence[Placement]) -> None:
        """Store defensive copies of already resolved placements."""
        if not placements:
            raise ValueError("resolved placements must be non-empty")
        self._placements = tuple(_copy_placement(item) for item in placements)

    def get_placement(
        self,
        cluster: Any,
        isolate_accelerator: bool = True,
    ) -> list[Placement]:
        """Return the pre-resolved placements without resolving configuration again."""
        del cluster, isolate_accelerator
        return [_copy_placement(item) for item in self._placements]


def _copy_placement(placement: Placement) -> Placement:
    return replace(
        placement,
        visible_accelerators=list(placement.visible_accelerators),
        local_hardware_ranks=list(placement.local_hardware_ranks),
    )


def _validate_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    return value


def _project_component(
    component: str,
    placements: Sequence[Placement],
    *,
    node_gpu_count: int,
) -> tuple[ResolvedGPUWorker, ...]:
    if not placements:
        raise ValueError(f"RLix {component} placement must be non-empty")

    workers: dict[int, ResolvedGPUWorker] = {}
    owned_gpus: set[int] = set()
    for placement in placements:
        rank = _validate_int(placement.rank, field=f"{component} rank")
        if rank < 0:
            raise ValueError(f"{component} rank must be non-negative, got {rank}")
        if rank in workers:
            raise ValueError(f"{component} placement contains duplicate rank {rank}")

        node_rank = _validate_int(
            placement.cluster_node_rank,
            field=f"{component} rank {rank} cluster_node_rank",
        )
        if node_rank != 0:
            raise ValueError(
                f"{component} rank {rank} must be on cluster node 0, got {node_rank}"
            )
        if placement.accelerator_type != AcceleratorType.NV_GPU:
            raise ValueError(
                f"{component} rank {rank} must use {AcceleratorType.NV_GPU.value}, "
                f"got {placement.accelerator_type!r}"
            )
        hardware_ranks = placement.local_hardware_ranks
        if not isinstance(hardware_ranks, list) or len(hardware_ranks) != 1:
            raise ValueError(
                f"{component} rank {rank} must own exactly one local GPU, "
                f"got {hardware_ranks!r}"
            )
        gpu = _validate_int(hardware_ranks[0], field=f"{component} rank {rank} GPU")
        if gpu < 0 or gpu >= node_gpu_count:
            raise ValueError(
                f"{component} rank {rank} GPU {gpu} is outside node GPU range "
                f"0..{node_gpu_count - 1}"
            )
        if gpu in owned_gpus:
            raise ValueError(
                f"{component} placement assigns GPU {gpu} to multiple workers"
            )
        owned_gpus.add(gpu)
        workers[rank] = ResolvedGPUWorker(
            rank=rank,
            cluster_node_rank=node_rank,
            local_gpu=gpu,
            accelerator_type=AcceleratorType.NV_GPU,
        )

    expected_ranks = list(range(len(workers)))
    if sorted(workers) != expected_ranks:
        raise ValueError(
            f"{component} ranks must be contiguous {expected_ranks}, got {sorted(workers)}"
        )
    return tuple(workers[rank] for rank in expected_ranks)


def _device_union(*worker_groups: Iterable[ResolvedGPUWorker]) -> tuple[int, ...]:
    return tuple(
        sorted({worker.local_gpu for group in worker_groups for worker in group})
    )


def build_rlix_placement_plan(
    *,
    actor_placements: Sequence[Placement],
    rollout_placements: Sequence[Placement],
    env_placements: Sequence[Placement],
    node_gpu_count: int,
) -> RLixPlacementPlan:
    """Validate resolved placements and derive the exact five-cluster topology."""
    node_gpu_count = _validate_int(node_gpu_count, field="node_gpu_count")
    if node_gpu_count <= 0:
        raise ValueError("node_gpu_count must be positive")

    actor_workers = _project_component(
        "actor", actor_placements, node_gpu_count=node_gpu_count
    )
    rollout_workers = _project_component(
        "rollout", rollout_placements, node_gpu_count=node_gpu_count
    )
    env_workers = _project_component(
        "environment", env_placements, node_gpu_count=node_gpu_count
    )

    rollout_ranks = {worker.rank for worker in rollout_workers}
    env_ranks = {worker.rank for worker in env_workers}
    if rollout_ranks != env_ranks:
        raise ValueError(
            "rollout and environment ranks must match exactly; "
            f"rollout={sorted(rollout_ranks)}, environment={sorted(env_ranks)}"
        )

    env_by_rank = {worker.rank: worker for worker in env_workers}
    bundles: list[tuple[int, tuple[int, ...]]] = []
    seen_bundle_gpus: set[int] = set()
    bundle_width: int | None = None
    for rollout_worker in rollout_workers:
        env_worker = env_by_rank[rollout_worker.rank]
        bundle = tuple(dict.fromkeys((rollout_worker.local_gpu, env_worker.local_gpu)))
        if bundle_width is None:
            bundle_width = len(bundle)
        elif len(bundle) != bundle_width:
            raise ValueError(
                "actor_infer bundles must have uniform width; "
                f"rank {rollout_worker.rank} has width {len(bundle)}, expected {bundle_width}"
            )
        overlap = seen_bundle_gpus & set(bundle)
        if overlap:
            raise ValueError(
                "actor_infer bundles must be disjoint; "
                f"rank {rollout_worker.rank} overlaps GPUs {sorted(overlap)}"
            )
        seen_bundle_gpus.update(bundle)
        bundles.append((rollout_worker.rank, bundle))

    actor_infer_devices = tuple(sorted(seen_bundle_gpus))
    plan = RLixPlacementPlan(
        actor_workers=actor_workers,
        rollout_workers=rollout_workers,
        env_workers=env_workers,
        initialization_devices=_device_union(
            actor_workers, rollout_workers, env_workers
        ),
        actor_train_devices=_device_union(actor_workers),
        actor_infer_devices=actor_infer_devices,
        policy_sync_devices=_device_union(actor_workers, rollout_workers),
        evaluation_devices=_device_union(rollout_workers, env_workers),
        actor_infer_bundles=tuple(bundles),
    )
    validate_rlix_placement_plan(plan)
    return plan


def resolve_rlix_placements(
    component_placement: Any, cluster: Any
) -> ResolvedRLixPlacements:
    """Resolve actor, rollout, and environment strategies exactly once."""
    resolved: dict[str, list[Placement]] = {}
    for component in ("actor", "rollout", "env"):
        try:
            strategy = component_placement.get_strategy(component)
            resolved[component] = strategy.get_placement(
                cluster, isolate_accelerator=True
            )
        except Exception as exc:
            raise ValueError(
                f"failed to resolve RLix {component} placement: {exc}"
            ) from exc
    node_gpu_count = cluster.get_node_info(0).num_accelerators
    plan = build_rlix_placement_plan(
        actor_placements=resolved["actor"],
        rollout_placements=resolved["rollout"],
        env_placements=resolved["env"],
        node_gpu_count=node_gpu_count,
    )
    return ResolvedRLixPlacements(
        plan=plan,
        actor_strategy=ResolvedPlacementStrategy(resolved["actor"]),
        rollout_strategy=ResolvedPlacementStrategy(resolved["rollout"]),
        env_strategy=ResolvedPlacementStrategy(resolved["env"]),
    )


def validate_rlix_placement_plan(plan: RLixPlacementPlan) -> None:
    """Self-check immutable plan relationships before registration."""
    if not isinstance(plan, RLixPlacementPlan):
        raise TypeError("plan must be an RLixPlacementPlan")
    bundle_union = {gpu for _, bundle in plan.actor_infer_bundles for gpu in bundle}
    if bundle_union != set(plan.actor_infer_devices) or len(bundle_union) != len(
        plan.actor_infer_devices
    ):
        raise ValueError(
            "actor_infer bundle union must exactly equal actor_infer_devices"
        )
    expected = {
        "initialization_devices": _device_union(
            plan.actor_workers, plan.rollout_workers, plan.env_workers
        ),
        "actor_train_devices": _device_union(plan.actor_workers),
        "policy_sync_devices": _device_union(plan.actor_workers, plan.rollout_workers),
        "evaluation_devices": _device_union(plan.rollout_workers, plan.env_workers),
    }
    for field, expected_devices in expected.items():
        if getattr(plan, field) != expected_devices:
            raise ValueError(f"{field} does not match its canonical worker union")


__all__ = [
    "RLixPlacementPlan",
    "ResolvedGPUWorker",
    "ResolvedPlacementStrategy",
    "ResolvedRLixPlacements",
    "build_rlix_placement_plan",
    "resolve_rlix_placements",
    "validate_rlix_placement_plan",
]
