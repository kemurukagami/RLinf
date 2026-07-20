"""Driver-side construction and access helpers for the RLinf coordinator."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import ray
from rlix_core.protocol.types import COORDINATOR_ACTOR_NAME_PREFIX

from .coordinator import RLixResizeCoordinator
from .protocol import CoordinatorStatus, ElasticCollectionContext, PolicySyncLease


def _extract_ranked_worker_handles(worker_group: Any, *, label: str) -> dict[int, Any]:
    """Extract exact public WorkerRank entries without invoking group wrappers."""
    entries = getattr(worker_group, "worker_info_list", None)
    if entries is None:
        raise TypeError(f"{label} worker group has no worker_info_list")
    handles: dict[int, Any] = {}
    for entry in entries:
        rank = getattr(entry, "rank", None)
        handle = getattr(entry, "worker", None)
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError(f"{label} worker group contains an invalid rank")
        if handle is None:
            raise ValueError(f"{label} worker rank {rank} has no actor handle")
        if rank in handles:
            raise ValueError(f"{label} worker group contains duplicate rank {rank}")
        handles[rank] = handle
    if tuple(sorted(handles)) != tuple(range(len(handles))):
        raise ValueError(f"{label} worker ranks must be non-empty and contiguous")
    return handles


class RLixStageController:
    """Own one named coordinator actor and its driver-facing async API."""

    def __init__(
        self,
        *,
        pipeline_id: str,
        ray_namespace: str,
        env_worker_group: Any,
        rollout_worker_group: Any,
        operation_timeout_s: float = 300.0,
        worker_max_concurrency: int | None = None,
    ) -> None:
        """Validate ranked handles and create the exact callback actor."""
        if not isinstance(pipeline_id, str) or not pipeline_id:
            raise ValueError("pipeline_id must be a non-empty string")
        if not isinstance(ray_namespace, str) or not ray_namespace:
            raise ValueError("ray_namespace must be a non-empty string")
        if worker_max_concurrency is not None:
            if not isinstance(worker_max_concurrency, int) or isinstance(
                worker_max_concurrency, bool
            ):
                raise TypeError("worker_max_concurrency must be an integer")
            if worker_max_concurrency < 2:
                raise ValueError("elastic workers require max_concurrency >= 2")
        env_workers = _extract_ranked_worker_handles(
            env_worker_group, label="environment"
        )
        rollout_workers = _extract_ranked_worker_handles(
            rollout_worker_group, label="rollout"
        )
        if set(env_workers) != set(rollout_workers):
            raise ValueError("environment and rollout worker ranks must match")

        self.pipeline_id = pipeline_id
        self.ray_namespace = ray_namespace
        self.actor_name = f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}"
        remote_class = ray.remote(RLixResizeCoordinator)
        self.coordinator = remote_class.options(
            name=self.actor_name,
            namespace=ray_namespace,
            max_concurrency=1000,
            max_restarts=0,
            max_task_retries=0,
        ).remote(
            pipeline_id=pipeline_id,
            env_workers=env_workers,
            rollout_workers=rollout_workers,
            operation_timeout_s=operation_timeout_s,
        )
        self._closed = False

    async def configure_collection(
        self,
        context: ElasticCollectionContext,
        *,
        env_input_channel: Any,
        rollout_request_channel: Any,
        reward_channel: Any | None = None,
        actor_channel: Any | None = None,
    ) -> None:
        """Configure one lifecycle before requesting elastic generation."""
        await self.coordinator.configure_collection.remote(
            context,
            env_input_channel=env_input_channel,
            rollout_request_channel=rollout_request_channel,
            reward_channel=reward_channel,
            actor_channel=actor_channel,
        )

    async def get_status(self) -> CoordinatorStatus:
        """Return the coordinator's observational status."""
        return await self.coordinator.get_status.remote()

    async def get_rank_results(self, rank: int, *, wait: bool = False) -> Any:
        """Observe matching environment and rollout run results for one rank."""
        return await self.coordinator.get_rank_results.remote(rank, wait=wait)

    @asynccontextmanager
    async def policy_sync(
        self, *, expected_policy_version: int
    ) -> AsyncIterator[PolicySyncLease]:
        """Hold the coordinator's exact policy-sync lease around caller work."""
        lease = await self.coordinator.begin_policy_sync.remote(
            expected_policy_version=expected_policy_version
        )
        body_error: BaseException | None = None
        try:
            yield lease
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                await self.coordinator.end_policy_sync.remote(lease)
            except Exception as cleanup_error:
                if body_error is None:
                    raise
                body_error.add_note(
                    "Policy-sync lease cleanup also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )

    async def close(self) -> None:
        """Guard coordinator shutdown and then terminate its owner-scoped actor."""
        if self._closed:
            return
        await self.coordinator.close.remote()
        ray.kill(self.coordinator, no_restart=True)
        self._closed = True


__all__ = ["RLixStageController"]
