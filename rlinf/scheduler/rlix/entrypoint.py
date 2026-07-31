"""CPU-testable Task 6 helpers for the synchronous embodied entrypoint."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .placement import ResolvedRLixPlacements, resolve_rlix_placements
from .validation import validate_elastic_vla_placement


@dataclass(frozen=True, slots=True)
class LaunchedRLixWorkers:
    """Lightweight worker groups and their inactive registered runtime."""

    actor: Any
    rollout: Any
    env: Any
    runtime: Any


def preflight_rlix_placements(
    component_placement: Any,
    cluster: Any,
    *,
    resolver: Callable[[Any, Any], ResolvedRLixPlacements] = resolve_rlix_placements,
    validator: Callable[[Any, Any], None] = validate_elastic_vla_placement,
) -> ResolvedRLixPlacements:
    """Resolve and validate the entire RLix topology before worker construction."""
    resolved = resolver(component_placement, cluster)
    validator(resolved.plan, cluster)
    return resolved


def launch_standalone_worker_groups(
    *,
    cluster: Any,
    actor_group_factory: Callable[[], Any],
    rollout_group_factory: Callable[[], Any],
    env_group_factory: Callable[[], Any],
    actor_name: str,
    rollout_name: str,
    env_name: str,
    actor_placement: Any,
    rollout_placement: Any,
    env_placement: Any,
) -> tuple[Any, Any, Any]:
    """Preserve the legacy launch order and arguments when RLix is disabled."""
    actor = actor_group_factory().launch(
        cluster, name=actor_name, placement_strategy=actor_placement
    )
    rollout = rollout_group_factory().launch(
        cluster, name=rollout_name, placement_strategy=rollout_placement
    )
    env = env_group_factory().launch(
        cluster, name=env_name, placement_strategy=env_placement
    )
    return actor, rollout, env


def launch_registered_rlix_workers(
    *,
    cluster: Any,
    actor_group: Any,
    rollout_group: Any,
    env_group: Any,
    actor_name: str,
    rollout_name: str,
    env_name: str,
    resolved: ResolvedRLixPlacements,
    worker_max_concurrency: int,
    operation_timeout_s: float,
    enable_gpu_tracing: bool,
    bootstrapper: Callable[..., Awaitable[Any]],
    completed_bundle_handoff: str = "retain_overlap",
) -> LaunchedRLixWorkers:
    """Launch replayed placements and atomically bootstrap their registration."""
    groups = [actor_group, rollout_group, env_group]
    try:
        actor = actor_group.launch(
            cluster,
            name=actor_name,
            placement_strategy=resolved.actor_strategy,
        )
        rollout = rollout_group.launch(
            cluster,
            name=rollout_name,
            placement_strategy=resolved.rollout_strategy,
            max_concurrency=worker_max_concurrency,
        )
        env = env_group.launch(
            cluster,
            name=env_name,
            placement_strategy=resolved.env_strategy,
            max_concurrency=worker_max_concurrency,
        )
        bootstrap_kwargs = {
            "env_worker_group": env,
            "rollout_worker_group": rollout,
            "placement_plan": resolved.plan,
            "worker_max_concurrency": worker_max_concurrency,
            "operation_timeout_s": operation_timeout_s,
            "enable_gpu_tracing": enable_gpu_tracing,
        }
        # Preserve the exact legacy bootstrap call for default-mode callers,
        # including injected bootstrappers with the old keyword signature.
        if completed_bundle_handoff != "retain_overlap":
            bootstrap_kwargs["completed_bundle_handoff"] = completed_bundle_handoff
        runtime = asyncio.run(bootstrapper(**bootstrap_kwargs))
    except BaseException as primary_error:
        from .runtime import close_worker_groups_after_bootstrap_failure

        close_worker_groups_after_bootstrap_failure(groups, primary_error)
        raise
    return LaunchedRLixWorkers(
        actor=actor,
        rollout=rollout,
        env=env,
        runtime=runtime,
    )


def run_registered_rlix_runner(*, runner: Any, runtime: Any) -> None:
    """Run enabled initialization/training and preserve errors during close."""
    primary_error: BaseException | None = None
    try:
        runner.init_workers()
        runner.run()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            runtime.close_sync()
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                "RLix runtime close also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )


__all__ = [
    "LaunchedRLixWorkers",
    "launch_registered_rlix_workers",
    "launch_standalone_worker_groups",
    "preflight_rlix_placements",
    "run_registered_rlix_runner",
]
