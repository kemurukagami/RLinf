"""CPU-only tests for Task 6 embodied entrypoint helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rlinf.scheduler.rlix.entrypoint import (
    launch_registered_rlix_workers,
    launch_standalone_worker_groups,
    preflight_rlix_placements,
)


class _Group:
    def __init__(
        self,
        label: str,
        events: list[str],
        *,
        fail_launch: bool = False,
    ) -> None:
        self.label = label
        self.events = events
        self.fail_launch = fail_launch
        self.launch_kwargs = None

    def launch(self, cluster, **kwargs):
        self.events.append(f"launch:{self.label}")
        self.launch_kwargs = kwargs
        if self.fail_launch:
            raise RuntimeError(f"{self.label} launch failed")
        return f"{self.label}-handle"

    def _close(self) -> None:
        self.events.append(f"close:{self.label}")


def _resolved():
    return SimpleNamespace(
        plan="immutable-plan",
        actor_strategy=object(),
        rollout_strategy=object(),
        env_strategy=object(),
    )


def test_preflight_resolves_and_validates_before_launch() -> None:
    events: list[str] = []
    resolved = _resolved()

    def resolver(component_placement, cluster):
        events.append("resolve")
        return resolved

    def validator(plan, cluster):
        events.append("validate")
        assert plan == "immutable-plan"

    result = preflight_rlix_placements(
        "components",
        "cluster",
        resolver=resolver,
        validator=validator,
    )
    actor = _Group("actor", events)
    rollout = _Group("rollout", events)
    env = _Group("env", events)

    async def bootstrapper(**kwargs):
        events.append("bootstrap")
        return "runtime"

    launch_registered_rlix_workers(
        cluster="cluster",
        actor_group=actor,
        rollout_group=rollout,
        env_group=env,
        actor_name="Actor",
        rollout_name="Rollout",
        env_name="Env",
        resolved=result,
        worker_max_concurrency=3,
        operation_timeout_s=12.0,
        enable_gpu_tracing=False,
        bootstrapper=bootstrapper,
    )

    assert events == [
        "resolve",
        "validate",
        "launch:actor",
        "launch:rollout",
        "launch:env",
        "bootstrap",
    ]


def test_preflight_failure_launches_no_groups() -> None:
    events: list[str] = []

    def validator(plan, cluster):
        events.append("validate")
        raise ValueError("invalid placement")

    with pytest.raises(ValueError, match="invalid placement"):
        preflight_rlix_placements(
            "components",
            "cluster",
            resolver=lambda *_: _resolved(),
            validator=validator,
        )

    assert events == ["validate"]


def test_enabled_launch_reuses_strategies_and_scopes_concurrency() -> None:
    events: list[str] = []
    resolved = _resolved()
    actor = _Group("actor", events)
    rollout = _Group("rollout", events)
    env = _Group("env", events)
    bootstrap_kwargs = {}

    async def bootstrapper(**kwargs):
        bootstrap_kwargs.update(kwargs)
        return "registered-runtime"

    launched = launch_registered_rlix_workers(
        cluster="cluster",
        actor_group=actor,
        rollout_group=rollout,
        env_group=env,
        actor_name="Actor",
        rollout_name="Rollout",
        env_name="Env",
        resolved=resolved,
        worker_max_concurrency=4,
        operation_timeout_s=20.0,
        enable_gpu_tracing=True,
        bootstrapper=bootstrapper,
    )

    assert actor.launch_kwargs == {
        "name": "Actor",
        "placement_strategy": resolved.actor_strategy,
    }
    assert rollout.launch_kwargs == {
        "name": "Rollout",
        "placement_strategy": resolved.rollout_strategy,
        "max_concurrency": 4,
    }
    assert env.launch_kwargs == {
        "name": "Env",
        "placement_strategy": resolved.env_strategy,
        "max_concurrency": 4,
    }
    assert bootstrap_kwargs == {
        "env_worker_group": "env-handle",
        "rollout_worker_group": "rollout-handle",
        "placement_plan": "immutable-plan",
        "worker_max_concurrency": 4,
        "operation_timeout_s": 20.0,
        "enable_gpu_tracing": True,
    }
    assert launched.runtime == "registered-runtime"


def test_disabled_launch_preserves_legacy_order_and_arguments() -> None:
    events: list[str] = []
    actor = _Group("actor", events)
    rollout = _Group("rollout", events)
    env = _Group("env", events)
    placements = [object(), object(), object()]

    def factory(group: _Group):
        def create():
            events.append(f"create:{group.label}")
            return group

        return create

    handles = launch_standalone_worker_groups(
        cluster="cluster",
        actor_group_factory=factory(actor),
        rollout_group_factory=factory(rollout),
        env_group_factory=factory(env),
        actor_name="Actor",
        rollout_name="Rollout",
        env_name="Env",
        actor_placement=placements[0],
        rollout_placement=placements[1],
        env_placement=placements[2],
    )

    assert events == [
        "create:actor",
        "launch:actor",
        "create:rollout",
        "launch:rollout",
        "create:env",
        "launch:env",
    ]
    assert handles == ("actor-handle", "rollout-handle", "env-handle")
    for group, name, placement in zip(
        (actor, rollout, env),
        ("Actor", "Rollout", "Env"),
        placements,
    ):
        assert group.launch_kwargs == {
            "name": name,
            "placement_strategy": placement,
        }


def test_enabled_launch_failure_closes_all_groups_in_reverse_order() -> None:
    events: list[str] = []
    actor = _Group("actor", events)
    rollout = _Group("rollout", events, fail_launch=True)
    env = _Group("env", events)

    async def bootstrapper(**kwargs):
        raise AssertionError("bootstrap must not run")

    with pytest.raises(RuntimeError, match="rollout launch failed"):
        launch_registered_rlix_workers(
            cluster="cluster",
            actor_group=actor,
            rollout_group=rollout,
            env_group=env,
            actor_name="Actor",
            rollout_name="Rollout",
            env_name="Env",
            resolved=_resolved(),
            worker_max_concurrency=2,
            operation_timeout_s=10.0,
            enable_gpu_tracing=False,
            bootstrapper=bootstrapper,
        )

    assert events == [
        "launch:actor",
        "launch:rollout",
        "close:env",
        "close:rollout",
        "close:actor",
    ]
