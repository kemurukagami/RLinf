"""Tests for Task 6 registered-pipeline bootstrap ownership."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from rlinf.scheduler.rlix.runtime import (
    bootstrap_registered_rlix_pipeline,
    close_worker_groups_after_bootstrap_failure,
)


class _Plan:
    def registration_payload(self):
        return {
            "cluster_tp_configs": {
                "initialization": 1,
                "actor_train": 1,
                "actor_infer": 1,
                "policy_sync": 1,
                "evaluation": 1,
            },
            "cluster_device_mappings": {
                "initialization": [0, 1, 2],
                "actor_train": [2],
                "actor_infer": [0, 1],
                "policy_sync": [0, 2],
                "evaluation": [0, 1],
            },
            "cluster_allocation_policies": {
                "initialization": "fixed",
                "actor_train": "fixed",
                "actor_infer": "elastic",
                "policy_sync": "fixed",
                "evaluation": "fixed",
            },
            "cluster_dp_device_mappings": {"actor_infer": {0: [0, 1]}},
        }


class _ControlPlane:
    def __init__(
        self,
        events: list[str],
        *,
        fail_register: bool = False,
        fail_admit: bool = False,
        fail_unregister: bool = False,
        env_vars: dict[str, str] | None = None,
    ) -> None:
        self.events = events
        self.fail_register = fail_register
        self.fail_admit = fail_admit
        self.fail_unregister = fail_unregister
        self.registration = None
        self.env_vars = env_vars

    def allocate_pipeline_id(self, pipeline_type: str) -> str:
        self.events.append(f"allocate:{pipeline_type}")
        return "rlinf_123456789abc"

    def register_pipeline(self, **kwargs) -> None:
        self.events.append("register")
        self.registration = kwargs
        if self.fail_register:
            raise RuntimeError("register failed")

    def admit_pipeline(self, *, pipeline_id: str):
        self.events.append("admit")
        if self.fail_admit:
            raise RuntimeError("admit failed")
        return SimpleNamespace(scheduler="scheduler-handle")

    def unregister_pipeline(self, *, pipeline_id: str) -> None:
        self.events.append("unregister")
        if self.fail_unregister:
            raise RuntimeError("unregister failed")


class _Controller:
    def __init__(
        self, events: list[str], *, fail_close: bool = False, **kwargs
    ) -> None:
        self.events = events
        self.fail_close = fail_close
        self.kwargs = kwargs
        events.append("controller")

    async def close(self) -> None:
        self.events.append("close_controller")
        if self.fail_close:
            raise RuntimeError("controller close failed")


def _bootstrap(
    control_plane: _ControlPlane,
    *,
    controller_fail_close: bool = False,
    enable_gpu_tracing: bool = False,
):
    controller_holder = {}

    def controller_factory(**kwargs):
        controller = _Controller(
            control_plane.events,
            fail_close=controller_fail_close,
            **kwargs,
        )
        controller_holder["value"] = controller
        return controller

    runtime = asyncio.run(
        bootstrap_registered_rlix_pipeline(
            env_worker_group="env-group",
            rollout_worker_group="rollout-group",
            placement_plan=_Plan(),
            worker_max_concurrency=3,
            operation_timeout_s=15.0,
            enable_gpu_tracing=enable_gpu_tracing,
            control_plane_factory=lambda **kwargs: (
                setattr(control_plane, "env_vars", kwargs["env_vars"]) or control_plane
            ),
            controller_factory=controller_factory,
        )
    )
    return runtime, controller_holder.get("value")


def test_bootstrap_uses_allocated_identity_and_exact_order() -> None:
    events: list[str] = []
    control_plane = _ControlPlane(events)

    runtime, controller = _bootstrap(control_plane, enable_gpu_tracing=True)

    assert events == ["allocate:rlinf", "controller", "register", "admit"]
    assert runtime.pipeline_id == "rlinf_123456789abc"
    assert runtime.ray_namespace == "rlix_core_pipeline_rlinf_123456789abc"
    assert runtime.scheduler == "scheduler-handle"
    assert control_plane.env_vars == {"RLIX_ENABLE_GPU_TRACING": "1"}
    assert controller.kwargs == {
        "pipeline_id": runtime.pipeline_id,
        "ray_namespace": runtime.ray_namespace,
        "env_worker_group": "env-group",
        "rollout_worker_group": "rollout-group",
        "operation_timeout_s": 15.0,
        "worker_max_concurrency": 3,
    }
    assert set(control_plane.registration["cluster_device_mappings"]) == {
        "initialization",
        "actor_train",
        "actor_infer",
        "policy_sync",
        "evaluation",
    }
    assert control_plane.registration["cluster_dp_device_mappings"] == {
        "actor_infer": {0: [0, 1]}
    }


@pytest.mark.parametrize(
    ("failure", "message", "expected_events"),
    [
        (
            {"fail_register": True},
            "register failed",
            [
                "allocate:rlinf",
                "controller",
                "register",
                "unregister",
                "close_controller",
            ],
        ),
        (
            {"fail_admit": True},
            "admit failed",
            [
                "allocate:rlinf",
                "controller",
                "register",
                "admit",
                "unregister",
                "close_controller",
            ],
        ),
    ],
)
def test_bootstrap_failure_rolls_back_registration_before_controller(
    failure: dict[str, bool],
    message: str,
    expected_events: list[str],
) -> None:
    events: list[str] = []
    control_plane = _ControlPlane(events, **failure)

    with pytest.raises(RuntimeError, match=message):
        _bootstrap(control_plane)

    assert events == expected_events


def test_bootstrap_cleanup_errors_do_not_replace_primary_error() -> None:
    events: list[str] = []
    control_plane = _ControlPlane(
        events,
        fail_register=True,
        fail_unregister=True,
    )

    with pytest.raises(RuntimeError, match="register failed") as exc_info:
        _bootstrap(control_plane, controller_fail_close=True)

    notes = getattr(exc_info.value, "__notes__", [])
    assert any("registration cleanup" in note for note in notes)
    assert any("coordinator cleanup" in note for note in notes)


def test_runtime_close_is_ordered_and_idempotent() -> None:
    events: list[str] = []
    control_plane = _ControlPlane(events)
    runtime, _ = _bootstrap(control_plane)
    events.clear()

    asyncio.run(runtime.close())
    asyncio.run(runtime.close())

    assert events == ["unregister", "close_controller"]


def test_runtime_close_can_retry_after_cleanup_failure() -> None:
    events: list[str] = []
    control_plane = _ControlPlane(events)
    runtime, _ = _bootstrap(control_plane)
    events.clear()
    control_plane.fail_unregister = True

    with pytest.raises(RuntimeError, match="unregister failed"):
        asyncio.run(runtime.close())
    control_plane.fail_unregister = False
    asyncio.run(runtime.close())

    assert events == [
        "unregister",
        "close_controller",
        "unregister",
        "close_controller",
    ]


def test_partial_worker_groups_close_in_reverse_without_replacing_error() -> None:
    events: list[str] = []

    class _Group:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def _close(self) -> None:
            events.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} close failed")

    primary = ValueError("bootstrap failed")
    close_worker_groups_after_bootstrap_failure(
        [_Group("actor"), _Group("rollout", fail=True), _Group("environment")],
        primary,
    )

    assert events == ["environment", "rollout", "actor"]
    assert str(primary) == "bootstrap failed"
    assert "rollout close failed" in primary.__notes__[0]
