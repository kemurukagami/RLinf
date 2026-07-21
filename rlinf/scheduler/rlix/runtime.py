"""Owner-scoped registration bootstrap for an inactive RLix pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from rlix_core.control_plane import ControlPlane
from rlix_core.protocol.types import get_pipeline_namespace

from .controller import RLixStageController
from .placement import RLixPlacementPlan


def _add_cleanup_note(primary: BaseException, operation: str, error: Exception) -> None:
    primary.add_note(
        f"RLix {operation} cleanup also failed: {type(error).__name__}: {error}"
    )


def close_worker_groups_after_bootstrap_failure(
    worker_groups: list[Any], primary_error: BaseException
) -> None:
    """Best-effort close partially launched groups in reverse construction order."""
    for group in reversed(worker_groups):
        try:
            group._close()
        except Exception as cleanup_error:
            _add_cleanup_note(primary_error, "worker-group", cleanup_error)


@dataclass(slots=True)
class RegisteredRLixPipeline:
    """Own an admitted but inactive pipeline until the Task 7 runner adopts it."""

    control_plane: Any
    scheduler: Any
    controller: Any
    pipeline_id: str
    ray_namespace: str
    placement_plan: RLixPlacementPlan
    _closed: bool = field(default=False, init=False, repr=False)

    async def close(self) -> None:
        """Unregister before closing the coordinator; repeated calls are harmless."""
        if self._closed:
            return
        primary_error: Exception | None = None
        try:
            self.control_plane.unregister_pipeline(pipeline_id=self.pipeline_id)
        except Exception as exc:
            primary_error = exc
        try:
            await self.controller.close()
        except Exception as exc:
            if primary_error is None:
                primary_error = exc
            else:
                _add_cleanup_note(primary_error, "coordinator", exc)
        if primary_error is not None:
            raise primary_error
        self._closed = True


async def bootstrap_registered_rlix_pipeline(
    *,
    env_worker_group: Any,
    rollout_worker_group: Any,
    placement_plan: RLixPlacementPlan,
    worker_max_concurrency: int,
    operation_timeout_s: float,
    enable_gpu_tracing: bool = False,
    control_plane_factory: Callable[..., Any] = ControlPlane,
    controller_factory: Callable[..., Any] = RLixStageController,
) -> RegisteredRLixPipeline:
    """Create, register, and admit one pipeline without requesting allocation."""
    if not isinstance(enable_gpu_tracing, bool):
        raise TypeError("enable_gpu_tracing must be a boolean")
    control_plane = control_plane_factory(
        env_vars={"RLIX_ENABLE_GPU_TRACING": "1"} if enable_gpu_tracing else {}
    )
    pipeline_id = control_plane.allocate_pipeline_id(pipeline_type="rlinf")
    ray_namespace = get_pipeline_namespace(pipeline_id)
    controller = controller_factory(
        pipeline_id=pipeline_id,
        ray_namespace=ray_namespace,
        env_worker_group=env_worker_group,
        rollout_worker_group=rollout_worker_group,
        operation_timeout_s=operation_timeout_s,
        worker_max_concurrency=worker_max_concurrency,
    )
    registration_attempted = False
    try:
        registration_attempted = True
        control_plane.register_pipeline(
            pipeline_id=pipeline_id,
            ray_namespace=ray_namespace,
            **placement_plan.registration_payload(),
        )
        admission = control_plane.admit_pipeline(pipeline_id=pipeline_id)
    except BaseException as primary_error:
        if registration_attempted:
            try:
                control_plane.unregister_pipeline(pipeline_id=pipeline_id)
            except Exception as cleanup_error:
                _add_cleanup_note(primary_error, "registration", cleanup_error)
        try:
            await controller.close()
        except Exception as cleanup_error:
            _add_cleanup_note(primary_error, "coordinator", cleanup_error)
        raise

    return RegisteredRLixPipeline(
        control_plane=control_plane,
        scheduler=admission.scheduler,
        controller=controller,
        pipeline_id=pipeline_id,
        ray_namespace=ray_namespace,
        placement_plan=placement_plan,
    )


__all__ = [
    "RegisteredRLixPipeline",
    "bootstrap_registered_rlix_pipeline",
    "close_worker_groups_after_bootstrap_failure",
]
