"""Shared acceptance-control actor for Task 8 two-driver orchestration."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray
from task8_acceptance_support import (
    AcceptanceEvent,
    AcceptanceEventProducer,
    AcceptanceProducerContext,
    normalize_manifest,
    validate_run_id,
)

_DEFAULT_GATES = frozenset(
    {
        "both_drivers_initialized",
        "allow_b_policy_sync",
        "b_policy_sync_completed",
        "allow_a_policy_sync",
        "a_policy_sync_completed",
        "allow_a_collection",
        "a_generation_requested",
        "a_generation_granted",
        "a_target_bootstrap_dispatched",
        "a_target_chunk_started",
        "a_first_rank_completed",
        "a_target_rank_completed",
        "allow_b_collection",
        "b_generation_requested",
        "b_generation_granted",
        "allow_b_demand",
        "transfer_to_b_observed",
        "b_useful_work_observed",
        "both_batches_sealed",
        "allow_training",
        "allow_a_training",
        "allow_b_training",
        "a_batch_sealed",
        "b_batch_sealed",
        "a_training_started",
        "b_training_started",
        "a_training_completed",
        "b_training_completed",
        "both_training_completed",
        "allow_b_release",
        "b_release_observed",
        "a_resume_observed",
        "b_resume_observed",
    }
)


@dataclass(frozen=True, slots=True)
class AcceptanceControlConfig:
    """Immutable run-owned control settings."""

    run_id: str
    event_log_path: str
    target_bundle: tuple[int, ...]
    target_rank: int = 0

    def validate(self) -> None:
        """Validate the run-owned control-plane contract."""
        validate_run_id(self.run_id)
        if self.target_rank < 0:
            raise ValueError("target_rank must be non-negative")
        if not self.target_bundle or any(gpu_id < 0 for gpu_id in self.target_bundle):
            raise ValueError("target_bundle must contain non-negative GPU IDs")


class AcceptanceControlCore:
    """Single-writer event sequencer plus deterministic orchestration gates."""

    def __init__(
        self,
        config: AcceptanceControlConfig,
        *,
        clock_ns: Callable[[], int] = time.time_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        config.validate()
        self._config = config
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._sink = _PersistentAcceptanceSink(
            run_id=config.run_id,
            event_log_path=Path(config.event_log_path),
            clock_ns=clock_ns,
        )
        self._gates = dict.fromkeys(_DEFAULT_GATES, False)
        self._driver_initialized: set[str] = set()
        self._policy_sync_completed: set[str] = set()
        self._generation_requested: set[str] = set()
        self._generation_granted: set[str] = set()
        self._batches_sealed: set[str] = set()
        self._training_completed: set[str] = set()
        self._failure: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def manifest(self) -> dict[str, Any]:
        """Return JSON-safe run control metadata for driver readiness."""
        with self._lock:
            return normalize_manifest(
                {
                    "run_id": self._config.run_id,
                    "event_log_path": self._config.event_log_path,
                    "target_bundle": self._config.target_bundle,
                    "target_rank": self._config.target_rank,
                    "gates": self._gates,
                }
            )

    def gate_status(self) -> dict[str, bool]:
        """Return a JSON-safe snapshot of all gates."""
        with self._lock:
            return dict(self._gates)

    def release_gate(self, gate: str) -> dict[str, bool]:
        """Release one explicit orchestration gate."""
        with self._lock:
            self._require_gate(gate)
            self._raise_if_failed()
            self._gates[gate] = True
            return dict(self._gates)

    def fail(self, *, role: str, error_type: str, error: str) -> None:
        """Publish a terminal driver/control failure to every waiter."""
        if role not in {"a", "b", "core", "orchestrator"}:
            raise ValueError("failure role must be a Task 8 role")
        if not error_type or not error:
            raise ValueError("failure requires error_type and error")
        with self._lock:
            self._failure = {
                "role": role,
                "error_type": error_type,
                "error": error,
            }

    def wait_for_gate(
        self,
        gate: str,
        *,
        timeout_s: float | None,
        poll_interval_s: float = 0.05,
    ) -> dict[str, bool]:
        """Block until one gate is released or fail closed."""
        self._require_gate(gate)
        if (timeout_s is not None and timeout_s <= 0) or poll_interval_s <= 0:
            raise ValueError("gate timeout and poll interval must be positive")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._gates[gate]:
                    return dict(self._gates)
                self._raise_if_failed()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for acceptance gate {gate}")
            self._sleep(poll_interval_s)

    def record_event(self, event: AcceptanceEvent) -> AcceptanceEvent:
        """Record one event, then update observation-derived gates."""
        with self._lock:
            self._raise_if_failed()
            try:
                stamped = self._sink.record(event)
            except Exception as exc:
                raise ValueError(
                    "acceptance event rejected: "
                    f"{_summarize_event_for_error(event)} reason={type(exc).__name__}: {exc}"
                ) from exc
            self._observe(stamped)
            return stamped

    def events(self) -> tuple[AcceptanceEvent, ...]:
        """Return stamped events in total sink order."""
        with self._lock:
            return self._sink.events

    def _observe(self, event: AcceptanceEvent) -> None:
        if event.event == "initialized" and event.driver_role in {"a", "b"}:
            self._driver_initialized.add(event.driver_role)
            if self._driver_initialized == {"a", "b"}:
                self._gates["both_drivers_initialized"] = True
        if event.event == "policy_synchronized" and event.driver_role in {"a", "b"}:
            self._policy_sync_completed.add(event.driver_role)
            self._gates[f"{event.driver_role}_policy_sync_completed"] = True
        if event.event == "generation_requested" and event.driver_role in {"a", "b"}:
            self._generation_requested.add(event.driver_role)
            self._gates[f"{event.driver_role}_generation_requested"] = True
        if event.event == "generation_granted" and event.driver_role in {"a", "b"}:
            self._generation_granted.add(event.driver_role)
            self._gates[f"{event.driver_role}_generation_granted"] = True
            if (
                event.driver_role == "b"
                and event.gpu_ids
                and set(self._config.target_bundle).issubset(set(event.gpu_ids))
            ):
                self._gates["transfer_to_b_observed"] = True
        if (
            event.event == "batch_sealed"
            and event.component == "runner"
            and event.driver_role in {"a", "b"}
        ):
            self._batches_sealed.add(event.driver_role)
            self._gates[f"{event.driver_role}_batch_sealed"] = True
            if self._batches_sealed == {"a", "b"}:
                self._gates["both_batches_sealed"] = True
        if event.event == "training_started" and event.driver_role in {"a", "b"}:
            self._gates[f"{event.driver_role}_training_started"] = True
        if event.event == "training_completed" and event.driver_role in {"a", "b"}:
            self._training_completed.add(event.driver_role)
            self._gates[f"{event.driver_role}_training_completed"] = True
            if self._training_completed == {"a", "b"}:
                self._gates["both_training_completed"] = True
        if (
            event.driver_role == "a"
            and event.event == "bootstrap_dispatched"
            and event.dp_rank == self._config.target_rank
            and event.gpu_ids == self._config.target_bundle
        ):
            self._gates["a_target_bootstrap_dispatched"] = True
        if (
            event.driver_role == "a"
            and event.event == "chunk_started"
            and event.dp_rank == self._config.target_rank
            and event.gpu_ids == self._config.target_bundle
        ):
            self._gates["a_target_chunk_started"] = True
        if (
            event.driver_role == "a"
            and event.component == "rollout"
            and event.event == "rank_completed"
        ):
            self._gates["a_first_rank_completed"] = True
        if (
            event.driver_role == "a"
            and event.event == "rank_completed"
            and event.dp_rank == self._config.target_rank
            and event.gpu_ids == self._config.target_bundle
        ):
            self._gates["a_target_rank_completed"] = True
        if (
            event.driver_role == "b"
            and event.event in {"policy_request_completed", "chunk_committed"}
            and event.gpu_ids == self._config.target_bundle
        ):
            self._gates["b_useful_work_observed"] = True
        if (
            event.driver_role == "b"
            and event.event == "allocation_committed"
            and event.gpu_ids == self._config.target_bundle
        ):
            self._gates["transfer_to_b_observed"] = True
        if (
            event.driver_role == "b"
            and event.event == "release_committed"
            and event.gpu_ids == self._config.target_bundle
        ):
            self._gates["b_release_observed"] = True
        if (
            event.driver_role == "a"
            and event.event in {"restore_validated", "resumed_bootstrap_dispatched"}
            and event.dp_rank == self._config.target_rank
            and event.gpu_ids == self._config.target_bundle
        ):
            self._gates["a_resume_observed"] = True
        if (
            event.driver_role == "b"
            and event.event in {"restore_validated", "resumed_bootstrap_dispatched"}
            and event.dp_rank == self._config.target_rank
            and event.gpu_ids == self._config.target_bundle
        ):
            self._gates["b_resume_observed"] = True

    def _require_gate(self, gate: str) -> None:
        if gate not in self._gates:
            raise ValueError(f"unknown acceptance gate {gate!r}")

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "Task 8 acceptance control failed: "
                f"{self._failure['role']} {self._failure['error_type']}: "
                f"{self._failure['error']}"
            )


def _summarize_event_for_error(event: Any) -> str:
    """Return a compact event summary for Ray exception text."""
    details = getattr(event, "details", None)
    detail_keys: tuple[str, ...] = ()
    if isinstance(details, dict):
        detail_keys = tuple(str(key) for key in sorted(details))
    transition = getattr(event, "transition_identity", None)
    return (
        f"event={getattr(event, 'event', None)!r} "
        f"role={getattr(event, 'driver_role', None)!r} "
        f"component={getattr(event, 'component', None)!r} "
        f"pipeline_id={getattr(event, 'pipeline_id', None)!r} "
        f"dp_rank={getattr(event, 'dp_rank', None)!r} "
        f"lifecycle_generation={getattr(event, 'lifecycle_generation', None)!r} "
        f"policy_version={getattr(event, 'policy_version', None)!r} "
        f"transition_identity={transition!r} "
        f"gpu_ids={getattr(event, 'gpu_ids', None)!r} "
        f"producer_sequence={getattr(event, 'producer_sequence', None)!r} "
        f"detail_keys={detail_keys!r}"
    )


class _PersistentAcceptanceSink:
    """AcceptanceEventSink-compatible append-only JSONL sink."""

    def __init__(
        self,
        *,
        run_id: str,
        event_log_path: Path,
        clock_ns: Callable[[], int],
    ) -> None:
        from task8_acceptance_support import AcceptanceEventSink

        self._sink = AcceptanceEventSink(run_id=run_id, clock_ns=clock_ns)
        self._path = event_log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("x", encoding="utf-8"):
            pass

    @property
    def events(self) -> tuple[AcceptanceEvent, ...]:
        """Return an immutable snapshot of accepted events."""
        return self._sink.events

    def record(self, event: AcceptanceEvent) -> AcceptanceEvent:
        """Stamp, validate, and append one event atomically from this process."""
        stamped = self._sink.record(event)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    normalize_manifest(asdict(stamped)),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            stream.write("\n")
        return stamped


@ray.remote(num_cpus=0, max_concurrency=16)
class AcceptanceControlActor(AcceptanceControlCore):
    """Named Ray actor wrapper around the Task 8 control core."""


def acceptance_control_actor_name(*, run_id: str) -> str:
    """Derive the named Ray actor identity for one run."""
    validate_run_id(run_id)
    return f"task8_acceptance_control_{run_id}"


def create_acceptance_control_actor(
    config: AcceptanceControlConfig,
    *,
    namespace: str | None = None,
) -> Any:
    """Create one named, detached acceptance-control actor for a run."""
    config.validate()
    return AcceptanceControlActor.options(
        name=acceptance_control_actor_name(run_id=config.run_id),
        namespace=namespace,
        lifetime="detached",
        get_if_exists=False,
    ).remote(config)


class _RayEventSubmitter:
    def __init__(self, control_actor: Any) -> None:
        self._control_actor = control_actor

    def __call__(self, event: AcceptanceEvent) -> AcceptanceEvent:
        return ray.get(self._control_actor.record_event.remote(event))


class AcceptanceControlObserverProxy:
    """Serializable worker observer that submits events to the shared actor."""

    def __init__(
        self,
        *,
        run_id: str,
        driver_role: str,
        component: str,
        dp_rank: int | None,
        context_provider: Callable[[], AcceptanceProducerContext],
        control_actor: Any,
    ) -> None:
        self._producer = AcceptanceEventProducer(
            run_id=run_id,
            driver_role=driver_role,
            component=component,
            dp_rank=dp_rank,
            context_provider=context_provider,
            sink_record=_RayEventSubmitter(control_actor),
        )

    def __call__(self, observation: Any) -> AcceptanceEvent:
        """Submit one worker observation and fail if the actor rejects it."""
        return self._producer.record(observation)


def event_to_json(event: AcceptanceEvent) -> dict[str, Any]:
    """Return a deterministic JSON-safe event payload."""
    event.validate()
    return normalize_manifest(asdict(event))
