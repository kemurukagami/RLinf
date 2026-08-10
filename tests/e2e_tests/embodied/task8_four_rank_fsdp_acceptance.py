"""Two-pipeline proof for four-rank FSDP training on one four-GPU node."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray
from omegaconf import OmegaConf
from rlix_core.protocol.types import ACTOR_TRAIN_CLUSTER_NAME, RLIX_NAMESPACE
from task8_acceptance_artifacts import atomic_write_json
from task8_acceptance_control import (
    AcceptanceControlConfig,
    acceptance_control_actor_name,
    create_acceptance_control_actor,
)
from task8_gpu_profiler import Task8GPUProfiler
from task8_two_pipeline_acceptance import (
    DriverProcessResult,
    _actor_id,
    _parse_control_bundles,
    prepare_preliminary_driver_layout,
    run_driver_pair,
    validate_generation_iteration_counts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("disaggregated",), required=True)
    parser.add_argument("--bundles", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="optional wall-clock limit; omitted means no timeout",
    )
    parser.add_argument(
        "--phase-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--acceptance-instrumentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record and validate detailed worker acceptance evidence",
    )
    parser.add_argument(
        "--residency-validation-mode",
        choices=("deep", "receipt", "off"),
        default="deep",
    )
    parser.add_argument(
        "--snapshot-validation-mode",
        choices=("deep", "receipt"),
        default="deep",
    )
    parser.add_argument(
        "--stream-driver-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gpu-event-poll-interval-s",
        "--gpu-profile-interval-s",
        dest="gpu_event_poll_interval_s",
        type=float,
        default=0.5,
        help=(
            "poll interval for acceptance events; nvidia-smi is queried only "
            "when a coalesced stage transition is observed"
        ),
    )
    parser.add_argument(
        "--gpu-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _validate_four_rank_ready(ready: Mapping[str, Mapping[str, Any]]) -> None:
    """Enforce the all-GPU fixed-training acceptance contract before gates."""
    expected = [0, 1, 2, 3]
    for role, payload in ready.items():
        mapping = payload.get("candidate_mapping", {})
        if mapping.get(ACTOR_TRAIN_CLUSTER_NAME) != expected:
            raise ValueError(
                f"driver {role} actor_train mapping must be {expected}; "
                f"got {mapping.get(ACTOR_TRAIN_CLUSTER_NAME)!r}"
            )
        bundles = payload.get("actor_infer_bundles")
        if bundles != [[0, 2], [1, 3]]:
            raise ValueError(
                f"driver {role} generation bundles changed unexpectedly: {bundles!r}"
            )


def _drive_four_rank_fsdp_gates(
    control_actor: Any,
    *,
    timeout_s: float | None,
    acceptance_instrumentation: bool = True,
) -> None:
    """Start deterministic initialization, then observe queue-driven execution."""
    ray.get(
        control_actor.wait_for_gate.remote(
            "both_drivers_initialized", timeout_s=timeout_s
        )
    )
    ray.get(control_actor.release_gate.remote("allow_b_policy_sync"))
    ray.get(
        control_actor.wait_for_gate.remote(
            "b_policy_sync_completed", timeout_s=timeout_s
        )
    )
    ray.get(control_actor.release_gate.remote("allow_a_policy_sync"))
    ray.get(
        control_actor.wait_for_gate.remote(
            "a_policy_sync_completed", timeout_s=timeout_s
        )
    )
    ray.get(control_actor.release_gate.remote("allow_a_collection"))
    ray.get(
        control_actor.wait_for_gate.remote("a_generation_granted", timeout_s=timeout_s)
    )
    # Queue B while A still owns both bundles. A completed-rank release can
    # therefore transfer directly to B without waiting for this orchestrator.
    ray.get(control_actor.release_gate.remote("allow_b_collection"))
    ray.get(
        control_actor.wait_for_gate.remote(
            "b_generation_requested", timeout_s=timeout_s
        )
    )
    if not acceptance_instrumentation:
        return
    ray.get(
        control_actor.wait_for_gate.remote(
            "a_first_rank_completed", timeout_s=timeout_s
        )
    )
    ray.get(control_actor.wait_for_gate.remote("a_batch_sealed", timeout_s=timeout_s))
    ray.get(
        control_actor.wait_for_gate.remote("a_training_started", timeout_s=timeout_s)
    )
    ray.get(
        control_actor.wait_for_gate.remote("a_training_completed", timeout_s=timeout_s)
    )
    ray.get(control_actor.wait_for_gate.remote("b_batch_sealed", timeout_s=timeout_s))
    ray.get(
        control_actor.wait_for_gate.remote(
            "both_training_completed", timeout_s=timeout_s
        )
    )


def _validate_four_rank_lifecycle(
    events: list[Any] | tuple[Any, ...], *, iterations: int
) -> None:
    """Require two-rank B preemption/resume and four-rank A training evidence."""
    required_worker_events = {
        "environment": {
            "drain_requested",
            "snapshot_completed",
            "environment_offload_verified",
            "environment_onload_verified",
            "restore_validated",
        },
        "rollout": {
            "drain_requested",
            "rollout_offload_verified",
            "rollout_onload_verified",
        },
    }
    for component, required in required_worker_events.items():
        for rank in (0, 1):
            observed = {
                event.event
                for event in events
                if event.driver_role == "b"
                and event.component == component
                and event.dp_rank == rank
            }
            missing = sorted(required - observed)
            if missing:
                raise ValueError(
                    f"B {component} rank {rank} lacks preemption evidence {missing}"
                )
    trained_actor_ranks = {
        event.dp_rank
        for event in events
        if event.driver_role == "a"
        and event.component == "actor"
        and event.event == "training_completed"
    }
    if trained_actor_ranks != {0, 1, 2, 3}:
        raise ValueError(
            "A training did not complete on every FSDP rank: "
            f"observed={sorted(trained_actor_ranks)}"
        )
    fixed_sync_acquisitions = [
        event
        for event in events
        if event.event == "stage_acquired"
        and event.details.get("stage") == "policy_sync"
    ]
    if fixed_sync_acquisitions:
        raise ValueError(
            "asynchronous CPU policy prefetch acquired the fixed sync stage"
        )
    expected_promotions = set(range(1, iterations + 1))
    expected_consumed_versions = set(range(1, iterations))
    for role in ("a", "b"):
        promoted = {
            event.details.get("policy_version")
            for event in events
            if event.driver_role == role
            and event.component == "actor"
            and event.event == "policy_cache_promoted"
            and event.details.get("promoted") is True
        }
        if promoted != expected_promotions:
            raise ValueError(
                f"driver {role} policy cache promotions mismatch: "
                f"expected={sorted(expected_promotions)}, observed={sorted(promoted)}"
            )
        consumed = {
            event.details.get("policy_version")
            for event in events
            if event.driver_role == role
            and event.component == "rollout"
            and event.event == "async_policy_update_committed"
        }
        if not expected_consumed_versions.issubset(consumed):
            raise ValueError(
                f"driver {role} lacks asynchronous rollout versions: "
                f"missing={sorted(expected_consumed_versions - consumed)}"
            )


def _summary(
    results: Mapping[str, DriverProcessResult],
    *,
    gpu_profile: Mapping[str, Any],
    acceptance_instrumentation: bool,
    residency_validation_mode: str,
    snapshot_validation_mode: str,
) -> dict[str, Any]:
    return {
        "status": "passed",
        "scope": "four_rank_fsdp_generation_proof",
        "task8_accepted": False,
        "detailed_acceptance_evidence": acceptance_instrumentation,
        "residency_validation_mode": residency_validation_mode,
        "snapshot_validation_mode": snapshot_validation_mode,
        "policy_sync_mode": "async_cpu_prefetch",
        "gpu_profile": dict(gpu_profile),
        "drivers": {
            role: {
                "pid": result.pid,
                "pipeline_id": result.ready["pipeline_id"],
                "result": dict(result.result),
            }
            for role, result in results.items()
        },
    }


def main() -> None:
    args = _parse_args()
    if args.timeout_s is not None and args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if args.gpu_event_poll_interval_s <= 0:
        raise ValueError("--gpu-event-poll-interval-s must be positive")
    cfg = OmegaConf.load(args.config)
    if tuple(cfg.smoke.actor_gpus) != (0, 1, 2, 3):
        raise ValueError("config must place four FSDP actor ranks on GPUs 0,1,2,3")
    if cfg.smoke.completed_bundle_handoff != "release_before_training":
        raise ValueError("config must release completed bundles before training")
    if cfg.smoke.policy_sync_mode != "async_cpu_prefetch":
        raise ValueError("config must enable asynchronous CPU policy prefetch")
    local_actor_batch = (
        int(cfg.smoke.total_num_envs)
        * int(cfg.smoke.rollout_epoch)
        // len(tuple(cfg.smoke.actor_gpus))
    )
    if local_actor_batch % int(cfg.smoke.group_size) != 0:
        raise ValueError(
            "config must route complete GRPO groups to every actor rank: "
            f"local_actor_batch={local_actor_batch}, "
            f"group_size={int(cfg.smoke.group_size)}"
        )

    bundles = _parse_control_bundles(args.bundles, mode=args.mode)
    layout = prepare_preliminary_driver_layout(
        args.output_dir,
        run_id=args.run_id,
        scope="generation_proof_only",
    )
    gpu_profiler = None
    gpu_profile_summary: Mapping[str, Any] = {"enabled": False}
    ray_context = ray.init(
        address=args.address,
        namespace=RLIX_NAMESPACE,
        ignore_reinit_error=True,
    )
    driver_address = str(ray_context.address_info["address"])
    control_actor = create_acceptance_control_actor(
        AcceptanceControlConfig(
            run_id=args.run_id,
            event_log_path=str(layout.core / "events.jsonl"),
            target_bundle=bundles[1],
            target_rank=1,
        ),
        namespace=RLIX_NAMESPACE,
    )
    if args.gpu_profile:
        gpu_profiler = Task8GPUProfiler(
            layout.root / "gpu_profile",
            interval_s=args.gpu_event_poll_interval_s,
        )
        gpu_profiler.start(event_source=lambda: ray.get(control_actor.events.remote()))
    environment = {
        "RLINF_TASK8_CONTROL_ACTOR": _actor_id(control_actor),
        "RLINF_TASK8_CONTROL_NAME": acceptance_control_actor_name(run_id=args.run_id),
        "RLINF_TASK8_CONTROL_NAMESPACE": RLIX_NAMESPACE,
        "RLINF_TASK8_RESIDENCY_VALIDATION_MODE": args.residency_validation_mode,
        "RLINF_TASK8_SNAPSHOT_VALIDATION_MODE": args.snapshot_validation_mode,
    }
    driver = Path(__file__).with_name("task8_four_rank_fsdp_driver.py")
    command = [
        sys.executable,
        str(driver),
        "--generation-proof-only",
        "--address",
        driver_address,
        "--run-id",
        args.run_id,
        "--mode",
        args.mode,
        "--bundles",
        args.bundles,
        "--config",
        str(args.config.resolve()),
        "--phase-diagnostics" if args.phase_diagnostics else "--no-phase-diagnostics",
        (
            "--acceptance-instrumentation"
            if args.acceptance_instrumentation
            else "--no-acceptance-instrumentation"
        ),
    ]
    if args.timeout_s is None:
        command.append("--no-timeout")
    else:
        command.extend(("--timeout-s", str(args.timeout_s)))
    try:
        results = run_driver_pair(
            layout=layout,
            commands={"a": command, "b": command},
            timeout_s=args.timeout_s,
            environment=environment,
            stream_driver_logs=args.stream_driver_logs,
            after_start=lambda ready: (
                _validate_four_rank_ready(ready),
                _drive_four_rank_fsdp_gates(
                    control_actor,
                    timeout_s=args.timeout_s,
                    acceptance_instrumentation=args.acceptance_instrumentation,
                ),
            ),
        )
        validate_generation_iteration_counts(
            results,
            expected_iterations=int(cfg.smoke.max_train_steps),
        )
        if args.acceptance_instrumentation:
            _validate_four_rank_lifecycle(
                ray.get(control_actor.events.remote()),
                iterations=int(cfg.smoke.max_train_steps),
            )
    except BaseException as exc:
        try:
            ray.get(
                control_actor.fail.remote(
                    role="orchestrator",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
        except Exception:
            pass
        atomic_write_json(
            layout.root / "pair_failure.json",
            {
                "status": "failed",
                "scope": "four_rank_fsdp_generation_proof",
                "task8_accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if gpu_profiler is not None:
            try:
                gpu_profile_summary = gpu_profiler.stop()
            except Exception:
                if sys.exc_info()[0] is None:
                    raise
        if ray.is_initialized():
            try:
                ray.kill(control_actor, no_restart=True)
            except Exception:
                pass
            ray.shutdown()

    summary = _summary(
        results,
        gpu_profile=gpu_profile_summary,
        acceptance_instrumentation=args.acceptance_instrumentation,
        residency_validation_mode=args.residency_validation_mode,
        snapshot_validation_mode=args.snapshot_validation_mode,
    )
    atomic_write_json(layout.root / "pair_result.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
