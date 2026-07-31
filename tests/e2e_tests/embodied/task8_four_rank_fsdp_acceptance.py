"""Two-pipeline proof for four-rank FSDP training on one four-GPU node."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--phase-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stream-driver-logs",
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


def _wait_for_both_b_ranks_working(control_actor: Any, *, timeout_s: float) -> None:
    """Wait until both B environment ranks have entered a real Wan chunk."""
    deadline = time.monotonic() + timeout_s
    while True:
        events = ray.get(control_actor.events.remote())
        active = {
            event.dp_rank
            for event in events
            if event.driver_role == "b"
            and event.component == "environment"
            and event.event == "chunk_started"
            and event.dp_rank in {0, 1}
        }
        completed = {
            event.dp_rank
            for event in events
            if event.driver_role == "b"
            and event.component == "rollout"
            and event.event == "rank_completed"
            and event.dp_rank in {0, 1}
        }
        if completed:
            raise RuntimeError(
                "B completed a rank before both ranks were staged for preemption: "
                f"completed={sorted(completed)}"
            )
        if active == {0, 1}:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for B to start real chunks on ranks 0 and 1"
            )
        time.sleep(0.05)


def _drive_four_rank_fsdp_gates(control_actor: Any, *, timeout_s: float) -> None:
    """Stage both B ranks before releasing A's all-GPU training request."""
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
    ray.get(
        control_actor.wait_for_gate.remote(
            "a_target_rank_completed", timeout_s=timeout_s
        )
    )
    ray.get(control_actor.release_gate.remote("allow_b_collection"))
    ray.get(
        control_actor.wait_for_gate.remote(
            "b_generation_requested", timeout_s=timeout_s
        )
    )
    ray.get(control_actor.wait_for_gate.remote("a_batch_sealed", timeout_s=timeout_s))
    _wait_for_both_b_ranks_working(control_actor, timeout_s=timeout_s)
    ray.get(control_actor.release_gate.remote("allow_a_training"))
    ray.get(
        control_actor.wait_for_gate.remote("a_training_started", timeout_s=timeout_s)
    )
    ray.get(
        control_actor.wait_for_gate.remote("a_training_completed", timeout_s=timeout_s)
    )
    ray.get(control_actor.wait_for_gate.remote("b_batch_sealed", timeout_s=timeout_s))
    ray.get(control_actor.release_gate.remote("allow_b_training"))
    ray.get(
        control_actor.wait_for_gate.remote(
            "both_training_completed", timeout_s=timeout_s
        )
    )


def _validate_four_rank_lifecycle(events: list[Any] | tuple[Any, ...]) -> None:
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


def _summary(results: Mapping[str, DriverProcessResult]) -> dict[str, Any]:
    return {
        "status": "passed",
        "scope": "four_rank_fsdp_generation_proof",
        "task8_accepted": False,
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
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    cfg = OmegaConf.load(args.config)
    if tuple(cfg.smoke.actor_gpus) != (0, 1, 2, 3):
        raise ValueError("config must place four FSDP actor ranks on GPUs 0,1,2,3")
    if cfg.smoke.completed_bundle_handoff != "release_before_training":
        raise ValueError("config must release completed bundles before training")
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
    environment = {
        "RLINF_TASK8_CONTROL_ACTOR": _actor_id(control_actor),
        "RLINF_TASK8_CONTROL_NAME": acceptance_control_actor_name(run_id=args.run_id),
        "RLINF_TASK8_CONTROL_NAMESPACE": RLIX_NAMESPACE,
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
        "--timeout-s",
        str(args.timeout_s),
        "--phase-diagnostics" if args.phase_diagnostics else "--no-phase-diagnostics",
    ]
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
                ),
            ),
        )
        validate_generation_iteration_counts(
            results,
            expected_iterations=int(cfg.smoke.max_train_steps),
        )
        _validate_four_rank_lifecycle(ray.get(control_actor.events.remote()))
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
        if ray.is_initialized():
            ray.shutdown()

    summary = _summary(results)
    atomic_write_json(layout.root / "pair_result.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
