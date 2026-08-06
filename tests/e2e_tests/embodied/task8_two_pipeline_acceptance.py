"""Fail-closed subprocess orchestration for Task 8 two-pipeline acceptance."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import ray
from omegaconf import ListConfig, OmegaConf
from rlix_core.protocol.types import RLIX_NAMESPACE
from task8_acceptance_artifacts import (
    AcceptanceArtifactLayout,
    atomic_write_json,
    prepare_acceptance_artifacts,
)
from task8_acceptance_control import (
    AcceptanceControlConfig,
    acceptance_control_actor_name,
    create_acceptance_control_actor,
)
from task8_acceptance_support import RunManifest, validate_run_id


@dataclass(frozen=True, slots=True)
class DriverProcessResult:
    """Completed identity, result, and log paths for one driver process."""

    role: str
    pid: int
    ready: Mapping[str, Any]
    result: Mapping[str, Any]
    stdout_path: Path
    stderr_path: Path


def load_acceptance_matrix_manifest(
    config_path: str | Path,
    *,
    run_id: str,
    environment: str,
    mode: str,
    scenario: str,
    checkpoint_digests: Mapping[str, str],
) -> RunManifest:
    """Validate one matrix config and freeze its acceptance manifest."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Task 8 matrix config does not exist: {path}")
    cfg = OmegaConf.load(path)
    if cfg.get("acceptance") is None or cfg.get("smoke") is None:
        raise ValueError("Task 8 matrix config requires acceptance and smoke sections")
    acceptance = cfg.acceptance
    if str(acceptance.environment) != environment:
        raise ValueError("matrix environment does not match the requested environment")
    if str(acceptance.mode) != mode:
        raise ValueError("matrix mode does not match the requested placement mode")
    if str(cfg.smoke.get("adv_type")) != "grpo":
        raise ValueError("Task 8 matrix config must use GRPO")
    training_iterations = int(cfg.smoke.get("max_train_steps", 0))
    if training_iterations < 2:
        raise ValueError("Task 8 matrix config requires at least two training steps")

    try:
        expected_bundles = tuple(
            tuple(int(gpu_id) for gpu_id in bundle)
            for bundle in acceptance.expected_bundles
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "acceptance.expected_bundles must be an integer matrix"
        ) from exc
    rollout_gpus = _config_gpu_ids(cfg.smoke.get("rollout_gpu"), field="rollout_gpu")
    env_gpus = _config_gpu_ids(cfg.smoke.get("env_gpu"), field="env_gpu")
    if len(rollout_gpus) != len(env_gpus):
        raise ValueError("rollout_gpu and env_gpu must contain the same rank count")
    configured_bundles = (
        tuple(zip(rollout_gpus, env_gpus, strict=True))
        if mode == "disaggregated"
        else tuple((gpu_id,) for gpu_id in rollout_gpus)
    )
    if mode == "collocated" and rollout_gpus != env_gpus:
        raise ValueError(
            "collocated matrix requires identical rollout and env GPU lists"
        )
    if expected_bundles != configured_bundles:
        raise ValueError(
            "acceptance bundles do not match the configured rollout/environment ranks: "
            f"expected={expected_bundles}, configured={configured_bundles}"
        )

    manifest = RunManifest(
        run_id=run_id,
        environment=environment,
        mode=mode,
        scenario=scenario,
        expected_bundles=expected_bundles,
        checkpoint_digests=dict(checkpoint_digests),
        algorithm="grpo",
        training_iterations=training_iterations,
        repetitions=int(acceptance.get("repetitions", 5)),
        warmup_repetitions=int(acceptance.get("warmup_repetitions", 1)),
        minimum_throughput_improvement=float(
            acceptance.get("minimum_throughput_improvement", 0.05)
        ),
        minimum_idle_reduction=float(acceptance.get("minimum_idle_reduction", 0.05)),
        sample_interval_ms=int(acceptance.get("sample_interval_ms", 100)),
    )
    manifest.validate()
    return manifest


def _config_gpu_ids(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, ListConfig)):
        raise ValueError(f"smoke.{field} must be a GPU list")
    try:
        gpu_ids = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"smoke.{field} must contain integer GPU IDs") from exc
    if (
        not gpu_ids
        or len(gpu_ids) != len(set(gpu_ids))
        or any(item < 0 for item in gpu_ids)
    ):
        raise ValueError(f"smoke.{field} must contain unique non-negative GPU IDs")
    return gpu_ids


def validate_driver_ready_pair(
    ready_by_role: Mapping[str, Mapping[str, Any]],
) -> None:
    """Prove shared core identity and distinct pipeline-owned identities."""
    if set(ready_by_role) != {"a", "b"}:
        raise ValueError("driver readiness requires exactly roles 'a' and 'b'")
    a = ready_by_role["a"]
    b = ready_by_role["b"]
    common = {
        "scope",
        "role",
        "pid",
        "control_plane_actor_id",
        "scheduler_actor_id",
        "pipeline_id",
        "pipeline_namespace",
        "candidate_mapping",
        "candidate_dp_mapping",
        "role_names",
    }
    scopes = {payload.get("scope") for payload in ready_by_role.values()}
    if len(scopes) != 1 or scopes.pop() not in {
        "acceptance_control_only",
        "connectivity_only",
        "generation_proof_only",
        "model_init_only",
    }:
        raise ValueError("drivers must report one matching supported readiness scope")
    scope = a["scope"]
    if scope == "acceptance_control_only":
        required = {
            "scope",
            "role",
            "pid",
            "acceptance_control_actor_id",
            "event_log_path",
        }
    else:
        required = (
            common
            | (
                {"runtime_state", "actor_infer_bundles", "residencies"}
                | {"generation_preemption_mode"}
                if scope in {"generation_proof_only", "model_init_only"}
                else set()
            )
            | (
                {"acceptance_control_actor_id", "event_log_path"}
                if scope == "generation_proof_only"
                else set()
            )
        )
    for role, payload in (("a", a), ("b", b)):
        if set(payload) != required:
            raise ValueError(f"driver {role} readiness schema is incomplete")
        if payload["role"] != role:
            raise ValueError(f"driver {role} readiness reports the wrong role")
        if not isinstance(payload["pid"], int) or isinstance(payload["pid"], bool):
            raise ValueError(f"driver {role} readiness has an invalid pid")
        if any(not payload[field] for field in required - {"pid", "role"}):
            raise ValueError(f"driver {role} readiness contains an empty identity")
        if scope == "acceptance_control_only":
            continue
        names = payload["role_names"]
        if not isinstance(names, Mapping) or len(names) != len(set(names.values())):
            raise ValueError(f"driver {role} has invalid role-owned names")
        if scope in {"generation_proof_only", "model_init_only"}:
            if payload["generation_preemption_mode"] != "fixed_stage_only":
                raise ValueError(
                    f"driver {role} did not register stage-aware generation"
                )
            if payload["runtime_state"] != "inactive":
                raise ValueError(f"driver {role} model runtime is not inactive")
            bundles = payload["actor_infer_bundles"]
            if not isinstance(bundles, list) or not bundles:
                raise ValueError(f"driver {role} has invalid actor-infer bundles")
            residencies = payload["residencies"]
            if not isinstance(residencies, list) or not residencies:
                raise ValueError(f"driver {role} has no model residency evidence")
            for residency in residencies:
                if not isinstance(residency, Mapping) or set(residency) != {
                    "component",
                    "rank",
                    "model_resident",
                    "optimizer_resident",
                    "cuda_graph_captured",
                    "policy_version",
                    "safe_to_release",
                }:
                    raise ValueError(
                        f"driver {role} has malformed model residency evidence"
                    )
                if residency["safe_to_release"] is not True or any(
                    residency[field] is not False
                    for field in (
                        "model_resident",
                        "optimizer_resident",
                        "cuda_graph_captured",
                    )
                ):
                    raise ValueError(
                        f"driver {role} has accelerator-resident model state"
                    )
    if scope == "acceptance_control_only":
        if a["acceptance_control_actor_id"] != b["acceptance_control_actor_id"]:
            raise ValueError("drivers resolved different acceptance control actors")
        if a["event_log_path"] != b["event_log_path"]:
            raise ValueError("drivers resolved different acceptance event logs")
        return
    for field in ("control_plane_actor_id", "scheduler_actor_id"):
        if a[field] != b[field]:
            raise ValueError(f"drivers resolved different {field}")
    if scope == "generation_proof_only":
        if a["acceptance_control_actor_id"] != b["acceptance_control_actor_id"]:
            raise ValueError("drivers resolved different acceptance control actors")
        if a["event_log_path"] != b["event_log_path"]:
            raise ValueError("drivers resolved different acceptance event logs")
    for field in ("pipeline_id", "pipeline_namespace"):
        if a[field] == b[field]:
            raise ValueError(f"drivers share pipeline-owned identity {field}")
    if set(a["role_names"].values()) & set(b["role_names"].values()):
        raise ValueError("driver role-owned names collide")
    if a["candidate_mapping"] != b["candidate_mapping"]:
        raise ValueError("drivers must register intentionally overlapping candidates")
    if a["candidate_dp_mapping"] != b["candidate_dp_mapping"]:
        raise ValueError("drivers must register the same candidate rank bundles")


def validate_generation_iteration_counts(
    results: Mapping[str, DriverProcessResult], *, expected_iterations: int
) -> None:
    """Require both generation drivers to complete every configured iteration."""
    if expected_iterations <= 0 or set(results) != {"a", "b"}:
        raise ValueError("generation iteration validation requires two drivers")
    for role, result in results.items():
        configured = result.result.get("configured_iterations")
        completed = result.result.get("completed_iterations")
        if configured != expected_iterations or completed != expected_iterations:
            raise ValueError(
                f"driver {role} did not complete every configured iteration: "
                f"expected={expected_iterations}, configured={configured}, "
                f"completed={completed}"
            )


def run_driver_pair(
    *,
    layout: AcceptanceArtifactLayout,
    commands: Mapping[str, Sequence[str]],
    timeout_s: float,
    environment: Mapping[str, str] | None = None,
    after_start: Callable[[Mapping[str, Mapping[str, Any]]], None] | None = None,
    stream_driver_logs: bool = False,
    clock: Callable[[], float] = time.monotonic,
    poll_interval_s: float = 0.05,
) -> dict[str, DriverProcessResult]:
    """Launch two OS drivers, validate readiness, release, and collect results."""
    if set(commands) != {"a", "b"}:
        raise ValueError("commands must contain exactly roles 'a' and 'b'")
    if timeout_s <= 0 or poll_interval_s <= 0:
        raise ValueError("driver timeout and poll interval must be positive")
    for role, command in commands.items():
        if not command or any(not argument for argument in command):
            raise ValueError(f"driver {role} command must not be empty")

    deadline = clock() + timeout_s
    control_dir = layout.root / "control"
    control_dir.mkdir(parents=True, exist_ok=False)
    start_path = control_dir / "start.json"
    processes: dict[str, subprocess.Popen[str]] = {}
    streams: list[Any] = []
    pump_threads: list[threading.Thread] = []
    terminal_lock = threading.Lock()
    try:
        for role in ("a", "b"):
            driver_dir = layout.drivers / role
            stdout_path = driver_dir / "stdout.log"
            stderr_path = driver_dir / "stderr.log"
            stdout = stdout_path.open("x", encoding="utf-8")
            stderr = stderr_path.open("x", encoding="utf-8")
            streams.extend((stdout, stderr))
            driver_environment = dict(os.environ)
            if environment is not None:
                driver_environment.update(environment)
            driver_environment.update(
                {
                    "RLINF_TASK8_ROLE": role,
                    "RLINF_TASK8_READY": str(driver_dir / "ready.json"),
                    "RLINF_TASK8_START": str(start_path),
                    "RLINF_TASK8_RESULT": str(driver_dir / "result.json"),
                }
            )
            if stream_driver_logs:
                driver_environment["PYTHONUNBUFFERED"] = "1"
            processes[role] = subprocess.Popen(
                list(commands[role]),
                stdout=subprocess.PIPE if stream_driver_logs else stdout,
                stderr=subprocess.PIPE if stream_driver_logs else stderr,
                env=driver_environment,
                start_new_session=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if stream_driver_logs:
                process = processes[role]
                if process.stdout is None or process.stderr is None:
                    raise RuntimeError("driver log pipes were not created")
                for source, sink, terminal, stream_name in (
                    (process.stdout, stdout, sys.stdout, "stdout"),
                    (process.stderr, stderr, sys.stderr, "stderr"),
                ):
                    thread = threading.Thread(
                        target=_pump_driver_log,
                        args=(source, sink, terminal),
                        kwargs={
                            "prefix": f"[driver {role} {stream_name}] ",
                            "lock": terminal_lock,
                        },
                        name=f"task8-{role}-{stream_name}",
                        daemon=True,
                    )
                    thread.start()
                    pump_threads.append(thread)

        ready = {
            role: _wait_for_json(
                layout.drivers / role / "ready.json",
                role=role,
                processes=processes,
                deadline=deadline,
                clock=clock,
                poll_interval_s=poll_interval_s,
            )
            for role in ("a", "b")
        }
        validate_driver_ready_pair(ready)
        atomic_write_json(
            start_path,
            {
                "status": "released",
                "scope": ready["a"]["scope"],
                **(
                    {
                        "control_plane_actor_id": ready["a"]["control_plane_actor_id"],
                        "scheduler_actor_id": ready["a"]["scheduler_actor_id"],
                    }
                    if ready["a"]["scope"] != "acceptance_control_only"
                    else {
                        "acceptance_control_actor_id": ready["a"][
                            "acceptance_control_actor_id"
                        ],
                    }
                ),
            },
        )
        if after_start is not None:
            after_start(ready)

        results: dict[str, DriverProcessResult] = {}
        for role in ("a", "b"):
            payload = _wait_for_json(
                layout.drivers / role / "result.json",
                role=role,
                processes=processes,
                deadline=deadline,
                clock=clock,
                poll_interval_s=poll_interval_s,
            )
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("driver pair exceeded its completion deadline")
            return_code = processes[role].wait(timeout=remaining)
            if return_code != 0:
                raise RuntimeError(f"driver {role} exited with code {return_code}")
            if payload.get("status") != "passed" or payload.get("role") != role:
                raise ValueError(f"driver {role} did not report a passing result")
            if (
                payload.get("scope") != ready[role]["scope"]
                or payload.get("task8_accepted") is not False
            ):
                raise ValueError(
                    f"driver {role} reported an invalid preliminary result scope"
                )
            results[role] = DriverProcessResult(
                role=role,
                pid=processes[role].pid,
                ready=ready[role],
                result=payload,
                stdout_path=layout.drivers / role / "stdout.log",
                stderr_path=layout.drivers / role / "stderr.log",
            )
        return results
    except BaseException:
        _stop_processes(processes)
        raise
    finally:
        for thread in pump_threads:
            thread.join(timeout=5.0)
        for stream in streams:
            stream.close()


def _pump_driver_log(
    source: Any,
    sink: Any,
    terminal: Any,
    *,
    prefix: str,
    lock: threading.Lock,
) -> None:
    """Copy one child stream to its artifact and the parent terminal."""
    try:
        for line in iter(source.readline, ""):
            sink.write(line)
            sink.flush()
            with lock:
                terminal.write(f"{prefix}{line}")
                terminal.flush()
    finally:
        source.close()


def _wait_for_json(
    path: Path,
    *,
    role: str,
    processes: Mapping[str, subprocess.Popen[Any]],
    deadline: float,
    clock: Callable[[], float],
    poll_interval_s: float,
) -> Mapping[str, Any]:
    while not path.is_file():
        return_code = processes[role].poll()
        if return_code is not None:
            raise RuntimeError(
                f"driver {role} exited with code {return_code} before writing {path.name}"
            )
        if clock() >= deadline:
            raise TimeoutError(f"timed out waiting for driver {role} {path.name}")
        time.sleep(poll_interval_s)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"driver {role} wrote unreadable {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"driver {role} {path.name} must contain a JSON object")
    return payload


def _stop_processes(processes: Mapping[str, subprocess.Popen[Any]]) -> None:
    live = [process for process in processes.values() if process.poll() is None]
    for process in live:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for process in live:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in live:
        if process.poll() is None:
            process.wait(timeout=5.0)


def prepare_preliminary_driver_layout(
    output_dir: str | Path,
    *,
    run_id: str,
    scope: str,
) -> AcceptanceArtifactLayout:
    """Create an isolated non-acceptance run tree for preliminary drivers."""
    validate_run_id(run_id)
    if scope not in {
        "acceptance_control_only",
        "connectivity_only",
        "generation_proof_only",
        "model_init_only",
    }:
        raise ValueError("unsupported preliminary driver scope")
    root = Path(output_dir).resolve() / run_id
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"preliminary run directory is not empty: {root}")
    layout = AcceptanceArtifactLayout(
        root=root,
        drivers=root / "drivers",
        core=root / "core",
        gpu=root / "gpu",
        reference=root / "reference",
        analysis=root / "analysis",
    )
    for directory in (layout.root, layout.drivers / "a", layout.drivers / "b"):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        layout.root / "preliminary_manifest.json",
        {
            "run_id": run_id,
            "scope": scope,
            "task8_accepted": False,
        },
    )
    return layout


def _parse_control_bundles(raw: str, *, mode: str) -> tuple[tuple[int, ...], ...]:
    expected_width = 2 if mode == "disaggregated" else 1
    bundles: list[tuple[int, ...]] = []
    try:
        for encoded_bundle in raw.split(";"):
            bundles.append(tuple(int(gpu_id) for gpu_id in encoded_bundle.split(",")))
    except ValueError as exc:
        raise ValueError("control bundles must contain integer GPU IDs") from exc
    if len(bundles) < 2 or any(len(bundle) != expected_width for bundle in bundles):
        raise ValueError(
            f"{mode} acceptance-control smoke requires at least two "
            f"width-{expected_width} bundles"
        )
    flattened = [gpu_id for bundle in bundles for gpu_id in bundle]
    if any(gpu_id < 0 for gpu_id in flattened) or len(flattened) != len(set(flattened)):
        raise ValueError("control bundles require disjoint non-negative GPU IDs")
    return tuple(bundles)


def _actor_id(handle: Any) -> str:
    actor_id = handle._actor_id  # noqa: SLF001 - identity is acceptance evidence.
    to_hex = getattr(actor_id, "hex", None)
    return to_hex() if callable(to_hex) else str(actor_id)


def _drive_control_gates(control_actor: Any, *, timeout_s: float) -> None:
    ray.get(
        control_actor.wait_for_gate.remote(
            "both_drivers_initialized", timeout_s=timeout_s
        )
    )
    ray.get(control_actor.release_gate.remote("allow_a_collection"))
    ray.get(
        control_actor.wait_for_gate.remote(
            "a_target_chunk_started", timeout_s=timeout_s
        )
    )
    ray.get(control_actor.release_gate.remote("allow_b_demand"))
    ray.get(
        control_actor.wait_for_gate.remote(
            "transfer_to_b_observed", timeout_s=timeout_s
        )
    )
    ray.get(control_actor.release_gate.remote("allow_b_release"))
    ray.get(
        control_actor.wait_for_gate.remote("a_resume_observed", timeout_s=timeout_s)
    )


def _drive_generation_proof_gates(control_actor: Any, *, timeout_s: float) -> None:
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
    # Publish B's demand before A releases either bundle. The scheduler can
    # then commit the first free bundle immediately without an orchestrator
    # round trip on the resource-release critical path.
    ray.get(control_actor.release_gate.remote("allow_b_collection"))
    ray.get(
        control_actor.wait_for_gate.remote(
            "b_generation_requested", timeout_s=timeout_s
        )
    )
    ray.get(
        control_actor.wait_for_gate.remote(
            "a_target_rank_completed", timeout_s=timeout_s
        )
    )
    ray.get(
        control_actor.wait_for_gate.remote(
            "transfer_to_b_observed", timeout_s=timeout_s
        )
    )
    ray.get(
        control_actor.wait_for_gate.remote(
            "b_useful_work_observed", timeout_s=timeout_s
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=(
            "acceptance-control-only",
            "connectivity-only",
            "generation-proof-only",
            "model-init-only",
            "acceptance-preflight",
        ),
        required=True,
    )
    parser.add_argument("--address", default="auto")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("disaggregated", "collocated"), required=True
    )
    parser.add_argument("--bundles")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--environment", choices=("wan", "opensora"))
    parser.add_argument(
        "--scenario", choices=("reference", "recovery", "utilization", "all")
    )
    parser.add_argument("--vla-checkpoint-digest")
    parser.add_argument("--environment-checkpoint-digest")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--phase-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable acceptance-only fine-grained world-model phase markers",
    )
    parser.add_argument(
        "--stream-driver-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="mirror both driver stdout/stderr streams to this terminal",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scope = args.scope.replace("-", "_")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if scope == "acceptance_preflight":
        required = {
            "--config": args.config,
            "--environment": args.environment,
            "--scenario": args.scenario,
            "--vla-checkpoint-digest": args.vla_checkpoint_digest,
            "--environment-checkpoint-digest": args.environment_checkpoint_digest,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "acceptance preflight is missing required arguments: "
                + ", ".join(missing)
            )
        manifest = load_acceptance_matrix_manifest(
            args.config,
            run_id=args.run_id,
            environment=args.environment,
            mode=args.mode,
            scenario=args.scenario,
            checkpoint_digests={
                "vla": args.vla_checkpoint_digest,
                args.environment: args.environment_checkpoint_digest,
            },
        )
        layout = prepare_acceptance_artifacts(args.output_dir, run_manifest=manifest)
        summary = {
            "status": "passed",
            "scope": scope,
            "task8_accepted": False,
            "run_manifest": str(layout.root / "run_manifest.json"),
            "message": (
                "Matrix preflight passed; no drivers or models were started and "
                "this is not Task 8 acceptance evidence."
            ),
        }
        atomic_write_json(layout.root / "preflight_result.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if scope in {"generation_proof_only", "model_init_only"} and args.config is None:
        raise ValueError("--config is required for model initialization/generation")
    if (
        scope in {"acceptance_control_only", "connectivity_only"}
        and args.config is not None
    ):
        raise ValueError("--config is only valid for model initialization/generation")
    if args.bundles is None:
        raise ValueError("--bundles is required for preliminary driver scopes")

    layout = prepare_preliminary_driver_layout(
        args.output_dir,
        run_id=args.run_id,
        scope=scope,
    )
    environment: dict[str, str] = {}
    control_actor = None
    driver_address = args.address
    if scope in {"acceptance_control_only", "generation_proof_only"}:
        bundles = _parse_control_bundles(args.bundles, mode=args.mode)
        if ray.is_initialized():
            raise RuntimeError("orchestrator unexpectedly initialized Ray early")
        ray_context = ray.init(
            address=args.address, namespace=RLIX_NAMESPACE, ignore_reinit_error=True
        )
        driver_address = str(ray_context.address_info["address"])
        control_actor = create_acceptance_control_actor(
            AcceptanceControlConfig(
                run_id=args.run_id,
                event_log_path=str(layout.core / "events.jsonl"),
                # Actor training uses GPU 0 in the four-GPU acceptance layout.
                # Hand off the non-overlapping rank while retaining [0, 2].
                target_bundle=bundles[1],
                target_rank=1,
            ),
            namespace=RLIX_NAMESPACE,
        )
        environment.update(
            {
                "RLINF_TASK8_CONTROL_ACTOR": _actor_id(control_actor),
                "RLINF_TASK8_CONTROL_NAME": acceptance_control_actor_name(
                    run_id=args.run_id
                ),
                "RLINF_TASK8_CONTROL_NAMESPACE": RLIX_NAMESPACE,
            }
        )
    driver = Path(__file__).with_name("task8_two_pipeline_driver.py")
    command = [
        sys.executable,
        str(driver),
        f"--{args.scope}",
        "--address",
        driver_address,
        "--run-id",
        args.run_id,
        "--mode",
        args.mode,
        "--bundles",
        args.bundles,
        "--timeout-s",
        str(args.timeout_s),
        ("--phase-diagnostics" if args.phase_diagnostics else "--no-phase-diagnostics"),
    ]
    if args.config is not None:
        command.extend(("--config", str(args.config.resolve())))
    try:
        results = run_driver_pair(
            layout=layout,
            commands={"a": command, "b": command},
            timeout_s=args.timeout_s,
            environment=environment,
            stream_driver_logs=args.stream_driver_logs,
            after_start=(
                (
                    lambda _: (
                        _drive_generation_proof_gates(
                            control_actor, timeout_s=args.timeout_s
                        )
                        if scope == "generation_proof_only"
                        else _drive_control_gates(
                            control_actor, timeout_s=args.timeout_s
                        )
                    )
                )
                if scope in {"acceptance_control_only", "generation_proof_only"}
                else None
            ),
        )
        if scope == "generation_proof_only":
            expected_iterations = int(OmegaConf.load(args.config).smoke.max_train_steps)
            validate_generation_iteration_counts(
                results, expected_iterations=expected_iterations
            )
    except BaseException as exc:
        if (
            scope in {"acceptance_control_only", "generation_proof_only"}
            and control_actor is not None
        ):
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
                "scope": scope,
                "task8_accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if (
            scope in {"acceptance_control_only", "generation_proof_only"}
            and ray.is_initialized()
        ):
            ray.shutdown()
    summary = {
        "status": "passed",
        "scope": scope,
        "task8_accepted": False,
        "drivers": {
            role: {
                "pid": result.pid,
                **(
                    {"pipeline_id": result.ready["pipeline_id"]}
                    if "pipeline_id" in result.ready
                    else {
                        "acceptance_control_actor_id": result.ready[
                            "acceptance_control_actor_id"
                        ],
                    }
                ),
                "result": dict(result.result),
            }
            for role, result in results.items()
        },
    }
    atomic_write_json(layout.root / "pair_result.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
