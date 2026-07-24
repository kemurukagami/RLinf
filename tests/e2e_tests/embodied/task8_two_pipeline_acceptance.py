"""Fail-closed subprocess orchestration for Task 8 two-pipeline acceptance."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from task8_acceptance_artifacts import AcceptanceArtifactLayout, atomic_write_json


@dataclass(frozen=True, slots=True)
class DriverProcessResult:
    """Completed identity, result, and log paths for one driver process."""

    role: str
    pid: int
    ready: Mapping[str, Any]
    result: Mapping[str, Any]
    stdout_path: Path
    stderr_path: Path


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
        "connectivity_only",
        "model_init_only",
    }:
        raise ValueError("drivers must report one matching supported readiness scope")
    scope = a["scope"]
    required = common | (
        {"runtime_state", "actor_infer_bundles", "residencies"}
        if scope == "model_init_only"
        else set()
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
        names = payload["role_names"]
        if not isinstance(names, Mapping) or len(names) != len(set(names.values())):
            raise ValueError(f"driver {role} has invalid role-owned names")
        if scope == "model_init_only":
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
    for field in ("control_plane_actor_id", "scheduler_actor_id"):
        if a[field] != b[field]:
            raise ValueError(f"drivers resolved different {field}")
    for field in ("pipeline_id", "pipeline_namespace"):
        if a[field] == b[field]:
            raise ValueError(f"drivers share pipeline-owned identity {field}")
    if set(a["role_names"].values()) & set(b["role_names"].values()):
        raise ValueError("driver role-owned names collide")
    if a["candidate_mapping"] != b["candidate_mapping"]:
        raise ValueError("drivers must register intentionally overlapping candidates")
    if a["candidate_dp_mapping"] != b["candidate_dp_mapping"]:
        raise ValueError("drivers must register the same candidate rank bundles")


def run_driver_pair(
    *,
    layout: AcceptanceArtifactLayout,
    commands: Mapping[str, Sequence[str]],
    timeout_s: float,
    environment: Mapping[str, str] | None = None,
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
    processes: dict[str, subprocess.Popen[bytes]] = {}
    streams: list[Any] = []
    try:
        for role in ("a", "b"):
            driver_dir = layout.drivers / role
            stdout_path = driver_dir / "stdout.log"
            stderr_path = driver_dir / "stderr.log"
            stdout = stdout_path.open("xb")
            stderr = stderr_path.open("xb")
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
            processes[role] = subprocess.Popen(
                list(commands[role]),
                stdout=stdout,
                stderr=stderr,
                env=driver_environment,
                start_new_session=True,
            )

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
                "control_plane_actor_id": ready["a"]["control_plane_actor_id"],
                "scheduler_actor_id": ready["a"]["scheduler_actor_id"],
            },
        )

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
        for stream in streams:
            stream.close()


def _wait_for_json(
    path: Path,
    *,
    role: str,
    processes: Mapping[str, subprocess.Popen[bytes]],
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


def _stop_processes(processes: Mapping[str, subprocess.Popen[bytes]]) -> None:
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
    run_component = Path(run_id)
    if (
        run_component.is_absolute()
        or len(run_component.parts) != 1
        or run_id in {"", ".", ".."}
    ):
        raise ValueError("run_id must be one safe output-directory component")
    if scope not in {"connectivity_only", "model_init_only"}:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("connectivity-only", "model-init-only"),
        required=True,
    )
    parser.add_argument("--address", default="auto")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("disaggregated", "collocated"), required=True
    )
    parser.add_argument("--bundles", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scope = args.scope.replace("-", "_")
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    if scope == "model_init_only" and args.config is None:
        raise ValueError("--config is required for model initialization")
    if scope == "connectivity_only" and args.config is not None:
        raise ValueError("--config is only valid for model initialization")

    layout = prepare_preliminary_driver_layout(
        args.output_dir,
        run_id=args.run_id,
        scope=scope,
    )
    driver = Path(__file__).with_name("task8_two_pipeline_driver.py")
    command = [
        sys.executable,
        str(driver),
        f"--{args.scope}",
        "--address",
        args.address,
        "--run-id",
        args.run_id,
        "--mode",
        args.mode,
        "--bundles",
        args.bundles,
        "--timeout-s",
        str(args.timeout_s),
    ]
    if args.config is not None:
        command.extend(("--config", str(args.config.resolve())))
    try:
        results = run_driver_pair(
            layout=layout,
            commands={"a": command, "b": command},
            timeout_s=args.timeout_s,
        )
    except BaseException as exc:
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
    summary = {
        "status": "passed",
        "scope": scope,
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
    atomic_write_json(layout.root / "pair_result.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
