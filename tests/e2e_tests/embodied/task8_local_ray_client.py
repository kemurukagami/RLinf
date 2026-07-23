"""Subprocess client used by the opt-in Task 8 local-Ray isolation test."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import ray
from rlix_core.client import connect
from rlix_core.protocol.types import GENERATION_CLUSTER_NAME, get_pipeline_namespace
from task8_acceptance_support import derive_role_names


def _actor_id(handle: Any) -> str:
    actor_id = handle._actor_id  # noqa: SLF001 - identity is the acceptance contract.
    to_hex = getattr(actor_id, "hex", None)
    return to_hex() if callable(to_hex) else str(actor_id)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _wait_for(path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for gate {path}")
        time.sleep(0.02)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role", choices=("a", "b"), required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--disconnect", type=Path, required=True)
    parser.add_argument("--recheck", type=Path)
    parser.add_argument("--rechecked", type=Path)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    control_plane = connect(address=args.address, create_if_missing=False)
    pipeline_id = ray.get(
        control_plane.allocate_pipeline_id.remote(pipeline_type="rlinf")
    )
    namespace = get_pipeline_namespace(pipeline_id)
    names = derive_role_names(run_id=args.run_id, role=args.role)
    candidate_mapping = {GENERATION_CLUSTER_NAME: [0, 1]}
    candidate_dp_mapping = {GENERATION_CLUSTER_NAME: {0: [0], 1: [1]}}
    ray.get(
        control_plane.register_pipeline.remote(
            pipeline_id=pipeline_id,
            ray_namespace=namespace,
            cluster_tp_configs={GENERATION_CLUSTER_NAME: 1},
            cluster_device_mappings=candidate_mapping,
            cluster_dp_device_mappings=candidate_dp_mapping,
        )
    )
    admission = ray.get(control_plane.admit_pipeline.remote(pipeline_id=pipeline_id))
    payload = {
        "pid": os.getpid(),
        "role": args.role,
        "control_plane_actor_id": _actor_id(control_plane),
        "scheduler_actor_id": _actor_id(admission.scheduler),
        "pipeline_id": pipeline_id,
        "pipeline_namespace": namespace,
        "candidate_mapping": candidate_mapping,
        "candidate_dp_mapping": candidate_dp_mapping,
        "role_names": {
            field: getattr(names, field)
            for field in names.__dataclass_fields__
            if field != "prefix"
        },
    }
    _write_json(args.ready, payload)

    if args.recheck is not None:
        if args.rechecked is None:
            raise ValueError("--rechecked is required with --recheck")
        _wait_for(args.recheck, args.timeout_s)
        current_scheduler = ray.get(control_plane.get_scheduler.remote())
        _write_json(
            args.rechecked,
            {
                "control_plane_actor_id": _actor_id(control_plane),
                "scheduler_actor_id": _actor_id(current_scheduler),
                "pipeline_id": pipeline_id,
            },
        )

    _wait_for(args.disconnect, args.timeout_s)
    ray.shutdown()


if __name__ == "__main__":
    main()
