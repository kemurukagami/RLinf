"""Stage-transition GPU snapshots for Task 8 hardware acceptance runs."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True, slots=True)
class GPUSample:
    """One nvidia-smi device observation."""

    wall_time_ns: int
    monotonic_ns: int
    gpu_index: int
    gpu_uuid: str
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    trigger: str = "unspecified"
    driver_role: str | None = None
    component: str | None = None
    policy_version: int | None = None
    lifecycle_generation: int | None = None


class Task8GPUProfiler:
    """Query GPUs only when the acceptance stream crosses a stage boundary."""

    _QUERY = "index,uuid,utilization.gpu,memory.used,memory.total"
    _TRANSITION_EVENTS = frozenset(
        {
            "allocation_committed",
            "release_committed",
            "generation_granted",
            "batch_sealed",
            "policy_prefetch_started",
            "policy_prefetch_completed",
            "training_started",
            "training_completed",
            "stage_acquired",
            "stage_released",
            "environment_offload_verified",
            "rollout_offload_verified",
            "environment_onload_verified",
            "rollout_onload_verified",
            "useful_cuda_work_started",
        }
    )

    def __init__(self, output_dir: Path, *, interval_s: float = 0.5) -> None:
        if interval_s <= 0:
            raise ValueError("GPU profile interval must be positive")
        self.output_dir = output_dir.resolve()
        self.interval_s = float(interval_s)
        self.samples_path = self.output_dir / "gpu_samples.jsonl"
        self.summary_path = self.output_dir / "gpu_profile_summary.json"
        self._samples: list[GPUSample] = []
        self._errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._event_source: Callable[[], Iterable[Any]] | None = None
        self._seen_event_count = 0
        self._transition_keys: set[tuple[object, ...]] = set()
        self._lock = threading.Lock()

    def _query(
        self,
        *,
        trigger: str,
        driver_role: str | None = None,
        component: str | None = None,
        policy_version: int | None = None,
        lifecycle_generation: int | None = None,
    ) -> list[GPUSample]:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={self._QUERY}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(5.0, self.interval_s * 4),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "nvidia-smi GPU query failed: "
                f"exit_code={completed.returncode} stderr={completed.stderr.strip()!r}"
            )
        wall_time_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        samples = []
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 5:
                raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
            samples.append(
                GPUSample(
                    wall_time_ns=wall_time_ns,
                    monotonic_ns=monotonic_ns,
                    gpu_index=int(fields[0]),
                    gpu_uuid=fields[1],
                    utilization_percent=float(fields[2]),
                    memory_used_mib=float(fields[3]),
                    memory_total_mib=float(fields[4]),
                    trigger=trigger,
                    driver_role=driver_role,
                    component=component,
                    policy_version=policy_version,
                    lifecycle_generation=lifecycle_generation,
                )
            )
        if not samples:
            raise RuntimeError("nvidia-smi returned no GPU samples")
        return samples

    def start(self, *, event_source: Callable[[], Iterable[Any]] | None = None) -> None:
        """Persist a baseline and optionally monitor stage-transition events."""
        if self._thread is not None:
            raise RuntimeError("GPU profiler is already started")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        baseline = self._query(trigger="profiler_started")
        self._append_samples(baseline)
        self._event_source = event_source
        if event_source is None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="task8-gpu-profiler",
            daemon=True,
        )
        self._thread.start()

    def _append_samples(self, samples: list[GPUSample]) -> None:
        with self._lock:
            self._samples.extend(samples)
            with self.samples_path.open("a", encoding="utf-8") as stream:
                for sample in samples:
                    stream.write(json.dumps(asdict(sample), sort_keys=True) + "\n")

    @staticmethod
    def _event_value(event: Any, field: str, default: Any = None) -> Any:
        if isinstance(event, dict):
            return event.get(field, default)
        return getattr(event, field, default)

    def _transition_key(self, event: Any) -> tuple[object, ...]:
        details = self._event_value(event, "details", {}) or {}
        policy_version = self._event_value(event, "policy_version")
        lifecycle_generation = self._event_value(event, "lifecycle_generation")
        return (
            self._event_value(event, "driver_role"),
            self._event_value(event, "event"),
            details.get("stage"),
            policy_version
            if policy_version is not None
            else details.get("policy_version"),
            lifecycle_generation
            if lifecycle_generation is not None
            else details.get("lifecycle_generation"),
            tuple(self._event_value(event, "gpu_ids", ()) or ()),
        )

    def record_transition(self, event: Any) -> bool:
        """Record one coalesced stage transition; return whether it was sampled."""
        event_name = self._event_value(event, "event")
        if event_name not in self._TRANSITION_EVENTS:
            return False
        key = self._transition_key(event)
        if key in self._transition_keys:
            return False
        self._transition_keys.add(key)
        details = self._event_value(event, "details", {}) or {}
        policy_version = self._event_value(event, "policy_version")
        lifecycle_generation = self._event_value(event, "lifecycle_generation")
        samples = self._query(
            trigger=str(event_name),
            driver_role=self._event_value(event, "driver_role"),
            component=self._event_value(event, "component"),
            policy_version=(
                policy_version
                if policy_version is not None
                else details.get("policy_version")
            ),
            lifecycle_generation=(
                lifecycle_generation
                if lifecycle_generation is not None
                else details.get("lifecycle_generation")
            ),
        )
        self._append_samples(samples)
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                assert self._event_source is not None
                events = tuple(self._event_source())
                new_events = events[self._seen_event_count :]
                self._seen_event_count = len(events)
                for event in new_events:
                    self.record_transition(event)
            except Exception as exc:  # pragma: no cover - hardware fault path
                self._errors.append(f"{type(exc).__name__}: {exc}")

    def stop(self) -> dict[str, Any]:
        """Stop monitoring and write sparse per-GPU transition statistics."""
        if not self._samples:
            raise RuntimeError("GPU profiler was not started")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 4))
            if self._thread.is_alive():
                raise TimeoutError("GPU profiler thread did not stop")
        try:
            self._append_samples(self._query(trigger="profiler_stopped"))
        except Exception as exc:  # pragma: no cover - hardware fault path
            self._errors.append(f"{type(exc).__name__}: {exc}")
        summary = self._summarize()
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.summary_path)
        return summary

    def _summarize(self) -> dict[str, Any]:
        devices: dict[str, dict[str, Any]] = {}
        for gpu_index in sorted({sample.gpu_index for sample in self._samples}):
            samples = [s for s in self._samples if s.gpu_index == gpu_index]
            baseline_memory = samples[0].memory_used_mib
            devices[str(gpu_index)] = {
                "gpu_uuid": samples[0].gpu_uuid,
                "snapshot_count": len(samples),
                "peak_utilization_percent": max(s.utilization_percent for s in samples),
                "baseline_memory_used_mib": baseline_memory,
                "peak_memory_used_mib": max(s.memory_used_mib for s in samples),
                "peak_memory_above_baseline_mib": max(
                    s.memory_used_mib for s in samples
                )
                - baseline_memory,
                "memory_total_mib": samples[0].memory_total_mib,
            }
        return {
            "schema_version": 2,
            "scope": "node_total_nvidia_smi",
            "sampling_mode": "acceptance_stage_transitions",
            "event_poll_interval_seconds": self.interval_s,
            "samples_path": str(self.samples_path),
            "transition_snapshot_count": len(
                {sample.monotonic_ns for sample in self._samples}
            ),
            "notes": [
                "Utilization and memory are node-total device measurements; unrelated GPU processes are included.",
                "Sparse transition snapshots are not used to integrate GPU-seconds.",
            ],
            "errors": list(self._errors),
            "devices": devices,
        }
