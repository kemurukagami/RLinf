from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).parents[1] / "e2e_tests" / "embodied" / "task8_gpu_profiler.py"
)
_SPEC = importlib.util.spec_from_file_location("task8_gpu_profiler", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
GPUSample = _MODULE.GPUSample
Task8GPUProfiler = _MODULE.Task8GPUProfiler


def test_gpu_profile_summarizes_sparse_transition_snapshots(tmp_path: Path) -> None:
    profiler = Task8GPUProfiler(tmp_path, interval_s=0.5)
    profiler._samples = [  # noqa: SLF001 - deterministic profiler unit fixture.
        GPUSample(10, 0, 0, "GPU-0", 50.0, 100.0, 1000.0),
        GPUSample(20, 1_000_000_000, 0, "GPU-0", 100.0, 300.0, 1000.0),
        GPUSample(30, 3_000_000_000, 0, "GPU-0", 0.0, 150.0, 1000.0),
    ]

    summary = profiler._summarize()  # noqa: SLF001
    gpu = summary["devices"]["0"]

    assert gpu["snapshot_count"] == 3
    assert gpu["peak_utilization_percent"] == 100.0
    assert gpu["peak_memory_used_mib"] == 300.0
    assert gpu["peak_memory_above_baseline_mib"] == 200.0


def test_gpu_profile_keeps_devices_separate(tmp_path: Path) -> None:
    profiler = Task8GPUProfiler(tmp_path)
    profiler._samples = [  # noqa: SLF001
        GPUSample(10, 0, 0, "GPU-0", 10.0, 100.0, 1000.0),
        GPUSample(10, 0, 1, "GPU-1", 90.0, 200.0, 1000.0),
        GPUSample(20, 1_000_000_000, 0, "GPU-0", 10.0, 100.0, 1000.0),
        GPUSample(20, 1_000_000_000, 1, "GPU-1", 90.0, 200.0, 1000.0),
    ]

    devices = profiler._summarize()["devices"]  # noqa: SLF001

    assert set(devices) == {"0", "1"}
    assert devices["0"]["peak_utilization_percent"] == 10.0
    assert devices["1"]["peak_utilization_percent"] == 90.0


def test_gpu_profile_queries_only_coalesced_stage_transitions(tmp_path: Path) -> None:
    profiler = Task8GPUProfiler(tmp_path)
    queries: list[str] = []

    def query(**kwargs):
        queries.append(kwargs["trigger"])
        return [
            GPUSample(
                10,
                len(queries),
                0,
                "GPU-0",
                50.0,
                100.0,
                1000.0,
                trigger=kwargs["trigger"],
            )
        ]

    profiler._query = query  # type: ignore[method-assign]  # noqa: SLF001
    irrelevant = {"event": "chunk_started", "driver_role": "a", "details": {}}
    transition = {
        "event": "training_started",
        "driver_role": "a",
        "component": "actor",
        "details": {"policy_version": 2},
    }

    assert not profiler.record_transition(irrelevant)
    assert profiler.record_transition(transition)
    assert not profiler.record_transition(dict(transition, dp_rank=1))
    assert profiler.record_transition(dict(transition, policy_version=3))
    assert queries == ["training_started", "training_started"]
