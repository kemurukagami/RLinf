"""CPU tests for Task 8 acceptance evidence and analysis."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.scheduler.rlix.validation import (
    validate_elastic_vla_config,
    validate_rank_early_completion_config,
)


def _load_acceptance_support():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_acceptance_support.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task8_acceptance_support", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Task 8 acceptance support from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_support = _load_acceptance_support()


def _load_acceptance_analysis():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_acceptance_analysis.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task8_acceptance_analysis", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Task 8 acceptance analysis from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_analysis = _load_acceptance_analysis()


def _load_acceptance_artifacts():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_acceptance_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task8_acceptance_artifacts", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load Task 8 acceptance artifacts from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_artifacts = _load_acceptance_artifacts()


def _load_acceptance_control():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_acceptance_control.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task8_acceptance_control", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Task 8 acceptance control from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


_control = _load_acceptance_control()


def _load_two_pipeline_acceptance():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_two_pipeline_acceptance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task8_two_pipeline_acceptance", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Task 8 orchestrator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_orchestrator = _load_two_pipeline_acceptance()


def _load_two_pipeline_driver():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_two_pipeline_driver.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task8_two_pipeline_driver", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Task 8 driver from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


_driver = _load_two_pipeline_driver()


def _load_single_pipeline_diagnostic():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_single_pipeline_diagnostic.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task8_single_pipeline_diagnostic", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load Task 8 single-pipeline diagnostic from {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(module_path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


_single_pipeline = _load_single_pipeline_diagnostic()
_workers = sys.modules["task8_acceptance_workers"]
AcceptanceEvent = _support.AcceptanceEvent
AcceptanceEventProducer = _support.AcceptanceEventProducer
AcceptanceEventSink = _support.AcceptanceEventSink
AcceptanceProducerContext = _support.AcceptanceProducerContext
AllocationSlice = _support.AllocationSlice
SchedulerCommitRecord = _support.SchedulerCommitRecord
GpuSample = _support.GpuSample
GpuUtilizationSummary = _support.GpuUtilizationSummary
RunManifest = _support.RunManifest
TransitionIdentity = _support.TransitionIdentity
UtilizationTrial = _support.UtilizationTrial
compare_manifests = _support.compare_manifests
derive_role_names = _support.derive_role_names
evaluate_utilization_trials = _support.evaluate_utilization_trials
logical_tensor_bytes = _support.logical_tensor_bytes
normalize_manifest = _support.normalize_manifest
normalize_scheduler_commit_marker = _support.normalize_scheduler_commit_marker
summarize_gpu_samples = _support.summarize_gpu_samples
validate_run_id = _support.validate_run_id
validate_exclusive_ownership = _support.validate_exclusive_ownership
analyze_snapshot_size = _analysis.analyze_snapshot_size
BatchManifest = _analysis.BatchManifest
build_analysis_summary = _analysis.build_analysis_summary
compare_reference_batches = _analysis.compare_reference_batches
GrpoIterationEvidence = _analysis.GrpoIterationEvidence
TransitionRecord = _analysis.TransitionRecord
validate_grpo_training_loop = _analysis.validate_grpo_training_loop
correlate_scheduler_transfer_commits = _analysis.correlate_scheduler_transfer_commits
validate_transfer_timeline = _analysis.validate_transfer_timeline
extract_scheduler_commits_from_perfetto = (
    _artifacts.extract_scheduler_commits_from_perfetto
)
AcceptanceReportMetadata = _artifacts.AcceptanceReportMetadata
atomic_write_csv = _artifacts.atomic_write_csv
atomic_write_json = _artifacts.atomic_write_json
atomic_write_jsonl = _artifacts.atomic_write_jsonl
emit_acceptance_artifact_set = _artifacts.emit_acceptance_artifact_set
parse_perfetto_commit_csv = _artifacts.parse_perfetto_commit_csv
prepare_acceptance_artifacts = _artifacts.prepare_acceptance_artifacts
verify_completion_marker = _artifacts.verify_completion_marker
write_completion_marker = _artifacts.write_completion_marker
AcceptanceControlConfig = _control.AcceptanceControlConfig
AcceptanceControlCore = _control.AcceptanceControlCore
AcceptanceControlObserverProxy = _control.AcceptanceControlObserverProxy
acceptance_control_actor_name = _control.acceptance_control_actor_name
event_to_json = _control.event_to_json
run_driver_pair = _orchestrator.run_driver_pair
validate_driver_ready_pair = _orchestrator.validate_driver_ready_pair
validate_generation_iteration_counts = (
    _orchestrator.validate_generation_iteration_counts
)
prepare_preliminary_driver_layout = _orchestrator.prepare_preliminary_driver_layout
load_acceptance_matrix_manifest = _orchestrator.load_acceptance_matrix_manifest
parse_canonical_bundles = _driver.parse_canonical_bundles
configure_role_owned_surface = _driver.configure_role_owned_surface
configure_generation_proof_artifacts = _driver.configure_generation_proof_artifacts
compose_wan_model_driver_config = _driver.compose_wan_model_driver_config
configure_single_pipeline_artifacts = (
    _single_pipeline.configure_single_pipeline_artifacts
)
derive_single_pipeline_names = _single_pipeline.derive_single_pipeline_names
summarize_sealed_batch = _single_pipeline.summarize_sealed_batch
AcceptanceWorkerRecorderMixin = _workers.AcceptanceWorkerRecorderMixin
RecordingEnvWorkerMixin = _workers.RecordingEnvWorkerMixin


def _transition() -> TransitionIdentity:
    return TransitionIdentity(
        worker_rank=0,
        lifecycle_generation=3,
        episode_generation=4,
        transition_id=5,
    )


def _event(**overrides) -> AcceptanceEvent:
    values = {
        "run_id": "run-123",
        "producer_sequence": 0,
        "producer_time_ns": 100,
        "producer_pid": 42,
        "driver_role": "a",
        "pipeline_id": "rlinf_123456789abc",
        "lifecycle_generation": 3,
        "policy_version": 0,
        "component": "environment",
        "dp_rank": 0,
        "event": "chunk_started",
        "transition_identity": _transition(),
        "gpu_ids": (2, 4),
        "details": {"chunk_index": 1},
    }
    values.update(overrides)
    return AcceptanceEvent(**values)


def _batch_manifest(*, lifecycle_generation: int = 3) -> BatchManifest:
    identities = (
        TransitionIdentity(0, lifecycle_generation, 0, 0),
        TransitionIdentity(1, lifecycle_generation, 0, 0),
    )
    transitions = tuple(
        TransitionRecord(
            identity=identity,
            policy_version=7,
            payload={
                "reward": torch.tensor([float(identity.worker_rank)]),
                "done": False,
            },
        )
        for identity in identities
    )
    return BatchManifest(
        pipeline_id="rlinf_123456789abc",
        lifecycle_generation=lifecycle_generation,
        policy_version=7,
        assigned_trajectories_by_rank={0: 1, 1: 1},
        completed_trajectories_by_rank={0: 1, 1: 1},
        expected_transition_identities=identities,
        transitions=transitions,
        expected_actor_ranks=(0, 1),
        actor_batch_ranks=(0, 1),
        final_conditioning={"image_queue": torch.arange(4, dtype=torch.float32)},
    )


def test_run_manifest_freezes_supported_topology_and_thresholds() -> None:
    manifest = RunManifest(
        run_id="run-123",
        environment="wan",
        mode="disaggregated",
        scenario="all",
        expected_bundles=((2, 4), (3, 5)),
        checkpoint_digests={"vla": "abc", "wan": "def"},
    )

    manifest.validate()
    with pytest.raises(ValueError, match="uniform bundle width 1"):
        replace(manifest, mode="collocated").validate()
    with pytest.raises(ValueError, match="disjoint"):
        replace(manifest, expected_bundles=((2, 4), (2, 5))).validate()
    with pytest.raises(ValueError, match="five repetitions"):
        replace(manifest, repetitions=4).validate()
    with pytest.raises(ValueError, match="algorithm='grpo'"):
        replace(manifest, algorithm="gae").validate()
    with pytest.raises(ValueError, match="at least two GRPO iterations"):
        replace(manifest, training_iterations=1).validate()


def test_grpo_training_loop_requires_reward_update_and_policy_reuse() -> None:
    first = GrpoIterationEvidence(0, 3, 3, 8, 8, 8, "grpo", True, 4)
    second = GrpoIterationEvidence(1, 4, 4, 8, 8, 8, "grpo", True, 5)

    assert validate_grpo_training_loop((first, second)) == (first, second)
    with pytest.raises(ValueError, match="incomplete batch or reward"):
        validate_grpo_training_loop((replace(first, rewarded_trajectories=7), second))
    with pytest.raises(ValueError, match="GRPO advantages"):
        validate_grpo_training_loop((replace(first, advantage_type="gae"), second))
    with pytest.raises(ValueError, match="not synchronized"):
        validate_grpo_training_loop(
            (
                first,
                replace(
                    second,
                    collection_policy_version=5,
                    sealed_policy_version=5,
                    produced_policy_version=6,
                ),
            )
        )
    with pytest.raises(ValueError, match="rewards and advantages must be finite"):
        validate_grpo_training_loop((replace(first, rewards_finite=False), second))


def test_acceptance_artifacts_are_atomic_immutable_and_completion_sealed(
    tmp_path: Path,
) -> None:
    manifest = RunManifest(
        run_id="run-123",
        environment="wan",
        mode="disaggregated",
        scenario="all",
        expected_bundles=((2, 4), (3, 5)),
        checkpoint_digests={"vla": "abc", "wan": "def"},
    )
    layout = prepare_acceptance_artifacts(tmp_path, run_manifest=manifest)

    events_path = layout.root / "events.jsonl"
    samples_path = layout.gpu / "samples.csv"
    atomic_write_jsonl(events_path, ({"sequence": 0}, {"sequence": 1}))
    atomic_write_csv(
        samples_path,
        fieldnames=("timestamp_ns", "gpu_id"),
        rows=({"timestamp_ns": 100, "gpu_id": 2},),
    )
    marker_path = write_completion_marker(
        layout,
        relative_artifacts=("events.jsonl", "gpu/samples.csv"),
    )

    assert (layout.root / "run_manifest.json").is_file()
    marker = json.loads(marker_path.read_text())
    assert [entry["path"] for entry in marker["artifacts"]] == [
        "events.jsonl",
        "gpu/samples.csv",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in marker["artifacts"])
    assert verify_completion_marker(layout) == (events_path, samples_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        atomic_write_json(events_path, {"replacement": True})
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_acceptance_artifacts(tmp_path, run_manifest=manifest)
    with pytest.raises(ValueError, match="relative paths"):
        write_completion_marker(layout, relative_artifacts=("../outside",))
    events_path.write_text('{"sequence": 2}\n')
    with pytest.raises(ValueError, match="changed after completion"):
        verify_completion_marker(layout)


def test_acceptance_artifact_layout_rejects_run_id_path_traversal(
    tmp_path: Path,
) -> None:
    manifest = RunManifest(
        run_id="../escaped",
        environment="wan",
        mode="disaggregated",
        scenario="all",
        expected_bundles=((2, 4), (3, 5)),
        checkpoint_digests={"vla": "abc", "wan": "def"},
    )

    with pytest.raises(ValueError, match=r"Got '../escaped'"):
        prepare_acceptance_artifacts(tmp_path, run_manifest=manifest)


def test_two_os_driver_orchestrator_rendezvous_and_isolates_identities(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = RunManifest(
        run_id="run-processes",
        environment="wan",
        mode="disaggregated",
        scenario="recovery",
        expected_bundles=((0, 2), (1, 3)),
        checkpoint_digests={"vla": "abc", "wan": "def"},
    )
    layout = prepare_acceptance_artifacts(tmp_path, run_manifest=manifest)
    script = """
import json
import os
import sys
import time
from pathlib import Path

role = os.environ["RLINF_TASK8_ROLE"]
ready = {
    "scope": "connectivity_only",
    "role": role,
    "pid": os.getpid(),
    "control_plane_actor_id": "shared-control",
    "scheduler_actor_id": "shared-scheduler",
    "pipeline_id": f"pipeline-{role}",
    "pipeline_namespace": f"namespace-{role}",
    "candidate_mapping": {"actor_infer": [0, 1, 2, 3]},
    "candidate_dp_mapping": {"actor_infer": {"0": [0, 2], "1": [1, 3]}},
    "role_names": {"actor": f"actor-{role}", "env": f"env-{role}"},
}
Path(os.environ["RLINF_TASK8_READY"]).write_text(json.dumps(ready), encoding="utf-8")
start = Path(os.environ["RLINF_TASK8_START"])
while not start.is_file():
    time.sleep(0.01)
Path(os.environ["RLINF_TASK8_RESULT"]).write_text(
    json.dumps({
        "status": "passed",
        "role": role,
        "scope": "connectivity_only",
        "task8_accepted": False,
    }),
    encoding="utf-8",
)
print(f"driver {role} completed")
print(f"driver {role} stderr", file=sys.stderr)
"""
    command = (sys.executable, "-c", script)

    results = run_driver_pair(
        layout=layout,
        commands={"a": command, "b": command},
        timeout_s=10.0,
        stream_driver_logs=True,
    )

    assert set(results) == {"a", "b"}
    assert results["a"].pid != results["b"].pid
    assert results["a"].ready["scheduler_actor_id"] == "shared-scheduler"
    assert "driver a completed" in results["a"].stdout_path.read_text()
    assert "driver b stderr" in results["b"].stderr_path.read_text()
    captured = capsys.readouterr()
    assert "[driver a stdout] driver a completed" in captured.out
    assert "[driver b stderr] driver b stderr" in captured.err
    assert (layout.root / "control" / "start.json").is_file()


def test_preliminary_driver_layout_is_explicitly_non_acceptance(
    tmp_path: Path,
) -> None:
    layout = prepare_preliminary_driver_layout(
        tmp_path,
        run_id="control-pair",
        scope="acceptance_control_only",
    )

    manifest = json.loads((layout.root / "preliminary_manifest.json").read_text())
    assert manifest == {
        "run_id": "control-pair",
        "scope": "acceptance_control_only",
        "task8_accepted": False,
    }
    assert (layout.drivers / "a").is_dir()
    assert (layout.drivers / "b").is_dir()
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_preliminary_driver_layout(
            tmp_path,
            run_id="control-pair",
            scope="acceptance_control_only",
        )


def test_generation_pair_requires_all_ten_iterations() -> None:
    passing = {
        role: SimpleNamespace(
            result={"configured_iterations": 10, "completed_iterations": 10}
        )
        for role in ("a", "b")
    }

    validate_generation_iteration_counts(passing, expected_iterations=10)
    failing = {
        **passing,
        "b": SimpleNamespace(
            result={"configured_iterations": 10, "completed_iterations": 9}
        ),
    }
    with pytest.raises(ValueError, match="driver b.*completed=9"):
        validate_generation_iteration_counts(failing, expected_iterations=10)


def test_run_id_validation_rejects_unsafe_artifact_components(
    tmp_path: Path,
) -> None:
    validate_run_id("generation-proof-20260728T120000Z")

    with pytest.raises(ValueError, match=r"Got 'bad/run'"):
        prepare_preliminary_driver_layout(
            tmp_path,
            run_id="bad/run",
            scope="generation_proof_only",
        )
    with pytest.raises(ValueError, match="do not include"):
        derive_role_names(run_id="bad:run", role="a")


def test_control_driver_ready_pair_requires_shared_acceptance_actor() -> None:
    def ready(role: str) -> dict:
        return {
            "scope": "acceptance_control_only",
            "role": role,
            "pid": 1 if role == "a" else 2,
            "acceptance_control_actor_id": "control-actor",
            "event_log_path": "/tmp/task8/events.jsonl",
        }

    pair = {role: ready(role) for role in ("a", "b")}
    validate_driver_ready_pair(pair)
    pair["b"]["acceptance_control_actor_id"] = "other-control-actor"
    with pytest.raises(ValueError, match="different acceptance control"):
        validate_driver_ready_pair(pair)


def test_driver_ready_pair_rejects_different_scheduler() -> None:
    base = {
        "scope": "connectivity_only",
        "role": "a",
        "pid": 1,
        "control_plane_actor_id": "control",
        "scheduler_actor_id": "scheduler-a",
        "pipeline_id": "pipeline-a",
        "pipeline_namespace": "namespace-a",
        "candidate_mapping": {"actor_infer": [0, 1]},
        "candidate_dp_mapping": {"actor_infer": {"0": [0], "1": [1]}},
        "role_names": {"actor": "actor-a"},
    }
    other = {
        **base,
        "role": "b",
        "pid": 2,
        "scheduler_actor_id": "scheduler-b",
        "pipeline_id": "pipeline-b",
        "pipeline_namespace": "namespace-b",
        "role_names": {"actor": "actor-b"},
    }

    with pytest.raises(ValueError, match="different scheduler_actor_id"):
        validate_driver_ready_pair({"a": base, "b": other})


def test_model_init_ready_pair_requires_offloaded_residencies() -> None:
    residency = {
        "component": "actor",
        "rank": 0,
        "model_resident": False,
        "optimizer_resident": False,
        "cuda_graph_captured": False,
        "policy_version": 0,
        "safe_to_release": True,
    }

    def ready(role: str) -> dict:
        return {
            "scope": "model_init_only",
            "role": role,
            "pid": 1 if role == "a" else 2,
            "control_plane_actor_id": "control",
            "scheduler_actor_id": "scheduler",
            "pipeline_id": f"pipeline-{role}",
            "pipeline_namespace": f"namespace-{role}",
            "candidate_mapping": {"actor_infer": [0, 1, 2, 3]},
            "candidate_dp_mapping": {"actor_infer": {"0": [0, 2], "1": [1, 3]}},
            "generation_preemption_mode": "fixed_stage_only",
            "role_names": {"actor": f"actor-{role}"},
            "runtime_state": "inactive",
            "actor_infer_bundles": [[0, 2], [1, 3]],
            "residencies": [dict(residency)],
        }

    pair = {role: ready(role) for role in ("a", "b")}
    validate_driver_ready_pair(pair)
    incompatible = {role: ready(role) for role in ("a", "b")}
    incompatible["b"]["generation_preemption_mode"] = "gap_ratio"
    with pytest.raises(ValueError, match="stage-aware generation"):
        validate_driver_ready_pair(incompatible)
    pair["b"]["residencies"][0]["model_resident"] = True
    pair["b"]["residencies"][0]["safe_to_release"] = False
    with pytest.raises(ValueError, match="accelerator-resident"):
        validate_driver_ready_pair(pair)


def test_connectivity_driver_parses_four_gpu_canonical_bundles() -> None:
    assert parse_canonical_bundles("0,2;1,3", mode="disaggregated") == (
        (0, 2),
        (1, 3),
    )
    assert parse_canonical_bundles("0;1", mode="collocated") == ((0,), (1,))
    with pytest.raises(ValueError, match="width-2"):
        parse_canonical_bundles("0;1", mode="disaggregated")
    with pytest.raises(ValueError, match="disjoint"):
        parse_canonical_bundles("0,1;1,2", mode="disaggregated")


def test_driver_applies_role_owned_worker_output_and_channel_names(
    tmp_path: Path,
) -> None:
    cfg = OmegaConf.create(
        {
            "actor": {"group_name": "ActorGroup"},
            "rollout": {"group_name": "RolloutGroup"},
            "env": {"group_name": "EnvGroup"},
            "runner": {
                "logger": {"log_path": "old", "experiment_name": "old"},
                "per_worker_log_path": "old-workers",
            },
        }
    )
    names = derive_role_names(run_id="run-role", role="b")

    channels = configure_role_owned_surface(
        cfg,
        names=names,
        driver_dir=tmp_path / "driver-b",
    )

    assert cfg.actor.group_name == names.actor_group
    assert cfg.rollout.group_name == names.rollout_group
    assert cfg.env.group_name == names.env_group
    assert cfg.runner.logger.experiment_name == names.prefix
    assert cfg.runner.logger.log_path == str((tmp_path / "driver-b" / "logs").resolve())
    assert cfg.runner.per_worker_log_path == str(
        (tmp_path / "driver-b" / "worker_logs").resolve()
    )
    assert channels == names.runner_channel_names()


def test_generation_driver_installs_observers_on_all_worker_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[object] = []
    diagnostic_flags: list[bool] = []

    class ConfigureObserver:
        @staticmethod
        def remote(observer: object) -> str:
            installed.append(observer)
            return "ok"

    class ConfigureDiagnostics:
        @staticmethod
        def remote(enabled: bool) -> str:
            diagnostic_flags.append(enabled)
            return "ok"

    class FakeWorker:
        configure_acceptance_observer = ConfigureObserver()
        configure_task8_phase_diagnostics = ConfigureDiagnostics()

    def group(size: int) -> SimpleNamespace:
        return SimpleNamespace(
            worker_info_list=[
                SimpleNamespace(rank=rank, worker=FakeWorker()) for rank in range(size)
            ]
        )

    launched = SimpleNamespace(
        actor=group(1),
        rollout=group(2),
        env=group(2),
        runtime=SimpleNamespace(
            pipeline_id="rlinf_123456789abc",
            placement_plan=SimpleNamespace(
                actor_workers=(SimpleNamespace(rank=0, local_gpu=0),),
                actor_infer_bundles=((0, (0, 2)), (1, (1, 3))),
            ),
        ),
    )
    monkeypatch.setattr(_driver.ray, "get", lambda values: values)

    _driver._install_acceptance_worker_observers(
        launched,
        run_id="run-123",
        role="a",
        control_actor=object(),
        phase_diagnostics=True,
    )

    assert len(installed) == 5
    assert all(callable(observer) for observer in installed)
    assert diagnostic_flags == [True, True]


def test_worker_recorder_enriches_events_from_elastic_cursor() -> None:
    observations = []

    class Recorder(AcceptanceWorkerRecorderMixin):
        _elastic_cursor = SimpleNamespace(
            lifecycle_generation=3,
            policy_version=7,
            expected_transition_id={
                "lifecycle_generation": 3,
                "worker_rank": 0,
                "episode_generation": 4,
                "transition_id": 5,
            },
        )

    recorder = Recorder()
    recorder.configure_acceptance_observer(observations.append)

    recorder._record_acceptance("policy_request_started", env_output={"token": "x"})

    assert len(observations) == 1
    assert observations[0].details["lifecycle_generation"] == 3
    assert observations[0].details["policy_version"] == 7
    assert observations[0].details["transition_id"] == {
        "episode_generation": 4,
        "lifecycle_generation": 3,
        "transition_id": 5,
        "worker_rank": 0,
    }


def test_environment_records_bootstrap_only_after_elastic_send_completes() -> None:
    observations = []
    order = []

    class ProductionEnv:
        async def _send_elastic_observation(
            self, rollout_channel, env_output, *, final_bootstrap=False
        ):
            order.append(("send_completed", final_bootstrap))

    class Recorder(RecordingEnvWorkerMixin, ProductionEnv):
        _rollout_cursor = SimpleNamespace(
            lifecycle_generation=3,
            policy_version=7,
            expected_transition_id=_transition(),
        )
        _task8_resume_dispatch_pending = False

    recorder = Recorder()
    recorder.configure_acceptance_observer(
        lambda event: (order.append(event.event), observations.append(event))
    )

    import asyncio

    asyncio.run(
        recorder._send_elastic_observation(
            object(),
            {"transition_id": _transition()},
            final_bootstrap=True,
        )
    )

    assert order == [("send_completed", True), "bootstrap_dispatched"]
    assert observations[0].details["lifecycle_generation"] == 3
    assert observations[0].details["policy_version"] == 7
    assert observations[0].details["final_bootstrap"] is True


def test_environment_records_world_model_diagnostic_phases_without_changing_result() -> (
    None
):
    observations = []
    production_calls = []
    expected = {"observation": torch.tensor([1.0])}

    class FakeWorldEnvironment:
        _chunk_step_diagnostic_observer = None

        def set_chunk_step_diagnostic_observer(self, observer):
            self._chunk_step_diagnostic_observer = observer

        def emit(self, event, **details):
            if self._chunk_step_diagnostic_observer is not None:
                self._chunk_step_diagnostic_observer(event, details)

        def onload(self):
            production_calls.append("onload")

        def _infer_next_chunk_frames(self, actions):
            production_calls.append(("diffusion", actions))

        def _infer_next_chunk_rewards(self):
            production_calls.append("reward")
            return torch.tensor([1.0])

        def chunk_step(self, actions):
            self.onload()
            self._infer_next_chunk_frames(actions)
            self._infer_next_chunk_rewards()
            self.emit("reward_returned")
            self.emit("done_check_completed", has_past_dones=False)
            self.emit("chunk_step_returning")
            return expected

    environment = FakeWorldEnvironment()

    class ProductionEnv:
        env_list = [environment]

        def env_interact_step(self, chunk_actions, stage_id):
            return self.env_list[stage_id].chunk_step(chunk_actions)

    class Recorder(RecordingEnvWorkerMixin, ProductionEnv):
        _rollout_cursor = SimpleNamespace(
            lifecycle_generation=3,
            policy_version=7,
            expected_transition_id=_transition(),
        )

    recorder = Recorder()
    recorder.configure_acceptance_observer(observations.append)
    recorder.configure_task8_phase_diagnostics(True)
    actions = torch.tensor([2.0])

    result = recorder.env_interact_step(actions, 0)

    assert result is expected
    assert production_calls == ["onload", ("diffusion", actions), "reward"]
    assert [event.event for event in observations] == [
        "chunk_started",
        "world_model_onload_started",
        "world_model_onload_completed",
        "world_model_diffusion_started",
        "world_model_diffusion_completed",
        "world_model_reward_started",
        "world_model_reward_completed",
        "world_model_reward_returned",
        "world_model_done_check_completed",
        "world_model_chunk_step_returning",
        "env_output_constructed",
        "chunk_committed",
    ]
    assert all(
        event.details["elapsed_seconds"] >= 0
        for event in observations
        if event.event
        in {
            "world_model_onload_completed",
            "world_model_diffusion_completed",
            "world_model_reward_completed",
        }
    )
    assert "onload" not in environment.__dict__
    assert "_infer_next_chunk_frames" not in environment.__dict__
    assert "_infer_next_chunk_rewards" not in environment.__dict__
    assert environment._chunk_step_diagnostic_observer is None


def test_environment_phase_diagnostics_can_be_disabled() -> None:
    observations = []

    class ProductionEnv:
        def env_interact_step(self, chunk_actions, stage_id):
            return {"actions": chunk_actions, "stage_id": stage_id}

    class Recorder(RecordingEnvWorkerMixin, ProductionEnv):
        pass

    recorder = Recorder()
    recorder.configure_acceptance_observer(observations.append)
    recorder.configure_task8_phase_diagnostics(False)

    result = recorder.env_interact_step(torch.tensor([2.0]), 0)

    assert result["stage_id"] == 0
    assert [event.event for event in observations] == [
        "chunk_started",
        "chunk_committed",
    ]


def test_four_gpu_wan_driver_config_composes_two_rank_topology(tmp_path: Path) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_wan_disaggregated.yaml"
    )
    names = derive_role_names(run_id="run-config", role="a")

    def pure_validator(cfg):
        validate_elastic_vla_config(cfg)
        return cfg

    cfg, channels = compose_wan_model_driver_config(
        config_path,
        names=names,
        driver_dir=tmp_path / "driver-a",
        validator=pure_validator,
    )

    assert OmegaConf.to_container(cfg.cluster.component_placement) == {
        "actor": "0",
        "rollout": "0,1",
        "env": "2,3",
    }
    assert cfg.env.train.total_num_envs == 16
    assert cfg.env.train.rollout_epoch == 1
    assert cfg.env.train.get("stop_rank_when_all_done", False) is False
    assert cfg.env.train.max_episode_steps == 256
    assert cfg.env.train.max_steps_per_rollout_epoch == 256
    assert cfg.env.train.num_inference_steps == 5
    assert cfg.algorithm.adv_type == "grpo"
    assert cfg.algorithm.group_size == 8
    assert cfg.algorithm.reward_type == "action_level"
    assert cfg.algorithm.filter_rewards is True
    assert cfg.algorithm.rewards_lower_bound == 0.0
    assert cfg.algorithm.rewards_upper_bound == 5.0
    assert cfg.rlix.monitor_poll_interval_s == 0.1
    assert cfg.actor.group_name == names.actor_group
    assert channels == names.runner_channel_names()


def test_four_rank_fsdp_variant_composes_all_gpu_actor_world(tmp_path: Path) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_wan_disaggregated_four_rank_fsdp.yaml"
    )
    names = derive_role_names(run_id="four-rank-fsdp", role="a")

    def pure_validator(cfg):
        validate_elastic_vla_config(cfg)
        return cfg

    cfg, _ = compose_wan_model_driver_config(
        config_path,
        names=names,
        driver_dir=tmp_path / "driver-a",
        validator=pure_validator,
    )

    assert OmegaConf.to_container(cfg.cluster.component_placement) == {
        "actor": "0,1,2,3",
        "rollout": "0,1",
        "env": "2,3",
    }
    assert cfg.actor.training_backend == "fsdp"
    assert (
        OmegaConf.select(cfg, "actor.model.tensor_model_parallel_size", default=1) == 1
    )
    assert (
        OmegaConf.select(cfg, "actor.model.pipeline_model_parallel_size", default=1)
        == 1
    )
    assert cfg.actor.global_batch_size == 16
    assert cfg.actor.micro_batch_size == 1
    assert cfg.env.train.total_num_envs == 16
    assert cfg.env.train.rollout_epoch == 4
    assert cfg.env.train.stop_rank_when_all_done is True
    assert OmegaConf.load(config_path).smoke.completed_bundle_handoff == (
        "release_before_training"
    )


def test_single_pipeline_diagnostic_preserves_task8_config_and_enables_artifacts(
    tmp_path: Path,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_wan_disaggregated.yaml"
    )
    names = derive_single_pipeline_names(run_id="single-control")

    def pure_validator(cfg):
        validate_elastic_vla_config(cfg)
        return cfg

    cfg, channels = compose_wan_model_driver_config(
        config_path,
        names=names,
        driver_dir=tmp_path / "single-control",
        validator=pure_validator,
    )
    trajectory_dir = configure_single_pipeline_artifacts(
        cfg, run_dir=tmp_path / "single-control"
    )
    validate_rank_early_completion_config(cfg)

    assert names.prefix == "t8_single-control_single"
    assert channels == names.runner_channel_names()
    assert OmegaConf.to_container(cfg.cluster.component_placement) == {
        "actor": "0",
        "rollout": "0,1",
        "env": "2,3",
    }
    assert cfg.runner.max_steps == 10
    assert cfg.env.train.total_num_envs == 16
    assert cfg.env.train.group_size == 8
    assert cfg.env.train.rollout_epoch == 1
    assert cfg.env.train.max_episode_steps == 256
    assert cfg.env.train.max_steps_per_rollout_epoch == 256
    assert cfg.env.train.num_inference_steps == 5
    assert cfg.algorithm.adv_type == "grpo"
    assert cfg.algorithm.filter_rewards is True
    assert cfg.algorithm.rewards_lower_bound == 0.0
    assert cfg.algorithm.rewards_upper_bound == 5.0
    assert cfg.env.train.stop_rank_when_all_done is True
    assert cfg.env.train.video_cfg.save_video is True
    assert Path(cfg.env.train.video_cfg.video_base_dir) == trajectory_dir / "videos"
    assert Path(cfg.runner.task8_single_pipeline_artifact_dir) == trajectory_dir


def test_wan_driver_config_rejects_non_boolean_rank_early_completion(
    tmp_path: Path,
) -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "e2e_tests"
        / "embodied"
        / "task8_wan_disaggregated.yaml"
    )
    source = OmegaConf.load(source_path)
    source.smoke.stop_rank_when_all_done = "yes"
    config_path = tmp_path / "invalid-rank-early-completion.yaml"
    OmegaConf.save(source, config_path)

    with pytest.raises(
        ValueError, match="smoke.stop_rank_when_all_done must be a boolean"
    ):
        compose_wan_model_driver_config(
            config_path,
            names=derive_role_names(run_id="invalid-early-completion", role="a"),
            driver_dir=tmp_path / "driver-a",
            validator=lambda cfg: cfg,
        )


def test_generation_proof_enables_role_local_trajectory_videos(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "train": {
                    "video_cfg": {
                        "save_video": False,
                        "video_base_dir": "old",
                        "info_on_video": False,
                    }
                }
            }
        }
    )

    trajectory_dir = configure_generation_proof_artifacts(
        cfg, driver_dir=tmp_path / "drivers" / "a"
    )

    assert trajectory_dir == (tmp_path / "drivers" / "a" / "trajectories").resolve()
    assert cfg.env.train.video_cfg.save_video is True
    assert Path(cfg.env.train.video_cfg.video_base_dir) == trajectory_dir / "videos"
    assert cfg.env.train.video_cfg.info_on_video is True


def test_single_pipeline_reward_summary_detects_all_zero_filtered_batch() -> None:
    batch = {
        "rewards": torch.zeros((2, 4, 3), dtype=torch.bfloat16),
        "loss_mask": torch.zeros((2, 4, 3), dtype=torch.bool),
        "terminations": torch.zeros((2, 4, 3), dtype=torch.bool),
        "truncations": torch.tensor(
            [
                [[False] * 3 for _ in range(4)],
                [[False, False, True] for _ in range(4)],
            ]
        ),
        "versions": torch.zeros((2, 4, 3), dtype=torch.int64),
    }

    summary = summarize_sealed_batch(
        batch,
        group_size=2,
        filter_rewards=True,
        rewards_lower_bound=0.5,
        rewards_upper_bound=4.5,
    )

    assert summary["reward_all_zero"] is True
    assert summary["trajectory_reward_sums"] == [0.0, 0.0, 0.0, 0.0]
    assert summary["reward_filter"]["accepted_groups"] == [False, False]
    assert summary["reward_filter"]["all_groups_filtered"] is True
    assert summary["post_filter_active_trajectory_count"] == 0
    assert summary["terminal_counts"] == {
        "terminations": 0,
        "truncations": 4,
    }
    assert summary["policy_versions"] == [0]


def test_single_pipeline_names_reject_unsafe_run_id() -> None:
    with pytest.raises(ValueError, match="run_id must be"):
        derive_single_pipeline_names(run_id="../unsafe")


@pytest.mark.parametrize(
    ("filename", "environment", "mode", "bundles", "training_iterations"),
    (
        (
            "task8_wan_disaggregated.yaml",
            "wan",
            "disaggregated",
            ((0, 2), (1, 3)),
            10,
        ),
        ("task8_wan_collocated.yaml", "wan", "collocated", ((0,), (1,)), 2),
        (
            "task8_opensora_disaggregated.yaml",
            "opensora",
            "disaggregated",
            ((0, 2), (1, 3)),
            2,
        ),
        (
            "task8_opensora_collocated.yaml",
            "opensora",
            "collocated",
            ((0,), (1,)),
            2,
        ),
    ),
)
def test_acceptance_matrix_configs_freeze_matching_manifests(
    filename: str,
    environment: str,
    mode: str,
    bundles: tuple[tuple[int, ...], ...],
    training_iterations: int,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "e2e_tests" / "embodied" / filename
    )

    manifest = load_acceptance_matrix_manifest(
        config_path,
        run_id="matrix-preflight",
        environment=environment,
        mode=mode,
        scenario="all",
        checkpoint_digests={"vla": "vla-digest", environment: "wm-digest"},
    )

    assert manifest.expected_bundles == bundles
    assert manifest.algorithm == "grpo"
    assert manifest.training_iterations == training_iterations
    assert manifest.repetitions == 5


def test_acceptance_matrix_preflight_rejects_cli_and_placement_drift(
    tmp_path: Path,
) -> None:
    config = OmegaConf.create(
        {
            "acceptance": {
                "environment": "wan",
                "mode": "collocated",
                "expected_bundles": [[0], [1]],
            },
            "smoke": {
                "rollout_gpu": [0, 1],
                "env_gpu": [0, 2],
                "adv_type": "grpo",
                "max_train_steps": 2,
            },
        }
    )
    path = tmp_path / "matrix.yaml"
    OmegaConf.save(config, path)

    with pytest.raises(ValueError, match="identical rollout and env"):
        load_acceptance_matrix_manifest(
            path,
            run_id="matrix-preflight",
            environment="wan",
            mode="collocated",
            scenario="all",
            checkpoint_digests={"vla": "vla", "wan": "wan"},
        )
    with pytest.raises(ValueError, match="requested environment"):
        load_acceptance_matrix_manifest(
            path,
            run_id="matrix-preflight",
            environment="opensora",
            mode="collocated",
            scenario="all",
            checkpoint_digests={"vla": "vla", "opensora": "opensora"},
        )


def test_role_names_are_deterministic_and_collision_free() -> None:
    names_a = derive_role_names(run_id="run-123", role="a")
    names_b = derive_role_names(run_id="run-123", role="b")

    assert names_a == derive_role_names(run_id="run-123", role="a")
    assert names_a.prefix == "t8_run-123_a"
    assert names_a.actor_group == "t8_run-123_a_ActorGroup"
    assert names_a.runner_channel_names() == {
        "env": "t8_run-123_a_env_input",
        "rollout": "t8_run-123_a_rollout_request",
        "actor": "t8_run-123_a_actor_batch",
    }
    assert {
        names_a.actor_group,
        names_a.rollout_group,
        names_a.env_group,
        names_a.env_input_channel,
        names_a.rollout_request_channel,
        names_a.actor_channel,
        names_a.event_producer,
    }.isdisjoint(
        {
            names_b.actor_group,
            names_b.rollout_group,
            names_b.env_group,
            names_b.env_input_channel,
            names_b.rollout_request_channel,
            names_b.actor_channel,
            names_b.event_producer,
        }
    )
    with pytest.raises(ValueError, match="role must be"):
        derive_role_names(run_id="run-123", role="core")


def test_event_sink_assigns_total_order_and_rejects_producer_regression() -> None:
    timestamps = iter((1000, 1001))
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: next(timestamps))

    first = sink.record(_event())
    second = sink.record(
        _event(
            producer_sequence=1,
            event="chunk_committed",
            producer_time_ns=200,
        )
    )

    assert (first.sink_sequence, first.sink_time_ns) == (0, 1000)
    assert (second.sink_sequence, second.sink_time_ns) == (1, 1001)
    assert sink.events == (first, second)
    with pytest.raises(ValueError, match="increase strictly"):
        sink.record(_event(producer_sequence=1))


@pytest.mark.parametrize(
    "event_name",
    ("barrier_send_started", "barrier_send_completed", "barrier_send_failed"),
)
def test_event_sink_accepts_barrier_phase_events_with_transition_identity(
    event_name: str,
) -> None:
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: 1000)
    event = _event(event=event_name)

    stamped = sink.record(event)

    assert stamped.event == event_name
    assert stamped.transition_identity == _transition()
    assert stamped.sink_sequence == 0


@pytest.mark.parametrize(
    "event_name",
    ("barrier_send_started", "barrier_send_completed", "barrier_send_failed"),
)
def test_barrier_phase_events_require_transition_identity(event_name: str) -> None:
    with pytest.raises(ValueError, match="requires transition_identity"):
        _event(event=event_name, transition_identity=None).validate()


def test_event_producer_enriches_identity_and_advances_only_after_sink() -> None:
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: 500)
    producer_times = iter((100, 101))
    attempts = 0

    def flaky_record(event):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("sink unavailable")
        return sink.record(event)

    producer = AcceptanceEventProducer(
        run_id="run-123",
        driver_role="a",
        component="environment",
        dp_rank=0,
        context_provider=lambda: AcceptanceProducerContext(
            pipeline_id="rlinf_123456789abc",
            lifecycle_generation=3,
            policy_version=0,
            transition_identity=_transition(),
            gpu_ids=(2, 4),
        ),
        sink_record=flaky_record,
        clock_ns=lambda: next(producer_times),
        producer_pid=42,
    )
    observation = SimpleNamespace(event="chunk_started", details={"chunk_index": 1})

    with pytest.raises(RuntimeError, match="sink unavailable"):
        producer.record(observation)
    stamped = producer.record(observation)

    assert stamped.producer_sequence == 0
    assert stamped.producer_time_ns == 101
    assert stamped.transition_identity == _transition()
    assert stamped.gpu_ids == (2, 4)
    assert stamped.sink_sequence == 0


def test_event_producer_derives_transition_context_from_details() -> None:
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: 500)
    producer = AcceptanceEventProducer(
        run_id="run-123",
        driver_role="a",
        component="rollout",
        dp_rank=0,
        context_provider=lambda: AcceptanceProducerContext(
            pipeline_id="rlinf_123456789abc",
            lifecycle_generation=None,
            policy_version=None,
            transition_identity=None,
            gpu_ids=(2, 4),
        ),
        sink_record=sink.record,
        clock_ns=lambda: 100,
        producer_pid=42,
    )

    stamped = producer.record(
        SimpleNamespace(
            event="policy_request_started",
            details={
                "request": {
                    "transition_id": {
                        "lifecycle_generation": 3,
                        "env_worker_rank": 0,
                        "episode_generation": 4,
                        "transition_id": 5,
                    },
                    "policy_version": 7,
                }
            },
        )
    )

    assert stamped.lifecycle_generation == 3
    assert stamped.policy_version == 7
    assert stamped.transition_identity == _transition()


def test_event_producer_accepts_worker_rank_transition_context() -> None:
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: 500)
    producer = AcceptanceEventProducer(
        run_id="run-123",
        driver_role="a",
        component="environment",
        dp_rank=0,
        context_provider=lambda: AcceptanceProducerContext(
            pipeline_id="rlinf_123456789abc",
            lifecycle_generation=None,
            policy_version=None,
            transition_identity=None,
            gpu_ids=(2, 4),
        ),
        sink_record=sink.record,
        clock_ns=lambda: 100,
        producer_pid=42,
    )

    stamped = producer.record(
        SimpleNamespace(
            event="chunk_started",
            details={
                "transition_id": {
                    "lifecycle_generation": 3,
                    "worker_rank": 0,
                    "episode_generation": 4,
                    "transition_id": 5,
                },
                "policy_version": 7,
            },
        )
    )

    assert stamped.lifecycle_generation == 3
    assert stamped.policy_version == 7
    assert stamped.transition_identity == _transition()


def test_event_producer_accepts_production_rollout_transition_context() -> None:
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: 500)
    producer = AcceptanceEventProducer(
        run_id="run-123",
        driver_role="a",
        component="environment",
        dp_rank=1,
        context_provider=lambda: AcceptanceProducerContext(
            pipeline_id="rlinf_123456789abc",
            lifecycle_generation=None,
            policy_version=0,
            transition_identity=None,
            gpu_ids=(1, 3),
        ),
        sink_record=sink.record,
        clock_ns=lambda: 100,
        producer_pid=42,
    )

    stamped = producer.record(
        SimpleNamespace(
            event="bootstrap_dispatched",
            details={
                "transition_id": {
                    "lifecycle_generation": 1,
                    "env_worker_rank": 1,
                    "stage_id": 0,
                    "sequence": 0,
                }
            },
        )
    )

    assert stamped.lifecycle_generation == 1
    assert stamped.policy_version == 0
    assert stamped.transition_identity == TransitionIdentity(
        lifecycle_generation=1,
        worker_rank=1,
        episode_generation=0,
        transition_id=0,
    )


@pytest.mark.parametrize("transition_ranks", ((0, 1), (1, 0)))
def test_actor_batch_seal_validates_all_transitions_without_selecting_one(
    transition_ranks: tuple[int, int],
) -> None:
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: 500)
    producer = AcceptanceEventProducer(
        run_id="run-123",
        driver_role="a",
        component="actor",
        dp_rank=0,
        context_provider=lambda: AcceptanceProducerContext(
            pipeline_id="rlinf_123456789abc",
            lifecycle_generation=None,
            policy_version=None,
            transition_identity=None,
            gpu_ids=(0,),
        ),
        sink_record=sink.record,
        clock_ns=lambda: 100,
        producer_pid=42,
    )
    transitions = [
        {
            "lifecycle_generation": 3,
            "env_worker_rank": rank,
            "stage_id": 0,
            "sequence": index,
        }
        for index, rank in enumerate(transition_ranks)
    ]

    stamped = producer.record(
        SimpleNamespace(
            event="batch_sealed",
            details={
                "receipt": {
                    "lifecycle_generation": 3,
                    "policy_version": 7,
                    "contributing_dp_ranks": [0, 1],
                    "transition_count": 2,
                },
                "transition_ids": transitions,
            },
        )
    )

    assert stamped.dp_rank == 0
    assert stamped.transition_identity is None
    assert stamped.lifecycle_generation == 3
    assert stamped.policy_version == 7


def test_actor_batch_seal_rejects_incomplete_transition_rank_coverage() -> None:
    event = _event(
        event="batch_sealed",
        component="actor",
        dp_rank=0,
        lifecycle_generation=3,
        policy_version=7,
        transition_identity=None,
        gpu_ids=(0,),
        details={
            "receipt": {
                "lifecycle_generation": 3,
                "policy_version": 7,
                "contributing_dp_ranks": [0, 1],
                "transition_count": 1,
            },
            "transition_ids": [
                {
                    "lifecycle_generation": 3,
                    "env_worker_rank": 1,
                    "stage_id": 0,
                    "sequence": 0,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="source ranks"):
        event.validate()


def test_batch_seal_rejects_singular_top_level_transition_identity() -> None:
    with pytest.raises(ValueError, match="aggregate evidence"):
        _event(event="batch_sealed").validate()


def test_event_sink_rejects_central_clock_regression() -> None:
    timestamps = iter((1001, 1000))
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: next(timestamps))
    sink.record(_event())

    with pytest.raises(ValueError, match="clock must not regress"):
        sink.record(_event(producer_sequence=1, event="chunk_committed"))


def test_event_validation_requires_rank_bundle_and_transition_identity() -> None:
    with pytest.raises(ValueError, match="non-negative dp_rank"):
        replace(_event(), dp_rank=None).validate()
    with pytest.raises(ValueError, match="requires transition_identity"):
        replace(_event(), transition_identity=None).validate()
    with pytest.raises(ValueError, match="unique non-negative"):
        replace(_event(), gpu_ids=(2, 2)).validate()
    with pytest.raises(ValueError, match="worker rank"):
        replace(
            _event(),
            transition_identity=replace(_transition(), worker_rank=1),
        ).validate()
    with pytest.raises(ValueError, match="non-empty GPU bundle"):
        _event(
            event="allocation_committed",
            transition_identity=None,
            gpu_ids=(),
        ).validate()


def test_event_sink_rejects_duplicate_terminal_event() -> None:
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: 1000)
    committed = _event(event="chunk_committed")
    sink.record(committed)

    with pytest.raises(ValueError, match="duplicate terminal"):
        sink.record(replace(committed, producer_sequence=1))


def test_event_sink_accepts_training_completion_for_each_policy_version() -> None:
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: 1000)
    first = _event(
        event="training_completed",
        component="actor",
        transition_identity=None,
        lifecycle_generation=None,
        policy_version=0,
        details={"policy_version": 0},
    )

    sink.record(first)
    second = sink.record(
        replace(
            first,
            producer_sequence=1,
            policy_version=1,
            details={"policy_version": 1},
        )
    )

    assert second.policy_version == 1


def _control_core(tmp_path: Path) -> AcceptanceControlCore:
    timestamps = iter(range(1000, 1100))
    return AcceptanceControlCore(
        AcceptanceControlConfig(
            run_id="run-123",
            event_log_path=str(tmp_path / "control" / "events.jsonl"),
            target_bundle=(2, 4),
            target_rank=0,
        ),
        clock_ns=lambda: next(timestamps),
        sleep=lambda _: None,
    )


def test_acceptance_control_persists_total_order_and_observation_gates(
    tmp_path: Path,
) -> None:
    control = _control_core(tmp_path)

    initialized_a = control.record_event(
        _event(
            event="initialized",
            transition_identity=None,
            gpu_ids=(),
            dp_rank=None,
        )
    )
    assert initialized_a.sink_sequence == 0
    assert control.gate_status()["both_drivers_initialized"] is False

    initialized_b = control.record_event(
        _event(
            producer_sequence=0,
            driver_role="b",
            event="initialized",
            transition_identity=None,
            gpu_ids=(),
            dp_rank=None,
        )
    )
    chunk_started = control.record_event(_event(producer_sequence=1))
    bootstrap_dispatched = control.record_event(
        _event(event="bootstrap_dispatched", producer_sequence=2)
    )
    allocation = control.record_event(
        _event(
            producer_sequence=1,
            driver_role="b",
            event="allocation_committed",
            transition_identity=None,
            gpu_ids=(2, 4),
        )
    )
    released = control.record_event(
        _event(
            producer_sequence=2,
            driver_role="b",
            event="release_committed",
            transition_identity=None,
            gpu_ids=(2, 4),
        )
    )
    resumed = control.record_event(
        _event(
            producer_sequence=3,
            event="resumed_bootstrap_dispatched",
        )
    )

    assert control.events() == (
        initialized_a,
        initialized_b,
        chunk_started,
        bootstrap_dispatched,
        allocation,
        released,
        resumed,
    )
    gate_status = control.gate_status()
    assert (
        gate_status
        | {
            "both_drivers_initialized": True,
            "allow_a_collection": False,
            "a_target_bootstrap_dispatched": True,
            "a_target_chunk_started": True,
            "allow_b_demand": False,
            "transfer_to_b_observed": True,
            "allow_b_release": False,
            "b_release_observed": True,
            "a_resume_observed": True,
        }
        == gate_status
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "control" / "events.jsonl").read_text().splitlines()
    ]
    assert [row["sink_sequence"] for row in rows] == list(range(7))
    assert rows[-1]["event"] == "resumed_bootstrap_dispatched"


def test_acceptance_control_tracks_generation_proof_gates(tmp_path: Path) -> None:
    control = _control_core(tmp_path)

    control.record_event(
        _event(
            event="policy_synchronized",
            producer_sequence=0,
            driver_role="b",
            component="runner",
            dp_rank=None,
            lifecycle_generation=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )
    control.record_event(
        _event(
            event="policy_synchronized",
            producer_sequence=0,
            component="runner",
            dp_rank=None,
            lifecycle_generation=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )
    control.record_event(
        _event(
            event="rank_completed",
            producer_sequence=3,
            component="rollout",
        )
    )
    control.record_event(
        _event(
            event="generation_requested",
            producer_sequence=1,
            component="runner",
            dp_rank=None,
            lifecycle_generation=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )
    control.record_event(
        _event(
            event="generation_granted",
            producer_sequence=2,
            component="runner",
            dp_rank=None,
            transition_identity=None,
            gpu_ids=(2, 4),
        )
    )
    control.record_event(
        _event(
            event="generation_requested",
            producer_sequence=1,
            driver_role="b",
            pipeline_id="rlinf_b123456789ab",
            component="runner",
            dp_rank=None,
            lifecycle_generation=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )
    control.record_event(
        _event(
            event="generation_granted",
            producer_sequence=2,
            driver_role="b",
            pipeline_id="rlinf_b123456789ab",
            component="runner",
            dp_rank=None,
            transition_identity=None,
            gpu_ids=(2, 4),
        )
    )
    control.record_event(
        _event(
            event="batch_sealed",
            producer_sequence=3,
            component="runner",
            dp_rank=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )
    control.record_event(
        _event(
            event="batch_sealed",
            producer_sequence=3,
            driver_role="b",
            pipeline_id="rlinf_b123456789ab",
            component="runner",
            dp_rank=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )
    control.record_event(
        _event(
            event="training_started",
            producer_sequence=4,
            component="runner",
            dp_rank=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )
    control.record_event(
        _event(
            event="training_completed",
            producer_sequence=5,
            component="runner",
            dp_rank=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )
    control.record_event(
        _event(
            event="training_started",
            producer_sequence=4,
            driver_role="b",
            pipeline_id="rlinf_b123456789ab",
            component="runner",
            dp_rank=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )
    control.record_event(
        _event(
            event="training_completed",
            producer_sequence=5,
            driver_role="b",
            pipeline_id="rlinf_b123456789ab",
            component="runner",
            dp_rank=None,
            transition_identity=None,
            gpu_ids=(),
        )
    )

    assert control.gate_status()["b_policy_sync_completed"] is True
    assert control.gate_status()["a_policy_sync_completed"] is True
    assert control.gate_status()["a_generation_requested"] is True
    assert control.gate_status()["a_generation_granted"] is True
    assert control.gate_status()["a_first_rank_completed"] is True
    assert control.gate_status()["a_target_rank_completed"] is True
    assert control.gate_status()["b_generation_requested"] is True
    assert control.gate_status()["transfer_to_b_observed"] is True
    assert control.gate_status()["both_batches_sealed"] is True
    assert control.gate_status()["a_batch_sealed"] is True
    assert control.gate_status()["b_batch_sealed"] is True
    assert control.gate_status()["a_training_started"] is True
    assert control.gate_status()["b_training_started"] is True
    assert control.gate_status()["a_training_completed"] is True
    assert control.gate_status()["b_training_completed"] is True
    assert control.gate_status()["both_training_completed"] is True


def test_generation_proof_queues_b_before_a_target_rank_completes(monkeypatch) -> None:
    calls = []

    class RemoteMethod:
        def __init__(self, operation):
            self.operation = operation

        def remote(self, gate, **kwargs):
            calls.append((self.operation, gate))
            return (self.operation, gate)

    control_actor = SimpleNamespace(
        wait_for_gate=RemoteMethod("wait"),
        release_gate=RemoteMethod("release"),
    )
    monkeypatch.setattr(_orchestrator.ray, "get", lambda value: value)

    _orchestrator._drive_generation_proof_gates(control_actor, timeout_s=1.0)

    assert calls.index(("release", "allow_b_collection")) < calls.index(
        ("wait", "b_generation_requested")
    )
    assert calls.index(("wait", "b_generation_requested")) < calls.index(
        ("wait", "a_target_rank_completed")
    )
    assert ("release", "allow_a_training") not in calls
    assert calls.index(("wait", "b_useful_work_observed")) < calls.index(
        ("wait", "a_training_started")
    )
    assert calls.index(("wait", "a_training_completed")) < calls.index(
        ("wait", "b_batch_sealed")
    )
    assert ("release", "allow_b_training") not in calls


def test_generation_proof_driver_has_no_first_iteration_training_barrier() -> None:
    gate_names = {
        constant
        for constant in _driver.run_generation_proof_driver.__code__.co_consts
        if isinstance(constant, str)
    }

    assert "allow_b_training" not in gate_names
    assert "both_training_completed" not in gate_names


def test_first_rank_completion_gate_does_not_require_configured_target(
    tmp_path: Path,
) -> None:
    control = _control_core(tmp_path)

    control.record_event(
        _event(
            event="rank_completed",
            component="rollout",
            dp_rank=1,
            transition_identity=None,
            gpu_ids=(7, 9),
        )
    )

    assert control.gate_status()["a_first_rank_completed"] is True
    assert control.gate_status()["a_target_rank_completed"] is False


def test_acceptance_control_explicit_gates_and_fail_closed_driver_loss(
    tmp_path: Path,
) -> None:
    control = _control_core(tmp_path)

    assert control.release_gate("allow_a_collection")["allow_a_collection"] is True
    assert control.wait_for_gate("allow_a_collection", timeout_s=0.01)
    with pytest.raises(ValueError, match="unknown acceptance gate"):
        control.release_gate("not-a-gate")

    control.fail(role="b", error_type="RuntimeError", error="driver exited")
    with pytest.raises(RuntimeError, match="driver exited"):
        control.wait_for_gate("allow_b_demand", timeout_s=0.01)
    with pytest.raises(RuntimeError, match="driver exited"):
        control.record_event(_event())


def test_acceptance_control_rejects_duplicate_without_appending(
    tmp_path: Path,
) -> None:
    control = _control_core(tmp_path)
    committed = _event(event="chunk_committed")

    control.record_event(committed)
    with pytest.raises(ValueError, match="duplicate terminal"):
        control.record_event(replace(committed, producer_sequence=1))

    lines = (tmp_path / "control" / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "chunk_committed"


def test_acceptance_control_rejection_includes_event_context(tmp_path: Path) -> None:
    control = _control_core(tmp_path)
    invalid = _event(
        event="policy_request_started",
        transition_identity=None,
        policy_version=None,
        details={"env_output": {"kind": "missing-transition"}},
    )

    with pytest.raises(ValueError) as exc_info:
        control.record_event(invalid)

    message = str(exc_info.value)
    assert "acceptance event rejected" in message
    assert "event='policy_request_started'" in message
    assert "component='environment'" in message
    assert "policy_version=None" in message
    assert "requires transition_identity" in message
    assert "detail_keys=('env_output',)" in message


def test_acceptance_control_observer_proxy_submits_worker_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[AcceptanceEvent] = []

    class RemoteRecord:
        @staticmethod
        def remote(event: AcceptanceEvent) -> AcceptanceEvent:
            recorded.append(event)
            return replace(event, sink_sequence=0, sink_time_ns=500)

    class FakeControlActor:
        record_event = RemoteRecord()

    monkeypatch.setattr(_control.ray, "get", lambda value: value)
    proxy = AcceptanceControlObserverProxy(
        run_id="run-123",
        driver_role="a",
        component="environment",
        dp_rank=0,
        context_provider=lambda: AcceptanceProducerContext(
            pipeline_id="rlinf_123456789abc",
            lifecycle_generation=3,
            policy_version=0,
            transition_identity=_transition(),
            gpu_ids=(2, 4),
        ),
        control_actor=FakeControlActor(),
    )

    stamped = proxy(SimpleNamespace(event="chunk_started", details={"chunk_index": 1}))

    assert stamped.sink_sequence == 0
    assert recorded[0].producer_sequence == 0
    assert recorded[0].driver_role == "a"
    assert event_to_json(stamped)["sink_time_ns"] == 500


def test_acceptance_control_actor_name_is_run_owned() -> None:
    assert acceptance_control_actor_name(run_id="run-123") == (
        "task8_acceptance_control_run-123"
    )
    with pytest.raises(ValueError, match="run_id"):
        acceptance_control_actor_name(run_id="bad run")


def test_manifest_normalization_is_typed_stable_and_tolerance_aware() -> None:
    value = {
        "reward": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "done": np.asarray([False, True]),
        "identity": _transition(),
    }
    first = normalize_manifest(value)
    second = normalize_manifest(value)

    assert first == second
    assert first["reward"]["shape"] == [2]
    assert first["reward"]["dtype"] == "float32"
    assert first["reward"]["logical_bytes"] == 8
    close = normalize_manifest(
        {**value, "reward": torch.tensor([1.00001, 2.0], dtype=torch.float32)}
    )
    assert compare_manifests(first, close, rtol=1e-4, atol=1e-6) == ()
    assert compare_manifests(first, close) == ("$.reward.values[0]",)
    with pytest.raises(ValueError, match="non-finite scalar"):
        normalize_manifest({"reward": float("nan")})
    with pytest.raises(ValueError, match="non-finite tensor"):
        normalize_manifest(torch.tensor([float("inf")]))


def test_manifest_normalization_preserves_bfloat16_evidence() -> None:
    value = torch.tensor([1.0, -2.5], dtype=torch.bfloat16)

    first = normalize_manifest(value)
    second = normalize_manifest(value.clone())

    assert first == second
    assert first["shape"] == [2]
    assert first["dtype"] == "bfloat16"
    assert first["logical_bytes"] == 4
    assert first["values"] == [1.0, -2.5]
    assert logical_tensor_bytes({"policy_result": value}) == 4
    with pytest.raises(ValueError, match="non-finite tensor"):
        normalize_manifest(torch.tensor([float("inf")], dtype=torch.bfloat16))


def test_manifest_rejects_cuda_state_and_counts_nested_tensor_bytes() -> None:
    state = {
        "observation": torch.zeros((2, 3), dtype=torch.float32),
        "queue": [np.ones((4,), dtype=np.int16)],
    }
    assert logical_tensor_bytes(state) == 32
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="CPU tensors"):
            normalize_manifest(torch.zeros(1, device="cuda"))


def test_batch_manifest_accepts_complete_single_version_batch() -> None:
    manifest = _batch_manifest()

    manifest.validate()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda manifest: replace(
                manifest,
                completed_trajectories_by_rank={0: 1, 1: 0},
            ),
            "incomplete trajectory",
        ),
        (
            lambda manifest: replace(manifest, sealed=False),
            "must be sealed",
        ),
        (
            lambda manifest: replace(
                manifest,
                transitions=(manifest.transitions[0], manifest.transitions[0]),
            ),
            "duplicate transition",
        ),
        (
            lambda manifest: replace(
                manifest,
                transitions=manifest.transitions[:1],
            ),
            "transition identity mismatch",
        ),
        (
            lambda manifest: replace(manifest, actor_batch_ranks=(0,)),
            "contributions are incomplete",
        ),
        (
            lambda manifest: replace(
                manifest,
                transitions=(
                    replace(manifest.transitions[0], policy_version=8),
                    manifest.transitions[1],
                ),
            ),
            "policy version",
        ),
    ],
)
def test_batch_manifest_rejects_partial_duplicate_or_mixed_evidence(
    mutation, message
) -> None:
    with pytest.raises(ValueError, match=message):
        mutation(_batch_manifest()).validate()


def test_reference_comparison_normalizes_only_lifecycle_generation() -> None:
    reference = _batch_manifest(lifecycle_generation=3)
    recovery = replace(
        _batch_manifest(lifecycle_generation=4),
        pipeline_id="rlinf_bbbbbbbbbbbb",
    )

    assert compare_reference_batches(reference, recovery, rtol=1e-5, atol=1e-6) == ()
    changed_transition = replace(
        recovery.transitions[0],
        payload={"reward": torch.tensor([0.1]), "done": False},
    )
    changed = replace(
        recovery,
        transitions=(changed_transition, recovery.transitions[1]),
    )
    assert compare_reference_batches(reference, changed, rtol=1e-5, atol=1e-6)


def test_snapshot_analysis_reports_fields_and_rejects_model_sized_payload() -> None:
    snapshot = {
        "current_obs": torch.zeros((2, 3), dtype=torch.float32),
        "image_queue": [torch.ones((4,), dtype=torch.float32)],
        "metadata": {"next_transition": 2},
    }

    report = analyze_snapshot_size(
        snapshot,
        encoded_bytes=80,
        continuation_ceiling_bytes=1024,
        field_ceiling_bytes=512,
    )

    assert report.logical_bytes == 40
    assert report.field_logical_bytes == {
        "current_obs": 24,
        "image_queue": 16,
        "metadata": 0,
    }
    with pytest.raises(ValueError, match="model-parameter-sized"):
        analyze_snapshot_size(
            snapshot,
            encoded_bytes=80,
            continuation_ceiling_bytes=1024,
            field_ceiling_bytes=20,
        )


def _transfer_evidence():
    sink = AcceptanceEventSink(run_id="run-123", clock_ns=lambda: 1000)
    pipeline_a = "rlinf_aaaaaaaaaaaa"
    pipeline_b = "rlinf_bbbbbbbbbbbb"
    committed = _transition()
    resumed = replace(committed, transition_id=6)
    sibling = TransitionIdentity(1, 3, 4, 8)
    specifications = [
        (pipeline_a, 0, "chunk_started", committed, (2, 4)),
        (pipeline_b, 0, "request_enqueued", None, ()),
        (pipeline_a, 0, "drain_requested", committed, (2, 4)),
        (pipeline_a, 0, "chunk_committed", committed, (2, 4)),
        (pipeline_a, 1, "chunk_committed", sibling, (3, 5)),
        (pipeline_a, 0, "snapshot_completed", resumed, (2, 4)),
        (pipeline_a, 0, "environment_offload_verified", None, (2, 4)),
        (pipeline_a, 0, "rollout_offload_verified", None, (2, 4)),
        (pipeline_a, 0, "release_committed", None, (2, 4)),
        (pipeline_b, 0, "allocation_committed", None, (2, 4)),
        (pipeline_b, 0, "useful_cuda_work_started", None, (2, 4)),
        (pipeline_b, 0, "release_committed", None, (2, 4)),
        (pipeline_a, 0, "allocation_committed", None, (2, 4)),
        (pipeline_a, 0, "environment_onload_verified", None, (2, 4)),
        (pipeline_a, 0, "rollout_onload_verified", None, (2, 4)),
        (pipeline_a, 0, "restore_validated", resumed, (2, 4)),
        (pipeline_a, 0, "resumed_bootstrap_dispatched", resumed, (2, 4)),
    ]
    for sequence, (pipeline_id, rank, event, transition, gpu_ids) in enumerate(
        specifications
    ):
        sink.record(
            _event(
                pipeline_id=pipeline_id,
                producer_sequence=sequence,
                driver_role="a" if pipeline_id == pipeline_a else "b",
                dp_rank=rank,
                event=event,
                transition_identity=transition,
                gpu_ids=gpu_ids,
            )
        )
    allocations = (
        AllocationSlice(pipeline_a, 0, (2, 4), 0, 100),
        AllocationSlice(pipeline_b, 0, (2, 4), 100, 200),
        AllocationSlice(pipeline_a, 0, (2, 4), 200, 300),
        AllocationSlice(pipeline_a, 1, (3, 5), 0, 300),
    )
    return (
        sink.events,
        allocations,
        pipeline_a,
        pipeline_b,
        committed,
        resumed,
    )


def test_transfer_analyzer_proves_exact_a_to_b_to_a_handoff() -> None:
    events, allocations, pipeline_a, pipeline_b, committed, resumed = (
        _transfer_evidence()
    )

    timeline = validate_transfer_timeline(
        events,
        allocations,
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
        selected_rank=0,
        sibling_rank=1,
        bundle=(2, 4),
        committed_transition=committed,
        resumed_transition=resumed,
    )

    assert timeline.a_release_committed < timeline.b_allocation_committed
    assert timeline.b_release_committed < timeline.a_reallocation_committed
    assert timeline.a_restore_validated < timeline.a_resume_dispatched
    assert timeline.a_drain_requested < timeline.sibling_chunk_committed


def test_transfer_analyzer_rejects_allocation_before_release() -> None:
    events, allocations, pipeline_a, pipeline_b, committed, resumed = (
        _transfer_evidence()
    )
    events = list(events)
    events[8] = replace(events[8], sink_sequence=9)
    events[9] = replace(events[9], sink_sequence=8)

    with pytest.raises(ValueError, match="invalid A-to-B-to-A transfer order"):
        validate_transfer_timeline(
            events,
            allocations,
            pipeline_a=pipeline_a,
            pipeline_b=pipeline_b,
            selected_rank=0,
            sibling_rank=1,
            bundle=(2, 4),
            committed_transition=committed,
            resumed_transition=resumed,
        )


def _commit_payload(**operations):
    payload = {"shrinks": [], "removes": [], "allocates": [], "expands": []}
    payload.update(operations)
    return payload


def test_scheduler_commit_markers_normalize_exact_registered_bundles() -> None:
    pipeline_a = "rlinf_aaaaaaaaaaaa"
    pipeline_b = "rlinf_bbbbbbbbbbbb"
    cluster_a = f"{pipeline_a}_actor_infer"
    cluster_b = f"{pipeline_b}_actor_infer"
    tracked = {cluster_a: pipeline_a, cluster_b: pipeline_b}
    bundles = {pipeline_a: {0: (2, 4)}, pipeline_b: {0: (2, 4)}}

    release = normalize_scheduler_commit_marker(
        marker_name="Commit C7",
        timestamp_ns=150,
        payload=_commit_payload(
            shrinks=[{"cluster_id": cluster_a, "dp_rank": 0, "gpus_freed": [4, 2]}]
        ),
        tracked_clusters=tracked,
        canonical_bundles=bundles,
    )
    allocation = normalize_scheduler_commit_marker(
        marker_name="Commit C8",
        timestamp_ns=200,
        payload=_commit_payload(
            expands=[
                {
                    "cluster_id": cluster_b,
                    "gpus_allocated": [2, 4],
                    "dp_ranks_added": [0],
                    "dp_rank_to_gpus": {"0": [2, 4]},
                }
            ]
        ),
        tracked_clusters=tracked,
        canonical_bundles=bundles,
    )

    assert release == (
        SchedulerCommitRecord(7, 150, "release", cluster_a, pipeline_a, 0, (2, 4)),
    )
    assert allocation == (
        SchedulerCommitRecord(8, 200, "allocation", cluster_b, pipeline_b, 0, (2, 4)),
    )
    with pytest.raises(ValueError, match="not a scheduler commit marker"):
        normalize_scheduler_commit_marker(
            marker_name="Exec C8",
            timestamp_ns=190,
            payload=_commit_payload(),
            tracked_clusters=tracked,
            canonical_bundles=bundles,
        )
    with pytest.raises(ValueError, match="bundle mismatch"):
        normalize_scheduler_commit_marker(
            marker_name="Commit C9",
            timestamp_ns=210,
            payload=_commit_payload(
                expands=[
                    {
                        "cluster_id": cluster_b,
                        "gpus_allocated": [2, 5],
                        "dp_ranks_added": [0],
                        "dp_rank_to_gpus": {0: [2, 5]},
                    }
                ]
            ),
            tracked_clusters=tracked,
            canonical_bundles=bundles,
        )


def _perfetto_csv(*rows: tuple[int, str, dict]) -> str:
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("timestamp_ns", "marker_name", "commit_json"))
    for timestamp_ns, marker_name, payload in rows:
        writer.writerow((timestamp_ns, marker_name, json.dumps(payload)))
    return output.getvalue()


def test_perfetto_csv_parser_extracts_only_post_commit_rank_evidence() -> None:
    pipeline_a = "rlinf_aaaaaaaaaaaa"
    pipeline_b = "rlinf_bbbbbbbbbbbb"
    cluster_a = f"{pipeline_a}_actor_infer"
    cluster_b = f"{pipeline_b}_actor_infer"
    tracked = {cluster_a: pipeline_a, cluster_b: pipeline_b}
    bundles = {pipeline_a: {0: (2, 4)}, pipeline_b: {0: (2, 4)}}
    csv_text = _perfetto_csv(
        (
            150,
            "Commit C7",
            _commit_payload(
                shrinks=[
                    {
                        "cluster_id": cluster_a,
                        "dp_rank": 0,
                        "gpus_freed": [2, 4],
                    }
                ]
            ),
        ),
        (
            200,
            "Commit C8",
            _commit_payload(
                expands=[
                    {
                        "cluster_id": cluster_b,
                        "gpus_allocated": [2, 4],
                        "dp_ranks_added": [0],
                        "dp_rank_to_gpus": {"0": [2, 4]},
                    }
                ]
            ),
        ),
    )

    records = parse_perfetto_commit_csv(
        csv_text,
        tracked_clusters=tracked,
        canonical_bundles=bundles,
    )

    assert tuple(record.operation for record in records) == (
        "release",
        "allocation",
    )
    assert tuple(record.cycle_counter for record in records) == (7, 8)
    with pytest.raises(ValueError, match="increase by timestamp and cycle"):
        parse_perfetto_commit_csv(
            _perfetto_csv(
                (
                    200,
                    "Commit C8",
                    _commit_payload(
                        expands=[
                            {
                                "cluster_id": cluster_b,
                                "gpus_allocated": [2, 4],
                                "dp_ranks_added": [0],
                                "dp_rank_to_gpus": {0: [2, 4]},
                            }
                        ]
                    ),
                ),
                (
                    150,
                    "Commit C7",
                    _commit_payload(
                        shrinks=[
                            {
                                "cluster_id": cluster_a,
                                "dp_rank": 0,
                                "gpus_freed": [2, 4],
                            }
                        ]
                    ),
                ),
            ),
            tracked_clusters=tracked,
            canonical_bundles=bundles,
        )


def test_perfetto_extractor_checks_tool_trace_and_command_failure(
    tmp_path: Path,
) -> None:
    pipeline_a = "rlinf_aaaaaaaaaaaa"
    cluster_a = f"{pipeline_a}_actor_infer"
    tracked = {cluster_a: pipeline_a}
    bundles = {pipeline_a: {0: (2, 4)}}
    trace_path = tmp_path / "scheduler.perfetto-trace"
    trace_path.write_bytes(b"trace")
    csv_text = _perfetto_csv(
        (
            150,
            "Commit C7",
            _commit_payload(
                shrinks=[
                    {
                        "cluster_id": cluster_a,
                        "dp_rank": 0,
                        "gpus_freed": [2, 4],
                    }
                ]
            ),
        )
    )
    calls = []

    def successful_runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=csv_text, stderr="")

    records = extract_scheduler_commits_from_perfetto(
        trace_path,
        trace_processor_shell="/bin/true",
        tracked_clusters=tracked,
        canonical_bundles=bundles,
        runner=successful_runner,
    )

    assert records[0].operation == "release"
    assert calls[0][0][0] == "/bin/true"
    assert calls[0][0][1:3] == ["--csv", "-Q"]
    assert calls[0][0][-1] == str(trace_path)
    assert calls[0][1] == {
        "capture_output": True,
        "text": True,
        "check": False,
    }

    def failed_runner(command, **kwargs):
        return SimpleNamespace(returncode=2, stdout="", stderr="bad trace")

    with pytest.raises(RuntimeError, match="bad trace"):
        extract_scheduler_commits_from_perfetto(
            trace_path,
            trace_processor_shell="/bin/true",
            tracked_clusters=tracked,
            canonical_bundles=bundles,
            runner=failed_runner,
        )
    with pytest.raises(ValueError, match="unavailable"):
        extract_scheduler_commits_from_perfetto(
            trace_path,
            trace_processor_shell=tmp_path / "missing-processor",
            tracked_clusters=tracked,
            canonical_bundles=bundles,
        )


def test_scheduler_commits_correlate_with_worker_residency_events() -> None:
    pipeline_a = "rlinf_aaaaaaaaaaaa"
    pipeline_b = "rlinf_bbbbbbbbbbbb"
    cluster_a = f"{pipeline_a}_actor_infer"
    cluster_b = f"{pipeline_b}_actor_infer"
    bundle = (2, 4)
    commits = (
        SchedulerCommitRecord(7, 150, "release", cluster_a, pipeline_a, 0, bundle),
        SchedulerCommitRecord(8, 200, "allocation", cluster_b, pipeline_b, 0, bundle),
        SchedulerCommitRecord(9, 300, "release", cluster_b, pipeline_b, 0, bundle),
        SchedulerCommitRecord(10, 400, "allocation", cluster_a, pipeline_a, 0, bundle),
    )
    specifications = (
        (pipeline_a, "snapshot_completed", 100),
        (pipeline_a, "environment_offload_verified", 110),
        (pipeline_a, "rollout_offload_verified", 120),
        (pipeline_b, "useful_cuda_work_started", 250),
        (pipeline_a, "environment_onload_verified", 410),
        (pipeline_a, "rollout_onload_verified", 420),
    )
    events = tuple(
        _event(
            pipeline_id=pipeline_id,
            driver_role="a" if pipeline_id == pipeline_a else "b",
            producer_sequence=sequence,
            producer_time_ns=timestamp_ns,
            event=event_name,
            transition_identity=(
                _transition() if event_name == "snapshot_completed" else None
            ),
            gpu_ids=bundle,
        )
        for sequence, (pipeline_id, event_name, timestamp_ns) in enumerate(
            specifications
        )
    )

    correlation = correlate_scheduler_transfer_commits(
        events,
        commits,
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
        selected_rank=0,
        bundle=bundle,
    )

    assert correlation.a_release_ns == 150
    assert correlation.a_reallocation_ns == 400
    early_release = (replace(commits[0], commit_time_ns=115), *commits[1:])
    with pytest.raises(ValueError, match="before snapshot and verified offload"):
        correlate_scheduler_transfer_commits(
            events,
            early_release,
            pipeline_a=pipeline_a,
            pipeline_b=pipeline_b,
            selected_rank=0,
            bundle=bundle,
        )


def test_analysis_summary_is_generated_only_from_complete_raw_evidence() -> None:
    events, allocations, pipeline_a, pipeline_b, committed, resumed = (
        _transfer_evidence()
    )
    timeline = validate_transfer_timeline(
        events,
        allocations,
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
        selected_rank=0,
        sibling_rank=1,
        bundle=(2, 4),
        committed_transition=committed,
        resumed_transition=resumed,
    )
    manifest = RunManifest(
        run_id="run-123",
        environment="wan",
        mode="disaggregated",
        scenario="all",
        expected_bundles=((2, 4), (3, 5)),
        checkpoint_digests={"vla": "abc", "wan": "def"},
    )
    snapshot = analyze_snapshot_size(
        {"state": torch.ones(4)},
        encoded_bytes=32,
        continuation_ceiling_bytes=1024,
        field_ceiling_bytes=512,
    )
    gpu_summaries = {
        gpu_id: GpuUtilizationSummary(gpu_id, 100, 50.0, 20) for gpu_id in (2, 3, 4, 5)
    }
    utilization = evaluate_utilization_trials(
        [
            trial
            for repetition in range(5)
            for trial in (
                UtilizationTrial(repetition, "static", 100, 4, 10),
                UtilizationTrial(repetition, "dynamic", 110, 4, 10),
            )
        ]
    )
    artifacts = {
        "events": "events.jsonl",
        "scheduler": "core/scheduler_timeline.jsonl",
        "gpu_samples": "gpu/samples.csv",
        "reference": "reference/manifest.json",
    }

    summary, markdown = build_analysis_summary(
        run_manifest=manifest,
        reference_differences=(),
        transfer_timeline=timeline,
        snapshot_report=snapshot,
        gpu_summaries=gpu_summaries,
        utilization=utilization,
        raw_artifacts=artifacts,
    )

    assert summary["status"] == "passed"
    assert summary["scope"]["intra_shard_idleness_solved"] is False
    assert "Status: **PASSED**" in markdown
    assert "events.jsonl" in markdown
    with pytest.raises(ValueError, match="raw_artifacts"):
        build_analysis_summary(
            run_manifest=manifest,
            reference_differences=(),
            transfer_timeline=timeline,
            snapshot_report=snapshot,
            gpu_summaries=gpu_summaries,
            utilization=utilization,
            raw_artifacts={"events": "events.jsonl"},
        )


def test_artifact_set_enforces_idle_gate_and_emits_complete_report(
    tmp_path: Path,
) -> None:
    events, allocations, pipeline_a, pipeline_b, committed, resumed = (
        _transfer_evidence()
    )
    timeline = validate_transfer_timeline(
        events,
        allocations,
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
        selected_rank=0,
        sibling_rank=1,
        bundle=(2, 4),
        committed_transition=committed,
        resumed_transition=resumed,
    )
    manifest = RunManifest(
        run_id="run-final",
        environment="wan",
        mode="disaggregated",
        scenario="all",
        expected_bundles=((2, 4), (3, 5)),
        checkpoint_digests={"vla": "abc", "wan": "def"},
    )
    layout = prepare_acceptance_artifacts(tmp_path, run_manifest=manifest)
    raw_artifacts = {
        "events": "events.jsonl",
        "scheduler": "core/scheduler_timeline.jsonl",
        "gpu_samples": "gpu/samples.csv",
        "reference": "reference/manifest.json",
    }
    atomic_write_jsonl(layout.root / raw_artifacts["events"], ({"event": "raw"},))
    atomic_write_jsonl(
        layout.root / raw_artifacts["scheduler"], ({"operation": "release"},)
    )
    atomic_write_csv(
        layout.root / raw_artifacts["gpu_samples"],
        fieldnames=("timestamp_ns", "gpu_id"),
        rows=({"timestamp_ns": 1, "gpu_id": 2},),
    )
    atomic_write_json(layout.root / raw_artifacts["reference"], {"sealed": True})
    write_completion_marker(layout, relative_artifacts=tuple(raw_artifacts.values()))
    snapshot = analyze_snapshot_size(
        {"state": torch.ones(4)},
        encoded_bytes=32,
        continuation_ceiling_bytes=1024,
        field_ceiling_bytes=512,
    )
    summaries = {
        gpu_id: GpuUtilizationSummary(gpu_id, 100, 50.0, 20) for gpu_id in (2, 3, 4, 5)
    }
    trials = tuple(
        trial
        for repetition in range(5)
        for trial in (
            UtilizationTrial(repetition, "static", 100, 4, 10, idle_fraction=0.5),
            UtilizationTrial(repetition, "dynamic", 110, 4, 10, idle_fraction=0.4),
        )
    )
    metadata = AcceptanceReportMetadata(
        root_commit="root-sha",
        rlinf_commit="rlinf-sha",
        dirty_worktree=False,
        hardware=("4 x NVIDIA test GPU",),
        commands=("run-task8",),
        limitations=("Test fixture only.",),
    )
    training = {
        pipeline_id: (
            GrpoIterationEvidence(0, 3, 3, 8, 8, 8, "grpo", True, 4),
            GrpoIterationEvidence(1, 4, 4, 8, 8, 8, "grpo", True, 5),
        )
        for pipeline_id in (pipeline_a, pipeline_b)
    }

    finalizer_kwargs = {
        "layout": layout,
        "reference_differences": (),
        "transfer_timeline": timeline,
        "snapshot_report": snapshot,
        "gpu_summaries": summaries,
        "utilization_trials": trials,
        "training_iterations": training,
        "latencies_ms": {"safe_point": (1.0, 2.0, 3.0)},
        "raw_artifacts": raw_artifacts,
        "metadata": metadata,
    }
    with pytest.raises(ValueError, match="does not match the stored manifest"):
        emit_acceptance_artifact_set(
            run_manifest=replace(manifest, minimum_throughput_improvement=0.06),
            **finalizer_kwargs,
        )
    with pytest.raises(ValueError, match="exactly one static and dynamic pair"):
        emit_acceptance_artifact_set(
            run_manifest=manifest,
            **{**finalizer_kwargs, "utilization_trials": (*trials, *trials[-2:])},
        )
    summary, report = emit_acceptance_artifact_set(
        run_manifest=manifest,
        **finalizer_kwargs,
    )

    assert summary["status"] == "passed"
    assert "Median direct-GPU idle reduction: 20.00%" in report
    assert "## Provenance" in report
    assert "## Batch and policy versions" in report
    assert summary["training_iterations"][pipeline_a][1]["produced_policy_version"] == 5
    assert {path.name for path in layout.analysis.iterdir()} == {
        "REPORT.md",
        "correctness.json",
        "latency_summary.csv",
        "ownership.json",
        "utilization_repetitions.csv",
        "utilization_summary.json",
    }


def test_idle_reduction_is_a_fail_closed_utilization_gate() -> None:
    trials = [
        trial
        for repetition in range(5)
        for trial in (
            UtilizationTrial(repetition, "static", 100, 4, 10, idle_fraction=0.5),
            UtilizationTrial(repetition, "dynamic", 110, 4, 10, idle_fraction=0.49),
        )
    ]

    result = evaluate_utilization_trials(
        trials,
        minimum_median_idle_reduction=0.05,
    )

    assert result.median_improvement == pytest.approx(0.1)
    assert result.median_idle_reduction == pytest.approx(0.02)
    assert result.passed is False
    with pytest.raises(ValueError, match="idle_fraction is required"):
        evaluate_utilization_trials(
            [replace(trial, idle_fraction=None) for trial in trials],
            minimum_median_idle_reduction=0.05,
        )
    with pytest.raises(ValueError, match="idle_fraction must be within"):
        evaluate_utilization_trials(
            [replace(trial, idle_fraction=float("nan")) for trial in trials],
            minimum_median_idle_reduction=0.05,
        )


def test_ownership_validation_accepts_handoff_boundary_and_rejects_overlap() -> None:
    safe = [
        AllocationSlice("pipeline-a", 0, (2, 4), 0, 100),
        AllocationSlice("pipeline-b", 0, (2, 4), 100, 200),
        AllocationSlice("pipeline-a", 1, (3, 5), 0, 200),
    ]
    validate_exclusive_ownership(safe)

    with pytest.raises(ValueError, match="GPU ownership overlap"):
        validate_exclusive_ownership(
            [*safe, AllocationSlice("pipeline-b", 0, (3,), 150, 180)]
        )


def test_gpu_samples_use_time_weighted_integration_and_reject_gaps() -> None:
    samples = [
        GpuSample(0, 2, 0.0, 10.0, 100),
        GpuSample(10, 2, 50.0, 20.0, 200),
        GpuSample(30, 2, 100.0, 30.0, 300),
    ]
    summary = summarize_gpu_samples(
        samples,
        max_gap_ns=20,
        idle_threshold=5.0,
    )[2]

    assert summary.duration_ns == 30
    assert summary.mean_sm_utilization == pytest.approx(1000 / 30)
    assert summary.idle_ns == 10
    with pytest.raises(ValueError, match="sample gap"):
        summarize_gpu_samples(samples, max_gap_ns=19, idle_threshold=5.0)


def test_gpu_samples_fail_on_sampler_error_or_duplicate_timestamp() -> None:
    with pytest.raises(ValueError, match="sampler reported"):
        summarize_gpu_samples(
            [
                GpuSample(0, 0, 0, 0, 0, error="NVML unavailable"),
                GpuSample(1, 0, 0, 0, 0),
            ],
            max_gap_ns=10,
            idle_threshold=5,
        )
    with pytest.raises(ValueError, match="timestamps must increase"):
        summarize_gpu_samples(
            [GpuSample(0, 0, 0, 0, 0), GpuSample(0, 0, 0, 0, 0)],
            max_gap_ns=10,
            idle_threshold=5,
        )


def test_utilization_acceptance_enforces_paired_material_improvement() -> None:
    trials = []
    for repetition in range(5):
        trials.extend(
            [
                UtilizationTrial(repetition, "static", 100, 2, 10.0),
                UtilizationTrial(repetition, "dynamic", 110, 2, 10.0),
            ]
        )

    result = evaluate_utilization_trials(trials)

    assert result.passed
    assert result.median_improvement == pytest.approx(0.1)
    assert result.improved_repetitions == 5


def test_utilization_failed_dynamic_run_counts_as_zero_and_fails_gate() -> None:
    trials = []
    for repetition in range(5):
        trials.extend(
            [
                UtilizationTrial(repetition, "static", 100, 2, 10.0),
                UtilizationTrial(
                    repetition,
                    "dynamic",
                    110,
                    2,
                    10.0,
                    correctness_passed=repetition != 4,
                ),
            ]
        )

    result = evaluate_utilization_trials(trials)

    assert not result.passed
    assert result.paired_improvements[-1] == -1.0


def test_utilization_rejects_missing_or_duplicate_pairs() -> None:
    with pytest.raises(ValueError, match="static and dynamic"):
        evaluate_utilization_trials(
            [UtilizationTrial(0, "static", 1, 1, 1)],
            minimum_pairs=1,
            minimum_improved_pairs=1,
        )
    with pytest.raises(ValueError, match="duplicate static"):
        evaluate_utilization_trials(
            [
                UtilizationTrial(0, "static", 1, 1, 1),
                UtilizationTrial(0, "static", 1, 1, 1),
            ],
            minimum_pairs=1,
            minimum_improved_pairs=1,
        )
