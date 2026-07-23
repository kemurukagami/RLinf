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
atomic_write_csv = _artifacts.atomic_write_csv
atomic_write_json = _artifacts.atomic_write_json
atomic_write_jsonl = _artifacts.atomic_write_jsonl
parse_perfetto_commit_csv = _artifacts.parse_perfetto_commit_csv
prepare_acceptance_artifacts = _artifacts.prepare_acceptance_artifacts
verify_completion_marker = _artifacts.verify_completion_marker
write_completion_marker = _artifacts.write_completion_marker


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

    with pytest.raises(ValueError, match="one safe output-directory component"):
        prepare_acceptance_artifacts(tmp_path, run_manifest=manifest)


def test_role_names_are_deterministic_and_collision_free() -> None:
    names_a = derive_role_names(run_id="run-123", role="a")
    names_b = derive_role_names(run_id="run-123", role="b")

    assert names_a == derive_role_names(run_id="run-123", role="a")
    assert names_a.prefix == "t8_run-123_a"
    assert names_a.actor_group == "t8_run-123_a_ActorGroup"
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
