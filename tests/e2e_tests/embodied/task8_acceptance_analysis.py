"""Pure correctness and transfer analysis for Task 8 acceptance artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping

from task8_acceptance_support import (
    AcceptanceEvent,
    AllocationSlice,
    GpuUtilizationSummary,
    RunManifest,
    SchedulerCommitRecord,
    TransitionIdentity,
    UtilizationAcceptance,
    compare_manifests,
    logical_tensor_bytes,
    normalize_manifest,
    validate_exclusive_ownership,
)


@dataclass(frozen=True, slots=True)
class TransitionRecord:
    """One ordered transition and its normalized semantic payload."""

    identity: TransitionIdentity
    policy_version: int
    payload: Any

    def validate(self, *, lifecycle_generation: int, policy_version: int) -> None:
        """Validate identity, lifecycle, version, and payload normalization."""
        self.identity.validate()
        if self.identity.lifecycle_generation != lifecycle_generation:
            raise ValueError("transition lifecycle generation does not match batch")
        if self.policy_version != policy_version:
            raise ValueError("transition policy version does not match batch")
        normalize_manifest(self.payload)


@dataclass(frozen=True, slots=True)
class BatchManifest:
    """Pre-training evidence for one complete sealed actor batch."""

    pipeline_id: str
    lifecycle_generation: int
    policy_version: int
    assigned_trajectories_by_rank: Mapping[int, int]
    completed_trajectories_by_rank: Mapping[int, int]
    expected_transition_identities: tuple[TransitionIdentity, ...]
    transitions: tuple[TransitionRecord, ...]
    expected_actor_ranks: tuple[int, ...]
    actor_batch_ranks: tuple[int, ...]
    final_conditioning: Any
    sealed: bool = True

    def validate(self) -> None:
        """Reject partial, duplicate, stale, or mixed-version batch evidence."""
        if not self.pipeline_id:
            raise ValueError("batch pipeline_id must not be empty")
        if self.lifecycle_generation < 0 or self.policy_version < 0:
            raise ValueError("batch lifecycle and policy version must be non-negative")
        assigned = dict(self.assigned_trajectories_by_rank)
        completed = dict(self.completed_trajectories_by_rank)
        if not assigned or set(assigned) != set(completed):
            raise ValueError("assigned and completed rank sets must be identical")
        if any(
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 0
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for rank, count in assigned.items()
        ):
            raise ValueError(
                "assigned trajectory counts must use non-negative integers"
            )
        if any(completed[rank] != count for rank, count in assigned.items()):
            raise ValueError("batch contains incomplete trajectory assignments")
        if not self.sealed:
            raise ValueError("batch must be sealed before acceptance or training")

        expected_ids = self.expected_transition_identities
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("expected transition identities contain duplicates")
        for identity in expected_ids:
            identity.validate()
            if identity.lifecycle_generation != self.lifecycle_generation:
                raise ValueError("expected transition has a stale lifecycle generation")
            if identity.worker_rank not in assigned:
                raise ValueError("expected transition belongs to an unassigned rank")

        actual_ids = tuple(record.identity for record in self.transitions)
        if len(actual_ids) != len(set(actual_ids)):
            raise ValueError("batch contains duplicate transition identities")
        missing = set(expected_ids) - set(actual_ids)
        extra = set(actual_ids) - set(expected_ids)
        if missing or extra:
            raise ValueError(
                "batch transition identity mismatch: "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )
        if actual_ids != expected_ids:
            raise ValueError("batch transitions are not in expected order")
        for record in self.transitions:
            record.validate(
                lifecycle_generation=self.lifecycle_generation,
                policy_version=self.policy_version,
            )

        expected_actor_ranks = self.expected_actor_ranks
        actor_batch_ranks = self.actor_batch_ranks
        if not expected_actor_ranks or len(expected_actor_ranks) != len(
            set(expected_actor_ranks)
        ):
            raise ValueError("expected actor ranks must be non-empty and unique")
        if len(actor_batch_ranks) != len(set(actor_batch_ranks)):
            raise ValueError("actor batch contributions contain duplicates")
        if set(actor_batch_ranks) != set(expected_actor_ranks):
            raise ValueError("actor batch contributions are incomplete")
        normalize_manifest(self.final_conditioning)


@dataclass(frozen=True, slots=True)
class GrpoIterationEvidence:
    """One complete collection, reward, GRPO update, and produced policy."""

    iteration: int
    collection_policy_version: int
    sealed_policy_version: int
    expected_trajectories: int
    received_trajectories: int
    rewarded_trajectories: int
    advantage_type: str
    actor_update_completed: bool
    produced_policy_version: int
    rewards_finite: bool = True
    advantages_finite: bool = True

    def validate(self) -> None:
        """Reject partial rewards, mixed versions, or non-GRPO updates."""
        integer_fields = (
            self.iteration,
            self.collection_policy_version,
            self.sealed_policy_version,
            self.expected_trajectories,
            self.received_trajectories,
            self.rewarded_trajectories,
            self.produced_policy_version,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in integer_fields
        ):
            raise ValueError("GRPO iteration fields must be non-negative integers")
        if self.expected_trajectories <= 0:
            raise ValueError("GRPO iteration requires trajectories")
        if not (
            self.expected_trajectories
            == self.received_trajectories
            == self.rewarded_trajectories
        ):
            raise ValueError("GRPO iteration has incomplete batch or reward evidence")
        if self.sealed_policy_version != self.collection_policy_version:
            raise ValueError("sealed batch policy version does not match collection")
        if self.advantage_type != "grpo":
            raise ValueError("Task 8 training evidence must use GRPO advantages")
        if not self.rewards_finite or not self.advantages_finite:
            raise ValueError("Task 8 GRPO rewards and advantages must be finite")
        if not self.actor_update_completed:
            raise ValueError("GRPO actor update did not complete")
        if self.produced_policy_version != self.collection_policy_version + 1:
            raise ValueError("GRPO update must produce the next policy version")


def validate_grpo_training_loop(
    iterations: Iterable[GrpoIterationEvidence],
) -> tuple[GrpoIterationEvidence, ...]:
    """Prove repeated GRPO training and next-collection policy synchronization."""
    ordered = tuple(iterations)
    if len(ordered) < 2:
        raise ValueError("Task 8 requires at least two linked GRPO iterations")
    for index, iteration in enumerate(ordered):
        iteration.validate()
        if iteration.iteration != index:
            raise ValueError("GRPO iteration indices must be contiguous from zero")
        if index and (
            iteration.collection_policy_version
            != ordered[index - 1].produced_policy_version
        ):
            raise ValueError(
                "updated GRPO policy was not synchronized into the next collection"
            )
    return ordered


def compare_reference_batches(
    reference: BatchManifest,
    recovery: BatchManifest,
    *,
    rtol: float,
    atol: float,
) -> tuple[str, ...]:
    """Compare recovery with reference after lifecycle-only normalization."""
    reference.validate()
    recovery.validate()
    differences: list[str] = []
    if dict(reference.assigned_trajectories_by_rank) != dict(
        recovery.assigned_trajectories_by_rank
    ):
        differences.append("$.assigned_trajectories_by_rank")
    if dict(reference.completed_trajectories_by_rank) != dict(
        recovery.completed_trajectories_by_rank
    ):
        differences.append("$.completed_trajectories_by_rank")
    if reference.policy_version != recovery.policy_version:
        differences.append("$.policy_version")
    if reference.expected_actor_ranks != recovery.expected_actor_ranks:
        differences.append("$.expected_actor_ranks")
    if reference.actor_batch_ranks != recovery.actor_batch_ranks:
        differences.append("$.actor_batch_ranks")

    reference_transitions = _normalized_transitions(reference.transitions)
    recovery_transitions = _normalized_transitions(recovery.transitions)
    differences.extend(
        compare_manifests(
            normalize_manifest(reference_transitions),
            normalize_manifest(recovery_transitions),
            rtol=rtol,
            atol=atol,
        )
    )
    differences.extend(
        compare_manifests(
            normalize_manifest(reference.final_conditioning),
            normalize_manifest(recovery.final_conditioning),
            rtol=rtol,
            atol=atol,
        )
    )
    return tuple(differences)


def _normalized_transitions(
    transitions: tuple[TransitionRecord, ...],
) -> tuple[TransitionRecord, ...]:
    return tuple(
        replace(
            record,
            identity=replace(record.identity, lifecycle_generation=0),
        )
        for record in transitions
    )


@dataclass(frozen=True, slots=True)
class SnapshotSizeReport:
    """Validated logical and encoded continuation-state sizes."""

    logical_bytes: int
    encoded_bytes: int
    field_logical_bytes: Mapping[str, int]


def analyze_snapshot_size(
    snapshot: Mapping[str, Any],
    *,
    encoded_bytes: int,
    continuation_ceiling_bytes: int,
    field_ceiling_bytes: int,
) -> SnapshotSizeReport:
    """Reject empty, GPU-resident, oversized, or model-sized snapshots."""
    if encoded_bytes <= 0:
        raise ValueError("encoded snapshot size must be positive")
    if continuation_ceiling_bytes <= 0 or field_ceiling_bytes <= 0:
        raise ValueError("snapshot ceilings must be positive")
    field_sizes = {
        str(field): logical_tensor_bytes(value) for field, value in snapshot.items()
    }
    logical_bytes = sum(field_sizes.values())
    if logical_bytes <= 0:
        raise ValueError("snapshot must contain continuation tensors")
    if logical_bytes > continuation_ceiling_bytes:
        raise ValueError("logical snapshot size exceeds continuation-state ceiling")
    if encoded_bytes > continuation_ceiling_bytes:
        raise ValueError("encoded snapshot size exceeds continuation-state ceiling")
    oversized = {
        field: size for field, size in field_sizes.items() if size > field_ceiling_bytes
    }
    if oversized:
        raise ValueError(f"snapshot contains model-parameter-sized fields: {oversized}")
    normalize_manifest(snapshot)
    return SnapshotSizeReport(
        logical_bytes=logical_bytes,
        encoded_bytes=encoded_bytes,
        field_logical_bytes=field_sizes,
    )


@dataclass(frozen=True, slots=True)
class TransferTimeline:
    """Key sink sequences proving one A-to-B-to-A bundle transfer."""

    a_chunk_started: int
    b_request_enqueued: int
    a_drain_requested: int
    a_chunk_committed: int
    a_snapshot_completed: int
    a_offload_verified: int
    a_release_committed: int
    b_allocation_committed: int
    b_useful_work_started: int
    b_release_committed: int
    a_reallocation_committed: int
    a_onload_verified: int
    a_restore_validated: int
    a_resume_dispatched: int
    sibling_chunk_committed: int


@dataclass(frozen=True, slots=True)
class SchedulerTransferCommits:
    """Trusted core commit timestamps for one exact A-to-B-to-A handoff."""

    a_release_ns: int
    b_allocation_ns: int
    b_release_ns: int
    a_reallocation_ns: int


def correlate_scheduler_transfer_commits(
    events: Iterable[AcceptanceEvent],
    commits: Iterable[SchedulerCommitRecord],
    *,
    pipeline_a: str,
    pipeline_b: str,
    selected_rank: int,
    bundle: tuple[int, ...],
) -> SchedulerTransferCommits:
    """Join post-commit core evidence to worker offload, work, and onload events."""
    worker_events = tuple(events)
    for event in worker_events:
        event.validate()
    if any(
        event.event == "callback_failed"
        and event.pipeline_id in {pipeline_a, pipeline_b}
        and event.dp_rank == selected_rank
        for event in worker_events
    ):
        raise ValueError("failed resize callback cannot produce transfer evidence")

    ordered_commits = tuple(commits)
    previous_key: tuple[int, int] | None = None
    for commit in ordered_commits:
        commit.validate()
        key = (commit.commit_time_ns, commit.cycle_counter)
        if previous_key is not None and key < previous_key:
            raise ValueError("scheduler commit records are not chronological")
        previous_key = key

    def matching_commits(
        pipeline_id: str, operation: str
    ) -> list[SchedulerCommitRecord]:
        return [
            commit
            for commit in ordered_commits
            if commit.pipeline_id == pipeline_id
            and commit.operation == operation
            and commit.dp_rank == selected_rank
            and commit.gpu_ids == bundle
        ]

    b_allocations = matching_commits(pipeline_b, "allocation")
    b_releases = matching_commits(pipeline_b, "release")
    if len(b_allocations) != 1 or len(b_releases) != 1:
        raise ValueError("expected exactly one B allocation and release commit")
    b_allocation = b_allocations[0]
    b_release = b_releases[0]

    a_releases = [
        commit
        for commit in matching_commits(pipeline_a, "release")
        if commit.commit_time_ns < b_allocation.commit_time_ns
    ]
    a_reallocations = [
        commit
        for commit in matching_commits(pipeline_a, "allocation")
        if commit.commit_time_ns > b_release.commit_time_ns
    ]
    if len(a_releases) != 1 or len(a_reallocations) != 1:
        raise ValueError(
            "expected one A release before B and one A reallocation after B"
        )
    a_release = a_releases[0]
    a_reallocation = a_reallocations[0]
    commit_order = (
        a_release.commit_time_ns,
        b_allocation.commit_time_ns,
        b_release.commit_time_ns,
        a_reallocation.commit_time_ns,
    )
    if tuple(sorted(commit_order)) != commit_order or len(set(commit_order)) != 4:
        raise ValueError(f"invalid scheduler A-to-B-to-A commit order: {commit_order}")

    def one_event(pipeline_id: str, event_name: str) -> AcceptanceEvent:
        matches = [
            event
            for event in worker_events
            if event.pipeline_id == pipeline_id
            and event.event == event_name
            and event.dp_rank == selected_rank
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {event_name} for {pipeline_id} rank {selected_rank}"
            )
        match = matches[0]
        if (
            event_name
            in {
                "environment_offload_verified",
                "rollout_offload_verified",
                "useful_cuda_work_started",
                "environment_onload_verified",
                "rollout_onload_verified",
            }
            and match.gpu_ids != bundle
        ):
            raise ValueError(f"{event_name} does not reference the committed bundle")
        return match

    snapshot = one_event(pipeline_a, "snapshot_completed")
    offloads = (
        one_event(pipeline_a, "environment_offload_verified"),
        one_event(pipeline_a, "rollout_offload_verified"),
    )
    b_work = one_event(pipeline_b, "useful_cuda_work_started")
    onloads = (
        one_event(pipeline_a, "environment_onload_verified"),
        one_event(pipeline_a, "rollout_onload_verified"),
    )
    if (
        max(snapshot.producer_time_ns, *(event.producer_time_ns for event in offloads))
        >= a_release.commit_time_ns
    ):
        raise ValueError("A release committed before snapshot and verified offload")
    if not (
        b_allocation.commit_time_ns < b_work.producer_time_ns < b_release.commit_time_ns
    ):
        raise ValueError("B useful work is outside its committed ownership interval")
    if a_reallocation.commit_time_ns >= min(
        event.producer_time_ns for event in onloads
    ):
        raise ValueError("A onload was not verified after reallocation commit")

    return SchedulerTransferCommits(
        a_release_ns=a_release.commit_time_ns,
        b_allocation_ns=b_allocation.commit_time_ns,
        b_release_ns=b_release.commit_time_ns,
        a_reallocation_ns=a_reallocation.commit_time_ns,
    )


def validate_transfer_timeline(
    events: Iterable[AcceptanceEvent],
    allocations: Iterable[AllocationSlice],
    *,
    pipeline_a: str,
    pipeline_b: str,
    selected_rank: int,
    sibling_rank: int,
    bundle: tuple[int, ...],
    committed_transition: TransitionIdentity,
    resumed_transition: TransitionIdentity,
) -> TransferTimeline:
    """Prove exact A-to-B-to-A ordering and selected-rank elasticity."""
    ordered = sorted(events, key=_sink_sequence)
    if not ordered:
        raise ValueError("transfer analysis requires acceptance events")
    for expected_sequence, event in enumerate(ordered):
        event.validate()
        if event.sink_sequence != expected_sequence:
            raise ValueError("sink sequences must be complete and contiguous")
        if event.sink_time_ns is None:
            raise ValueError("accepted events require sink timestamps")
        if (
            expected_sequence
            and event.sink_time_ns < ordered[expected_sequence - 1].sink_time_ns
        ):
            raise ValueError("sink timestamps must not regress")
    validate_exclusive_ownership(allocations)

    def one(
        pipeline_id: str,
        event_name: str,
        *,
        rank: int,
        transition: TransitionIdentity | None = None,
        require_bundle: bool = False,
    ) -> AcceptanceEvent:
        matches = [
            event
            for event in ordered
            if event.pipeline_id == pipeline_id
            and event.event == event_name
            and event.dp_rank == rank
            and (transition is None or event.transition_identity == transition)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {event_name} for {pipeline_id} rank {rank}, "
                f"got {len(matches)}"
            )
        match = matches[0]
        if require_bundle and match.gpu_ids != bundle:
            raise ValueError(
                f"{event_name} bundle {match.gpu_ids} does not match {bundle}"
            )
        return match

    a_chunk_started = one(
        pipeline_a,
        "chunk_started",
        rank=selected_rank,
        transition=committed_transition,
    )
    b_request = one(pipeline_b, "request_enqueued", rank=selected_rank)
    a_drain = one(
        pipeline_a,
        "drain_requested",
        rank=selected_rank,
        transition=committed_transition,
    )
    a_chunk_committed = one(
        pipeline_a,
        "chunk_committed",
        rank=selected_rank,
        transition=committed_transition,
    )
    sibling_commit = one(pipeline_a, "chunk_committed", rank=sibling_rank)
    a_snapshot = one(
        pipeline_a,
        "snapshot_completed",
        rank=selected_rank,
        transition=resumed_transition,
    )
    a_environment_offload = one(
        pipeline_a,
        "environment_offload_verified",
        rank=selected_rank,
        require_bundle=True,
    )
    a_rollout_offload = one(
        pipeline_a,
        "rollout_offload_verified",
        rank=selected_rank,
        require_bundle=True,
    )
    a_release = one(
        pipeline_a,
        "release_committed",
        rank=selected_rank,
        require_bundle=True,
    )
    b_allocation = one(
        pipeline_b,
        "allocation_committed",
        rank=selected_rank,
        require_bundle=True,
    )
    b_work = one(
        pipeline_b,
        "useful_cuda_work_started",
        rank=selected_rank,
        require_bundle=True,
    )
    b_release = one(
        pipeline_b,
        "release_committed",
        rank=selected_rank,
        require_bundle=True,
    )
    a_allocations = [
        event
        for event in ordered
        if event.pipeline_id == pipeline_a
        and event.event == "allocation_committed"
        and event.dp_rank == selected_rank
        and event.gpu_ids == bundle
        and _sink_sequence(event) > _sink_sequence(b_release)
    ]
    if len(a_allocations) != 1:
        raise ValueError("expected exactly one A reallocation after B release")
    a_reallocation = a_allocations[0]
    a_environment_onload = one(
        pipeline_a,
        "environment_onload_verified",
        rank=selected_rank,
        require_bundle=True,
    )
    a_rollout_onload = one(
        pipeline_a,
        "rollout_onload_verified",
        rank=selected_rank,
        require_bundle=True,
    )
    a_restore = one(
        pipeline_a,
        "restore_validated",
        rank=selected_rank,
        transition=resumed_transition,
    )
    a_resume = one(
        pipeline_a,
        "resumed_bootstrap_dispatched",
        rank=selected_rank,
        transition=resumed_transition,
    )

    offload_sequence = max(
        _sink_sequence(a_environment_offload),
        _sink_sequence(a_rollout_offload),
    )
    onload_sequence = max(
        _sink_sequence(a_environment_onload),
        _sink_sequence(a_rollout_onload),
    )
    required_order = (
        _sink_sequence(a_chunk_started),
        _sink_sequence(b_request),
        _sink_sequence(a_drain),
        _sink_sequence(a_chunk_committed),
        _sink_sequence(a_snapshot),
        offload_sequence,
        _sink_sequence(a_release),
        _sink_sequence(b_allocation),
        _sink_sequence(b_work),
        _sink_sequence(b_release),
        _sink_sequence(a_reallocation),
        onload_sequence,
        _sink_sequence(a_restore),
        _sink_sequence(a_resume),
    )
    if tuple(sorted(required_order)) != required_order or len(
        set(required_order)
    ) != len(required_order):
        raise ValueError(f"invalid A-to-B-to-A transfer order: {required_order}")
    if not (
        _sink_sequence(a_drain)
        < _sink_sequence(sibling_commit)
        < _sink_sequence(a_reallocation)
    ):
        raise ValueError("A sibling did not continue during the selected-rank transfer")

    return TransferTimeline(
        a_chunk_started=_sink_sequence(a_chunk_started),
        b_request_enqueued=_sink_sequence(b_request),
        a_drain_requested=_sink_sequence(a_drain),
        a_chunk_committed=_sink_sequence(a_chunk_committed),
        a_snapshot_completed=_sink_sequence(a_snapshot),
        a_offload_verified=offload_sequence,
        a_release_committed=_sink_sequence(a_release),
        b_allocation_committed=_sink_sequence(b_allocation),
        b_useful_work_started=_sink_sequence(b_work),
        b_release_committed=_sink_sequence(b_release),
        a_reallocation_committed=_sink_sequence(a_reallocation),
        a_onload_verified=onload_sequence,
        a_restore_validated=_sink_sequence(a_restore),
        a_resume_dispatched=_sink_sequence(a_resume),
        sibling_chunk_committed=_sink_sequence(sibling_commit),
    )


def _sink_sequence(event: AcceptanceEvent) -> int:
    if event.sink_sequence is None:
        raise ValueError("accepted event is missing sink_sequence")
    return event.sink_sequence


def build_analysis_summary(
    *,
    run_manifest: RunManifest,
    reference_differences: tuple[str, ...],
    transfer_timeline: TransferTimeline,
    snapshot_report: SnapshotSizeReport,
    gpu_summaries: Mapping[int, GpuUtilizationSummary],
    utilization: UtilizationAcceptance,
    raw_artifacts: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    """Generate deterministic JSON and Markdown views from validated inputs."""
    run_manifest.validate()
    required_artifacts = {"events", "scheduler", "gpu_samples", "reference"}
    if set(raw_artifacts) != required_artifacts or any(
        not path for path in raw_artifacts.values()
    ):
        raise ValueError(
            "raw_artifacts must contain non-empty events, scheduler, "
            "gpu_samples, and reference paths"
        )
    if not gpu_summaries:
        raise ValueError("analysis requires direct GPU utilization summaries")
    expected_gpus = {
        gpu_id for bundle in run_manifest.expected_bundles for gpu_id in bundle
    }
    if set(gpu_summaries) != expected_gpus:
        raise ValueError(
            "GPU summaries do not match the run manifest actor-infer devices"
        )
    correctness_passed = not reference_differences
    status = "passed" if correctness_passed and utilization.passed else "failed"
    summary = normalize_manifest(
        {
            "schema_version": run_manifest.schema_version,
            "run_id": run_manifest.run_id,
            "status": status,
            "scope": {
                "environment": run_manifest.environment,
                "mode": run_manifest.mode,
                "scenario": run_manifest.scenario,
                "intra_shard_idleness_solved": False,
            },
            "gates": {
                "reference_equivalence": correctness_passed,
                "ownership_transfer": True,
                "utilization": utilization.passed,
            },
            "reference_differences": reference_differences,
            "transfer_timeline": asdict(transfer_timeline),
            "snapshot": asdict(snapshot_report),
            "gpu_utilization": {
                str(gpu_id): asdict(summary)
                for gpu_id, summary in sorted(gpu_summaries.items())
            },
            "utilization": asdict(utilization),
            "raw_artifacts": dict(raw_artifacts),
        }
    )
    markdown = "\n".join(
        (
            f"# Task 8 Acceptance Report: {run_manifest.run_id}",
            "",
            f"Status: **{status.upper()}**",
            "",
            "## Tested scope",
            "",
            f"- Environment: `{run_manifest.environment}`",
            f"- Placement: `{run_manifest.mode}`",
            f"- Scenario: `{run_manifest.scenario}`",
            "- Cross-pipeline atomic-bundle sharing only; intra-shard VLA/world-model idleness remains deferred.",
            "",
            "## Hard gates",
            "",
            f"- Reference equivalence: {'PASS' if correctness_passed else 'FAIL'}",
            "- A-to-B-to-A ownership transfer: PASS",
            f"- Utilization threshold: {'PASS' if utilization.passed else 'FAIL'}",
            f"- Median throughput improvement: {utilization.median_improvement:.2%}",
            "- Median direct-GPU idle reduction: "
            + (
                f"{utilization.median_idle_reduction:.2%}"
                if utilization.median_idle_reduction is not None
                else "NOT MEASURED"
            ),
            "",
            "## Raw evidence",
            "",
            *(f"- {name}: `{path}`" for name, path in sorted(raw_artifacts.items())),
            "",
        )
    )
    return summary, markdown
