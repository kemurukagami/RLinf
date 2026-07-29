"""Filesystem and external-tool adapters for Task 8 acceptance artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

from task8_acceptance_analysis import (
    GrpoIterationEvidence,
    SnapshotSizeReport,
    TransferTimeline,
    build_analysis_summary,
    validate_grpo_training_loop,
)
from task8_acceptance_support import (
    GpuUtilizationSummary,
    RunManifest,
    SchedulerCommitRecord,
    UtilizationAcceptance,
    UtilizationTrial,
    evaluate_utilization_trials,
    normalize_manifest,
    normalize_scheduler_commit_marker,
    validate_run_id,
)

_COMMIT_QUERY = """
SELECT
  slice.ts AS timestamp_ns,
  slice.name AS marker_name,
  COALESCE(args.string_value, args.display_value) AS commit_json
FROM slice
JOIN args USING (arg_set_id)
WHERE slice.name GLOB 'Commit C*'
  AND args.key IN ('commit_json', 'debug.commit_json')
ORDER BY slice.ts, slice.name
""".strip()


@dataclass(frozen=True, slots=True)
class AcceptanceArtifactLayout:
    """Canonical directories for one isolated Task 8 acceptance run."""

    root: Path
    drivers: Path
    core: Path
    gpu: Path
    reference: Path
    analysis: Path


@dataclass(frozen=True, slots=True)
class AcceptanceReportMetadata:
    """Operator-supplied provenance rendered into the generated report."""

    root_commit: str
    rlinf_commit: str
    dirty_worktree: bool
    hardware: tuple[str, ...]
    commands: tuple[str, ...]
    limitations: tuple[str, ...]
    invalid_repetitions: tuple[str, ...] = ()

    def validate(self) -> None:
        """Reject incomplete provenance rather than emitting an ambiguous report."""
        required = (self.root_commit, self.rlinf_commit, *self.hardware, *self.commands)
        if (
            not self.hardware
            or not self.commands
            or any(not value.strip() for value in required)
        ):
            raise ValueError("report metadata requires commits, hardware, and commands")


def emit_acceptance_artifact_set(
    layout: AcceptanceArtifactLayout,
    *,
    run_manifest: RunManifest,
    reference_differences: tuple[str, ...],
    transfer_timeline: TransferTimeline,
    snapshot_report: SnapshotSizeReport,
    gpu_summaries: Mapping[int, GpuUtilizationSummary],
    utilization_trials: Sequence[UtilizationTrial],
    training_iterations: Mapping[str, Sequence[GrpoIterationEvidence]],
    latencies_ms: Mapping[str, Sequence[float]],
    raw_artifacts: Mapping[str, str],
    metadata: AcceptanceReportMetadata,
) -> tuple[dict[str, Any], str]:
    """Verify sealed raw evidence and atomically emit the complete derived set."""
    run_manifest.validate()
    metadata.validate()
    _validate_stored_manifest(layout, run_manifest)
    verified = {path.resolve() for path in verify_completion_marker(layout)}
    _validate_raw_artifact_links(layout, raw_artifacts, verified)
    _validate_repetition_set(utilization_trials, run_manifest.repetitions)
    validated_training = _validate_training_iterations(
        training_iterations, run_manifest.training_iterations
    )
    _validate_gpu_summaries(gpu_summaries)

    minimum_improved_pairs = max(1, (4 * run_manifest.repetitions + 4) // 5)
    utilization = evaluate_utilization_trials(
        utilization_trials,
        minimum_pairs=run_manifest.repetitions,
        minimum_median_improvement=run_manifest.minimum_throughput_improvement,
        minimum_improved_pairs=minimum_improved_pairs,
        minimum_median_idle_reduction=run_manifest.minimum_idle_reduction,
    )
    summary, base_report = build_analysis_summary(
        run_manifest=run_manifest,
        reference_differences=reference_differences,
        transfer_timeline=transfer_timeline,
        snapshot_report=snapshot_report,
        gpu_summaries=gpu_summaries,
        utilization=utilization,
        raw_artifacts=raw_artifacts,
    )
    summary["training_iterations"] = {
        pipeline_id: [asdict(iteration) for iteration in iterations]
        for pipeline_id, iterations in sorted(validated_training.items())
    }
    latency_rows = _latency_rows(latencies_ms)
    trial_rows = _utilization_rows(utilization_trials, utilization)

    derived_paths = (
        layout.analysis / "correctness.json",
        layout.analysis / "ownership.json",
        layout.analysis / "latency_summary.csv",
        layout.analysis / "utilization_repetitions.csv",
        layout.analysis / "utilization_summary.json",
        layout.analysis / "REPORT.md",
    )
    existing = [path for path in derived_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite derived artifacts: {existing!r}")
    atomic_write_json(layout.analysis / "correctness.json", summary)
    atomic_write_json(layout.analysis / "ownership.json", asdict(transfer_timeline))
    atomic_write_csv(
        layout.analysis / "latency_summary.csv",
        fieldnames=("metric", "count", "minimum_ms", "median_ms", "maximum_ms"),
        rows=latency_rows,
    )
    atomic_write_csv(
        layout.analysis / "utilization_repetitions.csv",
        fieldnames=(
            "repetition",
            "static_throughput_per_gpu",
            "dynamic_throughput_per_gpu",
            "throughput_improvement",
            "static_idle_fraction",
            "dynamic_idle_fraction",
            "idle_reduction",
        ),
        rows=trial_rows,
    )
    atomic_write_json(layout.analysis / "utilization_summary.json", asdict(utilization))
    report = _render_operator_report(
        base_report,
        run_manifest,
        summary,
        metadata,
        snapshot_report=snapshot_report,
        gpu_summaries=gpu_summaries,
        utilization_trials=utilization_trials,
        utilization=utilization,
    )
    _atomic_write_text(layout.analysis / "REPORT.md", report)
    return summary, report


def _validate_stored_manifest(
    layout: AcceptanceArtifactLayout, run_manifest: RunManifest
) -> None:
    path = layout.root / "run_manifest.json"
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("stored run manifest is missing or unreadable") from exc
    if stored != normalize_manifest(asdict(run_manifest)):
        raise ValueError("analysis run manifest does not match the stored manifest")


def _validate_repetition_set(
    trials: Sequence[UtilizationTrial], expected_repetitions: int
) -> None:
    repetitions = [trial.repetition for trial in trials]
    expected = set(range(expected_repetitions))
    if set(repetitions) != expected or len(repetitions) != 2 * expected_repetitions:
        raise ValueError(
            "utilization trials must contain exactly one static and dynamic pair "
            "for every configured repetition"
        )


def _validate_training_iterations(
    training: Mapping[str, Sequence[GrpoIterationEvidence]], expected_iterations: int
) -> dict[str, tuple[GrpoIterationEvidence, ...]]:
    if len(training) != 2 or any(not pipeline_id for pipeline_id in training):
        raise ValueError("training evidence requires exactly two pipeline IDs")
    validated: dict[str, tuple[GrpoIterationEvidence, ...]] = {}
    for pipeline_id, iterations in training.items():
        evidence = validate_grpo_training_loop(iterations)
        if len(evidence) != expected_iterations:
            raise ValueError(
                "training evidence must match the configured iteration count"
            )
        validated[pipeline_id] = evidence
    return validated


def _validate_gpu_summaries(
    summaries: Mapping[int, GpuUtilizationSummary],
) -> None:
    for gpu_id, summary in summaries.items():
        if (
            gpu_id != summary.gpu_id
            or summary.duration_ns <= 0
            or not math.isfinite(summary.mean_sm_utilization)
            or not 0.0 <= summary.mean_sm_utilization <= 100.0
            or not 0 <= summary.idle_ns <= summary.duration_ns
        ):
            raise ValueError(f"invalid direct GPU summary for GPU {gpu_id}")


def _validate_raw_artifact_links(
    layout: AcceptanceArtifactLayout,
    raw_artifacts: Mapping[str, str],
    verified: set[Path],
) -> None:
    required = {"events", "scheduler", "gpu_samples", "reference"}
    if set(raw_artifacts) != required:
        raise ValueError(
            f"raw artifact links must contain exactly {sorted(required)!r}"
        )
    root = layout.root.resolve()
    for name, relative in raw_artifacts.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"raw artifact link {name!r} is unsafe")
        resolved = (root / path).resolve()
        if resolved not in verified:
            raise ValueError(f"raw artifact link {name!r} is not sealed")


def _latency_rows(latencies_ms: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    if not latencies_ms:
        raise ValueError("at least one measured latency family is required")
    rows: list[dict[str, Any]] = []
    for metric, values in sorted(latencies_ms.items()):
        measured = tuple(float(value) for value in values)
        if (
            not metric
            or not measured
            or any(not math.isfinite(value) or value < 0.0 for value in measured)
        ):
            raise ValueError("latency metrics require a name and non-negative samples")
        rows.append(
            {
                "metric": metric,
                "count": len(measured),
                "minimum_ms": min(measured),
                "median_ms": median(measured),
                "maximum_ms": max(measured),
            }
        )
    return rows


def _utilization_rows(
    trials: Sequence[UtilizationTrial],
    acceptance: UtilizationAcceptance,
) -> list[dict[str, Any]]:
    pairs: dict[int, dict[str, UtilizationTrial]] = {}
    for trial in trials:
        pairs.setdefault(trial.repetition, {})[trial.mode] = trial
    rows: list[dict[str, Any]] = []
    for index, (repetition, pair) in enumerate(sorted(pairs.items())):
        static = pair["static"]
        dynamic = pair["dynamic"]
        rows.append(
            {
                "repetition": repetition,
                "static_throughput_per_gpu": static.useful_throughput_per_gpu,
                "dynamic_throughput_per_gpu": dynamic.useful_throughput_per_gpu,
                "throughput_improvement": acceptance.paired_improvements[index],
                "static_idle_fraction": static.idle_fraction,
                "dynamic_idle_fraction": dynamic.idle_fraction,
                "idle_reduction": acceptance.paired_idle_reductions[index],
            }
        )
    return rows


def _render_operator_report(
    base_report: str,
    manifest: RunManifest,
    summary: Mapping[str, Any],
    metadata: AcceptanceReportMetadata,
    *,
    snapshot_report: SnapshotSizeReport,
    gpu_summaries: Mapping[int, GpuUtilizationSummary],
    utilization_trials: Sequence[UtilizationTrial],
    utilization: UtilizationAcceptance,
) -> str:
    invalid = metadata.invalid_repetitions or ("None",)
    reference_differences = summary["reference_differences"] or ["None"]
    training_rows = tuple(
        f"| `{pipeline_id}` | {iteration['iteration']} | "
        f"{iteration['collection_policy_version']} | {iteration['received_trajectories']} | "
        f"{iteration['produced_policy_version']} |"
        for pipeline_id, iterations in sorted(summary["training_iterations"].items())
        for iteration in iterations
    )
    gpu_rows = tuple(
        f"| {gpu_id} | {gpu.mean_sm_utilization:.2f}% | "
        f"{gpu.idle_ns / gpu.duration_ns:.2%} |"
        for gpu_id, gpu in sorted(gpu_summaries.items())
    )
    trial_rows = _utilization_rows(utilization_trials, utilization)
    utilization_rows = tuple(
        f"| {row['repetition']} | {row['static_throughput_per_gpu']:.6f} | "
        f"{row['dynamic_throughput_per_gpu']:.6f} | "
        f"{row['throughput_improvement']:.2%} | {row['idle_reduction']:.2%} |"
        for row in trial_rows
    )
    return base_report + "\n".join(
        (
            "## Provenance",
            "",
            f"- Root commit: `{metadata.root_commit}`",
            f"- RLinf commit: `{metadata.rlinf_commit}`",
            f"- Dirty worktree: `{str(metadata.dirty_worktree).lower()}`",
            *(f"- Hardware: {item}" for item in metadata.hardware),
            f"- Checkpoint digests: `{json.dumps(dict(manifest.checkpoint_digests), sort_keys=True)}`",
            "",
            "## Reference equivalence",
            "",
            *(f"- {item}" for item in reference_differences),
            "",
            "## Batch and policy versions",
            "",
            "| Pipeline | Iteration | Collected policy | Trajectories | Produced policy |",
            "| --- | ---: | ---: | ---: | ---: |",
            *training_rows,
            "",
            "## Transfer timeline",
            "",
            "```json",
            json.dumps(summary["transfer_timeline"], indent=2, sort_keys=True),
            "```",
            "",
            "## Snapshot and GPU measurements",
            "",
            f"- Snapshot logical bytes: {snapshot_report.logical_bytes}",
            f"- Snapshot encoded bytes: {snapshot_report.encoded_bytes}",
            "- Latency distributions: `latency_summary.csv`",
            "",
            "| GPU | Mean SM utilization | Idle fraction |",
            "| ---: | ---: | ---: |",
            *gpu_rows,
            "",
            "## Static versus dynamic utilization",
            "",
            "| Repetition | Static throughput/GPU | Dynamic throughput/GPU | Improvement | Idle reduction |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *utilization_rows,
            "",
            "## Invalid repetitions",
            "",
            *(f"- {item}" for item in invalid),
            "",
            "## Limitations",
            "",
            *(f"- {item}" for item in metadata.limitations),
            "",
            "## Commands",
            "",
            *(f"- `{command}`" for command in metadata.commands),
            "",
        )
    )


def prepare_acceptance_artifacts(
    output_dir: str | Path,
    *,
    run_manifest: RunManifest,
) -> AcceptanceArtifactLayout:
    """Create an isolated run tree, refusing an existing non-empty run."""
    run_manifest.validate()
    output_root = Path(output_dir)
    validate_run_id(run_manifest.run_id)
    run_root = output_root / run_manifest.run_id
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"acceptance run directory is not empty: {run_root}")
    layout = AcceptanceArtifactLayout(
        root=run_root,
        drivers=run_root / "drivers",
        core=run_root / "core",
        gpu=run_root / "gpu",
        reference=run_root / "reference",
        analysis=run_root / "analysis",
    )
    for directory in (
        layout.root,
        layout.drivers / "a",
        layout.drivers / "b",
        layout.core,
        layout.gpu,
        layout.reference,
        layout.analysis,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(layout.root / "run_manifest.json", asdict(run_manifest))
    return layout


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write one normalized JSON artifact atomically without overwriting."""
    normalized = normalize_manifest(payload)
    text = (
        json.dumps(
            normalized,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    _atomic_write_text(Path(path), text)


def atomic_write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    """Write normalized JSONL rows atomically without overwriting."""
    encoded = [
        json.dumps(
            normalize_manifest(row),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        for row in rows
    ]
    if not encoded:
        raise ValueError("JSONL artifact must contain at least one row")
    _atomic_write_text(Path(path), "\n".join(encoded) + "\n")


def atomic_write_csv(
    path: str | Path,
    *,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write a rectangular CSV artifact atomically without overwriting."""
    if not fieldnames or len(set(fieldnames)) != len(fieldnames):
        raise ValueError("CSV fieldnames must be non-empty and unique")
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    row_count = 0
    for row in rows:
        if set(row) != set(fieldnames):
            raise ValueError("CSV row fields do not match the declared fieldnames")
        writer.writerow({name: row[name] for name in fieldnames})
        row_count += 1
    if row_count == 0:
        raise ValueError("CSV artifact must contain at least one data row")
    _atomic_write_text(Path(path), output.getvalue())


def write_completion_marker(
    layout: AcceptanceArtifactLayout,
    *,
    relative_artifacts: Sequence[str | Path],
) -> Path:
    """Seal closed raw artifacts with size and SHA-256 evidence."""
    if not relative_artifacts:
        raise ValueError("completion marker requires raw artifacts")
    root = layout.root.resolve()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for relative in relative_artifacts:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                "completion artifacts must be relative paths within the run"
            )
        normalized_relative = relative_path.as_posix()
        if normalized_relative in seen:
            raise ValueError("completion artifacts contain duplicates")
        seen.add(normalized_relative)
        artifact = (root / relative_path).resolve()
        if root not in artifact.parents or not artifact.is_file():
            raise ValueError(f"completion artifact is missing: {relative_path}")
        data = artifact.read_bytes()
        if not data:
            raise ValueError(f"completion artifact is empty: {relative_path}")
        entries.append(
            {
                "path": normalized_relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    marker = layout.root / "RAW_ARTIFACTS_COMPLETE.json"
    atomic_write_json(marker, {"artifacts": entries})
    return marker


def verify_completion_marker(
    layout: AcceptanceArtifactLayout,
) -> tuple[Path, ...]:
    """Verify the completion marker and every sealed artifact digest."""
    marker = layout.root / "RAW_ARTIFACTS_COMPLETE.json"
    if not marker.is_file():
        raise ValueError("raw artifact completion marker is missing")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("raw artifact completion marker is unreadable") from exc
    if not isinstance(payload, Mapping) or set(payload) != {"artifacts"}:
        raise ValueError("raw artifact completion marker has an invalid schema")
    entries = payload["artifacts"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("raw artifact completion marker has no artifacts")
    root = layout.root.resolve()
    verified: list[Path] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("raw artifact completion entry has an invalid schema")
        if (
            not isinstance(entry["path"], str)
            or not isinstance(entry["bytes"], int)
            or isinstance(entry["bytes"], bool)
            or entry["bytes"] <= 0
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
        ):
            raise ValueError("raw artifact completion entry has invalid field types")
        relative_path = Path(entry["path"])
        normalized_relative = relative_path.as_posix()
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or normalized_relative in seen
        ):
            raise ValueError(
                "raw artifact completion entry has an unsafe or duplicate path"
            )
        seen.add(normalized_relative)
        artifact = (root / relative_path).resolve()
        if root not in artifact.parents or not artifact.is_file():
            raise ValueError(f"sealed raw artifact is missing: {relative_path}")
        data = artifact.read_bytes()
        if (
            entry["bytes"] != len(data)
            or entry["sha256"] != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError(
                f"sealed raw artifact changed after completion: {relative_path}"
            )
        verified.append(artifact)
    return tuple(verified)


def _atomic_write_text(path: Path, text: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {path}"
            ) from exc
        temporary_path.unlink()
        temporary_path = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_perfetto_commit_csv(
    csv_text: str,
    *,
    tracked_clusters: Mapping[str, str],
    canonical_bundles: Mapping[str, Mapping[int, Sequence[int]]],
) -> tuple[SchedulerCommitRecord, ...]:
    """Parse trace-processor CSV output into validated scheduler commits."""
    reader = csv.DictReader(StringIO(csv_text))
    required_fields = {"timestamp_ns", "marker_name", "commit_json"}
    if reader.fieldnames is None or set(reader.fieldnames) != required_fields:
        raise ValueError(
            "Perfetto commit CSV must contain exactly timestamp_ns, marker_name, and commit_json"
        )
    records: list[SchedulerCommitRecord] = []
    previous_marker: tuple[int, int] | None = None
    marker_count = 0
    for row_number, row in enumerate(reader, start=2):
        try:
            timestamp_ns = int(row["timestamp_ns"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid Perfetto timestamp on CSV row {row_number}"
            ) from exc
        marker_name = row["marker_name"]
        if marker_name is None:
            raise ValueError(f"missing Perfetto marker name on CSV row {row_number}")
        raw_payload = row["commit_json"]
        try:
            payload = json.loads(raw_payload) if raw_payload is not None else None
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid commit_json on CSV row {row_number}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"commit_json on CSV row {row_number} must be an object")
        marker_records = normalize_scheduler_commit_marker(
            marker_name=marker_name,
            timestamp_ns=timestamp_ns,
            payload=payload,
            tracked_clusters=tracked_clusters,
            canonical_bundles=canonical_bundles,
        )
        if marker_records:
            marker_key = (
                timestamp_ns,
                marker_records[0].cycle_counter,
            )
            if previous_marker is not None and marker_key <= previous_marker:
                raise ValueError(
                    "Perfetto commit markers must increase by timestamp and cycle"
                )
            previous_marker = marker_key
            records.extend(marker_records)
        marker_count += 1
    if marker_count == 0:
        raise ValueError("Perfetto trace contains no scheduler commit markers")
    if not records:
        raise ValueError(
            "Perfetto trace contains no commits for tracked acceptance clusters"
        )
    return tuple(records)


def extract_scheduler_commits_from_perfetto(
    trace_path: str | Path,
    *,
    trace_processor_shell: str | Path,
    tracked_clusters: Mapping[str, str],
    canonical_bundles: Mapping[str, Mapping[int, Sequence[int]]],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[SchedulerCommitRecord, ...]:
    """Run Perfetto's trace processor and normalize post-commit markers.

    The trace processor is an explicit acceptance dependency. This adapter does
    not download tooling or fall back to plan markers when it is unavailable.
    """
    trace = Path(trace_path)
    if not trace.is_file() or trace.stat().st_size <= 0:
        raise ValueError(f"Perfetto trace is missing or empty: {trace}")
    processor_input = str(trace_processor_shell)
    processor = shutil.which(processor_input)
    if processor is None:
        candidate = Path(processor_input)
        if not candidate.is_file():
            raise ValueError(f"trace_processor_shell is unavailable: {processor_input}")
        processor = str(candidate)
    result = runner(
        [processor, "--csv", "-Q", _COMMIT_QUERY, str(trace)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "no stderr"
        raise RuntimeError(
            f"trace_processor_shell failed with code {result.returncode}: {detail}"
        )
    return parse_perfetto_commit_csv(
        result.stdout,
        tracked_clusters=tracked_clusters,
        canonical_bundles=canonical_bundles,
    )
