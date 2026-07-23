"""Filesystem and external-tool adapters for Task 8 acceptance artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from task8_acceptance_support import (
    RunManifest,
    SchedulerCommitRecord,
    normalize_manifest,
    normalize_scheduler_commit_marker,
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


def prepare_acceptance_artifacts(
    output_dir: str | Path,
    *,
    run_manifest: RunManifest,
) -> AcceptanceArtifactLayout:
    """Create an isolated run tree, refusing an existing non-empty run."""
    run_manifest.validate()
    output_root = Path(output_dir)
    run_component = Path(run_manifest.run_id)
    if (
        run_component.is_absolute()
        or len(run_component.parts) != 1
        or run_manifest.run_id in {".", ".."}
    ):
        raise ValueError("run_id must be one safe output-directory component")
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
    text = json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write_text(Path(path), text)


def atomic_write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    """Write normalized JSONL rows atomically without overwriting."""
    encoded = [
        json.dumps(normalize_manifest(row), sort_keys=True, ensure_ascii=False)
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
