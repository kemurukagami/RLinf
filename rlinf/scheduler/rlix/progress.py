"""Pure progress aggregation for elastic embodied collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from rlinf.workers.elastic_rollout_lifecycle import (
    ElasticRankProgress,
    ElasticRankState,
)


@dataclass(frozen=True, slots=True)
class ElasticPipelineProgress:
    """Framework-neutral values used to construct an RLix progress report."""

    step_target_trajectories: int
    metrics: dict[str, int | list[int] | str]


class ElasticProgressTracker:
    """Combine controller-owned cold assignments with activated worker progress."""

    def __init__(
        self,
        *,
        lifecycle_generation: int,
        assigned_trajectories_by_rank: Mapping[int, int],
        mode: str = "train",
    ) -> None:
        """Initialize one collection lifecycle and its immutable assignments."""
        if not isinstance(lifecycle_generation, int) or isinstance(
            lifecycle_generation, bool
        ):
            raise TypeError("lifecycle_generation must be an integer")
        if lifecycle_generation <= 0:
            raise ValueError("lifecycle_generation must be positive")
        if not isinstance(mode, str) or not mode:
            raise ValueError("mode must be a non-empty string")
        if not isinstance(assigned_trajectories_by_rank, Mapping) or not (
            assigned_trajectories_by_rank
        ):
            raise ValueError(
                "assigned_trajectories_by_rank must be a non-empty mapping"
            )

        assignments: dict[int, int] = {}
        for rank, assigned in assigned_trajectories_by_rank.items():
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
                raise ValueError(f"Invalid assigned DP rank {rank!r}")
            if not isinstance(assigned, int) or isinstance(assigned, bool):
                raise TypeError("Assigned trajectory counts must be integers")
            if assigned <= 0:
                raise ValueError("Assigned trajectory counts must be positive")
            assignments[rank] = assigned
        expected_ranks = list(range(len(assignments)))
        if sorted(assignments) != expected_ranks:
            raise ValueError(
                f"Assigned DP ranks must be contiguous {expected_ranks!r}, "
                f"got {sorted(assignments)!r}"
            )

        self._lifecycle_generation = lifecycle_generation
        self._assignments = assignments
        self._mode = mode
        self._worker_progress: dict[int, ElasticRankProgress] = {}

    @property
    def lifecycle_generation(self) -> int:
        """Return the controller-owned collection lifecycle generation."""
        return self._lifecycle_generation

    @property
    def step_target_trajectories(self) -> int:
        """Return total assigned complete trajectories for the lifecycle."""
        return sum(self._assignments.values())

    def update_worker_progress(self, progress: ElasticRankProgress) -> None:
        """Accept one activated worker snapshot after identity validation."""
        if not isinstance(progress, ElasticRankProgress):
            raise TypeError("progress must be ElasticRankProgress")
        if progress.dp_rank not in self._assignments:
            raise ValueError(f"Unknown DP rank {progress.dp_rank}")
        if progress.lifecycle_generation != self._lifecycle_generation:
            raise ValueError(
                "Worker progress lifecycle generation does not match the tracker"
            )
        expected_assignment = self._assignments[progress.dp_rank]
        if progress.assigned_trajectories != expected_assignment:
            raise ValueError(
                f"Worker rank {progress.dp_rank} assignment does not match tracker"
            )
        previous = self._worker_progress.get(progress.dp_rank)
        if previous is not None and (
            progress.completed_trajectories < previous.completed_trajectories
        ):
            raise ValueError("Completed trajectory progress cannot decrease")
        self._worker_progress[progress.dp_rank] = progress

    def snapshot(
        self,
        *,
        active_dp_ranks: set[int],
        reserved_dp_ranks: set[int] | None = None,
    ) -> ElasticPipelineProgress:
        """Build deterministic wire metrics without importing rlix-core."""
        if not isinstance(active_dp_ranks, set):
            raise TypeError("active_dp_ranks must be a set of integers")
        invalid_active = {
            rank
            for rank in active_dp_ranks
            if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
        }
        if invalid_active:
            raise ValueError(f"Invalid active DP ranks {sorted(invalid_active)!r}")
        unknown_active = active_dp_ranks - self._assignments.keys()
        if unknown_active:
            raise ValueError(f"Unknown active DP ranks {sorted(unknown_active)!r}")
        include_reserved = reserved_dp_ranks is not None
        reserved_dp_ranks = set() if reserved_dp_ranks is None else reserved_dp_ranks
        if not isinstance(reserved_dp_ranks, set):
            raise TypeError("reserved_dp_ranks must be a set of integers")
        if not reserved_dp_ranks.issubset(active_dp_ranks):
            raise ValueError("reserved DP ranks must be active")

        resumable: set[int] = set()
        completed: set[int] = set()
        safe: set[int] = set()
        completed_trajectories = 0

        for rank in sorted(self._assignments):
            progress = self._worker_progress.get(rank)
            if progress is None:
                resumable.add(rank)
                continue

            completed_trajectories += progress.completed_trajectories
            if (
                progress.state
                in {
                    ElasticRankState.INACTIVE_COLD,
                    ElasticRankState.PAUSED,
                }
                and progress.completed_trajectories < progress.assigned_trajectories
            ):
                resumable.add(rank)
            if progress.state is ElasticRankState.COMPLETED:
                completed.add(rank)
            if progress.snapshot_ready and progress.state in {
                ElasticRankState.SNAPSHOTTING,
                ElasticRankState.PAUSED,
            }:
                safe.add(rank)

        metrics: dict[str, int | list[int] | str] = {
            "mode": self._mode,
            "completed": completed_trajectories,
            "active_dp_ranks": sorted(active_dp_ranks),
            "resumable_dp_ranks": sorted(resumable),
            "completed_dp_ranks": sorted(completed),
            "at_safe_point_dp_ranks": sorted(safe),
        }
        if include_reserved:
            metrics["reserved_dp_ranks"] = sorted(reserved_dp_ranks)
        return ElasticPipelineProgress(
            step_target_trajectories=self.step_target_trajectories,
            metrics=metrics,
        )
