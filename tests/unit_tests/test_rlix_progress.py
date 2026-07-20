from __future__ import annotations

import pytest

from rlinf.scheduler.rlix.progress import ElasticProgressTracker
from rlinf.workers.elastic_rollout_lifecycle import (
    ElasticRankProgress,
    ElasticRankState,
)
from rlinf.workers.env.env_worker import EnvWorker


def _progress(
    rank: int,
    state: ElasticRankState,
    *,
    assigned: int = 2,
    completed: int = 0,
    generation: int = 3,
    snapshot_ready: bool = False,
) -> ElasticRankProgress:
    return ElasticRankProgress(
        dp_rank=rank,
        lifecycle_generation=generation,
        state=state,
        assigned_trajectories=assigned,
        completed_trajectories=completed,
        snapshot_ready=snapshot_ready,
        failed=state is ElasticRankState.FAILED_RESIDENT,
    )


def test_tracker_synthesizes_cold_rank_eligibility() -> None:
    tracker = ElasticProgressTracker(
        lifecycle_generation=3,
        assigned_trajectories_by_rank={0: 2, 1: 2},
    )

    snapshot = tracker.snapshot(active_dp_ranks=set())

    assert snapshot.step_target_trajectories == 4
    assert snapshot.metrics == {
        "mode": "train",
        "completed": 0,
        "active_dp_ranks": [],
        "resumable_dp_ranks": [0, 1],
        "completed_dp_ranks": [],
        "at_safe_point_dp_ranks": [],
    }


def test_worker_progress_replaces_cold_projection() -> None:
    tracker = ElasticProgressTracker(
        lifecycle_generation=3,
        assigned_trajectories_by_rank={0: 2, 1: 2},
    )
    tracker.update_worker_progress(_progress(0, ElasticRankState.ACTIVE))
    tracker.update_worker_progress(
        _progress(
            1,
            ElasticRankState.PAUSED,
            snapshot_ready=True,
        )
    )

    snapshot = tracker.snapshot(active_dp_ranks={0})

    assert snapshot.metrics["active_dp_ranks"] == [0]
    assert snapshot.metrics["resumable_dp_ranks"] == [1]
    assert snapshot.metrics["at_safe_point_dp_ranks"] == [1]


def test_completed_rank_can_remain_scheduler_active_until_release() -> None:
    tracker = ElasticProgressTracker(
        lifecycle_generation=3,
        assigned_trajectories_by_rank={0: 2, 1: 2},
    )
    tracker.update_worker_progress(
        _progress(0, ElasticRankState.COMPLETED, completed=2)
    )

    before_release = tracker.snapshot(active_dp_ranks={0})
    after_release = tracker.snapshot(active_dp_ranks=set())

    assert before_release.metrics["completed"] == 2
    assert before_release.metrics["active_dp_ranks"] == [0]
    assert before_release.metrics["completed_dp_ranks"] == [0]
    assert before_release.metrics["resumable_dp_ranks"] == [1]
    assert after_release.metrics["active_dp_ranks"] == []
    assert after_release.metrics["completed_dp_ranks"] == [0]


def test_failed_rank_is_not_synthesized_as_resumable() -> None:
    tracker = ElasticProgressTracker(
        lifecycle_generation=3,
        assigned_trajectories_by_rank={0: 2},
    )
    tracker.update_worker_progress(_progress(0, ElasticRankState.FAILED_RESIDENT))

    snapshot = tracker.snapshot(active_dp_ranks={0})

    assert snapshot.metrics["active_dp_ranks"] == [0]
    assert snapshot.metrics["resumable_dp_ranks"] == []
    assert snapshot.metrics["completed_dp_ranks"] == []


def test_tracker_rejects_stale_generation_assignment_and_counter_regression() -> None:
    tracker = ElasticProgressTracker(
        lifecycle_generation=3,
        assigned_trajectories_by_rank={0: 2},
    )
    with pytest.raises(ValueError, match="lifecycle generation"):
        tracker.update_worker_progress(
            _progress(0, ElasticRankState.ACTIVE, generation=2)
        )
    with pytest.raises(ValueError, match="assignment"):
        tracker.update_worker_progress(
            _progress(0, ElasticRankState.ACTIVE, assigned=1)
        )

    tracker.update_worker_progress(_progress(0, ElasticRankState.ACTIVE, completed=1))
    with pytest.raises(ValueError, match="cannot decrease"):
        tracker.update_worker_progress(
            _progress(0, ElasticRankState.ACTIVE, completed=0)
        )


def test_env_worker_progress_requires_elastic_preparation() -> None:
    worker = object.__new__(EnvWorker)
    worker._rollout_cursor = None

    with pytest.raises(RuntimeError, match="controller-owned"):
        worker.get_elastic_progress()
