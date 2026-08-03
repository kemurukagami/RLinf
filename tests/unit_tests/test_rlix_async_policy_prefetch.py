from __future__ import annotations

import asyncio

import pytest
import torch

from rlinf.hybrid_engines.weight_syncer.versioned_cache import (
    PolicyCacheReceiver,
    VersionedPolicyCache,
    build_policy_cache,
)
from rlinf.scheduler.rlix.model_update_service import AsyncPolicyUpdateService


class _CacheOwner:
    def __init__(self, state: dict[str, torch.Tensor], version: int) -> None:
        self.cache = VersionedPolicyCache()
        manifest, buckets = build_policy_cache(
            state, policy_version=version, bucket_size_bytes=8
        )
        self.cache.store_candidate(manifest, buckets)
        self.cache.promote(version)
        self.bucket_reads = 0

    def acquire_policy_cache(self, version):
        return self.cache.acquire(version)

    def get_policy_cache_manifest(self, lease):
        return self.cache.manifest_for_lease(lease)

    def get_policy_cache_bucket(self, lease, bucket_index):
        self.bucket_reads += 1
        return self.cache.bucket_for_lease(lease, bucket_index)

    def release_policy_cache(self, lease):
        self.cache.release(lease)


class _Rollout:
    def __init__(
        self,
        state: dict[str, torch.Tensor],
        *,
        version: int,
        event_log: list[str] | None = None,
    ) -> None:
        self.state = state
        self.version = version
        self.receiver = PolicyCacheReceiver(committed_version=version)
        self.begin_count = 0
        self.fail_bucket_once = False
        self.event_log = event_log
        self.begin_entered: asyncio.Event | None = None
        self.begin_release: asyncio.Event | None = None

    def get_async_policy_status(self):
        return {
            "committed_version": self.receiver.committed_version,
            "staging_version": self.receiver.staging_version,
            "worker_version": self.version,
        }

    async def begin_async_policy_update(self, *, transfer_id, manifest):
        self.begin_count += 1
        if self.event_log is not None:
            self.event_log.append("sync_begin")
        if self.begin_entered is not None:
            self.begin_entered.set()
        if self.begin_release is not None:
            await self.begin_release.wait()
        # Let a concurrent ensure reach the coalescing path.
        await asyncio.sleep(0)
        self.receiver.begin(
            transfer_id=transfer_id, manifest=manifest, state_dict=self.state
        )

    def apply_async_policy_bucket(self, *, transfer_id, bucket):
        if self.receiver.transfer_id != transfer_id:
            raise ValueError("wrong transfer")
        if self.fail_bucket_once:
            self.fail_bucket_once = False
            raise RuntimeError("injected bucket failure")
        return self.receiver.apply_bucket(bucket, state_dict=self.state)

    def commit_async_policy_update(self, *, transfer_id):
        if self.receiver.transfer_id != transfer_id:
            raise ValueError("wrong transfer")
        receipt = self.receiver.commit()
        self.version = receipt.applied_version
        if self.event_log is not None:
            self.event_log.append("sync_commit")
        return receipt

    def abort_async_policy_update(self, *, transfer_id):
        self.receiver.abort(transfer_id=transfer_id)


def _source() -> dict[str, torch.Tensor]:
    return {
        "a": torch.arange(8, dtype=torch.bfloat16),
        "b": torch.tensor([3.5, -2.0], dtype=torch.float32),
    }


def test_service_streams_complete_active_version_with_bounded_acknowledgements() -> (
    None
):
    async def run() -> None:
        source = _source()
        owner = _CacheOwner(source, 2)
        rollout = _Rollout(
            {name: torch.zeros_like(value) for name, value in source.items()},
            version=1,
        )
        service = AsyncPolicyUpdateService(
            pipeline_id="pipeline",
            actor_cache_owner=owner,
            rollout_workers={0: rollout},
            operation_timeout_s=1,
        )

        receipt = (await service.wait_version(expected_policy_version=2))[0]

        assert receipt.prior_version == 1
        assert receipt.applied_version == 2
        assert owner.bucket_reads == receipt.bucket_count
        assert owner.cache.status()["in_flight_transfers"] == 0
        for name, value in source.items():
            assert torch.equal(rollout.state[name], value)

    asyncio.run(run())


def test_start_returns_while_all_rank_updates_continue_in_background() -> None:
    async def run() -> None:
        source = _source()
        owner = _CacheOwner(source, 2)
        rollouts = {
            rank: _Rollout(
                {name: torch.zeros_like(value) for name, value in source.items()},
                version=1,
            )
            for rank in (0, 1)
        }
        entered = asyncio.Event()
        release = asyncio.Event()
        for rollout in rollouts.values():
            rollout.begin_entered = entered
            rollout.begin_release = release
        service = AsyncPolicyUpdateService(
            pipeline_id="pipeline",
            actor_cache_owner=owner,
            rollout_workers=rollouts,
            operation_timeout_s=1,
        )

        status = await service.start_version(expected_policy_version=2)
        await entered.wait()

        assert {item["target_rank"] for item in status["in_flight"]} == {0, 1}
        assert len(service.status()["in_flight"]) == 2

        release.set()
        receipts = await service.wait_version(expected_policy_version=2)
        assert {
            rank: receipt.applied_version for rank, receipt in receipts.items()
        } == {
            0: 2,
            1: 2,
        }
        assert service.status()["in_flight"] == []

    asyncio.run(run())


def test_same_version_is_idempotent_and_does_not_lease_or_read_cache() -> None:
    async def run() -> None:
        source = _source()
        owner = _CacheOwner(source, 2)
        rollout = _Rollout(dict(source), version=2)
        service = AsyncPolicyUpdateService(
            pipeline_id="pipeline",
            actor_cache_owner=owner,
            rollout_workers={0: rollout},
            operation_timeout_s=1,
        )

        receipt = (await service.wait_version(expected_policy_version=2))[0]

        assert receipt.bucket_count == 0
        assert rollout.begin_count == 0
        assert owner.bucket_reads == 0

    asyncio.run(run())


def test_duplicate_start_and_wait_are_coalesced() -> None:
    async def run() -> None:
        source = _source()
        owner = _CacheOwner(source, 3)
        rollout = _Rollout(
            {name: torch.zeros_like(value) for name, value in source.items()},
            version=2,
        )
        service = AsyncPolicyUpdateService(
            pipeline_id="pipeline",
            actor_cache_owner=owner,
            rollout_workers={0: rollout},
            operation_timeout_s=1,
        )

        await service.start_version(expected_policy_version=3)
        first, second = await asyncio.gather(
            service.wait_version(expected_policy_version=3),
            service.wait_version(expected_policy_version=3),
        )

        assert first[0] == second[0]
        assert rollout.begin_count == 1

    asyncio.run(run())


def test_bucket_failure_aborts_and_retries_from_bucket_zero() -> None:
    async def run() -> None:
        source = _source()
        owner = _CacheOwner(source, 4)
        rollout = _Rollout(
            {name: torch.zeros_like(value) for name, value in source.items()},
            version=3,
        )
        rollout.fail_bucket_once = True
        service = AsyncPolicyUpdateService(
            pipeline_id="pipeline",
            actor_cache_owner=owner,
            rollout_workers={0: rollout},
            operation_timeout_s=1,
            max_retries=1,
        )

        receipt = (await service.wait_version(expected_policy_version=4))[0]

        assert receipt.applied_version == 4
        assert rollout.begin_count == 2
        assert rollout.receiver.staging_version is None
        assert owner.cache.status()["in_flight_transfers"] == 0

    asyncio.run(run())


def test_newer_receiver_fails_closed() -> None:
    async def run() -> None:
        source = _source()
        owner = _CacheOwner(source, 2)
        rollout = _Rollout(dict(source), version=3)
        service = AsyncPolicyUpdateService(
            pipeline_id="pipeline",
            actor_cache_owner=owner,
            rollout_workers={0: rollout},
            operation_timeout_s=1,
        )

        with pytest.raises(RuntimeError, match="newer"):
            await service.wait_version(expected_policy_version=2)

    asyncio.run(run())


def test_service_completes_before_caller_requests_activation() -> None:
    async def run() -> None:
        source = _source()
        events: list[str] = []
        owner = _CacheOwner(source, 1)
        rollout = _Rollout(
            {name: torch.zeros_like(value) for name, value in source.items()},
            version=0,
            event_log=events,
        )
        service = AsyncPolicyUpdateService(
            pipeline_id="pipeline",
            actor_cache_owner=owner,
            rollout_workers={0: rollout},
            operation_timeout_s=1,
        )

        await service.start_version(expected_policy_version=1)
        await service.wait_version(expected_policy_version=1)
        events.append("prepare")

        assert events == ["sync_begin", "sync_commit", "prepare"]

    asyncio.run(run())
