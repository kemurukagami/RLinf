"""Per-pipeline orchestration for asynchronous CPU rollout-policy updates."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from rlinf.hybrid_engines.weight_syncer.versioned_cache import (
    PolicyBucketApplyReceipt,
    PolicyVersionReceipt,
)


class AsyncPolicyUpdateService:
    """Prefetch promoted policy versions into inactive CPU rollout models.

    Transfers are started immediately after training and are deliberately not
    awaited by the training stage.  The runner waits for the exact version
    before it submits the next generation GPU request.
    """

    def __init__(
        self,
        *,
        pipeline_id: str,
        actor_cache_owner: Any,
        rollout_workers: dict[int, Any],
        operation_timeout_s: float,
        max_retries: int = 1,
    ) -> None:
        """Validate handles and initialize per-target request coalescing."""
        if actor_cache_owner is None:
            raise ValueError("actor_cache_owner is required")
        if not rollout_workers:
            raise ValueError("rollout_workers must be non-empty")
        if operation_timeout_s <= 0:
            raise ValueError("operation_timeout_s must be positive")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool):
            raise TypeError("max_retries must be an integer")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.pipeline_id = pipeline_id
        self._actor_cache_owner = actor_cache_owner
        self._rollout_workers = dict(rollout_workers)
        self._operation_timeout_s = float(operation_timeout_s)
        self._max_retries = max_retries
        self._inflight: dict[tuple[int, int], asyncio.Task[PolicyVersionReceipt]] = {}
        self._inflight_lock = asyncio.Lock()
        self._last_failure: str | None = None

    @staticmethod
    async def _invoke(target: Any, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method = getattr(target, method_name)
        remote = getattr(method, "remote", None)
        result = (
            remote(*args, **kwargs) if callable(remote) else method(*args, **kwargs)
        )
        if inspect.isawaitable(result):
            return await result
        return result

    async def start_version(self, *, expected_policy_version: int) -> dict[str, object]:
        """Start every rollout-rank update and return serializable status."""
        await self._start_tasks(expected_policy_version=expected_policy_version)
        return self.status()

    async def _start_tasks(
        self, *, expected_policy_version: int
    ) -> dict[int, asyncio.Task[PolicyVersionReceipt]]:
        """Start or reuse background updates for every rollout rank."""
        tasks = {}
        for target_rank in sorted(self._rollout_workers):
            tasks[target_rank] = await self._get_or_create_task(
                target_rank=target_rank,
                expected_policy_version=expected_policy_version,
            )
        return tasks

    async def wait_version(
        self, *, expected_policy_version: int
    ) -> dict[int, PolicyVersionReceipt]:
        """Wait for complete, exact receipts without starting GPU allocation."""
        tasks = await self._start_tasks(expected_policy_version=expected_policy_version)
        try:
            receipts = await asyncio.gather(
                *(asyncio.shield(tasks[rank]) for rank in sorted(tasks))
            )
            return dict(zip(sorted(tasks), receipts, strict=True))
        finally:
            async with self._inflight_lock:
                for rank, task in tasks.items():
                    key = (rank, expected_policy_version)
                    if task.done() and self._inflight.get(key) is task:
                        del self._inflight[key]

    async def _get_or_create_task(
        self, *, target_rank: int, expected_policy_version: int
    ) -> asyncio.Task[PolicyVersionReceipt]:
        """Return the strongly held task for one exact rank/version pair."""
        if target_rank not in self._rollout_workers:
            raise ValueError(f"unknown rollout rank {target_rank}")
        if not isinstance(expected_policy_version, int) or isinstance(
            expected_policy_version, bool
        ):
            raise TypeError("expected_policy_version must be an integer")
        if expected_policy_version < 0:
            raise ValueError("expected_policy_version must be non-negative")
        key = (target_rank, expected_policy_version)
        async with self._inflight_lock:
            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(
                    self._update_version_impl(
                        target_rank=target_rank,
                        expected_policy_version=expected_policy_version,
                    ),
                    name=(
                        f"policy-prefetch-{self.pipeline_id}-{target_rank}-"
                        f"v{expected_policy_version}"
                    ),
                )
                self._inflight[key] = task
        return task

    async def _update_version_impl(
        self, *, target_rank: int, expected_policy_version: int
    ) -> PolicyVersionReceipt:
        worker = self._rollout_workers[target_rank]
        status = await self._invoke(worker, "get_async_policy_status")
        committed = int(status["committed_version"])
        worker_version = int(status["worker_version"])
        if committed != worker_version:
            raise RuntimeError(
                f"rollout rank {target_rank} committed/worker version mismatch: "
                f"committed={committed}, worker={worker_version}"
            )
        if status.get("staging_version") is not None:
            raise RuntimeError(
                f"rollout rank {target_rank} has an unresolved policy transfer"
            )
        if committed == expected_policy_version:
            return PolicyVersionReceipt(
                transfer_id=f"already-current/{target_rank}/{expected_policy_version}",
                prior_version=committed,
                applied_version=committed,
                manifest_hash="",
                bucket_count=0,
                total_bytes=0,
            )
        if committed > expected_policy_version:
            raise RuntimeError(
                f"rollout rank {target_rank} is newer than requested policy: "
                f"committed={committed}, requested={expected_policy_version}"
            )

        for attempt in range(self._max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._transfer_once(
                        worker=worker,
                        target_rank=target_rank,
                        expected_policy_version=expected_policy_version,
                    ),
                    timeout=self._operation_timeout_s,
                )
            except TimeoutError:
                self._last_failure = f"rank {target_rank} policy {expected_policy_version} transfer timed out"
                # The remote mutation may still be completing. Do not issue a
                # second transfer into uncertain receiver state.
                raise
            except Exception as exc:
                self._last_failure = f"{type(exc).__name__}: {exc}"
                if attempt >= self._max_retries:
                    raise
        raise AssertionError("unreachable policy transfer retry state")

    async def _transfer_once(
        self, *, worker: Any, target_rank: int, expected_policy_version: int
    ) -> PolicyVersionReceipt:
        lease = await self._invoke(
            self._actor_cache_owner,
            "acquire_policy_cache",
            expected_policy_version,
        )
        begun = False
        completed = False
        uncertain = False
        try:
            manifest = await self._invoke(
                self._actor_cache_owner, "get_policy_cache_manifest", lease
            )
            if manifest.policy_version != expected_policy_version:
                raise RuntimeError("source policy manifest has the wrong version")
            await self._invoke(
                worker,
                "begin_async_policy_update",
                transfer_id=lease.transfer_id,
                manifest=manifest,
            )
            begun = True
            for bucket_index in range(manifest.bucket_count):
                bucket = await self._invoke(
                    self._actor_cache_owner,
                    "get_policy_cache_bucket",
                    lease,
                    bucket_index,
                )
                receipt = await self._invoke(
                    worker,
                    "apply_async_policy_bucket",
                    transfer_id=lease.transfer_id,
                    bucket=bucket,
                )
                if not isinstance(receipt, PolicyBucketApplyReceipt):
                    raise TypeError("rollout worker returned an invalid bucket receipt")
                expected_descriptor = manifest.bucket_descriptors[bucket_index]
                if (
                    receipt.transfer_id != lease.transfer_id
                    or receipt.policy_version != expected_policy_version
                    or receipt.bucket_index != bucket_index
                    or receipt.checksum != expected_descriptor.checksum
                    or receipt.byte_count != expected_descriptor.byte_count
                ):
                    raise RuntimeError(
                        "rollout worker bucket receipt does not match source"
                    )
            result = await self._invoke(
                worker,
                "commit_async_policy_update",
                transfer_id=lease.transfer_id,
            )
            if not isinstance(result, PolicyVersionReceipt):
                raise TypeError("rollout worker returned an invalid policy receipt")
            if (
                result.transfer_id != lease.transfer_id
                or result.applied_version != expected_policy_version
                or result.manifest_hash != manifest.manifest_hash
                or result.bucket_count != manifest.bucket_count
                or result.total_bytes != manifest.total_bytes
            ):
                raise RuntimeError(
                    "rollout worker policy receipt does not match source"
                )
            completed = True
            return result
        except asyncio.CancelledError:
            # A timed-out Ray method may still be mutating the remote worker.
            # Do not race it with abort or retry; leave staging unresolved so
            # coordinator admission fails closed.
            uncertain = True
            raise
        finally:
            if begun and not completed and not uncertain:
                try:
                    await self._invoke(
                        worker,
                        "abort_async_policy_update",
                        transfer_id=lease.transfer_id,
                    )
                except Exception:
                    # The coordinator will fail closed if retry cannot observe a
                    # clean receiver. Preserve the primary transfer exception.
                    pass
            await self._invoke(self._actor_cache_owner, "release_policy_cache", lease)

    def status(self) -> dict[str, object]:
        """Return bounded service diagnostics without tensor data."""
        return {
            "pipeline_id": self.pipeline_id,
            "in_flight": [
                {"target_rank": rank, "policy_version": version}
                for rank, version in sorted(self._inflight)
            ],
            "last_failure": self._last_failure,
        }

    async def close(self) -> None:
        """Observe every strongly held task before coordinator destruction."""
        async with self._inflight_lock:
            tasks = list(self._inflight.values())
        if tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks), return_exceptions=True
            )
        async with self._inflight_lock:
            self._inflight.clear()


__all__ = ["AsyncPolicyUpdateService"]
