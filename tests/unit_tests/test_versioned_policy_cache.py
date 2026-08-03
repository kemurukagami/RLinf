# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from rlinf.hybrid_engines.weight_syncer.versioned_cache import (
    PolicyBucket,
    PolicyCacheReceiver,
    VersionedPolicyCache,
    build_policy_cache,
)


def _state(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "bias": torch.randn(5, generator=generator, dtype=torch.float32),
        "counter": torch.arange(3, dtype=torch.int64),
        "policy": torch.randn(3, 4, generator=generator, dtype=torch.bfloat16),
    }


def _cache(version: int, *, seed: int = 0, bucket_size: int = 24):
    return build_policy_cache(
        _state(seed),
        policy_version=version,
        bucket_size_bytes=bucket_size,
    )


def _apply_all(
    receiver: PolicyCacheReceiver,
    target: dict[str, torch.Tensor],
    manifest,
    buckets,
    *,
    transfer_id: str = "transfer",
):
    receiver.begin(
        transfer_id=transfer_id,
        manifest=manifest,
        state_dict=target,
    )
    receipts = [receiver.apply_bucket(bucket, state_dict=target) for bucket in buckets]
    return receipts, receiver.commit()


def test_build_and_apply_preserves_mixed_dtype_bytes_exactly() -> None:
    source = _state(7)
    manifest, buckets = build_policy_cache(
        source,
        policy_version=3,
        bucket_size_bytes=20,
    )
    target = {name: torch.zeros_like(value) for name, value in source.items()}
    receiver = PolicyCacheReceiver(committed_version=2)

    receipts, receipt = _apply_all(receiver, target, manifest, buckets)

    assert manifest.bucket_count >= 2
    assert [item.bucket_index for item in receipts] == list(
        range(manifest.bucket_count)
    )
    assert receipt.prior_version == 2
    assert receipt.applied_version == 3
    assert receipt.manifest_hash == manifest.manifest_hash
    assert receiver.committed_version == 3
    assert receiver.staging_version is None
    for name in source:
        assert target[name].dtype == source[name].dtype
        assert torch.equal(target[name], source[name])


def test_manifest_is_deterministic_independent_of_mapping_order() -> None:
    source = _state(4)
    reversed_source = dict(reversed(list(source.items())))
    first, first_buckets = build_policy_cache(
        source, policy_version=1, bucket_size_bytes=31
    )
    second, second_buckets = build_policy_cache(
        reversed_source, policy_version=1, bucket_size_bytes=31
    )

    assert first == second
    assert [bucket.payload.tolist() for bucket in first_buckets] == [
        bucket.payload.tolist() for bucket in second_buckets
    ]


def test_large_tensor_is_not_split_across_buckets() -> None:
    source = {
        "large": torch.arange(32, dtype=torch.float32),
        "small": torch.ones(1, dtype=torch.float32),
    }
    manifest, _ = build_policy_cache(source, policy_version=1, bucket_size_bytes=16)

    large_descriptors = [
        tensor
        for bucket in manifest.bucket_descriptors
        for tensor in bucket.tensors
        if tensor.name == "large"
    ]
    assert len(large_descriptors) == 1
    assert large_descriptors[0].byte_count == 128


@pytest.mark.parametrize("version", [-1, True, 1.5])
def test_invalid_policy_version_is_rejected(version) -> None:
    with pytest.raises((TypeError, ValueError)):
        build_policy_cache(_state(), policy_version=version, bucket_size_bytes=16)


def test_missing_selected_name_fails_before_publication() -> None:
    with pytest.raises(KeyError, match="missing"):
        build_policy_cache(
            _state(),
            policy_version=1,
            bucket_size_bytes=16,
            selected_names=["policy", "missing"],
        )


def test_candidate_must_be_complete_and_checksum_valid() -> None:
    manifest, buckets = _cache(1)
    cache = VersionedPolicyCache()
    with pytest.raises(ValueError, match="count"):
        cache.store_candidate(manifest, buckets[:-1])

    corrupted_payload = buckets[0].payload.clone()
    corrupted_payload[0] ^= 1
    corrupted = (replace(buckets[0], payload=corrupted_payload), *buckets[1:])
    with pytest.raises(ValueError, match="checksum"):
        cache.store_candidate(manifest, corrupted)
    assert cache.status()["active_version"] is None
    assert cache.status()["candidate_version"] is None


def test_promotion_is_atomic_and_only_accepts_current_candidate() -> None:
    cache = VersionedPolicyCache()
    manifest, buckets = _cache(1)
    cache.store_candidate(manifest, buckets)

    with pytest.raises(ValueError, match="current candidate"):
        cache.promote(2)
    assert cache.active_manifest() is None

    promoted = cache.promote(1)
    assert promoted == manifest
    assert cache.active_manifest() == manifest
    with pytest.raises(ValueError, match="newer"):
        cache.store_candidate(manifest, buckets)


def test_transfer_lease_prevents_gc_until_release() -> None:
    cache = VersionedPolicyCache(max_cached_versions=1)
    manifest_one, buckets_one = _cache(1, seed=1)
    cache.store_candidate(manifest_one, buckets_one)
    cache.promote(1)
    lease = cache.acquire(1)

    manifest_two, buckets_two = _cache(2, seed=2)
    cache.store_candidate(manifest_two, buckets_two)
    cache.promote(2)
    assert cache.status()["cached_versions"] == [1, 2]
    assert cache.bucket_for_lease(lease, 0) == buckets_one[0]

    cache.release(lease)
    assert cache.status()["cached_versions"] == [2]
    with pytest.raises(ValueError, match="not active"):
        cache.manifest_for_lease(lease)


def test_receiver_rejects_schema_mismatch_before_mutation() -> None:
    manifest, _ = _cache(1)
    target = _state(9)
    target["policy"] = torch.zeros(3, 5, dtype=torch.bfloat16)
    receiver = PolicyCacheReceiver(committed_version=0)

    with pytest.raises(ValueError, match="shape mismatch"):
        receiver.begin(
            transfer_id="bad-schema",
            manifest=manifest,
            state_dict=target,
        )
    assert receiver.committed_version == 0
    assert receiver.staging_version is None


def test_receiver_rejects_non_cpu_target() -> None:
    manifest, _ = _cache(1)
    target = _state(9)
    target["policy"] = torch.empty(3, 4, dtype=torch.bfloat16, device="meta")
    receiver = PolicyCacheReceiver(committed_version=0)

    with pytest.raises(ValueError, match="CPU-offloaded"):
        receiver.begin(
            transfer_id="resident",
            manifest=manifest,
            state_dict=target,
        )


def test_receiver_does_not_commit_incomplete_or_out_of_order_transfer() -> None:
    manifest, buckets = _cache(1, bucket_size=8)
    assert len(buckets) > 1
    target = {name: torch.zeros_like(value) for name, value in _state().items()}
    receiver = PolicyCacheReceiver(committed_version=0)
    receiver.begin(transfer_id="partial", manifest=manifest, state_dict=target)

    with pytest.raises(ValueError, match="order mismatch"):
        receiver.apply_bucket(buckets[1], state_dict=target)
    receiver.apply_bucket(buckets[0], state_dict=target)
    with pytest.raises(RuntimeError, match="incomplete"):
        receiver.commit()
    assert receiver.committed_version == 0
    assert receiver.staging_version == 1


def test_corrupted_bucket_fails_without_advancing_version() -> None:
    manifest, buckets = _cache(1)
    target = {name: torch.zeros_like(value) for name, value in _state().items()}
    receiver = PolicyCacheReceiver(committed_version=0)
    receiver.begin(transfer_id="corrupt", manifest=manifest, state_dict=target)
    payload = buckets[0].payload.clone()
    payload[-1] ^= 1

    with pytest.raises(ValueError, match="checksum"):
        receiver.apply_bucket(
            PolicyBucket(descriptor=buckets[0].descriptor, payload=payload),
            state_dict=target,
        )
    assert receiver.committed_version == 0


def test_abort_then_complete_retry_overwrites_partial_model() -> None:
    source = _state(12)
    manifest, buckets = build_policy_cache(
        source, policy_version=2, bucket_size_bytes=8
    )
    target = {name: torch.zeros_like(value) for name, value in source.items()}
    receiver = PolicyCacheReceiver(committed_version=1)
    receiver.begin(transfer_id="first", manifest=manifest, state_dict=target)
    receiver.apply_bucket(buckets[0], state_dict=target)
    receiver.abort(transfer_id="first")

    _, receipt = _apply_all(
        receiver,
        target,
        manifest,
        buckets,
        transfer_id="retry",
    )

    assert receipt.applied_version == 2
    for name, value in source.items():
        assert torch.equal(target[name], value)


def test_same_or_older_receiver_version_is_rejected() -> None:
    manifest, _ = _cache(2)
    receiver = PolicyCacheReceiver(committed_version=2)
    with pytest.raises(ValueError, match="newer"):
        receiver.begin(
            transfer_id="stale",
            manifest=manifest,
            state_dict=_state(),
        )


def test_exact_lease_and_bucket_bounds_are_enforced() -> None:
    cache = VersionedPolicyCache()
    manifest, buckets = _cache(1)
    cache.store_candidate(manifest, buckets)
    cache.promote(1)
    lease = cache.acquire(1)

    with pytest.raises(IndexError, match="out of range"):
        cache.bucket_for_lease(lease, manifest.bucket_count)
    cache.release(lease)
    with pytest.raises(ValueError, match="not active"):
        cache.release(lease)
