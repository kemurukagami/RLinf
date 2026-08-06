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
    capture_policy_cache,
    finalize_policy_capture,
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


def test_capture_is_cpu_owned_and_finalization_preserves_exact_bytes() -> None:
    source = _state(8)
    capture = capture_policy_cache(
        source,
        policy_version=4,
        bucket_size_bytes=19,
    )

    assert capture.policy_version == 4
    assert capture.total_bytes == sum(
        value.numel() * value.element_size() for value in source.values()
    )
    assert all(bucket.payload.device.type == "cpu" for bucket in capture.buckets)
    assert all(bucket.payload.dtype is torch.uint8 for bucket in capture.buckets)

    manifest, buckets = finalize_policy_capture(capture)
    target = {name: torch.zeros_like(value) for name, value in source.items()}
    receiver = PolicyCacheReceiver(committed_version=3)
    _apply_all(receiver, target, manifest, buckets)

    assert manifest.total_bytes == capture.total_bytes
    for name, value in source.items():
        assert torch.equal(target[name], value)


def test_finalization_rejects_inconsistent_capture_total() -> None:
    capture = capture_policy_cache(_state(), policy_version=1, bucket_size_bytes=16)
    malformed = replace(capture, total_bytes=capture.total_bytes + 1)

    with pytest.raises(ValueError, match="total byte count"):
        finalize_policy_capture(malformed)


def test_finalization_rejects_capture_fragment_gap_before_hashing() -> None:
    capture = capture_policy_cache(_state(), policy_version=1, bucket_size_bytes=16)
    first_bucket = capture.buckets[0]
    first_fragment = first_bucket.tensors[0]
    malformed_fragment = replace(
        first_fragment,
        start_byte=first_fragment.start_byte + 1,
        end_byte=first_fragment.end_byte + 1,
    )
    malformed_bucket = replace(
        first_bucket,
        tensors=(malformed_fragment, *first_bucket.tensors[1:]),
    )
    malformed = replace(
        capture,
        buckets=(malformed_bucket, *capture.buckets[1:]),
    )

    with pytest.raises(ValueError, match="gap or overlap"):
        finalize_policy_capture(malformed)


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


@pytest.mark.parametrize("bucket_size", [1, 2, 3, 7, 16, 31, 128])
def test_fragmented_roundtrip_is_exact_across_bucket_boundaries(bucket_size) -> None:
    source = {
        "bf16": torch.tensor(
            [1.5, -0.25, float("inf"), float("nan")], dtype=torch.bfloat16
        ),
        "bool": torch.tensor([True, False, True], dtype=torch.bool),
        "fp16": torch.linspace(-1, 1, 7, dtype=torch.float16),
        "fp32": torch.linspace(-2, 2, 11, dtype=torch.float32).reshape(1, 11),
        "int64": torch.tensor([-(2**40), 0, 2**40], dtype=torch.int64),
    }
    manifest, buckets = build_policy_cache(
        source,
        policy_version=9,
        bucket_size_bytes=bucket_size,
    )
    target = {name: torch.zeros_like(value) for name, value in source.items()}
    receiver = PolicyCacheReceiver(committed_version=8)

    _apply_all(receiver, target, manifest, buckets)

    assert all(bucket.payload.numel() <= bucket_size for bucket in buckets)
    for name, value in source.items():
        assert torch.equal(
            target[name].reshape(-1).view(torch.uint8),
            value.reshape(-1).view(torch.uint8),
        )


def test_large_tensor_is_split_into_bounded_exact_fragments() -> None:
    source = {
        "large": torch.arange(32, dtype=torch.float32),
        "small": torch.ones(1, dtype=torch.float32),
    }
    manifest, buckets = build_policy_cache(
        source, policy_version=1, bucket_size_bytes=16
    )

    large_descriptors = [
        tensor
        for bucket in manifest.bucket_descriptors
        for tensor in bucket.tensors
        if tensor.name == "large"
    ]
    assert len(large_descriptors) == 8
    assert all(bucket.descriptor.byte_count <= 16 for bucket in buckets)
    assert [
        (descriptor.tensor_start_byte, descriptor.tensor_end_byte)
        for descriptor in large_descriptors
    ] == [(start, start + 16) for start in range(0, 128, 16)]

    target = {name: torch.zeros_like(value) for name, value in source.items()}
    receiver = PolicyCacheReceiver(committed_version=0)
    _apply_all(receiver, target, manifest, buckets)
    for name, value in source.items():
        assert torch.equal(target[name], value)


def test_builder_copies_directly_without_tensor_concatenation(monkeypatch) -> None:
    def reject_cat(*args, **kwargs):
        raise AssertionError("policy-cache builder must not call torch.cat")

    monkeypatch.setattr(torch, "cat", reject_cat)

    manifest, buckets = build_policy_cache(
        _state(5), policy_version=1, bucket_size_bytes=17
    )

    assert manifest.bucket_count == len(buckets)
    assert all(bucket.payload.is_contiguous() for bucket in buckets)
    assert all(bucket.descriptor.byte_count <= 17 for bucket in buckets)


def test_empty_and_scalar_tensors_round_trip() -> None:
    source = {
        "empty": torch.empty(0, dtype=torch.float32),
        "scalar": torch.tensor(7, dtype=torch.int64),
    }
    manifest, buckets = build_policy_cache(
        source, policy_version=1, bucket_size_bytes=3
    )
    target = {name: torch.zeros_like(value) for name, value in source.items()}
    receiver = PolicyCacheReceiver(committed_version=0)

    _apply_all(receiver, target, manifest, buckets)

    assert target["empty"].shape == (0,)
    assert torch.equal(target["scalar"], source["scalar"])


def test_noncontiguous_cpu_target_is_staged_and_committed() -> None:
    source = {"policy": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    manifest, buckets = build_policy_cache(
        source, policy_version=2, bucket_size_bytes=7
    )
    backing = torch.zeros(4, 3, dtype=torch.float32)
    target = {"policy": backing.transpose(0, 1)}
    assert not target["policy"].is_contiguous()
    receiver = PolicyCacheReceiver(committed_version=1)

    _apply_all(receiver, target, manifest, buckets)

    assert torch.equal(target["policy"], source["policy"])


def test_aborted_noncontiguous_target_remains_unmodified() -> None:
    source = {"policy": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
    manifest, buckets = build_policy_cache(
        source, policy_version=2, bucket_size_bytes=7
    )
    backing = torch.zeros(4, 3, dtype=torch.float32)
    target = {"policy": backing.transpose(0, 1)}
    receiver = PolicyCacheReceiver(committed_version=1)
    receiver.begin(
        transfer_id="abort-noncontiguous",
        manifest=manifest,
        state_dict=target,
    )

    receiver.apply_bucket(buckets[0], state_dict=target)
    receiver.abort(transfer_id="abort-noncontiguous")

    assert torch.count_nonzero(target["policy"]) == 0
    assert receiver.staging_version is None


def test_receiver_rejects_tensor_fragment_gap_before_mutation() -> None:
    manifest, _ = build_policy_cache(
        {"large": torch.arange(12, dtype=torch.float32)},
        policy_version=1,
        bucket_size_bytes=16,
    )
    second_bucket = manifest.bucket_descriptors[1]
    second_fragment = second_bucket.tensors[0]
    malformed_fragment = replace(
        second_fragment,
        tensor_start_byte=second_fragment.tensor_start_byte + 1,
        tensor_end_byte=second_fragment.tensor_end_byte + 1,
    )
    malformed_bucket = replace(second_bucket, tensors=(malformed_fragment,))
    malformed_manifest = replace(
        manifest,
        bucket_descriptors=(
            manifest.bucket_descriptors[0],
            malformed_bucket,
            manifest.bucket_descriptors[2],
        ),
    )
    target = {"large": torch.zeros(12, dtype=torch.float32)}
    receiver = PolicyCacheReceiver(committed_version=0)

    with pytest.raises(ValueError, match="gap or overlap"):
        receiver.begin(
            transfer_id="gap",
            manifest=malformed_manifest,
            state_dict=target,
        )

    assert torch.count_nonzero(target["large"]) == 0
    assert receiver.staging_version is None


def test_receiver_rejects_bucket_fragment_overlap_before_mutation() -> None:
    manifest, _ = build_policy_cache(
        {
            "first": torch.arange(2, dtype=torch.float32),
            "second": torch.arange(2, dtype=torch.float32),
        },
        policy_version=1,
        bucket_size_bytes=32,
    )
    bucket = manifest.bucket_descriptors[0]
    first, second = bucket.tensors
    malformed_second = replace(
        second,
        start_byte=first.end_byte - 1,
        end_byte=second.end_byte - 1,
    )
    malformed_manifest = replace(
        manifest,
        bucket_descriptors=(replace(bucket, tensors=(first, malformed_second)),),
    )
    target = {
        "first": torch.zeros(2, dtype=torch.float32),
        "second": torch.zeros(2, dtype=torch.float32),
    }
    receiver = PolicyCacheReceiver(committed_version=0)

    with pytest.raises(ValueError, match="gap or overlap"):
        receiver.begin(
            transfer_id="overlap",
            manifest=malformed_manifest,
            state_dict=target,
        )

    assert receiver.staging_version is None


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

    oversized = (
        replace(
            buckets[0],
            payload=torch.cat((buckets[0].payload, torch.zeros(1, dtype=torch.uint8))),
        ),
        *buckets[1:],
    )
    with pytest.raises(ValueError, match="payload size"):
        cache.store_candidate(manifest, oversized)
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
