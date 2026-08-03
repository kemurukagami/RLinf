# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replayable CPU policy buckets for asynchronous RLix rollout updates."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch.distributed.tensor import DTensor

from rlinf.utils.utils import materialize_tensor

_FORMAT_VERSION = 1


def _require_policy_version(policy_version: int) -> int:
    if not isinstance(policy_version, int) or isinstance(policy_version, bool):
        raise TypeError("policy_version must be an integer")
    if policy_version < 0:
        raise ValueError("policy_version must be non-negative")
    return policy_version


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    if not name.startswith("torch."):
        raise TypeError(f"unsupported tensor dtype {dtype!r}")
    return name.removeprefix("torch.")


def _resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"unsupported policy-cache dtype {name!r}")
    return dtype


def _sha256_bytes(payload: torch.Tensor) -> str:
    if payload.device.type != "cpu" or payload.dtype is not torch.uint8:
        raise TypeError("policy bucket checksum requires a CPU uint8 tensor")
    return hashlib.sha256(payload.contiguous().numpy().tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyTensorDescriptor:
    """Location and interpretation of one tensor within a byte bucket."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    start_byte: int
    end_byte: int

    @property
    def byte_count(self) -> int:
        return self.end_byte - self.start_byte


@dataclass(frozen=True, slots=True)
class PolicyBucketDescriptor:
    """Immutable metadata and checksum for one policy bucket."""

    index: int
    byte_count: int
    tensors: tuple[PolicyTensorDescriptor, ...]
    checksum: str


@dataclass(frozen=True, slots=True)
class PolicyCacheManifest:
    """Identity and ordered layout of one complete policy version."""

    policy_version: int
    format_version: int
    model_schema_hash: str
    bucket_descriptors: tuple[PolicyBucketDescriptor, ...]
    total_bytes: int
    manifest_hash: str

    @property
    def bucket_count(self) -> int:
        return len(self.bucket_descriptors)


@dataclass(frozen=True, slots=True)
class PolicyBucket:
    """One transportable CPU byte bucket and its exact descriptor."""

    descriptor: PolicyBucketDescriptor
    payload: torch.Tensor


@dataclass(frozen=True, slots=True)
class PolicyCacheBuildReceipt:
    """One actor rank's participation in candidate construction."""

    actor_rank: int
    policy_version: int
    retained: bool
    manifest_hash: str | None
    bucket_count: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class PolicyCachePromotionReceipt:
    """Evidence that an owner made one complete version active."""

    actor_rank: int
    policy_version: int
    promoted: bool
    manifest_hash: str | None


@dataclass(frozen=True, slots=True)
class PolicyTransferLease:
    """Reference preventing a source policy version from being reclaimed."""

    transfer_id: str
    policy_version: int
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class PolicyBucketApplyReceipt:
    """Receiver acknowledgement for exactly one verified bucket."""

    transfer_id: str
    policy_version: int
    bucket_index: int
    checksum: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class PolicyVersionReceipt:
    """Receiver evidence for a fully applied policy version."""

    transfer_id: str
    prior_version: int
    applied_version: int
    manifest_hash: str
    bucket_count: int
    total_bytes: int


def _schema_payload(
    bucket_descriptors: Iterable[PolicyBucketDescriptor],
) -> list[dict[str, object]]:
    return [
        {
            "index": descriptor.index,
            "byte_count": descriptor.byte_count,
            "tensors": [
                {
                    "name": tensor.name,
                    "shape": list(tensor.shape),
                    "dtype": tensor.dtype,
                    "start_byte": tensor.start_byte,
                    "end_byte": tensor.end_byte,
                }
                for tensor in descriptor.tensors
            ],
        }
        for descriptor in bucket_descriptors
    ]


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_policy_cache(
    state_dict: Mapping[str, torch.Tensor | DTensor],
    *,
    policy_version: int,
    bucket_size_bytes: int,
    selected_names: Iterable[str] | None = None,
) -> tuple[PolicyCacheManifest, tuple[PolicyBucket, ...]]:
    """Materialize a complete state dict into deterministic CPU byte buckets.

    Every caller participating in an FSDP/DTensor collective must invoke this
    function with the same names and ordering. The returned payload is always
    CPU-owned and preserves the exact source dtype bytes, including BF16.
    """

    policy_version = _require_policy_version(policy_version)
    if not isinstance(bucket_size_bytes, int) or isinstance(bucket_size_bytes, bool):
        raise TypeError("bucket_size_bytes must be an integer")
    if bucket_size_bytes <= 0:
        raise ValueError("bucket_size_bytes must be positive")

    names = (
        sorted(state_dict) if selected_names is None else sorted(set(selected_names))
    )
    if not names:
        raise ValueError("policy cache requires at least one tensor")
    missing = [name for name in names if name not in state_dict]
    if missing:
        raise KeyError(f"policy cache names missing from state dict: {missing}")

    buckets: list[PolicyBucket] = []
    pending: list[tuple[str, torch.Tensor]] = []
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        byte_views: list[torch.Tensor] = []
        tensor_descriptors: list[PolicyTensorDescriptor] = []
        offset = 0
        for name, tensor in pending:
            contiguous = tensor.detach().contiguous()
            byte_view = (
                contiguous.view(torch.uint8)
                .reshape(-1)
                .to(device="cpu", non_blocking=False)
                .contiguous()
            )
            byte_views.append(byte_view)
            end = offset + byte_view.numel()
            tensor_descriptors.append(
                PolicyTensorDescriptor(
                    name=name,
                    shape=tuple(contiguous.shape),
                    dtype=_dtype_name(contiguous.dtype),
                    start_byte=offset,
                    end_byte=end,
                )
            )
            offset = end
        payload = torch.cat(byte_views).contiguous()
        descriptor = PolicyBucketDescriptor(
            index=len(buckets),
            byte_count=payload.numel(),
            tensors=tuple(tensor_descriptors),
            checksum=_sha256_bytes(payload),
        )
        buckets.append(PolicyBucket(descriptor=descriptor, payload=payload))
        pending = []
        pending_bytes = 0

    for name in names:
        value = state_dict[name]
        if not isinstance(value, (torch.Tensor, DTensor)):
            raise TypeError(f"policy cache value {name!r} is not a tensor")
        materialized = materialize_tensor(value)
        value_bytes = materialized.numel() * materialized.element_size()
        if pending and pending_bytes + value_bytes > bucket_size_bytes:
            flush()
        pending.append((name, materialized))
        pending_bytes += value_bytes
    flush()

    descriptors = tuple(bucket.descriptor for bucket in buckets)
    schema_payload = _schema_payload(descriptors)
    model_schema_hash = _hash_json(schema_payload)
    manifest_payload = {
        "policy_version": policy_version,
        "format_version": _FORMAT_VERSION,
        "model_schema_hash": model_schema_hash,
        "total_bytes": sum(item.byte_count for item in descriptors),
        "buckets": [
            {**schema, "checksum": descriptor.checksum}
            for schema, descriptor in zip(schema_payload, descriptors, strict=True)
        ],
    }
    manifest = PolicyCacheManifest(
        policy_version=policy_version,
        format_version=_FORMAT_VERSION,
        model_schema_hash=model_schema_hash,
        bucket_descriptors=descriptors,
        total_bytes=int(manifest_payload["total_bytes"]),
        manifest_hash=_hash_json(manifest_payload),
    )
    return manifest, tuple(buckets)


class VersionedPolicyCache:
    """Thread-safe candidate/active CPU cache with transfer-aware GC."""

    def __init__(self, *, max_cached_versions: int = 2) -> None:
        if not isinstance(max_cached_versions, int) or isinstance(
            max_cached_versions, bool
        ):
            raise TypeError("max_cached_versions must be an integer")
        if max_cached_versions < 1:
            raise ValueError("max_cached_versions must be positive")
        self._max_cached_versions = max_cached_versions
        self._entries: dict[
            int, tuple[PolicyCacheManifest, tuple[PolicyBucket, ...]]
        ] = {}
        self._candidate_version: int | None = None
        self._active_version: int | None = None
        self._leases: dict[str, PolicyTransferLease] = {}
        self._lock = threading.RLock()

    def store_candidate(
        self, manifest: PolicyCacheManifest, buckets: tuple[PolicyBucket, ...]
    ) -> None:
        """Validate and store a complete unpublished version."""
        if not isinstance(manifest, PolicyCacheManifest):
            raise TypeError("manifest must be a PolicyCacheManifest")
        if not isinstance(buckets, tuple):
            raise TypeError("buckets must be a tuple")
        if len(buckets) != manifest.bucket_count:
            raise ValueError("candidate bucket count does not match manifest")
        for index, (bucket, expected) in enumerate(
            zip(buckets, manifest.bucket_descriptors, strict=True)
        ):
            if bucket.descriptor != expected or expected.index != index:
                raise ValueError("candidate bucket descriptor does not match manifest")
            if _sha256_bytes(bucket.payload) != expected.checksum:
                raise ValueError(f"candidate bucket {index} checksum mismatch")
        with self._lock:
            if (
                self._active_version is not None
                and manifest.policy_version <= self._active_version
            ):
                raise ValueError("candidate version must be newer than active version")
            self._entries[manifest.policy_version] = (manifest, buckets)
            self._candidate_version = manifest.policy_version
            self._garbage_collect_locked()

    def promote(self, policy_version: int) -> PolicyCacheManifest:
        """Atomically promote an existing complete candidate."""
        policy_version = _require_policy_version(policy_version)
        with self._lock:
            if policy_version != self._candidate_version:
                raise ValueError("only the current candidate may be promoted")
            if policy_version not in self._entries:
                raise RuntimeError("candidate cache is missing")
            self._active_version = policy_version
            self._candidate_version = None
            self._garbage_collect_locked()
            return self._entries[policy_version][0]

    def active_manifest(self) -> PolicyCacheManifest | None:
        """Return the immutable active manifest, if one has been promoted."""
        with self._lock:
            if self._active_version is None:
                return None
            return self._entries[self._active_version][0]

    def acquire(self, policy_version: int) -> PolicyTransferLease:
        """Lease the exact active version for bounded replay."""
        policy_version = _require_policy_version(policy_version)
        with self._lock:
            if policy_version != self._active_version:
                raise ValueError(
                    f"requested policy version {policy_version} is not active; "
                    f"active={self._active_version}"
                )
            manifest = self._entries[policy_version][0]
            lease = PolicyTransferLease(
                transfer_id=uuid.uuid4().hex,
                policy_version=policy_version,
                manifest_hash=manifest.manifest_hash,
            )
            self._leases[lease.transfer_id] = lease
            return lease

    def manifest_for_lease(self, lease: PolicyTransferLease) -> PolicyCacheManifest:
        """Return source metadata after validating an exact live lease."""
        with self._lock:
            current = self._leases.get(lease.transfer_id)
            if current != lease:
                raise ValueError("policy transfer lease is not active")
            return self._entries[lease.policy_version][0]

    def bucket_for_lease(
        self, lease: PolicyTransferLease, bucket_index: int
    ) -> PolicyBucket:
        """Return one immutable bucket under an exact live lease."""
        if not isinstance(bucket_index, int) or isinstance(bucket_index, bool):
            raise TypeError("bucket_index must be an integer")
        with self._lock:
            current = self._leases.get(lease.transfer_id)
            if current != lease:
                raise ValueError("policy transfer lease is not active")
            buckets = self._entries[lease.policy_version][1]
            if bucket_index < 0 or bucket_index >= len(buckets):
                raise IndexError("policy bucket index is out of range")
            return buckets[bucket_index]

    def release(self, lease: PolicyTransferLease) -> None:
        """Release an exact transfer lease and run cache GC."""
        with self._lock:
            if self._leases.get(lease.transfer_id) != lease:
                raise ValueError("policy transfer lease is not active")
            del self._leases[lease.transfer_id]
            self._garbage_collect_locked()

    def status(self) -> dict[str, object]:
        """Return JSON-compatible cache diagnostics."""
        with self._lock:
            return {
                "active_version": self._active_version,
                "candidate_version": self._candidate_version,
                "cached_versions": sorted(self._entries),
                "cached_bytes": sum(
                    entry[0].total_bytes for entry in self._entries.values()
                ),
                "leased_versions": sorted(
                    lease.policy_version for lease in self._leases.values()
                ),
                "in_flight_transfers": len(self._leases),
            }

    def _garbage_collect_locked(self) -> None:
        protected = {
            version
            for version in (self._active_version, self._candidate_version)
            if version is not None
        }
        protected.update(lease.policy_version for lease in self._leases.values())
        removable = sorted(set(self._entries) - protected)
        while len(self._entries) > self._max_cached_versions and removable:
            del self._entries[removable.pop(0)]


class PolicyCacheReceiver:
    """Externally atomic transaction for an inactive CPU model state dict."""

    def __init__(self, *, committed_version: int) -> None:
        self._committed_version = _require_policy_version(committed_version)
        self._manifest: PolicyCacheManifest | None = None
        self._transfer_id: str | None = None
        self._received: set[int] = set()
        self._received_bytes = 0

    @property
    def committed_version(self) -> int:
        return self._committed_version

    @property
    def staging_version(self) -> int | None:
        return self._manifest.policy_version if self._manifest is not None else None

    @property
    def transfer_id(self) -> str | None:
        return self._transfer_id

    def begin(
        self,
        *,
        transfer_id: str,
        manifest: PolicyCacheManifest,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Open a new complete-version transaction after schema validation."""
        if not isinstance(transfer_id, str) or not transfer_id:
            raise ValueError("transfer_id must be a non-empty string")
        if self._manifest is not None:
            raise RuntimeError("a policy receive transaction is already active")
        if manifest.policy_version <= self._committed_version:
            raise ValueError(
                "received policy version must be newer than committed version"
            )
        for descriptor in manifest.bucket_descriptors:
            for tensor in descriptor.tensors:
                target = state_dict.get(tensor.name)
                if target is None:
                    raise KeyError(f"rollout model lacks policy tensor {tensor.name!r}")
                if tuple(target.shape) != tensor.shape:
                    raise ValueError(
                        f"shape mismatch for policy tensor {tensor.name!r}"
                    )
                if _dtype_name(target.dtype) != tensor.dtype:
                    raise ValueError(
                        f"dtype mismatch for policy tensor {tensor.name!r}"
                    )
                if target.device.type != "cpu":
                    raise ValueError("policy receiver requires CPU-offloaded tensors")
        self._manifest = manifest
        self._transfer_id = transfer_id
        self._received = set()
        self._received_bytes = 0

    @torch.no_grad()
    def apply_bucket(
        self,
        bucket: PolicyBucket,
        *,
        state_dict: Mapping[str, torch.Tensor],
    ) -> PolicyBucketApplyReceipt:
        """Verify and apply the next exact bucket to inactive CPU parameters."""
        manifest = self._manifest
        transfer_id = self._transfer_id
        if manifest is None or transfer_id is None:
            raise RuntimeError("no policy receive transaction is active")
        expected_index = len(self._received)
        if bucket.descriptor.index != expected_index:
            raise ValueError(
                f"policy bucket order mismatch: expected={expected_index}, "
                f"received={bucket.descriptor.index}"
            )
        expected = manifest.bucket_descriptors[expected_index]
        if bucket.descriptor != expected:
            raise ValueError("policy bucket descriptor does not match manifest")
        if _sha256_bytes(bucket.payload) != expected.checksum:
            raise ValueError("policy bucket checksum mismatch")
        for tensor in expected.tensors:
            byte_slice = bucket.payload[tensor.start_byte : tensor.end_byte]
            value = byte_slice.view(_resolve_dtype(tensor.dtype)).reshape(tensor.shape)
            state_dict[tensor.name].copy_(value)
        self._received.add(expected_index)
        self._received_bytes += expected.byte_count
        return PolicyBucketApplyReceipt(
            transfer_id=transfer_id,
            policy_version=manifest.policy_version,
            bucket_index=expected_index,
            checksum=expected.checksum,
            byte_count=expected.byte_count,
        )

    def commit(self) -> PolicyVersionReceipt:
        """Publish the receiver version only after complete bucket coverage."""
        manifest = self._manifest
        transfer_id = self._transfer_id
        if manifest is None or transfer_id is None:
            raise RuntimeError("no policy receive transaction is active")
        if len(self._received) != manifest.bucket_count:
            raise RuntimeError(
                f"policy transfer incomplete: received={len(self._received)}, "
                f"expected={manifest.bucket_count}"
            )
        if self._received_bytes != manifest.total_bytes:
            raise RuntimeError("policy transfer byte count does not match manifest")
        prior_version = self._committed_version
        self._committed_version = manifest.policy_version
        receipt = PolicyVersionReceipt(
            transfer_id=transfer_id,
            prior_version=prior_version,
            applied_version=manifest.policy_version,
            manifest_hash=manifest.manifest_hash,
            bucket_count=manifest.bucket_count,
            total_bytes=manifest.total_bytes,
        )
        self._manifest = None
        self._transfer_id = None
        self._received = set()
        self._received_bytes = 0
        return receipt

    def abort(self, *, transfer_id: str) -> None:
        """Discard staging identity while retaining the old committed version."""
        if self._transfer_id != transfer_id:
            raise ValueError("policy transfer id does not match active transaction")
        self._manifest = None
        self._transfer_id = None
        self._received = set()
        self._received_bytes = 0
