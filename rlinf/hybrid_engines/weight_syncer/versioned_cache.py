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
import math
import threading
import uuid
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch.distributed.tensor import DTensor

from rlinf.utils.utils import materialize_tensor

_FORMAT_VERSION = 2


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
    contiguous = payload.contiguous()
    return hashlib.sha256(memoryview(contiguous.numpy())).hexdigest()


@dataclass(frozen=True, slots=True)
class PolicyTensorDescriptor:
    """Location of one tensor fragment in a bucket and the complete tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    start_byte: int
    end_byte: int
    tensor_start_byte: int
    tensor_end_byte: int

    @property
    def byte_count(self) -> int:
        return self.end_byte - self.start_byte

    @property
    def tensor_byte_count(self) -> int:
        return self.tensor_end_byte - self.tensor_start_byte


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
class CapturedPolicyBucket:
    """Unsealed CPU payload captured before checksum and manifest creation."""

    index: int
    tensors: tuple[PolicyTensorDescriptor, ...]
    payload: torch.Tensor


@dataclass(frozen=True, slots=True)
class PolicyCacheCapture:
    """Complete CPU byte capture that is not yet publishable or leasable."""

    policy_version: int
    format_version: int
    buckets: tuple[CapturedPolicyBucket, ...]
    total_bytes: int


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
                    "tensor_start_byte": tensor.tensor_start_byte,
                    "tensor_end_byte": tensor.tensor_end_byte,
                }
                for tensor in descriptor.tensors
            ],
        }
        for descriptor in bucket_descriptors
    ]


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def capture_policy_cache(
    state_dict: Mapping[str, torch.Tensor | DTensor],
    *,
    policy_version: int,
    bucket_size_bytes: int,
    selected_names: Iterable[str] | None = None,
) -> PolicyCacheCapture:
    """Capture a complete state dict into deterministic unsealed CPU buckets.

    Every caller participating in an FSDP/DTensor collective must invoke this
    function with the same names and ordering. Returned payloads are CPU-owned
    and byte-exact, but cannot be published until :func:`finalize_policy_capture`
    creates and checksums their immutable manifest.
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

    bucket_fragments: list[list[PolicyTensorDescriptor]] = []
    bucket_byte_counts: list[int] = []
    fragments_by_name: dict[str, list[tuple[int, PolicyTensorDescriptor]]] = {}
    for name in names:
        value = state_dict[name]
        if not isinstance(value, (torch.Tensor, DTensor)):
            raise TypeError(f"policy cache value {name!r} is not a tensor")
        shape = tuple(value.shape)
        dtype = _dtype_name(value.dtype)
        tensor_bytes = value.numel() * value.element_size()
        tensor_offset = 0
        if not bucket_fragments or bucket_byte_counts[-1] == bucket_size_bytes:
            bucket_fragments.append([])
            bucket_byte_counts.append(0)
        if tensor_bytes == 0:
            descriptor = PolicyTensorDescriptor(
                name=name,
                shape=shape,
                dtype=dtype,
                start_byte=bucket_byte_counts[-1],
                end_byte=bucket_byte_counts[-1],
                tensor_start_byte=0,
                tensor_end_byte=0,
            )
            bucket_fragments[-1].append(descriptor)
            fragments_by_name[name] = [(len(bucket_fragments) - 1, descriptor)]
            continue
        while tensor_offset < tensor_bytes:
            if bucket_byte_counts[-1] == bucket_size_bytes:
                bucket_fragments.append([])
                bucket_byte_counts.append(0)
            bucket_index = len(bucket_fragments) - 1
            bucket_start = bucket_byte_counts[bucket_index]
            fragment_bytes = min(
                bucket_size_bytes - bucket_start,
                tensor_bytes - tensor_offset,
            )
            bucket_end = bucket_start + fragment_bytes
            tensor_end = tensor_offset + fragment_bytes
            descriptor = PolicyTensorDescriptor(
                name=name,
                shape=shape,
                dtype=dtype,
                start_byte=bucket_start,
                end_byte=bucket_end,
                tensor_start_byte=tensor_offset,
                tensor_end_byte=tensor_end,
            )
            bucket_fragments[bucket_index].append(descriptor)
            bucket_byte_counts[bucket_index] = bucket_end
            fragments_by_name.setdefault(name, []).append((bucket_index, descriptor))
            tensor_offset = tensor_end

    payloads: list[torch.Tensor | None] = [None] * len(bucket_fragments)
    for name in names:
        contiguous = materialize_tensor(state_dict[name]).detach().contiguous()
        source_bytes = contiguous.reshape(-1).view(torch.uint8)
        for bucket_index, descriptor in fragments_by_name[name]:
            payload = payloads[bucket_index]
            if payload is None:
                payload = torch.empty(
                    bucket_byte_counts[bucket_index], dtype=torch.uint8
                )
                payloads[bucket_index] = payload
            payload[descriptor.start_byte : descriptor.end_byte].copy_(
                source_bytes[descriptor.tensor_start_byte : descriptor.tensor_end_byte],
                non_blocking=False,
            )
        del source_bytes
        del contiguous

    captured_buckets: list[CapturedPolicyBucket] = []
    for index, (fragments, byte_count, payload) in enumerate(
        zip(bucket_fragments, bucket_byte_counts, payloads, strict=True)
    ):
        if payload is None:
            payload = torch.empty(byte_count, dtype=torch.uint8)
        if payload.device.type != "cpu" or payload.dtype is not torch.uint8:
            raise RuntimeError("policy capture payload must be a CPU uint8 tensor")
        if payload.numel() != byte_count:
            raise RuntimeError("policy capture payload size is inconsistent")
        captured_buckets.append(
            CapturedPolicyBucket(
                index=index,
                tensors=tuple(fragments),
                payload=payload,
            )
        )
    return PolicyCacheCapture(
        policy_version=policy_version,
        format_version=_FORMAT_VERSION,
        buckets=tuple(captured_buckets),
        total_bytes=sum(bucket.payload.numel() for bucket in captured_buckets),
    )


def finalize_policy_capture(
    capture: PolicyCacheCapture,
) -> tuple[PolicyCacheManifest, tuple[PolicyBucket, ...]]:
    """Checksum and seal a complete CPU capture into a publishable candidate."""
    if not isinstance(capture, PolicyCacheCapture):
        raise TypeError("capture must be a PolicyCacheCapture")
    if capture.format_version != _FORMAT_VERSION:
        raise ValueError("policy capture format is unsupported")
    if not capture.buckets:
        raise ValueError("policy capture must contain at least one bucket")

    tensor_ranges: dict[str, list[tuple[int, int]]] = {}
    tensor_byte_counts: dict[str, int] = {}
    tensor_schemas: dict[str, tuple[tuple[int, ...], str]] = {}
    observed_bytes = 0
    for index, captured in enumerate(capture.buckets):
        if captured.index != index:
            raise ValueError("policy capture bucket indices must be sequential")
        payload = captured.payload
        if payload.device.type != "cpu" or payload.dtype is not torch.uint8:
            raise TypeError("policy capture payload must be a CPU uint8 tensor")
        byte_count = payload.numel()
        bucket_cursor = 0
        for tensor in captured.tensors:
            if tensor.start_byte != bucket_cursor:
                raise ValueError("policy capture bucket contains a gap or overlap")
            if tensor.end_byte < tensor.start_byte or tensor.end_byte > byte_count:
                raise ValueError("policy capture tensor exceeds bucket bounds")
            if tensor.tensor_start_byte < 0:
                raise ValueError("policy capture tensor range is invalid")
            if tensor.byte_count != tensor.tensor_byte_count:
                raise ValueError("policy capture fragment byte counts do not match")
            schema = (tensor.shape, tensor.dtype)
            if tensor.name in tensor_schemas and tensor_schemas[tensor.name] != schema:
                raise ValueError("policy capture tensor schema is inconsistent")
            tensor_schemas[tensor.name] = schema
            expected_bytes = (
                math.prod(tensor.shape)
                * torch.empty((), dtype=_resolve_dtype(tensor.dtype)).element_size()
            )
            if tensor.tensor_end_byte > expected_bytes:
                raise ValueError("policy capture fragment exceeds tensor bounds")
            tensor_byte_counts[tensor.name] = expected_bytes
            tensor_ranges.setdefault(tensor.name, []).append(
                (tensor.tensor_start_byte, tensor.tensor_end_byte)
            )
            bucket_cursor = tensor.end_byte
        if bucket_cursor != byte_count:
            raise ValueError("policy capture bucket does not cover its payload")
        observed_bytes += byte_count
    for name, ranges in tensor_ranges.items():
        tensor_cursor = 0
        for start, end in sorted(ranges):
            if start != tensor_cursor:
                raise ValueError(
                    f"policy capture tensor {name!r} contains a gap or overlap"
                )
            tensor_cursor = end
        if tensor_cursor != tensor_byte_counts[name]:
            raise ValueError(f"policy capture tensor {name!r} is incomplete")
    if observed_bytes != capture.total_bytes:
        raise ValueError("policy capture total byte count is inconsistent")

    buckets: list[PolicyBucket] = []
    for index, captured in enumerate(capture.buckets):
        payload = captured.payload
        byte_count = payload.numel()
        descriptor = PolicyBucketDescriptor(
            index=index,
            byte_count=byte_count,
            tensors=captured.tensors,
            checksum=_sha256_bytes(payload),
        )
        buckets.append(PolicyBucket(descriptor=descriptor, payload=payload))

    descriptors = tuple(bucket.descriptor for bucket in buckets)
    schema_payload = _schema_payload(descriptors)
    model_schema_hash = _hash_json(schema_payload)
    manifest_payload = {
        "policy_version": capture.policy_version,
        "format_version": _FORMAT_VERSION,
        "model_schema_hash": model_schema_hash,
        "total_bytes": sum(item.byte_count for item in descriptors),
        "buckets": [
            {**schema, "checksum": descriptor.checksum}
            for schema, descriptor in zip(schema_payload, descriptors, strict=True)
        ],
    }
    manifest = PolicyCacheManifest(
        policy_version=capture.policy_version,
        format_version=_FORMAT_VERSION,
        model_schema_hash=model_schema_hash,
        bucket_descriptors=descriptors,
        total_bytes=int(manifest_payload["total_bytes"]),
        manifest_hash=_hash_json(manifest_payload),
    )
    return manifest, tuple(buckets)


def build_policy_cache(
    state_dict: Mapping[str, torch.Tensor | DTensor],
    *,
    policy_version: int,
    bucket_size_bytes: int,
    selected_names: Iterable[str] | None = None,
) -> tuple[PolicyCacheManifest, tuple[PolicyBucket, ...]]:
    """Capture and synchronously finalize a policy-cache compatibility path."""
    capture = capture_policy_cache(
        state_dict,
        policy_version=policy_version,
        bucket_size_bytes=bucket_size_bytes,
        selected_names=selected_names,
    )
    return finalize_policy_capture(capture)


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
        if manifest.format_version != _FORMAT_VERSION:
            raise ValueError("candidate policy-cache format is unsupported")
        if not isinstance(buckets, tuple):
            raise TypeError("buckets must be a tuple")
        if len(buckets) != manifest.bucket_count:
            raise ValueError("candidate bucket count does not match manifest")
        if manifest.total_bytes != sum(
            descriptor.byte_count for descriptor in manifest.bucket_descriptors
        ):
            raise ValueError("candidate manifest total byte count is inconsistent")
        for index, (bucket, expected) in enumerate(
            zip(buckets, manifest.bucket_descriptors, strict=True)
        ):
            if bucket.descriptor != expected or expected.index != index:
                raise ValueError("candidate bucket descriptor does not match manifest")
            if bucket.payload.numel() != expected.byte_count:
                raise ValueError(
                    "candidate bucket payload size does not match manifest"
                )
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
        self._noncontiguous_staging: dict[str, torch.Tensor] = {}
        self._noncontiguous_targets: dict[str, torch.Tensor] = {}

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
        if manifest.format_version != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported policy-cache format {manifest.format_version}; "
                f"expected={_FORMAT_VERSION}"
            )
        if manifest.total_bytes != sum(
            descriptor.byte_count for descriptor in manifest.bucket_descriptors
        ):
            raise ValueError("policy manifest total byte count is inconsistent")
        tensor_ranges: dict[str, list[tuple[int, int]]] = {}
        tensor_byte_counts: dict[str, int] = {}
        tensor_schemas: dict[str, tuple[tuple[int, ...], str]] = {}
        noncontiguous_staging: dict[str, torch.Tensor] = {}
        noncontiguous_targets: dict[str, torch.Tensor] = {}
        for expected_index, descriptor in enumerate(manifest.bucket_descriptors):
            if descriptor.index != expected_index:
                raise ValueError("policy bucket indices must be sequential")
            if descriptor.byte_count < 0:
                raise ValueError("policy bucket byte count must be non-negative")
            bucket_cursor = 0
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
                if not target.is_contiguous():
                    noncontiguous_targets[tensor.name] = target
                    if tensor.name not in noncontiguous_staging:
                        noncontiguous_staging[tensor.name] = torch.empty(
                            tuple(target.shape),
                            dtype=target.dtype,
                            device="cpu",
                        )
                if tensor.start_byte != bucket_cursor:
                    raise ValueError("policy bucket fragments contain a gap or overlap")
                if tensor.end_byte < tensor.start_byte:
                    raise ValueError("policy bucket fragment range is invalid")
                if tensor.end_byte > descriptor.byte_count:
                    raise ValueError("policy bucket fragment exceeds bucket bounds")
                if tensor.tensor_start_byte < 0:
                    raise ValueError("policy tensor fragment range is invalid")
                target_bytes = target.numel() * target.element_size()
                if tensor.tensor_end_byte > target_bytes:
                    raise ValueError("policy tensor fragment exceeds tensor bounds")
                if tensor.byte_count != tensor.tensor_byte_count:
                    raise ValueError("policy tensor fragment byte counts do not match")
                schema = (tensor.shape, tensor.dtype)
                if (
                    tensor.name in tensor_schemas
                    and tensor_schemas[tensor.name] != schema
                ):
                    raise ValueError("policy tensor fragment schema is inconsistent")
                tensor_schemas[tensor.name] = schema
                tensor_byte_counts[tensor.name] = target_bytes
                tensor_ranges.setdefault(tensor.name, []).append(
                    (tensor.tensor_start_byte, tensor.tensor_end_byte)
                )
                bucket_cursor = tensor.end_byte
            if bucket_cursor != descriptor.byte_count:
                raise ValueError("policy bucket descriptors do not cover its payload")
        for name, ranges in tensor_ranges.items():
            tensor_cursor = 0
            for start, end in sorted(ranges):
                if start != tensor_cursor:
                    raise ValueError(
                        f"policy tensor {name!r} fragments contain a gap or overlap"
                    )
                tensor_cursor = end
            if tensor_cursor != tensor_byte_counts[name]:
                raise ValueError(
                    f"policy tensor {name!r} fragments do not cover the tensor"
                )
        self._manifest = manifest
        self._transfer_id = transfer_id
        self._received = set()
        self._received_bytes = 0
        self._noncontiguous_staging = noncontiguous_staging
        self._noncontiguous_targets = noncontiguous_targets

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
        if bucket.payload.numel() != expected.byte_count:
            raise ValueError("policy bucket payload size does not match manifest")
        if _sha256_bytes(bucket.payload) != expected.checksum:
            raise ValueError("policy bucket checksum mismatch")
        for tensor in expected.tensors:
            byte_slice = bucket.payload[tensor.start_byte : tensor.end_byte]
            destination = self._noncontiguous_staging.get(
                tensor.name, state_dict[tensor.name]
            )
            target_bytes = destination.detach().reshape(-1).view(torch.uint8)
            target_bytes[tensor.tensor_start_byte : tensor.tensor_end_byte].copy_(
                byte_slice
            )
        self._received.add(expected_index)
        self._received_bytes += expected.byte_count
        return PolicyBucketApplyReceipt(
            transfer_id=transfer_id,
            policy_version=manifest.policy_version,
            bucket_index=expected_index,
            checksum=expected.checksum,
            byte_count=expected.byte_count,
        )

    @torch.no_grad()
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
        for name, staging in self._noncontiguous_staging.items():
            self._noncontiguous_targets[name].copy_(staging)
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
        self._noncontiguous_staging = {}
        self._noncontiguous_targets = {}
        return receipt

    def abort(self, *, transfer_id: str) -> None:
        """Discard staging identity while retaining the old committed version."""
        if self._transfer_id != transfer_id:
            raise ValueError("policy transfer id does not match active transaction")
        self._manifest = None
        self._transfer_id = None
        self._received = set()
        self._received_bytes = 0
        self._noncontiguous_staging = {}
        self._noncontiguous_targets = {}
