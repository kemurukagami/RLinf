# Task 2 Detailed Implementation Plan: Local Safe-Point Lifecycle

## 1. Status and source of truth

Status: MVP implementation complete; the paired local
drain/offload/onload/resume lifecycle and its functional timing matrix pass.

This document expands Task T2, "Local safe-point lifecycle," from:

- `../VLA_COMPATIBILITY_DESIGN.md`, especially safe-point interruption,
  recovery state, and resize lifecycle;
- `RLIX_ELASTIC_VLA_ROLLOUT_IMPLEMENTATION_PLAN.md`, especially the shared
  T1/T2/T5 safe-point invariant and the T2 edit list;
- `../VLA_COMPATIBILITY_IMPLEMENTATION_PLAN.md`, which defines the canonical
  T0-T8 task sequence; and
- `TASK_1_ROLLOUT_SNAPSHOT_RESUME_IMPLEMENTATION_PLAN.md`, which defines the
  implemented continuation-state boundary on which T2 builds.

If this document conflicts with `../VLA_COMPATIBILITY_DESIGN.md`, the
architecture document wins. If implementation work discovers that a safe
drain requires a different channel or lifecycle contract, update the
architecture document and the task-organized plan in the same change.

T1 is complete for the supported same-actor synchronous path. In particular,
`EnvWorker` already owns a persistent `EnvRolloutCursor`, can stop in
`BOOTSTRAP_PENDING`, can capture a CPU-only `EnvRolloutResumeState`, and can
restore that state without sending the pending bootstrap. T2 must use those
primitives rather than introduce a second continuation format.

Implementation progress:

- [x] Add transition identity, strict merge behavior, and the routed
  observation/drain-barrier payload data model.
- [x] Integrate identified observation/result/barrier traffic into both worker
  elastic run paths and await routed delivery end to end.
- [x] Add a shared rank lifecycle, operation tokens, status, and result types.
- [x] Add an elastic-only `EnvWorker` run path that drains after a committed
  chunk and retains the next bootstrap.
- [x] Add the persistent `MultiStepRolloutWorker` cursor and elastic-only run
  path that closes its outstanding receive at a validated drain barrier.
- [x] Add rollout activation/status/drain, identified inference, awaited result
  delivery, final-bootstrap progress, safe-point token, and fail-closed guards.
- [x] Add complete rollout policy/expert/feature-model offload/onload, residency
  verification, and resume preparation.
- Deferred from the T2 MVP: further shared legacy/elastic loop cleanup beyond
  the common train-result construction helper. Current legacy compatibility is
  regression-tested and this refactor is not required for correct pause/resume.
- [x] Add environment offload/onload operations with residency verification and
  fail-closed state changes.
- [x] Add the environment half of same-rank resume with policy-version and
  transition validation and prove one paired retained-transition dispatch.
- [x] Add isolated identity, envelope routing, lifecycle, receipt, cursor, and
  legacy split/merge regression tests.
- [x] Add the paired fake-channel pause/offload/onload/resume transaction with
  matching tokens and empty queues.
- [x] Add core snapshot, environment/rollout offload, rollout onload,
  environment restore, and partial-pair failure injection tests.
- [x] Add selected-rank isolation: rank 0 pauses and resumes while rank 1
  completes without a barrier or lifecycle change.
- [x] Add repeated completed-worker lifecycle-generation tests for both peers.
- [x] Add deterministic uninterrupted-versus-resumed transition and partial
  trajectory ordering equivalence.
- [x] Add the late-drain timing race where an already-sent observation must
  finish before pausing at the following transition.
- [x] Add the functional drain timing matrix, including requests queued during
  prediction/chunk execution, both channel waits, a late send, and final
  bootstrap completion.
- [x] Add richer Wan uninterrupted-versus-resumed world-state, last-observation,
  reward/return, reset, episode, and metric equivalence assertions.
- Deferred from the T2 MVP: specialized CUDA-graph and cache-residency failure
  injection. Core snapshot/offload/onload/restore failures remain covered.
- [x] Run the complete local T2 verification set after worker integration
  (`98 passed`), plus Ruff, formatting, Python compilation, and diff checks.

Foundation status on 2026-07-15: the embodied data layer now defines the
structured transition identity, collision-free elastic request envelope,
logical batch-size metadata, and routed split/merge/inference helpers. The
shared lifecycle module validates the resolved failure/reuse transition table,
requests, safe-point tokens, outcomes, and residency receipts. The rollout
worker implements its elastic entry points through `PAUSE_READY`, verified
offload, and resume preparation. EnvWorker now emits the matching barrier only
after a committed chunk, owns the CPU snapshot, verifies environment residency,
and resumes the retained bootstrap. The local paired transaction passes; the
local hardening also covers core fail-closed errors, partial paired failure,
repeated lifecycles, the complete functional drain-timing matrix,
selective-rank isolation, deterministic transition/trajectory equivalence, and
richer Wan world-state/metric equivalence. Specialized CUDA-graph/cache
failure injection is deferred as non-MVP hardening. Real two-pipeline GPU reuse
remains T8 acceptance rather than T2 scheduler integration.

## 2. Required outcome

After T2, a caller inside RLinf can operate one selected synchronous
rollout/environment DP shard as a local transaction without involving RLix:

```text
activate rank at lifecycle L and policy version V
run environment and rollout peers
request drain D while either peer may be busy
finish the already-started policy/chunk exchange
retain EnvOutput n+1 without dispatching policy request n+1
close rollout channel traffic at a drain barrier
snapshot EnvWorker continuation state to CPU
offload the selected environment and rollout models
report PAUSED only after both local workers are non-resident

later:
validate D, L, V, and transition n+1
onload the same rollout and environment actors
restore the T1 continuation state
restart both peer loops
send retained EnvOutput n+1 once
continue the partial trajectory on the same ranks
```

The defining transaction property is:

```text
committed transitions before pause
+ committed transitions after resume
= one uninterrupted ordered transition sequence
```

The following must be observable in CPU tests:

- transition identities are strictly ordered and unique;
- a drain requested during policy inference or world-model diffusion is not
  acknowledged until that exchange and chunk commit finish;
- the next observation is present in the T1 snapshot but absent from the
  rollout worker's inference log while paused;
- a successful pause receipt is impossible until environment and rollout GPU
  residency have both been released;
- resumption uses the same lifecycle generation, policy version, worker ranks,
  stage, and next transition;
- a sibling shard not selected for drain continues to completion; and
- malformed, stale, duplicated, or partially failed operations stop the shard
  in a fail-closed state.

T2 proves local correctness only. It does not transfer logical GPU ownership
to RLix.

## 3. Scope boundaries

### 3.1 Supported by T2

- Synchronous `EnvWorker` and Hugging Face `MultiStepRolloutWorker`.
- `EmbodiedRolloutResult` and the T1 world-model snapshot contract.
- Complete-model, data-parallel replicas.
- One rollout worker rank paired with the same-numbered environment rank.
- `rollout.pipeline_stage_num == 1`.
- Wan and OpenSora world-model environments supported by T1.
- Same-rank pause and resume in the same existing Ray actors.
- Drain at the end of a committed `chunk_step()` only.
- Selected-rank environment, VLA, auxiliary rollout-model, and CUDA-graph
  offload/onload.
- Local fake-channel and fake-model tests without scheduler decisions.
- Legacy uninterrupted `interact()` and `generate()` behavior through wrappers
  around the refactored core loops.

### 3.2 Explicitly not implemented by T2

- RLix registration, allocation, planning, release, or callback lookup.
- The `RLixResizeCoordinator`; T5 consumes the APIs defined here.
- Rank-specific progress reporting or voluntary scheduler release; those are
  T4 responsibilities.
- Placement conversion, configuration validation, or production worker launch
  changes; those are T6 responsibilities.
- Runner stage orchestration, complete-batch sealing, and training barriers;
  those are T7 responsibilities.
- Cross-rank migration, request rebalancing, or sticky-map clearing.
- Tensor parallelism, model pipeline parallelism, application rollout
  pipelining, async embodied collection, or decoupled channels.
- Forced cancellation inside policy inference or diffusion.
- Process restart recovery or durable snapshot persistence.
- Selective rollout weight synchronization during expansion.
- GPU utilization acceptance or another pipeline reusing the released bundle;
  those are T8 responsibilities.

### 3.3 T1 feature gates remain authoritative

The elastic entry points must call the existing T1 capability validation before
activation or drain. They must continue to reject:

- multiple rollout pipeline stages;
- decoupled mode;
- online LeRobot state;
- RLT pending transitions;
- history-buffer reward without a snapshot contract;
- training pipelining;
- enabled data-collection wrappers with unsupported mutable state;
- non-fixed world-model reset-state selection; and
- environments without the T1 snapshot/prepare/commit contract.

T2 adds gates for the paired lifecycle itself:

- environment and rollout world sizes must be equal for local pairing;
- the selected environment rank must equal the selected rollout rank;
- rollout `enable_offload` and train-environment `enable_offload` must be true;
- the applied rollout policy version must already be established;
- a new lifecycle cannot start while a previous lifecycle is active, draining,
  snapshotting, expanding, or failed resident; and
- evaluation, weight synchronization, or another generation call cannot enter
  a worker while its elastic shard is active or paused.

Production validation of these conditions belongs to T6, but T2 worker APIs
must still fail locally if called incorrectly.

## 4. Existing implementation and concrete gaps

### 4.1 `EnvWorker`

File: `rlinf/workers/env/env_worker.py`

T1 added:

- `EnvRolloutCursor` and `RolloutCursorPhase`;
- persistent current outputs, pending bootstraps, metrics, and partial rollout;
- `_set_pending_bootstrap()` and `_send_pending_bootstrap()`;
- `snapshot_rollout_stage()`, validation, and two-phase restore; and
- resume-aware iteration from `BOOTSTRAP_PENDING`.

The remaining T2 gaps are concrete:

- `_run_interact_once()` calls `_send_pending_bootstrap()` immediately after
  `_set_pending_bootstrap()`;
- no drain request can be recorded or correlated with one lifecycle;
- `interact()` returns only at complete collection;
- the receive path has no channel-visible transition identity;
- `_rollout_call_active` distinguishes only running/not-running, not the
  canonical rank lifecycle;
- environment offload occurs only after the complete `interact()` method;
- no public selected-rank pause or resume receipt exists; and
- errors do not distinguish a safely CPU-resident failure from uncertain GPU
  residency.

### 4.2 `MultiStepRolloutWorker`

File: `rlinf/workers/rollout/hf/huggingface_worker.py`

The current `generate()` and `generate_one_epoch()` methods keep epoch, chunk,
and stage position in coroutine locals. After sending result `n`, the worker
immediately starts a receive for observation `n+1`. It has no way to distinguish
that next observation from a stale or duplicate message.

The current model residency helpers also need stronger semantics:

- `offload_model()` and `reload_model()` have no lifecycle preconditions or
  idempotence state;
- CUDA graph release/capture is not represented in worker status;
- `expert_model` is not offloaded by the current helper;
- auxiliary model and buffer residency is not verified recursively;
- `self.version` is not tied to a paused transition token; and
- a partial failure can leave model residency unknown without changing worker
  state to `FAILED_RESIDENT`.

### 4.3 Embodied channel payloads

File: `rlinf/data/embodied_io_struct.py`

`EnvOutput` and `RolloutResult` currently carry no transition identity.
`EnvOutput.to_dict()`, `RolloutResult.merge_rollout_results()`,
`MultiStepRolloutWorker._merge_obs_batches()`, and
`MultiStepRolloutWorker._split_rollout_result()` therefore cannot preserve or
validate identity.

The channel itself must not be snapshotted. This creates one additional race
that the high-level pseudocode leaves implicit:

```text
rollout sends result n
rollout waits on the channel for observation n+1
environment commits chunk n
environment observes drain and retains observation n+1
```

Without a control payload, the rollout call remains blocked on an outstanding
receive forever. Abandoning or cancelling that receive is unsafe because it
could later consume the resumed observation. T2 therefore needs a typed drain
barrier on the existing routed channel. The barrier is not an `EnvOutput` and
does not dispatch policy inference.

### 4.4 Worker actor concurrency

The supported worker methods are async Ray actor methods, but synchronous model
inference and `chunk_step()` intentionally occupy the actor until their current
operation completes. Drain requests may run only at explicit async yield
points. T2 must add those yield points at transaction boundaries and use async
channel waits in the elastic path.

Changing production `WorkerGroup.launch(..., max_concurrency=...)` belongs to
T6/T7. T2 APIs and tests must document that the production elastic launch needs
at least two interleavable actor tasks: the long-running collection method and
one lifecycle-control method. T2 must not claim remote drain support from a
worker launched with effective concurrency one.

## 5. Core invariants and design decisions

### 5.1 One transition identity across both peers

Use a structured identity rather than a bare integer:

```python
@dataclass(frozen=True, slots=True, order=True)
class RolloutTransitionIdentity:
    lifecycle_generation: int
    env_worker_rank: int
    stage_id: int
    sequence: int
```

`sequence` is the T1 `next_transition_ids[stage_id]`. The other fields prevent
transition `0` from a previous collection or another source rank from being
accepted after a queue delay. For the supported topology:

```text
identity.lifecycle_generation == cursor.lifecycle_generation
identity.env_worker_rank == EnvWorker rank == rollout worker rank
identity.stage_id == 0
identity.sequence == cursor.next_transition_ids[0]
```

The structured value is stored in the optional `transition_id` field of both
`EnvOutput` and `RolloutResult`. Optionality preserves legacy payloads. An
elastic method requires a non-`None` value; a legacy method continues to use
`None`. Mixing identified and unidentified shards during merge fails.

### 5.2 Exactly-once means validate before work or mutation

On the rollout peer:

1. Receive an identified observation.
2. Validate lifecycle, source rank, stage, and exact expected sequence.
3. Only then invoke policy inference.
4. Copy the same identity to `RolloutResult`.
5. Await successful result delivery.
6. Record that identity as completed and advance the expected sequence.

On the environment peer:

1. Receive an identified `RolloutResult`.
2. Validate it against the observation request currently in flight.
3. Only then append policy data or mutate world-model state.
4. Commit `chunk_step()` and all reward/reset/metric effects.
5. Build the next identified `EnvOutput`.

A stale, skipped, duplicated, wrong-rank, wrong-stage, or wrong-lifecycle value
raises before inference or trajectory mutation and moves the elastic shard to
`FAILED_RESIDENT`. T2 does not silently drop, replay, or deduplicate messages.

### 5.3 The safe point remains after commit and before dispatch

The only pausable worker cursor phase is T1 `BOOTSTRAP_PENDING`:

```text
result n validated and appended
world-model chunk n committed
reward/reset/metric/history effects committed
EnvOutput n+1 created and retained
drain barrier sent and acknowledged by local call completion
no inference for n+1 started
```

No safe point is introduced:

- between receiving result `n` and appending it;
- between trajectory append and `chunk_step()`;
- during diffusion;
- between world-model commit and reward/reset bookkeeping; or
- after sending observation `n+1` but before receiving its result.

If observation `n+1` was already sent before the drain request was observed,
that exchange is the current in-flight transaction. Both peers finish it and
pause at `n+2`.

### 5.4 A drain barrier closes an outstanding rollout receive

Add an elastic-only channel envelope:

```python
class ElasticRolloutRequestKind(str, Enum):
    OBSERVATION = "observation"
    DRAIN_BARRIER = "drain_barrier"


@dataclass(kw_only=True, slots=True)
class ElasticRolloutRequest:
    kind: ElasticRolloutRequestKind
    transition_id: RolloutTransitionIdentity
    logical_batch_size: int
    env_output: dict[str, Any] | None
    drain_request_id: str | None = None
```

For `OBSERVATION`, `env_output` is the existing CPU observation dictionary and
`drain_request_id` is `None`. For `DRAIN_BARRIER`, `env_output` is `None` and
the transition identity names the retained, unsent observation. The barrier
also carries the caller-provided drain request ID. `logical_batch_size` is the
batch size of that retained observation for both request kinds; it lets
`Worker.recv_from()` retain its normal routed batch-size validation even though
a barrier has no observation body. The collision-free elastic name avoids
confusion with `rlinf.data.io_struct.RolloutRequest`.

The barrier has these rules:

- it uses the same route key and train channel as the next observation;
- it is routed through `Worker.send_to()`/`recv_from()`, not direct `Channel`
  queue addressing;
- it is sent only after the pending bootstrap is stored;
- its synchronous/awaited put must complete before the environment run method
  reports a drained boundary;
- the rollout peer accepts it only if it has a matching local drain request;
- it never calls `_predict_rollout_actions()` and never produces a
  `RolloutResult`;
- it does not consume or clear the environment's pending bootstrap; and
- the rollout peer returns from its elastic run method with the same next
  transition identity.

Legacy `interact()` and `generate()` continue to exchange their current raw
dictionary payloads. The new envelope is confined to elastic-only entry points
until T7 adopts them.

### 5.5 Channel sends are part of the transaction

The current rollout result send uses `async_op=True` without waiting for the
returned work. In the elastic path, every observation, result, and drain
barrier send must be awaited before advancing the peer cursor. A method cannot
report a safe point while a result or control message is only queued in a
local send work object.

This is a T2 correctness rule, not a throughput optimization. Later profiling
may batch or overlap sends only if it preserves the same completion boundary.

### 5.6 Same actor, same rank, same policy version

T2 never serializes rollout model parameters. The paused actor retains them on
CPU. Expansion requires:

```text
current worker rank == pause token rank
current lifecycle generation == pause token lifecycle
self.version == pause token policy version
EnvRolloutResumeState.cursor.policy_version == self.version
rollout next transition == environment snapshot next transition
```

Weight synchronization is forbidden while a shard is active, draining,
snapshotting, paused, or expanding. T7 supplies the all-rank fixed policy-sync
stage before collection; T2 enforces only the local guard.

### 5.7 No hidden work while paused

While `PAUSED`, reject policy prediction, evaluation, generation, environment
interaction, bootstrap prefetch, weight sync, and global-step changes that can
alter model behavior. Status queries, idempotent pause inspection, failure
reporting, and the matching resume operation remain allowed.

This guard is also why T2 does not add rollout RNG state to the T1 snapshot:
the actor is preserved and no allowed paused operation may consume policy RNG.
If CUDA-graph capture or a supported model's reload path changes inference RNG,
that model must expose and restore its generator state or fail the elastic
capability check.

## 6. Shared lifecycle data model

### 6.1 New module

Define `RolloutTransitionIdentity`, `ElasticRolloutRequestKind`, and
`ElasticRolloutRequest` in `rlinf/data/embodied_io_struct.py`; channel data must
not import upward from worker modules. Add
`rlinf/workers/elastic_rollout_lifecycle.py` for lifecycle, token, status, and
receipt types used by both worker classes. It may import and re-export the
transition identity, which gives T5 one stable worker-facing protocol without a
data-to-worker dependency or circular import.

The module must not import RLix or Ray actor handles.

### 6.2 Rank lifecycle

```python
class ElasticRankState(str, Enum):
    INACTIVE_COLD = "inactive_cold"
    EXPANDING = "expanding"
    ACTIVE = "active"
    DRAIN_REQUESTED = "drain_requested"
    SNAPSHOTTING = "snapshotting"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED_RESIDENT = "failed_resident"
```

Allowed transitions are exactly:

```text
INACTIVE_COLD -> EXPANDING -> ACTIVE
ACTIVE -> DRAIN_REQUESTED -> SNAPSHOTTING -> PAUSED
PAUSED -> EXPANDING -> ACTIVE
ACTIVE -> COMPLETED
COMPLETED -> EXPANDING -> ACTIVE       # strictly newer lifecycle only
DRAIN_REQUESTED -> COMPLETED        # work finished before a pausable boundary
ACTIVE | DRAIN_REQUESTED | SNAPSHOTTING | EXPANDING -> FAILED_RESIDENT
```

Do not add a transition out of `FAILED_RESIDENT` in T2. Administrative recovery
must inspect actual residency first and is outside the automatic protocol.
`ACTIVE -> FAILED_RESIDENT` is required for transition, version, or barrier
protocol corruption detected after activation. `COMPLETED -> EXPANDING` is
available only through `prepare_elastic_collection()` with a strictly newer
lifecycle after prior cursor, request, token, snapshot, failure, and outstanding
channel-work state has been cleared and model residency verified.

### 6.3 Operation and safe-point tokens

```python
@dataclass(frozen=True, slots=True)
class DrainRequest:
    request_id: str
    worker_rank: int
    lifecycle_generation: int
    expected_policy_version: int


@dataclass(frozen=True, slots=True)
class SafePointToken:
    request_id: str
    worker_rank: int
    lifecycle_generation: int
    policy_version: int
    next_transition_id: RolloutTransitionIdentity
```

The caller generates a unique `request_id` and submits the identical request to
the environment and rollout peers. Repeating the same request is idempotent:
return the existing status or token. Reusing an ID with different fields or
submitting a second ID while a drain is pending fails.

### 6.4 Status and run results

```python
class ElasticRunOutcome(str, Enum):
    PAUSE_READY = "pause_ready"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ElasticRankStatus:
    state: ElasticRankState
    worker_rank: int
    lifecycle_generation: int | None
    policy_version: int | None
    expected_transition_id: RolloutTransitionIdentity | None
    drain_request_id: str | None
    snapshot_ready: bool
    model_resident: bool | None
    cuda_graph_captured: bool | None
    failure: str | None


@dataclass(frozen=True, slots=True)
class ElasticRunResult:
    outcome: ElasticRunOutcome
    token: SafePointToken | None
    metrics: dict[str, torch.Tensor] | None
```

`PAUSE_READY` means traffic is closed and the environment snapshot exists, but
does not itself mean GPUs are released. The worker state remains
`SNAPSHOTTING` until explicit offload succeeds. Only a status with
`state == PAUSED` and `model_resident is False` can contribute to a successful
paired pause receipt in T5.

### 6.5 Residency receipts

```python
@dataclass(frozen=True, slots=True)
class ResidencyReceipt:
    token: SafePointToken
    state: ElasticRankState
    model_resident: bool
    cuda_graph_captured: bool
```

Environment receipts may report `cuda_graph_captured=False` unconditionally.
A pause operation succeeds only with `PAUSED`, `model_resident=False`, and no
live CUDA graph. A resume preparation succeeds only in `EXPANDING`, with model
residency verified and the snapshot/token identity still matching.

## 7. Embodied payload changes

File: `rlinf/data/embodied_io_struct.py`

### 7.1 `EnvOutput`

Add:

```python
transition_id: RolloutTransitionIdentity | None = None
```

Required behavior:

- `__post_init__()` validates the field type but does not move or mutate it;
- `to_dict()` preserves it;
- `merge_env_outputs()` requires all non-legacy inputs to carry the same
  identity and rejects a mix of `None` and identified values; and
- T1 cloning automatically includes it through `dataclasses.fields()`.

Because an elastic T1 snapshot now includes a channel-visible identity, bump
`ENV_ROLLOUT_RESUME_SCHEMA_VERSION` from 1 to 2 in `env_worker.py`. The world
environment schema does not change. T2 does not silently load version-1 worker
snapshots into an elastic lifecycle.

### 7.2 `RolloutResult`

Add the same optional field:

```python
transition_id: RolloutTransitionIdentity | None = None
```

Update:

- `merge_rollout_results()` to preserve one identical identity;
- the result constructor in every elastic train inference path;
- `_split_rollout_result()` to copy the identity to every batch shard; and
- fake/result helpers in affected unit tests.

The identity is metadata for the whole routed local batch. It is not split
along batch dimension and must not be represented as a float tensor.

### 7.3 Merge rules

Use one shared helper with these cases:

| Inputs | Result |
| --- | --- |
| all `None` | `None` for legacy behavior |
| all equal non-`None` | that identity |
| mixture of `None` and non-`None` | fail |
| two different non-`None` identities | fail |

The supported elastic topology is one-to-one by rank, so different environment
worker identities reaching one rollout batch indicate an invalid routing or
unsupported topology, not a value that should be concatenated.

### 7.4 Drain request envelope

Define `ElasticRolloutRequestKind` and `ElasticRolloutRequest` in the same data
module because they are channel protocol, not worker lifecycle state. The
distinct name avoids colliding with the reasoning-stack `RolloutRequest`. Add
elastic split, merge, and batch-size inference helpers that:

- accepts only one request kind per merge;
- validates equal transition and drain request identities;
- preserves and validates a positive logical batch size;
- splits observation bodies while copying identity metadata;
- rejects barrier splitting in the supported one-to-one topology;
- merges observation dictionaries only for `OBSERVATION`; and
- returns a single `DRAIN_BARRIER` without fabricating an empty observation.

Keep this helper separate from legacy `_merge_obs_batches()`.

## 8. `EnvWorker` implementation

### 8.1 Persistent lifecycle fields

Initialize the following fields in `__init__()`:

```python
self._elastic_state = ElasticRankState.INACTIVE_COLD
self._elastic_expected_policy_version: int | None = None
self._elastic_drain_request: DrainRequest | None = None
self._elastic_safe_point_token: SafePointToken | None = None
self._elastic_resume_state: EnvRolloutResumeState | None = None
self._elastic_failure: str | None = None
self._elastic_mode = False
```

Do not use a thread lock around GPU work. Lifecycle mutation occurs on the
async actor event loop at explicit yield points. If a helper can also be called
from a synchronous unit test, keep state transition validation in a small
pure function.

### 8.2 Cold activation API

Add:

```python
def prepare_elastic_collection(
    self,
    *,
    lifecycle_generation: int,
    expected_policy_version: int,
) -> ElasticRankStatus: ...
```

This method:

1. runs T1/T2 feature gates;
2. requires `INACTIVE_COLD` or `COMPLETED`; for `COMPLETED`, requires a
   strictly newer lifecycle and clears the prior completed collection state
   before entering `EXPANDING`;
3. validates positive lifecycle generation and non-negative policy version;
4. initializes the T1 cursor with the supplied lifecycle generation rather
   than independently incrementing it;
5. records the expected policy version before the first result arrives;
6. clears previous drain/token/failure state;
7. onloads the training world-model environment if it is at the initialized
   offloaded baseline and verifies residency; and
8. moves `INACTIVE_COLD -> EXPANDING`.

The elastic interaction method moves `EXPANDING -> ACTIVE` only after the
environment is resident and its initial or restored bootstrap is ready.

T7 will own lifecycle-generation allocation. For T2 tests it is supplied by
the local harness.

### 8.3 Drain request API

Add an async control method:

```python
async def request_elastic_drain(
    self, request: DrainRequest
) -> ElasticRankStatus: ...
```

It must:

- record validated intent atomically when the control coroutine is first
  scheduled, so a request queued during synchronous GPU work cannot miss the
  next transaction-boundary yield;
- validate rank, active lifecycle, and expected policy version;
- transition `ACTIVE -> DRAIN_REQUESTED`;
- be idempotent for the exact same request;
- reject a competing request; and
- return after recording intent, not after waiting for the safe point.

It must not call `snapshot_rollout_stage()`, wait for the collection method, or
offload. A method running in the same actor must never block waiting for a
callback that needs that actor to progress.

### 8.4 Elastic channel send helper

Add:

```python
async def _send_elastic_rollout_request(
    self,
    rollout_channel: Channel,
    request: ElasticRolloutRequest,
) -> None: ...
```

For an observation it wraps the existing `EnvOutput.to_dict()` result. For a
barrier it sends no observation. It awaits the routed send work before
returning. The helper calls `self.send_to()` with the elastic split helper and
awaits the returned `AsyncRouteWork`; its rollout peer calls `self.recv_from()`
with the elastic merge and batch-size inference helpers. Neither peer builds or
addresses raw channel keys itself.

The initial bootstrap identity is:

```text
(lifecycle_generation, env rank, stage 0, sequence 0)
```

After `_set_pending_bootstrap()` advances the T1 cursor, assign the resulting
identity to both `_current_env_outputs[0]` and `_resume_bootstraps[0]` before
capture. The two references may be the same live object before snapshot, as in
T1, but the captured copies remain detached.

### 8.5 Elastic interaction entry point

Add a new method rather than changing the public return contract of
`interact()`:

```python
async def interact_until_pause_or_complete(
    self,
    input_channel: Channel,
    rollout_channel: Channel,
    reward_channel: Channel | None,
    actor_channel: Channel | None = None,
) -> ElasticRunResult: ...
```

Refactor `_run_interact_once()` so legacy and elastic wrappers share trajectory,
reward, environment, and completion logic. The elastic branch differs only at
bootstrap send/receive, transition validation, yield points, and the pending
bootstrap decision.

Do not copy the complete interaction loop into a second method.

### 8.6 Chunk-boundary drain logic

The elastic inner loop must implement:

```text
receive RolloutResult n asynchronously
validate identity n before mutation
record policy version and mark COMMITTING_CHUNK
append policy result n
run chunk_step n to completion
append transitions and commit metrics/reset/history
construct and store EnvOutput n+1
mark BOOTSTRAP_PENDING

await asyncio.sleep(0)  # allow queued drain control method to run

if state == DRAIN_REQUESTED:
    create SafePointToken for n+1
    send DRAIN_BARRIER(token) and await delivery
    state = SNAPSHOTTING
    snapshot = snapshot_rollout_stage()
    retain snapshot and return PAUSE_READY(token)
else:
    send OBSERVATION n+1 and await delivery
    clear pending bootstrap
    mark WAITING_FOR_POLICY
```

Set `SNAPSHOTTING` before calling the snapshot API so an exception cannot be
misreported as an active rank. On snapshot failure, record the exception and
move to `FAILED_RESIDENT`; the environment is still logically resident.

The barrier is sent before snapshot capture so the rollout peer can close its
receive while CPU cloning proceeds. Neither worker is yet releasable. If
snapshot capture then fails, T5 sees `FAILED_RESIDENT` and RLix must retain
ownership.

### 8.7 Final-bootstrap and completion handling

The final bootstrap policy call at the end of an epoch is an identified channel
exchange but not a world-model chunk. T2 does not pause between that result and
epoch finalization because it is not the documented safe point.

If a drain is first observed during final-bootstrap handling:

- finish that bootstrap result;
- if another epoch remains, continue through its reset/bootstrap and first
  committed chunk, then pause;
- if collection is complete, transition to `COMPLETED` and return completed
  metrics rather than manufacture a pause snapshot.

This prevents a completed shard from being resurrected as resumable work.

### 8.8 Snapshot ownership

Store exactly one `_elastic_resume_state` after successful capture. It remains
inside the same EnvWorker actor. Status and run results expose only the token,
not the potentially large snapshot.

Add a test-only or diagnostic getter only if needed:

```python
def get_elastic_resume_state(
    self, token: SafePointToken
) -> EnvRolloutResumeState: ...
```

It must validate the exact token and return a detached CPU clone. T5 should not
normally transfer this state through Ray.

### 8.9 Environment offload

Add:

```python
def offload_elastic_environment(
    self, token: SafePointToken
) -> ResidencyReceipt: ...
```

Preconditions:

- state is `SNAPSHOTTING`;
- the collection call returned `PAUSE_READY`;
- the stored snapshot and token agree;
- no policy request or environment chunk is active; and
- train-environment offload is enabled.

Operation:

1. call the concrete environment's idempotent `offload()`;
2. recursively verify all continuation tensors remain CPU-only;
3. use an environment residency capability or explicit known model/device
   inspection to prove no owned CUDA tensors remain;
4. synchronize the worker device before clearing the allocator cache; and
5. move to `PAUSED` only after verification succeeds.

If offload or verification fails, catch the original error, record a concise
failure string, move to `FAILED_RESIDENT`, and re-raise. Never return a success
receipt with uncertain residency.

T1 already makes Wan/OpenSora offload idempotent. T2 adds a common public
`is_offloaded`/residency verification hook to the base world environment;
private implementation flags and concrete-type branching are not sufficient
release evidence.

### 8.10 Environment resume preparation

Add:

```python
def prepare_elastic_resume(
    self, token: SafePointToken
) -> ResidencyReceipt: ...
```

Required order:

1. require `PAUSED` and exact token match;
2. revalidate the stored CPU snapshot before GPU mutation;
3. move to `EXPANDING`;
4. onload the same world-model environment;
5. call T1 `restore_rollout_stage()` with the token lifecycle and policy;
6. verify the restored cursor is `BOOTSTRAP_PENDING` at the token transition;
7. keep the pending bootstrap unsent; and
8. return an `EXPANDING` residency receipt.

The next call to `interact_until_pause_or_complete()` changes the state to
`ACTIVE` and consumes the T1 bootstrap once through the elastic envelope send.
The prepare method itself does not send.

Any failure after onload moves the shard to `FAILED_RESIDENT`, because GPU
ownership is then uncertain. A second resume call after success is rejected;
it cannot resend the bootstrap.

### 8.11 Legacy wrapper behavior

Keep `interact()` behavior and return type unchanged:

- no lifecycle envelope;
- no transition requirement;
- no drain observation;
- complete collection before return;
- existing final environment offload; and
- existing metrics.

The shared core should take an explicit mode enum or strategy object rather
than infer elastic behavior from the presence of a drain request.

## 9. `MultiStepRolloutWorker` implementation

### 9.1 Persistent rollout cursor

Add a small worker-local cursor:

```python
class RolloutPeerPhase(str, Enum):
    IDLE = "idle"
    WAITING_FOR_ENV = "waiting_for_env"
    PREDICTING = "predicting"
    SENDING_RESULT = "sending_result"
    PAUSE_READY = "pause_ready"
    FINALIZING = "finalizing"
    COMPLETED = "completed"


@dataclass(slots=True)
class RolloutPeerCursor:
    lifecycle_generation: int
    policy_version: int
    epoch_index: int
    committed_chunk_count: int
    expected_transition_id: RolloutTransitionIdentity
    phase: RolloutPeerPhase
```

The cursor remains in the same actor and is not serialized. It must be
sufficient to restart the loop at the exact remaining epoch/chunk count after
pause. In particular, resuming after the last chunk of an epoch skips action
generation and performs only that epoch's final-bootstrap call.

### 9.2 Lifecycle fields and activation

Initialize the same shared lifecycle fields as the environment worker plus:

```python
self._elastic_cursor: RolloutPeerCursor | None = None
self._model_resident = False  # init_worker() offloads in the supported config
self._cuda_graph_captured = False
```

Add:

```python
def prepare_elastic_collection(
    self,
    *,
    lifecycle_generation: int,
    expected_policy_version: int,
) -> ElasticRankStatus: ...
```

Validate `self.version == expected_policy_version`, initialize expected
transition sequence zero, load the rollout model if needed, and move
`INACTIVE_COLD -> EXPANDING`. The elastic generate method marks `ACTIVE` after
residency verification.

Use the same async `request_elastic_drain()` contract and idempotence rules as
the environment worker.

### 9.3 Elastic generation entry point

Add:

```python
async def generate_until_pause_or_complete(
    self,
    input_channel: Channel,
    output_channel: Channel,
) -> ElasticRunResult: ...
```

Refactor `generate_one_epoch()` into a cursor-driven core shared by the legacy
and elastic wrappers. Do not restart all epochs after resume.

Elastic receive logic:

```text
await one ElasticRolloutRequest

if OBSERVATION:
    validate exact expected transition before inference
    mark PREDICTING
    run policy prediction and optional bootstrap-value prediction
    construct RolloutResult with the same transition identity and self.version
    mark SENDING_RESULT
    send and await completion
    record completed transition and advance cursor
    mark WAITING_FOR_ENV
    await a control-method scheduling point

if DRAIN_BARRIER:
    require matching local DrainRequest
    require barrier transition == current expected transition
    require no prediction or result send in flight
    construct the same SafePointToken as EnvWorker
    mark PAUSE_READY/SNAPSHOTTING and return PAUSE_READY
```

The drain-request flag alone never abandons an already received observation.
The barrier is authoritative because it proves the environment retained the
named next observation. This closes the race in which a drain request is
processed after an observation was already sent.

### 9.4 Version enforcement

Before every elastic prediction, assert:

```text
self.version == cursor.policy_version == expected policy version
```

Copy `self.version` into normal action-result `versions` exactly as today. The
final-bootstrap result may retain its existing `versions=None` payload, but its
`transition_id` remains mandatory.

Reject `sync_model_from_actor()`, `set_global_step()`, or any model mutation
while the elastic lifecycle is not `INACTIVE_COLD` or `COMPLETED`. T7 performs
sync before activation.

### 9.5 Complete rollout-model offload

Replace the lifecycle use of the current `offload_model()` with a verified,
idempotent internal helper that covers every owned inference model:

- `hf_model`;
- `expert_model`, if configured and supported;
- `rlt_feature_model`, although the initial T2 feature gate rejects active RLT
  continuation state; and
- model-specific cached tensors or processors that declare device residency.

The helper order is:

```text
synchronize outstanding result send
release CUDA graph if captured
move all owned models and buffers to CPU
verify every parameter/buffer and declared cache is non-CUDA
device synchronize
empty allocator cache
record model_resident=False and cuda_graph_captured=False
```

Add:

```python
def offload_elastic_rollout(
    self, token: SafePointToken
) -> ResidencyReceipt: ...
```

It requires `SNAPSHOTTING`, exact token/cursor agreement, and no inference or
send in flight. Success moves to `PAUSED`; failure moves to
`FAILED_RESIDENT` and re-raises.

Do not weaken existing `offload_model()` behavior for standalone callers. It
may delegate to the new helper after compatibility tests establish equivalent
ordering.

### 9.6 Rollout onload and CUDA graph reconstruction

Add:

```python
def prepare_elastic_resume(
    self, token: SafePointToken
) -> ResidencyReceipt: ...
```

It requires `PAUSED`, validates token and `self.version`, moves to `EXPANDING`,
and then:

1. moves every supported owned model to the execution device;
2. verifies parameter/buffer residency;
3. recreates CUDA graphs exactly once if enabled;
4. verifies the graph state;
5. confirms the peer cursor still expects the token transition; and
6. returns without receiving an observation.

Onload or graph-capture failure moves to `FAILED_RESIDENT`. Repeated onload is
not treated as a successful second expansion.

### 9.7 Completion

When all epochs and the last bootstrap result finish:

- mark the peer cursor `COMPLETED`;
- move lifecycle `ACTIVE` or `DRAIN_REQUESTED -> COMPLETED`;
- return `ElasticRunResult(COMPLETED, token=None, metrics=None)`; and
- do not manufacture a resumable transition.

T4/T7 later decide when and how a completed rank is offloaded and released. T2
may reuse the verified offload helper in a completion test, but it does not
report scheduler progress.

### 9.8 Legacy wrapper behavior

Keep `generate()` and `generate_one_epoch()` externally compatible:

- raw legacy observation dictionaries;
- optional/absent transition identities;
- full static epoch loops;
- current progress display;
- reload at entry and offload at complete return; and
- current return value.

The old method names may wrap refactored helpers, but existing runner tests and
call sites must not be changed as part of T2.

## 10. Local paired transaction protocol

T2 does not add the T5 coordinator actor, but its tests need a small local
harness that uses the public worker APIs in the order T5 will later implement.

### 10.1 Activation

```text
assert env rank == rollout rank
env.prepare_elastic_collection(L, V)
rollout.prepare_elastic_collection(L, V)
start rollout.generate_until_pause_or_complete(...)
start env.interact_until_pause_or_complete(...)
```

Starting the rollout receive before the environment bootstrap avoids relying
on channel capacity. Both methods must tolerate the reverse scheduling order.

### 10.2 Drain

```text
D = DrainRequest(unique ID, rank, L, V)
submit D to environment and rollout peers
await both long-running call results

require both outcomes == PAUSE_READY
require tokens are identical
require env snapshot exists and is CPU-only

offload environment and rollout peers, preferably concurrently
require both receipts == PAUSED and model_resident=False
only now report local paired pause success
```

If either long-running call completes normally, the local result is completion,
not pause. If tokens differ, stop both sides fail closed.

### 10.3 Resume

```text
require matching PAUSED statuses and tokens
prepare rollout resume/onload
prepare environment resume/onload/restore
require matching EXPANDING receipts

start rollout.generate_until_pause_or_complete(...)
start env.interact_until_pause_or_complete(...)

environment sends retained token transition once
rollout accepts it once and continues
```

Preparation may run concurrently for disaggregated workers. In collocated mode,
T6/T7 must order onload according to memory capacity; T2 tests use fakes and do
not assume simultaneous residency on one GPU.

### 10.4 No actor-local callback wait

Neither worker may wait inside its long-running method for an external
coordinator to call its offload method. The long-running method returns
`PAUSE_READY`; the caller then invokes offload. This avoids deadlocking an actor
whose control call must be scheduled after collection returns.

With async actor concurrency, a status/control call may interleave earlier at
the documented yield points, but the protocol remains correct even if offload
is processed only after the long-running call returns.

## 11. Failure and retry semantics

### 11.1 Fail before mutation where possible

Validate request ID, rank, lifecycle, policy version, transition identity,
cursor phase, and snapshot identity before:

- policy inference;
- trajectory append;
- environment `chunk_step()`;
- model onload;
- snapshot restore commit; or
- lifecycle state advancement.

### 11.2 `FAILED_RESIDENT`

Enter `FAILED_RESIDENT` when any of these occurs in an elastic lifecycle:

- transition identity violation after activation;
- drain barrier mismatch;
- snapshot capture failure;
- environment or rollout offload failure;
- inability to prove CPU residency;
- onload/restore/CUDA-graph reconstruction failure; or
- peer tokens disagree in the local transaction harness.

`FAILED_RESIDENT` means logical GPU ownership must be retained. It does not
claim the model is on GPU; it claims residency is not safe enough to release.

### 11.3 Idempotent operations

The exact same `DrainRequest` may be submitted more than once. Status and token
queries are repeatable. Concrete Wan/OpenSora offload remains idempotent.

The paired lifecycle operations are intentionally stricter:

- a second offload call may return the existing receipt only if state is
  already `PAUSED` and the token is identical;
- a second resume preparation after state is `EXPANDING` or `ACTIVE` fails;
- a consumed pending bootstrap can never be recreated by retrying resume; and
- a different request ID cannot replace an in-progress drain.

### 11.4 Error preservation

Worker APIs should re-raise the original exception after recording lifecycle
failure. T5 and T7 must be able to preserve that exception rather than receive
only `False`. Store only a concise non-sensitive diagnostic string in status;
do not serialize arbitrary traceback objects into snapshots.

### 11.5 Paired partial failure

If one peer offloads successfully and the other fails, the successful peer may
remain verifiably `PAUSED` and CPU-resident while the failing peer is
`FAILED_RESIDENT`. The composite shard is not successfully paused for release:
the local harness reports failure and later T5 retains the whole logical
bundle. It must not eagerly onload the successful peer as an automatic
rollback; rollback can itself fail and obscure actual residency.

## 12. Concurrency and ordering audit

### 12.1 Required yield points

Elastic `EnvWorker` yields:

- while asynchronously waiting for a policy result;
- immediately after storing a committed next bootstrap and before deciding to
  send it; and
- after an awaited barrier/observation send as needed for control fairness.

Elastic `MultiStepRolloutWorker` yields:

- while asynchronously waiting for an observation or drain barrier;
- after an awaited result send; and
- before entering the next receive.

Do not yield inside trajectory mutation, reward/reset commit, or the
world-model `chunk_step()` transaction.

### 12.2 Request timing matrix

Tests must cover:

| Request timing | Required result |
| --- | --- |
| rollout waiting for observation | drain recorded; wait for environment barrier |
| VLA prediction executing | prediction/result finish; pause after environment commits its chunk |
| environment waiting for result | record drain; consume and commit that result |
| world-model diffusion executing | diffusion and bookkeeping finish; retain next output |
| immediately before next send | barrier wins; next observation is not sent |
| immediately after next send | that new exchange finishes; pause at the following boundary |
| final bootstrap of final epoch | complete shard; do not create paused work |

### 12.3 Production actor requirement

T6/T7 must explicitly launch supported elastic environment and rollout worker
groups with `max_concurrency=2` (or a documented larger value). The long-running
methods must remain async. T2 should expose a lightweight initialization/status
assertion or document the requirement in the method docstrings, but must not
guess actor launch configuration from Hydra.

### 12.4 Channel queue audit

After both peers return `PAUSE_READY`, fake channels must show:

- result `n` was consumed;
- one drain barrier for `n+1` was consumed by rollout;
- no observation `n+1` is queued;
- no result `n+1` is queued; and
- the environment snapshot contains the pending `EnvOutput n+1`.

After resume and one continued step, the log must show exactly one observation
and one result for `n+1`.

## 13. Tests

### 13.1 New test file

Create `tests/unit_tests/test_elastic_env_rollout.py`. Keep the primary suite
CPU-only. Reuse T1 fake world environments and snapshot helpers where practical
without importing a test module as production code. Shared fake construction
may move to `tests/unit_tests/conftest.py` only if it improves both suites.

Use an in-memory fake routed channel with:

- async `put`/`get` work objects matching the methods used by workers;
- per-route FIFO queues;
- a complete immutable event log;
- hooks/events before and after receive, predict, result send, chunk commit,
  pending-bootstrap store, barrier send, offload, and onload; and
- queue inspection for the paused-boundary assertions.

### 13.2 Payload identity tests

Cover:

- `EnvOutput.to_dict()` preserves identity;
- observation-envelope merge preserves one identity;
- `RolloutResult` split copies identity to every shard;
- result merge preserves equal identity;
- all-legacy `None` identities remain accepted;
- mixed legacy/identified shards fail;
- different lifecycle, rank, stage, or sequence values fail; and
- drain barriers cannot contain observations or omit request IDs.

### 13.3 Normal elastic run regression

Run one fake shard without requesting drain. Assert:

- its transition sequence is ordered through action and final-bootstrap calls;
- trajectory lengths and metrics match the legacy reference;
- both peers end `COMPLETED`;
- no drain barrier is emitted; and
- the legacy `interact()`/`generate()` fake-channel test remains byte- or
  tensor-equivalent to its pre-T2 result.

### 13.4 Pause/resume equivalence

Compare:

```text
uninterrupted: transition 0 -> 1 -> 2 -> completion
paused:        transition 0 -> drain barrier at 1
               snapshot/offload/onload/restore
               transition 1 -> 2 -> completion
```

Assert equality of:

- ordered observation and result identities;
- fake policy actions and versions;
- world-model chunk outputs and conditioning state;
- partial and final `EmbodiedRolloutResult`;
- rewards, dones, truncations, resets, metrics, and last observations;
- final environment snapshot/state; and
- exactly one inference invocation per transition.

### 13.5 Drain timing tests

Parameterize the request timing matrix from section 12.2. Use fake hooks to
hold and release operations. Assert the token always names the first
transaction not dispatched after the request becomes observable.

The diffusion test must demonstrate that a request raised while `chunk_step()`
is held does not cause snapshot, barrier, or offload until the fake chunk
commit hook completes.

### 13.6 Selected-rank isolation

Run two independent paired shards on the same event loop:

- request drain only for rank 0;
- verify rank 0 reaches `PAUSE_READY` and then `PAUSED`;
- allow rank 1 to continue through additional transitions and complete;
- verify rank 1 receives no lifecycle request, barrier, model move, or cursor
  discontinuity; and
- resume rank 0 and verify its final sequence independently.

This is the T2 exit test for "unaffected ranks continue." It does not emulate
RLix resource reuse.

### 13.7 Stale and duplicate tests

Inject before mutation:

- duplicate observation;
- skipped observation;
- stale lifecycle observation;
- wrong source-rank or stage observation;
- duplicate result;
- result for a future transition;
- result with the right sequence but wrong lifecycle;
- drain barrier with the wrong request ID;
- barrier naming a transition after the retained bootstrap; and
- resume token from another rank or policy version.

Assert no extra inference, trajectory append, or `chunk_step()` occurs and the
affected shard enters `FAILED_RESIDENT`.

### 13.8 Policy-version tests

Cover:

- activation rejects `self.version != expected_policy_version`;
- first environment result must match its prepared expected version;
- a later result cannot change version;
- pause token records the applied version;
- weight sync and global-step mutation are rejected while paused; and
- resume fails before onload/restore for a stale expected version.

### 13.9 Offload/onload and CUDA-graph tests

With fake models, record exact ordering:

```text
result send complete
release graph
move every model to CPU
verify residency
empty cache
PAUSED receipt

move every model to device
verify residency
capture graph once
EXPANDING receipt
```

Cover no-graph and graph-enabled cases, optional expert/feature models,
idempotent same-token offload inspection, and a second resume rejection.

### 13.10 Failure tests

Inject failures in:

- snapshot capture;
- environment offload;
- rollout graph release;
- rollout model CPU move;
- CPU residency verification;
- environment onload;
- T1 restore preparation/commit; and
- CUDA graph recapture.

Every case must avoid a success receipt, preserve the original exception, set
`FAILED_RESIDENT`, and leave enough status/token information for T5 to retain
logical ownership.

Add a paired partial-failure test in which one peer offloads and the other
fails. The composite local result must be failure.

### 13.11 T1 snapshot regressions

Update `tests/unit_tests/test_world_model_resume.py` only where the worker
schema version and optional transition field require it. Preserve all T1
equivalence, validation-before-mutation, CPU-only, and aliasing tests.

Add one assertion that an elastic `EnvRolloutResumeState` contains a pending
bootstrap whose structured transition identity agrees with its cursor and
world-state `next_transition_id`.

### 13.12 Existing focused regressions

At minimum run:

- `test_world_model_resume.py`;
- `test_maniskill_offload_env.py`;
- `test_overlap_env_bootstrap.py`;
- `test_history_manager.py`;
- embodied IO structure tests, if present;
- rollout-worker tests covering merge/split and model offload; and
- runner tests that exercise unchanged standalone `interact()`/`generate()`
  ordering.

Unsupported T1 modes must still fail at capability validation rather than
partially entering the elastic lifecycle.

## 14. File-by-file edit list

### Required

`rlinf/data/embodied_io_struct.py`

- structured transition identity;
- optional identity on `EnvOutput` and `RolloutResult`;
- strict identity merge helper;
- merge/split preservation; and
- collision-free elastic observation/drain-barrier envelope with logical batch
  size and routed split/merge/inference helpers.

`rlinf/workers/elastic_rollout_lifecycle.py`

- lifecycle enum and transition validation;
- drain request and safe-point token;
- status, run result, and residency receipt types; and
- no RLix or Ray ownership.

`rlinf/workers/env/env_worker.py`

- worker snapshot schema bump;
- persistent elastic lifecycle state;
- explicit lifecycle activation, drain, status, offload, and resume APIs;
- async elastic channel exchange;
- transition validation before mutation;
- drain decision at T1 `BOOTSTRAP_PENDING`;
- stored CPU snapshot ownership;
- environment residency verification; and
- legacy wrapper preservation.

`rlinf/workers/rollout/hf/huggingface_worker.py`

- persistent peer cursor and lifecycle state;
- cursor-driven remaining-work iteration;
- elastic request/barrier handling;
- transition/version checks before inference;
- awaited result delivery;
- complete selected-rank model and CUDA-graph residency lifecycle;
- status/offload/resume APIs; and
- unchanged legacy wrappers.

`tests/unit_tests/test_elastic_env_rollout.py`

- fake channel, peer, model, and local transaction harness;
- normal, pause/resume, timing, isolation, identity, version, residency, and
  failure tests.

`tests/unit_tests/test_world_model_resume.py`

- worker schema/identity updates and T1 regressions only.

`rlinf/envs/world_model/base_world_env.py`

- add a public, implementation-neutral residency inspection and verification
  hook; private `_is_offloaded` fields are not release evidence.

### Conditional

`rlinf/models/embodiment/base_policy.py`

- add model cache/residency or RNG-state hooks only if parameters and buffers
  are insufficient to prove safe offload/resume for a supported policy.

`tests/unit_tests/conftest.py`

- host shared T1/T2 fake world-environment construction if duplication would
  otherwise obscure the tests.

### Explicitly deferred

`rlinf/scheduler/rlix/*`

- T5 owns coordinator/controller/protocol integration.

`rlinf/config.py` and `examples/embodiment/train_embodied_agent.py`

- T6 owns user configuration, production gates, rank pairing, and worker actor
  concurrency at launch.

`rlinf/runners/embodied_runner.py`

- T7 owns starting, waiting, draining, resuming, completing, and sealing local
  shard calls at runner stages.

## 15. Suggested implementation sequence

1. Add the shared transition identity and strict merge helper with isolated
   data-structure tests.
2. Add the elastic channel envelope and drain-barrier validation tests.
3. Add the shared lifecycle/token/status module with a pure transition-table
   test.
4. Harden the envelope for routed use: collision-free naming, logical batch
   size, split/merge/inference helpers, and awaited `send_to()`/`recv_from()`
   work tests.
5. Resolve active failure and completed-worker reuse in the shared transition
   table and validation tests.
6. Add the rollout peer cursor without changing legacy `generate()` behavior.
7. Refactor rollout epoch iteration to resume from a cursor and preserve the
   final-bootstrap call.
8. Add elastic rollout observation/result identity checks and awaited sends.
9. Add EnvWorker elastic activation and identified bootstrap/result handling.
10. Add the post-chunk yield, pending-bootstrap drain decision, barrier send,
   and T1 snapshot capture.
11. Add fake-channel uninterrupted and pause-boundary transaction tests before
   implementing GPU residency changes.
12. Add complete rollout model/CUDA-graph offload and onload with failure tests.
13. Add the public world-environment residency hook, then environment
    offload/onload/restore and residency receipts with failure
   tests.
14. Add paired pause/resume equivalence, timing-matrix, and two-rank isolation
   tests.
15. Test two consecutive completed lifecycles on reused actors.
16. Update T1 schema/identity regressions and run the broader embodied suite.
17. Run Ruff, format checking, compilation, and final diff review.

Keep commits aligned with these boundaries. In particular, review channel
exactly-once behavior separately from model-residency changes; both are
required for a safe pause, but failures in them have different causes.

## 16. Verification commands

From `/root/_VLAMP/RLinf`, using the repository's shared Python environment and
workspace `PYTHONPATH`:

```bash
export PYTHONPATH="/root/_VLAMP/RLinf${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_elastic_env_rollout.py \
  tests/unit_tests/test_world_model_resume.py
```

```bash
export PYTHONPATH="/root/_VLAMP/RLinf${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_elastic_env_rollout.py \
  tests/unit_tests/test_world_model_resume.py \
  tests/unit_tests/test_maniskill_offload_env.py \
  tests/unit_tests/test_overlap_env_bootstrap.py \
  tests/unit_tests/test_history_manager.py
```

Discover and add the existing embodied IO, HF rollout-worker, and runner test
files touched by the refactor rather than assuming names that are not present
in the checkout:

```bash
rg -l "EnvOutput|RolloutResult|MultiStepRolloutWorker|\.generate\(|\.interact\(" \
  tests/unit_tests
```

```bash
/root/.venv/bin/ruff check \
  rlinf/data/embodied_io_struct.py \
  rlinf/workers/elastic_rollout_lifecycle.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_elastic_env_rollout.py \
  tests/unit_tests/test_world_model_resume.py
```

```bash
/root/.venv/bin/ruff format --check \
  rlinf/data/embodied_io_struct.py \
  rlinf/workers/elastic_rollout_lifecycle.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_elastic_env_rollout.py \
  tests/unit_tests/test_world_model_resume.py
```

```bash
/root/.venv/bin/python -m compileall -q \
  rlinf/data \
  rlinf/workers/env \
  rlinf/workers/rollout/hf \
  tests/unit_tests/test_elastic_env_rollout.py
```

T2 has no required real-checkpoint GPU command. The existing Task 1 Wan
accelerator harness remains useful as a regression after environment lifecycle
edits, but passing it does not prove paired worker drain. Full two-pipeline Wan
and OpenSora GPU recovery remains T8.

## 17. Definition of done

T2 is complete only when all of the following are true:

- `EnvOutput` and `RolloutResult` carry one structured, channel-visible
  transition identity in elastic mode.
- Identity survives observation conversion, channel envelope merge, result
  split, and result merge.
- Both peers reject stale, skipped, duplicate, wrong-rank, wrong-stage, and
  wrong-lifecycle messages before inference or trajectory mutation.
- `EnvWorker` observes drain only at the committed T1 pending-bootstrap
  boundary.
- A typed drain barrier closes the rollout peer's outstanding next receive
  without sending the retained observation.
- Every elastic observation, result, and barrier send completes before its
  cursor advances or its method reports a boundary.
- `MultiStepRolloutWorker` resumes from a persistent remaining-work cursor and
  does not replay earlier epoch/chunk inference.
- Both peers produce the same `SafePointToken` for a drain.
- The T1 snapshot contains the exact unsent next observation and partial
  trajectory on CPU.
- Environment, rollout policy, auxiliary rollout models, and CUDA graphs are
  offloaded and verified before either worker reports `PAUSED`.
- Offload or onload uncertainty produces `FAILED_RESIDENT`, never a successful
  pause or expansion receipt.
- Resume validates rank, lifecycle, policy version, and next transition before
  GPU mutation and sends the stored bootstrap once.
- A completed shard becomes `COMPLETED`, not resumable paused work.
- One selected rank can pause while an unaffected sibling continues.
- Uninterrupted and local pause/offload/onload/resume fake executions have
  equivalent ordered transitions, trajectories, environment state, and
  metrics.
- Existing standalone `interact()`/`generate()` ordering and T1 snapshot tests
  remain unchanged except for the intentional worker schema version bump.

T2 completion does not mean a GPU can yet be returned to the shared scheduler.
That claim additionally requires T3 composite bundle accounting, T4 release
semantics, T5 callback transaction ordering, T6 production placement and actor
concurrency, T7 runner barriers, and T8 two-pipeline GPU acceptance.
