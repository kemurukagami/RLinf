# Task 5 Detailed Implementation Plan: RLinf Resize Coordinator

## 1. Status and source of truth

Status: completed on 2026-07-20. The coordinator transaction, driver
controller, worker hardening, fake-peer and stubbed-backend in-memory worker
suites, core fail-closed integration, and opt-in local Ray naming/handle test
pass. Production T6/T7 construction and runner wiring remain intentionally
deferred.

Test-double audit reverified on 2026-07-21: the coordinator/worker-focused
suite passed `78 passed, 1 skipped`; the opt-in real-Ray named-actor test passed
separately; and Ruff lint/format plus diff checks passed. The skip is the
opt-in real-Ray case in the restricted run, not missing coordinator coverage.

This document expands Task T5, "RLinf resize coordinator," from:

- `../VLA_COMPATIBILITY_DESIGN.md`, especially architecture, safe-point
  interruption, resize lifecycle, stage lifecycle, parity, and acceptance;
- `../VLA_COMPATIBILITY_IMPLEMENTATION_PLAN.md`, which defines the canonical
  T0-T8 sequence and T5 exit condition;
- `RLIX_ELASTIC_VLA_ROLLOUT_IMPLEMENTATION_PLAN.md`, especially the shared
  T1/T2/T5 safe-point invariant and T5 edit list;
- `TASK_1_ROLLOUT_SNAPSHOT_RESUME_IMPLEMENTATION_PLAN.md`, which defines the
  implemented CPU continuation contract;
- `TASK_2_LOCAL_SAFE_POINT_LIFECYCLE_IMPLEMENTATION_PLAN.md`, which defines the
  implemented worker state machine and local paired transaction;
- `../TASK_3_COMPOSITE_BUNDLE_SCHEDULING_IMPLEMENTATION_PLAN.md`, which defines
  canonical rank ownership and callback-before-commit ordering;
- `../TASK_4_ELASTIC_PROGRESS_RELEASE_IMPLEMENTATION_PLAN.md`, which defines
  rank eligibility, completed-rank release, and exact release waiters; and
- `../VLA_WM_UTILIZATION_FUTURE_DESIGN.md`, whose service-pool and shared-model
  alternatives remain deferred and do not change the paired-rank T5 contract.

If this plan conflicts with `../VLA_COMPATIBILITY_DESIGN.md`, the architecture
document wins. If implementation reveals that a callback cannot preserve the
documented safe point or logical-ownership transaction, update the architecture
and task-organized plans before changing behavior.

Current T5 dependency and implementation status:

- [x] T0 fixed allocation and fail-closed scheduler transactions are complete.
- [x] T1 same-actor CPU continuation state is complete for Wan and OpenSora.
- [x] T2 local paired drain/offload/onload/resume is MVP-complete.
- [x] T3 exact composite bundles, fixed `policy_sync`, and callback ordering are
  complete.
- [x] T4 eligibility, completed-trajectory progress, and rank-specific release
  are complete.
- [x] A named RLinf coordinator actor and driver-side construction helper use
  the exact `rlix-core` actor prefix and caller-supplied registered namespace.
- [x] `resize_infer` drives direct same-ranked worker handles in CPU fake tests;
  production construction remains T6/T7 wiring rather than a T5 entrypoint edit.
- [x] Completed elastic workers have public verified token-free offload
  operations and a failure-only `COMPLETED -> FAILED_RESIDENT` transition.
- [x] Coordinator protocol read models and a narrow public paired-failure
  worker surface are implemented with focused CPU tests.
- [x] Policy synchronization and resize share an exact condition-based lease
  gate with bounded callback waits and no automatic lease expiry.
- [ ] T6 placement/configuration and T7 runner adoption remain pending.

## 2. Required outcome

After T5, RLinf can create one named callback actor per admitted pipeline. The
actor implements the callback shape already used by `rlix-core`:

```python
async def resize_infer(
    self,
    dp_ranks_to_remove: list[int],
    dp_ranks_to_add: list[int],
) -> ActionResponse:
    ...
```

For every selected DP rank, the callback coordinates the existing
same-numbered `EnvWorker` and `MultiStepRolloutWorker` actors. It returns
success only after the complete local pair reaches the state required for the
pending scheduler transaction:

```text
shrink active rank:
    register the same drain request on both peers
    await both already-running calls
    require identical PAUSE_READY tokens
    offload and verify both peers
    return only when the pair is CPU-resident and PAUSED

shrink completed rank:
    require both calls and both peers are COMPLETED
    offload and verify both peers without creating resumable state
    return only when the completed pair is CPU-resident

expand cold rank:
    prepare both peers for lifecycle L and policy version V
    start both elastic run calls
    return only after both peers report ACTIVE

expand paused rank:
    validate the coordinator's stored matching token
    onload and restore both peers
    start both elastic run calls
    return only after both peers report ACTIVE
```

The callback is the safety boundary between physical residency and logical
ownership:

```text
RLinf pair transition succeeds
    -> callback returns
    -> rlix-core may commit the validated plan

RLinf pair transition fails or times out
    -> callback raises
    -> rlix-core does not commit the plan
    -> scheduler activity fails closed
```

The coordinator also supplies an explicit policy-sync lease. A resize cannot
begin while an all-rank policy synchronization lease is held, and a policy
sync cannot begin while a resize is running. T7 will use this lease around the
actual collective; T5 implements and tests the serialization primitive.

## 3. Scope boundaries

### 3.1 Supported by T5

- One named Ray coordinator actor for one RLinf pipeline.
- The exact `rlix-core` coordinator actor name and registered namespace
  protocol.
- Direct actor-handle access to equal-ranked environment and rollout workers.
- One active collection context at a time: lifecycle generation, policy
  version, channels, and supported ranks.
- Cold activation, active drain, paused resume, and completed-rank release.
- One stored safe-point token per paused rank.
- Concurrent work across different ranks inside one callback where safe.
- Strict shrink-before-expand ordering if a direct caller supplies both lists.
- Serialization of callbacks and policy synchronization.
- Configurable coordinator-side operation deadlines that never forcibly cancel
  VLA inference or world-model diffusion.
- Read-only coordinator status and rank-result surfaces for later T7 use.
- CPU-only unit tests with fake async workers and a small local Ray naming test.
- Minimal T2 lifecycle hardening required for completed-rank offload and
  coordinator-detected paired failure.

### 3.2 Explicitly not implemented by T5

- Deriving GPU IDs or DP pairs from placement objects; T6 owns this.
- User-facing Hydra configuration and unsupported-mode validation; T6 owns
  this.
- Changing worker launch concurrency; T6 must launch supported env and rollout
  actors with `max_concurrency >= 2`.
- Registering all cluster mappings or choosing stage allocation policies; T6
  owns topology conversion and registration.
- Rewriting `EmbodiedRunner` to use elastic collection, fixed policy sync,
  fixed training, batch sealing, or final cleanup; T7 owns this.
- Actor training, advantage computation, trajectory consumption, checkpointing,
  or evaluation orchestration.
- Selective rollout weight synchronization during expansion.
- Cross-rank state migration, rank renumbering, or request rebalancing.
- Tensor parallelism, model parallelism, application rollout pipelining, async
  runners, or decoupled channels.
- Forced cancellation inside policy inference or diffusion.
- Automatic recovery from `FAILED_RESIDENT`.
- Multi-pipeline accelerator reuse or utilization claims; T8 owns acceptance.
- A claim that collocated production residency is accepted. T5 preserves the
  opaque width-one ownership contract, while T6/T7 must establish a
  memory-safe runtime ordering and T8 must test it on real checkpoints.

### 3.3 No additional scheduler semantics

T5 does not modify `rlix-core` planning or commit behavior. In particular:

- callbacks remain outside the scheduler lock;
- all pipeline shrinks finish before any expansion starts;
- canonical T3 rank bundles remain authoritative;
- T4 release waiters complete only after callback success and commit;
- callback exceptions prevent commit; and
- the core scheduler, not the coordinator, owns global GPU allocation state.

If T5 tests expose a core transaction bug, fix it as a separately justified
core regression rather than duplicating allocation logic in RLinf.

## 4. Current implementation and concrete gaps

### 4.1 RLinf integration package

`rlinf/scheduler/rlix/` now contains:

- `progress.py`, with `ElasticProgressTracker`;
- `protocol.py`, with collection, lease, and coordinator status types;
- `coordinator.py`, with direct paired worker transactions and the callback;
- `controller.py`, with named actor construction and driver wrappers; and
- lazy public exports in `__init__.py` so unrelated RLinf imports do not eagerly
  require `rlix-core`.

The production embodied entrypoint does not construct these surfaces yet; that
placement/configuration work remains T6 and runner adoption remains T7.

### 4.2 Existing core callback lookup

`rlix-core` already resolves:

```text
actor name = f"rlix-core:coordinator:{pipeline_id}"
namespace  = registered pipeline namespace
method     = resize_infer(remove, add)
```

The scheduler invokes shrink callbacks, waits for all of them, then invokes
expansion callbacks. It commits only after the calls return. T5 must conform to
this protocol; it must not introduce a second callback registry or ask core to
import RLinf.

### 4.3 Worker group result wrappers are unsuitable inside the callback

`WorkerGroup.execute_on(...).method(...)` creates a background thread whose
failure path sends `SIGUSR1` to its owner process. A callback needs to catch a
worker exception, preserve it, mark the pair failed, and re-raise it to
`rlix-core`. It must therefore call the selected Ray actor handles directly
and await their object references. T5 must not use `WorkerGroupFuncResult` for
callback-critical calls.

The driver-side controller may extract rank-to-actor mappings through the
public `worker_info_list` surface and pass those serializable handles into the
coordinator actor.

### 4.4 The coordinator must own elastic run references

T2 long-running calls return only at `PAUSE_READY` or `COMPLETED`:

- `EnvWorker.interact_until_pause_or_complete(...)`; and
- `MultiStepRolloutWorker.generate_until_pause_or_complete(...)`.

A shrink callback must await those exact calls to compare their outcomes and
tokens. Launching them in the runner and hiding their references from the
coordinator would make the callback depend on polling and would lose the
paired result boundary. T5 therefore launches and stores the two run object
references for each activated rank. T7 later waits for collection completion
through controller/coordinator APIs rather than owning independent worker-group
handles for the elastic calls.

### 4.5 Completed-rank release foundation

Both elastic worker loops transition to `COMPLETED` while their models can
still be resident. The original T2 offload methods accept only a matching
`SafePointToken` in `SNAPSHOTTING`; completed work has no pending bootstrap and
correctly has no pause token. T5 now supplies the separate token-free verified
completed offload operations described below; the coordinator still needs to
consume them.

T4 allows immediate rank-specific release of a completed active rank, so T5
requires public completed-residency operations on both workers. They must:

- require `COMPLETED` and no active run call;
- offload all owned model/environment state;
- verify non-residency and clear CUDA graphs/caches as applicable;
- preserve `COMPLETED` and its durable trajectory count on success;
- be idempotent only after verified completed non-residency; and
- enter `FAILED_RESIDENT` if offload or verification fails.

The transition table now allows `COMPLETED -> FAILED_RESIDENT` for this
failure-only path, and the architecture records the clarification.
`COMPLETED -> EXPANDING` for a strictly new lifecycle remains unchanged.

### 4.6 Coordinator-detected pair failure surface

Worker-owned exceptions already call `_record_elastic_failure()`, but token
mismatch, peer-outcome mismatch, and partial pair success are detected by the
coordinator. T5 now provides this narrow public method on both peers:

```python
def fail_elastic_lifecycle(self, *, reason: str) -> ElasticRankStatus:
    ...
```

It records a concise diagnostic and transitions to `FAILED_RESIDENT` when the
current state permits. It must not accept traceback objects or arbitrary
serialized exceptions. Calling it on an already failed worker is idempotent.

### 4.7 Explicit policy-sync serialization

The coordinator remains async while waiting for worker safe points, so Ray may
interleave its methods. Setting actor concurrency to one would prevent
status/control calls and would not provide a lease around a collective that
executes outside the actor. T5 therefore implements this explicit
condition-based gate:

```text
resize begin      waits until no policy-sync lease and no other resize
policy-sync begin waits until no resize and no existing sync lease
policy-sync end   validates the lease token and wakes resize waiters
```

A lease is never expired automatically. Losing the lease owner fails closed;
silently unlocking could overlap weight mutation with activation. Tests cover
both exclusion directions, wrong-token completion, and a resize timeout that
leaves the lease held.

## 5. Core invariants and design decisions

### 5.1 Stable rank pairing

For every registered rank `r`:

```text
coordinator rank r
== rollout worker rank r
== environment worker rank r
== T3 canonical DP rank r
```

T5 receives this mapping; it does not derive or reorder it. Construction fails
before actor creation if env and rollout handle keys differ or are not exactly
the declared contiguous rank set.

### 5.2 Worker state is authoritative for physical safety

The coordinator keeps transaction records and tokens, but it validates them
against both workers before every transition. It never treats its own cached
record as proof that a model is non-resident.

Successful shrink requires two verified worker receipts/statuses. Successful
expansion requires both workers to report matching lifecycle, policy version,
rank, expected transition, and active residency.

### 5.3 One collection context at a time

Before an elastic generation request can trigger expansion, the driver-side
controller configures:

```python
ElasticCollectionContext(
    lifecycle_generation=L,
    policy_version=V,
    ranks=(0, 1, ...),
)
```

and binds the four existing channels needed by the worker methods. The context
is immutable for that collection. Reconfiguration is allowed only when:

- no resize is running;
- no policy-sync lease is held;
- no rank is active, draining, snapshotting, paused, or expanding;
- all prior run references have resolved; and
- the new lifecycle generation is strictly greater than the old one.

Cold activation gets `L` and `V` from this context. Paused activation also
requires the stored token to carry the same `L` and `V`.

### 5.4 One callback transaction at a time per pipeline

Use an async resize lock even though core normally issues at most one callback
per pipeline per scheduling phase. This protects against manual calls,
duplicate scheduler work, and policy-sync races. Different pipelines remain
independent because each has its own actor.

Within one callback, different ranks may be drained, offloaded, prepared, and
started concurrently. Each rank's environment and rollout peer operations
should also run concurrently in disaggregated mode unless an operation has an
explicit ordering requirement.

### 5.5 Shrink before expansion

Core calls remove-only and add-only phases, but the public method validates the
general signature. If both lists are non-empty, it executes all removals to
completion before starting any addition. A rank cannot appear in both lists.

### 5.6 No automatic rollback after partial residency mutation

If one peer offloads and the other fails, do not onload the successful peer.
If one peer onloads and the other fails, do not offload the successful peer as
an automatic rollback. A rollback is another fallible residency operation and
cannot restore scheduler truth.

Instead:

- mark the pair failed where possible;
- preserve all diagnostics;
- raise the original operation error, with peer context attached safely; and
- let `rlix-core` retain its prior logical allocation and fail scheduling
  activity closed.

For a failed expansion, the prior logical state is inactive even though a
partial model may now be resident. Scheduler fail-fast prevents that bundle
from being reused; administrative recovery is required.

### 5.7 Deadlines do not cancel in-flight model work

Coordinator waits may have explicit positive deadlines. On expiry:

- do not use `ray.cancel(..., force=True)`;
- do not kill a worker actor;
- do not interrupt diffusion or policy inference;
- mark coordinator/pair state failed;
- raise a timeout to core; and
- retain logical ownership according to the uncommitted scheduler plan.

The remote operation may finish later, but no later callback may treat it as a
successful transaction after the coordinator has failed closed.

### 5.8 Callback success is not scheduler commit

The coordinator may cache that a local transition was applied, but it must
name this state `callback_applied` or `locally_active`, not
`scheduler_committed`. Core owns final commit. T7 progress reporting must not
publish a new active-rank snapshot in the callback/commit gap.

### 5.9 Policy sync uses an explicit lease

The coordinator exposes begin/end calls with an opaque lease ID. The fixed
`policy_sync` allocation must be acquired before taking the lease: acquiring
that allocation may need resize callbacks on donor pipelines, and taking a
lease first could deadlock if an invalid stale allocation required a callback
on this pipeline. T7 follows:

```text
finish and release this pipeline's collection
clear its generation demand
acquire the fixed policy_sync allocation
lease = begin_policy_sync(expected_policy_version=V)
try:
    perform all-rank sync and verified offload
    publish applied version
finally:
    end_policy_sync(lease)
    release the fixed policy_sync allocation
```

`begin_policy_sync()` accepts the configured collection's policy version or
its immediate successor (the next post-training version); other versions fail
closed. It requires that this coordinator has no locally active,
draining, snapshotting, paused, or expanding rank. T5 does not perform the
collective or acquire the fixed allocation. It guarantees only mutual
exclusion and lease identity. A second begin, wrong end token, wrong expected
version, or collection activation during the lease fails closed.

## 6. RLinf protocol data model

Add `rlinf/scheduler/rlix/protocol.py`. It may import RLinf lifecycle types but
must not import runner implementations.

### 6.1 Collection context

```python
@dataclass(frozen=True, slots=True)
class ElasticCollectionContext:
    lifecycle_generation: int
    policy_version: int
    dp_ranks: tuple[int, ...]
```

Validation requires positive lifecycle generation, non-negative policy
version, non-empty unique non-boolean integer ranks, and contiguous rank
identity for the first milestone.

Channels are runtime handles rather than protocol identity. Store them in a
separate internal binding so context equality and logs do not serialize
channel internals.

### 6.2 Policy-sync lease

```python
@dataclass(frozen=True, slots=True)
class PolicySyncLease:
    lease_id: str
    expected_policy_version: int
```

Only the coordinator creates lease IDs. They must be non-empty and
unpredictably unique within the actor lifetime.

### 6.3 Per-rank callback record

Use an internal mutable record, not a second public lifecycle enum:

```python
@dataclass(slots=True)
class _RankRecord:
    token: SafePointToken | None = None
    env_run_ref: Any | None = None
    rollout_run_ref: Any | None = None
    last_env_result: ElasticRunResult | None = None
    last_rollout_result: ElasticRunResult | None = None
    callback_applied_active: bool = False
    failure: str | None = None
```

The T2 worker enums remain authoritative. Do not invent coordinator states
that can disagree silently with `ElasticRankState`.

### 6.4 Coordinator status

Expose a frozen, serializable read model for diagnostics and T7 integration:

```python
@dataclass(frozen=True, slots=True)
class CoordinatorStatus:
    pipeline_id: str
    collection: ElasticCollectionContext | None
    callback_applied_active_ranks: tuple[int, ...]
    paused_ranks: tuple[int, ...]
    completed_ranks: tuple[int, ...]
    failed_ranks: tuple[int, ...]
    resize_in_progress: bool
    policy_sync_lease: PolicySyncLease | None
```

Status is observational. It is not an allocation grant and cannot authorize
GPU use.

### 6.5 Completed residency receipt

Add a T2-adjacent receipt rather than fabricating a pause token:

```python
@dataclass(frozen=True, slots=True)
class CompletedResidencyReceipt:
    worker_rank: int
    lifecycle_generation: int
    policy_version: int
    state: ElasticRankState
    model_resident: bool
    cuda_graph_captured: bool
```

It is valid only for `COMPLETED`, `model_resident=False`, and
`cuda_graph_captured=False`.

## 7. Coordinator actor construction and lookup

### 7.1 Actor class

Add `rlinf/scheduler/rlix/coordinator.py` with an async
`RLixResizeCoordinator`. Keep core protocol imports narrow:

- `ActionResponse`;
- `COORDINATOR_ACTOR_NAME_PREFIX`; and
- the registered pipeline namespace supplied to the factory.

The coordinator is RLinf-owned and may import Ray, channel handles, worker
lifecycle types, and RLinf logging. `rlix-core` continues importing none of
them.

### 7.2 Factory and actor options

Add a controller factory that creates the actor with:

```text
name      = f"rlix-core:coordinator:{pipeline_id}"
namespace = exact non-empty namespace registered for this pipeline
max_restarts = 0
max_task_retries = 0
```

Do not silently replace the supplied registered namespace with
`get_pipeline_namespace(pipeline_id)`: current `rlix-core` registration accepts
any non-empty namespace and scheduler lookup uses the stored value verbatim.
The deterministic helper may be used by a caller only when that same value is
also passed during pipeline registration.

Do not use `get_if_exists=True`: attaching new worker handles to a stale
coordinator would violate rank identity. A duplicate name is a construction
error unless the controller explicitly proves it owns the same actor instance.

Prefer owner-scoped lifetime for T5. If later T7 requires detached lifetime,
add explicit generation identity, teardown, and stale-actor tests before
changing it. A dead coordinator must cause lookup/callback failure, not leave a
silently reusable stale callback.

### 7.3 Registration ordering

T6/T7 must eventually use this order:

```text
create worker actors
create named coordinator actor with exact ranked handles
register pipeline using that coordinator namespace
admit pipeline
configure collection before elastic generation can be requested
```

T5 provides construction helpers and tests actor discoverability. It does not
yet edit the embodied entrypoint.

### 7.4 Teardown ordering

The future caller must:

```text
release/finish all allocations
unregister the pipeline from core
verify no callback or sync lease is active
close the coordinator
then close worker actors/channels
```

Closing the coordinator while core can still schedule the pipeline is an
error. T5 exposes a guarded `close()` but T7 owns normal invocation.

## 8. Driver-side controller

Add `rlinf/scheduler/rlix/controller.py` with a non-Ray
`RLixStageController`. Its T5 responsibilities are deliberately small:

- validate and extract ranked actor handles from the two worker groups;
- create and retain the named coordinator handle;
- configure one collection context and channel binding;
- expose async/sync wrappers for status and rank-result queries;
- expose a policy-sync context manager backed by begin/end lease calls; and
- perform guarded coordinator teardown.

It must not:

- infer placement/GPU mappings;
- request stage GPUs;
- run policy sync, collection, actor training, or evaluation;
- aggregate trajectories; or
- decide when a batch is sealed.

Those operations are added to the controller protocol in T7 or composed around
the T5 primitives.

Avoid blocking `ray.get()` inside an already-running event loop. Provide an
async implementation and, only if current synchronous runner integration needs
it later, a thin synchronous boundary owned by T7.

## 9. Collection configuration

### 9.1 Configure before allocation

`configure_collection(context, channels)` must finish before T7 requests
elastic generation. Otherwise a scheduler expansion could invoke the callback
without lifecycle or policy identity.

The runtime channel binding contains:

- env input channel;
- rollout request channel;
- optional reward channel; and
- optional actor trajectory channel.

T5 accepts `None` only where the current worker signature accepts it. T6/T7
feature validation still decides which combinations are supported in
production.

### 9.2 Cold workers remain cold until expansion

Do not call `prepare_elastic_collection()` for every rank during configuration:
that method onloads the worker model/environment. It must run only inside the
expansion callback after core has planned that rank's bundle.

The configured context is controller-owned evidence that a cold rank has
assigned work; T4's progress tracker remains the source of cold eligibility.

### 9.3 New lifecycle reuse

For a worker pair previously `COMPLETED`, cold-style expansion into a newer
collection calls each worker's existing `prepare_elastic_collection(L, V)`.
Both worker methods already require a strictly newer lifecycle generation.
The coordinator clears old results, tokens, and failure-free transaction
records only after both peers validate the new context.

Paused ranks cannot be carried into a different collection context. The batch
must finish or fail before reconfiguration.

## 10. Expansion transaction

### 10.1 Preflight

Before mutating either worker:

1. Validate ranks structurally, sort deterministically, and reject duplicates.
2. Require a configured collection and channels.
3. Reject ranks already locally active or with unresolved run calls.
4. Query both worker statuses concurrently.
5. Require rank, lifecycle, policy, expected transition, and peer-state
   compatibility.
6. Classify each pair as cold/new-lifecycle or paused-current-lifecycle.
7. Reject completed-current-lifecycle, failed, draining, snapshotting, or
   partially mismatched pairs.

No rank may be silently dropped from the callback.

### 10.2 Cold or completed-new-lifecycle activation

For each eligible pair:

```text
env.prepare_elastic_collection(L, V)
rollout.prepare_elastic_collection(L, V)
require both return EXPANDING for rank r, lifecycle L, version V
launch and store both long-running elastic calls
wait until both status surfaces report ACTIVE
mark callback_applied_active
```

Call the two preparation methods concurrently for disaggregated fakes/tests.
Do not encode collocated simultaneous-residency assumptions in the generic
protocol; the later runtime integration may supply an ordering policy after T6
placement classification.

### 10.3 Paused resume

For each paused pair:

```text
token = coordinator stored token
require env status and rollout status match token
env.prepare_elastic_resume(token)
rollout.prepare_elastic_resume(token)
require matching EXPANDING receipts
launch and store both long-running calls
wait until both report ACTIVE with the token transition
clear the stored token only after both calls are active
mark callback_applied_active
```

The environment loop owns dispatch of the retained bootstrap. The coordinator
must never send it directly. Starting both run calls and waiting for `ACTIVE`
ensures the callback does not return after onload while the peer exchange is
still unstarted.

### 10.4 Activation acknowledgment

The existing run methods transition `EXPANDING -> ACTIVE` before their first
channel wait. T5 may poll `get_elastic_status()` with bounded cooperative
yields after dispatching the run calls. If implementation shows actor task
ordering makes this ambiguous, add a narrow worker activation acknowledgment;
do not treat an unresolved run object reference as proof of activation.

### 10.5 Expansion failure

If any prepare, receipt, launch, or active-status check fails:

- preserve the first exception;
- request `fail_elastic_lifecycle()` on both peers where possible;
- retain all run refs and diagnostics for inspection;
- do not mark the rank active;
- do not try another rank after a transaction-wide failure unless all
  previously started rank results are accounted for; and
- raise so core cannot commit any expansion in that callback batch.

## 11. Shrink transaction

### 11.1 Preflight

Before submitting drains:

1. Validate all ranks and require them to be locally callback-applied active.
2. Require stored env and rollout run references for each rank, unless both
   have already resolved as `COMPLETED`.
3. Query both peer statuses.
4. Reject failed, paused, cold, expanding, or mismatched pairs.
5. Build one unique `DrainRequest` per active rank from the configured
   lifecycle and policy version.

### 11.2 Active drain

For each active pair:

```text
submit identical DrainRequest D to env and rollout
await both request acknowledgments
await both stored run references
```

Then require:

- both outcomes are `PAUSE_READY`, or both outcomes are `COMPLETED` due to a
  final-boundary race;
- two `PAUSE_READY` results carry identical non-`None` tokens;
- token request ID, rank, lifecycle, policy version, and next transition match;
- neither peer has an unresolved channel operation; and
- environment snapshot readiness agrees with the token.

Mixed `PAUSE_READY`/`COMPLETED`, differing tokens, or a missing run result is a
paired protocol failure.

### 11.3 Paused offload

For matching pause tokens:

```text
env.offload_elastic_environment(token)
rollout.offload_elastic_rollout(token)
```

Require both receipts to contain the exact token, `PAUSED`, non-resident model
state, and no CUDA graph. Query final statuses as an independent check before
returning callback success. Store the token for later same-rank resume.

### 11.4 Completed-rank offload

For a rank already completed before drain, or whose drain raced with final
completion:

```text
env.offload_completed_elastic_environment()
rollout.offload_completed_elastic_rollout()
```

Require matching `CompletedResidencyReceipt`s and final non-resident
`COMPLETED` statuses. Do not store a safe-point token and do not add the rank to
resumable eligibility. T4 progress continues reporting it in
`completed_dp_ranks` until the next lifecycle is configured.

### 11.5 Shrink success

Only after every selected pair passes final verification:

- clear its callback-applied active marker;
- retain tokens only for paused ranks;
- retain completed results/metrics for T7 consumption; and
- return `ActionResponse(success=True)`.

Core currently treats a returned callback as success rather than inspecting
the response flag. Therefore T5 must raise on every failure and must never
return `ActionResponse(success=False)` as a substitute for an exception.

### 11.6 Shrink failure

On any failure or deadline:

- do not clear callback-applied active markers;
- do not erase a successfully captured token or receipt;
- do not rollback a successfully paused peer;
- mark affected pairs failed where possible; and
- re-raise the original exception.

This intentionally leaves coordinator local state reflecting partial physical
work while core retains the previous logical allocation. Scheduler fail-fast
prevents unsafe reuse.

## 12. Completion, progress, and T4 release handoff

T5 does not decide when to release a completed rank, but it must expose enough
state for T7 to implement the T4 sequence correctly.

### 12.1 Rank result observation

Provide a method that awaits or polls a selected pair's stored run references
and returns a paired result only when both are resolved. It validates matching
outcomes and records environment metrics without consuming them twice.

For `COMPLETED`:

- query `EnvWorker.get_elastic_progress()`;
- preserve the complete trajectory count;
- mark the rank completed in coordinator status; and
- leave callback-applied active true until T4 release commits.

For `PAUSE_READY`, only a resize callback may finalize offload and publish the
rank as paused.

### 12.2 Future T7 order

T7 will use:

```text
observe both rank calls COMPLETED
update ElasticProgressTracker with worker progress
report completed rank while it is still active
await scheduler.await_release_dp_ranks([rank])
callback verifies/offloads the completed pair
after release returns, report the rank inactive
```

T5 must not call `await_release_dp_ranks()` from inside `resize_infer`; that
would recursively wait on itself.

### 12.3 Paused progress

After a successful shrink callback, the coordinator can expose the paused token
and verified worker statuses. T7 updates progress with the worker snapshot and
core allocation state only after its scheduler request observes commit.

## 13. Policy synchronization serialization

### 13.1 Gate state

The coordinator owns:

- one `asyncio.Condition`;
- a boolean or operation ID for resize in progress; and
- zero or one `PolicySyncLease`.

`resize_infer()` waits for no lease, then marks resize active for the whole
callback. `begin_policy_sync()` waits for no resize and no lease, validates the
expected next/active policy version contract, creates a lease, and returns it.

### 13.2 End lease

`end_policy_sync()` requires exact lease equality. On success it clears the
lease and notifies waiters. Wrong, duplicate, or absent lease completion
raises. The controller context manager must call it in `finally` and preserve
an earlier collective exception if cleanup also fails.

### 13.3 Failure behavior

Do not automatically expire a lease. Expose lease age in diagnostics if useful,
but unlocking after a timeout could overlap a still-running all-rank
collective with expansion. Operator teardown is safer than guessing.

A coordinator callback waiting behind a lease may be bounded by its caller's
deadline. Its timeout raises and core retains the old allocation.

### 13.4 Worker mutation guards

Existing T2 guards reject weight sync while workers are active, draining,
snapshotting, paused, or expanding. T5 tests the gate itself with fake sync
work; T7 must additionally prove the actual worker collective runs only in an
allowed between-lifecycle stage.

## 14. Failure and retry semantics

### 14.1 Validate before mutation

Reject before worker calls:

- non-list callback values;
- boolean, non-integer, negative, duplicate, unknown, or overlapping ranks;
- missing collection context or channels;
- stale lifecycle or policy identity;
- missing run references for an active pair;
- adding a current-lifecycle completed rank;
- removing a cold or already paused rank;
- any `FAILED_RESIDENT` peer; and
- any policy-sync lease conflict that cannot be waited safely.

### 14.2 Idempotency boundary

Core does not currently retry a successful callback before commit. Keep the
public contract strict:

- empty remove/add is a validated no-op success;
- repeated worker offload with the same pause token remains worker-idempotent;
- repeating an already successful shrink callback is rejected because the
  rank is no longer locally active;
- repeating a successful expansion is rejected because the rank is active;
- a duplicate in-progress callback waits behind the resize lock and then is
  revalidated against the new state; and
- failed callbacks are not automatically retried.

If future core retry semantics require operation IDs, add them to the core
protocol explicitly; do not guess callback equivalence from rank lists.

### 14.3 Preserve original errors

Re-raise the first worker or timeout exception. Add rank/peer/operation context
using exception chaining or a coordinator exception whose `__cause__` is the
original error. Store only concise non-sensitive text in status.

### 14.4 Pair mismatch

Any disagreement in state, outcome, token, lifecycle, policy version,
transition ID, residency, or completion is a composite-rank failure. A single
healthy peer is insufficient evidence to release or acquire the bundle.

### 14.5 Actor and driver loss

- Coordinator actor loss makes core callback lookup/invocation fail closed.
- Worker actor loss makes the callback raise and prevents commit.
- Driver loss must not cause a surviving stale coordinator to accept a new
  pipeline instance under the same identity.
- No T5 path restarts actors or restores from durable checkpoints.

## 15. Concurrency and ordering audit

### 15.1 Required Ray worker concurrency

The callback sends control calls while long-running elastic methods may be
waiting or executing. Production worker actors require `max_concurrency >= 2`.
T6 owns the launch edit and validation; T5 construction accepts a declared
capability and fails before coordinator creation when a test/controller can
prove concurrency is insufficient.

Do not infer actual Ray concurrency from Hydra fields inside workers.

### 15.2 Safe interleaving

Control calls may interleave at T2's documented async yields. They must not
interleave inside:

- VLA prediction;
- trajectory append;
- world-model `chunk_step()`;
- reward/reset/metric commit;
- model offload/onload verification; or
- snapshot restore commit.

T5 does not add worker yields.

### 15.3 Multi-rank callback

For ranks `[0, 1]`, issue peer drains concurrently, then await all results.
Do not return early after rank 0 succeeds if rank 1 is unresolved. Any rank
failure fails the complete callback batch, matching core's atomic plan commit.

### 15.4 Policy-sync race matrix

Tests cover:

| First operation | Second operation | Required result |
| --- | --- | --- |
| resize | resize | second waits, then revalidates |
| resize | policy sync | sync waits until callback ends |
| policy sync | resize | callback waits until valid lease ends |
| policy sync | policy sync | second waits or is rejected deterministically |
| failed resize | policy sync | sync does not begin on failed coordinator state |
| wrong lease end | resize | lease remains held; resize cannot pass |

### 15.5 Final-completion race

A drain requested at the final bootstrap may observe both long-running calls
complete rather than pause. Treat this as successful completed release only
after completed offload verification. Never fabricate a resumable token.

## 16. Test plan

### 16.0 Test classification and doubles audit

T5 has no production end-to-end test. That is intentional at this task
boundary: T6 must construct placement and worker concurrency, T7 must route the
real runner through the coordinator, and T8 owns real Wan/OpenSora two-pipeline
GPU reuse. Do not describe the following scoped evidence as end to end:

- fake-peer unit tests run the production coordinator transaction with
  deterministic async worker protocol doubles;
- the in-memory worker protocol test runs production coordinator and public
  `EnvWorker`/`MultiStepRolloutWorker` lifecycle methods, but stubs model/world
  computation, channel transport, and snapshot validation/restore backends;
- the core fail-closed integration test runs production `SchedulerImpl` and
  production coordinator code, with fake workers and a thin `.remote()` adapter
  that invokes the real coordinator coroutine;
- the opt-in local Ray integration test uses real `ray.remote`, named actor
  creation, cross-actor handles, namespace lookup, status RPC, guarded close,
  and actor termination, while its workers expose only cold status; and
- the controller-options unit test monkeypatches `ray.remote` only to inspect
  construction options and is backed by the separate real-Ray integration
  test.

No mock replaces `RLixResizeCoordinator.resize_infer()` in tests that claim to
exercise coordinator behavior. Ordering is asserted from worker-observable
offload/prepare events without wrapping coordinator internals. These tests
satisfy T5's CPU callback-transaction boundary, not T6-T8 runtime acceptance.

### 16.1 Protocol validation

Add `tests/unit_tests/test_rlix_resize_coordinator.py` and cover:

- collection context and lease validation;
- empty, malformed, duplicate, unknown, and overlapping callback ranks;
- deterministic rank ordering;
- mismatched env/rollout handle maps; and
- context replacement rules.

### 16.2 Cold expansion

With fake async worker peers, assert:

- both prepare calls receive the same lifecycle and policy version;
- run calls launch once per rank;
- callback waits for both `ACTIVE` acknowledgments;
- one rank can activate without activating siblings;
- multi-rank activation returns only after every rank is active; and
- preparation, launch, active-check, and deadline failures raise without a
  false success response.

### 16.3 Active shrink

Cover drains requested while fake peers model each T2 timing case:

- waiting for observation;
- policy prediction;
- waiting for result;
- world-model chunk execution;
- immediately before next observation send;
- immediately after the send; and
- final completion.

Assert the callback returns only after matching results, tokens, offload
receipts, and final residency checks.

T2 already proves channel-level correctness. T5 fakes model public outcomes
rather than duplicate the entire channel implementation, plus one protocol
integration test uses the stubbed-backend in-memory worker pair.

### 16.4 Paused expansion

Cover:

- exact token resume;
- retained observation dispatched by EnvWorker exactly once;
- matching expected transition after activation;
- stale lifecycle, wrong rank, wrong policy, and wrong transition rejection;
- one peer onload/restore failure; and
- no automatic rollback after partial success.

### 16.5 Completed release

Add focused worker and coordinator tests proving:

- completed environment and rollout models are verified non-resident;
- completed trajectory progress survives offload;
- no pause token or resumable state is created;
- repeated verified completed offload is idempotent;
- completed offload failure enters `FAILED_RESIDENT`; and
- the callback raises so core ownership cannot commit release.

### 16.6 Pair failure matrix

Inject:

- one drain request rejection;
- one run call exception;
- `PAUSE_READY` versus `COMPLETED` mismatch;
- different safe-point tokens;
- environment snapshot not ready;
- one offload failure;
- one residency verification failure;
- one resume failure;
- one activation-status failure;
- worker actor death; and
- coordinator deadline expiry.

In every case, assert no success response, no unsafe active-marker update, no
automatic rollback, and concise preserved diagnostics.

### 16.7 Policy-sync gate

Use blocking fake operations to prove both directions of mutual exclusion,
lease-token validation, cleanup in `finally`, and no automatic lease expiry.

### 16.8 Selected-rank isolation

With two pairs:

1. activate both ranks;
2. shrink rank 0 while rank 1 continues;
3. require rank 0 paused/non-resident and rank 1 active;
4. expand rank 0 from its exact token;
5. complete both; and
6. release one completed rank without changing the sibling.

### 16.9 Core callback transaction integration

Use a lightweight Ray test or existing core fake boundary to prove:

- actor name and namespace lookup succeed;
- `rlix-core` invokes the RLinf actor with sorted exact ranks;
- callback failure prevents plan commit;
- shrink completion precedes expansion invocation; and
- T4 release waits through completed offload before commit.

Keep this CPU-only. Real Wan/OpenSora reuse belongs to T8.

### 16.10 Regression suites

Run:

- all focused T1/T2 lifecycle tests;
- T4 RLinf progress tests;
- relevant core scheduling/composite/release tests; and
- the broader embodied unit subset already recorded by prior tasks.

## 17. File-by-file edit list

### 17.1 Required new RLinf production files

`rlinf/scheduler/rlix/protocol.py`

- collection context;
- policy-sync lease;
- coordinator status/read models; and
- pure validation helpers.

`rlinf/scheduler/rlix/coordinator.py`

- named async callback actor implementation;
- per-rank run-reference and token ownership;
- cold expansion, active shrink, paused resume, and completed release;
- pair validation and failure propagation;
- operation deadlines;
- status/result surfaces; and
- policy-sync gate.

`rlinf/scheduler/rlix/controller.py`

- rank-to-actor extraction;
- coordinator construction and naming;
- collection/channel configuration wrappers;
- policy-sync context manager;
- status/result access; and
- guarded close.

### 17.2 Required existing RLinf production files

`rlinf/scheduler/rlix/__init__.py`

- export only the intended public T4/T5 surfaces;
- avoid importing optional `rlix_core` dependencies on unrelated RLinf paths
  if import-time compatibility requires lazy exports.

`rlinf/workers/elastic_rollout_lifecycle.py`

- add `CompletedResidencyReceipt`;
- allow `COMPLETED -> FAILED_RESIDENT` only for failed verified release or
  coordinator-declared paired failure; and
- retain all existing T2 transitions and validation.

`rlinf/workers/env/env_worker.py`

- add verified completed-environment offload;
- add public coordinator-declared failure recording;
- keep durable completed-trajectory progress unchanged; and
- expose enough final status to prove no active call/residency remains.

`rlinf/workers/rollout/hf/huggingface_worker.py`

- add verified completed-rollout offload including auxiliary models and CUDA
  graph cleanup;
- add public coordinator-declared failure recording; and
- preserve the existing pause/resume APIs.

### 17.3 Required tests

`tests/unit_tests/test_rlix_resize_coordinator.py`

- primary fake-peer callback, gate, concurrency, deadline, and failure suite.

`tests/unit_tests/test_elastic_env_rollout.py`

- completed offload receipts/failures;
- public failed-lifecycle method; and
- one stubbed-backend local worker pair driven through the coordinator
  transaction order.

`tests/unit_tests/test_rlix_progress.py`

- regression that completed progress survives callback-driven offload and is
  never made resumable.

Core tests should change only if the actor lookup integration needs a focused
fixture. Do not rewrite existing callback-order tests around RLinf.

### 17.4 Completed documentation updates

After all T5 exit criteria passed:

- mark T5 complete in `../VLA_COMPATIBILITY_DESIGN.md`;
- mark T5 complete and record commands/results in
  `../VLA_COMPATIBILITY_IMPLEMENTATION_PLAN.md`;
- update T5 status in `RLIX_ELASTIC_VLA_ROLLOUT_IMPLEMENTATION_PLAN.md`; and
- record the completed-offload lifecycle clarification in the architecture.

Do not mark T6-T8 complete or claim live cross-pipeline GPU reuse.

### 17.5 Explicitly deferred files

`rlinf/config.py`, placement modules, and
`examples/embodiment/train_embodied_agent.py`

- T6 owns configuration, topology, registration wiring, and worker launch
  concurrency.

`rlinf/runners/embodied_runner.py`

- T7 owns stage adoption, progress/release calls, batch sealing, and standalone
  no-op preservation.

GPU acceptance configs/scripts and operator docs

- T8 owns real two-pipeline Wan/OpenSora acceptance and utilization evidence.

## 18. Completed implementation sequence (preserved plan)

1. Add pure T5 protocol types and validation tests.
2. Add `CompletedResidencyReceipt` and the failure-only completed transition.
3. Implement completed environment offload with success, idempotency, and
   failure tests.
4. Implement completed rollout/auxiliary-model/CUDA-graph offload with the same
   tests.
5. Add the narrow public worker failed-lifecycle method and pair-mismatch tests.
6. Implement an in-process coordinator core with fake actor clients, no Ray
   naming yet.
7. Add collection context/channel binding and cold expansion.
8. Store run references and implement active drain through matching tokens.
9. Add paused resume and exact active acknowledgment.
10. Add completed-rank release and final-completion race handling.
11. Add multi-rank atomic callback behavior and selected-rank isolation.
12. Add operation deadlines without remote cancellation.
13. Add the condition-based policy-sync lease and its race matrix.
14. Wrap the tested core as the named Ray actor using exact core constants.
15. Add the driver-side controller and actor-handle extraction.
16. Add actor lookup/core failure-before-commit integration coverage.
17. Run focused T1/T2/T4/T5 suites, then broader RLinf and core regressions.
18. Run Ruff, format checks, compilation, inspect the full diff, and update
    completion documents only with recorded evidence.

Keep commits reviewable along these boundaries. In particular, land completed
offload semantics before the coordinator consumes them, and land the pure
coordinator transaction tests before adding Ray naming/lifecycle code.

## 19. Verification commands

From `/root/_VLAMP/RLinf`:

```bash
export PYTHONPATH="$PWD:/root/_VLAMP/rlix-core/src${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_rlix_resize_coordinator.py \
  tests/unit_tests/test_rlix_progress.py \
  tests/unit_tests/test_elastic_env_rollout.py \
  tests/unit_tests/test_world_model_resume.py
```

The real local-Ray actor name/namespace check is opt-in because Ray startup
requires loopback/network process discovery that may be unavailable in a
restricted test sandbox:

```bash
export PYTHONPATH="$PWD:/root/_VLAMP/rlix-core/src${PYTHONPATH:+:$PYTHONPATH}"
RLINF_RUN_LOCAL_RAY_TEST=1 /root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_rlix_resize_coordinator.py -k named_ray
```

Run the prior broader embodied subset:

```bash
export PYTHONPATH="$PWD:/root/_VLAMP/rlix-core/src${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_elastic_env_rollout.py \
  tests/unit_tests/test_world_model_resume.py \
  tests/unit_tests/test_maniskill_offload_env.py \
  tests/unit_tests/test_overlap_env_bootstrap.py \
  tests/unit_tests/test_history_manager.py \
  tests/unit_tests/test_rlix_progress.py \
  tests/unit_tests/test_rlix_resize_coordinator.py
```

Lint and format only the touched RLinf files first, then use the repository's
normal broader checks as available:

```bash
/root/.venv/bin/ruff check \
  rlinf/scheduler/rlix \
  rlinf/workers/elastic_rollout_lifecycle.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_rlix_resize_coordinator.py \
  tests/unit_tests/test_rlix_progress.py \
  tests/unit_tests/test_elastic_env_rollout.py
/root/.venv/bin/ruff format --check \
  rlinf/scheduler/rlix \
  rlinf/workers/elastic_rollout_lifecycle.py \
  rlinf/workers/env/env_worker.py \
  rlinf/workers/rollout/hf/huggingface_worker.py \
  tests/unit_tests/test_rlix_resize_coordinator.py \
  tests/unit_tests/test_rlix_progress.py \
  tests/unit_tests/test_elastic_env_rollout.py
/root/.venv/bin/python -m compileall -q \
  rlinf/scheduler/rlix \
  rlinf/workers/elastic_rollout_lifecycle.py
```

From `/root/_VLAMP`, verify core callback/release regressions remain green:

```bash
export PYTHONPATH="$PWD/rlix-core/src${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  rlix-core/tests/test_scheduling_cycle.py \
  rlix-core/tests/test_composite_bundle_scheduling.py \
  rlix-core/tests/test_rank_specific_release.py
/root/.venv/bin/python -m pytest -q rlix-core/tests
```

T5 requires no accelerator run. Do not turn a local fake callback pass into a
GPU-reuse claim.

## 20. Definition of done

T5 is complete only when all of the following are true:

- A named RLinf-owned coordinator is discoverable through the exact
  `rlix-core` actor name and registered namespace protocol.
- `resize_infer(remove, add)` validates exact ranks and preserves
  shrink-before-expand ordering.
- Cold expansion prepares the correct same-ranked pair for the configured
  lifecycle and policy version.
- Paused expansion validates the stored token, restores both peers, starts both
  run calls, and returns only after both are active.
- Active shrink waits for the current committed chunk, receives identical peer
  tokens, and verifies both peers non-resident before returning.
- Final-completion races and already completed ranks use a token-free verified
  completed offload path.
- Completed progress remains durable and completed ranks never become
  resumable in the same lifecycle.
- The coordinator owns and validates the exact long-running env/rollout object
  references for each activated rank.
- Selected-rank resize leaves siblings unchanged.
- Multi-rank callback success is atomic from core's perspective; one rank
  failure makes the callback raise.
- Partial offload/onload failure performs no automatic rollback and cannot
  produce callback success.
- Coordinator-detected peer mismatch moves the pair to a fail-closed diagnostic
  state where possible.
- Deadlines never force-cancel inference, diffusion, or worker actors.
- Policy synchronization and resize are mutually exclusive through an exact
  lease protocol.
- Callback failure prevents core plan commit and retains the previous logical
  allocation.
- Existing T1/T2/T4 behavior and core legacy callback semantics remain green.
- Focused lint, format, compilation, and regression commands pass or unchanged
  pre-existing findings are recorded precisely.

T5 completion means the callback transaction is implemented and tested with
CPU fakes/local Ray integration. It does not mean production placement is
validated, `EmbodiedRunner` uses the coordinator, or another pipeline has
reused a real Wan/OpenSora bundle. Those claims remain T6, T7, and T8.
