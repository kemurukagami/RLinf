# Task 7 Detailed Implementation Plan: Runner Stage Integration

## 1. Status and source of truth

Status: planned. T0-T6 are complete; T7 is the next implementation task and
T8 two-pipeline accelerator acceptance remains pending.

This document expands Task T7, "Runner stage integration," from these sources,
in descending order of authority:

1. `../VLA_COMPATIBILITY_DESIGN.md`, especially stage lifecycle, resize
   lifecycle, configuration, parity, and definition of done;
2. `../VLA_COMPATIBILITY_IMPLEMENTATION_PLAN.md`, which defines the canonical
   T0-T8 sequence and T7 exit condition;
3. `RLIX_ELASTIC_VLA_ROLLOUT_IMPLEMENTATION_PLAN.md`, especially the T7 stage
   pseudocode and shared safe-point invariant; and
4. the completed detailed T1-T6 plans, whose state, ownership, progress,
   callback, placement, and bootstrap contracts T7 must consume unchanged.

If this plan conflicts with `../VLA_COMPATIBILITY_DESIGN.md`, the architecture
document wins. If implementation work requires a different allocation,
safe-point, policy-version, or stage-ordering contract, update the architecture
document and both canonical implementation plans in the same change.

The deferred alternatives in `../VLA_WM_UTILIZATION_FUTURE_DESIGN.md` do not
change T7. In particular, T7 does not introduce unequal rollout/environment
pools, asynchronous shared inference, dynamic microbatching, model parallel
Wan/VLA execution, or intra-chunk overlap. It integrates the approved atomic
paired-rank design only.

Implementation checklist:

- [ ] Add a synchronous, runner-facing stage API to the registered runtime.
- [ ] Make fixed-stage acquisition and verified release one fail-closed
  transaction.
- [ ] Acquire `initialization` before model/environment initialization and
  establish the all-offloaded baseline before release.
- [ ] Serialize fixed `policy_sync` allocation with the T5 coordinator lease.
- [ ] Configure one immutable elastic collection lifecycle before generation
  allocation is requested.
- [ ] Publish the initial cold-rank progress snapshot before the blocking
  `actor_infer` request.
- [ ] Monitor rank completion, publish durable progress, and request exact
  completed-rank release in the T4/T5 order.
- [ ] Seal the actor batch only after every assigned trajectory has arrived and
  every transition has the expected policy version.
- [ ] Clear progress and release any remaining generation ownership only after
  the complete batch is sealed and worker residency is verified.
- [ ] Compute advantages only from the sealed CPU batch, then acquire
  `actor_train` around all-rank FSDP training and verify actor offload before
  release.
- [ ] Advance the runner policy version only after successful optimizer
  completion.
- [ ] Acquire fixed `evaluation` around evaluation after an all-rank policy
  sync, then verify rollout/environment offload before release.
- [ ] Preserve the exact standalone call order when `rlix_runtime is None`.
- [ ] Close the registered runtime on normal exit and preserve the primary
  exception when cleanup also fails.
- [ ] Add CPU fake-worker/runtime tests for ordering, partial activation, live
  resize, sealing, failure, cleanup, and disabled-mode parity.
- [ ] Run the focused T1-T7, complete `rlix-core`, lint, format, compilation,
  configuration, and dependency-boundary checks.
- [ ] Update the canonical status documents only after every definition-of-done
  item in section 19 passes.

Do not mark T7 complete from unit tests alone if no production entrypoint uses
the new runtime. Conversely, real two-pipeline Wan/OpenSora utilization and
recovery evidence belongs to T8, not T7.

## 2. Required outcome

When `cfg.rlix.enabled` is true, `EmbodiedRunner` must run every GPU-bearing
stage under the registered RLix runtime and must never use a GPU before its
corresponding allocation has been granted.

The enabled lifecycle is:

```text
T6 bootstrap
  register/admit five clusters
  create named resize coordinator
  hand inactive runtime to runner

fixed initialization
  acquire initialization union
  initialize rollout, environment, reward if CPU-only, and actor
  restore checkpoint if configured
  establish verified CPU-offloaded baseline
  release initialization union

for policy version V:
  fixed policy_sync
    acquire actor+all-rollout union at generation priority
    acquire coordinator policy-sync lease for V
    synchronize actor version V to every rollout rank
    verify all rollout ranks applied V and are offloaded
    end lease, then release policy_sync union

  elastic actor_infer
    configure lifecycle L with policy version V and canonical ranks
    publish cold/resumable progress and complete trajectory target
    start the actor trajectory receiver
    request generation; continue after any eligible rank activates
    observe completions while scheduler-driven shrink/expand continues
    for each completed active rank:
      update progress and report completion
      await exact rank release
      callback offloads/verifies both peers before scheduler commit
      report the committed inactive state
    wait for all assigned trajectories and all actor receives
    seal one complete, version-V batch
    clear progress and release any remaining generation ownership

  sealed batch processing
    compute advantages and returns from the sealed batch

  fixed actor_train
    acquire all-rank actor mapping
    train all FSDP ranks
    verify optimizer success and actor offload
    release actor_train union
    advance runner version to V+1

  optional fixed evaluation
    synchronize V+1 to every rollout rank under policy_sync
    acquire evaluation rollout+environment union
    evaluate to completion
    verify rollout/environment offload
    release evaluation union

normal shutdown
  assert no active collection, pending release, or fixed allocation
  clear progress defensively
  unregister the pipeline
  close the named coordinator
```

The defining stage invariant is:

```text
RLix logical ownership granted
    before worker GPU onload or use

worker stage completed and residency verified
    before RLix logical ownership release
```

The defining batch invariant is:

```text
sealed lifecycle L batch
== every rank's assigned complete trajectories
== no duplicate or missing transition identity
== one policy version V throughout
== the only batch eligible for advantage calculation and actor training
```

When `rlix_runtime is None`, the existing standalone `init_workers()`,
`update_rollout_weights()`, `evaluate()`, and `run()` ordering remains the
behavioral reference. No core import, allocation call, elastic worker method,
or extra offload barrier may enter that path.

## 3. Scope boundaries

### 3.1 Supported by T7

- The synchronous non-pipelined `EmbodiedRunner` selected by the T6 validation
  contract.
- The registered five-cluster runtime created by T6.
- Fixed initialization, policy synchronization, FSDP actor training, and
  evaluation stages.
- Elastic Hugging Face VLA plus Wan/OpenSora collection.
- Equal, contiguous rollout/environment DP ranks with one complete model per
  worker and uniform collocated or disaggregated bundles.
- Partial initial generation activation and later scheduler-driven live
  shrink/expansion through the T5 callback actor.
- Controller-owned cold progress plus activated EnvWorker progress.
- Exact release of completed ranks while productive siblings continue.
- A single complete, policy-version-consistent actor batch per runner step.
- Resume from an actor checkpoint at the restored global step, followed by a
  fresh collection lifecycle.
- Existing checkpoint, evaluation, profiling, timing, and metric behavior once
  their required fixed allocation is held.
- Normal and exceptional runtime cleanup without replacing the primary error.

### 3.2 Explicitly not implemented by T7

- Changes to RLix planning, priorities, bundle canonicalization, eligibility,
  callback ordering, commit, tracing, or rank-release semantics.
- Selective rollout weight synchronization during rank expansion.
- Resizing the FSDP actor group.
- Async embodied runners, `run_pipeline()`, training-pipeline overlap,
  environment bootstrap overlap, or decoupled channels.
- Tensor parallelism, model pipeline parallelism, or application rollout
  pipelining.
- Unequal rollout/environment world sizes, cross-rank migration, shared VLA
  service pools, or dynamic inference batching.
- Independently scheduled GPU reward workers.
- Elastic evaluation; evaluation remains a fixed all-rank stage.
- Actor training on a partial batch or policy-stale data.
- Process-restart recovery of a partially collected batch.
- Real two-pipeline Wan/OpenSora recovery, utilization, and throughput claims;
  those are T8 acceptance.
- The alternative utilization architectures in
  `VLA_WM_UTILIZATION_FUTURE_DESIGN.md`.

### 3.3 Fail-closed boundary

Enabled mode must reject or stop before mutation when any required runtime
capability is missing. In particular:

- the runner has an enabled config but no registered runtime;
- the runtime pipeline ID, placement ranks, or channel binding is inconsistent;
- a fixed grant differs from the exact registered device union;
- a rollout rank does not apply the expected policy version;
- a collection lifecycle is reused or decreases;
- completed progress decreases or disagrees with the fixed assignment;
- paired run outcomes disagree;
- the actor receiver observes the wrong trajectory count, lifecycle, rank, or
  policy version;
- a worker remains resident at a release boundary;
- scheduler progress, request, callback, or release fails; or
- cleanup cannot prove that the allocation is safe to return.

Failure never converts uncertain physical residency into a successful logical
release. The original exception remains primary; cleanup failures are attached
as notes.

## 4. Relationship to T0-T6

### 4.1 T0 supplies fixed atomic allocation

T7 uses the completed fixed-policy path for `initialization`, `policy_sync`,
`actor_train`, and `evaluation`. It does not emulate fixed ownership by directly
moving models or by requesting the elastic generation cluster.

### 4.2 T1 supplies continuation state

T7 never reads or edits Wan/OpenSora snapshots. It waits for T5/T2 receipts and
uses lifecycle, policy version, and transition identity as opaque correctness
evidence.

### 4.3 T2 supplies local worker lifecycle

T7 starts no legacy `env.interact()` or `rollout.generate()` calls in enabled
collection. The T5 coordinator alone launches
`interact_until_pause_or_complete()` and
`generate_until_pause_or_complete()` for selected ranks.

### 4.4 T3 supplies canonical allocation identity

T7 derives cluster IDs from the registered pipeline ID and standard cluster
names. It uses `placement_plan.actor_infer_bundles` to interpret returned
generation GPU IDs and never re-slices the flat mapping or treats bundle width
as tensor parallelism.

### 4.5 T4 supplies progress and release semantics

T7 owns the publication loop. It must preserve missing-versus-empty rank-set
semantics, report completed work before requesting release, and wait on
`await_release_dp_ranks()` before publishing the rank as inactive.

### 4.6 T5 supplies callback and policy-sync serialization

T7 invokes `RLixStageController.configure_collection()`, observes paired rank
results/status, and holds the exact coordinator policy-sync lease around all
rollout weight mutation. It never calls rank offload/onload directly during
elastic collection.

### 4.7 T6 supplies immutable topology and inactive handoff

T7 accepts only `RegisteredRLixPipeline`. It does not register, admit, resolve
placement, or create another coordinator. The first allocation request occurs
inside T7 initialization, after the T6 handoff.

## 5. Current implementation and concrete gaps

### 5.1 The runtime is registered but behaviorally unused

`EmbodiedRunner` stores `rlix_runtime`, but `init_workers()`, `run()`,
`update_rollout_weights()`, `evaluate()`, advantage calculation, and training
still use the standalone path without allocation requests.

### 5.2 The synchronous runner has no stage façade

The scheduler and controller expose async Ray calls, while `EmbodiedRunner` is
synchronous. Scattering `asyncio.run()` and raw scheduler actor calls through
the runner would make cleanup and exception ownership inconsistent. T7 needs
one synchronous runtime surface.

### 5.3 Initialization happens outside RLix ownership

The entrypoint currently calls `runner.init_workers()` after T6 registration
but before any `initialization` request. Enabled mode can therefore load models
without a logical allocation.

### 5.4 Policy sync is not coupled to allocation and lease

`update_rollout_weights()` invokes the two collective halves directly. It does
not acquire the fixed `policy_sync` mapping or the T5 lease, and it does not
return an explicit all-rank version/residency receipt.

### 5.5 The runner does not configure or monitor elastic collection

T5 can configure a collection and observe paired results, and T4 can aggregate
progress, but no production loop connects them to scheduler progress,
generation requests, completed-rank release, or final cleanup.

### 5.6 Actor receive completion is not a batch seal

`recv_rollout_trajectories()` waits for a statically derived number of channel
messages and stores `rollout_batch`, but it does not return a receipt that binds
the batch to lifecycle, policy version, rank contributions, and expected
trajectory count. Training therefore has no explicit T7 barrier.

### 5.7 Actor training does not expose a release receipt

FSDP training loads weights and optimizer as needed, but the default training
method does not establish a public verified all-offloaded result before the
runner could release `actor_train`.

### 5.8 Metrics assume legacy group handles

The current logging path consumes `WorkerGroupFuncResult` duration data and
later waits on the legacy environment handle. Elastic rank calls are owned by
the coordinator and may pause/resume several times, so T7 must aggregate final
environment metrics without pretending there was one legacy group handle.

### 5.9 Normal runtime teardown is missing

The entrypoint calls `runner.run()` without a `finally` block that unregisters
the pipeline and closes the named coordinator. `RegisteredRLixPipeline.close()`
also correctly refuses unsafe coordinator shutdown, so T7 must reach a verified
inactive state before normal close.

## 6. Core invariants and design decisions

### 6.1 One runner-facing runtime owns all enabled stage transitions

Extend `RegisteredRLixPipeline` with a narrow synchronous orchestration API or
compose it with a `RLixRunnerRuntime` owned by the same object. The runner may
call this API but must not call raw scheduler actor methods.

The façade owns:

- cluster ID construction;
- priorities and global-step metadata;
- synchronous waiting for Ray operations;
- fixed allocation bookkeeping;
- collection lifecycle and progress tracker state;
- canonical active-rank projection;
- cleanup ordering; and
- preservation of primary errors.

### 6.2 Stage state is explicit

Track one of:

```text
INACTIVE
FIXED_INITIALIZATION
FIXED_POLICY_SYNC
ELASTIC_COLLECTION
FIXED_ACTOR_TRAIN
FIXED_EVALUATION
FAILED_UNCERTAIN
CLOSED
```

Only one stage may be active for a pipeline. A transition to
`FAILED_UNCERTAIN` prevents later allocation requests and normal unregister
until physical state is administratively resolved.

### 6.3 Fixed grants must match the registered union

For every fixed request, compare the returned GPU set with the exact tuple in
`RLixPlacementPlan`. A subset, superset, duplicate, or different mapping is a
fatal protocol error before worker onload.

### 6.4 Release follows verified offload

The fixed-stage context manager does not automatically release merely because
the body returned. The body must provide a typed completion receipt proving
the stage-specific workers are nonresident. Only then may the runtime call
`notify_release_gpus()`.

If the body raises before verification, preserve ownership and mark the runtime
failed uncertain. Bootstrap-level cleanup may unregister the pipeline, but it
must not record a successful stage release.

### 6.5 Policy version is the runner global step

Use `policy_version = global_step` for collection. A checkpoint restored at
`global_step_N` begins by synchronizing and collecting version `N`. Training
success advances `global_step` to `N+1`; a failed optimizer step does not.

Every rollout rank must return the applied version from policy sync. Every
trajectory version in the sealed batch must equal the collection version.

### 6.6 Collection lifecycle is monotonic and separate from policy version

Maintain a runtime-owned positive `lifecycle_generation`. Increment it exactly
once before each new collection configuration. Never infer it from transition
counts. It may be initialized from the restored global step only if the formula
is documented and strictly increasing for every new local lifecycle; storing a
separate counter is preferred.

### 6.7 Assignments are computed once from worker configuration

For each canonical rank, the initial milestone assignment is the exact
EnvWorker contract:

```text
train_num_envs_per_stage * stage_num * rollout_epoch
```

Because T6 requires equal uniform workers and `stage_num == 1`, assignments
are uniform today. Store the explicit rank mapping anyway and require worker
progress to match it. Do not use chunk count or tensor metric length as a
trajectory count.

### 6.8 Progress precedes generation demand

Configure the coordinator and publish the initial progress snapshot before
calling blocking `request_gpus(actor_infer)`. Otherwise the gap-ratio planner
has no authoritative demand or eligibility.

### 6.9 Partial activation is normal

`request_gpus()` returns after at least one eligible generation rank is active.
T7 must not require every registered bundle in the first result. It starts the
actor receiver before or atomically with the request, then monitors until every
assignment completes while the scheduler may resize.

### 6.10 Completion is durable before release

For each rank the order is fixed:

```text
paired worker calls resolve COMPLETED
-> EnvWorker progress validates full assigned count
-> tracker update
-> scheduler progress report includes completed rank
-> await_release_dp_ranks(rank)
-> T5 completed offload and verification
-> scheduler commit and waiter return
-> next progress snapshot reports rank inactive
```

Do not release from inside `resize_infer()` and do not report an inactive
completed rank before the release waiter returns.

### 6.11 Batch seal precedes advantages

The actor receiver may begin early, but the actor may not calculate advantages
until:

- all rank assignments are durably complete;
- every rank's paired result is `COMPLETED`;
- all expected trajectory messages have arrived;
- the actor batch contains exactly the expected count;
- every version equals the collection version; and
- lifecycle/rank contribution metadata contains no duplicate or omission.

Return an immutable `ElasticBatchReceipt` and require it as input or validated
state for advantage calculation and training in enabled mode.

### 6.12 Disabled mode is structurally unchanged

Branch once at public stage entry points. Do not retrofit the standalone path
through elastic fakes or make `rlix-core` an import-time requirement for
disabled execution.

### 6.13 Cleanup preserves the primary error

For nested stage failures:

1. preserve the first worker/scheduler/runner exception;
2. attempt only cleanup whose safety preconditions are known;
3. attach cleanup failures with `BaseException.add_note()`;
4. do not notify release for uncertain residency; and
5. do not let logging-thread or runtime-close errors replace the primary
   exception.

## 7. Runner-facing runtime protocol

### 7.1 Cluster identities

Construct once:

```python
initialization = f"{pipeline_id}_initialization"
policy_sync = f"{pipeline_id}_policy_sync"
actor_infer = f"{pipeline_id}_actor_infer"
actor_train = f"{pipeline_id}_actor_train"
evaluation = f"{pipeline_id}_evaluation"
```

Use protocol constants rather than literal suffixes in production code.

### 7.2 Fixed-stage context

Provide a synchronous context manager with a typed finish operation:

```python
with runtime.fixed_stage(
    cluster_name=INITIALIZATION_CLUSTER_NAME,
    priority=Priority.INITIALIZATION,
    global_step=global_step,
) as stage:
    runner_work()
    stage.complete(residency_receipt)
```

The context validates state and exact devices before yielding. On clean exit it
requires `complete()`, validates the receipt, calls
`notify_release_gpus()`, and returns to `INACTIVE`.

`policy_sync` uses `Priority.GENERATION`; initialization and evaluation use
`Priority.INITIALIZATION`; actor training uses `Priority.ACTOR_TRAINING`.

### 7.3 Policy-sync context

Policy sync needs both scheduler ownership and the coordinator lease:

```text
request fixed policy_sync
begin coordinator policy-sync lease(V)
run actor/rollout collective
validate all applied V and offloaded rollout state
end exact lease
release fixed policy_sync
```

End the lease before notifying scheduler release. If collective mutation fails,
the lease cleanup still runs, but allocation release requires positive
residency evidence.

### 7.4 Collection session

Add one owner object, for example `ElasticCollectionSession`, containing:

- `ElasticCollectionContext`;
- explicit assignments;
- `ElasticProgressTracker`;
- expected policy version and target;
- canonical bundle mapping;
- observed completed and released ranks;
- final environment metrics by rank;
- batch receipt; and
- failure/closed state.

Only the runtime creates this object. Repeated begin, seal, or close calls fail
unless explicitly documented as idempotent cleanup.

### 7.5 Scheduler bridge

Keep all sync/async conversion in the runtime module. A helper may `ray.get()`
Ray object references and run controller coroutines, but it must:

- reject invocation from an already-running event loop if it cannot safely
  bridge it;
- never leave an unobserved task after timeout;
- preserve Ray task causes; and
- use the configured operation timeout for runner-owned waits.

Do not add an independent background event loop unless tests cover its thread,
shutdown, and exception propagation lifecycle.

## 8. Fixed initialization

### 8.1 Ordering

Enabled `init_workers()` becomes:

```text
acquire initialization
rollout.init_worker()
env.init_worker()
CPU reward init if present
wait rollout and env
actor.init_worker()
optional actor checkpoint restore
verify actor, rollout, and train/eval environment offload baseline
release initialization
```

Retain the legacy initialization order to control peak memory. T6 already
requires initialization offload, rollout offload, actor offload, and evaluation
offload when evaluation exists.

### 8.2 Initialization receipt

Add public worker/group inspection that returns only framework-neutral booleans
and versions needed for a typed receipt. Do not infer offload from private flags
in the runner. The receipt covers every GPU-bearing worker in the registered
initialization union.

### 8.3 Resume

Load the actor checkpoint while initialization ownership is held. Set
`global_step` only after successful load and validate it as a non-negative
policy version. Rollout weights are synchronized later under `policy_sync`; do
not assume checkpoint load updated rollout replicas.

## 9. Fixed policy synchronization

### 9.1 All-rank collective remains authoritative

Use the existing `actor.sync_model_to_rollout()` and
`rollout.sync_model_from_actor()` collective pair. Selective synchronization
on expansion remains deferred.

### 9.2 Version receipt

Extend the rollout sync result or add a post-sync query returning each rank's
applied version and residency status. Validate:

- all canonical rollout ranks responded;
- all applied exactly `expected_policy_version`;
- no rank retained an elastic active/paused lifecycle;
- all rollout models and CUDA graphs are released before fixed release; and
- actor weights/optimizer satisfy their configured post-sync offload contract.

### 9.3 Evaluation synchronization

After successful training increments `global_step`, `_maybe_eval_and_checkpoint`
must use the same fixed policy-sync transaction before evaluation so evaluation
uses the new actor version.

## 10. Elastic collection setup

### 10.1 Context and assignments

Build:

```python
ElasticCollectionContext(
    lifecycle_generation=next_lifecycle,
    policy_version=global_step,
    dp_ranks=placement_plan.canonical_dp_ranks,
)
```

Compute and store explicit assignments for the same ranks. Validate that their
sum matches the actor's expected complete batch contribution.

### 10.2 Bind channels before allocation

Call `controller.configure_collection()` with the runner's existing
`env_channel`, `rollout_channel`, optional reward channel, and actor channel.
The configuration must complete while all ranks are cold or completed and
offloaded.

### 10.3 Initial progress

Create `ElasticProgressTracker` with no worker snapshots. Its first snapshot
contains every rank in `resumable_dp_ranks`, zero completed trajectories, and
no active rank. Publish a core `ProgressReport` with the exact total target.

### 10.4 Actor receiver

Start `actor.recv_rollout_trajectories(actor_channel)` before generation can
complete. The receiver may block while only a subset of ranks is active. It may
not start advantage calculation or mutate policy weights.

### 10.5 Generation request

Request `actor_infer` at `Priority.GENERATION` with `global_step` and the exact
trajectory target. Validate every returned GPU belongs to a canonical bundle
and project only whole bundles to active ranks. Partial activation is accepted;
partial bundles are fatal.

## 11. Collection monitor and progress loop

### 11.1 Observation API

Extend the controller/coordinator runner surface with a single consistent rank
observation that includes:

- paired result if both current run calls have resolved;
- EnvWorker `ElasticRankProgress` when the rank has been prepared;
- paired lifecycle state and residency;
- callback-applied active flag; and
- failure text.

The observation is read-only and must not consume metrics or clear stored
results. Repeated polls return equivalent durable completion data.

### 11.2 Polling behavior

Poll with bounded backoff below the configured operation timeout. Each pass:

1. fail immediately on coordinator or worker failure;
2. validate lifecycle and policy identities;
3. update tracker snapshots only monotonically;
4. publish changed progress; and
5. launch or await completed-rank releases outside any coordinator callback.

Avoid a busy loop when no state changes. Do not use environment tensor metrics
as a completion signal.

### 11.3 Concurrent completed-rank release

Independent completed ranks may be released as one exact sorted batch when
observed in the same monitor pass. Preserve T4 waiter identity and do not issue
overlapping requests for the same rank. Productive siblings remain untouched.

### 11.4 Paused ranks

Scheduler-driven shrink can leave a rank `PAUSED`. Publish its worker progress
as resumable after the callback-applied state is visible. Do not call a release
API for a rank the scheduler has already shrunk. Later expansion is entirely
the scheduler/T5 transaction.

### 11.5 Completion condition

Collection is complete only when:

- tracker completed count equals target;
- completed rank set equals the canonical rank set;
- every paired final result is `COMPLETED`;
- actor receiver completed successfully; and
- every completed active rank's release waiter has returned or the final
  verified release transaction is about to run.

## 12. Batch sealing

### 12.1 Receipt data model

Add an immutable RLinf-owned receipt, for example:

```python
@dataclass(frozen=True, slots=True)
class ElasticBatchReceipt:
    lifecycle_generation: int
    policy_version: int
    contributing_dp_ranks: tuple[int, ...]
    expected_trajectories: int
    received_trajectories: int
    transition_count: int
```

If trajectory objects can expose lifecycle/rank identity, include deterministic
per-rank contribution counts or a digest. Keep tensors in actor-owned batch
state rather than duplicating them into the receipt.

### 12.2 Actor validation

Add a public actor method that validates the already received batch before
advantages:

- batch exists and is not already consumed;
- contribution count matches the session assignment;
- versions tensor exists and every element equals expected version;
- lifecycle/rank metadata matches the session when present;
- no transition identity is duplicated; and
- shapes needed by the selected algorithm are complete.

Validation occurs before any advantage or optimizer mutation.

### 12.3 Seal ordering

After the receipt is created:

1. publish the final completed progress snapshot;
2. release any still-active completed ranks through T4/T5;
3. publish the final inactive snapshot if needed;
4. clear scheduler progress; and
5. close the collection session as inactive.

If batch validation fails, do not train. Safe completed-rank release may still
finish, but preserve the validation error as primary.

## 13. Fixed actor training

### 13.1 Allocation boundary

Compute advantages from the sealed, CPU-owned batch only after collection is
sealed and generation ownership is fully released. Advantage calculation must
not onload actor parameters or use CUDA. Then acquire `actor_train` and validate
the fixed actor GPU union before parameter or optimizer onload.

### 13.2 Advantage and training order

Inside the fixed stage:

```text
validate exact batch receipt
compute advantages and returns
acquire actor_train and validate its exact fixed union
run all-rank FSDP training
wait for optimizer success on every rank
offload parameters, gradients, optimizer, and declared buffers
verify actor residency receipt
release actor_train
increment global_step
```

Keep the existing algorithm-specific actor APIs. If some T6-accepted actor
class cannot supply the seal or residency contract, narrow enabled validation
before implementation rather than silently skipping the barrier.

### 13.3 Version advancement

Do not set `global_step += 1` until training and verified offload succeed. A
failed stage retains policy version V and must not continue to evaluation or a
new collection.

## 14. Fixed evaluation and checkpointing

### 14.1 Evaluation order

When validation is due:

1. perform fixed policy sync for the current post-training version;
2. acquire fixed `evaluation`;
3. launch existing environment and rollout evaluation calls;
4. wait for both and aggregate metrics;
5. verify evaluation environment and rollout offload; and
6. release `evaluation`.

Evaluation does not use the elastic coordinator collection lifecycle.

### 14.2 Checkpoint order

Checkpointing uses actor state after successful training. If saving requires
actor GPU residency, either include it within a documented fixed actor stage
or implement CPU-state saving that needs no GPU allocation. Audit the current
FSDP checkpoint path before choosing; never load actor weights outside a fixed
grant merely because evaluation has completed.

### 14.3 Profiling

Existing profiling calls may onload or invoke CUDA work. In enabled mode, move
profiling start/stop calls inside the allocation of the component they profile,
or reject cross-stage profiling until it has a safe contract. Do not open a
global actor+rollout+environment profiling window while only one fixed or
elastic subset is allocated.

## 15. Failure, cleanup, and retry semantics

### 15.1 Fixed-stage failure

If worker work fails but residency verification proves every mapped worker is
offloaded, release may proceed as cleanup and the original error is re-raised.
If verification fails or is unavailable, retain logical ownership and mark the
runtime failed uncertain.

### 15.2 Collection failure

On progress, result, callback, or actor-receive failure:

- stop issuing new progress and release requests;
- observe coordinator status once for diagnostics;
- release only ranks already proven paused/completed and nonresident through a
  scheduler-committed transaction;
- do not clear progress if active work could still be scheduler-owned;
- preserve all partial snapshots and worker failure states; and
- re-raise the primary error.

There is no automatic continuation onto a new lifecycle after failure.

### 15.3 Timeout

Runner timeouts do not cancel in-flight VLA inference or world-model diffusion.
They stop runner progress and preserve allocation ownership. T5 remains the
authority for non-cancelling worker operation deadlines.

### 15.4 Normal close

Normal `RegisteredRLixPipeline.close()` requires:

- runtime state `INACTIVE`;
- no fixed allocation;
- no active/pending generation rank;
- no policy-sync lease;
- no unresolved rank run reference; and
- cleared progress.

Then unregister before closing the coordinator, as implemented by T6.

### 15.5 Entrypoint ownership

Wrap enabled initialization and run in a primary-error-preserving `try/finally`.
The runner/runtime closes the registered pipeline first; worker-group teardown
continues through existing cluster ownership. Disabled entrypoint behavior is
unchanged.

## 16. Compatibility audit

### 16.1 Original RLix parity preserved

T7 relies on, and does not alter:

- generation waking after any DP rank activates;
- remaining-demand gap-ratio planning;
- shrink-before-expand callbacks outside the scheduler lock;
- callback completion before scheduler commit;
- fixed generation-priority policy sync;
- resize/weight-sync serialization;
- exact composite bundle ownership; and
- legacy flat TP clients unrelated to RLinf.

### 16.2 Intentional RLinf differences remain explicit

- chunk-boundary drain rather than immediate abort;
- same-rank continuation rather than migration;
- rank completion eligibility and exact release;
- all-rank fixed policy sync rather than selective expansion sync; and
- fixed initialization/evaluation unions.

T7 adds no new scheduler deviation.

### 16.3 Standalone compatibility

Capture the current disabled-mode call trace in a regression test. With
`rlix_runtime=None`, assert no changes to worker launch, initialization,
weight-sync cadence, collection, advantage, training, evaluation, checkpoint,
logging, or close order.

## 17. Test plan

### 17.1 Runtime state and fixed stages

Add CPU tests for:

- exact cluster IDs, priorities, global steps, and expected device unions;
- successful initialization acquire/verify/release;
- missing `complete()` receipt;
- wrong returned GPU union;
- nested or overlapping stage rejection;
- worker failure with safe offload cleanup;
- worker failure with uncertain residency retaining ownership; and
- cleanup errors attached to the primary exception.

### 17.2 Policy synchronization

Cover:

- allocation before lease before collective;
- every rank applying the expected version;
- lease end before fixed release;
- wrong/mixed version rejection;
- collective failure and lease cleanup;
- rollout residency failure preventing release; and
- resize waiting while the lease is held.

### 17.3 Collection setup and partial activation

Use fake scheduler/controller/workers to prove:

- configure before progress before request;
- initial cold ranks are resumable without worker onload;
- actor receiver starts before first completion;
- a one-rank initial grant unblocks collection in a multi-rank topology;
- returned partial bundles fail closed; and
- inactive siblings are not called until the scheduler expands them.

### 17.4 Live shrink and resume

Drive one rank through ACTIVE -> PAUSED -> ACTIVE -> COMPLETED while another
rank continues. Assert matching lifecycle/version/transition identity, no
duplicate result consumption, monotonic progress, and exact resumed dispatch.

### 17.5 Completed-rank release

Assert the strict order:

```text
COMPLETED result
progress update/report
await_release_dp_ranks
completed offload callback
scheduler waiter returns
inactive progress report
```

Cover one rank, a same-pass multi-rank batch, a release failure, and a final
completion race with a concurrent scheduler shrink.

### 17.6 Batch sealing

Reject:

- missing rank contribution;
- duplicate contribution or transition identity;
- incomplete trajectory count;
- mixed policy versions;
- wrong lifecycle;
- actor receive failure; and
- a second seal or training attempt.

Prove advantages and training are not invoked in every rejection case.

### 17.7 Actor training

Assert generation is fully released before `actor_train` acquisition, training
uses the exact seal, optimizer failure does not advance the version, verified
offload precedes release, and success advances exactly once.

### 17.8 Evaluation, checkpoint, profiling, and metrics

Cover fixed post-training sync/evaluation order, evaluation offload failure,
checkpoint allocation requirements, supported profiling placement, final
environment metric aggregation across pause/resume calls, and preservation of
existing metric namespaces.

### 17.9 Entrypoint and disabled mode

Cover:

- enabled runtime passed to runner and closed once;
- initialization failure cleanup;
- run failure preserving the primary error;
- unsafe active runtime refusing false close;
- disabled execution making zero RLix calls; and
- the exact legacy call trace remaining unchanged.

### 17.10 Regression suites

Run all focused T1-T6 tests because T7 exercises their public contracts. Run
the complete `rlix-core` suite because the runner uses fixed, elastic,
progress, and release APIs together for the first time.

CPU fake tests are T7 completion evidence. They are not T8 GPU utilization or
real-model recovery acceptance.

## 18. File-by-file edit list and implementation sequence

### 18.1 Required RLinf production files

- `rlinf/runners/embodied_runner.py`
  - branch enabled stages from the unchanged standalone path;
  - integrate initialization, sync, collection, seal, training, evaluation,
    metrics, and cleanup.
- `rlinf/scheduler/rlix/runtime.py`
  - add the runner-facing synchronous state machine, fixed contexts,
    collection session, scheduler bridge, and safe close preconditions.
- `rlinf/scheduler/rlix/controller.py`
  - add durable runner-facing rank observation/progress helpers and sync
    version/residency access as needed.
- `rlinf/scheduler/rlix/coordinator.py`
  - expose read-only atomic rank observations if controller-only composition
    cannot guarantee a consistent view; add no scheduling decisions.
- `rlinf/scheduler/rlix/protocol.py`
  - add stage, batch, version, residency, and observation receipts that cross
    the driver/worker boundary.
- `rlinf/scheduler/rlix/progress.py`
  - keep deterministic snapshots and add only session helpers needed by the
    runner monitor.
- `rlinf/scheduler/rlix/__init__.py`
  - export new public types lazily without making disabled mode depend on Ray
    or `rlix-core`.
- `rlinf/workers/actor/fsdp_actor_worker.py`
  - add batch sealing/version validation and verified post-training residency
    receipt for the supported actor.
- `rlinf/workers/rollout/hf/huggingface_worker.py`
  - return/query applied sync version and verified post-sync/evaluation
    residency without weakening T2 mutation guards.
- `rlinf/workers/env/env_worker.py`
  - expose public fixed-stage residency inspection if existing environment
    verification cannot produce a driver-safe receipt.
- `examples/embodiment/train_embodied_agent.py`
  - make runtime ownership and normal/exceptional close explicit.
- `rlinf/scheduler/rlix/validation.py`
  - narrow enabled actor/checkpoint/profiling capabilities only where the T7
    audit finds no safe implementation contract.

Do not edit every conditional file preemptively. Prefer existing public worker
methods and add the smallest receipt/inspection surface that proves release
safety.

### 18.2 Required tests

- Add `tests/unit_tests/test_rlix_runner_runtime.py` for stage/session behavior.
- Add `tests/unit_tests/test_rlix_embodied_runner.py` for enabled and disabled
  runner ordering with fakes.
- Extend `tests/unit_tests/test_rlix_resize_coordinator.py` for atomic rank
  observation only if coordinator production code changes.
- Extend `tests/unit_tests/test_rlix_progress.py` for session snapshots.
- Extend `tests/unit_tests/test_rlix_entrypoint.py` for runtime close ownership.
- Extend focused actor/elastic worker tests for seal and residency receipts.

### 18.3 Explicitly excluded production files

- `rlix-core` scheduler/planner/validation/tracer code, unless an independently
  documented correctness bug is discovered;
- async runner and pipeline runner files;
- placement conversion and bundle construction except for discovered T6 bugs;
- Wan/OpenSora snapshot schemas except for discovered T1 regressions; and
- future utilization architecture code.

### 18.4 Suggested implementation sequence

1. Freeze disabled-mode call-order tests.
2. Define immutable stage, observation, batch, and residency receipts.
3. Implement the synchronous runtime bridge and explicit stage state machine.
4. Implement fixed-stage acquire/verify/release with failure tests.
5. Put enabled initialization under the fixed union and verify baseline
   offload.
6. Integrate the coordinator policy-sync lease and all-rank version receipt.
7. Add collection session construction, assignments, channel binding, and
   initial progress.
8. Add partial generation activation and the actor receiver barrier.
9. Add atomic rank observation and the progress monitor.
10. Add completed-rank report/release/report ordering.
11. Add actor batch sealing and all rejection tests.
12. Compute advantages from the sealed CPU batch, then put training under
    fixed actor ownership and add actor residency verification.
13. Integrate fixed evaluation, checkpoint, profiling, and metric behavior.
14. Add entrypoint close ownership and exceptional cleanup.
15. Run focused T1-T7 tests, then the full core and style suites.
16. Update canonical status documents and record exact evidence.

Keep runtime protocol, fixed stages, collection monitoring, and batch sealing
as reviewable units even if delivered in one pull request.

## 19. Verification commands and definition of done

From `/root/_VLAMP/RLinf`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_rlix_runner_runtime.py \
  tests/unit_tests/test_rlix_embodied_runner.py \
  tests/unit_tests/test_rlix_runtime.py \
  tests/unit_tests/test_rlix_entrypoint.py \
  tests/unit_tests/test_rlix_resize_coordinator.py \
  tests/unit_tests/test_rlix_progress.py \
  tests/unit_tests/test_elastic_env_rollout.py \
  tests/unit_tests/test_world_model_resume.py \
  tests/unit_tests/test_rlix_placement.py \
  tests/unit_tests/test_rlix_config.py
```

Run Ruff and compilation on every changed production and test file:

```bash
/root/.venv/bin/ruff check <changed-files>
/root/.venv/bin/ruff format --check <changed-files>
/root/.venv/bin/python -m compileall -q \
  rlinf/runners/embodied_runner.py \
  rlinf/scheduler/rlix \
  tests/unit_tests
```

From `/root/_VLAMP`:

```bash
export PYTHONPATH="$PWD/rlix-core/src${PYTHONPATH:+:$PYTHONPATH}"
~/.venv/bin/python -m pytest -q rlix-core/tests
~/.venv/bin/ruff check rlix-core/src rlix-core/tests
~/.venv/bin/ruff format --check rlix-core/src rlix-core/tests
~/.venv/bin/python -m compileall -q rlix-core/src rlix-core/tests
```

Also compose the RLix Wan example, run pure enabled/disabled validation, and
repeat the optional import boundary check with `rlix_core` unavailable for
disabled mode. Preserve the four pre-existing core format findings documented
by T3-T6 unless separately fixed.

T7 is done only when:

- the production synchronous embodied entrypoint uses the registered runtime;
- no enabled worker onloads or uses a GPU before the correct grant;
- every fixed release follows a typed verified offload receipt;
- policy sync holds both fixed allocation and the exact T5 lease;
- initial progress precedes generation request;
- partial initial activation works and later live resize remains callback-owned;
- completed ranks publish durable progress before exact release;
- callback/offload failure prevents logical release and training;
- the actor seals exactly one complete lifecycle/version-consistent batch;
- advantages and training cannot run on partial or mixed-version data;
- actor training is fixed, all-rank, and version advances only on success;
- evaluation is fixed and uses the post-training synchronized version;
- normal shutdown clears progress, unregisters, and closes the coordinator;
- exceptional cleanup preserves the primary error and never falsely releases
  uncertain residency;
- disabled standalone ordering is unchanged and has no RLix dependency; and
- focused T1-T7, complete core, style, compilation, configuration, and
  dependency-boundary checks pass with exact results recorded.

T7 completion does not claim cross-pipeline GPU reuse or Wan/OpenSora recovery
under real scheduling pressure. Those remain the T8 definition of done.
