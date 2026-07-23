# Task 8 Detailed Implementation Plan: Verification and GPU Acceptance

## 1. Status and source of truth

Status: in progress. T0-T7 are implemented and have focused CPU coverage. The
first T8 implementation slice now freezes the dependency-light run, event,
transition, allocation, GPU-sample, and utilization schemas and provides CPU
validators/analyzers for manifest normalization, tolerance comparison,
exclusive ownership, direct-sample integration, and paired throughput gates.
T8 is not complete: the local-Ray process-isolation proof now starts two
independent clients against one detached `rlix-core` scheduler, but no recorded
run has yet driven real Wan/OpenSora workers through composite-bundle transfer,
resumed the interrupted shard, and demonstrated a utilization improvement.

Implementation progress (2026-07-23):

- `tests/e2e_tests/embodied/task8_acceptance_support.py` contains the frozen
  schema version, fail-closed event sink, deterministic CPU tensor manifests,
  scheduler ownership slices, direct GPU sample integration, and paired
  static/dynamic throughput evaluation.
- `tests/e2e_tests/embodied/task8_acceptance_analysis.py` validates sealed
  single-version batches, lifecycle-normalized reference equivalence,
  continuation-state size ceilings, and the exact event/ownership ordering of
  one A-to-B-to-A bundle transfer while a sibling rank continues.
- `tests/unit_tests/test_rlix_gpu_acceptance.py` currently provides 34 passing
  CPU tests for schema/topology validation, central and producer ordering,
  duplicate terminal rejection, typed manifest comparison, snapshot tensor
  byte accounting and ceilings, transition/batch completeness, ownership
  exclusivity and transfer ordering, sampler degradation, and the default
  five-repetition material-improvement gate.
- The pure analyzer now generates deterministic JSON-safe and Markdown report
  views only from validated reference, transfer, snapshot, utilization, and
  raw-artifact inputs. Atomic raw-artifact layout, writing, and completion
  sealing are implemented; full artifact-set orchestration remains pending.
- Ruff lint, Ruff format checking, focused pytest, and compilation pass for
  the Task 8 helper and unit-test files. Production scheduling behavior is
  unchanged; `rlix-core` tracing now emits a distinct post-commit marker.
- `tests/unit_tests/test_rlix_multipipeline.py` now composes two production
  `RegisteredRLixPipeline` runtimes, one production `SchedulerImpl`, and two
  production `RLixResizeCoordinator` transactions on a persistent actor-like
  event loop. Its passing happy-path test activates A on two ranks, transfers
  one exact bundle to B, observes A sibling completion, releases B, resumes A
  once with the stored same-rank token, completes both pipelines, and seals two
  complete single-version actor batches. Its fail-closed test injects a peer
  safe-point-token mismatch and a one-peer pause-offload failure; both prove
  scheduler ownership does not commit to B and no batch is sealed.
- The same multipipeline file now includes an opt-in local-Ray test backed by
  `tests/e2e_tests/embodied/task8_local_ray_client.py`. Two independent OS
  clients resolve the same detached control-plane and scheduler actor IDs,
  register distinct pipeline IDs/namespaces and collision-free role-owned
  worker/channel/event names with intentionally overlapping candidate GPU
  mappings, and prove client B retains the same core after client A exits.
- `tests/e2e_tests/embodied/task8_acceptance_workers.py` provides
  acceptance-only `RecordingEnvWorker`, `RecordingMultiStepRolloutWorker`, and
  `RecordingEmbodiedRunner` subclasses. Their mixins bracket real chunk/policy,
  snapshot, offload/onload, restore, policy-sync, reward/collection, seal, and
  training boundaries while delegating computation and model movement. Five
  CPU tests prove identical outputs and base call order, no extra RNG
  consumption, correct policy-version advance, exact drain/barrier/resume
  ordering, sealed CPU batch capture, normalized snapshot/result evidence, and
  fail-closed observer errors.
- `AcceptanceEventProducer` now enriches those hooks with current pipeline,
  lifecycle, policy, transition, rank, and GPU-bundle identity before submitting
  them to the central sink. It advances producer sequence only after the sink
  accepts the event.
- The frozen manifest now requires GRPO and at least two training iterations.
  Linked-iteration analysis rejects incomplete rewards/batches, non-GRPO
  advantages, incomplete actor updates, skipped policy versions, and failure to
  synchronize the produced policy into the next collection.
- The scheduler now emits `Commit C<N>` only after successful state mutation,
  with exact per-rank bundle mappings for shrink, remove, allocation, and
  expansion. Acceptance-side normalization rejects pre-callback `Exec C<N>`
  markers, unknown ranks, and non-canonical bundles; correlation proves the
  A-to-B-to-A commits occur after A snapshot/offload, bracket B useful work,
  and precede A onload. A failed callback emits no commit marker.
- Commit markers now carry a deterministic `commit_json` trace argument. The
  acceptance artifact adapter invokes an operator-supplied Perfetto trace
  processor, parses only post-commit markers, and fails closed on missing
  tooling, empty traces, malformed CSV/JSON, clock/cycle regression, or command
  failure. Its CPU tests use an injected runner because trace-processor tooling
  is not installed in the shared environment.
- The artifact adapter now creates the canonical per-run directory layout,
  refuses non-empty prior runs and overwrites, atomically writes normalized
  JSON/JSONL/CSV, and seals closed raw inputs with size and SHA-256 evidence.
  Read-side verification rejects missing, unsafe, duplicated, or modified
  artifacts before analysis.
- The focused Task 8 command passes 42 tests with the local-Ray test skipped by
  default and 43 tests when `RLINF_RUN_LOCAL_RAY_TEST=1` enables it.
- The focused T1-T8 CPU regression available at this stage passes 284 tests
  with two optional skips across snapshot, safe-point, progress,
  coordinator, placement, configuration, runtime, runner, entrypoint, Task 8
  analysis, and standalone compatibility coverage.
- `../rlix-core/tests/test_t8_original_rlix_parity.py` adds six composed
  production-scheduler regressions covering an explicit two-pipeline
  A-to-B-to-A bundle cycle, any-rank wakeup, remaining-demand rebalance,
  shrink-before-expand, callbacks outside the scheduler lock,
  callback-before-commit, failure-before-commit, legacy TP slicing and partial
  activation, fixed lifecycle isolation, and policy-sync/resize
  serialization. The focused file passes 6 tests; the complete core suite
  passes 113 tests with one existing optional skip, and core lint and
  compilation pass.
- The complete core format check still identifies four pre-existing files
  (`client.py`, `test_gap_ratio.py`, `test_scheduling_cycle.py`, and
  `test_tracer.py`) that would be reformatted. The new T8 parity file passes
  its focused Ruff format check; unrelated formatting was not changed.
- Full artifact-set orchestration/report emission, real trace-processor
  execution, the two-driver acceptance orchestrator, driver/control event
  wiring, and real accelerator acceptance remain pending.

The existing real-model evidence is useful but does not satisfy T8:

- the T1 Wan tests prove same-environment snapshot/offload/restore equivalence
  in disaggregated and collocated modes;
- the T6 smoke proves placement, registration, and cold real-model loading;
- the T7 smoke proves one registered runner can initialize, synchronize,
  collect, seal, train, release, and close; and
- the VLA/world-model benchmarks measure resident and collocated model timing,
  but do not exercise Ray, RLix scheduling, or competing pipelines.

This document expands T8, "Verification and GPU acceptance," from these
sources, in descending order of authority:

1. `../VLA_COMPATIBILITY_DESIGN.md`, especially architecture, scheduling,
   safe-point interruption, resize lifecycle, parity, and definition of done;
2. `../VLA_COMPATIBILITY_IMPLEMENTATION_PLAN.md`, especially T8 and the
   whole-project definition of done;
3. `RLIX_ELASTIC_VLA_ROLLOUT_IMPLEMENTATION_PLAN.md`, especially the T8 GPU
   matrix and original-RLix parity assertions;
4. the completed detailed T1-T7 implementation plans; and
5. `../VLA_WM_UTILIZATION_FUTURE_DESIGN.md`, only for measurement guidance.

If this plan conflicts with `../VLA_COMPATIBILITY_DESIGN.md`, the architecture
document wins. A discovered correctness or ownership-boundary defect may be
fixed in production as part of T8. A new scheduling policy, unequal service
pool, dynamic VLA batching, model-parallel execution mode, or other architecture
change requires a design update before implementation.

The future-utilization note does not broaden T8. T8 accepts the approved
equal-world, paired-rank, atomic-bundle architecture. It must not claim to solve
the intra-rollout alternation in which the VLA waits for the world model and
the world model waits for the VLA.

Implementation checklist:

- [x] Freeze a machine-readable T8 evidence schema and CPU-test its validators.
- [x] Add a composed two-pipeline CPU integration suite using the production
  scheduler, runtime, and coordinator transaction boundaries.
- [x] Add explicit original-RLix parity regressions for every preserved
  behavior named in the architecture.
- [ ] Build an acceptance orchestrator that starts two independent OS driver
  processes against one existing Ray cluster.
- [x] Derive and CPU-test collision-free role-specific RLinf worker, channel,
  and event identities while retaining the one shared core namespace.
- [ ] Apply those identities to the full driver, output, and acceptance-control
  surface.
- [x] Prove two subprocess clients resolve the same detached control-plane and
  scheduler actor IDs and receive distinct pipeline IDs/namespaces.
- [x] Add transparent acceptance-only environment/rollout subclasses and prove
  fake-worker output, call-order, and RNG equivalence.
- [x] Complete acceptance-only worker/actor wiring around drain observation,
  barrier consumption, resumed dispatch, sealed CPU batch-manifest capture,
  GRPO advantages, and actor training.
- [x] Correlate core scheduler release/allocation commits and exact bundle reuse
  with the enriched worker event stream.
- [ ] Run at least two linked real GRPO iterations per acceptance pipeline:
  rollout/environment reward production, complete batch seal, GRPO advantage
  computation, actor update, policy-version increment, synchronization, and
  collection with the updated policy.
- [ ] Record an uninterrupted reference manifest before running preemption.
- [ ] Run the forced two-pipeline Wan recovery scenario in disaggregated mode.
- [ ] Run the same Wan recovery scenario in collocated mode.
- [ ] Run the forced two-pipeline OpenSora recovery scenario in disaggregated
  mode, and collocated mode when its real checkpoint fits the supported host.
- [ ] Compare transition, trajectory, version, reward, and final-conditioning
  manifests with the uninterrupted reference.
- [ ] Measure safe-point latency, snapshot size, memory reclaimed, resize cost,
  direct GPU utilization, makespan, and throughput per physical GPU.
- [ ] Run a matched static-partition control and prove a material utilization
  and throughput-per-GPU improvement with the dynamic elastic schedule.
- [ ] Produce self-contained JSON/JSONL/CSV/Perfetto/operator-report artifacts.
- [ ] Run the complete T1-T8 regression, lint, format, compilation, and shell
  checks.
- [ ] Mark T8 complete in the canonical documents only after both Wan and
  OpenSora satisfy every hard gate with recorded hardware evidence.

Do not check an item from a dry run, a fake-model run, a single driver with two
pipeline objects, inferred GPU idleness, or a scheduler-only trace. T8 requires
real models, real worker processes, real scheduler callbacks, and direct GPU
measurements.

## 2. Required outcome

After T8, the repository contains a repeatable acceptance workflow proving the
complete ownership transfer:

```text
driver A and driver B are separate OS processes
    -> both connect to the same detached ControlPlane actor
    -> both receive the same Scheduler actor
    -> each registers a distinct pipeline and RLinf coordinator namespace

pipeline A owns at least two canonical actor_infer bundles
    -> A rank r enters a real Wan/OpenSora diffusion chunk
    -> pipeline B creates competing demand
    -> core plans A rank r shrink
    -> T5 requests drain while the chunk is in progress
    -> the real chunk commits before the drain boundary is acknowledged
    -> T2 retains the next observation and T1 snapshots continuation to CPU
    -> rollout and environment residency are verified offloaded
    -> T5 callback returns
    -> core commits release of the complete canonical bundle
    -> pipeline B acquires and performs real work on that exact bundle
    -> B releases it safely
    -> A reacquires the same rank and exact bundle
    -> A restores and dispatches the retained transition exactly once
    -> both pipelines seal complete, single-version batches
    -> fixed actor training consumes only those sealed batches
```

The interrupted pipeline A result must be equivalent to an uninterrupted
reference with the same checkpoints, config, reset IDs, seeds, initial policy
version, rank assignments, and deterministic inference settings.

Task 8 accepts a real RL training pipeline, not only a resource-lifecycle
pipeline. The primary Wan and OpenSora acceptance configurations are frozen to
`algorithm.adv_type: grpo` and must execute at least two linked iterations:

```text
rollout + environment reward production
    -> complete single-version batch seal
    -> GRPO advantage computation
    -> actor optimization completes and produces policy version N+1
    -> policy N+1 synchronizes to rollout workers
    -> the next collection and sealed batch use policy N+1
```

A scheduling-only run, a fake actor update, one terminal training iteration,
or an update whose new policy is never collected does not satisfy Task 8.

The utilization result is a separate, matched experiment. It compares dynamic
two-rank sharing with a static one-bundle-per-pipeline partition on the same
physical GPU set and workload arrival pattern. Dynamic sharing must improve
the primary throughput-per-physical-GPU metric and reduce idle GPU time. The
forced preemption run proves correctness; it is not assumed by itself to
outperform an idealized no-resize execution because snapshot and residency
movement have real costs.

The defining correctness gates are:

```text
no logical release before real chunk commit + snapshot + verified offload
no overlapping logical or physical use of one bundle by two pipelines
one A -> B -> A transfer of the exact canonical bundle
one resumed dispatch of the exact retained next transition
no missing, skipped, duplicated, stale, wrong-rank, or wrong-version transition
no advantage calculation or actor training before a complete batch seal
GRPO reward coverage equals the complete trajectory count
each actor update produces exactly the next policy version
the next collection uses the policy produced by the preceding GRPO update
no callback failure, worker failure, or uncertain residency reported as success
```

The defining measurement gate is:

```text
same hardware + checkpoints + workload + arrival schedule
dynamic elastic sharing
    has higher completed useful work / (physical actor_infer GPUs * wall time)
    than static partition control
and direct time-weighted GPU busy samples corroborate reduced idleness
```

T8 completion supplies evidence for enabling elastic collection by default for
the supported configuration. It does not silently flip `rlix.enabled` in
existing examples. Any default change is a separate explicit configuration and
documentation decision after acceptance.

## 3. Scope boundaries

### 3.1 Supported by T8

- One homogeneous single-node NVIDIA CUDA Ray cluster.
- Two independent synchronous RLinf embodied driver processes.
- One detached `rlix-core` control plane, scheduler, and resource manager.
- Two distinct admitted pipeline IDs and pipeline-owned coordinators.
- Hugging Face complete-model VLA rollout replicas.
- Wan and OpenSora world-model training environments.
- FSDP actor groups used only through fixed stage ownership.
- Equal contiguous rollout/environment worlds with at least two composite
  ranks per competing topology.
- Disaggregated width-two rollout/environment bundles.
- Collocated width-one rollout/environment bundles.
- Scheduler-driven active shrink, safe-point pause, exact-rank resume, and
  completed-rank release.
- Real fixed initialization, policy synchronization, collection, batch seal,
  actor training, and cleanup.
- An uninterrupted reference, a forced preemption/recovery case, and a matched
  static-partition utilization control.
- Direct NVIDIA utilization and memory sampling plus scheduler Perfetto traces.
- CPU-only composed transaction, parity, schema, analysis, and failure tests.

### 3.2 Explicitly not implemented by T8

- Multi-node or heterogeneous accelerator acceptance.
- Tensor parallel VLA inference, model/pipeline parallel world models, or
  application rollout pipelining.
- Unequal VLA/environment service pools or many-to-many routing.
- Dynamic VLA microbatching or asynchronous actor-learner policy staleness.
- Cross-rank continuation migration or rank renumbering.
- Forced cancellation inside policy inference or diffusion.
- Elastic actor training or elastic evaluation.
- Selective rollout weight synchronization on expansion.
- Process-restart recovery of a partial batch.
- Solving VLA/world-model phase-local idleness inside one paired shard.
- Treating NVIDIA utilization percentage alone as useful throughput.
- Generalizing one-step Wan/OpenSora timing to full-quality diffusion settings.

### 3.3 Fail-closed acceptance boundary

The acceptance command fails, rather than skips or weakens assertions, when:

- either requested checkpoint is unavailable or incomplete;
- fewer than the configured physical GPUs are visible;
- NVML/direct utilization sampling or scheduler tracing cannot start;
- either driver resolves a different control-plane or scheduler actor;
- RLinf worker/channel names collide across drivers;
- pipeline IDs or pipeline namespaces collide;
- topology is not exactly the expected canonical rank-to-bundle mapping;
- the orchestrator cannot prove that drain was requested during a real chunk;
- a release, reuse, or resume event lacks exact pipeline/rank/bundle identity;
- event streams are incomplete, duplicated, or not monotonically ordered;
- reference manifests cannot be compared;
- either pipeline fails to seal or train a complete batch;
- post-release or final residency cannot be verified; or
- required utilization repetitions do not meet the declared threshold.

OpenSora may be unavailable on a particular developer host, but that produces
an explicit unexecuted result and leaves the OpenSora gate and T8 completion
pending. It is not a passing skip.

## 4. Relationship to T0-T7

### 4.1 T0 supplies the control experiment and fixed safety

T8 preserves fixed allocation for initialization, synchronization, actor
training, evaluation, and static-partition controls. It verifies fixed requests
remain atomic and are never routed through the resize callback.

### 4.2 T1 supplies the reference-equivalence state

T8 treats the T1 snapshot schema as authoritative. It records encoded snapshot
size and compares the resumed next chunk and final conditioning manifest with
an uninterrupted run. It does not add checkpoint persistence or restore to a
different actor.

### 4.3 T2 supplies the real safe point

T8 proves on real diffusion that the drain request can arrive after chunk
start, but snapshot/offload and callback success occur only after chunk commit.
It verifies the retained next bootstrap is dispatched exactly once on resume.

### 4.4 T3 supplies exact composite ownership and original parity

T8 consumes the canonical mapping in every event and report. A width-two rank
transfers as a whole, a width-one collocated rank remains one logical bundle,
and legacy flat TP registration remains unchanged in parity tests.

### 4.5 T4 supplies eligibility and exact release

T8 verifies paused ranks are resumable, completed ranks are not re-expanded in
the same lifecycle, productive siblings continue, and exact release waiters
return only for the ranks committed by their request.

### 4.6 T5 supplies callback-before-commit physical safety

T8 uses the production named coordinator and worker actor handles. It proves
offload/residency receipts precede scheduler commit and that policy sync does
not overlap resize. No acceptance harness may call worker offload/onload
directly to simulate a successful transfer.

### 4.7 T6 supplies topology, names, and registration

T8 derives candidate mappings from resolved placement, not test-authored
scheduler payloads. It adds unique per-driver RLinf group/channel identities
where needed, while both pipelines intentionally register overlapping global
candidate GPU IDs through the same detached core.

### 4.8 T7 supplies complete runner stages and batch sealing

T8 starts from the production enabled runner and its registered runtime. It may
subclass workers for acceptance-only event recording, but cannot bypass fixed
stages, collection monitoring, exact release, batch sealing, training, or safe
runtime close.

## 5. Current evidence and concrete gaps

### 5.1 Single-pipeline evidence is not multipipeline evidence

`task7_real_runner_smoke.py` creates one `Cluster`, one worker topology, one
registered runtime, and one runner in one driver. It verifies offloaded final
state but cannot expose cross-driver naming, detached-actor identity, scheduler
competition, or ownership transfer.

### 5.2 Default RLinf group names collide

Current examples use `ActorGroup`, `RolloutGroup`, and `EnvGroup`. Two drivers
in the same RLinf Ray namespace can therefore collide even though their core
pipeline namespaces are distinct. T8 must assign a run-ID/role prefix to all
driver-owned named actors and channels before worker launch and test that no
unprefixed global name remains.

This is acceptance isolation, not a request to give each pipeline a private
core scheduler. The core control-plane, scheduler, and resource-manager names
must remain shared singletons in `rlix-core`'s namespace.

### 5.3 The current smoke does not expose a deterministic preemption gate

Starting driver B after an arbitrary sleep cannot prove its demand arrived
during a real diffusion call. T8 needs acceptance-only chunk start/commit
events and a driver barrier so B is released precisely after A reports the
chosen rank inside `chunk_step()`.

### 5.4 Existing status receipts lack a complete cross-process timeline

T1-T7 receipts prove state at API boundaries but do not retain timestamps for
chunk start, drain visibility, snapshot completion, model moves, scheduler
commit, competing-pipeline acquisition, and resumed dispatch. T8 needs a
separate append-only evidence stream; it must not add scheduling decisions to
worker status objects.

### 5.5 Scheduler tracing is necessary but not sufficient

The core Perfetto trace records allocation slices, queue depth, active GPU
count, execution plans, and release markers. It proves logical ownership, but
not model non-residency or useful CUDA work. T8 correlates it with RLinf worker
events and direct NVIDIA samples.

### 5.6 Reference manifests are not emitted today

The T7 result records batch counts and final residency, not transition-level
identities, rewards, observations, conditioning state, or tensor fingerprints.
T8 needs an acceptance-only normalized manifest captured before actor training
mutates the policy.

### 5.7 Current benchmarks infer some idle shares

The existing VLA/world-model reports intentionally infer phase-local idle share
from sequential latency. T8 must collect direct time-series samples from the
actual two-driver run and must keep the deferred intra-shard utilization issue
separate from cross-pipeline reuse.

### 5.8 OpenSora has no recorded real acceptance result

OpenSora CPU fake snapshot tests exist, but the workspace previously lacked a
real checkpoint. T8 must provide a checkpoint-parameterized harness and retain
pending status until an actual run is recorded.

## 6. Core invariants and design decisions

### 6.1 Driver independence is observable

The two pipelines run in separate OS processes with distinct PIDs and separate
Ray client contexts. Constructing two runtime objects in one Python process is
a useful CPU unit test but is not GPU acceptance.

Each driver records:

- OS PID, hostname, Ray job/worker identity, and RLinf role;
- detached control-plane actor ID;
- scheduler actor ID returned through admission;
- allocated pipeline ID and registered namespace;
- every named worker group and acceptance actor/channel;
- resolved placement payload and canonical bundles; and
- checkpoint/config digests.

The two control-plane actor IDs and scheduler actor IDs must match. Pipeline
IDs, namespaces, worker names, and output directories must differ.

### 6.2 One run ID owns all acceptance artifacts

The operator supplies or receives an immutable `run_id`. Every file, named
acceptance actor, worker-group prefix, event, and subprocess log includes it.
Refuse an existing nonempty output directory unless an explicit resume/analyze
mode validates the same run manifest. Do not mix samples from separate runs.

### 6.3 Candidate overlap is allowed; active overlap is forbidden

Both pipelines may register identical candidate bundle mappings. Registration
does not allocate GPUs. At every timestamp, the correlated scheduler trace
must show at most one active owner for each physical GPU.

Worker evidence must additionally show that A reports verified non-residency
before B's first useful CUDA interval on the transferred bundle. A scheduler
slice alone cannot establish this physical-safety assertion.

### 6.4 At least one sibling continues during transfer

The accepted topology has at least two actor-infer ranks. When rank `r` of A
drains, at least one sibling A rank remains active and commits additional work.
This distinguishes selected-rank elasticity from whole-pipeline serialization.

### 6.5 Preemption is event-driven

The orchestrator blocks B before policy-sync/collection demand. It releases B
only after the acceptance event sink observes `chunk_started` for the selected
A rank and before that rank emits `chunk_committed`.

The hard ordering is:

```text
A.chunk_started(n)
< B.competing_request_enqueued
< A.drain_requested
< A.chunk_committed(n)
< A.snapshot_completed(next=n+1)
< A.rollout_offload_verified and A.environment_offload_verified
< core.release_committed(A, rank r, bundle X)
< core.allocation_committed(B, bundle X)
< B.useful_cuda_work_started(bundle X)
```

Transported events use a central sink timestamp for cross-process ordering and
also carry producer `monotonic_ns`, transition identity, CUDA device, and PID.
Critical CUDA boundaries synchronize the relevant stream before emission.

### 6.6 Resume is the reverse ownership transaction

After B safely releases bundle X, A must reacquire its registered rank `r`, not
a migrated rank. Acceptance requires:

```text
core.release_committed(B, bundle X)
< core.allocation_committed(A, rank r, bundle X)
< A.onload_verified(r)
< A.restore_validated(r, next transition n+1)
< A.resume_bootstrap_dispatched(n+1)
< A.policy_result_received(n+1)
```

The resumed bootstrap dispatch count for that transition must be exactly one.

### 6.7 Correctness manifests are normalized before hashing

Record typed scalar fields directly. For tensors/arrays record shape, dtype,
logical identity, and a deterministic CPU byte hash after canonical contiguous
conversion. Floating values used for semantic comparison retain raw CPU values
or summary arrays so the analyzer can apply the environment-specific
`rtol`/`atol`; do not use hash equality as a substitute for tolerance.

Manifest equality covers:

- ordered `RolloutTransitionIdentity` values;
- one policy version for all transitions;
- per-rank assigned and completed trajectory counts;
- action/log-probability/value/reward/done/truncation sequence metadata;
- observation tensors and next-visual-observation fingerprints;
- reset IDs, episode generations, chunk indices, and metrics;
- Wan `image_queue`, rolling `condition_action`, and deterministic seed
  identity;
- OpenSora latent queue and diffusion generator state; and
- final sealed-batch receipt and final conditioning manifest.

Use deterministic policy inference in reference/recovery comparison. If a
stochastic policy path is explicitly tested, capture and compare its dedicated
generator state rather than relying on process-global RNG order.

### 6.8 Measurement and correctness artifacts are immutable inputs

The analyzer reads files only after every producer has closed them and written
an atomic completion marker. It recomputes all derived metrics from raw events
and samples; a driver-supplied `status: passed` is not authoritative.

### 6.9 No silent evidence degradation

Missing direct utilization samples, missing trace markers, clock regressions,
sample gaps over the configured limit, or unparseable manifests fail the
measurement. The report may label secondary metrics unavailable only when they
are explicitly non-gating in the checked-in schema.

### 6.10 Production edits remain narrow

Prefer acceptance-only subclasses, wrappers, event actors, and read-only
queries under `tests/e2e_tests/embodied/`. Production code changes are allowed
only when the composed tests expose:

- a correctness defect;
- a collision or ownership-boundary bug affecting supported multipipeline use;
- missing read-only evidence that cannot be collected at a public boundary; or
- unsafe cleanup.

Do not add test timing sleeps or acceptance event dependencies to normal worker
execution.

## 7. Acceptance topology and process architecture

### 7.1 Required logical topology

Both drivers register the same candidate actor-infer topology so exact physical
reuse is possible.

Recommended disaggregated six-GPU mapping:

```text
actor ranks:   [0, 1]
rollout ranks: [2, 3]
env ranks:     [4, 5]

actor_infer rank 0 = [2, 4]
actor_infer rank 1 = [3, 5]
```

Recommended collocated four-GPU mapping:

```text
actor ranks:   [0, 1]
rollout ranks: [2, 3]
env ranks:     [2, 3]

actor_infer rank 0 = [2]
actor_infer rank 1 = [3]
```

GPU IDs are configurable, but every recorded run stores the resolved mapping.
Do not assume `CUDA_VISIBLE_DEVICES` text is the core GPU identity; T6 resolved
placement remains authoritative.

### 7.2 Process roles

Use three top-level processes:

```text
task8 orchestrator
  +-- driver A subprocess
  +-- driver B subprocess
  +-- NVML sampler subprocess or dedicated sampler actor
```

The orchestrator owns only barriers, raw artifact collection, timeout policy,
and final analysis. It does not own an RLinf runtime or issue scheduler resize
calls.

Each driver:

1. composes its role-specific config;
2. applies unique group/channel names;
3. resolves and validates placement;
4. launches the production worker groups, using acceptance subclasses only for
   observation/manifest recording;
5. bootstraps through `rlix_core.client.connect()`;
6. initializes and returns to verified inactive/offloaded state;
7. reports ready to the event sink;
8. waits for its role gate;
9. executes one or more production runner steps; and
10. verifies residency, closes its runtime, unregisters, and closes groups.

### 7.3 Shared control-plane preparation

Tracing configuration belongs to the first creation of the detached scheduler.
The launcher must therefore either:

- connect to a known fresh test Ray cluster and create the control plane with
  `RLIX_ENABLE_GPU_TRACING=1` and `RLIX_TRACE_OUTPUT_DIR=<run>/core`; or
- validate that an existing detached scheduler already has compatible tracing
  enabled and a run-owned output location.

Do not let driver A and driver B race to create differently configured core
actors. Record the actor IDs before starting either pipeline.

### 7.4 Collision-free RLinf identity

For role `a` or `b`, derive names such as:

```text
t8_<run_id>_<role>_ActorGroup
t8_<run_id>_<role>_RolloutGroup
t8_<run_id>_<role>_EnvGroup
t8_<run_id>_<role>_<channel-purpose>
```

Add a CPU/local-Ray test that launches the minimal two-driver naming surface
and enumerates expected actors. Core names are identical singletons; every
RLinf-owned name is role-scoped.

### 7.5 Orchestration barriers

Required gates:

1. both drivers initialized and offloaded;
2. B policy sync and then A policy sync completed sequentially while neither
   collection owns generation GPUs;
3. A owns both generation bundles;
4. selected A rank emitted `chunk_started`;
5. B may start its already-synchronized elastic collection demand;
6. at least one exact A bundle transferred to B;
7. B performed useful real model work on the transferred bundle;
8. B releases enough ownership for A to resume; and
9. both pipelines seal, train, return inactive, and close.

Every wait has a positive configured deadline. Timeout never force-cancels
diffusion. On timeout, preserve logs, query statuses, stop new work, and perform
only cleanup whose residency preconditions are known.

## 8. Evidence schema and acceptance observability

### 8.1 Run manifest

Write `run_manifest.json` before starting drivers with:

- schema version and run ID;
- git root commit and RLinf submodule commit;
- dirty-file list, without embedding file contents;
- Python, PyTorch, CUDA, Ray, driver, and GPU metadata;
- checkpoint paths plus stable metadata/file hashes;
- full resolved acceptance parameters;
- expected topology and process roles;
- repetition/warmup counts and thresholds; and
- explicit environment (`wan` or `opensora`) and mode (`disaggregated` or
  `collocated`).

### 8.2 Append-only event record

Use a frozen schema, for example:

```python
AcceptanceEvent(
    schema_version=1,
    run_id=...,
    sink_sequence=...,
    sink_time_ns=...,
    producer_time_ns=...,
    producer_pid=...,
    driver_role=...,
    pipeline_id=...,
    lifecycle_generation=...,
    policy_version=...,
    component=...,
    dp_rank=...,
    event=...,
    transition_identity=...,
    gpu_ids=...,
    details=...,
)
```

The sink assigns `sink_sequence` and `sink_time_ns`; producers cannot choose
them. Validate required identity per event type and reject unknown schema
versions, missing ranks, malformed GPU bundles, duplicate terminal events, and
nonmonotonic producer sequences.

### 8.3 Required event families

Driver/runtime:

- connected, registered, admitted, initialized, stage acquired/released;
- generation requested/granted;
- progress published;
- batch sealed, training started/completed; and
- runtime closed/unregistered.

Environment/rollout:

- policy request started/completed;
- `chunk_started` and `chunk_committed`;
- pending bootstrap retained;
- drain observed and barrier consumed;
- snapshot started/completed with encoded size;
- environment/rollout offload started/verified;
- environment/rollout onload started/verified;
- restore validated/committed;
- resumed bootstrap dispatched; and
- rank completed.

Scheduler correlation:

- request enqueue;
- shrink/expand plan with exact ranks/bundles;
- callback phase start/end/failure;
- release/allocation commit; and
- trace shutdown/final flush.

### 8.4 Acceptance-only worker instrumentation

Prefer test subclasses:

- `RecordingEnvWorker` wraps `env_interact_step()` so it brackets the real
  `chunk_step()` without replacing Wan/OpenSora computation;
- it records the post-commit output/conditioning manifest and delegates all
  lifecycle methods to `EnvWorker`;
- `RecordingMultiStepRolloutWorker` records policy-call and resume-dispatch
  identities while delegating real prediction/offload/onload;
- `RecordingEmbodiedFSDPActor` records the complete CPU batch manifest at the
  seal boundary before training consumes it.

The subclasses must not change return values, seeds, transition IDs, model
movement, channel calls, or scheduler interaction. Add CPU equivalence tests
showing an instrumented fake run has the same call trace and outputs as its
unwrapped worker.

If an essential event is below an overridable public/protected boundary, add
the smallest dependency-free observer callback in production, disabled by
default and covered by a no-observer regression. Do not parse human logs for a
hard ordering assertion.

### 8.5 Scheduler trace analyzer

Add a deterministic analyzer that extracts:

- queue intervals by pipeline/cluster;
- per-GPU allocation slices;
- DP rank and canonical bundle identity;
- shrink-before-expand execution markers;
- release markers; and
- active/idle GPU counter samples.

Assert every slice can be joined to the run's registration manifest and every
transferred GPU has exactly one owner. Keep raw `.perfetto-trace`; emit a
normalized `scheduler_timeline.jsonl` for review and unit testing.

### 8.6 Direct GPU sampler

Sample at a configurable interval no greater than 100 ms for the forced
preemption window and no greater than 250 ms for longer utilization trials.
Record per physical GPU:

- timestamp;
- SM utilization percentage;
- memory-controller utilization;
- total/used memory;
- power draw and clocks when available;
- active compute-process PIDs and per-process memory when available; and
- sampler errors/gaps.

Use NVML directly when available. A checked `nvidia-smi` CSV fallback is
acceptable only if it supplies the gating fields at the required cadence.
Synchronize worker event boundaries before treating a sample interval as useful
CUDA work.

## 9. Uninterrupted reference and equivalence analysis

### 9.1 Reference construction

For each environment/mode configuration, first run pipeline A alone with the
same two ranks, assignments, checkpoints, reset-state IDs, seeds, and policy
version intended for recovery. Disable evaluation/checkpoint side effects and
use one bounded training iteration unless the scenario specifically validates
them.

Record the collection manifest before optimizer mutation. The reference must
finish with a complete batch, expected transition count, verified offloaded
residency, and clean runtime close.

### 9.2 Recovery construction

Restore the same initial policy/checkpoint and repeat A's workload while B
causes the selected A rank to pause and later resume. B uses a distinct reset
seed/workload so cross-pipeline payloads cannot accidentally compare equal.

Do not compare a recovery run after A has already trained against a reference
from the pre-training policy version.

### 9.3 Exact identity assertions

For pipeline A, require equality of:

- canonical rank assignments;
- lifecycle-relative ordered transition identities, normalizing only the new
  run's lifecycle generation if necessary;
- policy version at every transition;
- trajectory and transition counts;
- done/truncation/reset sequence;
- resume transition identity; and
- one-use batch receipt.

There must be no extra inference invocation for the paused boundary and no
missing invocation after resume.

### 9.4 Numeric equivalence

Use checked-in per-environment tolerances no weaker than the existing T1 real
Wan harness unless a documented model kernel requires a justified relaxation.
Compare observations, rewards, metrics, actions, and final conditioning values.
Report max absolute/relative error for every tensor family.

Never relax tolerance automatically after a failure. A tolerance change
requires an evidence-backed plan/document update.

### 9.5 Snapshot size assertion

Record both recursive logical tensor bytes and encoded snapshot bytes. Assert:

- every reachable tensor is CPU resident;
- size is nonzero and bounded by a configuration-derived continuation-state
  ceiling;
- it contains no model-parameter-sized payload; and
- repeated capture at an identical fake state has stable encoded size.

The real report lists current observation, image/latent queue, action
conditioning, partial trajectory, and other dominant fields separately.

## 10. Forced preemption, reuse, and resume scenario

### 10.1 Preparation

1. Start a fresh run-owned event sink and GPU sampler.
2. Initialize both drivers sequentially through fixed ownership so peak model
   loading cannot overlap accidentally.
3. Verify both are inactive and all model/optimizer/CUDA-graph residency is
   offloaded.
4. Synchronize B and then A to the same expected policy version through their
   production fixed `policy_sync` stages, returning both to inactive/offloaded
   state. Pre-synchronizing B prevents its all-rollout fixed union from
   shrinking both A ranks when the intended test is selective elastic
   competition.
5. Allow A to begin collection with both canonical ranks.
6. Verify scheduler trace and callbacks agree that A owns both complete
   bundles.

### 10.2 Trigger

Select rank 0 by default. Hold the acceptance gate until its real world-model
worker emits `chunk_started(transition n)`. Immediately allow already-synced B
to enter `_collect_rlix_rollouts()` (or the equivalent production collection
stage API) without repeating policy sync. Record B's `actor_infer` request
enqueue. After sealing, B still trains through `_train_rlix_batch()` and the
normal fixed actor-training stage; the acceptance driver does not bypass any
T7 allocation or batch barrier.

The configured chunk must be long enough for the drain request to become
observable during diffusion. If the chunk commits before `drain_requested`,
the repetition is invalid and must be rerun with a production-valid longer
diffusion setting or additional environments; do not add a sleep inside
`chunk_step()` and call it real-model timing.

### 10.3 Shrink acceptance

Require:

- only selected A rank(s) receive drain;
- an unselected A sibling commits later work while the selected rank drains;
- the current real chunk commits exactly once;
- no next observation is sent before pause;
- peer safe-point tokens match;
- snapshot and both offloads complete;
- device/process memory drops by the declared reclaimed-memory threshold;
- callback returns after both residency receipts; and
- core release commit follows callback success for the whole bundle.

### 10.4 Competing reuse acceptance

Require B to acquire the exact released GPU set, first through any required
fixed sync ownership and then for real collection work. At least one recorded
B VLA prediction or Wan/OpenSora chunk must execute on that bundle while A's
selected rank remains paused.

Logical trace slices and physically resident process intervals must not overlap
between A and B on the transferred GPUs.

### 10.5 Resume acceptance

Allow B to complete/release enough work for A to become eligible. Require A to
expand the same canonical rank on the same registered bundle, validate the
stored token and policy version, onload both peers, restore, and dispatch the
retained bootstrap once.

A then completes all assigned work. Both A and B must release every elastic
rank, seal a complete batch, train only after the seal, verify fixed offload,
return inactive, unregister, and close.

## 11. Utilization experiment and thresholds

### 11.1 Why this is a separate experiment

Forced preemption includes snapshot/offload/onload overhead and exists to prove
correctness under contention. The utilization objective is measured with a
matched arrival pattern that gives dynamic sharing an opportunity to use GPUs
that a static partition leaves idle.

### 11.2 Static-partition control

Use the same physical actor-infer GPU set, checkpoints, total trajectories,
seeds, and driver arrival times. Restrict A to one disjoint bundle and B to the
other for the full trial. Neither pipeline may borrow the other's bundle.

This is the current fixed-capacity counterfactual. Do not compare against a
different batch size, fewer trajectories, different diffusion steps, or a
single model microbenchmark.

### 11.3 Dynamic-sharing trial

Give both pipelines the two-rank overlapping candidate topology. Start A first
so it can use both bundles while B is not ready; start B at the recorded arrival
offset; allow the scheduler to rebalance and later return capacity as demand
changes.

Run at least one warmup and five measured paired repetitions per required
environment/mode combination. Randomize control/dynamic order or alternate it
to reduce thermal/order bias. Persist each repetition independently.

### 11.4 Primary metric

Compute:

```text
useful_throughput_per_gpu =
    total accepted completed trajectory transitions
    / (number of physical actor_infer GPUs * measured wall seconds)
```

Only transitions present in complete sealed batches count as useful work.
Discarded, duplicate, partial, wrong-version, or failed-run transitions count
as zero, not as throughput.

Default hard threshold:

- median dynamic improvement is at least 5 percent over paired static control;
- at least four of five paired repetitions improve; and
- no dynamic repetition violates a correctness or residency gate.

Keep the threshold configurable in the run manifest but do not lower it after
observing results without updating this plan and explaining why the original
materiality criterion was inappropriate.

### 11.5 Corroborating utilization metrics

Report:

- time-weighted mean and percentile SM utilization per actor-infer GPU;
- physical GPU idle share (`SM utilization <= configured idle threshold`);
- scheduler-owned-but-physically-idle share;
- active bundle count over time;
- completed transitions per scheduler-owned bundle-second;
- total makespan and per-pipeline completion time;
- queue wait, safe-point, snapshot, offload, onload, restore, and resize costs;
- peak and reclaimed memory; and
- power/energy per accepted transition when available.

Require dynamic sharing to reduce aggregate sampled idle GPU-seconds by at
least 5 percent or increase time-weighted mean SM utilization by at least three
percentage points. Treat this as corroboration of the primary throughput gate,
not a replacement for it.

### 11.6 Statistical reporting

Publish every repetition and paired delta, not only averages. Include median,
mean, standard deviation, p50/p95/p99 latency, and a bootstrap confidence
interval for the paired throughput delta when the sample count supports it.
With only the minimum five repetitions, the hard sign/materiality criteria
remain authoritative.

## 12. Wan acceptance matrix

### 12.1 Disaggregated Wan recovery — required

Run the six-GPU topology with two width-two bundles and the real OpenVLA-OFT
and Wan checkpoints. This is the primary safe reuse proof because rollout and
environment residency must both clear different physical GPUs before transfer.

Required outputs:

- uninterrupted A reference;
- forced A/B preemption/reuse/resume run;
- static and dynamic utilization repetitions;
- transition/conditioning equivalence report;
- direct GPU samples and core trace; and
- final all-offloaded residency report.

### 12.2 Collocated Wan recovery — required

Run the width-one topology. The transferred GPU alternates VLA and Wan model
residency within each paired worker lifecycle. Prove neither A model remains
resident when B takes the GPU and that A can reload/restore exactly.

Because current collocated residency movement dominates cycle time, report
throughput and swap cost honestly. Correctness is required; the global T8
utilization threshold is judged on the designated disaggregated utilization
configuration unless the canonical design is changed to require collocated
performance parity.

### 12.3 Diffusion quality settings

Use the bounded low-step setting for routine acceptance, but record it
prominently. Before final production sign-off, run at least one forced recovery
with the intended production diffusion-step count or document why the bounded
setting covers the same safe-point/residency state machine and retain
performance claims only for the measured setting.

## 13. OpenSora acceptance matrix

### 13.1 Checkpoint and dependency preflight

Add explicit environment variables for the VLA checkpoint, OpenSora checkpoint,
and OpenSora source/dependency path. Validate model, VAE, reward-model, dataset,
and statistics entries before Ray worker launch.

The launcher writes `status: unavailable` with the precise missing prerequisite
when preflight fails, exits nonzero for an acceptance invocation, and does not
mark T8 complete.

### 13.2 Disaggregated OpenSora recovery — required

Run the same two-driver/two-rank forced transfer and reference comparison using
the real OpenSora model. In addition to common fields, compare latent queues and
the dedicated diffusion generator state. Perturbing process-global RNG must
not change the resumed result.

### 13.3 Collocated OpenSora recovery — conditional hardware matrix

Provide and validate the configuration. Run it when the real checkpoints fit
the supported GPU/host-memory envelope. If it does not fit, record measured
memory evidence and keep the collocated OpenSora row explicitly unaccepted;
do not substitute a fake model. The architecture's final completion wording
requires Wan and OpenSora two-pipeline recovery, so at least disaggregated
OpenSora remains a hard T8 gate.

### 13.4 OpenSora utilization

The required cross-pipeline utilization threshold must pass for Wan and for
OpenSora unless the canonical architecture is revised to designate only one
performance acceptance environment. Do not extrapolate Wan utilization from
OpenSora or vice versa.

## 14. Original-RLix parity and composed CPU verification

### 14.1 Original parity assertions

Add one focused core suite that proves, against current production code:

```text
generation wakes after any DP rank activates
remaining completed-trajectory demand drives gap-ratio planning
all shrink callbacks finish before any expansion callback starts
callbacks execute outside the scheduler lock
callback success precedes scheduler commit and waiter success
callback failure leaves the previous logical allocation unchanged
resize and fixed policy synchronization serialize correctly
legacy flat device_mapping/tp_size clients retain rank construction and cost
fixed auxiliary clusters never invoke resize callbacks
```

Use both explicit composite bundles and a legacy TP control. Avoid duplicating
T3/T4 assertions mechanically; the T8 suite should compose the preserved
behaviors in a two-pipeline cycle and name the parity contract directly.

### 14.2 Two-pipeline production-boundary CPU test

Construct two registered runtimes sharing one production `SchedulerImpl` with
production `RLixResizeCoordinator` instances and deterministic worker protocol
doubles. Drive:

1. A two-rank activation;
2. B competing demand;
3. selected A rank pause/offload;
4. exact bundle transfer to B;
5. A sibling continuation;
6. B completion/release;
7. A exact-token resume;
8. both complete-batch seals; and
9. cleanup/unregistration.

This test is CPU integration, not GPU acceptance. Its purpose is fast ordering
and failure localization.

### 14.3 Shared detached-control-plane local-Ray test

Start two subprocess clients against one local Ray instance and prove:

- same control-plane and scheduler actor IDs;
- distinct pipeline IDs and coordinator namespaces;
- collision-free RLinf names;
- both registrations visible to the same scheduler behavior; and
- one client disconnect does not stop the detached core or the other pipeline.

Keep this opt-in where restricted sandboxes cannot start Ray, but require it in
the T8 acceptance environment.

### 14.4 Failure matrix

At the composed CPU boundary inject:

- malformed/partial bundle grant;
- drain timeout while chunk work remains non-cancelled;
- mismatched peer safe-point tokens;
- snapshot failure;
- one-peer offload or onload failure;
- residency verification failure;
- callback failure before commit;
- stale or duplicate transition after resume;
- policy-version mismatch;
- incomplete or duplicate actor batch contribution;
- release waiter arriving during an in-flight callback;
- driver B crash while owning the transferred bundle;
- trace/sampler/event-sink failure; and
- cleanup failure after a primary runner error.

Assert ownership is retained or scheduling fails closed as documented, no
partial batch trains, and the primary error remains visible.

Do not deliberately kill a real GPU worker in the primary accelerator matrix
unless a separate recovery design defines safe administrative cleanup. CPU
failure tests cover fail-closed semantics without leaving uncertain real model
residency on a shared host.

## 15. Test plan

### 15.1 Evidence-schema tests

Add `tests/unit_tests/test_rlix_gpu_acceptance.py` covering:

- run/event schema validation;
- sink sequence assignment and producer-order rejection;
- pipeline/rank/bundle identity requirements;
- normalized tensor manifests and tolerance comparison;
- snapshot-size accounting;
- duplicate/missing transition detection;
- incomplete batch rejection;
- trace/event join validation;
- NVML sample gap and PID-attribution checks;
- metric calculation and threshold edges; and
- report generation from checked-in tiny fixtures.

### 15.2 Instrumentation transparency tests

With CPU fake workers, compare instrumented and uninstrumented execution:

- identical method inputs/outputs;
- identical channel and lifecycle call order;
- identical transition IDs and manifests;
- no extra RNG consumption;
- no implicit sleeps; and
- observer failure propagates before an acceptance claim rather than changing
  worker semantics silently.

### 15.3 Multipipeline isolation tests

Cover unique names for two roles, shared core IDs, distinct pipeline identity,
intentional candidate-GPU overlap, active-owner exclusivity, independent
cleanup, and no driver-global `Cluster` singleton leakage across subprocesses.

### 15.4 Reference analyzer tests

Use small deterministic manifests to cover exact equality, permitted numeric
tolerance, wrong policy version, lifecycle normalization, missing/duplicate
transitions, changed final conditioning, and unexpected global-RNG dependence.

### 15.5 Utilization analyzer tests

Use synthetic timelines with known integrals to verify:

- time-weighted utilization across irregular samples;
- idle GPU-seconds;
- scheduler-owned-but-idle intervals;
- throughput per physical GPU;
- paired repetition deltas;
- minimum-improvement thresholds; and
- rejection of gaps, warmup leakage, failed runs, or partial batches.

### 15.6 Wan real matrix

Run reference, forced recovery, and utilization control/dynamic cases in
disaggregated and collocated configurations. Require all correctness gates and
the designated performance gates.

### 15.7 OpenSora real matrix

Run the corresponding disaggregated matrix and the collocated configuration
where supported. A missing checkpoint is reported but not accepted.

### 15.8 Full regressions

Run all focused T1-T7 RLinf tests and the complete `rlix-core` suite. Preserve
disabled standalone behavior and the existing T1/T6/T7 real harnesses as
regressions; do not replace them with the larger T8 command.

## 16. Acceptance artifacts and operator report

### 16.1 Required directory layout

```text
<output>/<run_id>/
  run_manifest.json
  orchestrator.log
  events.jsonl
  drivers/
    a/{stdout.log,stderr.log,result.json,manifest.json}
    b/{stdout.log,stderr.log,result.json,manifest.json}
  core/
    rlix_gpu_timeline_*.perfetto-trace
    scheduler_timeline.jsonl
  gpu/
    samples.csv
    processes.csv
  reference/
    manifest.json
    result.json
  analysis/
    correctness.json
    ownership.json
    latency_summary.csv
    utilization_repetitions.csv
    utilization_summary.json
    REPORT.md
```

Write raw artifacts before derived ones. The report links every hard assertion
to its source file and event sequence.

### 16.2 Required report sections

- status and exact definition of the tested scope;
- commits, dirty state, environment, checkpoints, and hardware;
- resolved topology and shared/different actor identities;
- A-to-B-to-A transfer timeline;
- reference-equivalence table;
- batch completeness and policy-version table;
- memory and latency distributions;
- static versus dynamic utilization/throughput table;
- every failed/invalid repetition and reason;
- limitations, including diffusion steps and deferred intra-shard idleness;
- exact commands; and
- final pass/fail gates for Wan/OpenSora and collocated/disaggregated modes.

### 16.3 No hand-edited results

Generate `REPORT.md`, CSV, and JSON from raw artifacts. Operators may add a
separate notes file, but must not edit generated measurements or status. Do not
commit bulky raw traces/checkpoints by default; commit the implementation plan,
small schemas/fixtures, and a concise recorded report or artifact location as
project policy permits.

## 17. File-by-file edit list

### 17.1 Required new RLinf acceptance files

`tests/e2e_tests/embodied/task8_two_pipeline_acceptance.py`

- top-level subprocess orchestration;
- run manifest, barriers, deadlines, cleanup, and matrix selection;
- fresh/shared detached-core validation; and
- final analyzer invocation.

`tests/e2e_tests/embodied/task8_two_pipeline_driver.py`

- role-specific config composition and collision-free names;
- production worker/runtime/runner launch;
- event-gated initialization and run;
- residency and identity collection; and
- atomic driver result/manifest output.

`tests/e2e_tests/embodied/task8_acceptance_support.py`

- frozen evidence types and validators;
- event sink and acceptance-only worker subclasses;
- manifest normalization/comparison;
- trace and NVML sample analysis; and
- report generation shared by Wan/OpenSora cases.

If this module becomes too large, split dependency-light schema/analysis from
Ray/worker instrumentation. Do not place test-only event orchestration in
`rlinf/` merely for import convenience.

`tests/e2e_tests/embodied/task8_acceptance_artifacts.py`

- canonical isolated run-directory creation;
- atomic immutable JSON, JSONL, and CSV writers;
- raw-artifact completion-marker generation and digest verification; and
- tool-backed Perfetto post-commit extraction without automatic downloads.

`tests/e2e_tests/embodied/run_task8_two_pipeline_acceptance.sh`

- environment/checkpoint preflight;
- run-ID/output handling;
- offline dependency settings;
- trace directory and sampler requirements;
- scenario/mode selection; and
- direct nonzero exit on unavailable or failed hard gates.

`tests/e2e_tests/embodied/task8_wan_disaggregated.yaml`

- two-rank six-GPU recommended mapping and bounded workloads.

`tests/e2e_tests/embodied/task8_wan_collocated.yaml`

- two-rank four-GPU collocated mapping.

`tests/e2e_tests/embodied/task8_opensora_disaggregated.yaml`

- checkpoint-parameterized OpenSora two-rank mapping.

`tests/e2e_tests/embodied/task8_opensora_collocated.yaml`

- validated collocated OpenSora mapping and memory prerequisites.

Prefer a shared schema/default YAML only if Hydra composition keeps every
resolved acceptance value visible in `run_manifest.json`.

### 17.2 Required unit/integration tests

`tests/unit_tests/test_rlix_gpu_acceptance.py`

- evidence, instrumentation transparency, manifest, ownership, utilization,
  threshold, and report tests.

`tests/unit_tests/test_rlix_multipipeline.py`

- production-boundary two-pipeline transaction and failure matrix;
- optional two-subprocess local-Ray detached-core/name-isolation test.

### 17.3 Required core tests

`../rlix-core/tests/test_t8_original_rlix_parity.py`

- one composed explicit-bundle two-pipeline parity cycle;
- one legacy TP control;
- shrink-before-expand, callback-before-commit, failure-before-commit,
  partial wakeup, remaining-demand, and fixed-sync serialization.

Extend existing core tests instead if a new file would duplicate fixtures, but
retain a clearly discoverable T8 parity command and named assertions.

### 17.4 Conditional production files

Only edit these when a failing composed test demonstrates the need:

- `rlinf/scheduler/rlix/runtime.py`: read-only stage/allocation evidence or
  multipipeline-safe cleanup bug;
- `rlinf/scheduler/rlix/controller.py` and `coordinator.py`: read-only timestamp
  or receipt exposure that cannot be obtained at public boundaries;
- `rlinf/workers/env/env_worker.py`: optional disabled-by-default observer hook
  around real chunk commit/snapshot boundaries;
- `rlinf/workers/rollout/hf/huggingface_worker.py`: optional observer hook for
  inference/resume identity;
- `rlinf/workers/actor/fsdp_actor_worker.py`: optional pre-training manifest
  surface at batch seal;
- `rlinf/scheduler/rlix/entrypoint.py`: collision-free names if config-only
  role prefixing is insufficient; and
- `../rlix-core/src/rlix_core/scheduler/tracer.py`: only a missing correctness-
  critical trace identity/flush fix, never a second allocation ledger.

Every conditional production edit needs a focused regression proving normal
disabled/no-observer behavior is unchanged.

### 17.5 Documentation updates after acceptance

- mark T8 status and exact evidence in `../VLA_COMPATIBILITY_DESIGN.md`;
- update T8 and whole-project status in
  `../VLA_COMPATIBILITY_IMPLEMENTATION_PLAN.md`;
- update the canonical table and implementation order in
  `RLIX_ELASTIC_VLA_ROLLOUT_IMPLEMENTATION_PLAN.md`;
- add an operator section to the closest RLinf testing documentation; and
- record unavailable/unsupported OpenSora or collocated rows explicitly rather
  than collapsing the matrix to one overall word.

### 17.6 Explicitly excluded production changes

- new planner policy or priority;
- new allocation ledger in RLinf;
- local RLinf `ControlPlane` construction;
- model parallelism or unequal worker worlds;
- selective weight sync;
- async embodied runner adoption; and
- default-enable changes before every T8 gate passes.

## 18. Suggested implementation sequence

1. Freeze the run/event/manifest schemas and add pure validator tests.
2. Implement manifest normalization, equivalence, ownership-timeline, and
   utilization analysis against tiny checked-in fixtures.
3. Add the composed core T8 parity test with explicit and legacy mappings.
4. Add the two-pipeline production-boundary CPU test and failure matrix.
5. Add the two-subprocess local-Ray shared-core and name-isolation test.
6. Derive collision-free role-specific worker/channel names in the acceptance
   driver; fix production only if config-level isolation is insufficient.
7. Build transparent acceptance worker subclasses and prove fake-worker output
   equivalence.
8. Add the run-owned event sink, barriers, trace preparation, and NVML sampler.
9. Refactor the T7 real smoke construction into reusable test helpers only
   where this avoids copying production configuration logic; keep the T7
   command working unchanged.
10. Implement uninterrupted reference capture and analyzer validation for Wan.
11. Implement the disaggregated two-driver Wan forced preemption run.
12. Prove exact A-to-B-to-A transfer, reference equivalence, complete batches,
    and final residency before adding performance claims.
13. Add static/dynamic utilization repetitions and enforce thresholds.
14. Add collocated Wan recovery and report swap overhead separately.
15. Add OpenSora config/preflight and run disaggregated recovery/reference.
16. Run OpenSora utilization repetitions and collocated recovery where the
    hardware envelope supports it.
17. Run the full T1-T8/core/style suite and all shell/config preflight checks.
18. Generate reports solely from raw artifacts and inspect every failed or
    invalid repetition.
19. Update canonical design/status documents only after the required Wan and
    OpenSora rows pass.

Keep commits/reviews aligned with schema/analyzer, CPU composition, process
isolation, instrumentation, Wan correctness, utilization, and OpenSora
acceptance. Do not combine a newly discovered production correctness fix with
threshold/report changes unless they are inseparable and clearly documented.

## 19. Verification commands

### 19.1 Core parity and full core suite

From `/root/_VLAMP`:

```bash
export PYTHONPATH="$PWD/rlix-core/src${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  rlix-core/tests/test_t8_original_rlix_parity.py
/root/.venv/bin/python -m pytest -q rlix-core/tests
/root/.venv/bin/ruff check rlix-core/src rlix-core/tests
/root/.venv/bin/ruff format --check rlix-core/src rlix-core/tests
/root/.venv/bin/python -m compileall -q rlix-core/src rlix-core/tests
```

If the parity assertions are added to existing files, replace the first command
with the exact focused files and record them in the completion evidence.

### 19.2 Focused T8 CPU and local-Ray suites

From `/root/_VLAMP/RLinf`:

```bash
export PYTHONPATH="/root/_VLAMP/rlix-core/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_rlix_gpu_acceptance.py \
  tests/unit_tests/test_rlix_multipipeline.py \
  tests/unit_tests/test_task8_acceptance_workers.py
```

Run the opt-in real-Ray subprocess identity test in an environment that permits
Ray startup:

```bash
RLINF_RUN_LOCAL_RAY_TEST=1 /root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_rlix_multipipeline.py -k detached
```

### 19.3 Complete focused T1-T8 RLinf regression

```bash
export PYTHONPATH="/root/_VLAMP/rlix-core/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_world_model_resume.py \
  tests/unit_tests/test_elastic_env_rollout.py \
  tests/unit_tests/test_rlix_progress.py \
  tests/unit_tests/test_rlix_resize_coordinator.py \
  tests/unit_tests/test_rlix_placement.py \
  tests/unit_tests/test_rlix_config.py \
  tests/unit_tests/test_rlix_runtime.py \
  tests/unit_tests/test_rlix_runner_runtime.py \
  tests/unit_tests/test_rlix_embodied_runner.py \
  tests/unit_tests/test_rlix_entrypoint.py \
  tests/unit_tests/test_rlix_gpu_acceptance.py \
  tests/unit_tests/test_rlix_multipipeline.py \
  tests/unit_tests/test_task8_acceptance_workers.py \
  tests/unit_tests/test_maniskill_offload_env.py \
  tests/unit_tests/test_overlap_env_bootstrap.py \
  tests/unit_tests/test_history_manager.py
```

### 19.4 Style, compilation, config, and shell checks

```bash
/root/.venv/bin/ruff check \
  rlinf/scheduler/rlix \
  tests/unit_tests/test_rlix_gpu_acceptance.py \
  tests/unit_tests/test_rlix_multipipeline.py \
  tests/unit_tests/test_task8_acceptance_workers.py \
  tests/e2e_tests/embodied/task8_acceptance_support.py \
  tests/e2e_tests/embodied/task8_acceptance_artifacts.py \
  tests/e2e_tests/embodied/task8_acceptance_workers.py \
  tests/e2e_tests/embodied/task8_local_ray_client.py \
  tests/e2e_tests/embodied/task8_two_pipeline_driver.py \
  tests/e2e_tests/embodied/task8_two_pipeline_acceptance.py
/root/.venv/bin/ruff format --check <same-files>
/root/.venv/bin/python -m compileall -q \
  rlinf/scheduler/rlix \
  tests/unit_tests \
  tests/e2e_tests/embodied/task8_acceptance_support.py \
  tests/e2e_tests/embodied/task8_acceptance_artifacts.py \
  tests/e2e_tests/embodied/task8_acceptance_workers.py \
  tests/e2e_tests/embodied/task8_local_ray_client.py \
  tests/e2e_tests/embodied/task8_two_pipeline_driver.py \
  tests/e2e_tests/embodied/task8_two_pipeline_acceptance.py
bash -n tests/e2e_tests/embodied/run_task8_two_pipeline_acceptance.sh
```

Compose and preflight every Task 8 YAML without loading models. Validate both
role-specific resolved configs and their collision-free actor/channel names.

### 19.5 Real Wan commands

From `/root/_VLAMP/RLinf`, with a fresh or explicitly validated shared Ray
cluster:

```bash
bash tests/e2e_tests/embodied/run_task8_two_pipeline_acceptance.sh \
  --environment wan \
  --mode disaggregated \
  --scenario all \
  --config tests/e2e_tests/embodied/task8_wan_disaggregated.yaml
```

```bash
bash tests/e2e_tests/embodied/run_task8_two_pipeline_acceptance.sh \
  --environment wan \
  --mode collocated \
  --scenario recovery \
  --config tests/e2e_tests/embodied/task8_wan_collocated.yaml
```

The launcher interface may use positional arguments instead, but it must retain
explicit environment, mode, scenario, config, output, and run-ID values in the
manifest.

### 19.6 Real OpenSora commands

```bash
RLINF_OPENSORA_CHECKPOINT=/path/to/RLinf-OpenSora-LIBERO-Spatial \
bash tests/e2e_tests/embodied/run_task8_two_pipeline_acceptance.sh \
  --environment opensora \
  --mode disaggregated \
  --scenario all \
  --config tests/e2e_tests/embodied/task8_opensora_disaggregated.yaml
```

Run the collocated config separately when its preflighted memory envelope is
supported. Record exact commands, output directories, hardware, repetitions,
and results in the canonical completion update.

## 20. Definition of done

T8 and the T0-T8 project are complete only when all of the following are true:

- Two independent OS driver processes connect to the same detached core
  control-plane and scheduler actors.
- They receive distinct valid pipeline IDs/namespaces and have no RLinf-owned
  actor or channel name collision.
- Both register at least two exact canonical actor-infer bundles whose
  candidate physical GPU IDs intentionally overlap across pipelines.
- Pipeline A initially activates at least two whole bundles.
- Pipeline B's demand is emitted after a selected A rank starts a real Wan or
  OpenSora diffusion chunk and before that chunk commits.
- The chunk commits exactly once before snapshot/offload and callback success.
- A productive A sibling continues while the selected rank drains.
- Both selected peers snapshot/offload and prove non-residency before core
  release commit.
- The complete exact bundle transfers A to B with no logical or physical
  ownership overlap.
- B performs useful real-model work on the transferred bundle.
- B releases safely and A later expands the same canonical rank on its exact
  registered bundle.
- A validates/restores the same-rank snapshot and dispatches the retained next
  transition exactly once.
- Interrupted A matches its uninterrupted reference for transition order,
  trajectory counts, policy versions, rewards, done/reset state, observations,
  metrics, and Wan/OpenSora final conditioning state within declared tolerances.
- Both pipelines seal one complete lifecycle/version-consistent batch before
  advantage calculation or actor training.
- No partial, duplicate, missing, stale, or mixed-version batch trains.
- Every fixed and elastic release follows verified physical non-residency.
- Callback/offload/onload failure remains fail closed in composed tests and
  never commits unsafe ownership.
- Original-RLix partial wakeup, remaining-demand planning,
  shrink-before-expand, callback-before-commit, failure-before-commit,
  sync/resize serialization, and legacy TP behavior remain green.
- Direct GPU samples, scheduler traces, worker events, and manifests form one
  complete, internally consistent evidence timeline.
- Dynamic sharing improves median accepted throughput per physical
  actor-infer GPU by at least the declared material threshold versus the
  matched static partition and improves the corroborating idle/utilization
  metric.
- Safe-point latency, snapshot size, reclaimed memory, resize cost,
  utilization, throughput, and every repetition are reported from raw data.
- Disaggregated Wan reference/recovery/utilization acceptance passes.
- Collocated Wan recovery acceptance passes.
- Disaggregated OpenSora reference/recovery/utilization acceptance passes on a
  real checkpoint.
- Any required collocated OpenSora result is either passed or the canonical
  design is explicitly revised; a missing checkpoint is not counted as pass.
- Both pipelines finish inactive, all workers prove offloaded, registrations
  are removed, coordinators close, and the shared core remains healthy.
- Disabled standalone RLinf and the T1/T6/T7 real harnesses remain unchanged.
- Focused T1-T8, full core, Ruff, format, compilation, config, shell, and
  dependency-boundary checks pass with exact results recorded.
- The canonical design and implementation documents are updated from pending
  only after the generated reports and artifact locations have been reviewed.

Passing the correctness matrix without a material utilization improvement is
not T8 completion. Passing Wan without real OpenSora is not T8 completion.
Passing a scheduler/fake-model test without two real independent drivers is not
T8 completion.
