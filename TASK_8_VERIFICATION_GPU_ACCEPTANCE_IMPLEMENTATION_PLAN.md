# Task 8 Detailed Implementation Plan: Verification and GPU Acceptance

## 1. Status and source of truth

Status: in progress, updated 2026-08-05. T0-T7 are implemented and have focused
CPU coverage. T8 now has an executable two-OS-driver harness, shared detached
control plane, real Wan model initialization, real generation instrumentation,
safe-point drain/resume evidence, sealed-batch evidence, a single-pipeline
control, a ten-iteration two-pipeline workload, four-rank FSDP training, early
rank finalization, and asynchronous CPU policy delivery. Successive accelerator
debug runs have reached real policy inference, Wan diffusion/reward, chunk
commit, drain-barrier, snapshot/offload, completed-rank release, batch sealing,
GRPO training, candidate promotion, and all-rollout-rank version commit. The
latest checkpoint completed six linked updates per driver and exercised 11
two-rank preemption episodes before manual interruption during iteration 7.
T8 is not complete: no recorded two-driver run has completed the full linked
ten-iteration proof, and the Wan/OpenSora reference, recovery, and utilization
matrices have not passed. The current evidence is substantial implementation
progress, not Task 8 acceptance.

Implementation progress (through 2026-07-30; the dated entries below are an
append-only evidence ledger):

- `tests/e2e_tests/embodied/task8_acceptance_support.py` contains the frozen
  schema version, fail-closed event sink, deterministic CPU tensor manifests,
  scheduler ownership slices, direct GPU sample integration, and paired
  static/dynamic throughput evaluation.
- `tests/e2e_tests/embodied/task8_acceptance_analysis.py` validates sealed
  single-version batches, lifecycle-normalized reference equivalence,
  continuation-state size ceilings, and the exact event/ownership ordering of
  one A-to-B-to-A bundle transfer while a sibling rank continues.
- `tests/unit_tests/test_rlix_gpu_acceptance.py` currently provides 43 passing
  CPU tests for schema/topology validation, central and producer ordering,
  duplicate terminal rejection, typed manifest comparison, snapshot tensor
  byte accounting and ceilings, transition/batch completeness, ownership
  exclusivity and transfer ordering, sampler degradation, and the default
  five-repetition material-improvement gate.
- The pure analyzer now generates deterministic JSON-safe and Markdown report
  views only from validated reference, transfer, snapshot, utilization, and
  raw-artifact inputs. The artifact finalizer verifies the immutable stored
  manifest and sealed raw-input digests, requires the exact configured paired
  repetition set, enforces both throughput and direct-idle-reduction gates, and
  atomically emits the required correctness, ownership, latency, utilization,
  and operator-report views without overwriting prior evidence. Report emission
  also requires exactly two pipelines of linked, complete GRPO iteration
  evidence and renders batch/policy, transfer, snapshot, GPU, provenance,
  limitation, invalid-repetition, and exact-command sections from validated
  inputs.
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
- `task8_two_pipeline_acceptance.py` now supplies the fail-closed OS-process
  shell: exact role commands, exclusive stdout/stderr files, atomic
  ready/start/result rendezvous, one deadline, early-exit detection,
  terminate/kill cleanup, and validation of shared core identities, distinct
  pipeline ownership, collision-free names, and intentionally identical
  candidate mappings. A subprocess test executes two real Python processes.
- `task8_two_pipeline_driver.py` implements both the connectivity foundation
  and a model-initialization-only stage for the real driver. The latter uses
  the production Wan config, placement, worker launch, registered runtime, and
  runner initialization paths; it verifies every actor, rollout, and
  environment rank is cold/offloaded before writing readiness. Both scopes
  always label results `task8_accepted: false` because neither performs the
  required generation, preemption, or GRPO work.
- `task8_two_pipeline_acceptance.py` now has an executable preliminary-run
  interface in addition to its library API. It creates isolated role logs and
  manifests, enforces exact scope-specific readiness schemas, rejects any
  accelerator-resident model state, and writes either `pair_result.json` or
  `pair_failure.json` without allowing a preliminary run to claim acceptance.
- The orchestrator now also has an acceptance-matrix preflight that validates
  the requested environment/mode against the configured rank-to-GPU bundles,
  requires GRPO with at least two linked training iterations, freezes the hard
  utilization thresholds and operator-supplied checkpoint digests into the
  immutable run manifest, and still labels the result as non-acceptance. The
  checked-in launcher exposes this only through explicit `--preflight-only`
  until real two-driver generation is implemented. Wan and OpenSora now each
  have disaggregated and collocated matrix configs covered by CPU tests.
- A single four-A800 model-bearing Wan driver passed cold initialization in
  `287.14s`. Its immutable readiness evidence covers actor rank 0, rollout
  ranks 0-1, and environment ranks 0-1; all five records report model,
  optimizer, and CUDA-graph state non-resident after initialization, with
  canonical bundles `(0, 2)` and `(1, 3)` and an inactive registered runtime.
- The first two-driver model-initialization run correctly failed before
  readiness at the container memory boundary. Although the host exposes about
  1 TiB RAM, `/sys/fs/cgroup/memory.max` is `137438953472` bytes (128 GiB).
  One initialized pipeline used about 111 GiB across the actor (57.2 GiB), two
  rollouts (14.9 GiB each), and two environments (11.5 GiB each); loading the
  second reached `127.94/128.00 GiB`, after which Ray killed workers. GPU state
  was fully released. This is retained as failed preliminary evidence and is
  not a Task 8 acceptance result; the two-driver model stage requires a higher
  container memory limit or a separately validated memory reduction.
- After the container CPU-memory limit was raised, the two-driver Wan
  `model-init-only` run passed on 2026-07-27. Command:

  ```bash
  PYTHONPATH=/root/_VLAMP/rlix-core/src:/root/_VLAMP/RLinf:/root/_VLAMP/RLinf/tests/e2e_tests/embodied \
  /root/.venv/bin/python /root/_VLAMP/RLinf/tests/e2e_tests/embodied/task8_two_pipeline_acceptance.py \
    --scope model-init-only \
    --address 127.0.0.1:6379 \
    --run-id model-init-manual-2 \
    --output-dir /tmp/task8-model-init \
    --mode disaggregated \
    --bundles '0,2;1,3' \
    --config /root/_VLAMP/RLinf/tests/e2e_tests/embodied/task8_wan_disaggregated.yaml \
    --timeout-s 1800
  ```

  The run produced `/tmp/task8-model-init/model-init-manual-2/pair_result.json`
  with `status: passed`, `scope: model_init_only`, and
  `task8_accepted: false`. Driver A PID `84530` registered pipeline
  `rlinf_3b7af3522990`; driver B PID `84531` registered pipeline
  `rlinf_ce0ac00734e8`. Both drivers resolved the same detached control plane
  `9f30878de46e5fb18af6f0fc02000000` and scheduler
  `5f6ada4046615dbf15b7f7b502000000`, while retaining distinct pipeline
  namespaces and role-scoped RLinf worker/channel names. Both registered the
  four-GPU disaggregated actor-infer bundles `rank 0 -> (0, 2)` and
  `rank 1 -> (1, 3)`. Each driver reported five post-initialization residency
  records: actor rank 0, rollout ranks 0-1, and environment ranks 0-1. Every
  record had `model_resident: false`, `optimizer_resident: false`,
  `cuda_graph_captured: false`, and `safe_to_release: true`; both runtimes
  returned `inactive`. The repository helper `validate_driver_ready_pair()`
  accepted the two readiness files, and both driver stderr logs were scanned
  for traceback/exception/module/runtime/segmentation failure markers with no
  matches in the successful run. This proves two independent model-bearing
  Wan drivers can cold-initialize, verify physical offload, share one RLix
  core, and close cleanly on the four-GPU host. It does not prove generation,
  preemption, resume, GRPO, or utilization, so it remains preliminary evidence.
- After this slice, the focused Task 8 suite passes `51 passed, 1 skipped`, the
  complete focused Tasks 1-8 regression passes `294 passed, 2 skipped`, and
  the complete `rlix-core` suite passes `114 passed, 1 skipped`. Ruff lint
  passes for all changed surfaces. The repository-wide core format check still
  identifies three pre-existing unmodified tests (`test_gap_ratio.py`,
  `test_scheduling_cycle.py`, and `test_tracer.py`); changed-file format checks
  pass and those unrelated files were not rewritten.
- The new driver passed against a real four-A800 Ray cluster with separate PIDs
  and pipeline IDs, one shared scheduler actor, and overlapping disaggregated
  candidates `rank 0 -> (0, 2)` and `rank 1 -> (1, 3)`.
- `EmbodiedRunner` now accepts an optional exact channel-name mapping while
  preserving the original `Env`, `Rollout`, `Actor`, and `Reward` defaults.
  Task 8 role identities project into that mapping, removing the known channel
  collision when the model-bearing acceptance driver is connected.
- The driver now applies its role identity to actor, rollout, and environment
  group names, logger and per-worker output directories, experiment name, and
  runner channels in one validated configuration projection. This prevents a
  partially prefixed driver surface.
- `rlix-core` client creation failures now preserve and report the last actor
  construction exception. The real connectivity bring-up used that diagnostic
  to identify the existing Ray dashboard state-API prerequisite immediately;
  the corrected dashboard-enabled run then passed.
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
- Acceptance manifest normalization now rejects non-finite scalar, NumPy, and
  tensor values, strict JSON writers disable `NaN` emission, and linked GRPO
  evidence explicitly requires finite rewards and advantages. This closes a
  fail-open evidence path exposed by the real two-step Task 7 run below.
- The focused Task 8 command passes 48 tests with the local-Ray test skipped by
  default and 49 tests when `RLINF_RUN_LOCAL_RAY_TEST=1` enables it.
- The focused T1-T8 CPU regression available at this stage passes 291 tests
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
- Real trace-processor execution, the two-driver acceptance orchestrator,
  driver/control event wiring, and real accelerator acceptance remain pending.

Fresh prerequisite GPU results recorded on 2026-07-24 on one host with four
NVIDIA A800-SXM4-80GB GPUs:

- the Task 1 Wan/OpenVLA snapshot-resume harness passed in both collocated
  one-GPU and disaggregated two-GPU modes, with the expected
  `(1, 3, 1, 13, 256, 256)` CPU snapshot observation shape;
- the Task 6 three-GPU initialization smoke passed in 134.70 seconds with
  actor GPU 0 and canonical `actor_infer` bundle `(1, 2)`;
- the Task 7 collocated real runner passed two linked steps in 353.08 seconds,
  sealed 2/2 trajectories at final policy version 1, released its bundle, and
  closed the runtime; and
- the Task 7 metric table reported non-finite aggregate reward/advantage
  metrics for its zero-reward smoke batch. The smoke's lifecycle result remains
  valid, but Task 8 correctness analysis now rejects such evidence and this run
  does not satisfy the Task 8 GRPO numerical gate.

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
- [x] Build a preliminary acceptance orchestrator that starts two independent
  OS driver processes against one existing Ray cluster for connectivity,
  shared-control, and model-initialization scopes.
- [ ] Extend that orchestrator into the full generation acceptance path with
  preemption gates, reference capture, GRPO evidence, utilization sampling,
  and final report emission.
- [x] Derive and CPU-test collision-free role-specific RLinf worker, channel,
  and event identities while retaining the one shared core namespace.
- [x] Apply those identities to the full driver, output, and acceptance-control
  surface.
- [x] Prove two subprocess clients resolve the same detached control-plane and
  scheduler actor IDs and receive distinct pipeline IDs/namespaces.
- [x] Prove two independent model-bearing Wan drivers cold-initialize through
  production placement/launch/runner initialization, verify all actor,
  rollout, and environment ranks are offloaded, and close without claiming
  acceptance.
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
- Stage-aware completed-rank reservation: non-training-overlapping bundles are
  released immediately, while an overlapping completed bundle remains
  resident and owned until the batch is sealed.
- Atomic retained-generation shrink plus fixed actor-training enqueue in one
  scheduler cycle, with no lower-priority generation allocation interposed.
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
- a reserved rank is not simultaneously active, completed, canonically owned,
  and overlapping the declared next fixed stage;
- B acquires the retained training bundle before A's sealed-batch transition;
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

Required four-GPU disaggregated mapping on the supported host:

```text
fixed actor rank: [0]
rollout ranks:    [0, 1]
env ranks:        [2, 3]

actor_infer rank 0 = [0, 2]
actor_infer rank 1 = [1, 3]
```

The actor mapping intentionally overlaps generation GPU 0. This is safe only
because `actor_train` and `actor_infer` are different scheduler-owned stages:
the runner must release and verify physical offload of all generation bundles
before fixed actor training acquires GPU 0, and it must release actor training
before the next collection. The acceptance controller gates both pipelines so
pipeline B cannot enter a fixed actor stage while pipeline A is executing the
forced generation transfer. Any simultaneous fixed/elastic ownership of GPU 0
is an acceptance failure, not an allowed oversubscription.

Required four-GPU collocated mapping:

```text
actor ranks:   [2, 3]
rollout ranks: [0, 1]
env ranks:     [0, 1]

actor_infer rank 0 = [0]
actor_infer rank 1 = [1]
```

GPU IDs are configurable, but every recorded run stores the resolved mapping.
Do not assume `CUDA_VISIBLE_DEVICES` text is the core GPU identity; T6 resolved
placement remains authoritative.

Both independent pipelines register the same mapping. They initialize one at
a time, return fully CPU-offloaded/inactive, and only then enter the shared
generation experiment. Four GPUs are sufficient for the scheduling topology;
they do not remove the separately measured 128-GiB container-memory blocker.
The real two-driver run still requires either a higher cgroup memory limit or
a validated reduction in per-pipeline CPU-resident checkpoint state. The test
must fail preflight rather than substitute fake workers, fewer ranks, or one
driver when that host-memory prerequisite is not met.

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
4. selected A rank completed an acknowledged initial bootstrap dispatch;
5. B may start its already-synchronized elastic collection demand;
6. at least one exact A bundle transferred to B;
7. B performed useful real model work on the transferred bundle;
8. B releases enough ownership for A to resume; and
9. both pipelines seal, train, return inactive, and close.

Every wait has a positive configured deadline. Timeout never force-cancels
diffusion. On timeout, preserve logs, query statuses, stop new work, and perform
only cleanup whose residency preconditions are known.

### 7.6 Generation proof harness design

The next implementation step is a `generation-proof-only` scope that builds on
the passing `model-init-only` path. It remains `task8_accepted: false` until it
also emits reference equivalence, linked GRPO, utilization, and final report
artifacts. Its purpose is narrower: prove that two already-initialized
model-bearing drivers can interact through the real runner collection path and
perform one event-gated A-to-B-to-A actor-infer bundle transfer.

The harness uses the same four-GPU disaggregated topology already proven by
`model-init-manual-2`:

```text
pipeline A candidate actor_infer:
  rank 0 -> (0, 2)
  rank 1 -> (1, 3)

pipeline B candidate actor_infer:
  rank 0 -> (0, 2)
  rank 1 -> (1, 3)

fixed actor_train:
  GPU 0, mutually exclusive with actor_infer ownership
```

The two pipelines intentionally register identical candidate mappings. The
scheduler decides active ownership. The harness must prove that identical
candidate registration never becomes simultaneous active ownership of any
physical GPU. GPU 0 is reused across fixed actor and elastic generation stages
only by time sharing: no generation bundle may remain active or resident when
either pipeline acquires `actor_train`.

The generation-proof driver configuration is:

- use the same role-owned worker groups, channels, log directories, and runner
  channel names as `model-init-only`;
- launch `RecordingEnvWorker`, `RecordingMultiStepRolloutWorker`,
  `RecordingEmbodiedFSDPActor`, and `RecordingEmbodiedRunner`;
- configure every recording worker with an `AcceptanceControlObserverProxy`
  that submits to the named shared acceptance-control actor;
- install those proxies on the remote actor, rollout, and environment worker
  actors before `runner.init_workers()`; missing worker observers are a
  fail-closed harness error because generation hooks execute inside those Ray
  actors, not inside the local runner process;
- enrich worker-originated transition events from each worker's elastic cursor
  before submission so lifecycle generation, policy version, and transition
  identity are present even when the production method payload encodes them
  indirectly; the acceptance-control actor reports rejected event name, role,
  component, rank, lifecycle, policy, transition, GPU bundle, sequence, detail
  keys, and validation reason;
- initialize driver A and driver B sequentially, then verify both are
  `inactive` and all residencies are `safe_to_release`;
- run policy synchronization as a fixed stage before any elastic generation
  demand, with B synchronized first and A synchronized second; and
- after sync, require both pipelines to return to inactive/offloaded state
  before the forced generation interaction begins.

The frozen four-GPU Wan workload is:

| Setting | Value | Effective generation-proof behavior |
| --- | ---: | --- |
| physical GPUs | 4 | one single-node Ray cluster |
| actor training placement | GPU 0 | fixed stage, time-shared with generation |
| rollout placement | GPUs 0, 1 | one rollout worker per DP rank |
| environment placement | GPUs 2, 3 | one Wan worker per DP rank |
| canonical bundles | `(0, 2)`, `(1, 3)` | rollout/environment peers move together |
| `total_num_envs` | 16 | eight environments per DP rank |
| `group_size` | 8 | one complete GRPO group per rank, two groups per collection |
| `rollout_epoch` | 1 | one trajectory per environment |
| trajectories per rank | 8 | eight environments times one epoch |
| trajectories per collection | 16 | the scheduler's step target |
| `max_episode_steps` | 256 | production primitive-step truncation ceiling; success may end an episode earlier |
| `max_steps_per_rollout_epoch` | 256 | one complete production-length trajectory per environment |
| OpenVLA-OFT action chunk | 8 | eight primitive actions per policy result |
| policy/environment chunks | 32 | at most `256 / 8` interaction commits, unless sparse success terminates earlier |
| Wan `num_inference_steps` | 5 | production Wan generation quality for meaningful sparse reward classification |
| reward type | action-level | binary per-frame success predictions are retained through trajectory sealing |
| reward filtering | enabled, inclusive `[0.0, 5.0]` group-mean bounds | admit the complete binary-reward range, including uniform all-failure and all-success groups |
| RLix monitor interval | 0.1 seconds | bounded Ray status-query pressure |
| configured `max_train_steps` | 10 | ten linked collections and actor updates per pipeline |
| measured repetitions | 5 | full acceptance matrix only |
| warmup repetitions | 1 | full acceptance matrix only |

For this two-rank configuration, RLinf requires
`total_num_envs / env_world_size / pipeline_stage_num` to be divisible by
`group_size`. Here that is `16 / 2 / 1 = 8`, so each rank owns one complete
eight-trajectory GRPO group and no group crosses a rank boundary. The collection
still contains two independent groups, as the preceding 8-environment/group-4
diagnostic did, but each group now compares twice as many stochastic alternatives.
For an illustrative independent binary outcome with success probability 0.5,
the probability that a group has no reward variation falls from 12.5% at group
size 4 to about 0.78% at group size 8. Real Wan outcomes are correlated, so this
is a sizing intuition rather than an acceptance threshold. The cost is twice the
logical trajectories, environment state, video work, and maximum model/environment
interaction volume per collection.

`generation-proof-only` executes ten linked collections and GRPO actor updates
per pipeline. The first iteration deterministically waits for A rank 0 to
complete and release `(0, 2)`, runs useful B generation on that bundle, and then
uses A's sealed-batch actor-training request to pause and resume B. After both
first updates complete, iterations 2 through 10 use normal stage-aware shared
scheduler arbitration. Each later collection synchronizes the
policy produced by the preceding update before requesting generation. It does
not execute the one warmup plus five measured utilization repetitions, which
belong to the later full Task 8 acceptance matrix. With two pipelines, the proof
run therefore collects 320 trajectories in total. Each trajectory may terminate
early when the Wan reward model recognizes
success; otherwise it is truncated after 256 primitive environment steps. Wan
still computes sparse action-level reward predictions for every generated
eight-frame chunk, but GRPO does not resolve group rewards, apply the production
reward filter, or compute advantages until all sixteen trajectories in that
pipeline have reached a terminal boundary and the aggregate actor batch is
sealed. A collection with no reward-model success is valid generation/lifecycle
evidence: its all-zero group remains in the loss mask, produces finite zero
GRPO advantages and a zero-gradient actor update, and completes the iteration.
The proof still rejects incomplete batches, mixed policy versions, non-finite
values, and missing actor-update boundaries. Thus allowing a zero-success
iteration does not claim policy improvement or weaken lifecycle validation.

The `run_id` is shared by artifact paths, named Ray actors, worker groups, and
channels. It must be one safe component: letters, digits, `_`, `.`, `-`, or `=`,
starting with a letter or digit. It must not contain `/`, whitespace, or `:`.
For manual reruns, use compact UTC identifiers such as
`generation-proof-20260728T120000Z`, not path-shaped values or colon-separated
ISO timestamps.

The 100 ms collection monitor interval is intentional for the real-model proof.
The runtime still reacts much faster than a Wan/OpenVLA inference unit, while
avoiding the unbounded Ray task-event pressure seen with the former implicit
10 ms interval during multi-minute cold generation. This polling cadence does
not change transition, safe-point, or completion semantics.

The acceptance-control actor owns the following deterministic gates:

> **Gate compatibility note:** the control actor retains the old forced-
> preemption gate names for the dependency-light recovery smoke, but the real
> generation-proof orchestrator uses the completed-rank and per-role training
> gates defined in Section 10. Generation demand no longer drains incomplete A
> work in the RLinf scheduling mode.

```text
both_drivers_initialized
allow_b_policy_sync
b_policy_sync_completed
allow_a_policy_sync
a_policy_sync_completed
allow_a_collection
a_generation_granted
a_target_bootstrap_dispatched
a_target_chunk_started
a_target_rank_completed
allow_b_collection
b_generation_requested
transfer_to_b_observed
b_useful_work_observed
a_batch_sealed
allow_a_training
a_training_started
a_training_completed
b_resume_observed
b_batch_sealed
allow_b_training
b_training_started
b_training_completed
both_training_completed
```

The expected pipeline interaction is:

```text
1.  Orchestrator creates the run-owned acceptance-control actor and starts
    drivers A and B as separate OS processes.
2.  Driver A initializes real actor, rollout, and Wan environment workers,
    verifies offload, writes readiness, and waits.
3.  Driver B does the same independently with distinct pipeline ID, namespace,
    groups, channels, output directory, and PID.
4.  Driver B performs production fixed policy sync, then releases all fixed
    ownership and reports `policy_synchronized`.
5.  Driver A performs production fixed policy sync, then releases all fixed
    ownership and reports `policy_synchronized`.
6.  Driver A enters production collection. The scheduler grants A both
    actor-infer bundles. A rank 0 owns `(0, 2)`; A rank 1 owns `(1, 3)`.
7.  A rank 0 completes its assigned trajectories naturally, reports durable
    `COMPLETED`, and releases `(0, 2)` through completion-aware selected-rank
    release. This offloads both peers without creating a pause token.
8.  The orchestrator releases B collection demand. In `fixed_stage_only` mode,
    B may acquire `(0, 2)` after A's release commit but cannot drain A rank 1.
9.  B emits real policy or chunk work on `(0, 2)`. Meanwhile A finishes rank 1,
    releases it, validates the aggregate actor receipt, and emits runner-level
    `batch_sealed`.
10. Only after both B useful work and A seal are observed does the orchestrator
    release A to call `_train_rlix_batch()`. Its `ACTOR_TRAINING` request needs
    GPU 0 and therefore drains B's atomic rank 0 bundle `(0, 2)` at the next
    safe chunk boundary. A does not directly request a B drain.
11. B commits the current chunk exactly once, snapshots its retained
    continuation, offloads both peers, and returns the callback. Scheduler
    ownership transfers to A's fixed actor-training allocation only afterward.
12. A computes GRPO values, completes its actor update, offloads fixed state,
    and releases GPU 0. A then waits instead of starting iteration 2.
13. B's paused rank receives expansion preference, reacquires `(0, 2)`, validates
    its same-rank token/policy, restores, and dispatches the retained bootstrap
    exactly once.
14. B completes and seals its own batch. The orchestrator then permits B actor
    training. Both first updates must complete before either driver advances.
15. Iterations 2 through 10 repeat normal stage-aware arbitration without the
    first-iteration gates: completion-released bundles feed competing
    generation, and fixed training requests preempt only overlapping bundles.
16. After ten updates, both runtimes release ownership, verify final offload,
    unregister, close workers, and persist events/results.
```

Shrink completion-race semantics:

- A scheduler-driven shrink samples environment and rollout peers with separate
  Ray RPCs. The pair is not an atomic status object.
- A rank may be observed at the natural-completion boundary with one peer already
  reporting `COMPLETED` while the other still reports `ACTIVE` or
  `DRAIN_REQUESTED`. This is legal transient skew when the rank has just
  finished its assigned trajectories.
- The coordinator must not fail closed on that transient
  `COMPLETED`/`ACTIVE` or `COMPLETED`/`DRAIN_REQUESTED` sample. It waits for the
  stored environment and rollout run calls, then validates the real outcomes:
  both `COMPLETED` enters completed offload, both `PAUSE_READY` enters pause
  offload, and any true outcome mismatch remains fail-closed.
- Other peer-state mismatches are still invalid. Final post-shrink peer states
  must match and must be either `PAUSED` or `COMPLETED`, with both peers
  verified non-resident before the scheduler callback can release GPUs.
- `FAILED_RESIDENT` is not tolerated as transient skew. It means one peer
  recorded an exception while it still had model/GPU residency, so the
  coordinator must fail closed and preserve the existing scheduler allocation.
  Peer mismatch diagnostics must include both persisted worker failure strings
  and stored run-task states so a failing GPU run identifies the original
  worker-side exception without requiring a separate log search.

Generation-proof rerun `generation-proof-20260728T031142Z` demonstrated that
this diagnostic path works as intended. Both A rollout ranks received real Wan
observations and emitted `policy_request_started`; rank 0 then entered
`FAILED_RESIDENT` with `TypeError: Got unsupported ScalarType BFloat16`. The
failure was not a lifecycle race: the acceptance-only
`policy_request_completed` hook attempted to normalize the successful policy
result with `Tensor.numpy()`, but NumPy has no native bfloat16 scalar type. The
evidence encoder now handles CPU bfloat16 tensors explicitly: it preserves the
`bfloat16` dtype and two-byte logical size, hashes the original uint16 storage
bits, and widens only optional inline JSON values to float32. The production
policy result is not cast or mutated. A focused regression covers stable BF16
encoding, byte accounting, and non-finite rejection. This run remains failed
preliminary evidence and does not satisfy the generation-proof assertions.

Generation-proof rerun `generation-proof-20260728T033439Z` progressed through
both real policy calls and proved the BF16 evidence fix. Rank 1 entered
`env_interact_step()` at `03:44:30.163 UTC`; rank 0 accepted its paired drain
request and entered `env_interact_step()` at `03:44:30.742 UTC`. Neither rank
then emitted `chunk_committed` before the operator interrupted the run about
322 seconds later. There was no CUDA, NCCL, Ray object-store, OOM, or worker
exception; `pair_failure.json` records the operator `KeyboardInterrupt`. This
localizes the stall to the common synchronous Wan `chunk_step()` path, but the
old chunk-level hooks cannot distinguish model onload, diffusion/VAE work, and
reward inference.

The generation-proof recording environment now installs acceptance-only,
per-chunk diagnostic wrappers around those three production operations. It
emits `world_model_onload_{started,completed,failed}`,
`world_model_diffusion_{started,completed,failed}`, and
`world_model_reward_{started,completed,failed}`. A started marker is committed
to the central event log before entering the operation; completed and failed
markers include elapsed seconds, and failed markers also preserve exception
type and text. The wrappers delegate to the original bound methods, preserve
the exact return value, and restore the original environment surface in a
`finally` block. Therefore a killed rerun identifies the active Wan phase
without replacing or weakening production computation.

Generation-proof rerun `generation-proof-20260728T065253Z` advanced both Wan
environment ranks through the reward-call return marker but emitted no
`chunk_committed` marker. System-wide GPU utilization from an unrelated
operator workload is not acceptance evidence and must not be used to infer the
blocked project phase. The remaining boundary is post-reward processing inside
`WanEnv.chunk_step()` or immediate `EnvOutput` assembly.

The acceptance harness therefore supports switchable fine-grained phase
diagnostics. `--phase-diagnostics` is enabled by default for the current debug
runs; `--no-phase-diagnostics` disables every detailed world-model marker while
retaining the required acceptance events such as `chunk_started` and
`chunk_committed`. Worker instances default to detailed diagnostics disabled
and must be explicitly configured by the harness. When enabled, a temporary
Wan observer records reward return, reward-difference calculation, success
estimation, the natural truncation and done synchronization checks, optional
automatic reset, metric construction, render tensor conversion, chunk return,
and completed `EnvOutput` assembly. The observer is removed in a `finally`
block after each chunk. The checks record only compact booleans such as
`has_truncations` and `has_past_dones`; they do not serialize intermediate
model tensors or insert a synchronization ahead of the corresponding
production check.

Rerun `generation-proof-postreward-debug-1` reached
`world_model_chunk_step_returning` and `env_output_constructed` on both A
environment ranks. Both ranks reported `has_truncations: false` and
`has_past_dones: false`, and both completed metrics and render conversion. The
next acceptance hook attempted to attach the complete live `EnvOutput` tuple to
`chunk_committed`. Manifest normalization expands dataclasses with
`dataclasses.asdict()`, which recursively deep-copies their fields before the
CPU-tensor validator runs. The result contains nested accelerator tensors, so
observability introduced an unsafe copy/synchronization on the commit path.
`chunk_committed` is now a bounded identity/order marker containing stage,
transition, lifecycle, policy, rank, bundle, sequence, and timestamp context,
but no live result payload. Detailed numerical evidence remains the
responsibility of later CPU-sealed snapshots and batches. This changes only
acceptance evidence encoding; the production result returned to the rollout
path is unchanged.

Rerun `generation-proof-postreward-debug-2` proved that both environment ranks
now reach `chunk_committed`, but exposed a production drain-barrier routing bug.
The live coordinator observation showed A environment rank 0 in
`FAILED_RESIDENT` with `ValueError: split sizes must sum to the request logical
batch size`, its rollout peer still in `DRAIN_REQUESTED`, and A rank 1 naturally
`COMPLETED`. With four global environments and two one-to-one environment and
rollout ranks, the route planner assigns a local shard of two environments to
rank 0. Observation envelopes already describe that local shard. The drain
barrier incorrectly described the global training batch of four, so its
`[2]` route split could not satisfy a logical size of four and the barrier never
reached rollout rank 0.

Drain barriers now use the local per-rank, per-stage environment count for
their envelope logical batch size while retaining the global batch size as the
input to route planning. Regression coverage composes the real two-source,
two-destination route calculation with a four-environment global batch and
requires a single local barrier shard of size two. Split validation failures
report the request kind, calculated split sizes and sum, and declared logical
batch size.

Paired coordinator waits now retain legal asynchronous completion skew but
return on the first genuine task exception. Normal completion of only one peer
still waits for the other; an exception is surfaced immediately because the
other peer may be blocked on a message that the failed peer was responsible for
sending. This prevents a worker exception from appearing as a hang until the
900-second operation timeout. The coordinator remains fail-closed and preserves
resident allocation state on error.

When `--phase-diagnostics` is enabled, the acceptance environment also emits
`barrier_send_started`, `barrier_send_completed`, or `barrier_send_failed` with
compact exception type and text, followed by the required `drain_observed` only
after a successful send. These additional markers are disabled by
`--no-phase-diagnostics`; the production barrier protocol and required
acceptance events are unchanged.

The first rerun with these markers reached both real Wan `chunk_committed`
events, then failed before entering the production barrier send because the
strict acceptance vocabulary had not yet registered `barrier_send_started`.
All three barrier phase events are now members of the known-event and
transition-event contracts, so the sink both accepts them and requires their
rank, lifecycle, policy, and transition identity. Schema-level regressions pass
each marker through `AcceptanceEvent.validate()` and `AcceptanceEventSink`;
worker-only list-observer coverage is not considered sufficient for new event
types.

The first generation-proof implementation should stop after one bounded
collection and one actor update per pipeline if that is the smallest reliable
production path. It must not claim Task 8 acceptance until a follow-up scope
runs two linked GRPO iterations and proves that policy version `N+1` produced
by the first update is synchronized into the next collection.

Hard assertions for this scope:

- A and B readiness schemas remain identical to `model-init-only` plus
  acceptance-control actor identity and event-log path.
- A and B share the same detached control plane and scheduler actor IDs.
- A and B pipeline IDs, namespaces, worker groups, channel names, and output
  directories are distinct.
- A rank 0 acknowledged `bootstrap_dispatched` precedes B generation demand.
- A later `policy_request_completed` and `chunk_committed` prove that the
  scheduling milestone progressed through real policy and environment work.
- A rank 0 `drain_requested` occurs before A rank 0 `chunk_committed`.
- A rank 0 `chunk_committed` occurs exactly once for the selected transition.
- A rank 1 emits useful progress after A rank 0 is selected for drain.
- A snapshot and rollout/environment offload receipts precede the scheduler
  `release_committed` for `(0, 2)`.
- B `allocation_committed` for `(0, 2)` follows A release and precedes B useful
  work.
- A and B never overlap active scheduler ownership or reported physical
  residency on any GPU in `(0, 2)`.
- B release precedes A reallocation/resume.
- A emits exactly one retained-transition resume dispatch.
- Both batches are sealed before training starts.
- No fixed `actor_train` stage starts while any generation bundle using GPU 0
  is active or resident.
- Final residency for every actor, rollout, and environment rank is
  `safe_to_release`.

Failure handling is also part of the design. If a driver exits, a worker
observer rejects an event, a timeout fires, or a callback fails, the
acceptance-control actor publishes the failure to every waiter. The
orchestrator must then preserve partial logs and write `pair_failure.json`
without attempting to convert the run into a weaker single-driver or fake-model
case.

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

## 10. Training-readiness preemption, reuse, and resume scenario

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

Allow A to collect on both ranks until one rank completes its assigned
trajectories naturally. A reports that rank in `completed_dp_ranks` and invokes
the completion-aware selected-rank release. Only after callback-verified
offload and scheduler release commit may already-synchronized B enter
`_collect_rlix_rollouts()` and acquire that idle bundle. B's generation request
alone must not drain A's other active, incomplete generation rank.

When A's remaining rank also completes, A validates and seals its complete CPU
batch and submits the normal fixed `ACTOR_TRAINING` request. That pending fixed
request is the authoritative training-readiness signal. The scheduler drains
only B generation bundles that overlap A's fixed training mapping, at the
existing world-model chunk safe point. For the four-GPU test, actor training
uses GPU 0, so B rank 0's atomic `(0, 2)` bundle is drained even though GPU 2
is not used by the trainer; B rank 1 on `(1, 3)` may continue. A production
all-GPU FSDP mapping would drain both B bundles.

The acceptance driver must not infer training readiness from trajectory counts
alone or bypass batch sealing, reward resolution, advantage calculation, or
the T7 fixed-stage API.

### 10.3 Shrink acceptance

For the training-triggered B shrink, require:

- A's naturally completed rank was released without creating a pause token;
- B generation demand did not evict A's incomplete sibling;
- only B rank(s) overlapping A's fixed training request receive drain;
- the current real chunk commits exactly once;
- no next observation is sent before pause;
- peer safe-point tokens match;
- snapshot and both offloads complete;
- device/process memory drops by the declared reclaimed-memory threshold;
- callback returns after both residency receipts; and
- core release commit follows callback success for the whole bundle.

### 10.4 Competing reuse acceptance

Require B to acquire the exact bundle voluntarily released by A's completed
rank and perform at least one recorded VLA prediction or Wan/OpenSora chunk
while A's sibling continues. When A later requests training, B must retain any
non-overlapping bundle and safely pause/offload each overlapping bundle before
A's fixed allocation commits.

Logical trace slices and physically resident process intervals must not overlap
between A and B on the transferred GPUs.

### 10.5 Resume acceptance

After A trains, offloads, and releases fixed ownership, give B's paused
resumable rank preference over A's new collection demand. Require B to expand
the same canonical rank on the same registered bundle, validate the stored
token and policy version, onload both peers, restore, and dispatch the retained
bootstrap once.

B then completes its assigned work and may request its own fixed actor-training
stage, symmetrically interrupting overlapping generation from A's next
collection. Both pipelines must release every elastic rank, seal complete
batches, train only after their own seals, verify fixed offload, return
inactive, unregister, and close. Equal-priority training requests require
deterministic FIFO arbitration, and a just-trained pipeline must not starve
older resumed work by immediately starting a new collection.

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

Run the four-GPU topology with two width-two bundles and the real OpenVLA-OFT
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

- two-rank four-GPU mapping with stage-isolated fixed actor overlap and bounded
  workloads.

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
8. Add the run-owned event sink and shared acceptance-control barriers.
9. Prove the two-driver control-only and model-initialization-only scopes on
   the four-GPU Wan host, retaining `task8_accepted: false`.
10. Wire the recording worker subclasses into a `generation-proof-only` driver
   scope and configure every worker with a shared actor observer proxy.
11. Add event-gated policy-sync and collection gates so B's generation demand
   is released only after A rank 0 completes an acknowledged production
   bootstrap dispatch; retain later policy/chunk events as generation proof.
12. Prove one real disaggregated Wan A-to-B-to-A generation transfer with
   complete offload, useful B work, A resume dispatch, sealed batches, and
   final residency.
13. Add trace preparation and NVML sampler correlation to that generation run.
14. Refactor the T7 real smoke construction into reusable test helpers only
   where this avoids copying production configuration logic; keep the T7
   command working unchanged.
15. Implement uninterrupted reference capture and analyzer validation for Wan.
16. Extend the generation-proof run into the full disaggregated Wan forced
    preemption acceptance run.
17. Prove exact A-to-B-to-A transfer, reference equivalence, complete batches,
    and final residency before adding performance claims.
18. Add two linked GRPO iterations per pipeline and prove the produced policy
    version is collected by the next iteration.
19. Add static/dynamic utilization repetitions and enforce thresholds.
20. Add collocated Wan recovery and report swap overhead separately.
21. Add OpenSora config/preflight and run disaggregated recovery/reference.
22. Run OpenSora utilization repetitions and collocated recovery where the
    hardware envelope supports it.
23. Run the full T1-T8/core/style suite and all shell/config preflight checks.
24. Generate reports solely from raw artifacts and inspect every failed or
    invalid repetition.
25. Update canonical design/status documents only after the required Wan and
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

### 19.7 Mid-rollout continuation hardening result (2026-07-28)

The disaggregated Wan generation-proof run
`generation-proof-barrier-contract-fix-1` progressed through real chunk commit,
barrier consumption, drain observation, and snapshot construction. Pause
offload then failed validation because `last_observations` and
`last_intervened_info` were empty. This was downstream of the earlier barrier
and acceptance-instrumentation failures and occurred before release, ownership
transfer, onload, or actual resume.

The root cause was a resume-schema dependency on end-of-rollout caches:
`auto_reset=False` leaves those caches empty until rollout finalization, while
Task 8 pauses after a committed chunk. Worker resume schema version 3 now uses
the committed `current_env_outputs` and retained `resume_bootstraps` pair as
the canonical stage continuation. It normalizes observation/intervention
fields from the committed output, validates the complete snapshot before
reporting `snapshot_completed`, and repeats validation during offload and
restore. A divergent retained bootstrap fails closed. This behavior is
reset-mode independent and is covered by focused empty-cache, stale-cache,
divergence, ownership, and restore tests.

The next rerun reached this new construction-time validation and exposed a
second pre-existing assumption: the validator required one continuous
`EmbodiedRolloutResult.actions` entry per committed chunk. Real OpenVLA-OFT
instead retains its training target as `forward_inputs["action_tokens"]`; the
decoded `[environment, chunk, action]` tensor was valid and had already driven
the committed Wan chunk. Validation now accepts either a complete continuous
action sequence or complete model forward inputs, requires elastic transition
identity coverage, rejects partial/missing representations, and emits all
counts and forward-input keys on failure. The run still stopped before
`snapshot_completed`, offload, transfer, onload, or resume, so it is diagnostic
progress rather than Task 8 acceptance evidence.

The following `generation-proof-action-alignment-1` rerun accepted the real
OpenVLA-OFT token-action representation and advanced to reward alignment. It
showed the expected first-chunk safe-point ledger: one policy input, log-prob,
value, version, transition identity, and done boundary; zero materialized
rewards; and one reward retained in the pending `EnvOutput`. The validator had
incorrectly required that retained reward to be materialized already. The
phase-aware contract now requires `committed_chunks - 1` trajectory rewards,
one pending-bootstrap reward, and result-boundary counts that include prior
epoch finalization. Missing and duplicate pending rewards fail closed, snapshot
does not mutate the ledger, and pause/restore tests prove one-time later
materialization. This run also stopped before `snapshot_completed` and remains
diagnostic rather than acceptance evidence.

### 19.8 Aggregate batch-seal evidence contract (2026-07-29)

The `generation-proof-reward-ledger-1` rerun advanced beyond the earlier
continuation failures: A snapshotted and offloaded, B performed real generation
after ownership transfer, A restored and performed post-resume Wan work, and
the active rollout ranks completed. Production actor batch sealing then
validated its CPU batch, but the acceptance producer rejected the following
evidence event. It had recursively selected one rank-1 environment transition
from the aggregate batch details and attached it to actor producer rank 0.
Those ranks are different coordinate spaces: the actor rank identifies the
batch consumer, while every transition identity identifies its contributing
environment rank. A singleton actor is expected to consume transitions from
both environment ranks 0 and 1.

The acceptance contract therefore treats `batch_sealed` as aggregate evidence,
as its existing exclusion from the rank-local and transition-local event sets
already specified. It forbids a singular top-level `transition_identity`,
retains the complete transition list in the sealed details, and validates that
list independently for uniqueness, lifecycle, policy version, transition
count, and exact coverage of the receipt's `contributing_dp_ranks`. Rank-local
events retain the existing producer-rank/transition-rank equality check. Tests
exercise both transition arrival orders so acceptance cannot depend on which
environment trajectory reaches the actor first, and malformed contributor
coverage continues to fail closed. The rerun is generation/resume evidence but
is not full generation-proof acceptance because GRPO training did not start.

### 19.9 Completion-aware selected-rank release (2026-07-29)

The `generation-proof-aggregate-seal-fix-1` rerun passed real snapshot,
offload, ownership transfer, restore, and post-resume Wan generation. Pipeline
A's resumed rank 0 then completed its final assigned trajectory. The runtime's
rank observation still described the completed worker as callback-active, so
it published `completed=4`, `active_dp_ranks=[0]`, and
`completed_dp_ranks=[0, 1]`. The central scheduler reacted to zero remaining A
demand and independently released A's last rank for the still-incomplete B
pipeline. The runtime then submitted its exact-rank release from the earlier
observation; scheduler ownership was already empty, and the strict API raised
`No requested ranks are active`. This occurred immediately after A's final
`rank_completed` event and before reward aggregation, `batch_sealed`,
advantage calculation, or training. Later actor-death and expansion errors
were teardown consequences.

Selected-rank release is now an atomic completion-aware compare-and-release
operation under the scheduler lock. Every requested rank must belong to the
canonical registered mapping. Active ranks retain ownership validation and go
through the existing verified shrink callback before commit. Already-inactive
ranks are accepted only when the current progress snapshot includes them in
`completed_dp_ranks`; mixed requests shrink only the active subset. The same
partition is recomputed during planning so a background shrink committed in
the API-to-plan lock gap is also safe. Unknown ranks and inactive ranks without
current completion evidence still fail closed. Repeated completed release is
therefore idempotent without catching exception strings or weakening physical
ownership validation. New collection progress replaces the old snapshot and
sealing clears it, bounding the evidence to the current collection lifecycle.

### 19.10 Single-pipeline Wan control experiment (2026-07-29)

`tests/e2e_tests/embodied/task8_single_pipeline_diagnostic.py` is the control
experiment for separating generation/reward behavior from cross-pipeline
preemption behavior. It composes the same production
`wan_libero_spatial_grpo_openvlaoft_rlix` base and the same Task 8 overrides as
the two-driver run. The resulting pipeline has actor rank 0 on GPU 0, rollout
ranks on GPUs 0 and 1, environment ranks on GPUs 2 and 3, and canonical
actor-infer bundles `rank 0 -> (0, 2)` and `rank 1 -> (1, 3)`. It uses sixteen
training environments, GRPO group size 8, rollout epoch 1, 256 environment
steps per trajectory, five Wan diffusion steps per chunk, production reward
filter bounds, and ten configured training iterations. Each DP rank owns eight
environments and therefore contributes one complete GRPO group per collection;
the sealed actor batch contains two groups and sixteen trajectories.

Only one RLix pipeline is registered. There is no acceptance control actor,
second driver, generation gate, or competing scheduler demand, so no
cross-pipeline shrink, transfer, restore, or resume is expected. RLix remains
enabled deliberately: initialization, fixed stages, elastic collection, batch
sealing, and completed-rank release still use the same production paths as the
two-pipeline test. This makes the run a controlled comparison rather than a
legacy-runner or non-Ray smoke test. It is diagnostic evidence and cannot
satisfy Task 8's two-pipeline acceptance requirements.

The run refuses to reuse a non-empty output directory and writes a fully
resolved config before model launch. Training video capture is enabled at
`<run>/trajectories/videos/seed_*/<n>.mp4`; the normal environment
`finish_rollout()` path synchronously finalizes each MP4. For Wan, the current
chunk API exposes one returned observation per eight-action chunk, so these
videos are suitable for visual trajectory review but do not contain every
intermediate diffusion frame.

At each actor batch seal, before advantage calculation or training, the
diagnostic actor atomically writes
`<run>/trajectories/iteration_<policy>/sealed_batch.pt` and
`reward_summary.json`. The tensor artifact retains the complete CPU-owned
batch for offline inspection. The summary records reward shape/range/nonzero
count, per-trajectory sums, GRPO group sums and means, configured filter
bounds, accepted groups, post-filter active trajectories, terminal/truncation
counts, policy versions, and the elastic batch receipt. Therefore the first
generated batch remains available if a later production stage fails. Filtering
remains enabled but admits the complete binary group-mean range `[0.0, 5.0]`,
so a uniform group yields a finite zero-advantage update rather than an empty
mask. The harness records a nonzero exit in `result.json` plus `failure.txt`
when production execution fails.

The intended clean control run is:

```bash
cd /root/_VLAMP
/root/.venv/bin/ray stop --force
rm -rf /tmp/task8-single-pipeline/single-realistic-wan-16x8-1
rm -f /tmp/task8-single-pipeline/single-realistic-wan-16x8-1.console.log
mkdir -p /tmp/task8-single-pipeline

RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
/root/.venv/bin/ray start \
  --head --port=6379 --num-gpus=4 \
  --include-dashboard=true --dashboard-host=0.0.0.0 --disable-usage-stats

set -o pipefail
PYTHONPATH=/root/_VLAMP/rlix-core/src:/root/_VLAMP/RLinf:/root/_VLAMP/RLinf/tests/e2e_tests/embodied \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
/root/.venv/bin/python \
  /root/_VLAMP/RLinf/tests/e2e_tests/embodied/task8_single_pipeline_diagnostic.py \
    --address 127.0.0.1:6379 \
    --run-id single-realistic-wan-16x8-1 \
    --output-dir /tmp/task8-single-pipeline \
    --config /root/_VLAMP/RLinf/tests/e2e_tests/embodied/task8_wan_disaggregated.yaml \
    --bundles '0,2;1,3' \
  2>&1 | tee /tmp/task8-single-pipeline/single-realistic-wan-16x8-1.console.log
```

`tee` leaves all driver and worker-forwarded output visible in the terminal
while retaining the same stream in the adjacent `.console.log`. Structured
runner metrics remain under `<run>/logs/metrics.log`, and training videos remain
enabled under `<run>/trajectories/videos/seed_*/<n>.mp4`.

### 19.11 Zero-success-compatible two-pipeline proof and live artifacts (2026-07-29)

The two-driver generation proof now composes the same Wan workload used by the
single-pipeline control: 16 total environments, two environment DP ranks,
eight environments and one group per rank, GRPO group size 8, rollout epoch 1,
256 primitive steps per trajectory, five Wan diffusion steps, and ten linked
training iterations per driver. The first iteration is explicitly gated to
prove completed-rank reuse followed by training-triggered B pause/resume. After both first actor updates,
iterations 2 through 10 synchronize the preceding policy and run through normal
shared scheduler arbitration.

Acceptance producers derive policy version from each current cursor or batch
receipt instead of freezing worker context at policy 0. Aggregate terminal-event
uniqueness includes policy version, so `batch_sealed` and `training_completed`
may occur once for each of policies 0 through 9 while duplicates within one
policy still fail closed. The result requires both configured and completed
iteration counts to equal 10.

Reward filtering remains enabled, but its inclusive group-mean interval is
`[0.0, 5.0]`. These are the complete possible terminal-masked group means for
the current binary reward model with reward coefficient 5. An all-failure
group is therefore retained with zero rewards and finite zero advantages; an
all-success group is retained with uniform rewards and the same zero relative
advantages. Mixed groups remain the only groups capable of producing a useful
GRPO gradient. The generation proof accepts zero-success iterations as valid
lifecycle and execution evidence, not as learning-quality evidence. Batch
cardinality, rank contribution, policy version, transition identity, finite
reward/advantage values, and actor-update completion remain fail-closed.

Each driver enables the production `RecordVideo` wrapper at
`<run>/drivers/<role>/trajectories/videos/seed_*/<n>.mp4`. Videos are flushed
synchronously at rollout completion. They are tiled per environment worker and
retain the Wan limitation that only the observation returned for each
eight-frame chunk is recorded. The driver result includes its trajectory root.

The orchestrator's `--stream-driver-logs` option is enabled by default. It
mirrors both child stdout/stderr streams to the parent terminal with driver and
stream prefixes while preserving the unmodified streams in
`drivers/<role>/stdout.log` and `stderr.log`. `--no-stream-driver-logs` disables
only terminal mirroring and does not disable artifact logs.

The intended clean two-pipeline rerun is:

```bash
cd /root/_VLAMP
/root/.venv/bin/ray stop --force
rm -rf /tmp/task8-generation-proof/generation-proof-zero-reward-videos-1
rm -f /tmp/task8-generation-proof/generation-proof-zero-reward-videos-1.console.log
mkdir -p /tmp/task8-generation-proof

RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
/root/.venv/bin/ray start \
  --head --port=6379 --num-gpus=4 \
  --include-dashboard=true --dashboard-host=0.0.0.0 --disable-usage-stats

set -o pipefail
PYTHONPATH=/root/_VLAMP/rlix-core/src:/root/_VLAMP/RLinf:/root/_VLAMP/RLinf/tests/e2e_tests/embodied \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
/root/.venv/bin/python \
  /root/_VLAMP/RLinf/tests/e2e_tests/embodied/task8_two_pipeline_acceptance.py \
    --scope generation-proof-only \
    --address 127.0.0.1:6379 \
    --run-id generation-proof-zero-reward-videos-1 \
    --output-dir /tmp/task8-generation-proof \
    --mode disaggregated \
    --bundles '0,2;1,3' \
    --config /root/_VLAMP/RLinf/tests/e2e_tests/embodied/task8_wan_disaggregated.yaml \
    --phase-diagnostics \
    --stream-driver-logs \
    --timeout-s 21600 \
  2>&1 | tee /tmp/task8-generation-proof/generation-proof-zero-reward-videos-1.console.log
```

### 19.12 End-to-end progress and target arbitration correction (2026-07-29)

The end-to-end effort has progressed beyond connectivity and model-init smoke.
Two independent model-bearing processes have cold-initialized and offloaded on
the four-GPU host while sharing one detached scheduler. Successive two-driver
generation runs have executed real OpenVLA policy calls and Wan environment
chunks and have exercised scheduler shrink entry, paired worker observation,
drain-barrier routing, continuation snapshot construction, offload diagnostics,
completed-rank release, aggregate batch sealing, and GRPO boundaries. These
runs exposed real defects rather than producing acceptance passes: transient
peer-state skew, hidden `FAILED_RESIDENT` causes, an unconfigured observer,
strict event-vocabulary rejection, BF16 evidence encoding, unsafe live-result
normalization, global-versus-local barrier cardinality, paired-task exception
propagation, result-boundary accounting, aggregate seal semantics, and stale
completed-rank release. Each repaired boundary remains fail-closed and has
focused regression coverage.

The single-pipeline Wan control completed through production generation and
training without cross-pipeline preemption. It demonstrated that unsuccessful
trajectories are legal RL samples: the production runner can seal them,
calculate finite GRPO values, and complete an optimizer boundary even when a
uniform group supplies no useful relative-policy gradient. The diagnostic now
retains sealed CPU batches, reward summaries, videos, metrics, and terminal
logs. This isolates the remaining two-driver failures from the baseline Wan
pipeline and prevents zero task success from being misclassified as lifecycle
failure.

The current two-pipeline workload is 16 environments, two DP ranks, one
eight-trajectory GRPO group per rank, five Wan diffusion steps per chunk, up to
256 primitive environment steps per trajectory, inclusive reward-filter bounds
covering the full binary reward range, video recording, streamed and retained
driver logs, switchable phase diagnostics, and ten linked collections and
actor updates per driver. Acceptance identity and terminal-event uniqueness
are policy-version-aware. The focused regression set at that checkpoint
reported 94 passing tests, and focused Ruff lint and formatting checks passed.
These are code-level results, not accelerator acceptance.

The intended scheduler interaction is now clarified as stage-aware rather than
unrestricted generation-demand rebalancing:

```text
A generation rank completes naturally
  -> A reports durable completion and releases that exact bundle
  -> B may start generation on the idle bundle while A's sibling continues
  -> A completes its collection, seals its batch, and requests actor training
  -> the higher-priority training request drains only overlapping B bundles
  -> A trains, offloads, and releases
  -> B's paused ranks receive restoration preference and resume
  -> the same cycle may later occur with A and B reversed
```

Generation demand may consume idle or completion-released bundles but must not,
by itself, evict another pipeline's active incomplete generation. A sealed-batch
fixed-stage request may preempt overlapping generation at a safe point. The
pending `ACTOR_TRAINING` request, not `completed == target` inferred by the
scheduler, is the authoritative readiness signal because reward resolution,
batch validation, and sealing occur between trajectory completion and training.
After the fixed stage, paused work should be restored before new collection
demand to avoid churn and starvation.

This corrects the original first-iteration acceptance choreography, which
deliberately submitted B generation demand during A's active chunk to force an
A-to-B-to-A pause. That choreography remains a lower-level safe-point recovery
test. Core registration now defaults legacy callers to `gap_ratio`, while RLinf
registers `fixed_stage_only`; its generation demand cannot select active donors,
fixed generation-priority policy sync waits for availability, actor training
retains higher-priority overlapping-bundle reclaim, and paused ranks are
preferred when resources return. The ten-iteration harness now waits for A's
completed rank before B demand and uses A training to interrupt B. This code is
covered by focused regressions but still requires a clean real-GPU run. No
completed two-driver generation proof, utilization improvement, or OpenSora
acceptance is recorded yet.

### 19.13 Five-iteration two-pipeline partial proof and timeout finding (2026-07-30)

The two-driver run `generation-proof-dormant-lifecycle-fix-1` used the
four-GPU disaggregated Wan workload and completed five full linked lifecycles
per pipeline before termination. Its artifact root is:

```text
/tmp/task8-generation-proof/generation-proof-dormant-lifecycle-fix-1
```

The single 3,600-second harness deadline started at run creation, before cold
model initialization. The run was created at 15:39:24 UTC, both drivers were
ready at 15:43:51, work was released at 15:48:09, and the harness terminated it
at 16:39:25 while lifecycle 6 was active. The terminal
`pair_failure.json` contains `timed out waiting for driver a result.json`.
This is an orchestration-policy timeout: no worker or scheduler failure was
observed before termination. In particular, the event stream contains zero
callback, world-model onload, diffusion, reward, and barrier-send failures.
Pipeline B emitted 1,232 events and committed 61 lifecycle-6 Wan chunks in the
last five minutes, demonstrating that the run was progressing rather than
deadlocked.

Completed aggregate evidence is:

- five completed iterations per pipeline, ten total;
- 16 trajectories and 66 transition records per sealed batch;
- 160 completed trajectories and 660 transition records total;
- ten sealed batches, ten actor-training calls, and twenty completed rank runs;
- approximately 295.1 MiB of logical actor batch data per iteration;
- approximately 3.2 completed trajectories per minute over completed work;
- six policy syncs per pipeline: initial version 0 and versions 1 through 5;
- five of ten optimizer updates with nonzero gradients.

The per-iteration learning evidence is:

| Pipeline | Iteration | recorded `success_once` | GRPO reward | Grad norm | Total loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 1 | 0/16 | 0.000000 | 0.000 | 0.000000 |
| A | 2 | 9/16 | 0.016071 | 34.166 | 0.003793 |
| A | 3 | 0/16 | 0.000000 | 0.000 | 0.000000 |
| A | 4 | 0/16 | 0.000000 | 0.000 | 0.000000 |
| A | 5 | 9/16 | 0.018204 | 79.137 | 0.006605 |
| B | 1 | 1/16 | 0.001248 | 34.707 | 0.004411 |
| B | 2 | 8/16 | 0.013477 | 0.000 | 0.000000 |
| B | 3 | 1/16 | 0.001230 | 19.050 | 0.002504 |
| B | 4 | 0/16 | 0.000000 | 0.000 | 0.000000 |
| B | 5 | 12/16 | 0.028958 | 25.182 | 0.002471 |

`success_once` is a recorded world-model metric, not independently verified
visual task success. Zero-gradient rows are valid completed GRPO optimizer
boundaries when the group supplies no usable relative advantage; they are not
lifecycle failures.

This run repeatedly proves collection, batch seal, actor-training acquisition,
optimizer completion, actor policy publication from `N` to `N+1`, rollout
policy synchronization, fixed-stage release, and next-lifecycle admission.
Both pipelines published and synchronized version 5 and opened lifecycle 6.
It also exercises the dormant lifecycle rule: an offloaded canonical rank may
remain `COMPLETED(N)` while the pipeline opens collection `N+1`, contributes no
current progress, and is prepared with exact `N+1` lifecycle and policy
identity before onload.

The proof boundary is intentionally narrower than T8 completion. The run has
no drain, continuation snapshot, or resume events, so it does not demonstrate
the required training-triggered safe-point interruption and recovery. It also
does not demonstrate ten completed iterations per pipeline, reference
equivalence, a material utilization gain, or OpenSora. Because the harness
terminated the drivers, it emitted no terminal `summary.json`, driver
`result.json`, or final acceptance report. The retained central
`core/events.jsonl`, driver stdout/stderr logs, and twenty videos provide
normalized execution evidence, but no raw reloadable `.pt` trajectory tensors
were saved.

Timeout handling must distinguish slow startup from a stalled workload. The
target harness policy is a 1,800-second startup deadline, a 1,200-second stall
deadline refreshed by meaningful lifecycle/model events, and a 14,400-second
overall safety ceiling. Until those independent clocks are implemented, the
next run should use `--timeout-s 10800`. This only prevents premature
termination; it does not relax any correctness, interruption, utilization, or
acceptance gate.

Post-fix regression baselines are 347 passed with two skipped tests for the
broad RLinf RLix suite and 129 passed with one skipped test for `rlix-core`.
Ruff and diff checks pass. T8 remains in progress and the definition of done
below is unchanged.

The separate root document
`../FUTURE_FUNGIBLE_WORK_AND_FULL_GPU_TRAINING_DESIGN.md` records deferred
logical-work/physical-slot separation and an all-four-GPU actor-training
experiment. Those proposals do not alter this plan's current pinned-rank
acceptance contract or definition of done.

The initial all-four-GPU FSDP experiment was implemented on 2026-07-30 as
separate `task8_four_rank_fsdp_acceptance.py`,
`task8_four_rank_fsdp_driver.py`, and
`task8_wan_disaggregated_four_rank_fsdp.yaml` files. The existing Task 8 E2E
files remain unchanged. The opt-in `completed_bundle_handoff` runtime policy
defaults to `retain_overlap`; only the new experiment selects
`release_before_training`, allowing both completed bundles to reach the other
pipeline before a fixed `[0,1,2,3]` actor-training request preempts them. The
new gate holds A after its first seal until B has entered real chunks on both
generation ranks; terminal validation requires drain/offload/restore evidence
from both B ranks and completed training events from all four A actor ranks. CPU
configuration/runtime regressions pass, but no four-rank GPU run has yet been
recorded, so this is implementation progress rather than T8 acceptance.

The first GPU attempt reached real generation but exposed a local GRPO routing
constraint before batch seal: with one rollout epoch, sixteen global
trajectories became four per actor rank, which is not divisible by
`group_size: 8`. The first corrective overlay used 32 environments, yielding
eight trajectories per actor rank, and its driver rejects partial local GRPO
groups before Ray/model startup.

The next GPU attempt passed that preflight and reached production batch seal.
It failed because the previous actor receipt contract compared every actor's
local transition contributors with the complete global collection contributor
tuple `(0, 1)`. Four-way FSDP routing does not replicate the whole collection
to every actor: environment rank 0 supplies actor ranks 0 and 1, while
environment rank 1 supplies actor ranks 2 and 3. The correct local receipts are
therefore `(0,)`, `(0,)`, `(1,)`, and `(1,)`, each for eight trajectories.

The seal contract now separates local evidence from aggregate validation:

- each actor reports only the generation ranks observed in its local shard and
  rejects any contributor outside the declared collection;
- lifecycle, policy version, and exact per-actor trajectory count remain
  mandatory on every receipt;
- the registered runtime requires the union of actor-local contributors to
  equal the complete collection contributor set; and
- the runtime validates topology-derived sender fanout. With `S` environment
  senders and `A` actor receivers, each sender must occur in
  `lcm(S, A) / S` receipts, which is two for this two-sender/four-actor layout.

This is not a relaxation of complete-batch validation. A missing sender,
foreign sender, duplicated route, partial actor shard, or mixed lifecycle or
policy still fails before advantage calculation and training. The focused
four-rank/acceptance regression set passes 136 tests, and the broad RLix suite
passes 253 tests with two skips. Four-rank GPU rerun evidence remains pending.

The subsequent `four-rank-fsdp-preempt-3` GPU run completed three full
collection/training iterations in both pipelines and began the fourth before
operator interruption. This established the four-rank harness as the active
successor to the older one-rank trainer acceptance variant. Its next workload
revision returns to 16 simultaneously resident environments while increasing
`rollout_epoch` to 4 and enabling `stop_rank_when_all_done`. Each of the two
generation ranks now owns exactly one eight-environment group per epoch. Four
epochs produce 32 trajectories per generation rank and 64 per sealed
collection; splitting each generation shard between two FSDP receivers gives
16 trajectories, or two complete GRPO groups, per actor rank. Every epoch uses
the same policy version and may finalize early only after all eight sticky local
completion bits are set. Training still starts only after all four epochs seal,
then reclaims GPUs 0-3 for four-rank FSDP.

### 19.14 Asynchronous CPU policy update and transition profiling (2026-08-03)

The expansion-time `lazy_versioned` experiment was removed after a real run
showed that a roughly 14-GiB rank update held `resize_infer()` for about 100
seconds and could serialize two ranks into about 200 seconds. The isolated
four-rank harness now selects `async_cpu_prefetch`. Each successful FSDP update
builds and promotes a complete CPU candidate, then launches updates for all
CPU-offloaded rollout ranks before the training stage releases its GPUs. The
launch is nonblocking, so another pipeline can acquire GPUs while host-side
updates continue. Before its next collection, the runner waits for exact
all-rank receipts; only then can generation demand be published. Expansion no
longer invokes the update service. Paused or resident ranks reject mutation.

The old fixed all-rank sync remains available. Four-rank acceptance fails
if a fixed policy-sync stage is acquired, if owner promotions do not cover
versions 1 through the final trained version, or if rollout commits do not
cover every version consumed by a later collection. Acceptance-only events
record candidate size/hash, promotion, receiver transaction, each bucket, and
the final receipt.

The harness also starts switchable stage-transition GPU profiling by default;
it can be disabled with `--no-gpu-profile`. It polls acceptance events but
queries `nvidia-smi` only for coalesced allocation/release, generation, seal,
prefetch, fixed-stage, and training transitions. Raw JSONL and a sparse
per-GPU peak summary are written below `<run-root>/gpu_profile/`. Sparse
snapshots are not reported as utilization-weighted GPU-seconds. Measurements
remain node-total and include unrelated GPU processes.

CPU regression gates at this checkpoint include 368 passing focused RLix,
elastic lifecycle, cache/service/coordinator/runtime/runner, Task 8 config,
instrumentation, and profiler tests with two opt-in Ray skips. The complete
framework-neutral core suite passes 129 tests with one optional skip, and Ruff
lint passes for both the modified RLinf files and core. The repository's
unchanged core baseline still has two files that `ruff format --check` would
reformat; modified RLinf files pass the formatting gate. At this checkpoint,
real GPU evidence for `async_cpu_prefetch` remained pending; section 19.15
records the subsequent partial run.

### 19.15 Native baseline and async CPU policy GPU checkpoint (2026-08-05)

The native comparison is now maintained and executed entirely in the original
RLinf checkout. Its resolved ten-iteration workload is
`/root/RLinf/tests/e2e_tests/embodied/wan_libero_spatial_grpo_openvlaoft_10_iteration.yaml`;
it uses the native runner and does not import the `_VLAMP` launcher or RLix
runtime. The recorded run at
`/root/original-rlinf-baseline/native-wan-10-1/` completed ten of ten updates in
1:43:25 (about 5.80 updates/hour), with mean generation 321.43 seconds, actor
training 283.19 seconds, and native weight synchronization 15.94 seconds. It
produced 640 trajectories and 80 videos. Mean logged rollout reward was
0.007691 and mean `success_once` was 0.328125. Iteration 9 had zero return but
trained normally under reward filter `[0, 5]`; no OOM, NCCL, or runtime failure
was found.

The corresponding RLix artifact is
`/root/task8-four-rank-fsdp/four-rank-fsdp-async-cpu-prefetch-1/`. Each pipeline
completed six updates and began collection for iteration 7 before manual
interruption. The 12 completed aggregate updates took approximately 2:22:36
through the twelfth policy-ready point (about 5.05 updates/hour). The run
exercised 24 rank completions, 12 complete batch seals, 12 four-rank FSDP
updates, 22 rank drain/snapshot cycles, all-rank policy receipts and commits,
and 15 early-finalized rank epochs out of 96. Completed training batches kept
the required shapes: rewards `[32, 16, 8]`, dones `[33, 16, 8]`, and loss mask
`[32, 16, 8]`. Mean logged rollout reward over those updates was 0.007238 and
the success-marker rate was approximately 0.319. Driver A wrote 52 videos,
including partial iteration-7 output, and driver B wrote 48 videos for its six
completed iterations.

Mean scheduler wait for a generation grant was 137.8 seconds. Mean
grant-to-last-rank collection wall time was 662.8 seconds, but that interval
includes time paused for the other pipeline's training and is not active GPU
generation time. Unpreempted RLix generation examples of 325.5 and 340.4
seconds are close to the native 321.43-second mean. Early finalization exited
at a mean step of approximately 117 versus the 256-step limit and is estimated
to have avoided about 8.5 percent of aggregate environment steps.

The acceptance stream contains 64,555 events and is approximately 259 MB,
which also identifies instrumentation volume as an experiment variable. GPU
transition snapshots observed approximately 47.2--47.6 GiB peak memory and
100-percent peak utilization on each 80-GiB GPU. They are sparse, node-level
snapshots that include unrelated processes and therefore cannot establish a
utilization time integral.

The selected policy cache was 15,082,474,368 bytes in 143 buckets. Mean
training-through-cache time was 346.79 seconds and mean asynchronous CPU policy
delivery was 119.93 seconds, compared with native training plus synchronization
of 283.19 + 15.94 = 299.13 seconds. Current tracing supports only an inferred
decomposition: about 283.19 seconds of common optimizer work, 2.33 seconds of
collective materialization, a 60.02-second rank-0 CPU packaging tail, and a
1.25-second completion/offload tail. The collective value is inferred from the
non-owner cache marker and is not directly bracketed by an optimizer-complete
event.

These measurements identify CPU packing, repeated model-sized copies, byte
conversion, checksumming, and Ray delivery as the immediate optimization
target. Next implementation work should add exact markers around optimizer
completion, materialization, device-to-host copying, bucket assembly, hashing,
receiver apply, and commit; preallocate shared or pinned bucket storage; avoid
`torch.cat` and `.numpy().tobytes()` model-sized copies; and evaluate shared
immutable handles plus shard-aware cache ownership. Atomic promotion,
exact-version receipts, and fail-closed admission remain mandatory.

This checkpoint provides real-GPU evidence that the mechanism executes, but it
does not satisfy Task 8: `pair_failure.json` records a manual
`KeyboardInterrupt`, `status: failed`, and `task8_accepted: false`; neither
pipeline completed all ten declared iterations. The Definition of Done below
remains unchanged.

### 19.16 Queue-driven free-bundle acquisition (2026-08-06)

The generation-proof harness no longer uses first-iteration training barriers
to manufacture a deterministic pause/resume sequence. The prior sequence had
three control-plane distortions:

- B did not publish generation demand until after A's first completed-rank
  event reached the external orchestrator;
- B could seal a batch but could not publish its first training request until
  the orchestrator released `allow_b_training`; and
- both drivers waited at `both_training_completed` after their first updates,
  preventing either from independently publishing its next policy-ready
  collection demand.

The revised harness keeps deterministic cold initialization and initially
grants A generation so ownership transfer has an unambiguous source. As soon
as A's first generation allocation is observed, however, the orchestrator
releases B collection and waits for `b_generation_requested` *before* waiting
for any A rank completion. B's request is therefore already pending in the
scheduler when A releases a bundle. Scheduler wakeup and allocation commit no
longer depend on an event round trip through the acceptance process.

After collection, both drivers now request training immediately. Fixed-stage
priority, GPU exclusivity, shrink-before-expand, safe drain, and fair request
ordering are enforced by the production scheduler rather than acceptance
gates. After training, each driver may await its own policy receipt and request
its next collection without rendezvousing with the peer pipeline.

The parent acceptance process still waits for observed gates such as batch
seal, training start, training completion, and both-training completion. These
are passive assertions over durable events: no driver waits on them and they
cannot delay resource requests. Legacy gate names remain in the control actor
for the dependency-light acceptance-control smoke and old artifact decoding;
their presence does not imply that the real generation-proof drivers consume
them.

This change intentionally makes exact first-iteration interleaving a scheduler
outcome rather than a scripted outcome. It does not relax batch cardinality,
policy-version, lifecycle, physical non-residency, exclusive ownership,
snapshot/restore, or asynchronous policy-receipt validation. The four-rank
proof continues to require its declared preemption/resume evidence; failure to
produce that evidence is reported as an acceptance failure instead of being
hidden by resource-admission barriers.

## 20. Definition of done

T8 and the T0-T8 project are complete only when all of the following are true:

- Two independent OS driver processes connect to the same detached core
  control-plane and scheduler actors.
- They receive distinct valid pipeline IDs/namespaces and have no RLinf-owned
  actor or channel name collision.
- Both register at least two exact canonical actor-infer bundles whose
  candidate physical GPU IDs intentionally overlap across pipelines.
- Pipeline A initially activates at least two whole bundles, and one rank
  completes its assigned trajectories naturally while its sibling continues.
- A's completed rank publishes durable completion and releases its exact bundle
  without a pause token; callback success and verified non-residency precede
  core release commit.
- Pipeline B acquires that completion-released bundle with no logical or
  physical ownership overlap and performs useful real-model work on it.
- B's generation request does not evict A's active incomplete sibling.
- A's remaining rank completes, A seals a complete batch, and only its pending
  fixed actor-training request triggers shrink of overlapping B generation.
- Affected B chunks commit exactly once before matching snapshot/offload and
  callback success; non-overlapping B siblings remain productive.
- After A training and fixed release, B expands the same paused canonical rank
  on its exact registered bundle, validates/restores its same-rank snapshot,
  and dispatches the retained next transition exactly once.
- Paused resumable work is preferred over a just-trained pipeline's new
  collection, and equal-priority training requests have deterministic fair
  ordering.
- Interrupted B matches its uninterrupted reference for transition order,
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

## 18. Receipt-mode performance runs

The four-rank FSDP harness now separates correctness evidence from performance
observation with independent switches:

- detailed worker acceptance events default on and remain fail closed;
- phase diagnostics and sparse GPU profiling can be disabled independently;
- residency validation selects `deep`, `receipt`, or benchmark-only `off`;
- continuation snapshot validation selects `deep` or `receipt`;
- video capture remains controlled by the environment video configuration and
  is not disabled by any of these switches.

Receipt residency mode records synchronized movement generation, worker rank,
lifecycle, policy version, destination, and optional byte accounting. It skips
Python parameter/buffer traversal but rejects stale identity, wrong destination,
and unsynchronized evidence. Current backends do not all expose moved-byte
counts, so `moved_bytes=None` is explicit rather than falsely claiming complete
byte accounting.

Receipt snapshot mode deeply validates once at construction, then stores an
opaque receipt with the private immutable snapshot. Redundant continuation
fields remain in the schema but are empty; restore reconstructs them from the
canonical pending bootstrap. This removes repeated recursive validation during
offload, prepare, and restore without changing transition identity, partial
rollout contents, early-termination masks, or video metrics.

When detailed worker instrumentation is disabled, the harness retains only the
small runner/orchestrator event set needed for deterministic startup. It skips
the detailed lifecycle-evidence validator and labels the result accordingly.
Such a run must not be reported as Task 8 acceptance proof.

The four-rank performance harness has no wall-clock timeout by default. Its
orchestrator, child-driver filesystem rendezvous, acceptance-control gates, and
OS process waits all propagate an explicit unbounded wait. `--timeout-s` remains
available as an opt-in operator safeguard. Signal/error cleanup retains short
bounded termination waits so shutdown cannot become permanently stuck.
