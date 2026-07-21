# Task 6 Detailed Implementation Plan: Placement and Configuration

## 1. Status and source of truth

Status: completed on 2026-07-21. T0-T6 are complete; T7 runner integration and
T8 two-pipeline accelerator acceptance remain pending.

This plan refines the T6 requirements in these documents, in descending order
of authority:

1. `../VLA_COMPATIBILITY_DESIGN.md`
2. `../VLA_COMPATIBILITY_IMPLEMENTATION_PLAN.md`
3. `RLIX_ELASTIC_VLA_ROLLOUT_IMPLEMENTATION_PLAN.md`
4. The completed detailed plans for T1-T5

The implementation baseline at the start of T6 was also authoritative where
those documents described older behavior. In particular:

- T3 already requires an explicit `actor_infer` DP mapping with `tp_size=1`;
- the core had no cluster name capable of representing the documented
  fixed initialization or fixed evaluation device unions;
- T4 already consumes rank eligibility from progress reports;
- T5 already provides `RLixStageController` and the named resize coordinator;
- `HybridComponentPlacement` resolves Hydra placement strings into concrete
  `Placement` objects;
- the embodied entrypoint launched environment and rollout actors
  without an explicit Ray `max_concurrency` value; and
- no `rlinf.scheduler.rlix.placement` or `.validation` module existed.

T6 owns configuration, topology conversion, worker launch concurrency, and the
production registration bootstrap defined by T5. It creates the named
coordinator, registers and admits the pipeline, and hands an inactive runtime
context to the runner. T7 owns allocation requests, collection configuration,
stage ordering, progress/release calls, and training-loop behavior.

Implementation progress and inherited foundation:

- [x] T0 fixed allocation and fail-closed scheduler transactions are complete.
- [x] T1 versioned Wan/OpenSora continuation state is complete for the scoped
  synchronous same-actor contract.
- [x] T2 local paired safe-point lifecycle is MVP-complete.
- [x] T3 explicit composite bundles, canonical ownership, fixed `policy_sync`,
  and generic fixed-cluster scheduling are complete.
- [x] T4 rank eligibility, progress, and rank-specific release are complete.
- [x] T5 named resize coordination, policy-sync serialization, and bootstrap
  construction primitives are complete.
- [x] Add fixed-only `initialization` and `evaluation` core cluster constants
  with protocol, parsing, allocation, validation, and no-callback tests.
- [x] Add the opt-in RLix Hydra schema, disabled default, and complete
  supported-mode rejection matrix.
- [x] Project resolved actor, rollout, and environment `Placement` records into
  immutable, rank-stable one-GPU ownership records.
- [x] Validate the live single-GPU-node Ray topology and its exact agreement
  with RLinf and `rlix-core` GPU numbering.
- [x] Build and self-check the five exact flat mappings, explicit generation
  bundles, TP sizes, and fixed/elastic allocation policies.
- [x] Cover valid collocated/disaggregated mappings and every rank, topology,
  bundle-width, overlap, backend, wrapper, and offload rejection.
- [x] Reuse the exact preflighted placements for worker launch without parsing
  Hydra strings or resolving placement a second time.
- [x] Launch environment and rollout actors with explicit
  `max_concurrency >= 2` only in enabled mode.
- [x] Construct the named coordinator, register the same namespace/topology,
  admit the pipeline, and return an inactive registered runtime in T5 order.
- [x] Implement reverse-order bootstrap cleanup for controller, registration,
  admission, and partial worker-launch failures without replacing the primary
  error.
- [x] Wire the embodied entrypoint and optional runner handoff while preserving
  the disabled standalone path and making no T7 allocation request.
- [x] Add the elastic config group and a dedicated Wan example without changing
  existing standalone examples.
- [x] Run focused T1-T6, core auxiliary-cluster, lint, format, compilation, and
  dependency-boundary verification and record exact results here.
- [x] Mark T6 complete in the canonical design/task documents only after every
  section 17 acceptance criterion passes; T7/T8 remain pending.
- [x] Add the optional three-GPU T6 bootstrap and real rollout/environment
  model-loading smoke, and cover Wan cold offload before the first explicit
  reset (33 focused tests passed).
- [x] Rerun the T6-focused real-model smoke through `result.json` after the
  pristine Wan queue offload fix; it passed on three RTX 4090 GPUs.

Check an item only when its production changes and focused tests are complete.
Record completion dates, exact test counts, skips, and any intentional
deviation in this section rather than inferring progress from later prose.

Completion evidence on 2026-07-21: the complete `rlix-core` suite passed with
107 tests and 1 skip. The focused T1-T6 RLinf suite, including existing
placement regressions, passed with 236 tests and 1 skip. Ruff lint and
changed-file format checks, compilation, whitespace checks, standalone Hydra
composition/pure validation, and an import test with `rlix_core` deliberately
unavailable all passed. A repository-wide core format check still reports four
unrelated pre-existing files; no unrelated formatting was changed. The
optional T6 bootstrap/model-loading smoke later passed on real accelerators as
recorded in section 13.7; T7 integration and T8 two-pipeline accelerator
acceptance remain explicitly pending.

## 2. Required outcome

When `rlix.enabled` is true, RLinf must validate the supported elastic VLA mode
and derive the exact RLix registration topology from the same resolved
placements later used to launch the Ray workers.

For a two-rank disaggregated placement:

```text
rollout rank 0 -> GPU 0       env rank 0 -> GPU 2
rollout rank 1 -> GPU 1       env rank 1 -> GPU 3

actor_infer DP bundles = {0: [0, 2], 1: [1, 3]}
actor_infer flat mapping = [0, 1, 2, 3]
```

For a collocated placement:

```text
rollout rank 0 -> GPU 0       env rank 0 -> GPU 0
rollout rank 1 -> GPU 1       env rank 1 -> GPU 1

actor_infer DP bundles = {0: [0], 1: [1]}
actor_infer flat mapping = [0, 1]
```

The produced registration data must contain:

```python
cluster_tp_configs = {
    "initialization": 1,
    "actor_train": 1,
    "actor_infer": 1,
    "policy_sync": 1,
    "evaluation": 1,
}
cluster_allocation_policies = {
    "initialization": "fixed",
    "actor_train": "fixed",
    "actor_infer": "elastic",
    "policy_sync": "fixed",
    "evaluation": "fixed",
}
cluster_dp_device_mappings = {
    "actor_infer": {rank: bundle, ...},
}
```

The fixed mappings are the actor/rollout/environment union for
`initialization`, actor GPUs for `actor_train`, actor-plus-rollout GPUs for
`policy_sync`, and rollout-plus-environment GPUs for `evaluation`. All output
ordering is deterministic so registration, diagnostics, and tests are
reproducible.

Unsupported configurations must fail before worker launch and therefore
before model or environment initialization.

After lightweight worker actors exist, T6 must use the T5 ordering exactly:

```text
create named coordinator with exact ranked handles
register the pipeline with the same namespace and T6 placement payload
admit the pipeline
hand the inactive registered runtime to T7
```

No model initialization or stage allocation occurs as part of this bootstrap.

## 3. Scope boundaries

### 3.1 Supported by T6

- one Ray node containing NVIDIA CUDA GPUs;
- synchronous embodied training through
  `examples/embodiment/train_embodied_agent.py`;
- Hugging Face `MultiStepRolloutWorker` rollout;
- an FSDP actor whose world size and resolved placement remain fixed;
- Wan or OpenSora world-model training environments covered by T1/T2;
- one process and exactly one assigned GPU for each actor, rollout, and
  environment rank;
- equal, contiguous rollout and environment rank sets;
- collocated rollout/environment rank pairs;
- disaggregated rollout/environment rank pairs;
- one uniform bundle width per pipeline;
- explicit elastic-worker concurrency of at least two;
- fixed `initialization` and `evaluation` auxiliary cluster registration;
- named coordinator construction plus pipeline registration/admission; and
- a disabled mode that preserves the existing entrypoint behavior.

### 3.2 Explicitly not implemented by T6

- fixed initialization, policy-sync, training, or evaluation requests;
- elastic collection requests and progress-report forwarding;
- batch sealing or policy-version advancement;
- TP, model PP, pipeline rollout stages, or actor resizing;
- asynchronous or decoupled embodied runners;
- multi-node GPU-ID translation or environment migration;
- heterogeneous bundle widths within a pipeline;
- GPU reward workers;
- selective rollout weight synchronization; or
- a real GPU utilization or recovery claim.

These remain T7 or T8 work. T6 may expose immutable data and entrypoint wiring
that T7 consumes, but it must not partially implement stage ownership. T6 does
own rollback of a coordinator/registration bootstrap that fails before the
runner takes ownership.

T6 does make the minimal `rlix-core` protocol extension for the two fixed
auxiliary cluster names. It does not add a priority, planner policy, callback,
or resize semantic: both use the generic fixed-cluster path already completed
in T0/T3.

### 3.3 Fail-closed boundary

Validation is opt-in:

```text
rlix.enabled absent or false -> preserve the existing standalone path
rlix.enabled true            -> enforce every T6 requirement
```

Once enabled, missing fields are not silently interpreted as supported unless
the default is part of the documented T6 schema. Invalid placement must not
fall back to a flat legacy RLix mapping or a fixed rollout allocation. No Ray
worker may be launched after T6 preflight fails.

## 4. Relationship to T1-T5

### 4.1 T1 constrains the environment mode

T6 validates before launch the configuration gates currently checked late by
`EnvWorker._validate_snapshot_capability()`: one pipeline stage, fixed reset
state IDs, no data-collection wrapper, no online LeRobot, no RLT worker state,
and no history-buffer reward state.

T6 does not duplicate runtime snapshot validation. Worker checks remain the
authoritative defense against state that changes after launch.

### 4.2 T2 constrains worker topology and concurrency

Each elastic DP rank is a stable same-ranked environment/rollout pair. The two
actors need asynchronous control calls to interleave at documented yield
points while generation is active, so both groups are launched with
`max_concurrency >= 2` only in enabled mode.

### 4.3 T3 defines registration semantics

Bundle width is physical scheduling cost, never tensor parallelism. T6 always
registers `actor_infer` with `tp_size=1` plus an explicit DP mapping. The
bundle union must exactly equal the flat generation mapping.

T3's generic fixed-cluster planning is reused for `initialization` and
`evaluation`. These names are not generation clusters, never carry DP-rank
ownership, and never invoke `resize_infer`. `policy_sync` remains the only
fixed auxiliary cluster allowed at `Priority.GENERATION`.

### 4.4 T4 defines rank identity

Placement ranks become the canonical ranks used by progress and selective
release. T6 may not renumber a placement based on GPU order. Rank `n` is paired
only with rank `n`.

### 4.5 T5 defines callback identity

The T5 controller expects equal contiguous worker rank sets and
`max_concurrency >= 2`. T6 makes those properties true in production,
constructs the controller with the allocated pipeline ID and canonical
namespace, then registers that same identity. T7 uses the already registered
controller/runtime for collection and stage operations.

## 5. Pre-implementation gaps

### 5.1 No placement conversion layer

`HybridComponentPlacement` exposed strategies, hardware-rank summaries, and
world sizes, but did not produce RLix cluster mappings. Building mappings
from `cluster.component_placement` strings would duplicate parser semantics and
could diverge from actual worker placement.

### 5.2 Local hardware IDs are not globally valid on multiple nodes

`Placement.local_hardware_ranks` is node-local. It is a valid RLix GPU ID only
when the live Ray cluster has exactly one GPU-bearing node, not merely when
Hydra says `cluster.num_nodes: 1`. A later multi-node design must define a
canonical translation shared with `rlix-core`'s resource topology rather than
flattening local IDs speculatively.

### 5.3 Worker launch lacked control interleaving

The entrypoint called `launch(...)` for environment and rollout workers without
`max_concurrency`. T5 validated a configured value but did not control how
production actors were launched.

### 5.4 Configuration lacked an RLix schema

`validate_cfg()` normalized embodied settings but had no `rlix` section or
enabled-mode validation. Some unsupported values were discovered only inside
worker initialization, after expensive actors existed.

### 5.5 The existing world-model example was not an elastic contract

The original Wan example collocated actor, environment, and rollout and had the
model offload fields needed by the design, but did not opt into RLix or define
elastic timeouts/concurrency. T6 retains that standalone example and adds a
dedicated opt-in primary config.

## 6. Core invariants and design decisions

### 6.1 Resolved placement is the only topology source

Call each component strategy's `get_placement(cluster)` and derive mappings
from the returned `Placement` records. Do not parse Hydra placement text, use
`CUDA_VISIBLE_DEVICES`, or use `ComponentPlacement.get_hardware_ranks()` as a
substitute for per-process ownership.

Resolve each strategy once during enabled-mode preflight. The exact resolved
placements used to build the plan must also be passed to worker launch; avoid a
second resolution whose sorting or future dynamic behavior could differ.

If the existing launch API cannot consume a pre-resolved placement sequence,
add a small immutable placement strategy wrapper rather than rebuilding the
mapping from config.

### 6.2 Rank identity is stable

For every component, resolved placement ranks must be non-negative, unique,
and contiguous from zero. The plan stores mappings keyed by those ranks.

Rollout rank `r` pairs only with environment rank `r`. Pairing by list
position, local GPU, node rank, or visible-device order is forbidden.

### 6.3 One worker owns one physical GPU

Each actor, rollout, and environment `Placement` must have exactly one
`local_hardware_rank`. Its accelerator type must be `AcceleratorType.NV_GPU`,
and all placements must have `cluster_node_rank == 0` in the supported
single-node topology.

Before registration, inspect the live Ray topology and require exactly one
alive node with a positive `GPU` resource. Its GPU count must equal the RLinf
node accelerator count used to validate local IDs and the count observed by
the `rlix-core` resource manager. This makes local GPU `n` the same scheduler
GPU `n`; config-only single-node validation is insufficient.

`visible_accelerators` is an execution detail and is not accepted as ownership
evidence.

### 6.4 A bundle is the ordered union of a rank pair

Construct a bundle in component order, rollout then environment, and remove a
duplicate only when both placements name the same GPU:

```python
bundle = list(dict.fromkeys([rollout_gpu, env_gpu]))
```

Thus a supported bundle has width one or two. Bundle order is deterministic,
but ownership comparisons use set equality.

### 6.5 Generation bundles are disjoint and uniform

No GPU may appear in two `actor_infer` bundles. All bundles in one pipeline
must have the same width. A placement that mixes collocated and disaggregated
pairs is rejected even if each pair is individually valid.

Cross-cluster overlap is expected: actor GPUs may also occur in
`policy_sync`, and in a fully collocated configuration may occur in
`actor_infer`. RLix stage allocation prevents simultaneous ownership; T7 must
request stages in the designed order.

### 6.6 Flat mappings are canonical unions

Build every flat mapping as sorted unique GPU IDs:

```text
initialization = union(actor placements, rollout placements, env placements)
actor_train = union(actor placements)
actor_infer = union(all explicit bundles)
policy_sync = union(actor placements, rollout placements)
evaluation = union(rollout placements, env placements)
```

The explicit generation mapping retains rank and bundle order. Its flattened
set and length must equal `actor_infer` exactly.

### 6.7 Configuration validation has two phases

Pure config validation belongs in `validate_cfg()` and catches semantic modes
that require no live cluster. Resolved topology validation runs after
constructing `Cluster` and `HybridComponentPlacement`, but before launching any
worker.

Both phases raise `ValueError` or a dedicated `RLixConfigurationError` with the
config path and rejected value. Do not rely on Python `assert`, which can be
disabled and often yields poor operator diagnostics.

### 6.8 Disabled mode is behaviorally unchanged

When RLix is disabled:

- do not import `rlix_core` through the normal entrypoint path;
- do not resolve placement an extra time;
- do not alter worker concurrency;
- do not add a coordinator or controller argument to the runner; and
- preserve existing config defaults and launch order.

### 6.9 Registration identity and ordering are atomic bootstrap state

T6 allocates the pipeline ID through `ControlPlane.allocate_pipeline_id()` and
derives the callback namespace with `get_pipeline_namespace(pipeline_id)`.
Neither value is accepted from independent user configuration.

After worker launch, bootstrap ordering is:

1. create `RLixStageController` with exact environment/rollout handles;
2. register the T6 payload with the controller's namespace;
3. admit the registered pipeline; and
4. return one runtime context containing the control plane, scheduler handle,
   controller, pipeline identity, and immutable placement plan.

After a registration call is attempted, any registration or admission failure
must best-effort unregister the pipeline and then close the controller. This
also covers an ambiguous transport failure after the scheduler committed the
registration. Unregistering an absent pipeline is idempotent in the current
core. Preserve the primary error and attach cleanup failures as notes. No
allocation may be requested during T6.

## 7. Configuration contract

### 7.1 Minimal schema

Add an opt-in Hydra config group, for example
`examples/embodiment/config/rlix/elastic_vla.yaml`:

```yaml
enabled: true
rollout_allocation_policy: elastic
rollout_safe_point: world_model_chunk
progress_unit: trajectories
worker_max_concurrency: 2
operation_timeout_s: 300.0
enable_gpu_tracing: false
```

The entrypoint config includes it explicitly with:

```yaml
defaults:
  - rlix/elastic_vla@rlix
```

Do not add RLix to every embodied config's defaults. Existing examples remain
standalone unless they opt in.

Pipeline IDs and coordinator namespaces should not be user-composed in this
group. T6 allocates the ID and uses `get_pipeline_namespace(pipeline_id)` so
the registration and T5 actor lookup cannot disagree.

### 7.2 Normalization

When a root `rlix` section is absent, normalize only
`cfg.rlix.enabled = false`. When enabled, fill documented defaults inside an
`open_dict(cfg)` block, validate types before numeric comparisons, and reject
unknown contract values.

Required exact values for this milestone are:

```text
rollout_allocation_policy == "elastic"
rollout_safe_point == "world_model_chunk"
progress_unit == "trajectories"
worker_max_concurrency >= 2
operation_timeout_s > 0
enable_gpu_tracing is bool
```

### 7.3 Pure configuration rejection matrix

Enabled mode rejects at least:

- `cluster.num_nodes != 1`;
- the live Ray cluster has zero or more than one GPU-bearing node;
- `runner.task_type != "embodied"` or evaluation-only mode;
- an async entrypoint/mode;
- `runner.enable_decoupled_mode`;
- `runner.use_training_pipeline`;
- `runner.overlap_env_bootstrap`;
- `runner.weight_sync_interval != 1`;
- `actor.training_backend != "fsdp"`;
- `rollout.generation_backend != "huggingface"`;
- `rollout.pipeline_stage_num != 1`;
- actor or rollout tensor/model pipeline parallel sizes other than one when
  present;
- `actor.enable_offload`, `rollout.enable_offload`, or
  `env.train.enable_offload` not true;
- evaluation enabled while `env.eval.enable_offload` is not true;
- `env.train.enable_init_offload` explicitly false;
- a non-Wan/non-OpenSora training environment;
- `env.train.use_fixed_reset_state_ids` not true;
- enabled environment data collection;
- online LeRobot, RLT, or history-buffer reward state;
- a GPU/external reward worker (`reward.use_reward_model` unless an explicitly
  CPU-only supported path is designed later); and
- training bootstrap overlap or another stateful wrapper not covered by T1.

The implementation must inspect actual config paths used by current RLinf
rather than introduce aliases such as `tensor_parallel_size` if the repository
uses `tensor_model_parallel_size`.

## 8. Placement data model

### 8.1 Pure resolved record

Add an internal immutable projection rather than retaining mutable
`Placement` objects in the public plan:

```python
@dataclass(frozen=True, slots=True)
class ResolvedGPUWorker:
    rank: int
    cluster_node_rank: int
    local_gpu: int
    accelerator_type: AcceleratorType
```

Conversion validates every field and rejects booleans where integers are
required.

### 8.2 Registration plan

Add a frozen `RLixPlacementPlan` whose internal values are tuples:

```python
@dataclass(frozen=True, slots=True)
class RLixPlacementPlan:
    actor_workers: tuple[ResolvedGPUWorker, ...]
    rollout_workers: tuple[ResolvedGPUWorker, ...]
    env_workers: tuple[ResolvedGPUWorker, ...]
    initialization_devices: tuple[int, ...]
    actor_train_devices: tuple[int, ...]
    actor_infer_devices: tuple[int, ...]
    policy_sync_devices: tuple[int, ...]
    evaluation_devices: tuple[int, ...]
    actor_infer_bundles: tuple[tuple[int, tuple[int, ...]], ...]
```

Expose a method that returns fresh dictionaries/lists for the topology fields
of `ControlPlane.register_pipeline(...)`. Pipeline identity and Ray namespace
come from the runtime bootstrap, not the placement plan. Callers must not be
able to mutate the canonical plan through a returned payload.

The method sets all TP sizes and policies explicitly. It should import
`rlix_core` constants lazily so importing normal RLinf placement utilities does
not make `rlix-core` mandatory for standalone users.

### 8.3 Pre-resolved launch strategy

If needed, add a `ResolvedPlacementStrategy` that stores a tuple of validated
`Placement` copies and returns fresh copies from `get_placement()`. It must not
resort ranks, change visibility, or retain caller-owned mutable lists.

The entrypoint uses these exact actor, rollout, and environment strategies for
launch after preflight succeeds.

## 9. Placement conversion algorithm

### 9.1 Resolve components

After creating the real `Cluster` and `HybridComponentPlacement`:

```text
actor placements   = actor strategy.get_placement(cluster)
rollout placements = rollout strategy.get_placement(cluster)
env placements     = env strategy.get_placement(cluster)
```

Resolve with the same `isolate_accelerator` value used by launch. Reject node
placement or missing accelerator resources.

### 9.2 Validate component ranks

For each component:

1. validate non-empty records;
2. validate unique contiguous ranks;
3. validate one live GPU node, node zero, and NVIDIA type;
4. validate exactly one local hardware rank per process;
5. validate that the local rank is within the node's accelerator count; and
6. build `rank -> GPU` without reordering identity.

### 9.3 Pair rollout and environment

Require identical rank keys. For each sorted rank, build the ordered unique
bundle and then validate:

- width is one or two;
- the bundle contains both declared placements;
- bundle width equals the first bundle's width; and
- its GPU set is disjoint from every earlier rank bundle.

The actor world size need not equal rollout world size, but the actor mapping
must be a non-empty fixed FSDP GPU set with no duplicate worker ownership.

### 9.4 Build and self-check registration data

Construct the five flat mappings, explicit bundles, policies, and TP sizes.
Before returning, call a local structural validator. When `rlix_core` is
installed, also pass the payload through
`validate_register_pipeline(RegisterValidationInput(...))` in tests. T6 uses
that same checked payload for runtime registration; stage allocation remains
T7.

### 9.5 Registered runtime context

Add an owner-scoped runtime object, for example `RegisteredRLixPipeline`, that
contains the `ControlPlane`, admitted scheduler handle, `RLixStageController`,
pipeline ID, namespace, and `RLixPlacementPlan`. Construction follows section
6.9 and returns only after admission succeeds.

Before T7 requests any allocation, T6 bootstrap cleanup is:

```text
unregister pipeline
close coordinator
```

After T7 adopts the runtime, T7 must first release or finish allocations and
then invoke that same unregister-before-controller-close ordering.

## 10. Validation implementation

### 10.1 New module boundary

Add `rlinf/scheduler/rlix/validation.py` for enabled-mode config and plan
validation. Keep functions narrow:

```python
normalize_rlix_config(cfg) -> None
validate_elastic_vla_config(cfg) -> None
validate_elastic_vla_placement(plan, cluster) -> None
```

Pure config validation must not import Ray or initialize a cluster.

### 10.2 Ordering

Run validation in this order for deterministic errors:

1. RLix schema and scalar types;
2. entrypoint/runner mode;
3. actor and rollout backends/parallelism;
4. environment snapshot capability gates;
5. offload and weight-sync requirements;
6. reward/wrapper exclusions;
7. resolved node and accelerator topology;
8. rank pairing and one-GPU ownership;
9. bundle disjointness/uniformity; and
10. registration payload self-check.

The core protocol check must also prove `initialization` and `evaluation` are
known GPU clusters, default to or explicitly require fixed allocation, reject
explicit DP bundles, and are accepted only through the generic non-generation
fixed path. T7 requests both at `Priority.INITIALIZATION`; evaluation reuses
that existing highest-priority lifecycle barrier rather than adding a new
priority tier.

Only the first failure is raised. Tests should assert stable message fragments,
not entire messages.

### 10.3 Defense in depth

Do not remove existing worker-side T1/T2 validation. Configuration can be
mutated between validation and actor initialization, and runtime objects can
violate capabilities not visible in Hydra.

## 11. Embodied entrypoint wiring

### 11.1 Preflight before launch

Refactor the entrypoint into testable helpers instead of placing all logic in
the Hydra-decorated `main()`:

```text
validated cfg
    -> construct Cluster
    -> construct HybridComponentPlacement
    -> if enabled: resolve and validate RLixPlacementPlan
    -> launch workers using the validated resolved strategies
```

No actor, rollout, environment, or reward worker exists when placement
preflight raises.

### 11.2 Worker concurrency

In enabled mode pass `cfg.rlix.worker_max_concurrency` to both:

```python
MultiStepRolloutWorker.create_group(cfg).launch(..., max_concurrency=value)
EnvWorker.create_group(cfg).launch(..., max_concurrency=value)
```

Do not alter actor concurrency. Do not pass the value in disabled mode.

The worker method bodies remain non-interleavable except at existing async
yield points; Ray concurrency is permission for T2/T5 control traffic, not
permission to parallelize diffusion or model mutation.

### 11.3 Registration bootstrap and T7 handoff

After environment and rollout groups are launched, create the registered
runtime context using their exact handles and the preflighted placement plan.
Pass that context to `EmbodiedRunner` only when enabled, using an optional
keyword that defaults to `None`.

The T6 constructor change is a mechanical ownership handoff only: it may store
the context but must not request allocations or alter the training loop. T7
will use it for fixed initialization, policy sync, elastic collection,
training, progress/release, and final cleanup. Until T7 is implemented, the
enabled configuration is integration-incomplete and must not be advertised as
a runnable elastic training mode.

## 12. Compatibility and failure semantics

### 12.1 Standalone compatibility

Existing configs with no `rlix` section must produce the same worker classes,
placements, concurrency options, and runner arguments as before T6.

### 12.2 Invalid enabled mode

Any invalid semantic or topology property raises before worker launch. The
error names the component, rank, config path, or GPU set responsible, and no
actors need cleanup.

Coordinator creation, registration, or admission failures occur after worker
launch but before model/environment initialization. The entrypoint must close
the coordinator, best-effort unregister after any registration attempt, and
close all newly launched worker groups while preserving the original failure.

### 12.3 Resolution failure

Preserve the original placement exception as the cause and add RLix context:

```text
failed to resolve RLix rollout placement: <original error>
```

Do not retry with another placement parser or `get_hardware_ranks()`.

### 12.4 Optional dependency boundary

Standalone imports of `rlinf.config`, `rlinf.scheduler`, and the embodied
entrypoint must remain usable without `rlix_core`. Import T6 registration
constants and validators only inside enabled-mode functions.

## 13. Test plan

### 13.1 Pure placement conversion

Add `tests/unit_tests/test_rlix_placement.py` with lightweight `Placement`
fixtures. Cover:

- one-rank and multi-rank collocated mappings;
- one-rank and multi-rank disaggregated mappings;
- deterministic mapping and bundle ordering;
- initialization, actor, generation, policy-sync, and evaluation unions;
- all TP sizes equal one and exact fixed/elastic policies;
- returned registration payloads do not mutate the plan; and
- the generated payload passes `rlix-core` registration validation.

### 13.2 Placement rejection matrix

Cover each failure independently:

- empty component placement;
- duplicate, missing, negative, boolean, or non-contiguous ranks;
- unequal environment/rollout rank sets;
- zero or multiple GPUs assigned to a process;
- non-NVIDIA accelerator;
- node rank other than zero;
- zero or multiple live Ray GPU-bearing nodes despite single-node config;
- local GPU outside the node range;
- two DP bundles sharing a GPU;
- mixed width-one and width-two bundles;
- duplicate actor GPU ownership; and
- flat/explicit union mismatch injected into plan validation.

### 13.3 Configuration validation

Add `tests/unit_tests/test_rlix_config.py`. Start from a minimal valid
world-model config and parameterize every rejection in section 7.3. Also
verify:

- absent `rlix` normalizes to disabled;
- explicit disabled mode does not enforce elastic restrictions;
- enabled defaults are inserted deterministically;
- invalid scalar types, including booleans accepted as integers by Python, are
  rejected; and
- error messages identify the failing path.

### 13.4 Entrypoint preflight and launch wiring

Extract helpers so CPU tests can use fake cluster, placement, group, and runner
objects. Verify:

- placement validation completes before the first `launch()` call;
- failure launches zero groups;
- enabled mode reuses the resolved strategies;
- rollout and environment receive the configured `max_concurrency`;
- actor does not receive elastic concurrency;
- disabled mode preserves existing arguments and ordering;
- enabled mode creates the controller, registers, and admits in exact order;
- the registration namespace equals the controller namespace;
- the registration payload is the preflighted immutable plan; and
- failures at controller creation, registration, or admission perform the
  correct reverse cleanup without model initialization.

### 13.5 Registration bootstrap

Add `tests/unit_tests/test_rlix_runtime.py` with fake control-plane and
controller factories. Cover:

- pipeline IDs come from `allocate_pipeline_id()`;
- namespaces come from `get_pipeline_namespace()`;
- coordinator creation precedes registration and admission;
- registration receives all five mappings, policies, and explicit bundles;
- registration failure best-effort unregisters before closing the coordinator;
- admission failure unregisters before closing the coordinator;
- cleanup is idempotent before any allocation exists; and
- cleanup errors do not replace the primary bootstrap error.

### 13.6 Regression tests

Run the T1/T2/T4/T5 focused suites because T6 promotes their late validation
assumptions to the entrypoint. Run existing placement and config tests to catch
changes to `HybridComponentPlacement` behavior.

No GPU is required for T6 unit acceptance. T8 owns real placement, recovery,
and utilization evidence.

### 13.7 Optional T6 bootstrap and real-model loading smoke

`tests/e2e_tests/embodied/task6_real_model_init_smoke.py` and its launcher place
real actor, Hugging Face OpenVLA-OFT rollout, and Wan environment Ray workers
on three distinct GPUs. The test performs T6 preflight, coordinator creation,
registration, and admission, then loads the real rollout and environment
models through their production `init_worker()` methods. It verifies that the
coordinator has no collection, no DP rank was activated, and both initialized
components are cold and offloaded before orderly cleanup. Actor
`init_worker()` is intentionally excluded: it constructs full training
gradients and AdamW state, which is outside T6 placement/bootstrap ownership
and makes this smoke depend on training-scale actor memory. The test does not
request an allocation, collect, interrupt, resize, synchronize, train, or
evaluate.

The verified checkpoint paths in the current test environment are:

```text
VLA: /workspace/VLA/Openvla-oft-SFT-libero-spatial-traj1
Wan: /workspace/WM/RLinf-Wan-LIBERO-Spatial
```

Run it from `RLinf/`; these are also the config defaults, while the environment
variables permit overrides on another host:

```bash
RLINF_VLA_CHECKPOINT=/workspace/VLA/Openvla-oft-SFT-libero-spatial-traj1 \
RLINF_WM_CHECKPOINT=/workspace/WM/RLinf-Wan-LIBERO-Spatial \
bash tests/e2e_tests/embodied/run_task6_real_model_init_smoke.sh
```

The harness, configuration, CLI import, Ruff, compilation, and shell syntax
were verified on 2026-07-21. Hardware execution confirmed real rollout and Wan
weight loading and exposed a pristine-state bug: with
`env.train.auto_reset: false`, initial cold offload encountered the intentional
`None` placeholders in Wan's image queue. `WanEnv.offload()` now preserves
those slots until the first explicit reset, with focused regression coverage.
An attempted actor `init_worker()` also showed why actor training initialization
does not belong in this smoke: single-rank FSDP becomes `NO_SHARD` and exhausts
a 24 GiB GPU while warming full AdamW state. The adjusted T6-focused smoke
passed on 2026-07-21 in 176.18 seconds on three RTX 4090 GPUs. It recorded the
expected actor-infer bundle `{0: [1, 2]}`, all five cluster device mappings,
and `inactive_cold` rollout and environment states in `result.json`. This smoke
is additional evidence only; training initialization, interruption/recovery,
and utilization acceptance remain later-stage work.

## 14. File-by-file edit list

### 14.1 Required new production files

`rlinf/scheduler/rlix/placement.py`

- immutable resolved-worker and placement-plan types;
- pure rank pairing and bundle construction;
- registration payload generation; and
- optional pre-resolved placement strategy.

`rlinf/scheduler/rlix/validation.py`

- RLix config normalization;
- supported-mode rejection matrix; and
- resolved-plan/topology validation.

`rlinf/scheduler/rlix/runtime.py`

- owner-scoped registered runtime context;
- exact T5 coordinator/register/admit construction order; and
- reverse-order bootstrap cleanup before T7 allocation ownership.

`examples/embodiment/config/rlix/elastic_vla.yaml`

- opt-in supported defaults only.

### 14.2 Required existing production files

`rlinf/config.py`

- normalize the disabled default;
- invoke pure enabled-mode validation after existing defaults are resolved;
- avoid importing `rlix_core` on the disabled path.

`examples/embodiment/train_embodied_agent.py`

- split preflight/launch helpers;
- resolve placement once in enabled mode;
- validate before worker launch;
- pass explicit environment/rollout concurrency;
- construct the registered runtime after paired worker launch; and
- retain a clean T7 handoff without requesting an allocation.

`rlinf/scheduler/rlix/__init__.py`

- export dependency-light placement types without eagerly importing Ray or
  `rlix_core`.

### 14.3 Required tests

- `tests/unit_tests/test_rlix_placement.py`
- `tests/unit_tests/test_rlix_config.py`
- `tests/unit_tests/test_rlix_runtime.py`
- focused entrypoint helper tests, either in
  `tests/unit_tests/test_rlix_entrypoint.py` or the closest existing entrypoint
  test module

### 14.4 Configuration example

Add one dedicated elastic Wan configuration derived from
`wan_libero_spatial_grpo_openvlaoft.yaml`. Keep checkpoint paths as documented
placeholders. Prefer a new file over changing the existing standalone example,
so disabled compatibility stays observable.

### 14.5 Required `rlix-core` protocol edits

- `rlix_core/protocol/types.py`: add `INITIALIZATION_CLUSTER_NAME` and
  `EVALUATION_CLUSTER_NAME` to the known GPU cluster sets;
- `tests/test_cluster_name_constants.py`: prove both names are fixed-only,
  non-generation, and parseable; and
- `tests/test_fixed_auxiliary_clusters.py`: prove complete atomic allocation at
  `Priority.INITIALIZATION`, no DP ownership, and no resize callback.

No planner algorithm change should be necessary. If either cluster requires a
special-case planning branch, stop and update the architecture before
implementation because the generic fixed path assumption is false.

### 14.6 Explicitly excluded files

- `rlix-core` planner algorithms and resize transaction code;
- `rlinf/runners/embodied_runner.py`: stage behavior is T7; T6 may add only an
  optional constructor field for the inactive registered-runtime handoff;
- T1/T2 world-model and worker lifecycle logic, unless a test exposes a real
  config-validation mismatch; and
- GPU e2e scripts and utilization reports, which belong to T8.

## 15. Suggested implementation sequence

1. Add failing pure tests for collocated/disaggregated conversion and all
   placement invariants.
2. Implement immutable placement projections, bundle construction, and
   registration payload copying.
3. Add the two fixed auxiliary core cluster constants and focused protocol /
   generic fixed-allocation tests.
4. Cross-check generated payloads with `rlix-core` validation.
5. Add failing config normalization and rejection tests.
6. Implement the two-phase validation module and call pure validation from
   `validate_cfg()`.
7. Add a pre-resolved placement strategy if the entrypoint otherwise resolves
   placement twice.
8. Implement and test coordinator/register/admit bootstrap plus failure
   cleanup.
9. Refactor entrypoint preflight into CPU-testable helpers.
10. Wire enabled-only environment/rollout concurrency and verify disabled
   parity.
11. Add the opt-in config group and a dedicated elastic Wan example.
12. Run focused T1-T6 tests, full RLinf unit tests feasible in the shared
    environment, Ruff, format, and compilation checks.
13. Update this document with implemented file names, intentional deviations,
    exact test counts, and the completion date.
14. Mark T6 complete in the three canonical project documents only after every
    acceptance criterion below passes.

## 16. Verification commands

From `RLinf/`:

```bash
export PYTHONPATH="$PWD:/root/_VLAMP/rlix-core/src${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_rlix_placement.py \
  tests/unit_tests/test_rlix_config.py \
  tests/unit_tests/test_rlix_runtime.py \
  tests/unit_tests/test_rlix_entrypoint.py
```

Focused regression command (adjust only for actual retained test file names):

```bash
export PYTHONPATH="$PWD:/root/_VLAMP/rlix-core/src${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_world_model_resume.py \
  tests/unit_tests/test_elastic_env_rollout.py \
  tests/unit_tests/test_rlix_progress.py \
  tests/unit_tests/test_rlix_resize_coordinator.py \
  tests/unit_tests/test_rlix_placement.py \
  tests/unit_tests/test_rlix_config.py \
  tests/unit_tests/test_rlix_runtime.py \
  tests/unit_tests/test_rlix_entrypoint.py
```

Style and compilation:

```bash
/root/.venv/bin/ruff check \
  rlinf/scheduler/rlix \
  rlinf/config.py \
  examples/embodiment/train_embodied_agent.py \
  tests/unit_tests/test_rlix_placement.py \
  tests/unit_tests/test_rlix_config.py \
  tests/unit_tests/test_rlix_runtime.py \
  tests/unit_tests/test_rlix_entrypoint.py
/root/.venv/bin/ruff format --check \
  rlinf/scheduler/rlix \
  rlinf/config.py \
  examples/embodiment/train_embodied_agent.py \
  tests/unit_tests/test_rlix_placement.py \
  tests/unit_tests/test_rlix_config.py \
  tests/unit_tests/test_rlix_runtime.py \
  tests/unit_tests/test_rlix_entrypoint.py
/root/.venv/bin/python -m compileall -q \
  rlinf/scheduler/rlix \
  examples/embodiment/train_embodied_agent.py
```

From the repository root, retain the core registration regression:

```bash
export PYTHONPATH="$PWD/rlix-core/src${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  rlix-core/tests/test_cluster_name_constants.py \
  rlix-core/tests/test_fixed_auxiliary_clusters.py \
  rlix-core/tests/test_composite_bundle_registration.py \
  rlix-core/tests/test_composite_bundle_scheduling.py
```

## 17. Definition of done

T6 is complete only when all of the following are true:

- enabled-mode configuration is explicit and validated before worker launch;
- disabled mode preserves the standalone entrypoint behavior;
- actual resolved placements, not Hydra strings, produce registration data;
- rollout and environment ranks pair by stable equal rank identity;
- collocated and disaggregated mappings match the T3 explicit-bundle contract;
- every worker process owns exactly one NVIDIA GPU on one node;
- bundles are non-empty, disjoint, and uniform within a pipeline;
- `actor_train` and `policy_sync` mappings contain the correct fixed unions;
- `initialization` and `evaluation` are known fixed-only auxiliary clusters
  with exact unions and no callback/DP ownership;
- every RLix TP size is one and only `actor_infer` is elastic;
- environment and rollout actors launch with explicit concurrency of at least
  two in enabled mode;
- controller construction, pipeline registration, and admission follow the T5
  identity and ordering contract;
- bootstrap failure cleans up registration, coordinator, and launched groups
  without replacing the primary error;
- all unsupported modes in the design fail with actionable errors;
- registration payloads pass `rlix-core` validation;
- CPU unit, regression, lint, format, and compilation checks pass; and
- no T7 stage-integration or T8 GPU-acceptance claim is made.
