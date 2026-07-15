# DP-Only Elastic VLA Rollout Implementation Plan

## 1. Contract and shared tasks

The supported collection path is elastic. RLix resizes complete data-parallel
VLA/world-model shards through the original callback shape:

```python
resize_infer(dp_ranks_to_remove: list[int], dp_ranks_to_add: list[int])
```

The implementation excludes TP, model PP, application rollout pipelining,
actor resizing, and environment migration.

This document uses the shared project task list:

| ID | Task | Status |
| --- | --- | --- |
| T0 | Fixed allocation foundation | Completed |
| T1 | Versioned continuation state | Pending |
| T2 | Local safe-point lifecycle | Completed (MVP) |
| T3 | Composite bundle scheduling | Pending |
| T4 | Elastic progress and release | Pending |
| T5 | RLinf resize coordinator | Pending |
| T6 | Placement and configuration | Pending |
| T7 | Runner stage integration | Pending |
| T8 | Verification and GPU acceptance | Pending |

## 2. Original RLix behavior and deviations

Original elastic scheduling performs:

```text
derive DP workers from flat device_mapping / tp_size
compute remaining-demand gap ratios
plan rank shrinks and expansions
execute every shrink callback outside the scheduler lock
wait for all shrinks
execute every expansion callback
wait for all expansions
commit the plan under the lock
```

Original ROLL inference resizing performs:

```text
shrink:
    remove ranks from active routing
    abort and drain their requests
    clear sticky source-rank mappings
    offload selected workers

expand:
    selectively synchronize/load selected workers
    add ranks to routing
    abort some old requests to rebalance source ranks
```

The extracted core already differs in package/actor prefixes, removal of ROLL
runtime ownership, T0 fixed allocation, and shared-cluster fail-fast behavior.
Elastic parity refers to planning and transaction ordering.

Intentional RLinf deviations are assigned to tasks:

| Task | Original RLix | RLinf adaptation |
| --- | --- | --- |
| T3 | A DP worker is one inference TP bundle costing `tp_size` GPUs. | A DP worker is a rollout/environment bundle whose actual width is its GPU cost. |
| T3 | No separate policy-sync cluster. | Add fixed `policy_sync` for the actor-plus-rollout union. |
| T4 | Every inactive rank can expand. | Only paused ranks with remaining work can expand. |
| T4 | Planned release targets all active ranks. | Add rank-specific release while retaining the old API. |
| T2/T5 | Shrink aborts an inference request immediately. | Shrink drains at the end of the current world-model chunk. |
| T2 | Caller retries prompt/token state on any worker. | EnvWorker resumes saved embodied state on the same rank. |
| T2/T5 | Expansion aborts old requests to rebalance. | Expansion resumes its own shard without interrupting siblings. |
| T7 | ModelUpdateService selectively updates expanding workers. | Use a short fixed all-rank policy sync until selective RLinf sync exists. |
| T5 | RLix pipeline runtime owns the coordinator. | RLinf owns an adapter with the same callback signature and lookup pattern. |

No additional deviation is allowed without updating this table and T8 parity
tests.

## 3. Safe-point invariant shared by T1, T2, and T5

A rank can be released only after:

1. Its current VLA prediction and send finish.
2. EnvWorker appends that policy result to the partial trajectory.
3. Wan/OpenSora `chunk_step()` commits the next frame chunk.
4. Reward, done, truncation, reset, and metric state commit.
5. The next `EnvOutput` is retained but not sent after drain is requested.
6. No next VLA call is in flight.
7. Continuation tensors are on CPU.
8. The rollout and environment models have released GPU residency.

```text
receive RolloutResult(transition=n)
append transition n exactly once
run chunk_step to completion
build EnvOutput(transition=n+1)

if drain requested:
    save EnvOutput n+1 as resume_bootstrap
    snapshot and offload
    acknowledge safe point
else:
    send EnvOutput n+1
```

## 4. Task-organized edit plan

### T0: Fixed allocation foundation

Purpose:

- Keep short indivisible stages safe.
- Preserve a fallback if elastic GPU acceptance has not passed.

Completed files:

- `rlix-core/src/rlix_core/protocol/types.py`
- `rlix-core/src/rlix_core/protocol/validation.py`
- `rlix-core/src/rlix_core/control_plane.py`
- `rlix-core/src/rlix_core/scheduler/scheduler.py`
- `rlix-core/src/rlix_core/scheduler/planner.py`
- `rlix-core/src/rlix_core/scheduler/validation.py`
- `rlix-core/tests/test_fixed_allocation_policy.py`

Legacy pseudocode:

```text
every generation registration is elastic
every actor_infer may enter resize planning
```

Current behavior:

```text
legacy actor_infer defaults to elastic
fixed actor_infer receives its whole mapping
fixed allocations never enter resize or donor planning
```

Exit condition: already met. T0 must remain regression-tested but must not be
used as the normal supported rollout allocation.

### T1: Versioned continuation state

Detailed edit-level design and test plan:
`TASK_1_ROLLOUT_SNAPSHOT_RESUME_IMPLEMENTATION_PLAN.md`.

Purpose:

- Make world-model and partial rollout state sufficient for exact same-rank
  continuation.
- Reject stale, wrong-rank, or malformed recovery state before mutation.

Files and edits:

- `rlinf/envs/world_model/base_world_env.py`
  - Define `WorldEnvResumeState` and snapshot/onload/offload capability.
- `rlinf/envs/world_model/world_model_wan_env.py`
  - Add symmetric save/load covering missing conditioning state.
- `rlinf/envs/world_model/world_model_opensora_env.py`
  - Convert existing state methods to the versioned validated contract.
- `rlinf/workers/env/env_worker.py`
  - Define `EnvRolloutCursor` state owned above the environment object.
- New CPU tests under `tests/unit_tests/test_world_model_resume.py`.

World state contains:

```text
schema/environment/rank/stage/episode identity
chunk and transition IDs
current_obs
image_queue
Wan condition_action or None
task descriptions and reset metadata
elapsed steps, previous reward, success, returns
reset IDs and reset generator state
deterministic diffusion transition seed
```

EnvWorker state contains:

```text
lifecycle generation and policy version
rollout epoch/chunk cursor
partial EmbodiedRolloutResult
current EnvOutput and resume_bootstrap
last observation/intervention/done state
accumulated metrics
```

Legacy pseudocode:

```text
OpenSora saves frames/latents without schema identity
Wan saves partial state, omits image_queue and condition_action, has no loader
EnvWorker loop cursor lives only in coroutine locals
```

After edit:

```text
snapshot validates identity and contains CPU tensors only
Wan/OpenSora restore every input needed by their next chunk
EnvWorker restores its partial training trajectory and exact next transition
```

Why it fixes the problem: frames alone do not preserve Wan action conditioning
or already collected policy data. `init_ee_poses` remain reset metadata; a
future evolving absolute-pose accumulator must be added if introduced.

Exit condition:

- Uninterrupted and pause/restore two-chunk runs are equivalent.
- Wrong schema, rank, episode, transition, shape, or dtype fails before load.

### T2: Local safe-point lifecycle

Detailed edit-level design and test plan:
`TASK_2_LOCAL_SAFE_POINT_LIFECYCLE_IMPLEMENTATION_PLAN.md`.

Purpose:

- Prove selected-rank pause/resume without involving scheduler decisions.
- Make channel and trajectory delivery exactly once.

Status: MVP implementation complete as of 2026-07-15. Implemented foundation work includes
structured transition identity, strict identity merges, worker snapshot schema
version 2, a collision-free routed observation/barrier envelope with logical
batch size, split/merge/inference helpers, validated lifecycle operations and
receipts, result identity split preservation, and the persistent rollout peer
cursor. The rollout worker now implements lifecycle activation/status/drain,
cursor-driven identified generation, awaited sends, final-bootstrap progress,
barrier/token handling, fail-closed validation, complete owned-model
offload/onload, and resume preparation. EnvWorker now supplies the matching
activation/status/drain APIs, identified shared interaction branch,
post-commit barrier and snapshot, public residency verification, environment
offload/onload, and resume preparation. A paired in-memory routed test proves
matching tokens, empty pause queues, verified peer offload/onload, one retained
transition dispatch, and completion. The current suite passes (`98 passed`) and covers
core snapshot/offload/onload/restore failures, partial paired offload failure,
new-lifecycle reuse, late-drain ordering, selective two-rank pause/resume with
an unaffected completing sibling, the complete functional drain-timing matrix,
deterministic transition/trajectory ordering equivalence, and richer Wan
world-state/metric equivalence. Further shared-loop cleanup and specialized
CUDA-graph/cache-residency failure injection are deferred as non-MVP hardening.
Full two-pipeline Wan/OpenSora GPU reuse remains T8 acceptance. T2 is not
connected to RLix.

Files and edits:

- `rlinf/workers/env/env_worker.py`
  - Persist loop cursors, observe drain after `chunk_step()`, retain next output,
    expose drain/snapshot/offload/restore/resume APIs.
- `rlinf/workers/rollout/hf/huggingface_worker.py`
  - Add rank lifecycle gate, transition IDs, selected-rank model/CUDA-graph
    offload, onload, and applied-policy-version check.
- New fake-channel tests in `tests/unit_tests/test_elastic_env_rollout.py`.

Lifecycle:

```text
INACTIVE_COLD -> EXPANDING -> ACTIVE
ACTIVE -> DRAIN_REQUESTED -> SNAPSHOTTING -> PAUSED
PAUSED -> EXPANDING -> ACTIVE
ACTIVE -> COMPLETED
COMPLETED -> EXPANDING -> ACTIVE       # strictly newer lifecycle only
ACTIVE | DRAIN_REQUESTED | SNAPSHOTTING | EXPANDING -> FAILED_RESIDENT
```

`RolloutTransitionIdentity` and the collision-free
`ElasticRolloutRequest` channel envelope belong to
`rlinf/data/embodied_io_struct.py`. Observation and barrier traffic uses
`Worker.send_to()`/`recv_from()` rather than direct queue addressing. The
envelope includes a logical batch size so routed receive validation also works
for a barrier with no observation body, and all returned route work is awaited.
Supported environments and rollout models expose public residency verification
rather than relying on private offload flags.

Legacy pseudocode:

```text
generate/interact loop over a static world size
next observation is sent immediately
models offload when the whole method returns
```

After edit:

```text
drain waits for current predict/send and chunk_step
next observation remains stored while paused
only selected complete-model replicas offload
resume validates policy/transition then sends stored output once
```

Completed shards notify a driver/controller task; they do not block inside an
EnvWorker method waiting for a scheduler callback that must re-enter that actor.

Exit condition:

- No message or trajectory transition is lost or duplicated.
- One rank can remain paused while siblings continue.
- Offload failure produces `FAILED_RESIDENT`, never a successful pause.

### T3: Composite bundle scheduling

Purpose:

- Represent the true GPU cost of one paired rollout/environment DP shard.
- Add a short actor-plus-rollout sync allocation without bloating actor training.
- Preserve all legacy flat TP behavior.

Files and edits:

- `rlix-core/src/rlix_core/protocol/types.py`
  - Add `POLICY_SYNC_CLUSTER_NAME` and framework-neutral explicit bundle types.
- `rlix-core/src/rlix_core/protocol/validation.py`
  - Validate explicit bundle identity, disjointness, union, range, contiguous
    ranks, uniform RLinf width, and fixed-only policy sync.
- `rlix-core/src/rlix_core/control_plane.py`
  - Forward optional `cluster_dp_device_mappings`.
- `rlix-core/src/rlix_core/scheduler/scheduler.py`
  - Centralize canonical bundle lookup and add fixed auxiliary sync planning.
- `rlix-core/src/rlix_core/scheduler/planner.py`
  - Snapshot explicit bundles and account for actual bundle GPU width.
- `rlix-core/src/rlix_core/scheduler/validation.py`
  - Simulate exact bundle ownership.
- Scheduler/protocol tests for bundles and `policy_sync`.

Registration:

```python
cluster_tp_configs={"actor_infer": 1, "policy_sync": 1}
cluster_dp_device_mappings={
    "actor_infer": {
        0: [rollout_gpu_0, env_gpu_0],
        1: [rollout_gpu_1, env_gpu_1],
    }
}
```

Collocated bundles have width one; disaggregated bundles have width two. One
pipeline cannot mix widths in this milestone.

Legacy pseudocode:

```text
bundle(rank) = device_mapping[rank*tp_size : (rank+1)*tp_size]
worker GPU cost = tp_size
known clusters exclude policy_sync
```

After edit:

```text
if explicit bundles: use exact registered bundle and its length
else: use unchanged TP slice and tp_size cost
policy_sync is fixed and handled before elastic generation at the same priority
```

Keep existing `SchedGuidedAllocationOp.dp_rank_to_gpus_to_add`; it already
preserves canonical rank identity for initial and later elastic activation.

Exit condition:

- Legacy tests are unchanged.
- Exact one-/two-GPU bundles allocate, shrink, expand, trace, and validate.
- `policy_sync` is atomic and never invokes `resize_infer`.

### T4: Elastic progress and release

Purpose:

- Prevent expansion of ranks with no work.
- Release a completed rank without pausing productive siblings.
- Preserve original remaining-demand gap-ratio math and wire compatibility.

Files and edits:

- `rlix-core/src/rlix_core/protocol/validation.py`
  - Validate optional rank sets inside `ProgressReport.metrics`; do not change
    the `ProgressReport` dataclass.
- `rlix-core/src/rlix_core/scheduler/planner.py`
  - Intersect inactive expansion candidates with resumable ranks when supplied.
- `rlix-core/src/rlix_core/scheduler/scheduler.py`
  - Add rank-specific planned release through the existing phase-0.5 shrink
    transaction.
- RLinf controller/EnvWorker reporting code.
- Progress/planner/release tests.

Metrics:

```text
completed
active_dp_ranks
resumable_dp_ranks
completed_dp_ranks
at_safe_point_dp_ranks
```

Legacy pseudocode:

```text
every inactive registered rank can expand
await_release_gpus selects all active generation ranks
```

After edit:

```text
if eligibility absent: retain legacy candidates
if eligibility present: expand only resumable ranks
await_release_dp_ranks removes only validated requested active ranks
await_release_gpus remains unchanged
```

Exit condition:

- Completed ranks never expand.
- Paused ranks can expand.
- Completed active ranks release immediately without sibling release.
- Legacy clients remain behaviorally unchanged.

### T5: RLinf resize coordinator

Purpose:

- Connect original RLix callback ordering to RLinf worker safe points.
- Keep the runner and workers RLinf-owned.

Files and edits:

- `rlinf/scheduler/rlix/coordinator.py`
- `rlinf/scheduler/rlix/controller.py`
- `rlinf/scheduler/rlix/protocol.py`
- Coordinator failure/order tests.

Use `rlix-core`'s current coordinator prefix and registered namespace. Literal
names differ from `_rlix`; the lookup pattern and callback signature match.

Legacy original pseudocode:

```text
remove routing, abort requests, offload infer worker
load/sync infer worker, add routing, rebalance by abort
```

After edit:

```text
resize_infer(remove, add):
    serialize against policy sync
    if remove:
        request T2 drains
        await T1 snapshot and environment/rollout offload
        return only after residency checks
    if add:
        onload selected paired workers
        validate T1 state, policy version, transition ID
        resume stored output once
        return only after ranks are active
```

RLix retains its ordering: callbacks execute outside its lock, all shrinks
finish before expansions, and state commits only after callback success.

Exit condition:

- Failure/timeout never commits the planned resize.
- Previous logical ownership remains while extracted scheduler activity fails
  closed.

### T6: Placement and configuration

Purpose:

- Ensure registered bundle identity matches actual Ray worker placement.
- Fail unsupported semantics before expensive initialization.

Files and edits:

- `rlinf/scheduler/rlix/placement.py`
- `rlinf/scheduler/rlix/validation.py`
- `rlinf/config.py`
- `examples/embodiment/train_embodied_agent.py`
- Placement/validation unit tests and an elastic config group.

Placement uses resolved `get_placement()` and `local_hardware_ranks`, pairs
equal rollout/environment process ranks, unions collocated IDs, and registers
`tp_size=1` regardless of bundle width.

Reject:

```text
multi-node or non-NVIDIA placement
TP/PP values greater than one
pipeline_stage_num != 1
unequal rollout/environment world sizes
more than one GPU per worker process
mixed bundle widths
async, decoupled, training-pipeline, or bootstrap-overlap modes
missing offload or unsupported stateful wrappers/GPU reward
```

Exit condition: CPU tests cover both valid topologies and every rejection.

### T7: Runner stage integration

Purpose:

- Use the correct allocation policy at each RLinf-owned stage.
- Seal and train only complete version-consistent batches.
- Preserve the no-op standalone path.

Files and edits:

- `rlinf/runners/embodied_runner.py`
- RLix controller protocol/factory
- runner/config tests

Legacy pseudocode:

```text
sync all rollout weights
run whole static env/rollout call
wait most handles; env metrics wait occurs later
train actor
```

After edit:

```text
fixed initialization and offload baseline

per step:
    acquire fixed policy_sync actor+rollout union
    synchronize every rollout rank and record version
    offload/release policy_sync

    request elastic actor_infer with trajectory target
    start with any active eligible rank
    accept T5 shrink/expand until all shard work completes
    seal complete batch, clear progress, release remaining ranks

    compute advantages
    acquire fixed all-rank actor_train
    train, advance version, offload, release
```

Evaluation remains fixed initially. Failure cleanup preserves the original
exception and never releases uncertain GPU residency.

Exit condition:

- Fake-worker tests prove stage order, partial activation, live resize,
  environment completion barriers, batch versions, and no-op compatibility.

### T8: Verification and GPU acceptance

Purpose:

- Prove original elastic parity and the GPU-idleness objective.

Test edits:

- Original-parity scheduler tests.
- T1 snapshot and T2 transaction tests.
- T3 composite bundle and policy-sync tests.
- T4 eligibility and release tests.
- T5 callback ordering/failure tests.
- T6 placement/config tests.
- T7 runner/no-op tests.
- Wan/OpenSora two-pipeline GPU acceptance configs and documentation.

Original-parity assertions:

```text
generation wakes with any active DP rank
remaining demand drives gap ratio
shrinks complete before expansions
callbacks complete before commit
failure does not commit
legacy TP clients behave unchanged
```

GPU matrix:

1. Interrupt one rank during Wan/OpenSora diffusion.
2. Verify release only after chunk commit and offload.
3. Run another pipeline on the released exact bundle.
4. Expand and complete the original episode.
5. Compare transition IDs, trajectories, versions, rewards, and final
   conditioning state with an uninterrupted reference.
6. Measure safe-point latency, snapshot size, memory reclaimed, resize cost,
   utilization, and throughput.

Exit condition: utilization improves with no missing/duplicate transition,
partial batch training, policy mismatch, or unsafe reuse. Elastic becomes the
default for supported Wan/OpenSora only after this task passes.

## 5. Implementation order

```text
T0 completed
T1 -> T2 -> T3 -> T4 -> T5 -> T6 -> T7 -> T8
```

T1 and the framework-neutral parts of T3 may be developed independently, but
no later task is complete until all earlier exit conditions it depends on are
met.
