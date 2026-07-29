# Task 1 Detailed Implementation Plan: Rollout Snapshot and Resume

## 1. Status and source of truth

Status: implemented for the scoped same-actor synchronous contract. Common,
OpenSora, and Wan world-environment continuation state, worker
cursor/snapshot/restore integration, exactly-once bootstrap consumption, and
two-chunk Wan equivalence tests are complete. The implementation is verified
with CPU fakes and a real Wan/DiffSynth accelerator run. Real OpenSora
accelerator validation remains pending.

Implementation progress:

- [x] Add common snapshot dataclasses, cloning, fingerprinting, validation,
  two-phase prepare/commit restore, and episode-generation tracking.
- [x] Add isolated common-contract tests with a fake `BaseWorldEnv` subclass.
  The fake initializes its reset generator in the test fixture, matching the
  concrete environments' ownership and avoiding double initialization in the
  base class.
- [x] Migrate OpenSora and add deterministic generator tests.
- [x] Complete Wan capture/restore/offload and its tests.
- [x] Add worker cursor/state types without changing external loop behavior.
- [x] Extract the pending-bootstrap state-machine boundary.
- [x] Add worker capture/validate/restore APIs with fail-closed feature gates.
- [x] Convert `_run_interact_once()` to cursor-driven iteration and resume
  consumption.
- [x] Add two-chunk equivalence and regression tests.
- [x] Run final formatting, lint, focused tests, and the broader embodied unit
  subset.
- [x] Run real OpenVLA-OFT-to-Wan two-chunk snapshot/offload/restore
  equivalence on two RTX 4090 GPUs.

This document expands Task T1, "Versioned continuation state," from:

- `../VLA_COMPATIBILITY_DESIGN.md`, especially recovery state and safe-point
  interruption.
- `RLIX_ELASTIC_VLA_ROLLOUT_IMPLEMENTATION_PLAN.md`, especially T1 and the
  safe-point invariant shared by T1, T2, and T5.
- `../VLA_COMPATIBILITY_IMPLEMENTATION_PLAN.md`, which defines the canonical
  T0-T8 task sequence.

If this document conflicts with the architecture document, the architecture
document wins. Update both documents if implementation discoveries require a
contract change.

## 2. Required outcome

After T1, one synchronous world-model rollout shard can capture all state
needed to continue its current collection on the same existing Ray actors.
The snapshot is held in CPU memory and can optionally be encoded as bytes for
transport. Restoring it must produce the same next committed chunk and the same
partial training trajectory as an uninterrupted execution.

T1 provides the continuation primitives. It does not itself interrupt a
running Ray call or release a scheduler allocation. The completed T2 MVP now
calls these primitives at a drained chunk boundary and adds rollout-worker
lifecycle and channel transition enforcement. T3 now supplies composite
scheduler ownership, T4 supplies release semantics, and T5 supplies the
callback transaction. T6 now supplies production placement/concurrency and
registration; production ownership transfer still requires T7 runner
invocation, while T8 owns accelerator acceptance.

The defining equivalence is:

```text
reset -> chunk 0 -> chunk 1

equals

reset -> chunk 0 -> snapshot -> perturb/offload -> restore -> chunk 1
```

Equality covers:

- the next visual observation;
- world-model conditioning inputs;
- reward, done, truncation, reset, return, and success state;
- the partial `EmbodiedRolloutResult`;
- the next rollout cursor and transition identity;
- the unsent next `EnvOutput` retained at the safe point.

## 3. Scope boundaries

### 3.1 Supported by T1

- `OpenSoraEnv` and `WanEnv`.
- Synchronous `EnvWorker` collection.
- Hugging Face embodied rollout data represented by `EmbodiedRolloutResult`.
- Complete-model data-parallel replicas.
- `rollout.pipeline_stage_num == 1`.
- Restore into the same environment worker rank and stage.
- CPU-resident continuation state.
- Model parameters remaining in the same Ray actor and moving through the
  existing offload/onload path rather than being copied into the snapshot.

### 3.2 Explicitly not implemented by T1

- Process restart recovery or a durable checkpoint directory.
- Restore onto another rank or a newly constructed environment.
- Tensor parallelism, model pipeline parallelism, or application rollout
  pipelining.
- Async environment workers.
- Drain requests and resize callback coordination.
- Rollout-model offload/onload and CUDA graph lifecycle.
- Channel queue serialization or channel replay.
- Cross-rank rebalancing.
- Selective policy synchronization.

### 3.3 Fail-closed feature gates

The snapshot entry point must raise `NotImplementedError` before capture when
any stateful feature is enabled but not represented in the snapshot. The
initial gates are:

- `pipeline_stage_num != 1`;
- `enable_decoupled_mode`;
- online LeRobot collection (`EmbodiedLerobotRolloutResult`);
- RLT pending transitions;
- history-buffer reward unless its manager state is included as described in
  section 9.7;
- data-collection wrappers with open episode buffers;
- any environment wrapper that declares mutable continuation state but does
  not expose a snapshot contract.

Video recording is not part of training equivalence. It may remain supported
only for same-actor pause/resume where the wrapper is not recreated or mutated
while paused. Otherwise it must be rejected or given an explicit wrapper state
contract.

## 4. Baseline gaps addressed by T1

This section records the pre-T1 implementation gaps that motivated the edits.
The status checklist in section 1 and the verification evidence in section 15
describe the current implementation.

### 4.1 `BaseWorldEnv`

File: `rlinf/envs/world_model/base_world_env.py`

The base class owns common reward and reset state but defines no continuation
schema. It also has no common validation or CPU-only assertion.

Required changes:

- define snapshot identity and state types;
- own schema versioning and common validation;
- own recursive CPU checks;
- provide capture, validate, and restore template methods;
- add episode-generation identity that cannot be confused with a reused
  dataset reset-state ID.

### 4.2 `OpenSoraEnv`

File: `rlinf/envs/world_model/world_model_opensora_env.py`

The existing `get_state()` and `load_state()` preserve several useful fields,
including `current_obs` and `image_queue`, but:

- the payload has no schema or environment identity;
- rank, stage, episode, chunk, and transition identities are absent;
- keys, shapes, and dtypes are not validated before assignment;
- a malformed load can partially mutate the live environment;
- diffusion noise uses the process-global generator in `torch.randn()`;
- the state format is an opaque byte buffer at the environment API boundary.

### 4.3 `WanEnv`

File: `rlinf/envs/world_model/world_model_wan_env.py`

The existing `get_state()` omits the two values that determine the next Wan
generation:

- the full `image_queue`;
- the rolling `condition_action` tensor.

There is no matching `load_state()`. Offload/onload moves `current_obs` and
metrics but does not explicitly manage `condition_action` or document the
device contract of queue frames.

### 4.4 `EnvWorker`

File: `rlinf/workers/env/env_worker.py`

The following continuation state currently lives in locals inside
`_run_interact_once()`:

- `epoch`;
- `chunk_step_idx`;
- `stage_id`;
- `env_outputs`;
- `env_metrics`;
- `rlt_pending_obs`.

The method sends the new `EnvOutput` immediately after `chunk_step()`. There is
no stable point where the next observation has committed but has not been sent.
`rollout_results`, last observations, previous dones, and prefetched bootstrap
state are worker attributes, but they are not grouped into a validated
continuation contract.

## 5. Snapshot semantics and invariants

### 5.1 Safe point represented by a T1 snapshot

A T1 snapshot may be captured only after all of the following are true:

1. The policy result for transition `n` has been fully received.
2. Its actions, log probabilities, values, versions, and forward inputs have
   been appended to the partial rollout.
3. `chunk_step()` has completed for transition `n`.
4. Rewards, done flags, truncations, auto-reset effects, metrics, and history
   effects have committed.
5. The resulting `EnvOutput` for transition `n + 1` exists in worker memory.
6. That output has not been sent to the rollout worker.
7. There is no policy inference request in flight for `n + 1`.

T1 exposes this boundary in the local state machine. T2 decides when a drain
request causes capture at this boundary.

### 5.2 Identity invariants

Every snapshot must identify:

- schema version;
- environment implementation;
- environment configuration fingerprint;
- environment worker rank and world size;
- stage ID;
- lifecycle generation;
- per-environment episode generation and reset-state ID;
- committed chunk index;
- next transition ID;
- rollout policy version.

Restore accepts expected worker/lifecycle/policy identity from its caller. It
does not silently adopt those values from an untrusted snapshot.

### 5.3 CPU ownership

The snapshot must own its data. Capturing it must clone tensors and arrays,
rather than returning aliases into live state. Every tensor reachable from the
snapshot must be on CPU and contiguous where applicable.

The snapshot must not include:

- model parameters;
- CUDA graphs;
- channel objects or futures;
- environment or worker instances;
- open file handles, executors, or wrapper threads.

### 5.4 Validation before mutation

Restore is a two-phase operation:

```text
decode -> validate envelope -> validate every world state
       -> prepare cloned replacement values -> commit all replacements
```

Schema, identity, shape, dtype, bounds, and CPU checks happen before the first
assignment to the live environment or worker. Validation errors leave the
target bitwise/logically unchanged.

Conversion from prepared CPU state to the runtime device should also happen in
temporary values before field assignment. If device allocation fails, the
worker remains resident with its old logical state and the error propagates.

### 5.5 Exactly-once boundary

T1 guarantees that restore leaves one pending bootstrap associated with one
next transition ID. The worker state machine can consume that pending value
only once. The completed T2 MVP extends `EnvOutput` and `RolloutResult` with
transition IDs so the channel peer rejects duplicate or stale traffic after
failures.

## 6. Data model

Use typed dataclasses for in-process state. Do not make a free-form dictionary
the primary internal contract. Byte encoding can use a versioned dictionary
envelope derived from the dataclasses.

### 6.1 Snapshot context

Add to `base_world_env.py`:

```python
@dataclass(frozen=True, slots=True)
class WorldEnvSnapshotContext:
    worker_rank: int
    worker_world_size: int
    stage_id: int
    lifecycle_generation: int
    chunk_index: int
    next_transition_id: int
    episode_generations: torch.Tensor
    reset_state_ids: torch.Tensor
```

The worker constructs this context. Environment code validates it against its
own `worker_info`, `num_envs`, and current reset identity.

### 6.2 World environment state

Add to `base_world_env.py`:

```python
WORLD_ENV_RESUME_SCHEMA_VERSION = 1

@dataclass(frozen=True, slots=True)
class WorldEnvResumeState:
    schema_version: int
    environment_type: str
    config_fingerprint: str
    worker_rank: int
    worker_world_size: int
    stage_id: int
    lifecycle_generation: int
    chunk_index: int
    next_transition_id: int
    episode_generations: torch.Tensor
    reset_state_ids: torch.Tensor
    reset_generator_state: torch.Tensor
    diffusion_generator_state: torch.Tensor | None
    diffusion_seed: int | None
    current_obs: Any
    image_queue: tuple[tuple[Any, ...], ...]
    condition_action: torch.Tensor | None
    task_descriptions: tuple[str, ...]
    init_ee_poses: tuple[Any, ...]
    elapsed_steps: int
    prev_step_reward: torch.Tensor
    success_once: torch.Tensor | None
    returns: torch.Tensor | None
    is_start: bool
```

`condition_action` is `None` for OpenSora. Environment-specific additions must
be explicit versioned fields or a typed subclass; do not add an unchecked
`extras: dict[str, Any]` escape hatch.

The configuration fingerprint should hash only continuation-relevant resolved
values, for example:

- environment type;
- `num_envs`;
- chunk length;
- condition-frame length;
- image size;
- model/reward type;
- auto-reset and relative-reward behavior;
- Wan retain-action behavior;
- OpenSora VAE type and latent condition length.

Paths and logging/video settings should not affect this fingerprint.

### 6.3 Episode identity

Add `self.episode_generations`, an `int64[num_envs]` CPU or runtime-device
tensor, to `BaseWorldEnv`. Increment the affected elements whenever `reset()`
selects a new episode. Do not use `reset_state_ids` alone as episode identity,
because a dataset sample can be selected more than once.

Both Wan and OpenSora reset paths must call a common helper after the selected
episode indices are known:

```python
def _commit_episode_reset(self, env_indices: torch.Tensor | None = None) -> None:
    ...
```

The snapshot stores both episode generation and reset-state ID. The worker
envelope duplicates these identities so corruption or stage mismatches are
detectable before restore.

### 6.4 Worker cursor

Add to `env_worker.py`:

```python
class RolloutCursorPhase(str, Enum):
    IDLE = "idle"
    WAITING_FOR_POLICY = "waiting_for_policy"
    COMMITTING_CHUNK = "committing_chunk"
    BOOTSTRAP_PENDING = "bootstrap_pending"
    EPOCH_FINALIZING = "epoch_finalizing"
    COMPLETED = "completed"


@dataclass(slots=True)
class EnvRolloutCursor:
    schema_version: int
    lifecycle_generation: int
    policy_version: int
    epoch_index: int
    chunk_index: int
    stage_id: int
    next_transition_ids: tuple[int, ...]
    phase: RolloutCursorPhase
```

Snapshot is legal only in `BOOTSTRAP_PENDING`. With the T1
`pipeline_stage_num == 1` gate, `stage_id` must be zero. Keeping stage and a
tuple of transition IDs in the schema makes the unsupported topology explicit
and provides a clean extension point for later pipeline work.

### 6.5 Worker snapshot

Add to `env_worker.py`:

```python
ENV_ROLLOUT_RESUME_SCHEMA_VERSION = 3

@dataclass(frozen=True, slots=True)
class EnvRolloutResumeState:
    schema_version: int
    worker_rank: int
    worker_world_size: int
    stage_num: int
    cursor: EnvRolloutCursor
    world_states: tuple[WorldEnvResumeState, ...]
    rollout_results: tuple[EmbodiedRolloutResult, ...]
    current_env_outputs: tuple[EnvOutput, ...]
    resume_bootstraps: tuple[EnvOutput | None, ...]
    last_observations: tuple[Any, ...]
    last_intervened_info: tuple[Any, ...]
    train_prev_done: tuple[torch.Tensor, ...]
    env_metrics: dict[str, tuple[torch.Tensor, ...]]
    prefetched_train_bootstrap: tuple[EnvOutput, ...] | None
    history_state: Any | None
```

Schema history:

- version 1 introduced worker continuation snapshots;
- version 2 added channel-visible elastic transition identity; and
- version 3 defines `current_env_outputs` plus the unsent
  `resume_bootstraps` as the canonical committed-boundary continuation.

In version 3, `last_observations` and `last_intervened_info` are normalized
from the committed output. They are not copied from the live end-of-rollout
caches, because those caches may be empty under `auto_reset=False` or stale
until rollout finalization under `auto_reset=True`. Snapshot creation validates
the normalized record before returning it, and restore repopulates the runtime
caches from the validated record.

Partial `EmbodiedRolloutResult` validation must not assume that every model
stores continuous `actions`. Required result state covers every committed
boundary. Continuous actions and model `forward_inputs` are each optional, but
one complete representation must cover all committed chunks; populated
optional sequences cannot be partial. OpenVLA-OFT therefore resumes from its
`action_tokens` forward inputs without adding a synthetic continuous training
field. Elastic snapshots also require one transition identity per committed
result boundary, and diagnostics include all sequence counts and forward-input
keys.

At `BOOTSTRAP_PENDING`, reward materialization intentionally lags the committed
environment chunk. The partial trajectory contains rewards for all earlier
commits, while `current_env_outputs`/`resume_bootstraps` retains the newest raw
environment reward for the next bootstrap-value and optional external-reward
calculation. Validation requires `committed_chunks - 1` materialized rewards
and exactly one pending reward; snapshot never appends it. Done/termination,
value, and elastic transition sequences use
`committed_chunks + epoch_index` because completed epochs add a final bootstrap
boundary without another policy action.

Do not serialize `defaultdict` factories. Normalize metrics to plain mappings
and immutable tuples in the snapshot.

`EmbodiedRolloutResult` and `EnvOutput` already move their standard tensors to
CPU, but capture must still deep-clone them because their lists and nested
`forward_inputs` remain mutable.

### 6.6 Optional encoded form

Provide encoding at the worker boundary rather than hiding serialization in
each environment:

```python
def encode_rollout_resume_state(state: EnvRolloutResumeState) -> bytes: ...
def decode_rollout_resume_state(data: bytes) -> EnvRolloutResumeState: ...
```

The encoded envelope begins with a plain schema/type marker. Decode does not
mean validate; restore must still validate against the live worker. Treat
snapshot bytes as trusted internal data because `torch.load` is not a safe
untrusted-input format. Prefer a `weights_only=True` compatible tree of
primitives and tensors if the installed PyTorch version supports all required
values.

## 7. `BaseWorldEnv` implementation

### 7.1 Public template API

Add:

```python
def snapshot_resume_state(
    self, context: WorldEnvSnapshotContext
) -> WorldEnvResumeState: ...

def validate_resume_state(
    self,
    state: WorldEnvResumeState,
    expected: WorldEnvSnapshotContext,
) -> None: ...

def prepare_resume_state(
    self,
    state: WorldEnvResumeState,
    expected: WorldEnvSnapshotContext,
) -> PreparedWorldEnvState: ...

def commit_resume_state(self, prepared: PreparedWorldEnvState) -> None: ...
```

`prepare_resume_state()` first calls validation, then creates all device-local
replacement values without mutating `self`. `commit_resume_state()` contains
only assignments and reconstruction of queue containers.

`PreparedWorldEnvState` is private and mutable. It is never serialized.

### 7.2 Subclass hooks

The base implementation captures common fields and calls narrow hooks:

```python
def _environment_type(self) -> str: ...
def _continuation_config(self) -> dict[str, Any]: ...
def _snapshot_image_queue(self) -> tuple[tuple[Any, ...], ...]: ...
def _snapshot_condition_action(self) -> torch.Tensor | None: ...
def _snapshot_diffusion_state(self) -> tuple[torch.Tensor | None, int | None]: ...
def _validate_model_resume_state(self, state: WorldEnvResumeState) -> None: ...
def _prepare_model_resume_state(self, state: WorldEnvResumeState) -> Any: ...
```

Common code validates:

- exact schema version;
- exact environment type and config fingerprint;
- rank/world/stage/lifecycle identity;
- non-negative chunk and transition values;
- per-environment vector lengths;
- exact episode and reset identity match with the worker context;
- task/reset metadata length;
- scalar types;
- metrics presence matching `record_metrics`;
- recursively CPU-only tensors;
- finite floating metrics where required.

### 7.3 Compatibility with existing state methods

Search found no RLinf caller of the world-model `get_state()`/`load_state()`
pair outside their definitions. Replace them with the typed contract and keep
temporary compatibility wrappers only if downstream users require them.

If wrappers remain, they must require a `WorldEnvSnapshotContext`; they must not
invent rank/stage/transition identity defaults. Legacy unversioned loads should
raise a clear schema error instead of being guessed into the new format.

## 8. OpenSora implementation

### 8.1 Dedicated diffusion generator

Create a generator owned by the environment:

```python
self._diffusion_generator = torch.Generator(device=self.device)
self._diffusion_generator.manual_seed(self.seed)
```

Pass it to the noise creation in `_infer_next_chunk_frames()`:

```python
z = torch.randn(..., generator=self._diffusion_generator)
```

Capture `_diffusion_generator.get_state()` after the committed chunk. Restore
the state before the next inference. This prevents unrelated torch RNG calls in
the Ray actor from changing resumed output.

If the target accelerator backend cannot construct a device generator through
this API, isolate the backend-specific generator creation behind a helper and
add a capability check. Do not fall back silently to process-global RNG.

### 8.2 Captured OpenSora fields

- `current_obs`;
- every latent frame in every `deque` in `image_queue`;
- task descriptions and `init_ee_poses`;
- elapsed steps, previous reward, metrics, and `is_start`;
- reset-state IDs, episode generations, and reset generator;
- diffusion generator state;
- identity and configuration fields from the common schema.

### 8.3 OpenSora validation

Validate at minimum:

- `current_obs` is floating point and shaped
  `[num_envs, 3, 1, time, height, width]`;
- `time` is within the current condition/chunk buffer bound;
- queue count equals `num_envs`;
- every queue length equals `z_condition_frame_length` at a resumable point;
- latent channel/spatial shapes agree within and across queues;
- queue dtypes match the configured inference dtype;
- reset IDs and episode generations are `int64[num_envs]`;
- reset/diffusion generator states are CPU `uint8` tensors;
- reward, return, and success tensor shapes and dtypes are exact;
- `condition_action is None`.

### 8.4 Offload interaction

Snapshot happens before offload. Existing offload must remain idempotent.
Replace direct `torch.cuda.empty_cache()` with the base accelerator-cache helper
so the CPU unit path and non-CUDA platforms do not depend on CUDA imports.

Restore ordering for T1 tests is:

```text
onload if needed -> validate/prepare snapshot -> commit snapshot -> continue
```

T2 may refine ordering so validation occurs before expensive onload while final
device preparation occurs after onload.

## 9. Wan implementation

### 9.1 Captured Wan fields

Capture all common fields plus:

- all `condition_frame_length` frames for every environment in `image_queue`;
- `condition_action` after its rolling update;
- the deterministic diffusion seed used by the Wan pipeline;
- `retain_action`, gripper reset behavior, and environment family through the
  configuration fingerprint.

`init_ee_poses` remain reset metadata. They are not a substitute for
`condition_action`.

### 9.2 Wan validation

Validate at minimum:

- `current_obs` is floating point and shaped
  `[num_envs, 3, 1, time, height, width]`;
- queue count equals `num_envs`;
- every queue contains exactly `condition_frame_length` frames;
- each frame is floating point with shape `[3, 1, height, width]`;
- `condition_action` is floating point with shape
  `[num_envs, condition_frame_length, 7]`;
- condition-action dtype is compatible with the action path;
- the recorded diffusion seed matches the supported deterministic seed
  contract;
- common identity, reset, and metric validation passes.

### 9.3 Symmetric restore

Implement Wan restore through `prepare_resume_state()` and
`commit_resume_state()`. Reconstruct each queue as a new list and assign a new
`condition_action` tensor. Never append restored frames to the existing queue.

### 9.4 Offload/onload completion

- Move `condition_action` to CPU during offload and to the action execution
  device during onload.
- Keep queue frames on CPU if `_infer_next_chunk_frames()` immediately converts
  them to NumPy, and document that invariant. Otherwise move them symmetrically.
- Clear accelerator cache through `_clear_accelerator_cache()`.
- Preserve idempotence through `_is_offloaded`.

### 9.5 Deterministic seed identity

Wan currently calls its pipeline with `seed=0`. Extract this into a named
attribute such as `self._diffusion_seed`. Snapshot and validate it. If future
Wan code derives a seed per chunk, the snapshot schema must store the next seed
or its generator state before that change is accepted.

### 9.6 Avoid hidden NumPy RNG dependence

Reset can use NumPy when fixed reset-state IDs are disabled. The reset-state
torch generator is already the intended deterministic partitioning mechanism
for the supported elastic path. Validation/configuration must require fixed
reset-state selection for the initial T1 path or add a dedicated NumPy
generator whose bit-generator state is included in the snapshot.

Do not rely on global `np.random.get_state()` because it is shared with other
code in the actor.

### 9.7 History reward state

If history-buffer reward is included in T1 rather than gated off, add typed
state methods to `rlinf/workers/env/history_manager.py` covering:

- `history_entries` with cloned CPU values;
- `history_counts`;
- worker `history_lengths` for the current reward request.

Validate environment counts and configured history-buffer names/sizes before
restore. Otherwise the worker snapshot API must reject this reward mode.

## 10. `EnvWorker` state-machine refactor

### 10.1 Persistent fields

Initialize in `EnvWorker.__init__()`:

```python
self._rollout_cursor: EnvRolloutCursor | None = None
self._current_env_outputs: list[EnvOutput] | None = None
self._resume_bootstraps: list[EnvOutput | None] = [None] * self.stage_num
self._rollout_env_metrics: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
self._rlt_pending_obs: list[dict[str, Any] | None] = [None] * self.stage_num
self._rollout_call_active = False
self._policy_request_in_flight = False
```

The cursor lifecycle generation increments when a new top-level collection
call is initialized. Restore supplies the expected generation; it must not
increment as a side effect of loading a snapshot.

### 10.2 Extract loop operations

Split `_run_interact_once()` into operations with explicit pre/postconditions:

```python
def _start_or_resume_rollout(self, rollout_channel: Channel) -> None: ...

def _receive_policy_result(
    self, input_channel: Channel, reward_channel: Channel | None
) -> RolloutResult: ...

def _commit_policy_result(self, rollout_result: RolloutResult) -> None: ...

def _commit_environment_chunk(
    self, rollout_result: RolloutResult
) -> EnvOutput: ...

def _set_pending_bootstrap(self, env_output: EnvOutput) -> None: ...

def _send_pending_bootstrap(self, rollout_channel: Channel) -> None: ...

def _advance_cursor_after_send(self) -> None: ...
```

The exact extraction may combine receive/commit helpers to avoid excessive
argument plumbing, but the boundary between `_set_pending_bootstrap()` and
`_send_pending_bootstrap()` must remain explicit and testable.

### 10.3 Cursor transitions

Normal collection:

```text
IDLE
  -> bootstrap/reset and send transition 0
  -> WAITING_FOR_POLICY(transition 0)
  -> COMMITTING_CHUNK(transition 0)
  -> BOOTSTRAP_PENDING(transition 1)
  -> send transition 1
  -> WAITING_FOR_POLICY(transition 1)
```

Snapshot path exposed for T2:

```text
COMMITTING_CHUNK(transition n)
  -> BOOTSTRAP_PENDING(transition n+1)
  -> capture CPU snapshot without send
```

Restore path:

```text
validate all state
  -> commit state in BOOTSTRAP_PENDING(transition n+1)
  -> send stored bootstrap once
  -> WAITING_FOR_POLICY(transition n+1)
```

### 10.4 Policy version capture

For the supported synchronous HF path, all values in
`RolloutResult.versions` for a committed result must identify one applied
rollout version. Convert that value to an integer only after checking that all
non-sentinel entries agree. Store it in the cursor.

The first committed policy result establishes the cursor policy version.
Subsequent results in the same collection must match. Restore receives
`expected_policy_version` from the coordinator/rollout worker and rejects a
mismatch before mutation.

### 10.5 Transition identity in T1

T1 stores transition identity in `EnvRolloutCursor` and associates the pending
bootstrap with `next_transition_ids[stage_id]`. Add internal assertions around
receive, append, chunk commit, pending send, and cursor advance.

Do not modify channel payloads as part of T1 unless implementation testing
shows it is unavoidable. T2 owns adding an optional `transition_id` to
`EnvOutput`/`RolloutResult`, merge/split preservation, and peer-side stale or
duplicate rejection.

### 10.6 Worker capture API

Add:

```python
def snapshot_rollout_stage(self) -> EnvRolloutResumeState:
    self._validate_snapshot_capability()
    self._assert_snapshot_safe_point()
    ...
```

The method must:

1. Verify the cursor is `BOOTSTRAP_PENDING`.
2. Verify no policy request or environment chunk is active.
3. Verify one unsent bootstrap exists for every supported stage.
4. Build per-stage `WorldEnvSnapshotContext` objects.
5. Capture every world environment.
6. Deep-clone partial worker state to CPU.
7. Recursively assert the completed snapshot is CPU-only.
8. Return a detached snapshot without changing cursor state.

Capture is repeatable at the same safe point and must return equivalent but
non-aliased objects.

### 10.7 Worker validation API

Add:

```python
def validate_rollout_resume_state(
    self,
    state: EnvRolloutResumeState,
    *,
    expected_lifecycle_generation: int,
    expected_policy_version: int,
) -> None: ...
```

Validate:

- worker schema, rank, world size, and stage count;
- supported cursor phase and cursor bounds;
- lifecycle and policy version;
- exactly one world state/output/result for the T1 topology;
- world/cursor stage, chunk, transition, episode, and reset identity agreement;
- pending bootstrap exists while no already-consumed marker is set;
- partial rollout list lengths are consistent with the cursor;
- every rollout tensor and nested forward input is CPU resident;
- last observations, intervention fields, previous dones, and metric shapes
  agree with `train_num_envs_per_stage`;
- the environment validates its state against the constructed expected context;
- unsupported state fields are empty rather than silently discarded.

### 10.8 Worker restore API

Add:

```python
def restore_rollout_stage(
    self,
    state: EnvRolloutResumeState,
    *,
    expected_lifecycle_generation: int,
    expected_policy_version: int,
) -> None: ...
```

Implementation order:

1. Reject restore while `interact()` or a policy request is active.
2. Run complete worker and environment validation.
3. Deep-clone all replacement worker values into temporaries.
4. Ask every environment to prepare device-local replacement state.
5. Commit all environment prepared states.
6. Assign worker cursor, partial rollout, outputs, metrics, and history.
7. Leave the cursor in `BOOTSTRAP_PENDING`.
8. Do not send from the restore method.

Steps 5 and 6 form the only mutation block. T1 has one stage, so failure
between multiple environment commits is excluded. Later multi-stage support
will require rollback or an atomic staging object.

### 10.9 Resume consumption

`_start_or_resume_rollout()` checks for a restored `BOOTSTRAP_PENDING` cursor.
It sends the saved output through the existing train-bootstrap send path,
clears only that pending slot, and advances to `WAITING_FOR_POLICY`.

Clearing after the local send call succeeds gives local exactly-once behavior.
If the process fails after enqueue but before clearing, T2 transition IDs make
the receiver reject a duplicate retry.

### 10.10 Preserve current completion behavior

Epoch finalization, final bootstrap value collection, trajectory sending,
`finish_rollout()`, last-observation storage, offload, and returned metric
format must remain unchanged for uninterrupted `interact()` calls.

The refactor must not call `_prepare_rollout_results()` when resuming a saved
partial trajectory. It may only initialize a new result when starting a new
lifecycle generation.

## 11. Serialization and cloning details

### 11.1 No live aliases

Use `clone_nested_to_cpu()` for dictionaries/lists/tuples and explicit
dataclass reconstruction for:

- `EnvOutput`;
- `ChunkStepResult` contents already accumulated in
  `EmbodiedRolloutResult`;
- `EmbodiedRolloutResult` list fields;
- cursor and metric containers.

Do not use a shallow `dataclasses.replace()` on mutable fields.

### 11.2 Recursive CPU assertion

Add a private reusable walker that supports:

- dataclasses;
- mappings;
- lists and tuples;
- `deque`;
- torch tensors;
- NumPy arrays;
- primitive immutable values.

It should report the field path of the first CUDA tensor or unsupported object,
for example:

```text
resume_state.world_states[0].image_queue[2][3] must be on CPU, got cuda:0
```

### 11.3 Snapshot size

Do not duplicate model weights. `current_obs` and `image_queue` are expected to
dominate snapshot size. Capture metrics for encoded byte size later in T8, but
add a unit assertion that snapshot size scales with continuation tensors rather
than model parameter count.

## 12. Tests

Create `tests/unit_tests/test_world_model_resume.py`. Keep it CPU-only and avoid
importing real Wan/OpenSora model packages by loading modules with stubbed
optional dependencies or constructing minimal instances with `object.__new__`,
following the approach in `test_maniskill_offload_env.py`.

### 12.1 Common state tests

- Capture returns schema version 1 and the expected identity.
- Every nested tensor is CPU resident.
- Captured tensors and lists do not alias live state.
- Capture twice at one safe point produces equivalent independent snapshots.
- Encode/decode preserves the typed state.
- Config fingerprint changes when a continuation-relevant field changes.

### 12.2 Validation failure matrix

Parameterize corruption of:

- schema version;
- environment type;
- configuration fingerprint;
- worker rank and world size;
- stage ID;
- lifecycle generation;
- episode generation;
- reset-state ID;
- chunk index;
- transition ID;
- missing required state;
- unexpected `condition_action` for OpenSora;
- tensor device, shape, and dtype;
- queue count and queue length;
- generator-state dtype;
- metric presence.

For every case, assert the exception is specific and the destination state is
unchanged.

### 12.3 OpenSora tests

- Round-trip `current_obs`, latent queues, reset state, rewards, and metrics.
- Restore a diffusion generator and prove the next generated fake noise tensor
  equals uninterrupted generation.
- Perturb process-global torch RNG between snapshot and restore and prove it
  does not affect the resumed dedicated generator.
- Reject inconsistent latent shapes/dtypes.
- Offload/onload remains idempotent with CPU fakes.

### 12.4 Wan tests

- Round-trip every queue frame and `condition_action`.
- Prove the rolling action window after the next fake chunk matches an
  uninterrupted run.
- Prove next fake frame conditioning consumes the restored queue rather than a
  reset/default queue.
- Reject missing/short queues and malformed condition actions.
- Validate and restore the deterministic diffusion seed identity.
- Offload/onload moves or preserves queue/action state according to the
  documented device contract.

### 12.5 Worker snapshot tests

Construct a minimal `EnvWorker` with one fake world environment and populate:

- a `BOOTSTRAP_PENDING` cursor;
- a partial `EmbodiedRolloutResult` containing actions, log probabilities,
  values, rewards, dones, forward inputs, and versions;
- a pending `EnvOutput`;
- last observations/intervention state;
- previous done state and accumulated metrics.

Then verify:

- capture includes every field and owns CPU copies;
- restore recreates the cursor and partial trajectory after deliberate
  perturbation;
- resumption sends the pending bootstrap once and advances the cursor once;
- a second resume attempt cannot resend the consumed bootstrap;
- restore rejects a policy version mismatch;
- capture rejects unsafe cursor phases and in-flight flags;
- capture rejects each unsupported feature gate;
- uninterrupted and snapshot/restored two-chunk fake runs yield identical
  final rollout results and environment metrics.

### 12.6 Regression tests

Run existing tests most likely to cover affected contracts:

```bash
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_world_model_resume.py \
  tests/unit_tests/test_maniskill_offload_env.py \
  tests/unit_tests/test_overlap_env_bootstrap.py \
  tests/unit_tests/test_history_manager.py
```

Also run the embodied I/O structure tests if they are added or modified during
implementation.

## 13. File-by-file edit list

### Required

`rlinf/envs/world_model/base_world_env.py`

- schema constants and dataclasses;
- episode-generation tracking;
- config fingerprinting;
- common capture/validate/prepare/commit implementation;
- recursive CPU and identity validation;
- subclass hooks.

`rlinf/envs/world_model/world_model_opensora_env.py`

- migrate state methods to the common contract;
- dedicated diffusion generator;
- OpenSora queue/shape/dtype validation;
- prepared restore construction;
- accelerator-neutral cache clearing.

`rlinf/envs/world_model/world_model_wan_env.py`

- complete capture of queue and rolling actions;
- symmetric validation and restore;
- diffusion seed identity;
- complete offload/onload of continuation tensors.

`rlinf/workers/env/env_worker.py`

- cursor and worker resume dataclasses;
- persistent loop state;
- explicit pending-bootstrap boundary;
- capture/validate/restore APIs;
- cursor-driven `_run_interact_once()`;
- feature gates and policy-version checks.

`tests/unit_tests/test_world_model_resume.py`

- all new CPU unit tests and fake dependencies.

### Conditional

`rlinf/workers/env/history_manager.py`

- only if history-buffer reward is supported in T1; otherwise add a fail-closed
  worker gate.

`rlinf/utils/nested_dict_process.py`

- only if the recursive clone/CPU walker is made general-purpose rather than
  private to the resume implementation.

`rlinf/data/embodied_io_struct.py`

- not required for the initial T1 cursor-local transition identity;
- subsequently updated by T2 for channel-visible transition IDs and
  merge/split support.

## 14. Suggested implementation sequence

1. [x] Add common snapshot dataclasses, cloning, fingerprinting, and validation.
2. [x] Add isolated common-contract tests with a fake `BaseWorldEnv` subclass.
3. [x] Migrate OpenSora and add deterministic generator tests.
4. [x] Complete Wan capture/restore/offload and its tests.
5. [x] Add worker cursor/state types without changing external loop behavior.
6. [x] Extract the pending-bootstrap state-machine boundary.
7. [x] Add worker capture/validate/restore APIs.
8. [x] Convert `_run_interact_once()` to cursor-driven iteration.
9. [x] Add two-chunk equivalence and regression tests.
10. [x] Run formatting, lint, focused tests, and the broader embodied unit
    subset.

Keep commits aligned with these boundaries so environment-state correctness can
be reviewed separately from the higher-risk worker-loop refactor.

## 15. Verification commands

From `/root/_VLAMP/RLinf`, using the shared workspace environment:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_world_model_resume.py
```

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/root/.venv/bin/python -m pytest -q \
  tests/unit_tests/test_world_model_resume.py \
  tests/unit_tests/test_maniskill_offload_env.py \
  tests/unit_tests/test_overlap_env_bootstrap.py \
  tests/unit_tests/test_history_manager.py
```

```bash
/root/.venv/bin/ruff check \
  rlinf/envs/world_model/base_world_env.py \
  rlinf/envs/world_model/world_model_opensora_env.py \
  rlinf/envs/world_model/world_model_wan_env.py \
  rlinf/workers/env/env_worker.py \
  tests/unit_tests/test_world_model_resume.py
```

```bash
/root/.venv/bin/python -m compileall -q \
  rlinf/envs/world_model \
  rlinf/workers/env \
  tests/unit_tests/test_world_model_resume.py
```

### 15.1 Recorded real-model result

From the current workspace, run:

```bash
cd /root/_VLAMP/RLinf
bash tests/e2e_tests/embodied/run_task1_snapshot_resume.sh
```

The model-explicit configuration is
`tests/e2e_tests/embodied/task1_wan_snapshot_resume.yaml`. It uses:

- GPU 0: `/workspace/WM/RLinf-Wan-LIBERO-Spatial`;
- GPU 1: `/workspace/VLA/Openvla-oft-SFT-libero-spatial-traj1`;
- one world-model environment and `num_inference_steps: 1`;
- `/root/.venv` with Python 3.11, PyTorch 2.6.0/CUDA 12.4,
  Transformers 4.40.1, and `huggingface-hub` 0.36.2.

Result recorded on 2026-07-15 on a 4-CPU-core, 128-GB-RAM server with 8 RTX
4090 GPUs: pass. OpenVLA generated both real action chunks. After chunk 0, the
test captured and validated a CPU-only snapshot. It then compared uninterrupted
chunk 1 against chunk 1 after Wan offload, prepare/commit restore into the same
environment instance, and onload. Observations, rewards,
termination/truncation flags, metrics, and the full final
`WorldEnvResumeState` matched within `rtol=atol=1e-5`. The captured
`current_obs` shape was `(1, 3, 1, 13, 256, 256)`. The focused unit suite
reported `31 passed`; Ruff, format, Python compilation, shell syntax, and diff
checks passed for the new harness.

This run validates real Wan T1 continuation and offload/restore behavior. It
does not prove drain coordination, scheduler release, bundle reuse by a second
pipeline, or scheduler-controlled resumed worker-channel ordering. Local drain
and resumed channel ordering are now covered by T2, while T3 now covers the
framework-neutral ownership and bundle scheduler. Runtime release,
coordination, placement, runner integration, and two-pipeline reuse remain
outside this T1 result. T4-T6 now supply release, coordination, and placement;
T7 runner integration and full T8 GPU acceptance are still intentionally
deferred, and OpenSora hardware equivalence remains pending
because `/workspace/WM` contains no OpenSora checkpoint.

### 15.2 Recorded collocated real-model result

The same equivalence harness also has an explicit one-GPU placement:

```bash
cd /root/_VLAMP/RLinf
CUDA_VISIBLE_DEVICES=2 \
  bash tests/e2e_tests/embodied/run_task1_snapshot_resume_collocated.sh
```

The logical device in
`tests/e2e_tests/embodied/task1_wan_snapshot_resume_collocated.yaml` is
`cuda:0` for both OpenVLA and Wan. Because their combined weights exceed a
24 GB RTX 4090, the harness offloads Wan before policy inference and offloads
OpenVLA before each world-model chunk. It asserts that collocated placement
uses one device and fails rather than silently falling back to two GPUs.

Result recorded on 2026-07-15: pass on one RTX 4090. OpenVLA generated two real
action chunks, Wan generated the corresponding frame chunks, and the resumed
second-chunk output and final continuation state matched the uninterrupted
path within `rtol=atol=1e-5`. The snapshot `current_obs` shape was
`(1, 3, 1, 13, 256, 256)`.

## 16. Definition of done

T1 is complete only when all of the following are true:

- Wan and OpenSora emit versioned, typed, CPU-only continuation snapshots.
- Wan restores `image_queue` and `condition_action` symmetrically.
- OpenSora future diffusion does not depend on unrelated process-global RNG.
- Environment restore validates all identity and tensor structure before
  mutation.
- `EnvWorker` owns its rollout cursor and partial continuation state outside
  coroutine locals.
- A committed next `EnvOutput` can remain pending without being sent.
- Worker restore recreates the partial trajectory and one pending next
  transition.
- Uninterrupted and snapshot/restored two-chunk CPU executions are equivalent.
- Wrong schema, rank, stage, lifecycle, policy, episode, transition, shape, or
  dtype fails before mutation.
- Unsupported stateful modes fail closed.
- Existing uninterrupted rollout, offload, bootstrap, and history tests pass.

T1 completion does not claim that GPUs can yet be safely released. T2 now
supplies the local lifecycle prerequisite and T3 supplies composite bundle
accounting. T4 release semantics, the T5 scheduler callback transaction, and
T6 production placement/registration are now complete. T7 runner integration
and T8 real two-pipeline acceptance remain required before logical ownership
can move safely in production.
