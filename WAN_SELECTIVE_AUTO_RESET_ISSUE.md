# Issue: Wan Does Not Support Correct Selective Auto-Reset

## Status

Open. This is a follow-up design and implementation issue, not part of the
current Task 1 rollout snapshot/resume scope.

## Summary

`WanEnv` is vectorized, but its reset lifecycle is batch-synchronous. In the
stock Wan GRPO configuration, `auto_reset` is disabled. A logical environment
that terminates early remains in the Wan inference batch until the fixed
rollout horizon; transitions after its first termination are masked from
training. All environment slots are reset together at the next rollout epoch.

Enabling `auto_reset` does not provide normal per-environment recycling. When
any slot is done, `WanEnv._handle_auto_reset()` calls the full `reset()` method,
which replaces the state of every slot. This can prematurely discard unfinished
episodes and break GRPO group semantics.

The desired behavior is either:

1. selective auto-reset, where only completed slots start new episodes; or
2. explicit GRPO group-synchronous reset, where completed slots become inactive
   until every member of their group has completed or reached its horizon.

The intended mode must be selected as part of the design because the two modes
have different rollout-storage and advantage-computation requirements.

## Current Behavior

For the stock configuration with eight environments and `group_size: 8`:

```text
reset all 8 slots to the same task and initial state
  -> execute a fixed number of batched Wan chunks
  -> retain each slot's data through its first done
  -> mask that slot's later transitions
  -> reset all 8 slots at the next rollout epoch
```

This preserves one comparable set of eight trajectories for GRPO, but it
wastes Wan inference after individual slots terminate.

The current auto-reset path is also batch-wide:

```text
any slot is done
  -> WanEnv._handle_auto_reset()
  -> WanEnv.reset()
  -> all slots are replaced
```

This differs from IsaacLab, ManiSkill, and Libero wrappers, which can pass a
done-index mask to reset only selected simulator environments.

## Why Selective Reset Is Not a Local Change

### 1. Reset reconstructs the complete Wan batch

`WanEnv.reset()` builds and assigns all of the following for every slot:

- `current_obs`;
- `image_queue`;
- `condition_action`;
- task descriptions and initial end-effector poses;
- reset-state IDs;
- reward and episode metrics.

There is no indexed reset API. A selective implementation must replace every
piece of continuation state for the selected slots atomically while leaving
unfinished slots unchanged.

### 2. Episode age is batch-global

`BaseWorldEnv.elapsed_steps` is a scalar, and Wan increments it once per action
chunk for the whole batch. Selectively recycled slots require an integer step
counter per slot so that truncation is computed independently.

Changing this also affects snapshot validation and restore because the Task 1
schema currently records a scalar elapsed-step value.

### 3. Continuation state spans devices and representations

Selective reset must update a mixture of state without cross-episode leakage:

```text
GPU/runtime tensors:
  current_obs, condition_action, reward state

CPU tensors and Python containers:
  image_queue, task descriptions, initial poses, reset metadata

Identity and randomness:
  reset-state IDs, episode generations, reset RNG, diffusion RNG/seed
```

For example, replacing `current_obs[i]` without replacing `image_queue[i]`
would condition the next diffusion call on frames from the previous episode.

### 4. Wan advances slots through one batched diffusion call

One `WanVideoPipeline` invocation consumes the images and actions for the whole
local batch. A slot cannot reset in the middle of a generated action chunk. The
earliest valid reset point is after generation, decoding, reward evaluation,
and terminal detection for the chunk.

Keeping a constant batch size is possible by replacing completed slots before
the next call. Compacting only active slots is a separate optimization and may
change throughput, memory use, and deterministic random-number consumption.

### 5. Diffusion randomness is worker-level

Wan currently supplies a scalar diffusion seed to the batched pipeline.
Selective episode recycling needs a defined per-slot/per-episode randomness
contract so that resetting one slot does not unexpectedly change unfinished
slots and snapshot/resume remains reproducible.

### 6. Independent recycling breaks positional GRPO groups

The eight slots in a Wan GRPO group start from the same task and reset state.
GRPO compares their returns by reshaping adjacent samples according to
`group_size`. If a completed slot immediately starts a new task while the other
seven slots continue, adjacency no longer identifies a valid comparison group.

Correct selective recycling therefore requires explicit episode and GRPO-group
identities, plus a scheduler that only compares trajectories produced from the
same task, initial state, and compatible policy version.

### 7. Rollout storage assumes one trajectory per slot

With auto-reset, a slot may produce multiple episodes during one collection
window. Those episodes must be split into distinct samples. The current Wan
GRPO path computes one score per batch slot and relies on first-done masking in
the non-auto-reset mode. It does not represent an arbitrary number of completed
episodes per slot as separate GRPO samples.

Enabling selective reset without changing rollout segmentation could discard
later episodes or assign their actions an advantage from the wrong episode.

## Relationship to Task 1 Snapshot/Resume

Task 1 supports the current synchronous Wan lifecycle and captures each slot's
continuation tensors inside one worker-level snapshot. Selective auto-reset
would require extending that contract to include at least:

- per-slot elapsed-step counters;
- explicit active/done and episode identity state;
- per-slot or per-episode diffusion randomness, if adopted;
- partial GRPO-group membership;
- multiple pending/completed episode segments per physical slot.

Task 1 should not silently add these semantics. The snapshot schema should be
updated together with the selective-reset design if this issue is implemented.

## Candidate Designs

### A. GRPO group-synchronous reset

Treat each group of eight as the scheduling unit. Mark completed members
inactive, finalize the group when every member is done or truncated, and reset
the complete group together.

Advantages:

- preserves current GRPO grouping;
- requires fewer rollout-buffer changes;
- keeps task and policy-version identity simple.

Limitations:

- retains the long-tail problem;
- inactive slots may still consume diffusion compute unless active batches are
  compacted or the pipeline supports an effective inactive mask.

### B. Full selective slot recycling

Reset a completed slot at the next chunk boundary and immediately assign it to
a new episode.

Advantages:

- improves environment utilization;
- allows multiple episodes per slot per collection window.

Requirements:

- indexed Wan reset and per-slot counters;
- episode-oriented rollout segmentation;
- explicit GRPO group IDs and a completed-trajectory assembler;
- defined per-slot randomness and snapshot semantics;
- policy-version checks when assembling delayed groups.

### C. Dynamic active-batch compaction

Remove completed slots from Wan calls until their group is finalized, or fill
free capacity with newly scheduled episodes while maintaining a logical-to-
physical slot map.

This may reduce wasted inference but is the most invasive option. It changes
batch sizes, RNG behavior, output scattering, snapshot identity, and scheduler
state, and should be evaluated only after the episode/group model is correct.

## Recommended Direction

Implement group-synchronous reset first for Wan GRPO and make that lifecycle
explicit. Separately prototype selective recycling with an episode-oriented
collector before enabling `auto_reset` in production Wan configurations.

The environment API should eventually support an indexed reset operation even
if the GRPO scheduler initially invokes it one complete group at a time. This
keeps state ownership correct and creates a path for PPO or other algorithms
that do not require fixed groups.

## Proposed Implementation Work

- Add an indexed Wan reset API accepting selected environment indices and reset
  state IDs.
- Convert elapsed steps and any remaining batch-global episode state to
  per-slot tensors.
- Add one helper that atomically replaces all Wan continuation state for the
  selected indices.
- Define diffusion seed/generator ownership by worker, slot, and episode.
- Introduce explicit episode IDs and GRPO group IDs in rollout records.
- Segment completed episodes independently of physical environment slots.
- Assemble GRPO groups by identity instead of positional adjacency alone.
- Define behavior for policy-version changes while a group is incomplete.
- Extend snapshot/resume state and validation for partial groups and recycled
  slots.
- Add utilization metrics for inactive, masked, and recycled slot-chunks.

## Acceptance Criteria

- Resetting one Wan slot does not modify any continuation state for unfinished
  slots.
- A reset slot receives fresh observation, image queue, action conditioning,
  metrics, reset identity, episode generation, and RNG state.
- Truncation is based on each slot's own episode age.
- No generated transition is assigned to two episodes or to the wrong episode.
- Every GRPO comparison group contains trajectories from the same task and
  initial reset state under an allowed policy-version contract.
- Multiple episodes produced by one physical slot are emitted as separate
  training samples.
- Snapshot/restore during mixed episode generations produces the same next Wan
  outputs and completed groups as uninterrupted execution.
- Focused tests cover one slot finishing early, multiple slots finishing in the
  same chunk, group completion, horizon truncation, and snapshot/resume after a
  selective reset.
- A real GPU benchmark reports whether selective recycling improves completed
  valid trajectories per second relative to fixed-horizon masking.

## Relevant Files

- `rlinf/envs/world_model/world_model_wan_env.py`
- `rlinf/envs/world_model/base_world_env.py`
- `rlinf/workers/env/env_worker.py`
- `rlinf/data/embodied_io_struct.py`
- `rlinf/algorithms/utils.py`
- `rlinf/algorithms/advantages.py`
- `rlinf/utils/metric_utils.py`
- `examples/embodiment/config/env/wan_libero_spatial.yaml`
- `examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml`
- `TASK_1_ROLLOUT_SNAPSHOT_RESUME_IMPLEMENTATION_PLAN.md`
