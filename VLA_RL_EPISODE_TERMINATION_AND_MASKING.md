# VLA RL Episode Termination, Reset, and Post-Success Masking

## Status and scope

This document records the evidence and design rationale for episode termination,
environment reset, and post-success masking in VLA reinforcement learning. It
compares RLinf/RLinf-VLA, WoVR, and LeRobot and explains the implications for the
Wan world-model pipeline.

This is the semantic reference for the opt-in RLix rank-level early-completion
lifecycle. Wan selective slot reset remains out of scope and is tracked
separately in `WAN_SELECTIVE_AUTO_RESET_ISSUE.md`.

Sources were reviewed on 2026-07-31. GitHub citations are pinned to commits where
possible; paper citations identify the relevant section.

## Separate the four decisions

"Termination behavior" is ambiguous unless four independent decisions are
identified:

1. **Success detection:** what observation or simulator predicate counts as
   success?
2. **Logical episode boundary:** does the first success make the trajectory
   terminal, or must success still hold at the horizon?
3. **Collection lifecycle:** does a terminal slot reset immediately, become
   inactive/padded, or continue physical/model generation to a fixed batch
   horizon?
4. **Optimization semantics:** are post-terminal samples absent, padded and
   masked, or treated as valid training data?

Two systems may use the same `done` flag while making different choices for the
remaining three decisions. In particular, fixed-shape collection after a logical
termination does not imply that the continuation belongs to the training
trajectory.

## Common episodic RL behavior: LeRobot SAC

LeRobot's online HIL-SERL/SAC path follows the conventional immediate-reset
model:

- `RewardClassifierProcessorStep` evaluates the current image observation. Its
  defaults are `success_threshold=0.5`, `success_reward=1.0`, and
  `terminate_on_success=True`. A successful prediction sets the reward and the
  transition's `done` flag. The implementation contains no default
  consecutive-frame debounce.
- The online actor treats `done` or `truncated` as an episode boundary, sends the
  collected transitions, records episode statistics, and resets the environment
  and processor state.
- SAC prevents value bootstrapping across that terminal boundary with
  `rewards + (1 - done) * discount * next_value`.

Consequently, LeRobot normally does not generate a long continuation after a
classifier-declared success and then mask that continuation. This design is a
natural fit for a single physical environment actor and an off-policy replay
buffer; it does not need to preserve a fixed positional GRPO group in a dense
world-model batch.

## RLinf-VLA's three rollout modes

RLinf-VLA explicitly documents three modes:

| Mode | Logical objective | Collection behavior | Training behavior |
| --- | --- | --- | --- |
| Fixed episode length | `success_at_end` | Run to the maximum length | The continuation remains meaningful |
| Partial reset | `success_once` | Reset a completed sub-environment immediately | Start a new episode; do not train across the boundary |
| Valid action mask | `success_once` | Keep a fixed collection shape without resetting the slot | Use actions only through first completion; mask later actions |

For PPO, RLinf-VLA supports fixed episode length and partial reset. The paper
argues that partial reset improves sample efficiency when the target metric is
`success_once`, because a finished slot can begin collecting a new episode
instead of remaining idle.

For GRPO, RLinf-VLA presents the valid action mask as an effective design:

- each environment may run for at most `max_episode_steps`;
- only timesteps through task completion contribute to the objective;
- post-success actions are excluded as redundant under `success_once`;
- policy loss is normalized by each trajectory's valid length so short
  successful and long failed trajectories do not receive unequal weight merely
  from their lengths;
- members of a GRPO group represent the same task and initial state, analogous
  to multiple LLM responses to the same prompt.

The published result is task-dependent. RLinf-VLA reports a clear benefit from
valid action masking and length normalization on LIBERO-Goal, but no clear
benefit in its evaluated ManiSkill setting. The mechanism should therefore be
treated as a configurable objective and batching strategy, not a universal rule
for all VLA tasks.

## WoVR's world-model-specific rationale

WoVR provides a stronger reason for post-success masking in imagined rollouts.
Autoregressive video world models accumulate error with rollout depth. Once the
imagined trajectory has reached apparent success, later frames can be dominated
by visual drift or hallucination. Optimizing those later actions can teach the
policy to exploit simulator error rather than genuine task progress.

WoVR therefore:

- defines valid length through and including the first success;
- masks all later imagined steps;
- normalizes the GRPO objective by valid trajectory length; and
- combines masking with Keyframe-Initialized Rollouts (KIR), which start some
  trajectories near task-critical states to reduce effective prediction depth.

This rationale applies directly to Wan-based rollout. It is distinct from the
engineering reason for retaining a fixed batch shape: masking protects the
optimization signal, while fixed-shape generation satisfies the current model
and collector lifecycle.

## Is valid-action masking specific to GRPO?

Terminal masking as a principle is not GRPO-specific. RL algorithms must avoid
treating observations beyond a terminal boundary as an ordinary continuation of
the same Markov episode. The representation differs by algorithm:

- **SAC and other bootstrapped value methods:** the terminal flag suppresses
  next-state bootstrapping; the environment commonly resets immediately.
- **PPO:** completed vector environments commonly reset immediately. Fixed-size
  tensors may use masks for padding, but RLinf-VLA recommends partial reset for
  the `success_once` PPO configuration it studies.
- **GRPO:** in addition to terminal correctness, the collector must preserve
  comparison groups with the same task, initial state, and compatible policy
  version. RLinf-VLA names and studies its first-success form as the Valid Action
  Mask.
- **Offline preference methods:** DPO/TPO consume previously segmented
  trajectories, so online reset policy is not part of the optimizer itself.

In the current RLinf FSDP actor, the cumulative-done loss mask is constructed
before advantage-algorithm dispatch. The code path is therefore technically
reusable, although the paper's named design and ablations focus on GRPO.

## Implications for Wan and elastic collection

For the current Wan GRPO lifecycle, the safe semantics are:

1. first threshold-crossing success ends that slot's **logical** trajectory;
2. the success step remains valid;
3. subsequent fixed-horizon samples from that slot are excluded from policy
   optimization;
4. rank-level early completion is safe only when every trajectory assigned to
   that rank is terminal or truncated;
5. recycling a single physical slot is not safe until the collector carries
   explicit episode/group identity and can assemble GRPO groups independently
   of positional slots.

The last restriction is an inference from the documented GRPO group contract
and the current Wan batching/storage architecture. The papers do not claim that
selective reset is intrinsically invalid; it requires a collector capable of
preserving group identity across recycled slots.

## Implemented RLix rank-level early completion

RLix exposes the opt-in setting:

```yaml
env:
  train:
    stop_rank_when_all_done: true
```

It is disabled by default. Enabling it requires `auto_reset: false`,
`ignore_terminations: false`, and one rollout pipeline stage. These constraints
ensure first success remains a stable logical boundary and activates RLinf's
normal cumulative-done loss mask.

Each environment rank maintains a sticky success bit for every local
trajectory. A false flag after an earlier success cannot reopen that trajectory.
When the last local bit becomes true, the environment sends an explicitly
marked final bootstrap to its paired rollout rank. This natural completion wins
over a drain request arriving at the same committed chunk boundary. Partial
completion continues normally, and a drain before all local trajectories finish
still uses the existing snapshot/resume path.

Early output is padded before it is transferred to the actor:

- optimization fields such as rewards, old log probabilities, values, and
  optional decoded actions are extended to the configured chunk horizon with
  dtype/device/shape-matched zeros;
- model `forward_inputs` and observations repeat independent clones of the last
  valid entry, rather than invalid all-zero token sequences or attention masks;
- boundary-aligned `termination`, `truncation`, `done`, and value tensors are
  extended to `T+1`, preserving the real success boundary and using false/zero
  synthetic boundaries;
- padded policy-version entries retain the real collection policy version;
- no synthetic channel transition identities are invented.

OpenVLA rollout results may intentionally omit the optional decoded `actions`
field: their action tokens live in `forward_inputs`. Padding therefore derives
the collected chunk count from mandatory policy-version entries. The repeated
model inputs remain structurally valid if the actor executes its forward pass,
while `compute_loss_mask()` cumulatively scans the preserved real `done` and
makes every synthetic action post-terminal. This produces the same masked
rewards, log probabilities, GRPO scores/advantages, and policy-loss contribution
as an ordinary fixed-horizon run whose arbitrary post-terminal continuation is
masked. Padding exists only to retain fixed tensor shapes across data-parallel
actor ranks.

Worker logs expose the exact boundaries for later review:

```text
RLIX_TRAJECTORY_COMPLETED rank=0 env_index=7 ... epoch=0 chunk=13 step=104 outcome=success
RLIX_RANK_EARLY_FINALIZED rank=0 ... exit_chunk=13 exit_step=104 padded_chunks=19
```

The standalone Task 8 single-pipeline diagnostic enables this option and keeps
video and sealed-batch persistence enabled. The shared two-pipeline acceptance
configuration remains unchanged until this behavior has completed standalone
GPU validation.

## Benchmark scope relevant to this choice

LIBERO contains 130 tasks organized into LIBERO-Spatial, LIBERO-Object,
LIBERO-Goal, and LIBERO-100; LIBERO-100 is split into LIBERO-90 and LIBERO-10.
The first three controlled suites contain ten tasks each. Their controlled
variation matters when interpreting masking ablations:

- **Spatial:** the same core black-bowl-to-plate goal under ten different
  spatial descriptions/configurations, including relations to other objects and
  placement in/on fixtures.
- **Object:** the same basic pick-and-place behavior with ten different target
  objects placed into a basket.
- **Goal:** a shared kitchen-style object/fixture vocabulary with ten different
  goals, including opening drawers, turning on the stove, and placing different
  objects at different targets.
- **LIBERO-10:** ten longer, composition-heavy tasks reserved by the original
  benchmark for downstream lifelong-learning evaluation.
- **LIBERO-90:** ninety tasks used by the original benchmark as the pretraining
  portion of LIBERO-100.

Thus an ablation on LIBERO-Goal is evidence for a suite containing genuinely
different goals, not merely different initial poses. It does not by itself prove
the same effect size for the deliberately narrower Spatial suite.

## References

1. RLinf-VLA authors, **RLinf-VLA: A Unified and Efficient Framework for VLA+RL
   Training**, Sections 4.2.1, 4.2.2, and 5.3.2, 2025:
   <https://arxiv.org/html/2510.06710#S4.SS2> and
   <https://arxiv.org/html/2510.06710#S5.SS3.SSS2>.
2. WoVR authors, **WoVR: World Models as Reliable Simulators for Post-Training
   VLA Policies with RL**, Section 4.2, 2026:
   <https://arxiv.org/html/2602.13977#S4.SS2>.
3. Hugging Face, LeRobot `RewardClassifierProcessorStep`, commit
   `0d0737ab57f27c05d7b35fcf27e701f6003a5f3a`:
   <https://github.com/huggingface/lerobot/blob/0d0737ab57f27c05d7b35fcf27e701f6003a5f3a/src/lerobot/processor/hil_processor.py#L551-L647>.
4. Hugging Face, LeRobot online actor episode reset, same commit:
   <https://github.com/huggingface/lerobot/blob/0d0737ab57f27c05d7b35fcf27e701f6003a5f3a/src/lerobot/rl/actor.py#L384-L423>.
5. Hugging Face, LeRobot SAC terminal bootstrap mask, same commit:
   <https://github.com/huggingface/lerobot/blob/0d0737ab57f27c05d7b35fcf27e701f6003a5f3a/src/lerobot/rl/algorithms/sac/sac_algorithm.py#L303-L307>.
6. Lifelong Robot Learning, LIBERO repository overview and benchmark API,
   commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`:
   <https://github.com/Lifelong-Robot-Learning/LIBERO/blob/8f1084e3132a39270c3a13ebe37270a43ece2a01/README.md#libero-lifelong-robot-learning>.
7. LIBERO-Spatial official BDDL task definitions, same commit:
   <https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/bddl_files/libero_spatial>.
8. LIBERO-Object, LIBERO-Goal, LIBERO-10, and LIBERO-90 official BDDL task
   definitions, same commit:
   <https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01/libero/libero/bddl_files>.
9. Liu et al., **What Can RL Bring to VLA Generalization? An Empirical Study**,
   including PPO, GRPO, and DPO formulations, 2025:
   <https://arxiv.org/html/2505.19789>.
