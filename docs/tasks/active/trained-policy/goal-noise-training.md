# Pre-deployment training pass: PPO with goal-position noise

**Type:** task / training
**Owner:** DGX (`strafer_lab` lane — env config + training run)
**Priority:** P2 — gates *deployable-policy quality* for VLM-grounded
goals; not blocking the inference plumbing in
[`inference-package`](inference-package.md), which
validates against direct-pose goals.
**Estimate:** M (~1–2 days: config + training resume + evaluation
sweep). Depends on a converged baseline checkpoint already existing.
**Branch:** task/policy-goal-noise-training

## Story

As a **mission operator running missions where goals come from VLM
grounding (Qwen2.5-VL-3B with ±0.2–0.5 m localization error)**, I want
**the PPO policy retrained with `goal_position_noise_std` matching
the deployment goal-noise distribution**, so that **the deployed
policy converges smoothly to the goal instead of oscillating around
it because the training distribution had pixel-perfect goal
positions and the deployment distribution doesn't**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [strafer-inference-package.md](inference-package.md) — the
  inference brief whose Phase 5 (end-to-end acceptance) becomes
  reliable once this training pass lands. Without goal-noise
  training, deployment with VLM-grounded goals shows oscillation that
  is *not* an inference-side bug.
- [policy-export-tooling.md](../../completed/policy-export-tooling.md) — the next-step
  brief in the deployment chain. This brief produces the checkpoint;
  that brief converts it to a deployable artifact.

## Context

### The sim-to-deployment gap

Deployment goal-pose error has *three* sources, with distinct
distributions. The original framing of this brief covered only the
first; this revision adds the other two so the training distribution
matches what the policy actually sees on the real robot.

1. **VLM-grounding noise** (per-regrounding-event, large magnitude).
   Qwen2.5-VL-3B real-camera grounds exhibit ~0.2–0.5 m localization
   error per ground (see [`strafer_vlm/`](../../../../source/strafer_vlm/)
   ground-truth-vs-observed analysis). Each regrounding step is an
   ~independent draw from that distribution.
2. **SLAM continuous drift** (per-tick, small magnitude). RTAB-Map's
   `map → base_link` TF drifts continuously during long traversals
   at ~1 cm/s typical. Well-modeled by per-tick Gaussian goal noise
   in the policy's body frame (since the policy sees the goal
   through that TF chain).
3. **SLAM loop-closure jump** (per-event, large magnitude,
   discrete). RTAB-Map occasionally detects a loop closure and
   *snaps* the `map` frame, producing a 10–50 cm discrete goal
   shift visible to the policy as a sudden goal-pose change. Not
   modeled by Gaussian noise — this is an event distribution.

The current PPO training pipeline addresses none of these — the
`policy` group's `goal_relative` / `goal_distance` /
`goal_heading_to_goal` are derived from `command[:, :2]` which is
the exact ground-truth goal in sim, fixed per episode.

There IS an existing `randomize_goal_noise` event in
[`events.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py)
referenced by `EventsCfg_Realistic` with `noise_std=0.15` and
`mode="reset"`. That's a per-episode-reset offset — the policy sees
*the same* noisy goal for the entire episode. The brief's analysis
correctly identifies this as the "wrong" pattern: the policy learns
to converge precisely on the wrong point, then re-converge when the
episode resets. The fix must produce per-tick variation, not just
per-reset.

### Why all three noise types matter for deployment

At deployment with VLM-grounded missions:
- The mission emits a goal pose at start (VLM ground).
- Mid-mission, the VLM re-grounds every ~5–10 s (event of type 1).
- Between re-grounds, the SLAM frame drifts (event of type 2).
- Occasionally SLAM closes a loop (event of type 3).

A policy trained only with per-tick Gaussian noise (type 2) handles
the slow drift but discovers types 1 and 3 at deployment. A policy
trained only with per-reset noise (the current `randomize_goal_noise`
event) doesn't handle any of them.

### What's missing

[`source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py:363`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py)
— `GoalCommandCfg` does *not* expose a `goal_position_noise_std` field
today. The training-time implementation needs:

1. The config knob (new field on `GoalCommandCfg`).
2. The runtime perturbation in `GoalCommand._update_command()` (or wherever
   the published goal is read by observation terms — need to perturb
   per *observation tick*, not per-resample, so the policy sees a
   different noisy goal each step).
3. A resumed training run from a converged baseline checkpoint with the
   knob set.

### Why this is a *targeted final pass*, not a fresh train

Adding goal noise from epoch 0 makes the credit-assignment problem
harder (the policy can't distinguish goal-error from policy-error
early on). Standard practice: train to convergence with zero noise,
then resume from the converged checkpoint with noise enabled for a
final ~10–20% of total training budget. The policy adapts to the
noise distribution without forgetting the base navigation behavior.

## Approach

### Phase 1 — Add the config knobs (½ day)

In [`commands.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py):

- Add three fields to `GoalCommandCfg` (and the `GoalCommandProcRoom`
  subclass config), one per deployment noise source:
  ```python
  # Type 2 — per-tick continuous drift (SLAM continuous drift)
  goal_position_noise_std: float = 0.0
  """Per-step Gaussian noise (meters, isotropic xy) applied to the
  goal position seen by observation terms. Models SLAM continuous
  drift in the map → base_link TF. 0.0 disables (training default).
  0.05 m typical."""

  # Type 3 — per-event discrete jumps (SLAM loop closure)
  goal_jump_probability_per_step: float = 0.0
  """Per-step probability of a discrete goal jump event (SLAM loop
  closure). 0.0 disables. ~0.001 (so a jump every ~30 s at 30 Hz)
  matches RTAB-Map loop-closure cadence on a typical traversal."""
  goal_jump_std: float = 0.0
  """Std of the discrete jump (meters, isotropic xy) when a jump
  fires. 0.3 m typical for loop-closure-class events. Independent
  of goal_position_noise_std (events compose additively)."""
  ```

  Type 1 (VLM re-grounding) is intentionally NOT added as a new knob.
  Re-grounding events look identical in the policy's observation to
  loop-closure events from a model standpoint (a sudden ~0.2–0.5 m
  goal shift). Tune `goal_jump_probability_per_step` and
  `goal_jump_std` to cover both. A separate `vlm_reground_*` knob
  pair would over-specify the model.

- In `GoalCommand` (and `GoalCommandProcRoom`): the `command` property
  returns the *base* goal pose. Add a `noisy_command` property — or
  modify `command` to add per-step noise — that observation terms
  consume. The exact point of perturbation matters:
  - **Per-step noise (correct):** policy sees a different noisy goal
    every observation tick → trains to be robust to per-tick goal
    drift, which matches deployment.
  - **Per-resample noise (wrong, current behavior of
    `randomize_goal_noise` event):** policy sees the same noisy goal
    for the entire resample window → trains to converge to the wrong
    point precisely, doesn't help.
  Implementation: maintain a per-env persistent goal offset that
  evolves per-tick (drift + occasional jump). Sampling pattern:
  ```python
  # per env, per tick
  goal_offset += N(0, goal_position_noise_std)        # type 2 drift
  if rand() < goal_jump_probability_per_step:
      goal_offset += N(0, goal_jump_std)              # type 3 jump
  noisy_goal = base_goal + goal_offset
  ```
  The offset is persistent within an episode but resets at episode
  reset. The drift component is integrated, not re-sampled — that
  matches SLAM drift (cumulative), not white noise.
- Unit test in `source/strafer_lab/tests/test_commands.py` (or where
  command tests live): three cases:
  - At `goal_position_noise_std=0.0` and `goal_jump_probability_per_step=0.0`,
    the published goal is byte-identical to the base goal across
    ticks.
  - At `goal_position_noise_std=0.05`, `goal_jump_probability_per_step=0.0`,
    the published goal *integrates* per-tick (cumulative shift),
    with variance matching expected Brownian-motion bound.
  - At `goal_position_noise_std=0.0`, `goal_jump_probability_per_step=0.001`,
    `goal_jump_std=0.3`, the published goal stays at base for most
    ticks and exhibits discrete ~0.3 m jumps at the expected
    frequency over a long horizon.

  Cross-reference: this brief consumes Finding 3 in
  [`observation-contract-cleanup.md`](observation-contract-cleanup.md).

### Phase 2 — Identify or train baseline checkpoint

**MVP target: DEPTH variant** (since
[`inference-package`](inference-package.md) ships
`strafer_direct` against `PolicyVariant.DEPTH`). Skip if a converged
DEPTH baseline already exists from the ProcRoom-Depth env. Otherwise:

```
python Scripts/train_strafer_navigation.py \
    --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0
```

Train to convergence (target episode reward; episode-length-target
met). Save baseline as
`logs/rsl_rl/strafer_navigation/baseline_depth_no_noise/model_<step>.pt`.

NOCAM_SUBGOAL has its own baseline produced by
[`subgoal-env`](subgoal-env.md) Phase 5
— if/when that brief ships and a goal-noise pass is wanted for the
hybrid mode policy too, file a `policy-subgoal-noise-training.md`
follow-up. Subgoal noise from Nav2 path resolution (~5 cm at
MAP_RESOLUTION) is smaller than VLM goal noise but non-zero.

### Phase 3 — Goal-noise resume training

Edit the env config to set `goal_position_noise_std=0.25` (mid-range
of the measured 0.2–0.5 m VLM noise). Resume from the DEPTH baseline:

```
python Scripts/train_strafer_navigation.py \
    --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
    --resume logs/rsl_rl/strafer_navigation/baseline_depth_no_noise/model_<step>.pt \
    --max_iterations <baseline_iter * 0.15>
```

Save the noised checkpoint as
`logs/rsl_rl/strafer_navigation/goal_noise_0.25/model_<step>.pt`.

### Phase 4 — Comparative evaluation

Run gym evaluation (`Scripts/play_strafer_navigation.py` with
`--n-rollouts 100` or equivalent) on both checkpoints, sweeping
`goal_position_noise_std` at evaluation time over `[0.0, 0.25, 0.5]` m:

| | baseline checkpoint | noised checkpoint |
|---|---|---|
| eval noise 0.0 m | (control: should be ≈ same) | (control: should be ≈ same) |
| eval noise 0.25 m | high oscillation expected | flat performance expected |
| eval noise 0.5 m | severe oscillation expected | graceful degradation expected |

Metrics per cell: median final-distance-to-goal across episodes,
mean-time-to-reach-tolerance, episode-success-rate.

The noised checkpoint should be:
- Within 5% of baseline at eval-noise 0.0 (no degradation under
  no-noise eval — the noise training shouldn't have damaged the base
  policy).
- Substantially better at eval-noise 0.25 and 0.5.

If the noised checkpoint regresses meaningfully at eval-noise 0.0,
either the resume budget was too long (over-trained on noise) or the
baseline wasn't actually converged. Investigate before declaring done.

## Acceptance criteria

### Config + runtime
- [ ] Three fields exist on `GoalCommandCfg` and
      `GoalCommandProcRoomCfg`: `goal_position_noise_std`,
      `goal_jump_probability_per_step`, `goal_jump_std`. All default to
      `0.0` (training default preserves current behavior; explicit
      opt-in for the noise pass).
- [ ] Unit test confirms all three modes per Phase 1's case list
      (zero noise = byte-identical; per-tick drift integrates;
      per-step jumps fire at expected frequency).

### Checkpoints
- [ ] Noised DEPTH checkpoint exists at
      `logs/rsl_rl/strafer_navigation/depth_goal_noise_0.25/model_<step>.pt`
      with a metadata note (in commit message + a sidecar `.json` if
      [`policy-export-tooling.md`](../../completed/policy-export-tooling.md) has
      landed first; otherwise just the commit message) identifying
      it as a goal-noise variant trained from `<baseline_path>` of
      the DEPTH ProcRoom env.

### Comparative evaluation
- [ ] PR description includes a comparison table: baseline vs noised
      checkpoint, evaluated at goal-noise stds `0.0`, `0.25`, `0.5`,
      with median final-distance-to-goal and episode-success-rate per
      cell.
- [ ] Noised checkpoint is within 5% of baseline at eval-noise 0.0
      (no-noise control), and shows lower final-distance-to-goal and
      higher success rate at eval-noise 0.25 and 0.5.

### Maintenance
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py)
  — `GoalCommandCfg` (line 363), `Ranges` subclass (line 402), the
  `GoalCommand` class (line 23) and the `GoalCommandProcRoom` subclass
  (line 413). The noise field belongs in the cfg classes; the
  perturbation belongs in `_update_command` or a new property the
  observation terms consume.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  — `goal_relative` and friends; whichever property the noise gets
  added to needs to be the one these read.
- VLM noise measurement: ground-truth-vs-observed analysis in
  `source/strafer_vlm/` if present; otherwise the `0.2–0.5 m` figure
  comes from the deferred-work entry's empirical estimate. Tighten
  this if better data exists.
- [`Scripts/train_strafer_navigation.py`](../../../../Scripts/train_strafer_navigation.py)
  — training entry; check that it accepts `--resume` (or equivalent
  flag) for the Phase 3 resume.

## Out of scope

- **Training the baseline checkpoint.** Operator-side ongoing work;
  this brief consumes whatever converged baseline exists.
- **Production export.** That's
  [`policy-export-tooling.md`](../../completed/policy-export-tooling.md). The two
  briefs run sequentially: train baseline → train noised → export
  → deploy.
- **Inference-side integration.** That's
  [`inference-package`](inference-package.md).
  The Jetson consumes whatever artifact is exported; this brief
  ensures the artifact it gets behaves well under VLM-noise.
- **Other domain-randomization knobs** (sensor noise, friction
  randomization, lighting, etc.). Each is its own targeted training
  pass; file separate briefs as needed. Goal-position noise is the
  one with measured deployment evidence behind it.
- **Hybrid-mode subgoal-noise training.** That's
  [`hybrid-mode`](../../parked/trained-policy/hybrid-mode.md);
  hybrid trains a different `PolicyVariant` against subgoal poses and
  has its own noise considerations.
