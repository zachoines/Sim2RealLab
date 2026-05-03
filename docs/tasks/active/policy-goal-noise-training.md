# Pre-deployment training pass: PPO with goal-position noise

**Type:** task / training
**Owner:** DGX (`strafer_lab` lane — env config + training run)
**Priority:** P2 — gates *deployable-policy quality* for VLM-grounded
goals; not blocking the inference plumbing in
[`strafer-inference-package.md`](strafer-inference-package.md), which
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
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [strafer-inference-package.md](strafer-inference-package.md) — the
  inference brief whose Phase 5 (end-to-end acceptance) becomes
  reliable once this training pass lands. Without goal-noise
  training, deployment with VLM-grounded goals shows oscillation that
  is *not* an inference-side bug.
- [policy-export-tooling.md](policy-export-tooling.md) — the next-step
  brief in the deployment chain. This brief produces the checkpoint;
  that brief converts it to a deployable artifact.

## Context

### The sim-to-deployment gap

VLM grounding produces goal poses with ~0.2–0.5 m localization error
(measured on Qwen2.5-VL-3B real-camera grounds; see
[`strafer_vlm/`](../../../source/strafer_vlm/) ground-truth-vs-observed
analysis if available). The current PPO training pipeline uses zero
goal noise — the policy sees pixel-perfect goal positions throughout
training.

At deployment, the policy receives a noisy goal and converges to the
*wrong* point with high precision. Once the goal updates (next VLM
ground), the policy has to re-converge from scratch, producing
visible oscillation around the true target.

This is a textbook training-distribution-shift problem with a textbook
fix: add Gaussian noise to the goal position during training, std
matching the deployment-noise distribution.

### What's missing

[`source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py:363`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py)
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

### Phase 1 — Add the config knob (½ day)

In [`commands.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py):

- Add a field to `GoalCommandCfg` (and the `GoalCommandProcRoom`
  subclass config):
  ```python
  goal_position_noise_std: float = 0.0
  """Per-step Gaussian noise (meters, isotropic xy) applied to the
  goal position seen by observation terms. 0.0 disables (training
  default). 0.25 m matches measured VLM grounding noise on
  Qwen2.5-VL-3B; deployment-prep training uses this value."""
  ```
- In `GoalCommand` (and `GoalCommandProcRoom`): the `command` property
  returns the *base* goal pose. Add a `noisy_command` property — or
  modify `command` to add per-step noise — that observation terms
  consume. The exact point of perturbation matters:
  - **Per-step noise (correct):** policy sees a different noisy goal
    every observation tick → trains to be robust to per-tick goal
    drift, which matches deployment.
  - **Per-resample noise (wrong):** policy sees the same noisy goal
    for the entire resample window → trains to converge to the wrong
    point precisely, doesn't help.
- Unit test in `source/strafer_lab/tests/test_commands.py` (or where
  command tests live): instantiate `GoalCommand` with
  `goal_position_noise_std=0.25`, assert the published goal pose
  varies across consecutive ticks with the expected variance, assert
  it equals the base goal when std=0.

### Phase 2 — Identify or train baseline checkpoint

Skip if a converged baseline already exists. Otherwise:

```
python Scripts/train_strafer_navigation.py \
    --task strafer_navigation \
    --policy_variant NOCAM
```

Train to convergence (target episode reward; episode-length-target
met). Save baseline as
`logs/rsl_rl/strafer_navigation/baseline_no_noise/model_<step>.pt`.

### Phase 3 — Goal-noise resume training

Edit the env config to set `goal_position_noise_std=0.25` (mid-range
of the measured 0.2–0.5 m VLM noise). Resume from the baseline
checkpoint:

```
python Scripts/train_strafer_navigation.py \
    --task strafer_navigation \
    --resume logs/rsl_rl/strafer_navigation/baseline_no_noise/model_<step>.pt \
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
- [ ] `goal_position_noise_std` field exists on `GoalCommandCfg` and
      `GoalCommandProcRoomCfg`, defaults to `0.0` (training default
      preserves current behavior; explicit opt-in for the noise pass).
- [ ] Unit test confirms:
      - At `std=0.0`, the published goal is byte-identical to the base
        goal across ticks.
      - At `std=0.25`, the published goal varies per-tick with the
        expected variance.

### Checkpoints
- [ ] Noised checkpoint exists at
      `logs/rsl_rl/strafer_navigation/goal_noise_0.25/model_<step>.pt`
      with a metadata note (in commit message + a sidecar `.json` if
      [`policy-export-tooling.md`](policy-export-tooling.md) has
      landed first; otherwise just the commit message) identifying it
      as a goal-noise variant trained from `<baseline_path>`.

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

- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py)
  — `GoalCommandCfg` (line 363), `Ranges` subclass (line 402), the
  `GoalCommand` class (line 23) and the `GoalCommandProcRoom` subclass
  (line 413). The noise field belongs in the cfg classes; the
  perturbation belongs in `_update_command` or a new property the
  observation terms consume.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  — `goal_relative` and friends; whichever property the noise gets
  added to needs to be the one these read.
- VLM noise measurement: ground-truth-vs-observed analysis in
  `source/strafer_vlm/` if present; otherwise the `0.2–0.5 m` figure
  comes from the deferred-work entry's empirical estimate. Tighten
  this if better data exists.
- [`Scripts/train_strafer_navigation.py`](../../../Scripts/train_strafer_navigation.py)
  — training entry; check that it accepts `--resume` (or equivalent
  flag) for the Phase 3 resume.

## Out of scope

- **Training the baseline checkpoint.** Operator-side ongoing work;
  this brief consumes whatever converged baseline exists.
- **Production export.** That's
  [`policy-export-tooling.md`](policy-export-tooling.md). The two
  briefs run sequentially: train baseline → train noised → export
  → deploy.
- **Inference-side integration.** That's
  [`strafer-inference-package.md`](strafer-inference-package.md).
  The Jetson consumes whatever artifact is exported; this brief
  ensures the artifact it gets behaves well under VLM-noise.
- **Other domain-randomization knobs** (sensor noise, friction
  randomization, lighting, etc.). Each is its own targeted training
  pass; file separate briefs as needed. Goal-position noise is the
  one with measured deployment evidence behind it.
- **Hybrid-mode subgoal-noise training.** That's
  [`strafer-inference-hybrid-mode.md`](strafer-inference-hybrid-mode.md);
  hybrid trains a different `PolicyVariant` against subgoal poses and
  has its own noise considerations.
