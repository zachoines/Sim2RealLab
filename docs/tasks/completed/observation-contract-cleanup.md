# Audit and close sim-vs-real gaps in the policy observation contract

**Status:** Shipped 2026-05-24 in `781b636` (DGX). `body_velocity_xy`
now does encoder-derived FK from the same joint-velocity tensor the
real chassis driver reads when producing `/strafer/odom`. The "same
encoder noise *sample*" subgoal (acceptance criterion #3) requires a
per-tick noised-ticks cache plus a policy/critic obs-function split;
filed as the [`encoder-noise-shared-sample`](encoder-noise-shared-sample.md)
follow-up rather than expanded in scope here.

**PR:** https://github.com/zachoines/Sim2RealLab/pull/56

**Type:** bug / refactor
**Owner:** DGX (`strafer_lab` env config) + Either
(`strafer_shared.policy_interface` field list)
**Priority:** P1 — silent sim-to-real bug. The `policy` observation
group contains at least one field whose value in sim is sourced from
ground truth and whose real-robot equivalent is derived from a
different signal chain. The asymmetric actor-critic convention catches
*explicit* privileged info (critic-only fields), but a sim-only
ground-truth source feeding a `policy` field is the failure mode the
convention does *not* protect against, because the field exists on both
sides — its *meaning* just diverges. This must be resolved before
[`inference-package`](inference-package.md) ships, otherwise the
deployment-ready DEPTH checkpoint trains on one distribution and is
asked to act on another.
**Estimate:** S–M (~1 day: identify each sim-real-divergent field,
re-derive it on the sim side from the same signal chain the real robot
uses, and add a parity unit test).
**Branch:** task/observation-contract-cleanup

## Story

As a **DGX operator promoting a DEPTH PPO checkpoint to real-robot
deployment**, I want **every field in `PolicyVariant.DEPTH` /
`PolicyVariant.NOCAM` to be computed from the same signal chain in
training as the inference node will use at deployment**, so that **the
checkpoint's learned input distribution matches what the real robot
feeds it, instead of being subtly biased toward whatever sim-only
shortcut the env config happens to take today**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [inference-package.md](inference-package.md) — Phase 2's obs-parity
  acceptance criterion ("NOCAM-fields obs parity ≤ 1e-5 max abs delta")
  is what this brief makes achievable. Today the bound is technically
  met by the training env's sim-ground-truth `body_velocity_xy` and
  the inference node's odom-derived `body_velocity_xy` being
  ***numerically close*** but ***distributionally different*** — they
  agree on a single timestep but not on the noise profile across
  steps.
- [goal-noise-training.md](goal-noise-training.md) — sister brief for
  goal-pose noise; this brief covers the velocity / proprioception
  fields.
- [domain-randomization-audit.md](domain-randomization-audit.md) —
  sister brief for dynamics/sensor noise *ranges*. This brief is about
  the *source* of each obs field, not its noise scale.

## Context

### The shape of the gap

[`policy_interface.py`](../../../../source/strafer_shared/strafer_shared/policy_interface.py)
declares `_NOCAM_FIELDS`:

```python
_NOCAM_FIELDS = (
    ObsField("imu_accel", 3, ...),
    ObsField("imu_gyro", 3, ...),
    ObsField("encoder_vels_ticks", 4, ...),
    ObsField("goal_relative", 2, ...),
    ObsField("goal_distance", 1, ...),
    ObsField("goal_heading_to_goal", 1, ...),
    ObsField("body_velocity_xy", 2, ...),  # ← see below
    ObsField("last_action", 3, ...),
)
```

Each field has a sim implementation (in `mdp/observations.py`) and a
real-robot implementation (in the inference node, per the inference
brief). For most fields these match in spirit:

- `imu_accel` / `imu_gyro` — sim reads `wp.to_torch(sensor.data.lin_acc_b)`
  with noise applied per the REAL contract; real reads
  `/d555/imu/filtered`. Same physical quantity, same noise channel.
- `encoder_vels_ticks` — sim reads `joint_vel * RADIANS_TO_ENCODER_TICKS`
  with encoder noise applied per the REAL contract; real reads
  `/strafer/joint_states.velocity` via `wheel_vels_to_ticks_per_sec`.
  Same.
- `goal_relative` / `goal_distance` / `goal_heading_to_goal` — sim
  derives from `env.command_manager.get_command(...)` which is the
  *exact ground truth goal* in world frame, transformed to body frame
  via `quat_apply_inverse`. Real derives from a noisy VLM-grounded
  goal in `map` frame, transformed via TF `map → base_link`
  (RTAB-Map SLAM). The VLM noise dimension is the
  [`goal-noise-training`](goal-noise-training.md) brief's scope; the
  *SLAM drift* dimension is not currently in any brief — see Finding
  3 below.
- `last_action` — sim reads `env.action_manager.action` (raw policy
  output before clamp); real caches the previous tick's raw policy
  output. Aligned (the inference brief gets this right).

The outlier is `body_velocity_xy`:

#### Finding 1 — `body_velocity_xy` is sim ground truth, not encoder-derived odom

[`observations.py:442-456`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py):

```python
def body_velocity_xy(env: ManagerBasedEnv) -> torch.Tensor:
    """Get body-frame linear velocity (vx, vy) from simulation ground truth.

    Equivalent to running forward kinematics on encoder readings in the real
    system (encoder_ticks_to_body_velocity).
    """
    robot = env.scene["robot"]
    # root_lin_vel_b is body-frame linear velocity (x, y, z)
    return wp.to_torch(robot.data.root_lin_vel_b)[:, :2]
```

The docstring acknowledges "equivalent to running forward kinematics on
encoder readings" — but the implementation reads `root_lin_vel_b`,
which is the sim's integrated rigid-body velocity, *not* the encoder
FK estimate. These differ by:

- The encoder noise model (Real contract injects 2% velocity noise,
  missed-tick errors, quantization). Sim ground truth has none.
- Wheel slip (a real wheel slips; encoders over-report distance, FK
  over-reports velocity). Sim ground truth knows the actual chassis
  velocity through PhysX.
- Wheel radius mis-calibration (real wheels wear; the sim's wheel
  radius is exact). FK is biased by wheel-radius error.

On the real robot, the inference brief Phase 2 says:
> body_velocity_xy from odom.twist.linear.{x,y} — odom frame
> (body-convention). Pass-through.

But `/strafer/odom` on the Jetson is produced by the chassis driver
from encoder readings via mecanum FK. So at deployment the policy
sees `body_velocity_xy ≈ encoder_FK(noisy_encoder_ticks)`. At
training the policy sees `body_velocity_xy = sim_ground_truth`. The
policy learned to *trust* `body_velocity_xy` against ground-truth
behavior; at deployment it's correlated with `encoder_vels_ticks`
(both are encoder-derived) but the policy doesn't know this.

Worst case: the policy learned a small reliance on
`body_velocity_xy` to disambiguate cases where the encoder noise is
high (e.g., wheel slip episodes). At deployment, encoder noise and
`body_velocity_xy` noise are *the same noise*, not independent. The
policy's noise-cancelling behavior breaks.

#### Finding 2 — Three goal-derived fields encode redundant information from the same source

`goal_relative` (2) + `goal_distance` (1) + `goal_heading_to_goal` (1)
are all derived from `command[:, :2]` (the ground-truth goal in sim,
or the SLAM-frame VLM-grounded goal in deployment). Two of the three
are mathematically derivable from `goal_relative`:

- `goal_distance = norm(goal_relative)`
- `goal_heading_to_goal = atan2(goal_relative[1], goal_relative[0])`

Including all three doesn't *leak* information — they share a single
upstream — but it gives the policy three weighted ways to read the
same noisy upstream signal. If SLAM-frame drift makes `goal_relative`
shift by 5 cm, all three fields shift correlatedly; the policy's
"average across the three" trick that helps during training doesn't
help at deployment (correlated noise vs. independent).

This is a minor regularization concern, not a correctness bug, but
worth noting alongside Finding 1.

#### Finding 3 — SLAM drift on the goal is not modeled in training

[`goal-noise-training`](goal-noise-training.md) addresses
**VLM grounding noise** at goal-source. But the goal is in `map`
frame, and `map → base_link` TF is produced by RTAB-Map SLAM. SLAM
drift on the Jetson exhibits two distinct failure modes:

1. **Slow drift** (~1 cm/s during long traversals): a continuous
   small-magnitude shift, well-modeled by per-tick Gaussian goal
   noise.
2. **Loop-closure jump** (~10–50 cm discrete jump when RTAB-Map
   finds a loop): a discontinuous large-magnitude shift. NOT
   modeled by Gaussian noise — discrete events.

The `goal-noise-training` brief's per-step Gaussian model only
captures (1). The per-reset offset (existing `randomize_goal_noise`
event with `noise_std=0.15` at mode="reset") doesn't capture (2)
either, because it only fires at episode reset.

To match deployment, the training distribution should include
*occasional discrete goal jumps* — e.g., with low probability per
step, the goal shifts by `N(0, 0.3 m)`. This is a "perturbation
event" pattern, not a continuous noise pattern.

This is a cross-brief finding; documented here so the loader fix
covers it, and cross-referenced from `goal-noise-training` for the
training-side fix.

#### Finding 4 — `imu_orientation` and `imu_projected_gravity` are sim-only and not in the policy obs (verified)

[`observations.py:278-296`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
defines `imu_orientation` (quaternion) and `imu_projected_gravity`
(unit vector). Neither appears in any `ObsCfg_*.PolicyCfg` group, only
in unused options. Confirmed clean — listed here so a future audit
doesn't have to re-verify.

### Why fix in training, not at inference

Two viable fixes:

A. **Inference-side**: ignore the policy's `body_velocity_xy` reading
   at deployment, pass zeros. Loses the field but matches a known
   training distribution.

B. **Training-side** (this brief): make sim `body_velocity_xy` use
   encoder FK, matching what the real robot does.

Option A wastes a field, hides the issue from future re-trains, and
biases the deployment distribution from "policy sees an unusable
field" toward "policy must learn to ignore a zero field" — different
broken state, not a fix.

Option B is the principled fix. The fact that the docstring of
`body_velocity_xy` already says "equivalent to running forward
kinematics on encoder readings" suggests the original author meant to
do this and the implementation drifted.

### Why this can't wait for the next training round

The deployable checkpoint
([`inference-package`](inference-package.md) Phase 5 acceptance) is
gated on a fresh DEPTH ProcRoom training run. Fixing the obs
contract *before* that run lands ships the fix and the checkpoint at
the same time. Fixing it *after* requires re-training from scratch.

## Approach

### Phase 1 — Re-implement `body_velocity_xy` as encoder-derived FK

In
[`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py),
replace the body-frame ground-truth read with an encoder-FK
computation that mirrors what the Jetson `/strafer/odom` produces:

```python
def body_velocity_xy(env: ManagerBasedEnv) -> torch.Tensor:
    """Get body-frame (vx, vy) from encoder-derived FK.

    Mirrors the real-robot computation: encoder velocities → wheel
    angular velocities → mecanum FK → body twist. Matches the signal
    chain the Jetson inference node sees via /strafer/odom.

    Encoder noise is applied upstream by the wheel_encoder_velocities
    obs term + the REAL contract's encoder noise model, so the noise
    flows into this output by construction.
    """
    raw_ticks = wheel_encoder_velocities(env)  # (num_envs, 4)
    # Convert ticks/s → rad/s and run mecanum inverse FK
    wheel_rad_s = raw_ticks * ENCODER_TICKS_TO_RADIANS
    # Apply the inverse of the kinematic matrix (same one in actions.py)
    body_xy = wheel_rad_s @ KINEMATIC_INVERSE[:2, :].T  # (num_envs, 2)
    return body_xy
```

Where `KINEMATIC_INVERSE` is the pseudoinverse of
`KINEMATIC_MATRIX` (the matrix that maps `[vx, vy, omega] → wheel
rad/s`). Both live in `strafer_shared.mecanum_kinematics`; no new
dependency, the import is already in `actions.py`.

Acceptance check: with `IDEAL_SIM_CONTRACT` (zero encoder noise),
the new `body_velocity_xy` should match the old sim-ground-truth
`root_lin_vel_b[:, :2]` to within `1e-4` per axis (wheel slip / FK
round-off only).

With `REAL_ROBOT_CONTRACT`, the new value has the encoder noise band
applied — exactly as the real robot would see it.

### Phase 2 — Parity unit test

Add `source/strafer_lab/tests/test_obs_contract_parity.py`:

- Construct a tiny `ManagerBasedEnv` with a stub robot that has known
  wheel velocities (e.g., `(10.0, 10.0, 10.0, 10.0)` rad/s →
  body velocity ≈ pure forward).
- Call `body_velocity_xy(env)` under IDEAL — assert matches
  forward-kinematics result.
- Call `body_velocity_xy(env)` under REAL — assert the *expectation*
  matches and the *variance* matches the encoder noise model.
- Cross-check against `wheel_encoder_velocities(env)` to confirm both
  obs terms share the same encoder noise sample (correlated, not
  independent — that's the whole point).

### Phase 3 — Update the inference brief's obs-parity acceptance

[`inference-package`](inference-package.md)'s Phase 2 says:

> NOCAM-fields obs parity: with a recorded sim-in-the-loop rosbag,
> the inference node's assembled NOCAM-portion (first 19 dims)
> matches the gym-env obs at the same sim timestamp within float32
> noise (≤ 1e-5 max abs delta).

After this brief lands, the parity test is meaningful — both sides
compute `body_velocity_xy` from the same encoder chain. The 1e-5
tolerance should hold. Add a one-line note to the inference brief's
Context section pointing at this brief as the load-bearing predecessor
for that acceptance.

### Phase 4 — Cross-reference goal-noise-training for Finding 3

Edit [`goal-noise-training`](goal-noise-training.md) to include
loop-closure-jump perturbation in addition to per-tick Gaussian. Add
a `goal_jump_probability_per_step` + `goal_jump_std` config knob
alongside `goal_position_noise_std`. The implementation pattern:

```python
# Per-step Gaussian (already proposed by goal-noise-training)
goal_noisy = goal + N(0, sigma_per_step)

# Per-step low-prob discrete jump (this brief's addition)
if rand() < goal_jump_probability_per_step:
    goal_noisy += N(0, sigma_jump)
```

Don't duplicate the implementation here — file the cross-ref and let
goal-noise-training own it.

## Acceptance criteria

### Code

- [ ] `body_velocity_xy(env)` in
      [`observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
      returns encoder-derived FK output, not `root_lin_vel_b`.
- [ ] Under `IDEAL_SIM_CONTRACT`, new `body_velocity_xy` matches the
      old ground-truth implementation to within 1e-4 (sanity check
      that FK is correct; only wheel-slip / round-off differences).
- [ ] Under `REAL_ROBOT_CONTRACT`, new `body_velocity_xy` carries the
      same encoder noise sample as `wheel_encoder_velocities` (i.e.
      they are no longer independent noise channels).

### Tests

- [ ] `python -m pytest source/strafer_lab/tests/test_obs_contract_parity.py`
      passes — covers both IDEAL and REAL contracts.

### Cross-brief consistency

- [ ] [`inference-package.md`](inference-package.md)'s Phase 2
      Context section cross-references this brief as the prerequisite
      for the 1e-5 obs-parity tolerance.
- [ ] [`goal-noise-training.md`](goal-noise-training.md) updated to
      include loop-closure-jump perturbation per Finding 3, with a
      cross-reference back to this brief.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py:442-456`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  — `body_velocity_xy` current implementation.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py:125-159`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  — `wheel_encoder_velocities` (the encoder-FK input).
- [`source/strafer_shared/strafer_shared/mecanum_kinematics.py`](../../../../source/strafer_shared/strafer_shared/mecanum_kinematics.py)
  — `KINEMATIC_MATRIX` and (if not yet present) the pseudoinverse;
  same source-of-truth the actions.py uses.
- [`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py:679-712`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py)
  — `ObsCfg_NoCam_Realistic` — where `body_velocity_xy` enters the
  policy obs group (scaled by `_BODY_VEL_SCALE`).
- ANYmal upstream observation contract
  ([legged_robot.py](https://github.com/leggedrobotics/legged_gym))
  — peer reference for proprioception field sourcing; consistently
  uses encoder/IMU-only chains in the `policy` group, never
  rigid-body ground truth.

## Out of scope

- **Adding new observation fields.** This brief audits existing
  fields, doesn't propose new ones.
- **Removing `body_velocity_xy` from the obs.** That's an option but
  changes obs_dim, breaking all existing checkpoints. The fix here
  is in-place re-derivation, not removal.
- **`PolicyVariant.NOCAM_SUBGOAL` audit.** Filed under
  [`subgoal-env`](subgoal-env.md). NOCAM_SUBGOAL inherits NOCAM's
  field list per its current design, so the same fix flows through —
  no separate work.
- **Goal-pose noise modeling.** That's
  [`goal-noise-training`](goal-noise-training.md). This brief
  cross-refs and updates that brief; doesn't re-implement.
- **Sensor noise tuning.** That's
  [`domain-randomization-audit`](domain-randomization-audit.md).
  This brief is about *which signal chain* feeds each obs field, not
  the noise magnitude on that chain.
