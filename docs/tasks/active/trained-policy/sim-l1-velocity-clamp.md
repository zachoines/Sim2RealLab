# Apply the L1 velocity clamp in sim's action pipeline to match the deployment safety clamp

**Type:** task / sim-to-real contract tightening
**Owner:** DGX (`strafer_lab` lane — modifies the action processing path in
`mdp/actions.py` and lifts the helper into `strafer_shared`)
**Priority:** P2 — tightens the sim-to-real contract for the trained-policy
backend. Not blocking any current mission shape; not a regression on
already-shipped behavior. Worth picking up before the next DGX training
run so the converged checkpoint trains against the same body-frame
command shaping the real chassis sees.
**Estimate:** S (~1 day: lift helper to `strafer_shared`, wire into sim's
action processing, unit-test parity with the Jetson-side call site, sweep
any existing checkpoints for re-training implications)
**Branch:** task/sim-l1-velocity-clamp

## Story

As a **policy author training DEPTH (or any future variant) in `strafer_lab`**,
I want **sim's action processing to apply the same L1 body-frame velocity
clamp the Jetson inference node applies before publishing
`/strafer/cmd_vel`**, so that **the policy's training distribution matches
the deployment-time command-shaping rather than the distorted per-wheel-
saturation distribution sim currently exposes** — and the architectural-win
acceptance in [`strafer-direct-sim-validation`](strafer-direct-sim-validation.md)
isn't undermined by a contract gap the validation rosbag wouldn't catch.

## Context bundle

Read these before starting:

- [`completed/inference-package.md`](../../completed/inference-package.md)
  — Phase 3 spec for the L1 clamp; the helper that's currently Jetson-side
  in [`strafer_inference/obs_pipeline.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py)
  (`l1_clamp_velocity`). This brief lifts that helper into
  `strafer_shared` so both sides import from one source of truth.
- [`source/strafer_shared/strafer_shared/mecanum_kinematics.py`](../../../../source/strafer_shared/strafer_shared/mecanum_kinematics.py)
  — the natural home for the helper. Already carries the wheel-cap
  invariant (`np.clip(wheel_vels, -MAX_WHEEL_ANGULAR_VEL, MAX_WHEEL_ANGULAR_VEL)`
  in `twist_to_wheel_velocities`) and the kinematic matrices. The L1
  body-frame clamp sits one step earlier in the chain.

## Why this matters (the actual sim-to-real gap)

The chassis cannot deliver max-forward + max-strafe simultaneously
because each mecanum wheel has a single motor with a per-wheel cap
of `MAX_WHEEL_ANGULAR_VEL` (~32.67 rad/s). At a corner-case command
like `(vx, vy, omega) = (1.55, 1.55, 0)`, the FL wheel inverse
kinematics ask for `(vx + vy) / WHEEL_RADIUS ≈ 64.6 rad/s` — roughly
2× the cap. The two sides handle that constraint **differently** today:

**Real (Jetson inference node):**
```python
# inference_node._on_tick:
vx, vy, omega = interpret_action(action)
vx, vy, omega = l1_clamp_velocity(
    vx, vy, omega,
    vel_cap_linear_m_s=self._vel_cap_linear,
    vel_cap_angular_rad_s=self._vel_cap_angular,
)
# (vx, vy) scaled jointly so |vx| + |vy| <= cap; heading preserved.
```
The L1 clamp scales `vx` and `vy` by the same factor. The commanded
direction is preserved; only the magnitude is reduced.

**Sim (`mdp/actions.py:MecanumWheelAction.process_actions`):**
The action is denormalized to body-frame `(vx, vy, omega)` and then
fed straight through `twist_to_wheel_velocities`, which applies
`np.clip(wheel_vels, ±MAX_WHEEL_ANGULAR_VEL)` per-wheel. **Per-wheel
clipping distorts the commanded heading**: each wheel saturates
independently, so the resulting motion drifts off the policy's
intended direction.

So the policy in sim learns "outputting `(0.99, 0.99, 0)` produces
distorted diagonal motion" — and adapts its policy to that
distortion. On real, that same output produces clean (slower)
diagonal motion. The two are different action→outcome relationships,
and the gap is invisible to any parity check that compares only the
observation vector (because the gap is on the *action* side, not
the observation side).

The [`strafer-direct-sim-validation`](strafer-direct-sim-validation.md)
brief's parity bounds compare assembled observation vectors at the
same sim timestamp, NOT the action-to-motion transform. It would not
catch this gap by construction. That's why it's worth tightening
the contract before the next training run rather than waiting for a
deployment-side failure to surface it.

## What's missing

Three changes:

1. **`l1_clamp_velocity` lives in the wrong place.** Currently in
   [`strafer_inference/obs_pipeline.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py)
   (Jetson-only). Both lanes need it; the function belongs in
   `strafer_shared.mecanum_kinematics` per the existing pattern (see
   `wheel_axis_signs`, `twist_to_wheel_velocities`, etc. — shared
   chassis invariants).

2. **Sim's `MecanumWheelAction.process_actions` doesn't call it.** Add
   the L1 clamp between the denormalization step and the per-wheel
   kinematics. Order matters: L1 clamp first (body-frame, preserves
   heading), then `twist_to_wheel_velocities` (per-wheel, clips any
   residual over-cap due to omega contribution).

3. **The Jetson side should re-import from `strafer_shared`** instead
   of keeping the local copy. Mechanical cleanup, but anchors the
   one-source-of-truth claim.

## Approach

### Phase 1 — Lift the helper (½ day)

In [`source/strafer_shared/strafer_shared/mecanum_kinematics.py`](../../../../source/strafer_shared/strafer_shared/mecanum_kinematics.py):

```python
def l1_clamp_twist(
    vx: float, vy: float, omega: float,
    *,
    vel_cap_linear_m_s: float,
    vel_cap_angular_rad_s: float,
) -> tuple[float, float, float]:
    """Cap (vx, vy) jointly under an L1 budget; cap omega independently.

    The chassis cannot reach max forward + max lateral simultaneously
    (per-wheel motor cap). Scaling (vx, vy) by the same factor keeps
    the commanded heading; clipping each axis independently would
    skew it. omega clamps independently because it routes through a
    different per-wheel sign-correction pathway.
    """
    ...
```

Copy verbatim from
[`strafer_inference/obs_pipeline.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py)'s
`l1_clamp_velocity`; rename to `l1_clamp_twist` to match the
existing `twist_to_wheel_velocities` naming convention.

In `strafer_inference/obs_pipeline.py`:

```python
from strafer_shared.mecanum_kinematics import l1_clamp_twist as l1_clamp_velocity
# Local alias keeps the existing import name in inference_node.py;
# remove the alias at the next inference-side touch.
```

OR remove the local copy entirely and re-import in `inference_node.py`.
The picking-up agent chooses; the alias path is smaller-diff.

Unit-test parity: existing
[`test_obs_pipeline.py::TestL1ClampVelocity`](../../../../source/strafer_ros/strafer_inference/test/test_obs_pipeline.py)
should pass unchanged against the lifted helper.

### Phase 2 — Wire into sim (½ day)

In [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py):

Locate `MecanumWheelAction.process_actions` (or whatever name the
action term has — verify against actual code at PR-opening time).
Between the action denormalization (raw `[-1, 1]^3` → physical units
via the same scaling `interpret_action` does on the Jetson side)
and the per-wheel kinematics call, insert:

```python
from strafer_shared.mecanum_kinematics import l1_clamp_twist
from strafer_shared.constants import (
    MAX_ANGULAR_VEL,
    MAX_LINEAR_VEL,
    NAV_VEL_SCALE,
)

# Resolve cap once at env-config time:
vel_cap_linear = MAX_LINEAR_VEL * STRAFER_ENV_VEL_SCALE
vel_cap_angular = MAX_ANGULAR_VEL * STRAFER_ENV_VEL_SCALE
# where STRAFER_ENV_VEL_SCALE is 1.0 for sim training (matches
# STRAFER_NAV_VEL_SCALE=1.0 the sim-bringup launch uses).

# Per-tick, per-env:
vx, vy, omega = l1_clamp_twist(
    vx, vy, omega,
    vel_cap_linear_m_s=vel_cap_linear,
    vel_cap_angular_rad_s=vel_cap_angular,
)
```

Vectorize for `num_envs`-parallel execution. Don't loop —
`l1_clamp_twist` is pure float math; either port it to torch /
numpy for vectorized application, or write a sibling
`l1_clamp_twist_batched` in `strafer_shared` for the sim path.
Recommend the latter so the deployment-time scalar form stays
readable.

### Phase 3 — Validation + re-training implication audit (½ day)

Unit tests in `source/strafer_lab/tests/test_action_clamp.py`:

- Corner-case input `(0.99, 0.99, 0.99)` denormalized to physical
  units: clamp produces the same `(vx, vy, omega)` as the Jetson-side
  `l1_clamp_velocity` call given the same caps. **Cross-lane parity
  at the helper level**, anchoring the one-source-of-truth claim.
- Per-wheel kinematics on the clamped output produce wheel velocities
  ≤ `MAX_WHEEL_ANGULAR_VEL`. No per-wheel saturation activates
  downstream of the clamp.
- Within-budget commands (e.g. `(0.5, 0.3, 1.0)`) pass through
  unchanged. The clamp is a no-op below the cap.

Re-training implication:

- [ ] If a DEPTH-direct checkpoint has already trained against the
      unclamped sim, the converged policy may exhibit a small
      distribution shift when the clamp is added. Document the
      checkpoint training-time SHA in the PR description.
- [ ] If no converged DEPTH-direct checkpoint exists yet, this brief
      lands first and the next training run automatically picks up
      the clamp. **Recommend this sequencing** — the L1 clamp is a
      contract item the training run should be exposed to from the
      start, not a post-hoc graft.
- [ ] If the architectural-win metric (≥ 1.0 m/s sustained) was
      measured against a pre-clamp checkpoint and this brief lands
      after, [`strafer-direct-sim-validation`](strafer-direct-sim-validation.md)'s
      E2E acceptance should re-run on the post-clamp checkpoint. Not
      a likely scenario given the current state (no deployable
      checkpoint exists), but flag it.

## Acceptance criteria

### Code

- [ ] `l1_clamp_twist` lives in `strafer_shared.mecanum_kinematics`.
- [ ] `strafer_inference` consumes it from there (via direct import
      or via the alias migration path).
- [ ] `MecanumWheelAction.process_actions` (or equivalent) in
      `strafer_lab` applies it between denormalization and per-wheel
      kinematics.

### Tests

- [ ] Cross-lane parity test: same `(vx, vy, omega)` input, same
      caps, the Jetson-side call and the sim-side call produce
      byte-identical output. Lives in `strafer_lab/tests/` (DGX has
      torch + numpy; the test imports from `strafer_shared` and
      runs on both lanes' test runners).
- [ ] No-op below cap: within-budget commands pass through unchanged.
- [ ] At-cap pass-through: `|vx| + |vy| == cap` exactly produces the
      same input.
- [ ] Above-cap scaling preserves heading: `atan2(vy, vx)` unchanged
      after clamp.
- [ ] Post-clamp per-wheel velocities ≤ `MAX_WHEEL_ANGULAR_VEL`.
- [ ] `colcon test --packages-select strafer_inference` still
      passes (the existing `TestL1ClampVelocity` suite migrates with
      the helper).

### Re-training implications

- [ ] PR description records: is there a converged DEPTH-direct
      checkpoint that trained against the pre-clamp behavior?
      If yes, document the re-training plan. If no, confirm the
      next training run picks up the clamp by default.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module — particularly the action-contract section of
      [`completed/inference-package.md`](../../completed/inference-package.md)
      and the safety-acceptance bullet in
      [`strafer-direct-sim-validation`](strafer-direct-sim-validation.md)
      — update those in the same commit.

## Investigation pointers

- [`source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py)
  — the current `l1_clamp_velocity` implementation. The math to lift
  is verbatim.
- [`source/strafer_ros/strafer_inference/test/test_obs_pipeline.py`](../../../../source/strafer_ros/strafer_inference/test/test_obs_pipeline.py)
  `TestL1ClampVelocity` — the existing test suite. Same assertions
  migrate to the sim-side test file with the helper.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py)
  — sim's action processing. Verify the actual class name and the
  exact location to insert the clamp at PR-opening time.
- [`source/strafer_shared/strafer_shared/mecanum_kinematics.py`](../../../../source/strafer_shared/strafer_shared/mecanum_kinematics.py)
  — the destination. Existing functions show the API convention.
- [`source/strafer_shared/strafer_shared/constants.py`](../../../../source/strafer_shared/strafer_shared/constants.py)
  — `MAX_LINEAR_VEL`, `MAX_ANGULAR_VEL`, `NAV_VEL_SCALE` are the
  cap inputs.

## Out of scope

- **Changing the magnitude of the cap.** This brief is about the
  *shape* of the clamp (joint vs per-wheel), not the chosen cap
  values. Sim continues to use `NAV_VEL_SCALE * MAX_LINEAR_VEL =
  MAX_LINEAR_VEL` at `STRAFER_NAV_VEL_SCALE=1.0` and real continues
  to use `NAV_VEL_SCALE * MAX_LINEAR_VEL = 0.5 * MAX_LINEAR_VEL` by
  default. The cap-values audit is downstream of
  [`domain-randomization-audit`](domain-randomization-audit.md).
- **Action smoothing / rate limits / jerk limits.** Separate
  contract concerns. If a need surfaces, file as its own
  sim-to-real-contract brief.
- **Re-training a converged DEPTH-direct checkpoint** to pick up
  the new clamp. The training pipeline will pick it up
  automatically on the next run; if a converged checkpoint exists
  and validation must re-run, that's a separate training brief.
- **The Jetson-side migration to import from `strafer_shared`** as a
  hard requirement. The alias path (local re-import keeping the
  existing inference-node call site unchanged) is acceptable for
  this brief's PR; cleanup at the next inference-side touch.
- **Sim's recurrent-reset behavior on `SubgoalCommand` resampling.**
  Different contract gap; audit it during [`subgoal-env`](subgoal-env.md)
  Phase 5's training run, file as its own brief if the PPO trainer
  doesn't fire `reset()` on the mid-episode goal-update path. Not
  this brief's scope.
- **TF staleness in sim training.** Folded into
  [`domain-randomization-audit`](domain-randomization-audit.md) as a
  dedicated gap-table row + Phase 1 measurement step + Phase 2
  config update (sibling to the control-rate-jitter knob — jitter
  randomizes *when* the policy ticks, TF staleness randomizes
  *when* the policy's spatial reference frame last updated). Pick
  it up there, not here.
