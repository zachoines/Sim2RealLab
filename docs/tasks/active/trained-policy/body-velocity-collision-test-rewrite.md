# Rewrite `test_body_velocity_drops_on_collision` for the encoder-FK contract

**Type:** bug / test refactor
**Owner:** DGX (`strafer_lab/test/sensors/`)
**Priority:** P2 — the test fails on `main` after
[`observation-contract-cleanup`](../../completed/observation-contract-cleanup.md)
shipped. The failure is the new contract working as designed (encoder-FK
`body_velocity_xy` correctly reports wheel-slip motion during collision),
but a red `imu` suite makes CI noise indistinguishable from real
regressions. Rewriting the test under the new contract restores the
signal.
**Estimate:** S–M (~½ day: audit any other tests carrying the old
assumption; rewrite the one failing test to actively pin the new
behaviour; add a critic-side guard if useful).
**Branch:** task/body-velocity-collision-test-rewrite

## Story

As a **DGX operator reading CI output**, I want **the sensor test suite
to pass cleanly under the encoder-FK `body_velocity_xy` contract**, so
that **a future test failure here is a real regression rather than the
known mismatch between the old test's premise and the new obs
semantics**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [observation-contract-cleanup.md](../../completed/observation-contract-cleanup.md)
  — the predecessor ship that changed `body_velocity_xy` from sim
  ground truth (`root_lin_vel_b`) to encoder-FK over the same
  joint-velocity tensor the chassis driver reads. Its docstring on
  the obs function is the source-of-truth for the new contract.

## Context

### The failing test and why it fails

[`source/strafer_lab/test/sensors/test_imu_collision.py:335`](../../../../source/strafer_lab/test/sensors/test_imu_collision.py)
`TestCollisionProprioception::test_body_velocity_drops_on_collision`
asserts that body velocity drops on collision (Welch's t-test, one-
sided: free speed > collision speed).

The test's docstring states the premise:

> "When the robot hits a kinematic obstacle, its body velocity drops
> to near-zero **even though the action commands full forward and the
> wheels continue spinning**. This velocity-action mismatch is a
> strong proprioceptive collision signal available to the policy."

That premise is true for ground-truth body velocity (`root_lin_vel_b`).
It is **not** true for encoder-FK body velocity — wheel slip means
`body_velocity_xy = (r/4) · Σ(noisy_encoder_ω)` keeps reporting motion
while the chassis is physically stuck. The failure numbers reproduce
exactly this: collision speed ≈ 0.066 m/s, free speed ≈ 0.034 m/s
(collision > free), the wheel-slip signature.

The new contract is the **right** behavior — `/strafer/odom` on the
real robot does the same thing — and the test must be rewritten to
match.

### What's still valid

`test_collision_imu_mean_differs_from_free` (same file) passes — IMU
acceleration drops on collision because the chassis stops accelerating.
That's the real proprioceptive collision signal both in sim and on the
real robot. The failing test's own docstring acknowledges this as a
parallel signal.

A reviewed audit of other tests reading `body_velocity_xy` from the
policy obs vector or via `body_velocity_xy(env)` directly:

| Test | Assumption | Affected by encoder-FK rewrite? |
|---|---|---|
| `test/sensors/test_imu_collision.py::test_body_velocity_drops_on_collision` | body vel drops on collision | **Yes — this brief** |
| `test/observations/test_obs_functions.py::test_body_velocity_xy_shape` | output is `(N, 2)` | No (shape only) |
| `test/observations/test_obs_functions.py::test_body_velocity_near_zero_when_stationary` | mean speed < 0.5 m/s under zero action | No — stationary wheels → zero ticks → zero FK output |
| `test/env/test_obs_contract.py` (multiple) | field name / dim / scale match the contract | No (contract surface unchanged) |

So scope is bounded to the one test.

### What signal does the policy *actually* have for collision

The same proprioceptive cues the real robot has:

- **IMU accel deceleration** — body stops accelerating; mass·g-only.
- **Encoder-FK velocity high while commanded velocity is also high
  but the goal-pose / SLAM-pose stops advancing** — i.e. the policy
  has to compare `body_velocity_xy` (high) against `goal_distance`
  (not shrinking) over a temporal window. That's a recurrent-state
  inference, not a single-tick signal.

The DEPTH policy's GRU (`rnn_hidden_dim=128`) has the capacity to
learn the second pattern. The first pattern is a single-tick obs
($\sim$ IMU magnitude) and is already exposed to the policy.

## Approach

### Phase 1 — Audit (½ hour)

Re-grep `source/strafer_lab/test/` for any other test that reads
`body_velocity_xy` from the policy obs vector AND makes a value
assertion (not just shape / dim / scale):

```bash
grep -rn "body_velocity_xy\|BODY_VEL_SLICE\|root_lin_vel_b" source/strafer_lab/test/
```

If anything new turns up, decide delete / rewrite / move-to-critic
per the table below.

### Phase 2 — Rewrite `test_body_velocity_drops_on_collision`

Pick **one** of three patterns. Recommendation: **(B) actively pin
the new contract**, since that gives a future maintainer a positive
assertion to break against rather than a silent removal.

**(A) Delete the test.** Cleanest in line count, but loses the
collision-coverage shape entirely. Don't pick this unless Phase 1 shows
the IMU-side collision test fully subsumes the coverage.

**(B) Rewrite to actively pin the new contract (recommended).**
Replace the body of the test so it asserts the **opposite** direction
under collision: encoder-FK `body_velocity_xy` is **non-zero or
elevated** during collision (wheels slipping), demonstrating that the
policy cannot use this signal for collision detection. Use the same
fixture (`collision_env`, `_place_obstacle_in_front`, drive forward,
collect after impact). The docstring becomes a documentation of the
contract:

```python
def test_body_velocity_xy_does_not_drop_on_collision(self, collision_env):
    """`body_velocity_xy` is encoder-derived FK over the same joint-
    velocity tensor that produces /strafer/odom on the real robot. On
    collision the chassis stops moving but the wheels continue
    spinning, so encoder-FK reports wheel-slip motion rather than
    dropping to zero. This test pins that behaviour so a future change
    that silently reverts to ground-truth body velocity is caught.

    The IMU-based collision signal lives in
    `test_collision_imu_mean_differs_from_free` (same class) — that is
    the proprioceptive signal the policy can actually use.
    """
    # ... same fixture + driving as before ...
    # collision speed must be >= free speed * 0.5 (slip means the
    # wheels keep contributing motion to FK), not <= free speed * 0.5
    # as the old test asserted.
```

The exact threshold needs a tuning pass — but the assertion direction
flips. Set the bound generously so it pins the contract without being
brittle to env-load noise. Effect-size assertion is optional; the
direction is the load-bearing piece.

**(C) Move ground-truth-body-velocity coverage to the critic side.**
`mdp.privileged_ground_truth` (already in
[`observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py))
still returns `root_lin_vel_b[:, :2]` — the critic still sees ground-
truth body velocity. A new critic-side test could assert the original
drop-on-collision behaviour against `privileged_ground_truth(env, ...)`
output (slice `[:2]`). This preserves the "body actually stops on
collision" assertion in the test suite without lying about the policy
contract. Bundle with (B) if you want both pins.

Recommendation: **(B) standalone**, with (C) only if Phase 1 surfaces
additional value in the ground-truth signal beyond what the IMU test
already covers.

### Phase 3 — Run the suite + commit

```bash
$ISAACLAB -p source/strafer_lab/run_tests.py imu
```

Expect: 4/4 passing under the rewritten contract.

If Phase 1 surfaced additional affected tests, run the broader sensor
+ observation suites too:

```bash
$ISAACLAB -p source/strafer_lab/run_tests.py imu sensors observations
```

## Acceptance criteria

### Tests

- [ ] `python run_tests.py imu` passes 4/4 on DGX (was 3/4 before).
- [ ] The rewritten test (`test_body_velocity_xy_does_not_drop_on_collision`
      or whatever name you pick) has a docstring that documents the
      encoder-FK contract for a future reader, not a comment about
      what *used* to be tested.
- [ ] If Phase 1 surfaced additional tests, each has a delete /
      rewrite / move decision recorded in the PR description.

### Cross-brief consistency

- [ ] If you added a critic-side test under pattern (C), the
      Jetson-side `inference-package` brief gets a one-line note that
      ground-truth body velocity is still available *to the critic
      only*, not to the deployed policy — so collision-handling logic
      in the inference node cannot assume it.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`source/strafer_lab/test/sensors/test_imu_collision.py:335`](../../../../source/strafer_lab/test/sensors/test_imu_collision.py)
  — the failing test. Fixture setup (`collision_env`,
  `_move_obstacle_far_away`, `_place_obstacle_in_front`) is reusable.
- [`source/strafer_lab/test/sensors/test_imu_collision.py:275`](../../../../source/strafer_lab/test/sensors/test_imu_collision.py)
  — `test_collision_imu_mean_differs_from_free`, the sibling test
  that exercises the *valid* collision signal. Pattern reference.
- [`source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
  — `body_velocity_xy` (encoder-FK, policy obs) and
  `privileged_ground_truth` (ground-truth body vel, critic obs).
  Same file; the split is the asymmetric actor-critic convention.

## Out of scope

- **Adding new proprioceptive collision signals to the obs vector.**
  This brief audits and rewrites a test; it doesn't propose new obs
  fields. If a deployment regression shows the IMU signal alone is
  insufficient for collision recovery, file that separately.
- **EKF-fused odometry to recover ground-truth-quality body velocity
  for the policy.** That's the architecturally-right long-term fix,
  parked at
  [`robot_localization-ekf-fused-odom`](../../parked/reliability/robot_localization-ekf-fused-odom.md)
  on trigger "DEPTH MVP deployment shows wheel-slip-driven failures".
  Don't pre-empt — this brief is only about test correctness.
