# `robot_localization` EKF: fuse encoder odom + IMU + SLAM into `/odom_fused`

**Type:** new feature (Jetson sensor-fusion node)
**Owner:** Jetson (`strafer_ros` — new `strafer_localization` package
or extension to existing perception layer)
**Priority:** P3 — filed-on-trigger. Encoder-only `/strafer/odom` is
the convention on indoor mobile-robot stacks and works well most of
the time. The known failure mode (wheel slip during collision, sharp
turns, low-friction floors) is bounded; only worth the multi-day EKF
integration if real-robot deployment shows it biting.
**Estimate:** M (~1 wk: install + minimal launch + covariance tuning
+ regression eval; tuning is the bulk).
**Branch:** task/robot_localization-ekf-fused-odom

**Pickup gate:** Filed-on-trigger. Do NOT pick up preemptively.

## Story

As a **Jetson operator running the deployed DEPTH MVP whose
`body_velocity_xy` observation comes from encoder-derived `/strafer/odom`**,
I want **a fused-odometry topic `/odom_fused` that combines the
wheel encoders, the D555 IMU, and (where available) the RTAB-Map
SLAM pose into a single body-velocity estimate that's robust to
wheel slip**, so that **the policy at deployment sees a body-velocity
signal that does NOT over-report during collision / low-friction
events and does NOT drift like raw IMU integration — closing the
wheel-slip gap that encoder-only odometry leaves open**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`observation-contract-cleanup.md`](../../completed/observation-contract-cleanup.md)
  — the predecessor that landed encoder-FK `body_velocity_xy`. Its
  docstring + the
  [`body-velocity-collision-test-rewrite`](../../completed/body-velocity-collision-test-rewrite.md)
  follow-up document the wheel-slip case explicitly.
- [`inference-package.md`](../../completed/inference-package.md)
  Phase 2 — the obs-pipeline section that reads `body_velocity_xy`
  from `odom.twist.linear.{x,y}` today. This brief swaps the source
  topic.
- [`imu-yaw-drift-no-magnetometer`](imu-yaw-drift-no-magnetometer.md)
  — sibling parked brief on the orientation side of the IMU drift
  problem. The EKF tuning here interacts with whatever option that
  brief picks.

## Trigger detail (un-park conditions)

File this brief if real-robot deployment of the DEPTH MVP shows **at
least one** of:

- **Collision-recovery loops.** Policy commands forward, chassis is
  stuck against an obstacle, but encoder-FK reports motion. The
  policy keeps commanding forward instead of backing off / re-routing.
  Failure rate ≥ 10% on missions that route past obstacles.
- **Low-friction-floor wobble.** On smooth surfaces (polished
  concrete, wet tile) the wheels slip continuously and encoder-FK
  reports a body velocity that's systematically higher than reality.
  The policy under-shoots arrival points because it thinks it's
  going faster than it is.
- **Sharp-turn over-rotation.** Mecanum diagonal motions have the
  worst encoder-FK fidelity (all four wheels contribute fractionally,
  slip is more visible). If oscillation around the goal on diagonal
  approaches becomes a measurable problem, the EKF + IMU correction
  is exactly the right intervention.
- **A targeted gym eval of `body_velocity_xy` accuracy** (the kind a
  domain-randomization audit follow-up would run) shows the encoder-
  FK error exceeds the policy's effective dead-band by ≥ 2×.

If none of these bite within ~3 months of DEPTH MVP shipping, the
likely correct action is to **delete this brief** — the encoder-only
path was sufficient and the EKF complexity wasn't justified.

## Context

### Why encoder-only odometry leaves a gap

The Strafer chassis driver
([`roboclaw_node.py`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py))
computes `/strafer/odom.twist.linear` via
`encoder_ticks_to_body_velocity()` — wheel ticks → angular vel →
`INVERSE_KINEMATIC_MATRIX` → body twist. Accurate when wheels roll
without slip, which holds for indoor flat floors at moderate speeds
≥ 99% of the time.

The slip-class failure modes are real but rare and bounded. Today
the inference-package reads `body_velocity_xy` from this topic
directly, so a slipping wheel translates to over-reported body
velocity into the policy obs.

### Why IMU-integrated velocity is NOT the fix

Direct IMU integration drifts to chassis-scale velocity error in
seconds:

| Error source | BMI055 typical | Velocity error after 1 s |
|---|---|---|
| White noise (~150 µg/√Hz @ 200 Hz) | σ ≈ 0.021 m/s² | ~0.02 m/s |
| Bias (~30 mg, varies with temperature) | 0.29 m/s² | **0.29 m/s** |
| 1° pitch error → false gravity component | 0.17 m/s² | 0.17 m/s |

The bias term diverges linearly without an external anchor. IMU
without fusion is not a viable standalone velocity source.

### Why an EKF is the right architecture

A Kalman-style filter exploits each sensor's strengths:

```
encoder-FK (50 Hz, low noise, drifts on slip only)   ┐
IMU (200 Hz, high frequency, drifts continuously)    ├─► EKF ─► /odom_fused
RTAB-Map pose (1–10 Hz, absolute anchor, sparse)     ┘
```

- Encoder is the high-confidence baseline for the no-slip case.
- IMU corrects the slip case in the seconds before SLAM updates —
  if encoders say "I'm accelerating" but IMU says "I'm not," the
  EKF down-weights the encoder estimate.
- SLAM anchors the long-term drift IMU would otherwise accumulate.

ROS 2's
[`robot_localization`](https://github.com/cra-ros-pkg/robot_localization)
is the canonical implementation. It handles all three sources, is
well-tested, and the integration burden is mostly config (covariance
matrices + which dimensions each sensor contributes).

## Approach

### Phase 1 — Minimal `robot_localization` integration (~2 days)

- Add the `robot_localization` apt dependency to the Jetson bringup.
- New package `source/strafer_ros/strafer_localization/` (or extend
  an existing perception package), with a launch + config:
  - `ekf_localization_node` consuming:
    - `/strafer/odom` (encoder-FK, twist channel only — pose is
      fused into RTAB-Map separately)
    - `/d555/imu/filtered` (orientation + angular vel + linear accel)
    - `/rtabmap/odom` (optional, when RTAB-Map is healthy — provides
      the global anchor)
  - `/odom_fused` published at the encoder rate (~50 Hz).
- Defaults: start with `robot_localization`'s reference config for
  a diff-drive robot, adapt covariance matrices to the Strafer chassis.

### Phase 2 — Covariance tuning (~2 days)

The EKF is only as good as its covariance estimates. Tune:
- **Encoder covariance** (`twist_covariance`): increase the diagonal
  values during sharp turns / mecanum-strafe to account for known
  worse slip behaviour on diagonal motions. Bench-measure if possible.
- **IMU covariance** (`linear_acceleration_covariance` /
  `angular_velocity_covariance`): match the BMI055 datasheet noise
  density translated to per-sample std at 200 Hz.
- **SLAM covariance**: read from RTAB-Map's published pose covariance;
  if not populated, set conservatively wide.

Validate on a recorded rosbag: pure straight-line motion (EKF agrees
with encoder), pure rotation (EKF agrees with IMU + encoder), wheel-
slip scenarios (EKF down-weights encoder where IMU disagrees).

### Phase 3 — Inference-package source swap (~½ day)

In
[`source/strafer_ros/strafer_inference/`](../../../../source/strafer_ros/strafer_inference/)
(the package this brief assumes exists post-`inference-package` Phase 1
ship), change the `body_velocity_xy` source from `/strafer/odom` to
`/odom_fused`. Single-line config change in the inference YAML, plus
a launch coordination that brings up the EKF before the inference
node subscribes.

If the sim-in-the-loop bridge does NOT have an EKF in front of
`body_velocity_xy`, the sim contract diverges from real again — file
a parallel sim-side brief to either run `robot_localization` in
`--mode bridge` or to swap the bridge's `body_velocity_xy` source
to a sim-equivalent fused channel. (Pragmatic option: the existing
encoder-FK in [`observations.py`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py)
already matches encoder-only `/strafer/odom`. Once a real-side EKF
ships, decide whether to mirror it in sim or accept the (now
narrower) wheel-slip gap as the only divergence.)

### Phase 4 — Regression eval against the trigger symptom

Whichever of the four trigger conditions filed this brief — re-run
that symptom against the fused-odom inference path. The pass bar is
"the symptom that fired the trigger goes away." Don't fish for new
regressions; one targeted comparison closes the loop.

## Acceptance criteria

### Plumbing

- [ ] `robot_localization` package installed and a launchable
      `ekf_localization_node` configuration committed.
- [ ] `/odom_fused` published at ~50 Hz when bringup launches.
- [ ] Inference YAML / launch wired so `body_velocity_xy` source is
      `/odom_fused` rather than `/strafer/odom`.

### Tuning evidence

- [ ] On a recorded straight-line rosbag, `/odom_fused.twist.linear.x`
      matches `/strafer/odom.twist.linear.x` within EKF noise (no
      visible drift, no oscillation).
- [ ] On a recorded wheel-slip rosbag (the trigger symptom or a
      bench reproduction), `/odom_fused` is closer to ground truth
      (measured or visually obvious chassis position) than
      `/strafer/odom`.

### Regression

- [ ] The trigger symptom (whichever of the four conditions fired)
      no longer reproduces under the fused-odom inference path. PR
      description includes before/after comparison.

### Cross-brief consistency

- [ ] Sim-side divergence handled per Phase 3's pragmatic decision
      (either mirror the EKF in sim, accept the narrowed gap, or
      file a sim-side mirror brief).
- [ ] `body-velocity-collision-test-rewrite`'s pinned test still
      passes — the new contract for sim-side `body_velocity_xy`
      (encoder-FK over-reports on collision) is unchanged by this
      brief unless Phase 3's sim mirror lands.

### Maintenance

- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- [`robot_localization` documentation](https://docs.ros.org/en/humble/p/robot_localization/)
  — `ekf_localization_node` + `ukf_localization_node` configs,
  covariance handling, sensor odometry / IMU input format.
- [`source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py)
  — current source of `/strafer/odom`; encoder-FK math via
  `encoder_ticks_to_body_velocity` from `strafer_shared.mecanum_kinematics`.
- [`source/strafer_ros/strafer_perception/launch/perception.launch.py`](../../../../source/strafer_ros/strafer_perception/launch/perception.launch.py)
  — current Madgwick orientation filter for `/d555/imu/filtered`.
  EKF inputs assume orientation-quaternion-populated IMU; this is
  already the case.

## Out of scope

- **VIO** (visual-inertial odometry from D555 stereo + IMU). Heavier
  compute, harder integration; defer until a clear failure mode
  shows EKF + SLAM is insufficient.
- **GPS / external anchor.** Not relevant indoors.
- **Re-deriving the policy obs contract.** This brief swaps the
  *source* feeding `body_velocity_xy`; it does not change what the
  policy sees by name or shape. Re-training the policy is unnecessary
  unless the new signal distribution differs measurably from
  encoder-only (it should not — the EKF is mostly the encoder under
  normal driving, and the slip-case correction is precisely what
  we want the policy to see).
- **Sim-side mirror of the EKF.** Flagged in Phase 3 as a decision
  point; if picked, file separately under
  `active/trained-policy/` or `active/sim-performance/` per the
  scope of the sim-side work.
