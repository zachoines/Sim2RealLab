# Bounded yaw strategy for the D555 IMU without a magnetometer

**Type:** investigation → task (filed-on-trigger)
**Owner:** Either (DGX measures + Jetson tunes)
**Priority:** P2 — pickup gated on either (a) the first long-horizon
multi-room mission completing on real hardware where yaw drift
materially affects arrival accuracy, or (b) RTAB-Map loop-closure
becoming unreliable on the lab carpet such that yaw drift is no
longer corrected at <30-second cadence.
**Estimate:** L (~2–4 days; measure drift envelope, pick a bounding
strategy, ship the chosen strategy, retest under load)
**Branch:** task/imu-yaw-drift-no-magnetometer

## Story

As a **real-robot operator running long-horizon multi-room missions
(>5 minutes between RTAB-Map loop closures)**, I want **the chassis's
absolute yaw to stay bounded by *something* — periodic SLAM-anchored
correction, sun-compass, line-of-known-feature anchor, or VIO**,
so that **between loop closures the robot's `map → odom` heading
doesn't drift far enough to violate `align_to_goal_yaw` or break
`_navigate_via_staging`'s costmap-bounded staging math**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/rotate-in-place-sim-clock-deadline.md](../../completed/rotate-in-place-sim-clock-deadline.md)
  — touches the rotation skill the yaw drift affects.

## Context

The D555 carries a BMI055 (accel + gyro) IMU. **No magnetometer.**
[`perception.launch.py:85-98`](../../../../source/strafer_ros/strafer_perception/launch/perception.launch.py)
runs `imu_filter_madgwick` with `use_mag: False`. Result: the
Madgwick filter has accel for tilt (pitch + roll well bounded by
gravity) but **no global yaw reference**. Yaw is integrated from
gyro alone; gyro bias drift is unbounded.

Field consequences:
- On the lab carpet at 25 °C, a typical BMI055-class MEMS gyro drifts
  ~0.1°–0.5°/min after a 60 s warmup. Over a 10-minute mission, that's
  1°–5° of unbounded yaw error if SLAM doesn't anchor.
- RTAB-Map loop closure resets the `map → odom` transform whenever
  the robot revisits a known place. Single-room missions hit a
  closure every ~30 s, so drift never accumulates. Multi-room or
  long-corridor missions can go several minutes between closures.
- The current
  [`rtabmap-cold-start-determinism.md`](../../active/reliability/rtabmap-cold-start-determinism.md)
  brief's "Increment map id" branch fires partially *because* the
  identity-pose teleport at startup is interpreted as a yaw
  discontinuity — bounded yaw would make that signal cleaner.
- Wheel-odom yaw (from differential encoder integration on a
  mecanum chassis) is *worse* than gyro: integration error compounds
  with slip, and slip on hardwood/carpet transitions is the dominant
  noise source. The Madgwick filter currently uses gyro as the
  primary heading source — that's correct *given* no magnetometer.

The sim lane is unaffected: the DGX bridge publishes an oracle
`/strafer/odom` from Isaac Sim's perfect ground truth, so this
brief is filed-on-trigger pending real-robot observation.

## Approach

This is an *investigation* first; only once option A vs B vs C is
decided does the brief promote to a task with a concrete implementation
plan.

### A. SLAM-driven yaw anchoring is enough; add measurement only

Measure actual drift between RTAB-Map closures on real-robot
missions. If 95th-percentile inter-closure drift is < 2° on typical
missions, **no fix needed**; the brief lands as a measurement-only
investigation under [`investigations/`](../../investigations/) and
the IMU stays as configured.

Adds: a `/imu/drift_telemetry` topic that logs
`{closure_id, yaw_drift_rad, since_last_closure_s}` to Foxglove.
Ship only this if A holds.

### B. External magnetometer add-on

Bolt a HMC5883L / QMC5883 / LIS3MDL breakout to the chassis (the
Strafer has spare GPIO on the front compute box). Re-enable Madgwick
with `use_mag: True`. Pros: cheap (~$5 part, ~half-day of bringup);
right-direction-of-fit. Cons: magnetic interference from the motors
themselves and the surrounding building's HVAC — these can introduce
heading errors larger than the original gyro drift. Production teams
report mixed results on indoor robots (see
[ROS Mobile Robotics survey](https://discourse.ros.org/) — search
for "magnetometer indoor compass"). **Validate carefully if chosen.**

### C. Visual-Inertial Odometry (VIO) replacing wheel odom

RealSense ships a T265-style VIO. The D555 itself is RGB-D, not VIO;
adding a T265 would be a second camera. Or run an
[OpenVINS](https://github.com/rpng/open_vins) / VINS-Mono node on the
D555's IMU + RGB. Cons: significant compute hit on Jetson Orin Nano
(~25% CPU at 30 fps); a third front-end on top of RTAB-Map and Nav2.
**Park as a follow-up** — only pick up if A measures bad drift and B
isn't viable.

### D. Pseudo-anchor: line-of-known-feature lock

When the robot sees a known wall (long straight feature in the
costmap), use its visual orientation as a yaw anchor. Mostly a
research direction; not standard. **Park as experimental.**

**Recommended sequence:** A first (measurement). If 95p drift > 2°
between closures, escalate to B (cheap external mag) with careful
interference testing. C and D are research, file separately if
needed.

## Acceptance criteria (investigation phase — option A)

- [ ] Real-robot mission set captured: at least 5 missions of >3 min
      duration each, on lab carpet, with `/imu/drift_telemetry`
      logging enabled.
- [ ] Drift statistics computed: p50, p95, p99 of yaw error between
      successive RTAB-Map loop closures.
- [ ] Decision recorded in the brief: A (measure only, ship the
      telemetry), B (file add-mag brief), C (file VIO investigation),
      or "no action — drift is benign."
- [ ] If A: `/imu/drift_telemetry` topic and Foxglove panel ship in
      this PR; brief moves to `completed/` with the measurement
      methodology recorded.
- [ ] If B/C: file the chosen follow-up under
      `parked/reliability/<new-brief>.md`; this brief moves to
      `completed/` with the investigation outcome.

## Investigation pointers

- [`source/strafer_ros/strafer_perception/launch/perception.launch.py:85-98`](../../../../source/strafer_ros/strafer_perception/launch/perception.launch.py) —
  Madgwick config; `use_mag: False`.
- [`source/strafer_ros/strafer_slam/launch/slam.launch.py`](../../../../source/strafer_ros/strafer_slam/launch/slam.launch.py) —
  the `Reg/Force3DoF=true` 2-DoF SLAM optimizer constrains yaw to a
  scalar; useful for measuring drift cleanly.
- RTAB-Map's loop-closure tracking via `/rtabmap/info` — read the
  `loop_closure_id` field for the closure-event timestamps used in
  the drift telemetry.
- Madgwick filter documentation:
  [`imu_filter_madgwick`](https://github.com/CCNYRoboticsLab/imu_tools).
- D555 IMU spec: [BMI055 datasheet](https://www.bosch-sensortec.com/products/motion-sensors/imus/bmi055/).

## Out of scope

- **Replacing the IMU with a higher-grade unit** (e.g., a tactical-
  grade fiber-optic gyro). Mechanical change; outside reliability
  brief lane.
- **Sim-side magnetometer simulation** for end-to-end testing of
  option B. The Isaac Sim bridge would need a `ROS2PublishImu`
  variant with magnetometer fields, which lives in
  `source/strafer_lab/` (DGX lane). File separately if option B
  ships.
- **Re-running RTAB-Map's optimizer with a different
  `Optimizer/Strategy`.** Reg/Force3DoF + g2o is fine for yaw alone;
  this brief doesn't touch the optimizer.
- **Wheel-odom-only fallback when IMU drops out.** Different failure
  mode; file separately if it surfaces.
