# Publish `/strafer/joint_states` + `/d555/imu/filtered` from the sim bridge

**Type:** bug
**Owner:** DGX agent
**Priority:** P2
**Estimate:** S
**Branch:** task/bridge-imu-joint-states

## Story

As a **sim-in-the-loop operator**, I want **the DGX bridge to publish the
robot's wheel joint state and the D555 IMU on the same topics the real
robot uses**, so that **the trained-policy inference node reaches `ready`
and can run `strafer_direct` / hybrid missions in sim instead of
zero-twisting on stale observations**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/conventions.md](../../context/conventions.md)

## Context

**Symptom.** With `STRAFER_NAV_BACKEND=hybrid_nav2_strafer` (or
`strafer_direct`) and a real policy artifact loaded, `strafer_inference`
launches and advertises `/strafer_inference/navigate_to_pose`, but the node
never becomes `ready`: `ros2 param get /strafer_inference/strafer_inference
ready` → `False`, and the watchdog logs
`stale=['goal','imu','joint_states']`, publishing zero twist.

**Root cause.** The bridge's Python telemetry publisher
(`strafer_lab/bridge/async_publisher.py`) publishes only `/clock`,
`/strafer/odom`, and `/tf`. `ros2 topic info` on a live run shows
`/strafer/joint_states` and `/d555/imu/filtered` with **0 publishers**. But
`inference_node._assemble_observation_or_none()` returns `None` unless it
has IMU **and** joint_states **and** odom — for every variant — so no obs is
ever assembled and the node never flips `ready`. IMU was a documented
deferral (`bridge/graph.py`: Isaac Sim ships no `ROS2PublishImu`); wheel
joint_states was simply never published (on the robot the RoboClaw driver
publishes it, and that driver isn't started in sim).

**Fix.** The bridge already runs a Python `rclpy` publisher (the OmniGraph
IMU-publisher limitation doesn't apply to a Python publisher). Publish both
missing streams from `StraferAsyncPublisher.publish_state()` reading the
sim state already at the call site: wheel `joint_vel`/`joint_pos` from the
articulation, and `lin_acc_b`/`ang_vel_b` from the scene's `d555_imu`
sensor — the **same source the training obs uses**, so the policy sees an
in-distribution IMU. Signals are published **clean** (no synthetic DR
noise): training injects IMU/encoder noise at the observation-term level,
and the deployment obs pipeline adds none, so the bridge stream must match
the clean real-sensor contract (and stay deterministic for the ≤1e-5
obs-parity check in `strafer-direct-sim-validation`).

Because the inference obs reconstructs body velocity from wheel-FK via the
shared mecanum inverse-K (PR #56), publishing wheel `joint_states` also
fixes `body_velocity_xy` parity — one change closes both stale sources.

## Acceptance criteria

- [ ] The bridge publishes `/strafer/joint_states` (`sensor_msgs/JointState`,
      names = `WHEEL_JOINT_NAMES` in `[FL,FR,RL,RR]` order, velocity in
      rad/s) and `/d555/imu/filtered` (`sensor_msgs/Imu`, body-frame
      `linear_acceleration` incl. gravity + `angular_velocity`) each bridge
      tick, at a rate above the inference `obs_timeout_s` (0.2 s) floor.
- [ ] IMU accel/gyro are read from the scene `d555_imu` sensor
      (`lin_acc_b` / `ang_vel_b`) — the same source training's obs uses —
      and published **clean** (no re-injected DR noise).
- [ ] Wheel velocities are selected in canonical `WHEEL_JOINT_NAMES` order
      via the same index resolution the training env uses; covered by a
      pure unit test that runs without Kit.
- [ ] On a live DGX run (`make sim-bridge` + Jetson `make launch-sim` with a
      policy backend), `ros2 topic hz /strafer/joint_states` and
      `ros2 topic hz /d555/imu/filtered` both report > 5 Hz, and
      `ros2 param get /strafer_inference/strafer_inference ready` → `True`
      once the first obs assembles. **(DGX-side; not runnable on Jetson.)**
- [ ] Existing bridge telemetry (`/clock`, `/odom`, TF, `/cmd_vel`) is
      unchanged; callers that don't pass the new topics get the old
      behavior (publishers skipped).
- [ ] If your work invalidates a fact in any referenced context module,
      package README, top-level `Readme.md`, or guide under `docs/`, update
      those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression in the harness capture path (the second bridge
      instantiation site) — the new publishers are additive.

## Investigation pointers

- `source/strafer_lab/strafer_lab/bridge/async_publisher.py` —
  `publish_state()` already reads `root_lin_vel_b`/`root_ang_vel_b`; add
  `joint_vel`/`joint_pos` + IMU reads.
- `source/strafer_lab/strafer_lab/bridge/proprio.py` (new) — pure wheel
  index resolution, unit-tested in the pxr-free autonomy suite.
- `source/strafer_lab/scripts/run_sim_in_the_loop.py` — the two
  `StraferAsyncPublisher(...)` instantiation sites (bridge + harness);
  pass `imu_sensor=scene.sensors["d555_imu"]` and the new topics.
- `source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py` —
  `imu_linear_acceleration` / `imu_angular_velocity` /
  `_get_wheel_joint_indices` define the source contract to mirror.
- `source/strafer_ros/strafer_ros/.../inference_node.py:604-611` — the
  hard IMU+joint_states+odom obs requirement that gates `ready`.

## Out of scope

- Realistic sensor-noise injection in the bridge stream (train-time DR
  noise; a future opt-in knob if a noised sim-in-the-loop is wanted).
- The 200 Hz native IMU rate — publishing at the bridge step rate clears
  the obs freshness floor; a higher-rate IMU thread is a separate perf task.
- Any Jetson-side inference or launch change (the launch already
  subscribes to these topics; PR #126 wired the auto-launch).

## Notes

Authored + unit-tested on the Jetson (pure config + proprio tests green via
the autonomy suite's `strafer_lab` stub). The **live Kit publish and the
`ready` → True check must be validated on the DGX** (`make sim-bridge`) —
Isaac Sim isn't available on the Jetson.
</content>
</invoke>
