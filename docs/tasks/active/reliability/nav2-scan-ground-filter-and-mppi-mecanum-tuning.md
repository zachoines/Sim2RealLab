# Nav2 scan ground filter and MPPI mecanum dial-down

## Why
Three real-robot symptoms surfaced during lap tests after
[nav2-commit-and-follow-path-stability](../../completed/nav2-commit-and-follow-path-stability.md)
landed:

1. A phantom flat arc at ~3.5 m appears in `/scan` (Foxglove). It
   follows the robot/camera as it moves. Nav2 trajectories route
   around it as if it were a wall.
2. The generated path recalculates frequently. The robot rotates to
   face the new path → another path is generated → rotate again.
   Sometimes the robot cannot move forward at all.
3. The robot oscillates back/forth and side-to-side mid-mission,
   triggering bt_navigator's *"Navigation made no progress (>= 0.10 m)
   in the last 20 s"*. Particularly after mid-mission replans.

## Root cause
The `/scan` publisher is
[`depthimage_to_laserscan_node`](../../../../source/strafer_ros/strafer_slam/launch/slam.launch.py)
(slam.launch.py:67-86), which takes the **per-column min depth**
across a 60-row window centered on the image principal point
([depthimage_to_laserscan.yaml:15](../../../../source/strafer_ros/strafer_slam/config/depthimage_to_laserscan.yaml#L15)).

With the D555 mounted at `CAMERA_OFFSET_Z = 0.25 m` and a small
downward tilt, the bottom rows of that 60-row window project to the
floor at ~3.0–3.5 m. Because the per-column min is taken, *any* row
in the window seeing the floor pulls the entire column's reported
range to the floor distance. Result: a fan of points at ~3.5 m that
moves with the camera frame, not the world frame.

Items 2 and 3 cascade from item 1:

- Local-costmap obstacle layer marks the arc cells LETHAL → the
  10 Hz `IsPathValid` tick fails as the arc shifts under the existing
  path → SmacPlanner2D returns a fresh path biased around the new arc
  position.
- Omni motion model + `PathAlignCritic: 14.0`
  ([nav2_params.yaml:182](../../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml#L182))
  makes MPPI strafe to converge laterally on each new path. On
  real-robot mecanum, strafing is lossy/slow → forward progress stalls
  → progress-checker timeout.

## What

### A. Replace depth-to-laserscan with depth_image_proc + pointcloud-to-laserscan + ground filter

A proper Z filter prevents the floor from ever entering the scan,
fixing the artifact at its source rather than masking it. The depth
source must work on both lanes: real-robot
([perception.launch.py](../../../../source/strafer_ros/strafer_perception/launch/perception.launch.py)
starts the realsense node) and sim-in-the-loop
([bringup_sim_in_the_loop.launch.py](../../../../source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py)
skips it; the DGX bridge publishes depth + color images directly).
Both lanes already feed the existing `_sync` depth topics via
[timestamp_fixer](../../../../source/strafer_ros/strafer_perception/strafer_perception/timestamp_fixer.py),
so the projection step lives downstream of those topics in
slam.launch.py.

- In
  [slam.launch.py](../../../../source/strafer_ros/strafer_slam/launch/slam.launch.py),
  before the scan publisher, add a `depth_image_proc::point_cloud_xyz_node`:
  - `image_rect` ← `/d555/aligned_depth_to_color/image_sync`
  - `camera_info` ← `/d555/aligned_depth_to_color/camera_info_sync`
  - `points` → `/d555/aligned_depth_to_color/points`
- Replace the `depthimage_to_laserscan_node` block with a
  `pointcloud_to_laserscan_node`. Required params (file at
  `source/strafer_ros/strafer_slam/config/pointcloud_to_laserscan.yaml`):
  - `target_frame: base_link` (transform → robot frame; Z is true vertical)
  - `min_height: 0.05` (skip floor)
  - `max_height: 0.30` (Strafer body height; skip ceiling / people heads)
  - `range_min: DEPTH_MIN`, `range_max: DEPTH_MAX` (overridden at launch)
  - `angle_min`/`angle_max` = ±0.873 rad (~±50°) covers D555's ~87° H-FOV
  - Remap input cloud → `/d555/aligned_depth_to_color/points`, output → `/scan`
- Delete the old depthimage yaml.
- Swap `depthimage_to_laserscan` → `pointcloud_to_laserscan` and add
  `depth_image_proc` to
  [strafer_slam/package.xml](../../../../source/strafer_ros/strafer_slam/package.xml)
  exec_depend.
- Install on the Jetson:
  `sudo apt install ros-humble-pointcloud-to-laserscan ros-humble-depth-image-proc`.
- Update
  [test_slam_config.py](../../../../source/strafer_ros/strafer_slam/test/test_slam_config.py)
  to assert pointcloud_to_laserscan params (target frame, height
  bounds, ranges, symmetric angle sweep) instead of the depth-window params.

`perception.launch.py` and `timestamp_fixer.py` are unchanged by this
brief — the existing `_sync` depth pipeline is the right source for
both lanes, so enabling the realsense native pointcloud would only
work on real-robot.

### B. MPPI critic dial-down on the real-robot lane

The 14.0 PathAlignCritic real-robot baseline over-weights lateral
convergence for mecanum. Reduce it; raise PreferForwardCritic so the
controller biases toward forward motion.

- In [nav2_params.yaml](../../../../source/strafer_ros/strafer_navigation/config/nav2_params.yaml):
  - `PathAlignCritic.cost_weight: 14.0 → 8.0`
  - `PreferForwardCritic.cost_weight: 3.0 → 6.0`
- The sim absolute overrides in
  [navigation.launch.py](../../../../source/strafer_ros/strafer_navigation/launch/navigation.launch.py)
  `_patch_params` (`PathAlignCritic.cost_weight = 9.0`,
  `PreferForwardCritic.cost_weight = 10.0`) are absolute values, not
  derived from baselines, so they remain unchanged. Effective sim
  values stay where they are today; only the real-robot lane shifts.
  Refresh the stale inline comments referencing the old "14 → 9" /
  "3 → 10" arrows.
- Add a `TestMPPIController::test_real_robot_critic_baselines` test
  that pins the new baselines (8.0 / 6.0) so future drifts are caught
  by CI.

### C. Verify

- Lap test on real robot in Mission 1 (known room). Acceptance below.
- If item 3 persists after A+B, file follow-up for hysteresis on path
  acceptance (reject a new path whose first-waypoint heading deviates
  < N° from current heading when time-since-last-plan < M s).

### D. Decide: is path-lookahead pre-rotation still needed?

The `_path_lookahead_yaw` + `compute_path_to_pose` work in
[nav2-commit-and-follow-path-stability](../../completed/nav2-commit-and-follow-path-stability.md)
was diagnosed under an over-weighted `PathAlignCritic` (14.0). If
section B's dial-down lands the real-robot lane in a regime where MPPI
naturally rotates to face each waypoint as Nav2 dishes them out, the
pre-rotation becomes a band-aid for a problem that's already fixed.

Protocol on the same lap test as section C:

1. Baseline lap with current code (path-lookahead pre-rotation active).
2. Temporarily disable the lookahead in `_align_to_goal_yaw` (return the
   goal pose's yaw directly, or skip pre-alignment entirely behind a
   feature flag) and run the same lap.
3. Compare during the first 1 m of each navigate-to-pose call:
   chassis-yaw vs `/plan` direction; `|/cmd_vel.vy|` peak vs
   `|/cmd_vel.vx|` peak.

Decision tree:

- If lap 2 tracks paths as cleanly as lap 1 (vy/vx ratio comparable,
  no extra strafe), file follow-up to simplify `_align_to_goal_yaw`
  (drop the lookahead; possibly drop pre-alignment entirely) and
  retire `JetsonRosClient.compute_path_to_pose` if it has no other
  consumer.
- If lap 2 shows materially worse path-following, keep the current
  pre-rotation. Note the lap evidence in the follow-up so the
  decision is replayable.

## Acceptance
- Foxglove `/scan` shows no flat arc following the robot in known-empty
  space, robot stationary or moving.
- BT `IsPathValid` failures correlate with real obstacle insertion or
  goal change, not with robot motion alone (observed in `/plan` vs
  `/scan` overlay).
- 5 of 5 lap-test goals reach `succeeded` without `aborted` from the
  progress checker.
- `test_slam_config.py` updated to cover pointcloud_to_laserscan params.
- Section D protocol ran on at least one lap and recorded the
  vy/vx-with vs without-pre-rotation comparison in the PR description.
  The follow-up to simplify or keep `_align_to_goal_yaw` is filed
  (either direction) with the lap evidence cited.

## Risks
- pointcloud_to_laserscan CPU cost is higher than depthimage variant.
  Verify Jetson CPU headroom during a full mission; downsample via
  realsense `decimation_filter.enable: "true"` if needed.
- Realsense pointcloud at 640x360x30 may saturate USB bandwidth.
  Fallback: drop depth profile to 424x240 or enable decimation.
- Lower PathAlignCritic could increase path-tracking error in tight
  corridors. Mitigated by raised PreferForwardCritic; revisit if
  observed.

## Out of scope
- Architectural cleanup of `_patch_params` sim/real split — covered by
  [nav2-sim-real-promotion-architecture](../tooling/nav2-sim-real-promotion-architecture.md).
- Camera mount/tilt changes (hardware).
- Path-acceptance hysteresis (filed as follow-up only if items 2/3
  persist after A+B).
