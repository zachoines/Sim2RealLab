# Move alignment-bearing math from executor to perception node

**Type:** task / refactor
**Owner:** Jetson agent (`source/strafer_ros/strafer_perception/`, `source/strafer_ros/strafer_msgs/`, `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`, `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`)
**Priority:** P3 (filed-on-trigger; do not pick up until at least one of the bearing-varying behaviors below is on the active roadmap — `approach-from-angle`, `look-at-while-driving`, `face-target-while-translating`, or similar)
**Estimate:** S–M (~half to one day; one new `.srv` + ~15-line perception handler + ros_client method + ~10-line executor handler replacement + tests + colcon rebuild on both hosts)
**Branch:** task/perception-side-bearing-service

## Story

As **a maintainer planning future bearing-varying behaviors
(approach-from-angle, look-at-while-driving, face-target-while-
translating)**, I want **the bearing-to-target math to live in
`strafer_perception` — the package that already owns
`/d555/color/camera_info_sync`, the TF buffer, and the
`GoalPoseCandidate` cache — instead of in the autonomy executor**,
so that **the next feature that varies the alignment policy doesn't
have to thread or duplicate quaternion/yaw math through the
autonomy/executor layer.**

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [completed/align-after-scan-grounding.md](../completed/align-after-scan-grounding.md)
  — the brief that shipped today's `align_to_goal_yaw` skill and
  put the bearing math in the executor. This brief is the explicit
  follow-up flagged in that PR's summary.

## Context

The currently-shipped alignment path (see
`mission_runner._align_to_goal_yaw`):

1. The `project_detection_to_goal_pose` step stores a
   `GoalPoseCandidate` in `runtime.latest_goal_pose`. The candidate
   carries a `Pose3D` whose yaw is encoded in `qz`/`qw` (with
   `qx=qy=0`) — a planar-2D assumption baked into
   `goal_projection_node._compute_standoff_pose`.
2. The executor's `_align_to_goal_yaw` handler reads that
   `Pose3D`, extracts the yaw via
   `atan2(2*qw*qz, 1 - 2*qz²)`, reads the robot's current odometry
   yaw via `ros_client.get_robot_state()`, computes a shortest-arc
   delta, and dispatches to `rotate_in_place`.

This works for today's "rotate to face goal" use case. The cost
becomes visible only when a future change wants to *vary* the
bearing target. Three concrete future briefs that would benefit:

- **`approach-from-angle.md`** (hypothetical): for missions where
  the robot should arrive at the target's side rather than face-on
  (delivery from a particular aisle, photographing a target from a
  specific azimuth). The alignment yaw is `target_yaw + offset`,
  not `target_yaw`.
- **`look-at-while-driving.md`** (hypothetical): a controller that
  steers `vx`/`vy` along the path while continuously rotating to
  keep the camera framed on the target. Needs *instantaneous*
  bearing-to-target, not a one-shot value frozen at projection time.
- **`face-target-while-translating.md`** (hypothetical): a variant
  of `translate` that maintains the camera-on-target lock across
  the motion.

In all three, the bearing math lives in the executor today and
would have to either be threaded (compiler → handler args) or
duplicated. The cleaner layering is to expose the math from
`strafer_perception`, which already owns the relevant state, and
have the executor consume a pre-computed `yaw_delta_rad`.

A second incidental benefit: the bearing today is computed against
the TF lookup that happened during `project_detection_to_goal_pose`.
If the robot drifts in the window between projection and align, the
bearing is stale. A perception-side service that re-queries TF at
the moment of align always returns fresh data.

## Approach

### A. New service in `strafer_perception` (recommended)

Add a `ComputeTargetBearing.srv` to `strafer_msgs`:

```
# request
string  request_id          # for log correlation
float64 offset_yaw_rad      # 0.0 = "face target"; non-zero supports
                            # approach-from-angle without re-shipping
                            # the .srv
---
# response
bool    found
float64 yaw_delta_rad       # shortest-arc delta from robot's current
                            # map-frame yaw
float64 target_yaw_map_rad  # absolute target yaw (telemetry)
string  message
```

In `goal_projection_node`, add a service handler that:

1. Reads the cached latest `GoalPoseCandidate` (the node already
   retains state across the projection service for telemetry).
2. Performs a fresh `base_link → map` TF lookup *at call time*
   (not at projection time).
3. Computes
   `atan2(target_y - robot_y, target_x - robot_x) + offset_yaw_rad`
   and the shortest-arc delta vs. the robot's current yaw.
4. Returns the delta.

In `ros_client.py`, add `compute_target_bearing(request_id,
offset_yaw_rad=0.0) -> BearingResult` mirroring the existing
`project_detection_to_goal_pose` client method.

In `mission_runner._align_to_goal_yaw`, replace the quaternion-and-
odometry block with a single service call:

```python
def _align_to_goal_yaw(self, runtime, step):
    started_at = time.time()
    try:
        bearing = self._ros_client.compute_target_bearing(
            request_id=f"{runtime.mission_id}:{step.step_id}",
            offset_yaw_rad=float(step.args.get("offset_yaw_rad", 0.0)),
        )
    except Exception as exc:
        return self._failed_result(step, f"bearing query failed: {exc}",
                                   "bearing_failed", started_at)
    if not bearing.found:
        return self._failed_result(step, bearing.message or "no bearing",
                                   "align_prereq_missing", started_at)
    return self._ros_client.rotate_in_place(
        step_id=step.step_id,
        yaw_delta_rad=bearing.yaw_delta_rad,
        tolerance_rad=float(step.args.get("tolerance_rad", 0.087)),
        timeout_s=self._motion_timeout_s(
            step=step, magnitude=bearing.yaw_delta_rad, kind="angular",
        ),
    )
```

The executor loses its `Pose3D` import for alignment, its
quaternion-to-yaw math, and its robot-state fetch. All three move
inside the perception node.

### B. Extend the existing projection service response

Add `yaw_delta_rad` and `target_yaw_map_rad` fields to
`ProjectDetectionToGoalPose.srv`'s response. The executor still
reads `runtime.latest_goal_pose.yaw_delta_rad` instead of doing the
math itself.

Pros: no new service; smaller surface.

Cons: re-encodes the staleness problem (bearing is frozen at
projection time, not align time). Defeats one of the two motivating
benefits of this refactor. **Not recommended.**

### C. Keep today's shipped layout, threading new args per future feature

Punt: leave bearing math in the executor and thread an
`offset_yaw_rad` arg through the compiler when the first
bearing-varying feature lands.

Pros: zero work today.

Cons: every bearing-varying feature pays the cost; the executor
slowly accumulates camera/TF math it doesn't logically own.

**Recommended:** A. The two benefits compound: layering hygiene
*and* fresher TF at align time.

## Acceptance criteria

- [ ] `strafer_msgs/srv/ComputeTargetBearing.srv` exists with the
      request/response shape above (or a near equivalent — minor
      naming changes are author's discretion as long as the
      `offset_yaw_rad` field is present so approach-from-angle
      doesn't require a future re-ship of the `.srv`).
- [ ] `goal_projection_node` exposes
      `/strafer/compute_target_bearing` and the handler uses a
      **fresh TF lookup** at call time, not the lookup cached
      during the prior projection.
- [ ] `ros_client.compute_target_bearing(...)` exists, mirrors the
      pattern of the existing `project_detection_to_goal_pose`
      client method, and returns a typed result.
- [ ] `mission_runner._align_to_goal_yaw` no longer imports or
      uses `Pose3D`, no longer reads `runtime.latest_goal_pose.goal_pose`
      directly for yaw extraction, and no longer calls
      `ros_client.get_robot_state()` for the alignment path. The
      function is a thin pass-through ≤ ~15 lines.
- [ ] Unit tests for the perception-side handler cover: no cached
      `GoalPoseCandidate` → `found=False`; TF lookup failure →
      `found=False` with a clear message; happy path with non-zero
      `offset_yaw_rad` reflecting in the response.
- [ ] Executor handler tests still cover dispatch, prereq-missing,
      and pass-through to `rotate_in_place` — adapted to use the
      new client method instead of the previous quaternion path.
- [ ] No behavior change observable to the operator on a baseline
      `go_to_target` mission with `offset_yaw_rad = 0` against a
      sim target. The `cmd_vel.vy ≈ 0` assertion from
      `align-after-scan-grounding`'s acceptance still holds.
- [ ] `colcon build` clean on both Jetson and DGX after the
      `strafer_msgs` rebuild ripple.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit.

## Investigation pointers

- `source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py:399-435`
  — `_compute_standoff_pose` produces the planar-yaw encoding the
  executor consumes today. The new handler reuses the same math
  inline.
- `source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py:170-310`
  — the existing `_project_detection_to_goal_pose` service handler
  is the structural template for the new bearing handler (state
  cache, TF lookup, response composition).
- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`
  — `_align_to_goal_yaw` is the function that collapses. The shape
  to mirror is `_orient_to_direction` (cardinal-yaw rotation
  through `ros_client.rotate_in_place`), but with the target yaw
  pre-computed on the perception side.
- `source/strafer_ros/strafer_msgs/srv/ProjectDetectionToGoalPose.srv`
  — pattern for the new `.srv` file.
- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:753-810`
  — the existing `project_detection_to_goal_pose` client method is
  the structural template for `compute_target_bearing`.

## Out of scope

- **Implementing approach-from-angle / look-at-while-driving /
  face-target-while-translating.** Those are the *features* this
  refactor unblocks; brief them separately when on the roadmap.
- **Replacing the `Pose3D` quaternion encoding broadly.** Other
  consumers of `Pose3D` (Nav2 goal dispatch, verify_arrival,
  semantic-map node poses) keep the existing encoding. Only the
  alignment-bearing read path moves.
- **Touching `_compile_single_target_steps`.** The compiler still
  emits `align_to_goal_yaw` with the same step ordering and the
  same default `tolerance_rad`. Step shape unchanged.
- **Removing `runtime.latest_goal_pose` from the executor's
  runtime state.** It's still consumed by `navigate_to_pose`,
  `verify_arrival`, and the semantic-map short-circuit path.
- **Bumping `strafer_msgs`'s version number or migration policy.**
  Adding a new `.srv` is additive; no version bump needed beyond
  the colcon rebuild ripple.
