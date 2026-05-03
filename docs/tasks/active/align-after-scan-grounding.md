# Align robot to grounded target before navigate, to eliminate strafe-toward-goal

**Type:** task
**Owner:** Jetson agent
**Priority:** P2
**Estimate:** S–M (~half day; one new short rotate step + plan_compiler
edit + a small sim validation lap)
**Branch:** task/align-after-scan-grounding

## Story

As a **mission operator dispatching `go_to_target` missions**, I want
**the robot to start the navigate leg already roughly facing the
grounded target**, so that **MPPI drives forward-and-arc into the
goal pose instead of strafing diagonally — easier to interpret in
Foxglove, and the on-mecanum strafe is the symptom MPPI's critic
landscape resolves least cleanly today**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [completed/sim-velocity-attenuation.md](../completed/sim-velocity-attenuation.md)
  — same operator session that surfaced the velocity-attenuation
  symptom flagged this; useful for context on how MPPI behaves when
  the heading delta at navigate-start is large.

## Context

The current `go_to_target` plan is:

```
1. scan_for_target           # rotate-and-ground loop
2. project_detection_to_goal_pose   # bbox → map-frame goal at standoff
3. navigate_to_pose          # Nav2 / MPPI to that pose
4. verify_arrival            # CLIP top-k re-check
```

Two interacting code paths cause the operator-visible "robot strafes
toward the goal" symptom:

1. **Scan exits as soon as the bbox lands anywhere in FOV.**
   [`mission_runner.py:_scan_for_target`](../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
   captures, attempts grounding, and `return SkillResult(...)
   succeeded` the moment confidence ≥ `min_grounding_confidence`
   (default 0.5). The D555's perception-stream FOV is ~70°
   horizontal, so the robot can be up to ~35° off-axis from the
   target when scan succeeds.
2. **Goal pose's yaw faces the target.**
   [`goal_projection_node._compute_standoff_pose`](../../../source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py)
   sets `goal_yaw = math.atan2(uy, ux)` (the robot→target unit
   vector's angle). So `goal_pose.yaw` ≠ `robot_pose.yaw` at
   navigate-start by up to half-FOV.

`navigate_to_pose` then asks Nav2 to reach a pose whose position is
"forward of where the camera was pointing" *and* whose orientation
is some tens of degrees off the robot's current heading. MPPI on a
mecanum with the current critic balance resolves that as a
strafe-while-turn — which is valid motion, but visually unintuitive
and the symptom that's hardest to debug when something goes wrong.

## Approach

Pick the surgical fix:

**Option A (preferred): insert a brief alignment step in
`plan_compiler._compile_single_target_steps`.** Between
`project_detection_to_goal_pose` and `navigate_to_pose`, emit a
`rotate_to_pose_yaw` skill that turns the robot to face the goal
pose's yaw with a tight tolerance (~5°). Robot then starts navigate
with heading ≈ goal yaw, and Nav2 + MPPI plan a near-straight-line
forward path. The skill is small to add — pose math identical to
`_orient_to_direction` but takes a target yaw rather than a cardinal
key, and reuses `ros_client.rotate_in_place`.

**Option B: fine-rotate inside `_scan_for_target` after grounding.**
Compute the bbox center's pixel offset from image center, convert to
a yaw delta via the camera's horizontal FOV, and rotate that much
before returning success. Pro: no plan_compiler change; the fix is
local to the skill that already rotates. Con: couples scan logic to
camera intrinsics, and the post-rotate goal pose is still recomputed
in `project_detection_to_goal_pose` so you'd want the *next* capture
to reflect the alignment, otherwise step 2's projection is off.

**Option C: tune MPPI critics to prefer turn-then-drive over
strafe-while-turn.** Bump `PreferForwardCritic.cost_weight` (3.0 →
6–8) and trim `PathAlignCritic` weight. Already partially within
[`mppi-critic-tuning-for-sim-envelope.md`](../completed/mppi-critic-tuning-for-sim-envelope.md)
scope. This is less surgical and doesn't deterministically fix the
heading-mismatch root cause; it biases MPPI to prefer one resolution
of it.

A is recommended.

## Acceptance criteria

- [ ] `plan_compiler._compile_single_target_steps` (and the patrol /
      multi-target variants that share it) emit an alignment step
      between `project_detection_to_goal_pose` and `navigate_to_pose`
      that rotates the robot to within ~5° of the goal pose's yaw
      before the navigate leg starts. Skill name and arg shape are
      author's choice; the executor handler delegates to
      `ros_client.rotate_in_place` with the computed yaw delta.
- [ ] On a `go_to_target` mission against a target that's off-axis
      from the robot's start heading, observable in Foxglove or via
      `ros2 topic echo /cmd_vel`: the navigate leg's `cmd_vel.vy`
      stays close to zero (no strafe) once past the acceleration
      ramp, in contrast to the current behavior where `vy` carries
      meaningful magnitude.
- [ ] No regression on missions where the robot already faces the
      target (the alignment step is a no-op rotate within tolerance).
- [ ] No regression on `translate` / `rotate_by_degrees` /
      `rotate_to_direction` / `cancel` / `status` / `describe` /
      `query` — none of which use `_compile_single_target_steps`.
- [ ] `nav2-far-goal-staging.md`'s reference cross-room mission
      still completes end-to-end with the staging loop intact.
- [ ] Unit tests cover the new compiler emission shape and the
      executor handler's pass-through behavior.
- [ ] If your work invalidates a fact in any referenced context
      module, update that module in the same commit.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/planner/plan_compiler.py:61-95`
  — `_compile_single_target_steps`; insert the new alignment step
  between the project and navigate emissions here.
- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py:1538-1580`
  — `_orient_to_direction` is the closest existing template for an
  "align to a yaw" handler; the new handler differs only in where
  the target yaw comes from (a step arg, not a cardinal lookup).
- `source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py:399-435`
  — `_compute_standoff_pose` is where `goal_yaw` is computed; the
  new alignment step needs the same `goal_yaw`, available either by
  re-reading the `GoalPoseCandidate` from the previous step's outputs
  or recomputing from the projected target.

## Out of scope

- **Changing `_compute_standoff_pose` to use the robot's current
  yaw.** That would let the navigate leg drive in a straight line
  without an explicit alignment step, but at the cost of arriving at
  the goal not facing the target — bad for any subsequent
  approach-further or look-at-target step. Keep the goal-yaw
  semantics; align before the leg.
- **Changing scan_for_target's success criterion** (e.g., requiring
  bbox center to be near image center). That's Option B above and
  is explicitly not chosen here.
- **MPPI critic tuning to prefer turn-then-drive.** That's Option
  C / part of
  [`mppi-critic-tuning-for-sim-envelope.md`](../completed/mppi-critic-tuning-for-sim-envelope.md).
  The two changes are complementary; this brief delivers the fix at
  the planning layer regardless of MPPI tuning.
- **Real-robot validation.** D555 FOV and chassis kinematics differ
  from sim; a real-robot lap is worth doing if/when the change ships
  to the operator's robot, but this brief's acceptance is sim-side.
