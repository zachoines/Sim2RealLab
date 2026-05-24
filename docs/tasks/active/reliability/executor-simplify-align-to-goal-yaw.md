# Simplify `_align_to_goal_yaw` — remove the path-lookahead pre-rotation

## Why

The path-lookahead pre-rotation in `_align_to_goal_yaw` was added in
[completed/nav2-commit-and-follow-path-stability](../../completed/nav2-commit-and-follow-path-stability.md)
to avoid MPPI commanding `vy` strafe on curved paths. It rotates the
chassis to a 1 m-ahead path waypoint before starting `navigate_to_pose`,
querying Nav2's planner via a new
`JetsonRosClient.compute_path_to_pose` wrapper.

The Section D protocol in
[nav2-scan-ground-filter-and-mppi-mecanum-tuning](../../completed/nav2-scan-ground-filter-and-mppi-mecanum-tuning.md)
disabled the pre-rotation (via `STRAFER_SKIP_ALIGN_TO_GOAL_YAW=1`)
and the operator observed identical tracking quality on
sim-in-the-loop after the CPU-starvation issues were resolved. MPPI
with corrected sim critic weights (`PathAlign=9`, `PreferForward=10`)
handles starting heading via its critic landscape without a separate
pre-rotation step. The pre-rotation was a band-aid for the
over-tracking MPPI that the dial-down + CPU relief fixed.

This brief retires the pre-rotation and its supporting code so the
codebase reflects the actual production behavior.

## What

### A. Simplify `_align_to_goal_yaw`

In
[mission_runner.py](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
`_align_to_goal_yaw`:

- Pick one of (operator decision in the implementing PR):
  1. **Make it a no-op** — return `succeeded` without rotating. The
     skill stays in the registry so existing plans that include it
     don't break; the rotation just doesn't fire.
  2. **Simplify to goal-bearing rotation** — rotate to the
     map-frame bearing from robot to goal pose (NOT goal pose's yaw,
     NOT path lookahead). Cheaper than the planner query and avoids
     the goal-yaw-vs-path-direction mismatch the original brief
     diagnosed.

Recommended: start with (1) since the operator's section D evidence
shows MPPI tracks fine without any pre-rotation; keep the skill
entry point so plans don't break.

- Remove the `STRAFER_SKIP_ALIGN_TO_GOAL_YAW` env knob (becomes moot
  once the default is "no rotation").
- Remove the `_path_lookahead_yaw` helper.

### B. Retire `compute_path_to_pose` if unused

Audit consumers of
[`JetsonRosClient.compute_path_to_pose`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py):

- Only consumer today is `_path_lookahead_yaw` (deleted in A).
- If no other consumer surfaces, remove the method, its Protocol
  signature, and any related ros_client tests.

### C. Tests

- Remove `TestAlignToGoalYaw` tests for path-lookahead and goal-yaw
  fallback in
  [test_new_skills.py](../../../../source/strafer_autonomy/tests/test_new_skills.py).
- Keep / update `test_align_skipped_when_env_set` and
  `test_align_runs_when_env_unset` as appropriate for the chosen
  simplification — or remove them if the env knob is retired.
- If retaining option (2) goal-bearing rotation, add a test that
  asserts the rotation magnitude matches the map-frame bearing.

## Acceptance

- Mission "go to the couch" (or any sim-in-the-loop navigate-to-pose
  mission) succeeds with the same tracking quality the operator
  observed under `STRAFER_SKIP_ALIGN_TO_GOAL_YAW=1`.
- `_path_lookahead_yaw` and `compute_path_to_pose` no longer appear
  in the codebase (or have a documented remaining consumer).
- No env knob is needed at launch time to get the production behavior.
- Test suite green; coverage for the chosen simplification approach
  is in place.

## Out of scope

- Re-introducing pre-rotation under a different design (e.g. goal-yaw
  alignment for non-mecanum bases). File separately if it surfaces.
- Adjusting MPPI critic weights — this brief is purely about the
  executor-side pre-rotation surface.

## Risks

- Mecanum-base behavior on tightly-curved paths in *real-robot*
  conditions has not been validated under "no pre-rotation". The
  section D evidence is sim-only. Validation lap on real hardware
  is part of the implementing PR's acceptance.
