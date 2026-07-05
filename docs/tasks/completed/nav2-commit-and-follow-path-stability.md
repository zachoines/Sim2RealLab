# Commit-and-follow Nav2 global paths: prefer known-free, replan on invalidation

**Status:** Shipped 2026-05-23 in `fdf02bd` (Jetson). Global planner is
now `nav2_smac_planner/SmacPlanner2D` with `allow_unknown: true` +
`cost_travel_multiplier: 2.0` (soft-prefer known cells). BT replans
only when `IsPathValid` fails or `GlobalUpdatedGoal` fires, on every
lane. `_align_to_goal_yaw` rotates to a path lookahead via a new
`JetsonRosClient.compute_path_to_pose` plan-only query (falls back to
goal pose's yaw on planner failure). Follow-up
[`nav2-scan-ground-filter-and-mppi-mecanum-tuning`](../active/reliability/nav2-scan-ground-filter-and-mppi-mecanum-tuning.md)
covers real-robot symptoms surfaced during lap tests.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/50

**Type:** task / tuning
**Owner:** Jetson agent (`source/strafer_ros/strafer_navigation/`)
**Priority:** P2
**Estimate:** S–M (~half day; YAML + BT XML + nav-config tests, no Python logic)
**Branch:** task/nav2-commit-and-follow-path-stability

## Story

As a **mission operator running both sim-in-the-loop and real-robot
missions**, I want **Nav2's global plan to (a) prefer already-mapped
free space over unknown cells and (b) stay committed to the chosen
path until a real obstacle invalidates it — on every lane**, so that
**the robot drives obvious routes instead of cutting through unknown
bands, MPPI tracks a stable reference instead of a path that jitters
every replan tick, and behavior I validate in sim is the same
behavior I get on real**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [completed/nav2-startup-unknown-donut-path-noise.md](nav2-startup-unknown-donut-path-noise.md)
  — the predecessor that swapped Nav2's `<RateController hz="1.0">`
  for `<DistanceController distance="0.5">` and inserted SmoothPath
  on the sim lane.

## Context

Operator observation (2026-05-22) after the rotate-then-translate
shipping (PR #49): the executor's local heading bias is working, but
two upstream behaviors keep paths visibly poor in sim:

1. **Global plan routes through unknown regions even when a verified
   free path exists.** NavfnPlanner's potential-field search runs over
   the global costmap, and with `planner_server.GridBased.allow_unknown:
   true` (current default), `NO_INFORMATION` cells get a uniform
   `NEUTRAL_COST` that's not much higher than known-free. The
   shortest-distance heuristic then happily slices through unknown
   bands when they shave a few cells off the route — even when a
   known-free path of comparable length exists.

2. **The chosen path keeps mutating mid-traverse.** The current sim BT
   gates replanning on 0.5 m of motion
   (`<DistanceController distance="0.5">` — shipped in
   `nav2-startup-unknown-donut-path-noise`), but it still calls
   `ComputePathToPose` *unconditionally* every 0.5 m. As the costmap
   fills in (new free cells observed, transient sensor returns
   clearing), NavFn re-derives a slightly different shortest path on
   each replan tick. MPPI is then forced to track a reference whose
   topology shifts under it.

The behavior the operator wants is "commit and follow": pick a path,
stick with it, only replan when something concrete invalidates the
plan (current path blocked, goal changed). Symptom 1 wants the planner
to *prefer known cells*, and symptom 2 wants the BT to *defer
replanning to events, not timers*.

## Approach

Two changes, both applied **universally** (sim AND real). The
existing `envelope_factor > 1.0` gate exists for changes that
genuinely depend on the lifted velocity envelope (MPPI sampling stds,
MPPI critic rebalance); these path-shape changes don't, so they go in
the project's universal default instead. The architectural cleanup
that formalizes this split (and graduates the other currently-gated
knobs) is tracked in
[`nav2-sim-real-promotion-architecture`](nav2-sim-real-promotion-architecture.md).

### A. SmacPlanner2D with soft-prefer-known via `cost_travel_multiplier`

Swap the planner plugin from `nav2_navfn_planner/NavfnPlanner` to
`nav2_smac_planner/SmacPlanner2D`. NavFn's binary `allow_unknown`
proved too brittle in practice — even on a costmap that's visually
"fully filled in," the camera blind-spot donut and gaps between
depth-projected scan beams leave small unknown patches that NavFn
refuses to traverse, so far goals fail planning outright even though
known-free paths exist within the operator's mental model. SmacPlanner2D
exposes `cost_travel_multiplier`, which scales each cell's costmap
value in the A* cost. With `allow_unknown: true` + `cost_travel_multiplier:
2.0`, unknown cells (NEUTRAL_COST ≈ 50) cost ~100 to traverse while
free cells stay near 0 — the planner prefers free but allows unknown
when needed.

Pros: closes the prefer-known-free intent without the
goal-rejection brittleness. SmacPlanner2D is already shipped in
`nav2_smac_planner`; no extra packages.

Cons: SmacPlanner is somewhat slower than NavFn on identical
costmaps. We don't see this as a blocker (planning runs once per
goal-change rather than every BT tick, courtesy of the IsPathValid
gate from change B), but if real-robot planner latency surfaces, a
`max_planning_time` cap is already set in the YAML.

### B. IsPathValid-gated `Fallback` as the universal BT

In `navigate_to_pose_w_smoothing_and_recovery.xml`, replace the
`<DistanceController distance="0.5">` decorator with a
`<Fallback name="ReplanIfNeeded">` whose first child is a
`<Sequence>` of `<Inverter><GlobalUpdatedGoal/></Inverter>` and
`<IsPathValid path="{path}"/>`. `IsPathValid` succeeds when the
current `{path}` is still collision-free against the latest costmap;
the inverted `GlobalUpdatedGoal` succeeds when no new goal has
arrived since the last tick. When both succeed the `Fallback`
short-circuits without re-planning. When either fails (path blocked
or new goal), the fallback falls through to the existing
`ComputePathToPose → SmoothPath` sequence.

Use `GlobalUpdatedGoal`, **not** `GoalUpdated` — the latter resets
its remembered-goal state when the BT is halted between
back-to-back `navigate_to_pose` action calls, so the new goal is
treated as the "initial" goal on the first tick after halt and the
gate falls back on the stale `{path}` from the previous call.

Lift the BT swap in `_patch_params` out of the `if envelope_factor
> 1.0:` block so the BT path is injected unconditionally — sim and
real bringup both load this BT. (The injection itself must stay in
launch code rather than YAML because the absolute path is resolved
from `ament_index_python.get_package_share_directory` at launch
time.)

Net BT shape (universal):

```
PipelineSequence
├─ Fallback (ReplanIfNeeded)
│  ├─ Sequence
│  │   ├─ Inverter(GlobalUpdatedGoal)
│  │   └─ IsPathValid(path)
│  └─ RecoveryNode(ComputePathToPose → SmoothPath, ClearGlobalCostmap)
└─ RecoveryNode(FollowPath, ClearLocalCostmap)
```

Pros: replanning is event-driven (path invalid / goal changed) rather
than motion-driven; the path the controller tracks is now stable
between invalidation events on every lane. The plugin
(`nav2_is_path_valid_condition_bt_node`) is already linked in
`bt_navigator.plugin_lib_names` from the predecessor.

Cons: a transient sensor blip (one scan that briefly marks a cell
lethal across the current path) can trigger a one-shot replan that
mostly reverts on the next tick once the blip clears. The local
costmap clearing recovery in the existing BT covers the recovery
branch already. Real lidars are noisier than sim virtual lidars, so
real-robot is the side where flapping is more likely to surface —
the
[`nav2-sim-real-promotion-architecture`](nav2-sim-real-promotion-architecture.md)
follow-up tracks the real-robot validation lap and any debouncer
work that comes out of it.

**Sequence:** ship both at once — they're complementary fixes for the
same operator complaint. A on its own leaves symptom 2; B on its own
leaves symptom 1.

### C. `align_to_goal_yaw` rotates to a path lookahead, not the goal pose's yaw

`_align_to_goal_yaw` previously rotated the chassis to the goal
pose's yaw (= "face the target from the standoff"). On straight
paths this matches the immediate path direction so MPPI drives
forward cleanly. On curved paths (Nav2 routing around obstacles)
the goal yaw and the path direction diverge — MPPI's PathAlignCritic
then commands lateral `vy` to track the curve while the chassis is
oriented toward the goal, i.e. the chassis strafes.

Reworked: `_align_to_goal_yaw` calls `ros_client.compute_path_to_pose`
(plan-only Nav2 action), finds the first waypoint at least
`lookahead_m` (default 1.0 m) along the planned path, and rotates
the chassis to the bearing of that waypoint. Falls back to the goal
pose's yaw when planning is unavailable or no usable waypoint
exists.

A new method `JetsonRosClient.compute_path_to_pose(goal_pose=...)`
wraps the `compute_path_to_pose` action and returns the path as a
list of `(x, y)` tuples in the map frame; the executor consumes that
without any ROS imports of its own.

## Acceptance criteria

- [ ] On a sim mission whose goal has a known-free path AND an
      unknown-band shortcut both visible in the global costmap, Nav2
      emits a path that lies entirely inside known-free cells. The
      sim-side log + Foxglove `/plan` overlay should make this
      directly observable on a `translate forward 3 m` mission run
      after some warmup driving — no unknown-band cut-through.
- [ ] On the same mission, the `/plan` topic publishes once at the
      start (and again on `goal_update` or path-invalidating
      obstacles), not once per 0.5 m of robot motion. Operator can
      visually confirm in Foxglove that the path doesn't jitter.
- [ ] Cardinal far-goal regression on
      [`completed/nav2-far-goal-staging.md`](nav2-far-goal-staging.md)'s
      reference mission: mission still completes end-to-end with
      ≥ 2 staging legs. Plan stability + known-free preference may
      slow the path discovery in the worst case (planning fails until
      `explore_until_visible` widens known free); document any
      slowdown in the PR description.
- [ ] No regression on cornering smoke from
      [`completed/mppi-critic-tuning-for-sim-envelope.md`](mppi-critic-tuning-for-sim-envelope.md):
      `translate forward 1 m → rotate 90° → translate forward 1 m`
      lands within `xy_goal_tolerance=0.15` /
      `yaw_goal_tolerance=0.20`.
- [ ] Real-robot bringup *intentionally inherits* the new behavior:
      `_patch_params` injects the smoothing BT and YAML pins the
      SmacPlanner2D plugin at every `envelope_factor`. The validation
      lap for the real-robot side lives in
      [`nav2-sim-real-promotion-architecture`](nav2-sim-real-promotion-architecture.md);
      that brief tracks observed regressions and any rollback or
      debouncer follow-ups.
- [ ] On a sim mission whose Nav2 plan curves around an obstacle,
      `align_to_goal_yaw` rotates to the bearing of a waypoint ~1 m
      along the path (not the goal pose's yaw), and the resulting
      traverse keeps `/cmd_vel.vy` peak ≤ 0.3 × `/cmd_vel.vx` peak
      during the early-path phase. Operator can confirm in Foxglove
      by inspecting the chassis yaw at the start of `navigate_to_pose`
      vs the `/plan` direction at the same instant.
- [ ] `test_nav_config.py::TestConstantsInjection` MPPI assertions
      still pass — the velocity-coupled critic / sampling overrides
      stay gated on `envelope_factor > 1.0` and are unchanged here.
- [ ] Unit tests cover all three changes: YAML baseline pins the
      SmacPlanner2D plugin + `allow_unknown=True` +
      `cost_travel_multiplier > 1`; the BT swap fires at every
      `envelope_factor`; the BT structure contains `IsPathValid` +
      `GlobalUpdatedGoal` + `Inverter` + no `DistanceController` +
      keeps `SmoothPath` and `ComputePathToPose`; `_align_to_goal_yaw`
      uses the first lookahead waypoint when the planner returns one
      and falls back to the goal pose's yaw when planning is
      unavailable.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml`
  — `planner_server.GridBased`. SmacPlanner2D + `allow_unknown: true`
  + `cost_travel_multiplier: 2.0`.
- `source/strafer_ros/strafer_navigation/launch/navigation.launch.py`
  — `_patch_params`. The BT swap is applied unconditionally; the
  MPPI sampling / critic rebalance stays gated on
  `envelope_factor > 1.0`.
- `source/strafer_ros/strafer_navigation/config/navigate_to_pose_w_smoothing_and_recovery.xml`
  — the project's canonical navigate-to-pose BT, applied on every
  lane.
- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`
  — `JetsonRosClient.compute_path_to_pose` wraps Nav2's plan-only
  action; consumed by `_align_to_goal_yaw`.
- `source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py`
  — `_align_to_goal_yaw` + the `_path_lookahead_yaw` helper.

## Out of scope

- **Switching planners** to SmacPlannerHybrid / SmacPlannerLattice.
  SmacPlanner2D is the right fit for the omni mecanum chassis;
  Hybrid/Lattice add kinematic constraints we don't need.
- **Costmap layer tuning** (inflation_radius, cost_scaling_factor).
  Untouched here — the path-shape problem is the planner's cost
  evaluation, not the underlying costmap.
- **Real-robot validation lap** for the universal BT + YAML defaults.
  Tracked in
  [`nav2-sim-real-promotion-architecture`](nav2-sim-real-promotion-architecture.md),
  along with the migration plan for the remaining
  `envelope_factor > 1.0`-gated MPPI tuning.
- **A debouncer on `IsPathValid`**. Only file if real-robot or sim
  runs surface observable flapping on transient sensor blips.
- **Replacing `_patch_params`'s envelope_factor gate with a separate
  sim YAML**. The predecessor briefs evaluated and rejected this;
  option (C) in `mppi-critic-tuning-for-sim-envelope.md` is the
  canonical write-up.
