# Commit-and-follow Nav2 global paths: prefer known-free, replan on invalidation

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
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/nav2-startup-unknown-donut-path-noise.md](../../completed/nav2-startup-unknown-donut-path-noise.md)
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
[`nav2-sim-real-promotion-architecture`](../tooling/nav2-sim-real-promotion-architecture.md).

### A. `allow_unknown: false` as the universal YAML default

Flip the YAML baseline at
`planner_server.ros__parameters.GridBased.allow_unknown` from `true`
to `false`. NavfnPlanner then treats `NO_INFORMATION` as
untraversable, so the emitted path can only use cells the costmap
has observed at least once. Already-mapped free space wins over
unknown bands by construction on every lane.

Caveat — cross-room cold-start: if the goal sits in unmapped space,
planning fails outright. The executor already has the
`explore_until_visible` skill for that flow (frontier-driven discovery
before a final `navigate_to_pose`), so the cold-start path is covered.
A goal that's reachable only through unknown space *without* a prior
explore is a planner-level failure with `goal_unreachable` as the
error code — the operator can either issue an `explore` first or
relax `allow_unknown` per-launch if blocked.

Pros: one source of truth in YAML; closes the sim-to-real gap for
this knob immediately; aligns with the operator's "prefer verified
empty regions" intent on real *and* sim.

Cons: a goal projected just outside the known-free footprint (e.g.
right against the camera-blind-spot donut) becomes unreachable until
the executor's chassis-side rotation or a deliberate explore widens
the known footprint. Symptom is "no plan available" rather than a
wonky path, which is the easier failure mode to diagnose. Real-robot
sessions with persisted RTAB-Map state are unlikely to hit this in
practice (the cold-start unknown footprint is small); fresh real
sessions will hit it the same way sim does.

### B. IsPathValid-gated `Fallback` as the universal BT

In `navigate_to_pose_w_smoothing_and_recovery.xml`, replace the
`<DistanceController distance="0.5">` decorator with a
`<Fallback name="ReplanIfNeeded">` whose first child is a
`<Sequence>` of `<Inverter><GloballyUpdatedGoal/></Inverter>` and
`<IsPathValid path="{path}"/>`. `IsPathValid` succeeds when the
current `{path}` is still collision-free against the latest costmap;
the inverted `GloballyUpdatedGoal` succeeds when no new goal has
arrived since the last tick. When both succeed the `Fallback`
short-circuits without re-planning. When either fails (path blocked
or new goal), the fallback falls through to the existing
`ComputePathToPose → SmoothPath` sequence.

Use `GloballyUpdatedGoal`, **not** `GoalUpdated` — the latter resets
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
│  │   ├─ Inverter(GloballyUpdatedGoal)
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
[`nav2-sim-real-promotion-architecture`](../tooling/nav2-sim-real-promotion-architecture.md)
follow-up tracks the real-robot validation lap and any debouncer
work that comes out of it.

**Sequence:** ship both at once — they're complementary fixes for the
same operator complaint. A on its own leaves symptom 2; B on its own
leaves symptom 1.

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
      [`completed/nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md)'s
      reference mission: mission still completes end-to-end with
      ≥ 2 staging legs. Plan stability + known-free preference may
      slow the path discovery in the worst case (planning fails until
      `explore_until_visible` widens known free); document any
      slowdown in the PR description.
- [ ] No regression on cornering smoke from
      [`completed/mppi-critic-tuning-for-sim-envelope.md`](../../completed/mppi-critic-tuning-for-sim-envelope.md):
      `translate forward 1 m → rotate 90° → translate forward 1 m`
      lands within `xy_goal_tolerance=0.15` /
      `yaw_goal_tolerance=0.20`.
- [ ] Real-robot bringup *intentionally inherits* the new behavior:
      `_patch_params` injects the smoothing BT and YAML pins
      `allow_unknown: false` at every `envelope_factor`. The
      validation lap for the real-robot side lives in
      [`nav2-sim-real-promotion-architecture`](../tooling/nav2-sim-real-promotion-architecture.md);
      that brief tracks observed regressions and any rollback or
      debouncer follow-ups.
- [ ] `test_nav_config.py::TestConstantsInjection` MPPI assertions
      still pass — the velocity-coupled critic / sampling overrides
      stay gated on `envelope_factor > 1.0` and are unchanged here.
- [ ] Unit tests cover both knobs: YAML baseline asserts
      `allow_unknown=False` and `_patch_params` doesn't re-enable it
      at any `envelope_factor`; the BT swap fires at every
      `envelope_factor`; the BT structure parses and contains
      `IsPathValid` + `GloballyUpdatedGoal` + `Inverter` + no
      `DistanceController` + keeps `SmoothPath` and `ComputePathToPose`.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml`
  — `planner_server.GridBased.allow_unknown`. Universal YAML default
  is now `false`.
- `source/strafer_ros/strafer_navigation/launch/navigation.launch.py`
  — `_patch_params`. The BT swap is now applied unconditionally
  (lifted out of the `envelope_factor > 1.0` block); the MPPI
  sampling / critic rebalance stays gated.
- `source/strafer_ros/strafer_navigation/config/navigate_to_pose_w_smoothing_and_recovery.xml`
  — the project's canonical navigate-to-pose BT, applied on every
  lane.
- `source/strafer_ros/strafer_navigation/test/test_nav_config.py`
  — `TestSmoothingBT` + the `TestConstantsInjection` knob tests pin
  the universal-vs-gated split.

## Out of scope

- **Switching planners** (SmacPlanner2D, ThetaStar, SmacHybrid).
  NavfnPlanner is current; planner-swap requires its own validation.
- **Costmap layer tuning** (inflation_radius, cost_scaling_factor).
  Untouched here — the path-shape problem is the planner's cost
  evaluation, not the underlying costmap.
- **Real-robot validation lap** for the universal BT + YAML defaults.
  Tracked in
  [`nav2-sim-real-promotion-architecture`](../tooling/nav2-sim-real-promotion-architecture.md),
  along with the migration plan for the remaining
  `envelope_factor > 1.0`-gated MPPI tuning.
- **A debouncer on `IsPathValid`**. Only file if real-robot or sim
  runs surface observable flapping on transient sensor blips.
- **Replacing `_patch_params`'s envelope_factor gate with a separate
  sim YAML**. The predecessor briefs evaluated and rejected this;
  option (C) in `mppi-critic-tuning-for-sim-envelope.md` is the
  canonical write-up.
