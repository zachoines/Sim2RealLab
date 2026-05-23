# Commit-and-follow Nav2 global paths: prefer known-free, replan on invalidation

**Type:** task / tuning
**Owner:** Jetson agent (`source/strafer_ros/strafer_navigation/`)
**Priority:** P2
**Estimate:** S–M (~half day; YAML + BT XML + nav-config tests, no Python logic)
**Branch:** task/nav2-commit-and-follow-path-stability

## Story

As a **mission operator running sim-in-the-loop missions**, I want
**Nav2's global plan to (a) prefer already-mapped free space over
unknown cells and (b) stay committed to the chosen path until a real
obstacle invalidates it**, so that **the robot drives obvious routes
instead of cutting through unknown bands, and MPPI tracks a stable
reference instead of a path that jitters every replan tick**.

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

Two changes, both sim-lane only — real-robot bringup keeps stock Nav2
behavior verbatim. Mirror the existing `_patch_params` pattern that
gates the smoothing BT and MPPI critic overrides on `envelope_factor >
1.0`.

### A. `allow_unknown: false` on the sim lane

In `_patch_params`, when `envelope_factor > 1.0`, set
`planner_server.ros__parameters.GridBased.allow_unknown = false`.
NavfnPlanner then treats `NO_INFORMATION` as untraversable, so the
emitted path can only use cells that have been observed at least
once. Already-mapped free space wins over unknown bands by
construction.

Caveat — cross-room cold-start: if the goal sits in unmapped space,
planning fails outright. The executor already has the
`explore_until_visible` skill for that flow (frontier-driven discovery
before a final `navigate_to_pose`), so the cold-start path is covered.
A goal that's reachable only through unknown space *without* a prior
explore is a planner-level failure with `goal_unreachable` as the
error code — the operator can either issue an `explore` first or
relax `allow_unknown` per-launch if blocked.

Pros: surgical YAML diff in `_patch_params`; real-robot lane keeps
the permissive default; aligns with the operator's "prefer verified
empty regions" intent literally.

Cons: a goal projected just outside the known-free footprint (e.g.
right against the camera-blind-spot donut) becomes unreachable until
the executor's chassis-side rotation or a deliberate explore widens
the known footprint. Symptom is "no plan available" rather than a
wonky path, which is the easier failure mode to diagnose.

### B. IsPathValid-gated `Fallback` replaces `DistanceController`

In `navigate_to_pose_w_smoothing_and_recovery.xml`, replace the
`<DistanceController distance="0.5">` decorator with a
`<ReactiveFallback>` whose first child is a `<ReactiveSequence>` of
`<Inverter><GoalUpdated/></Inverter>` and
`<IsPathValid path="{path}"/>`. `IsPathValid` succeeds when the
current `{path}` is still collision-free against the latest costmap;
the inverted `GoalUpdated` succeeds when no new goal has arrived
since the last tick. When both succeed the `ReactiveFallback`
short-circuits without re-planning. When either fails (path blocked
or new goal), the fallback falls through to the existing
`ComputePathToPose → SmoothPath` sequence.

Net BT shape (sim lane only):

```
PipelineSequence
├─ ReactiveFallback (ReplanIfNeeded)
│  ├─ ReactiveSequence
│  │   ├─ Inverter(GoalUpdated)
│  │   └─ IsPathValid(path)
│  └─ RecoveryNode(ComputePathToPose → SmoothPath, ClearGlobalCostmap)
└─ RecoveryNode(FollowPath, ClearLocalCostmap)
```

Pros: replanning is event-driven (path invalid / goal changed) rather
than motion-driven; the path the controller tracks is now stable
between invalidation events. The plugin
(`nav2_is_path_valid_condition_bt_node`) is already linked in
`bt_navigator.plugin_lib_names` from the predecessor.

Cons: a transient sensor blip (one scan that briefly marks a cell
lethal across the current path) can trigger a one-shot replan that
mostly reverts on the next tick once the blip clears. The local
costmap clearing recovery in the existing BT covers the recovery
branch already. If the operator sees pathological flapping the fix
is a debouncer on `IsPathValid` (out of scope for this brief).

**Sequence:** A first to fix the route shape, then B to fix the path
stability. Either can ship without the other, but A on its own leaves
symptom 2, and B on its own leaves symptom 1.

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
- [ ] Real-robot bringup is unaffected: the
      `test_nav_config.py::TestConstantsInjection` test still
      asserts that `envelope_factor=1.0` leaves the controller config
      byte-identical to the YAML baseline, and a new assertion pins
      `default_nav_to_pose_bt_xml` absent + `allow_unknown=true`
      on the real-robot lane.
- [ ] Unit tests cover both knobs: the YAML override is gated on
      `envelope_factor > 1.0`, the new BT structure parses + contains
      `IsPathValid` + `GoalUpdated` + no `DistanceController` +
      keeps `SmoothPath` and `ComputePathToPose`.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- `source/strafer_ros/strafer_navigation/config/nav2_params.yaml:211–220`
  — `planner_server.GridBased.allow_unknown`. Currently `true`.
- `source/strafer_ros/strafer_navigation/launch/navigation.launch.py:120–193`
  — `_patch_params`. The sim-lane override block at
  `envelope_factor > 1.0` is the pattern to mirror for the new
  `allow_unknown` override.
- `source/strafer_ros/strafer_navigation/config/navigate_to_pose_w_smoothing_and_recovery.xml`
  — the sim BT. Today's `DistanceController distance="0.5"` is the
  knob to replace.
- `source/strafer_ros/strafer_navigation/test/test_nav_config.py:408–584`
  — `TestSmoothingBT`. `test_bt_uses_distance_controller_not_rate_controller`
  pins the predecessor's choice; rename + replace its assertions for
  the IsPathValid-gated structure.

## Out of scope

- **Switching planners** (SmacPlanner2D, ThetaStar, SmacHybrid).
  NavfnPlanner is current; planner-swap requires its own validation.
- **Costmap layer tuning** (inflation_radius, cost_scaling_factor).
  Untouched here — the path-shape problem is the planner's cost
  evaluation, not the underlying costmap.
- **Real-robot path-stability**. Stock Nav2 BT stays on the
  real-robot lane until sim validation lands; a follow-up brief can
  promote the IsPathValid pattern after that.
- **A debouncer on `IsPathValid`**. Only file if sim runs surface
  observable flapping on transient sensor blips.
- **Replacing `_patch_params`'s envelope_factor gate with a
  separate sim YAML**. The predecessor briefs evaluated and rejected
  this; option (C) in `mppi-critic-tuning-for-sim-envelope.md` is
  the canonical write-up.
