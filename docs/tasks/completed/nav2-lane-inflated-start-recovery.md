# The nav2 lane cannot recover from a start pose inside the inflation halo

**Status:** Shipped 2026-07-30 in `444f8cd` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/171

**Type:** task / reliability (navigation recovery)
**Owner:** Jetson (needs a rig + bridge to reproduce; the change itself is a BT edit)
**Priority:** P2 (a legitimate park wedges the lane until an operator intervenes)
**Estimate:** S (~half day: one BT edit, its structural tests, one rig reproduction)
**Branch:** `task/nav2-lane-inflated-start-recovery`

## Context bundle

- [context/conventions.md](../../context/conventions.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- Sibling: the hybrid lane's half of this shipped in
  [`deploy-hardening`](https://github.com/zachoines/Sim2RealLab/pull/169), which registers the
  `GridBasedRelaxed` planner this brief reuses.

## The problem (measured)

`SmacPlanner2D` refuses a start cell at or above the inscribed value, so a robot
that legitimately parks near an obstacle cannot be planned from. Measured on the
rig 2026-07-29 at cell cost **99, 0.200 m from lethal**:

```
GridBased          -> ABORTED    Starting point in lethal space! Cannot create feasible plan..
GridBasedRelaxed   -> SUCCEEDED  (57 poses)
```

The **hybrid** lane now escapes this itself: the rolling-subgoal generator retries
on `GridBasedRelaxed` and releases once the robot has moved clear.

The **nav2** lane does not, and its behavior tree cannot rescue it.
`config/navigate_to_pose_w_smoothing_and_recovery.xml` hardcodes the planner:

```xml
<ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
```

and every recovery available to it fails for this pose:

| Recovery | Why it does not clear the wedge |
|---|---|
| `ClearEntireCostmap` (global) | clears the **obstacle** layer; the wall is in the **static** layer from rtabmap and inflation is computed over the combined costmap. Also discards real obstacle knowledge. |
| `Spin` | rotates without translating — rotation cannot leave an inflation halo |
| `BackUp` | **measured: refuses** — `Collision Ahead - Exiting DriveOnHeading`, the same root cause |
| `Wait` | changes nothing |

So the outer `RecoveryNode` burns its 6 retries and the mission fails. Nothing
stock solves it either: Nav2's `navigate_to_pose_w_replanning_and_recovery.xml`
has the same hardcoded planner and the same actuating recoveries, and
`nav2_planner_selector_bt_node` (already in our `plugin_lib_names`) selects a
planner from a **topic** for external selection, not on failure.

## Proposed change

Make the plan step a `Fallback` over the two planners, inside the existing
`RecoveryNode` and before the costmap clear:

```xml
<Sequence name="PlanAndSmooth">
  <Fallback name="PlanWithRelaxedFallback">
    <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
    <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBasedRelaxed"/>
  </Fallback>
  <SmoothPath unsmoothed_path="{path}" smoothed_path="{path}" smoother_id="simple_smoother"/>
</Sequence>
```

`GridBasedRelaxed` is already registered in `nav2_params.yaml`, so this needs no
new plugin.

## The open question this brief must answer — RESOLVED

**A bare `Fallback` cannot tell *why* `GridBased` failed.** It fires on any
failure, including an unreachable goal — and `NavfnPlanner`'s binary
`allow_unknown` is more permissive than `SmacPlanner2D`'s soft
`cost_travel_multiplier`, so the fallback may return a path through unknown
space that the primary deliberately rejected. That would silently widen what the
nav2 lane is willing to drive.

The coordinator ruled the fallback must be gated on the failure's **cause**.
The three candidates resolved as follows, checked before shipping:

- **(a) is unavailable.** `nav2_msgs/action/ComputePathToPose` in this
  distribution has no `error_code` field, and the BT node's `providedPorts` are
  `path` / `goal` / `start` / `planner_id` only. Nor is there a stock condition
  node that could ask the question another way: the registered Humble BT
  vocabulary has no costmap-reading condition (`AreErrorCodesPresent` and
  `WouldAPlannerRecoveryHelp` are Iron and later), and BT.CPP is 3.8.7, so there
  is no scripting either.
- **(b) rejected.** `GridBasedRelaxed` is also the hybrid lane's escape hatch;
  tightening its `allow_unknown` would change hybrid-lane behaviour, which this
  PR must leave alone.
- **(c) not needed.**

**Shipped:** the cause is probed *outside* the tree and delivered through
`planner_selector`, the seam Nav2 already provides — `start_cell_planner_selector`
reads the robot's own cell in `global_costmap/costmap_raw` and names
`GridBasedRelaxed` only while that cell is at or above the inscribed value.
`<PlannerSelector default_planner="GridBased">` means the retry branch is a
plain re-run of the primary for every other failure, so the escape hatch never
sees a refusal it cannot fix. The probe takes the max over the planner's
downsample block, because `SmacPlanner2D` plans on a 2× downsampled grid whose
cells carry the max cost of the block they cover.

**What it over-admits:** the gate is evaluated at plan time rather than at
failure time, so a planner failure that is *not* the inflated start but happens
while the robot is parked in the halo also reaches the relaxed branch. Tolerable
because an inflated start is exactly the case where the primary cannot answer at
all, and because the window is bounded to the halo, which the robot leaves as
soon as it moves.

**What it costs:** when the gate is closed, a genuine planner failure runs the
primary twice per `RecoveryNode` cycle instead of once (bounded by
`max_planning_time: 2.0`). That is the price of never letting the escape hatch
pre-empt an attempt the primary would have won.

**Visibility:** both directions log at WARN, rate-limited — the gate opening and
closing from the node's own probe, and the relaxed branch actually firing, read
off `behavior_tree_log`.

## Acceptance

- [x] The nav2 lane plans from a pose that `GridBased` refuses — demonstrated on
      a **container** Nav2 stack (`strafer-cpu:humble`, synthetic map + TF), not
      the rig. Parked 0.20 m from lethal: probe reads 253, selector publishes
      `GridBasedRelaxed`, `GridBased` ABORTS with `Starting point in lethal
      space! Cannot create feasible plan..`, `GridBasedRelaxed` SUCCEEDS,
      `ComputePathToPoseRelaxed` reaches SUCCESS in `/behavior_tree_log`, both
      WARNs fire, `/plan` published (111 poses).
- [ ] **PENDING — rides the next rig session:** the same reproduction on the rig
      with the brief's own probe. Not synthesized here; the container run above
      is evidence about the BT and the gate, not about the rig.
- [x] The fallback's admissibility question is resolved per the section above,
      with the choice and its reasoning recorded here.
- [x] `test_nav_config.py`'s BT structural tests updated —
      `test_bt_keeps_planner_and_follower` now pins that the first
      `ComputePathToPose` names the primary literally, the replan-gating test
      pins that no planner call sits inside the path-validity gate, and
      `test_relaxed_planner_reachable_only_after_the_primary_fails` pins the
      two-gate structure. 18 new tests in all; ROS suite 627 passed.
- [x] Hybrid lane behaviour unchanged: `test_hybrid_lane_fallback_stays_independent`
      pins that the generator keeps its own `fallback_planner_id` and gains no
      selector-topic input. The two select the same registered planner by name
      on separate request paths and cannot double-engage.

## Found while verifying

`GridBasedRelaxed` clears only the robot's *own* cell, so it needs a free
neighbour to propagate into: measured planning from ~0.20 m off lethal and
refusing from ~0.15 m. Pre-existing behaviour, shared with the hybrid lane's
escape hatch, unchanged by this work — recorded in the sim-bridge cheatsheet so
an operator reading either lane's WARN knows the limit.

## Out of scope

- The rolling-subgoal generator's fallback, which already shipped.
- Registering `GridBasedRelaxed` — already in `nav2_params.yaml`.
- Changing the footprint or inflation radius to make the halo thinner; that is a
  safety change, not a recovery change.

## Triggered by

PR review of `deploy-hardening` (2026-07-29): "wouldn't the Nav2 paradigm to
handle a robot parked inside the costmap inflation halo be a behavior tree?" —
correct for this lane, which the deploy PR did not cover.
