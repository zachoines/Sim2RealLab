# The nav2 lane cannot recover from a start pose inside the inflation halo

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

## The open question this brief must answer

**A bare `Fallback` cannot tell *why* `GridBased` failed.** It fires on any
failure, including an unreachable goal — and `NavfnPlanner`'s binary
`allow_unknown` is more permissive than `SmacPlanner2D`'s soft
`cost_travel_multiplier`, so the fallback may return a path through unknown
space that the primary deliberately rejected. That would silently widen what the
nav2 lane is willing to drive.

Resolve one of:

- **(a)** `ComputePathToPose` in this distribution exposes an error-code output
  port — gate the fallback on the start-occupied code only. Check first; this is
  the clean answer if available.
- **(b)** Accept the wider behaviour and bound it: tighten `GridBasedRelaxed`'s
  `allow_unknown` to `false` so it can only escape into known-free space. Cheap,
  and it makes the fallback strictly more conservative than the primary except
  on the start cell.
- **(c)** Reject the BT approach for this lane and document the wedge as an
  operator-intervention case.

(b) is the likely answer; do not ship (a) on an assumption without checking the
port exists.

## Acceptance

- [ ] The nav2 lane plans from a pose that `GridBased` refuses, demonstrated on
      the rig with the same probe as above (cell cost >= 99, `GridBased`
      ABORTED, mission proceeds).
- [ ] The fallback's admissibility question is resolved per the section above,
      with the choice and its reasoning recorded here.
- [ ] `test_nav_config.py`'s BT structural tests updated — at minimum
      `test_bt_keeps_planner_and_follower` and the replan-gating test — and a new
      one pinning that the relaxed planner is reachable only after the primary
      fails.
- [ ] Hybrid lane behaviour unchanged: the generator's own fallback is
      independent of the BT and must not double-engage.

## Out of scope

- The rolling-subgoal generator's fallback, which already shipped.
- Registering `GridBasedRelaxed` — already in `nav2_params.yaml`.
- Changing the footprint or inflation radius to make the halo thinner; that is a
  safety change, not a recovery change.

## Triggered by

PR review of `deploy-hardening` (2026-07-29): "wouldn't the Nav2 paradigm to
handle a robot parked inside the costmap inflation halo be a behavior tree?" —
correct for this lane, which the deploy PR did not cover.
