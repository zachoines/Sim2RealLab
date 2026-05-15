# Add a frontier-exploration skill for cross-room target discovery

**Type:** new feature
**Owner:** Either (skill lives on Jetson executor lane; frontier
detector + costmap interaction is Nav2-side. The cross-lane
shape is the same as the autonomy-stack brief.)
**Priority:** P1 (without this primitive, every cross-room
mission to an unseen room fails by construction — see
`autonomy-stack`'s cold-start case)
**Estimate:** M (~3–5 days; frontier detector + skill wrapper +
behavior-tree integration + smoke test in a multi-room scene)
**Branch:** task/frontier-exploration-primitive

## Story

As an **operator running a cold-start mission to a target the
robot has never seen** ("go to the kitchen table" with a fresh
semantic map, robot in the living room), I want **the executor
to have a runtime-legal `explore_until_visible(label)` skill
that rotates / drives toward unmapped frontiers and re-scans
until the target is grounded or the explorable region is
exhausted**, so that **cold-start multi-room missions are
solvable using only the robot's own sensors — closing the gap
§1.10.1 of `STRAFER_AUTONOMY_NEXT.md` named as option 3 and
that `autonomy-stack`'s smoke test depends on**.

## Context bundle

Read these before starting:
- [`context/repo-topology.md`](../../context/repo-topology.md)
- [`context/ownership-boundaries.md`](../../context/ownership-boundaries.md)
- [`context/branching-and-prs.md`](../../context/branching-and-prs.md)
- [`context/conventions.md`](../../context/conventions.md)
- [`autonomy-stack`](autonomy-stack.md) — primary consumer.
  Its multi-room transit-or-explore plan compiler emits a
  call into this skill whenever the inferred target room is
  unknown or unreachable in `connectivity`.
- [`observation-derived-room-state`](observation-derived-room-state.md)
  — sibling. Frontier exploration finds new ground; this skill
  populates the semantic-map nodes that
  observation-derived-room-state then clusters into rooms.

## Context

### Why a primitive, not an ad-hoc loop in `_scan_for_target`

`_scan_for_target` today does a rotate-and-ground loop
([`mission_runner.py:620`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py#L620)).
Extending it to also drive toward frontiers conflates two
concerns: (a) "find a target visible from somewhere close" and
(b) "find a target whose room we have never entered." The
second has different cost, different failure modes, different
acceptance bar (LLM-emitted intent should be able to opt into
the more expensive primitive only when the room-level reasoning
in `autonomy-stack` flags an unknown target room). Keep them
separate skills.

### State of the art

The frontier-exploration literature settled on a small set of
patterns the strafer can lift directly:

- **Wavefront frontier detection** over the global costmap —
  cells on the boundary between known-free and unknown.
  Implemented in `nav2_wavefront_frontier_exploration` and
  `m-explore-ros2` (Nav2 port of `explore_lite`).
- **Gain function:** information gain (frontier cell count) vs.
  travel cost (Nav2 plan length). Tuned per platform but the
  shape is universal.
- **Dynamic-window variants** for large environments
  (`IRVLUTD/dynamic-window-frontier-exploration`).
- **LLM-guided variants** (CogNav, ICCV 2025; LFG; FSR-VLN)
  where the language model biases the gain function toward
  frontiers semantically consistent with the operator's command
  ("kitchen" → prefer frontiers in rooms classified as kitchen
  by the room-state inference). This is the cutting edge and a
  natural follow-on once the unguided primitive is shipping.

Recommended for v1: integrate `m-explore-ros2` or pull its
wavefront detector into the strafer skill set as the frontier
source, drive via Nav2 with the existing reactive staging loop,
ground at each arrival. LLM-guided gain is a P2 follow-up.

### Skill shape

```
explore_until_visible(label: str,
                      max_frontiers: int = 5,
                      max_distance_m: float = 6.0,
                      timeout_s: float = 180.0) -> SkillResult
```

Per frontier-attempt loop:
1. Pull the global costmap + frontier set.
2. Filter to frontiers within `max_distance_m` of the robot,
   rank by gain (cell count) / Nav2 plan cost.
3. Navigate to the top-ranked frontier via the existing
   `navigate_to_pose` skill (this gets the reactive staging
   loop from
   [`nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md)
   for free).
4. Run a small `scan_for_target` on arrival.
5. If grounded: return success with the grounding outputs.
6. Else: remove the frontier from the candidate set, loop.

Bounded by `max_frontiers` AND `timeout_s` — whichever fires
first ends the search. Returns `frontier_set_exhausted` on
failure.

### Integration with autonomy-stack

`autonomy-stack`'s plan-compiler invokes this skill when the
target room is unknown OR unreachable. Steps from the
compiler's perspective:

```
step_01: explore_until_visible(label)   ← this brief
step_02: project_detection_to_goal_pose  ← (uses the grounding
                                            from step_01)
step_03: align_to_goal_yaw
step_04: navigate_to_pose
step_05: verify_arrival
```

The skill's success outputs include `bbox_2d` + `depth` so
`project_detection_to_goal_pose` can run unchanged.

## Acceptance criteria

- [ ] **Skill registration.** `explore_until_visible` is
      registered in
      [`mission_runner.py`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)'s
      skill table (sibling to `scan_for_target`). The compiler
      can emit it as a step.
- [ ] **Frontier source.** Either vendoring `m-explore-ros2`
      under `source/strafer_ros/` OR pulling its wavefront
      detector into a new `strafer_navigation` helper. Choice
      documented in the brief commit, with one-line rationale.
- [ ] **Runtime-legal inputs only.** The skill reads only Nav2
      costmaps, RTAB-Map state, and the semantic map. A grep of
      the new code for `scene_metadata`, `scene_labels`, or
      `room_adjacency` returns zero hits. (Same sim-to-real
      rule as `autonomy-stack`.)
- [ ] **Cold-start smoke test.** In a multi-room Infinigen
      scene, start the robot in room A, mission "go to the
      <object> in room B" where the semantic map is empty. The
      compiler-emitted plan begins with `explore_until_visible`;
      mission succeeds end-to-end via the bridge harness.
      Mission-summary excerpt in the PR description.
- [ ] **Warm-start non-regression.** Repeating the same mission
      after one successful run, the plan no longer contains
      `explore_until_visible` (the room is now in
      `known_rooms`); the compiler picks the transit-step path
      from `autonomy-stack`.
- [ ] **Bounded termination.** With the target genuinely
      unreachable (e.g., behind a wall in a sealed room),
      `explore_until_visible` returns `frontier_set_exhausted`
      within `timeout_s`. No infinite loops.
- [ ] **Unit tests.** Frontier detector tested in isolation
      against a small synthetic occupancy grid. Skill tested
      against a recorded Nav2 costmap + grounding fixture.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide
      under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] **No regression** in single-room missions. Existing
      `scan_for_target`-only plans are unchanged.

## Investigation pointers

- Current scan loop:
  [`mission_runner.py:_scan_for_target`](../../../../source/strafer_autonomy/strafer_autonomy/executor/mission_runner.py)
  around line 620.
- Nav2 costmap query pattern: see the reactive staging loop
  shipped in
  [`nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md)
  — its costmap-bounds query is reusable.
- Frontier detector references:
  - [`m-explore-ros2`](https://github.com/robo-friends/m-explore-ros2)
    — explore_lite port to ROS 2; closest off-the-shelf option.
  - [`nav2_wavefront_frontier_exploration`](https://github.com/SeanReg/nav2_wavefront_frontier_exploration)
    — minimal wavefront detector, easy to vendor.
  - [`dynamic-window-frontier-exploration`](https://github.com/IRVLUTD/dynamic-window-frontier-exploration)
    — large-environment variant; probably overkill for indoor
    home scenes but useful reference for the gain function.
- LLM-guided exploration follow-ups (P2 brief if the unguided
  version is too inefficient): CogNav (ICCV 2025), Language
  Frontier Guide (arXiv:2310.10103), FSR-VLN (arXiv:2509.13733).

## Out of scope

- **LLM-guided frontier gain.** v1 ranks by geometric gain
  only. Biasing toward semantically-consistent frontiers
  (CogNav-style) is a P2 follow-up.
- **Multi-floor exploration.** Strafer mecanum cannot climb
  stairs; stay on the start floor.
- **Re-exploration after map staleness.** If the semantic map
  is stale (objects moved since last scan), this skill does not
  re-explore already-mapped regions. Filed under
  `STRAFER_AUTONOMY_NEXT.md`'s map-lifecycle section.
- **Active SLAM tuning.** RTAB-Map parameter changes for
  exploration efficiency stay out — use the
  [`nav2-far-goal-staging.md`](../../completed/nav2-far-goal-staging.md)
  defaults.
- **Real-D555 deployment validation.** Sim acceptance is the
  bar for this brief; a real-robot trial brief gets filed once
  the sim path is stable.
