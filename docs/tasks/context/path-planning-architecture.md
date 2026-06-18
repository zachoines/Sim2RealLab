# Path-planning architecture — one shared grid planner, many consumers

The spine every brief that plans a collision-free path through scene
geometry references instead of writing its own A*. It records the
**settled** decision (operator + orchestrator, 2026-06-17, off the
[`harness-architecture`](../active/harness/harness-architecture.md) Tier 2
review) that the project has **one** path-planning core, and that new
consumers adapt their scene representation onto it rather than growing a
second or third planner.

---

## The decision

**One scene-agnostic grid A\* core; per-scene-source occupancy-grid
adapters.** Any brief that needs a collision-free waypoint path —
RL subgoal training, harness oracle/scripted paths, mission-generator
waypoint validation, hard-negative trajectory generation — plans with
[`strafer_lab.tasks.navigation.path_planner`](../../../source/strafer_lab/strafer_lab/tasks/navigation/path_planner/). What differs
per consumer is **only the occupancy grid** fed in, not the planner.

This reverses the prior drift toward per-brief A* helpers: the
`mission-generator` brief sketched "reuse the connectivity-validation
brief's A* helper," `scene-connectivity-validation` had not yet written
one, and `subgoal-env` shipped a third for RL training. The decision
collapses those onto the subgoal-env planner, which already shipped and
is tested.

---

## The core (shipped by `subgoal-env`, PR #87)

`path_planner` is scene-agnostic — it takes a boolean grid, not scene
objects:

| Symbol | Contract |
|---|---|
| `plan_path(start_xy, goal_xy, free_space, *, grid_res, grid_origin_xy, discretization_m=0.05, snap_radius_m=0.3)` | Grid A* on a `(rows, cols)` bool `free_space` grid (**already robot-radius-inflated**). Returns `(N, 2)` env-local waypoints, arc-length-resampled, first==start / last==goal. Raises `InvalidEndpointError` / `NoPathError`. |
| `resample_polyline(points, spacing)` | Arc-length resample any polyline to fixed spacing. |
| `perturb_waypoints(...)` | Training-only path noise (Option-B robustness for RL). **Not** for demonstration/oracle paths. |
| `PathCursor` / `PathCursorState` | Rolling along-arc-length cursor (the subgoal lookahead mechanism). |

The planner does not know what a "room" or an "Infinigen object" is.
That is the whole point: it is reusable because the scene is abstracted
to a grid before it arrives.

---

## What is per-consumer — the occupancy grid

Each consumer rasterizes **its** scene representation into the same
`free_space` grid shape, then calls `plan_path`:

| Consumer | Scene source | Grid builder | Regime |
|---|---|---|---|
| `subgoal-env` RL training | ProcRoom procedural boxes | [`mdp/proc_room.py`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py) `_build_occupancy_grid` + `_inflate_obstacles` | GPU-batched `(B, G, G)`, per-reset, throughput-bound |
| `mission-generator` oracle / waypoint validation | Infinigen `scene_metadata.json` (object AABBs + room polygons) | **new** CPU/numpy adapter (this is the brief's work) | single-scene, episode-gen time, latency-tolerant |
| `scene-connectivity-validation` navigable-mask check | Infinigen `scene_metadata.json` | the same CPU adapter as mission-generator | single-scene |
| `grounding-negative-taxonomy` `trajectory_violation` | Infinigen `scene_metadata.json` | the same CPU adapter | single-scene |

The ProcRoom grid builder is GPU-batched for training-reset throughput;
the Infinigen consumers run **once per scene at generation time**, so
they want a plain CPU/numpy `scene_metadata.json → free_space` rasterizer
(object AABBs + room-boundary polygons, inflated by the robot radius to
the same grid convention). That adapter is shared across the three
Infinigen consumers — write it once.

---

## The invariant that keeps it coherent

**A new path-planning consumer writes a grid adapter, not a planner.**
If a brief finds itself implementing A* / BFS / RRT, that is the signal
it has drifted from this decision — stop and reuse `path_planner`. The
only sanctioned reasons to touch the core itself are a measured planner
deficiency (and then it is an architecture change reviewed like any
other) or the parked
[`batched-gpu-path-planner`](../parked/trained-policy/batched-gpu-path-planner.md)
unification, which is the one in-flight planner-core evolution.

`perturb_waypoints` is training-only. Demonstration, oracle, and
hard-negative paths must be clean (un-perturbed) — perturbation exists to
make the RL policy robust to planner-quirk noise, not to corrupt a
recorded trajectory.

---

## Distributional-gap caveat (carried from `subgoal-env`)

`subgoal-env` chose a custom grid A* (its "Option B") over Nav2's
offline planner. At deployment the hybrid backend tracks paths from
Nav2's *actual* planner, so the training planner's quirks differ from
Nav2's. The RL policy hedges this with `perturb_waypoints`. Harness
consumers don't have a Nav2-parity bar — their paths are demonstration
trajectories, not a tracked reference — so the same custom planner is
fine for them as-is. If a future consumer *does* need Nav2 parity, that
is a planner-core decision and revises this module.

---

## Maintenance contract

Same rule as the other context modules: a brief that adds, removes, or
re-homes a path-planning consumer updates the consumer table here in the
same PR. A change to the **core** (the planner API, the grid convention,
the perturbation policy) is an architecture change — it revises this
module and is reviewed like any other PR.
