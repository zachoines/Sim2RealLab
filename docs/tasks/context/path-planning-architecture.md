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
| `scene-connectivity-validation` (the **producer**) | Infinigen USD physics colliders | **generates** the cached occupancy via Isaac Sim's occupancy-map extension ([`validate_scene_connectivity.py`](../../../source/strafer_lab/scripts/validate_scene_connectivity.py)) → `<scene>/occupancy.npy` | single-scene, scene-gen time |
| `mission-generator` oracle / waypoint validation | the cached `<scene>/occupancy.npy` | **load** it + [`scene_connectivity.occupancy_to_free_space`](../../../source/strafer_lab/strafer_lab/tools/scene_connectivity.py) (invert + robot-radius disc inflation) | single-scene, episode-gen time, latency-tolerant |
| `grounding-negative-taxonomy` `trajectory_violation` | the cached `<scene>/occupancy.npy` | the same load + invert/inflate adapter | single-scene |
| `path-statistics` corridor/aperture measurement | either — ProcRoom grids from the CPU stub, scanned scenes from the cached `<scene>/occupancy.npy` | [`tools/path_statistics.py`](../../../source/strafer_lab/strafer_lab/tools/path_statistics.py) consumes whichever `free_space` the source already builds, plus that source's *raw* occupancy for clearance | offline, CPU, both sources measured identically |

The ProcRoom grid builder is GPU-batched for training-reset throughput.
The Infinigen consumers share a **cached-occupancy seam** instead of each
re-rasterizing the scene: `scene-connectivity-validation` generates the
occupancy *once per scene at generation time* from the USD's physics
colliders (Isaac Sim's occupancy-map extension `isaacsim.asset.gen.omap` —
more accurate than rasterizing object AABBs, and it sees the doorway
free-space a rasterizer misses) and caches it as `<scene>/occupancy.npy`
(+ `occupancy.json` for the grid→world mapping). Downstream consumers
(mission-generator oracle, grounding-negative trajectory checks) **load that
cached grid** and run the one shared invert/inflate adapter
`scene_connectivity.occupancy_to_free_space` (occupied → free, dilated by
the robot radius with a Euclidean disc) before calling `plan_path` — they do
not re-rasterize. The adapter is the one shared seam; the grid is written
once. The bridge/training **robot spawn** derivation
(`scene_connectivity.spawn_pool_from_occupancy`, via the env cfg's
`derive_infinigen_scene_spawn` and the coverage driver's `_derive_spawn_xy`)
is a further consumer of the same seam — it loads the cached grid and runs the
invert/inflate/seal adapters to pick free, in-room spawn cells — but it is
**plan-free**: it selects from `free_space` directly and does not call
`plan_path`, so a bridge spawn is "free on the inflated grid + in a room", not
per-leg planner-reachable. The inflation *radius* differs from the GPU training grid's by design:
the GPU grid uses the rotation-invariant **circumscribed** radius for RL
obstacle-avoidance, while the connectivity check uses the **inscribed** radius
(half-width) — the holonomic mecanum rotates to thread a ~0.55 m doorway, so
the circumscribed radius (~0.28 m, robot-as-0.56 m-disc) would wrongly seal
standard doorways and mark rooms unreachable. Same disc kernel, different
radius.

Because the omap reads *physics colliders*, a collider that does not match its
visual mesh distorts the grid: thin perimeter trim (skirting / baseboard) given a
`convexHull` approximation fills the whole room footprint with a phantom
floor-height slab, and the omap reads the entire navigable floor as occupied.
Authoring faithful colliders is the **scene provider's source-specific
postprocess** responsibility (the agnostic occupancy generator only rasterizes
what the colliders say), so the provider's bake assigns the exact mesh — not a
convex hull — to such structural trim. Note the 2D omap rasterizes a single
cell-thick horizontal slice just above the floor, not the full `[z_lo, z_hi]`
band: an in-slice collider is what the grid records, so raising the upper bound
does not catch taller obstacles.

This supersedes the earlier plan of a per-consumer CPU/numpy
`scene_metadata.json → free_space` rasterizer. That footprint+AABB
rasterizer survives only as `validate_scene_connectivity.py`'s
`--rasterize-fallback`, for when the occupancy-map extension is unavailable;
the cached-occupancy seam is the default.

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
