# Batched GPU wavefront planner for the subgoal training env

**Type:** refactor / performance
**Owner:** DGX (`strafer_lab` lane — replaces the reset-time path planner used by
[`subgoal-env`](../../completed/subgoal-env.md)'s `SubgoalCommand`)
**Priority:** P3 — pure optimization. The current numpy A* is not a measured
bottleneck (see Context); this brief is filed-on-trigger, not blocking anything.
**Estimate:** M (~2–3 days: GPU wavefront planner + path extraction + vectorized
`set_paths` + parity tests + benchmark)
**Branch:** task/batched-gpu-path-planner

## Un-park trigger

Un-park when **profiling** shows reset-time path planning is actually gating
training throughput — e.g. a config that resets many envs per step (much larger
parallel env counts than the current ~256, much shorter episodes, or a
curriculum that terminates aggressively), where the per-step planning cost rises
from the current sub-millisecond into the same order as the Kit sim step. Do not
un-park on intuition; attach a profile (`torch.profiler` or wall-clock around
`_resample_command`) showing the planner is a top cost. If the profile is flat,
leave it parked — the numpy A* is correct and adequate.

Un-park by `git mv parked/trained-policy/batched-gpu-path-planner.md active/trained-policy/batched-gpu-path-planner.md` in the PR that picks it up.

## Story

As a **`strafer_lab` operator scaling subgoal training to large parallel env
counts**, I want **the per-episode path planning to run batched on the GPU
instead of a per-env Python/numpy A\* loop**, so that **reset-time planning stays
negligible even when many environments reset on the same step**.

## Context bundle

Read these before starting:

- [`subgoal-env.md`](../../completed/subgoal-env.md) — the predecessor that built
  the planner this brief replaces. Its `## Approach > Phase 1` records why a
  custom planner (not Nav2's) is used at training time, and the
  collision-free / arc-length-discretization / `nav_msgs/Path`-shape output
  contract that this brief must preserve byte-for-byte semantically.
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md) —
  `proc_room.py` is shared by all ProcRoom envs; the planner module and
  `SubgoalCommand` are subgoal-only.
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
  — the planner does not touch the hashed obs/action contract, but its output
  is what the (unhashed, runtime) subgoal command emits.

## Context

The current planner is correct and, as measured during the subgoal-env PR, **not
a bottleneck** — this brief exists to remove a *future* scaling cliff, not a
present cost:

- `path_planner/planner.py::plan_path` is an 8-connected numpy A\* on the 80×80
  inflated occupancy grid. Measured **~0.37 ms mean / 0.92 ms p95** per call.
- `mdp/commands.py::SubgoalCommand._resample_command` calls it in a **per-env
  Python loop** over the resetting envs, then `perturb_waypoints` per env.
- `path_planner/cursor.py::PathCursor.set_paths` installs the resulting
  variable-length paths in a **per-env Python loop** (buffer write + one
  arc-length cumsum each).
- Both loops run only at **episode reset**, for the subset resetting that step.
  With ~600-step episodes staggered across envs this is ~1–3 envs/step
  (~0.7 ms/step) in steady state, plus a one-time ~0.1 s hitch when all envs
  reset at init. The overnight 256-env / 3000-iter baseline run was not gated by
  it.
- The genuinely hot path — `PathCursor.update`, run every step for every env —
  is **already fully vectorized** and is out of scope here.

A\* does not batch cleanly: it is inherently sequential (data-dependent
priority-queue expansion). The tractable batched formulation is a **GPU
wavefront / distance-field planner**, and the prior art already lives in this
repo: `proc_room.py::_gpu_bfs` runs a parallel BFS across all envs via iterative
morphological dilation (`max_pool2d`) on the inflated free-space grid. This brief
extends that pattern from "reachability mask" to "planned path."

## Approach

1. **Batched goal-rooted wavefront.** Run the `_gpu_bfs`-style dilation from each
   env's goal cell (not the room center), accumulating a per-cell **step count**
   (BFS layer index) → an integer distance-to-goal field per env, all envs at
   once on the GPU. Disconnected start (goal unreachable) is detectable as an
   un-reached start cell, replacing A\*'s `NoPathError`.
2. **Path extraction by gradient descent.** From each env's start cell, walk
   greedily to the lowest-distance 8-neighbor until the goal is reached — a
   batched gather loop bounded by the longest path length, not by env count.
   This yields the cell path; shortcut + resample to the same uniform arc-length
   spacing the current `resample_polyline` produces.
3. **Vectorized `set_paths`.** With paths produced as a padded GPU tensor (not a
   Python list), batch the buffer write and the arc-length cumsum across envs in
   one shot — removing the second per-env loop.
4. **Keep the numpy planner** as the reference oracle for the parity test (and
   optionally as a CPU fallback for single-env / non-CUDA use).

The waypoint-noise / lookahead semantics, the `nav_msgs/Path` output shape, and
the collision-free guarantee are unchanged — this is an implementation swap
behind the same `plan_path`-equivalent contract.

## Acceptance criteria

- [ ] The batched planner produces collision-free paths that reach the goal on
      the same synthetic obstacle configs the current
      `tests/navigation/test_path_planner.py` uses, and raises a meaningful
      error (or flags the env) when start/goal are disconnected.
- [ ] **Parity test:** on a batch of randomized rooms, the batched planner's
      paths match the numpy A\* reference within a bounded median deviation along
      arc length (wavefront is shortest-path-in-cells like A\*, so they should be
      close; allow for tie-breaking differences).
- [ ] **Benchmark recorded in the PR:** reset-time planning wall-clock vs the
      numpy loop at increasing env counts (e.g. 64 / 256 / 1024), showing the
      crossover where the GPU planner wins. If there is no crossover at
      realistic counts, that is itself a finding — report it and re-park.
- [ ] `PathCursor.update` stays untouched and still passes its vectorized-tracking
      tests; the subgoal command / reward / termination contracts are unchanged.
- [ ] All four `Isaac-Strafer-Nav-RLNoCam-Subgoal-*` task IDs still `gym.make` +
      step, and the composition golden hashes are unchanged (planner is runtime,
      not cfg).
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit.

## Investigation pointers

- `source/strafer_lab/strafer_lab/tasks/navigation/path_planner/planner.py` —
  the numpy A\* to replace / keep as oracle.
- `source/strafer_lab/strafer_lab/tasks/navigation/path_planner/cursor.py:79` —
  `PathCursor.set_paths`, the second per-env loop to vectorize.
- `source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py` —
  `SubgoalCommand._resample_command`, the per-env planning loop.
- `source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py` —
  `_gpu_bfs`, `_inflate_obstacles`, `_xy_to_grid` / `_grid_to_xy`: the existing
  batched-BFS prior art and grid coordinate helpers to reuse.
- The subgoal-env PR review threads that prompted this brief (vectorize the
  reset-time planner / `set_paths` loops), with the timing data.

## Out of scope

- **The per-step `PathCursor.update`** — already vectorized; do not rewrite it.
- **Path output semantics** — must stay `nav_msgs/Path`-shaped (uniform
  arc-length waypoints, exact endpoints) so the inference-time hybrid backend
  reads the same shape. This brief changes *how* the path is computed, not what
  it is.
- **The obstacle-inflation model** — the disc inflation shipped with subgoal-env
  is the input grid; this brief consumes it, doesn't change it.
- **Deployment-side planning** — Nav2 plans at deployment; this is a
  training-time optimization only.
