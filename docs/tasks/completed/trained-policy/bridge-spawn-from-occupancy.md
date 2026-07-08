# Converge the bridge/training robot spawn onto the occupancy sidecar and fix the cross-scene spawn-union bug

**Status:** Shipped 2026-06-30 in `e7ec3ae` (DGX). The bridge/training Infinigen env now derives the robot spawn from ONLY the loaded scene's `occupancy.npy` free-space (shared `scene_connectivity.spawn_pool_from_occupancy` core via `derive_infinigen_scene_spawn`) and pins `spawn_z` from that scene's floor — no cross-scene union, no pooled-max. `run_sim_in_the_loop --scene-usd` re-derives spawn/floor for the overridden scene; coverage's `_derive_spawn_xy` wraps the same core, behaviorally unchanged. Validated Kit-free against the corpus (256 in-room free cells, per-scene `spawn_z`, all three Infinigen cfgs; 688 lab tests green). Derive-at-consume — `scenes_metadata.json` schema preserved, so teleop + the bridge harness-smoke + `STRAFER_CFG` physics are untouched. A degenerate occupancy grid fails loud (never masked, no union fallback).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/127
**Remaining:** the live "robot settles inside the single-room+couch scene" operator Kit smoke (goal-(a) Prereq 4) is satisfied jointly with Brief B [`single-room-couch-scene-supply`](../../active/harness/single-room-couch-scene-supply.md), which supplies that scene — an operator-run gate, not unit-testable. Everything else closes against the existing corpus.


**Type:** bug + refactor (spawn sourcing)
**Owner:** DGX agent — every touched surface (`strafer_lab/` env cfgs, `scene_connectivity` tools, `run_sim_in_the_loop.py`, the sim-in-the-loop bridge) is DGX-lane.
**Priority:** P1 — unblocks the **spawn/scene precondition** of goal-(a) ([`strafer-nocam-subgoal-singleroom-sim-validation`](../../active/trained-policy/strafer-nocam-subgoal-singleroom-sim-validation.md) Prereq 4: any spawn must land in the same room as the target). Today the bridge spawns the robot at a foreign scene's coordinates, so no single-room mission can validate. Not P0 (the hybrid runtime is plumbing-correct and unit-tested; this is the spawn data path), but the goal-(a) spawn precondition cannot be met without it.
**Estimate:** M — the occupancy→spawn mechanism already exists and is proven on the coverage path; the work is promoting it to a shared util, re-pointing the bridge/training cfg at it per-loaded-scene, and adding the runtime `--scene-usd` re-derivation hook.
**Branch:** task/bridge-spawn-from-occupancy

## Story

As a **mission operator validating the trained NOCAM_SUBGOAL policy in the sim bridge**, I want **the robot to spawn from the loaded scene's own occupancy free-space — the same way coverage-capture already derives its start pose**, so that **the robot reliably appears inside the loaded single room (in the same room as the target) instead of at a different scene's coordinates, and `--scene-usd` / scene selection yields a coherent spawn automatically.**

## The problem

The bridge/training env bakes the robot spawn at config time by **unioning** `spawn_points_xy` across **every** scene in `scenes_metadata.json`, with no filter for the scene actually loaded. `_get_infinigen_spawn_points_xy()` (`source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py:1155-1164`):

```python
for scene_data in meta["scenes"].values():       # <-- EVERY scene
    spawn_points_xy.extend(scene_data.get("spawn_points_xy", []))
```

`_apply_infinigen_scene_setup(cfg)` (`strafer_env_cfg.py:1211-1237`) then writes that union into **both** spawn consumers and pins the robot's spawn-z from a cross-scene **max**:

```python
spawn_points_xy = _get_infinigen_spawn_points_xy()          # union
if spawn_points_xy:
    cfg.events.reset_robot.params["spawn_points_xy"] = spawn_points_xy   # robot reset
    cfg.commands.goal_command.spawn_points_xy = spawn_points_xy          # goal sampling
floor_top_z = _get_infinigen_floor_top_z()                  # pooled MAX z across scenes
if floor_top_z is not None:
    cfg.events.reset_robot.params["spawn_z"] = floor_top_z + 0.1         # <-- pooled MAX
```

This runs from `composed_env_cfg.py:481-482` for any `scene_source.kind == "infinigen"` cfg, which includes the goal-(a) bridge env `StraferNavCfg_BridgeAutonomy` (`composed_env_cfg.py:607`). The corpus scenes occupy mutually disjoint world ranges (e.g. seed1 ≈ (7, 9), seed5 ≈ (13, −0.9)), so sampling the union places the robot — and goals — at a non-loaded scene's coordinates, and the pooled-max `spawn_z` (seed1 floor 0.148 vs seed5 0.614) is wrong for whichever single scene is loaded.

**Which leak breaks goal-(a), and which is latent.** The load-bearing defect for goal-(a) is the **reset spawn** (the robot's start position): in `--mode bridge` the robot is driven by external `/cmd_vel` from the Jetson autonomy stack and goal-(a)'s goal comes from `strafer-autonomy-cli submit`, *not* the gym `goal_command` term. So a unioned reset spawn is what places the robot outside the loaded single room. The unioned `goal_command.spawn_points_xy` is a **latent correctness defect** for goal-objective training/harness modes (it places sampled goals in foreign-scene coords) but is *not* exercised by the bridge mission's external-cmd_vel driver. `StraferNavCfg_BridgeAutonomy` does inherit `objective.kind="goal"` (it does not set `kind="subgoal"`), so its `goal_command` pool *is* poisoned and is fixed here — but the framing matters: the reset-spawn fix is what unblocks goal-(a); the goal-command fix removes a latent foot-gun.

The runtime override does not save it: `run_sim_in_the_loop.py:555-557` swaps **geometry only** —

```python
if args.scene_usd is not None:
    env_cfg.scene.scene_geometry.spawn.usd_path = str(args.scene_usd.resolve())
```

— and never touches the already-baked `reset_robot.params["spawn_points_xy"]` / `goal_command.spawn_points_xy` / `spawn_z`. So pointing the bridge at scene X still spawns from the X+Y+Z union (blocker 1, confirmed end-to-end).

Separately, the spawn points themselves are not occupancy-aware: `generate_scenes_metadata._sample_floor_points` (`source/strafer_lab/scripts/generate_scenes_metadata.py:218-270`) draws area-weighted uniform samples from the **union of floor bounding boxes**, rejecting only points inside object AABBs — no config-space disc inflation, no doorway/collider awareness (blocker 3). A point can land in geometry the AABBs miss.

There is no second implementation problem to invent a solution for: the coverage driver already derives a correct spawn from the per-scene occupancy sidecar ([`coverage-spawn-from-occupancy`](../../completed/harness/coverage-spawn-from-occupancy.md), PR #113; the multiscene per-scene USD/floor binding from [`coverage-capture-multiscene-correctness`](../../completed/harness/coverage-capture-multiscene-correctness.md), PR #116). The bridge has the **same class of bugs the coverage path already fixed** but never received the fix. This brief converges the bridge/training spawn onto that shipped mechanism.

## Context bundle

Read these before starting:

- [`branching-and-prs`](../../context/branching-and-prs.md) — branch/PR flow; ship-in-PR-before-merge (universally relevant).
- [`conventions.md`](../../context/conventions.md) — no transient doc refs in code, no commit trailers, no workspace env names in source, user-facing-doc maintenance sweep.
- [`ownership-boundaries.md`](../../context/ownership-boundaries.md) — DGX owns `source/strafer_lab/`; `strafer_shared/` is append-only cross-host constants only (NOT a home for numpy spawn logic).
- [`path-planning-architecture`](../../context/path-planning-architecture.md) — `scene_connectivity.py` is the one shared occupancy/free-space seam; the new spawn util belongs in this neighborhood.
- [`coverage-spawn-from-occupancy`](../../completed/harness/coverage-spawn-from-occupancy.md) — **the exact mechanism this converges onto.** Its Out-of-scope ("The training env's spawn sampling — unchanged; this is capture-only.") is the exclusion **this brief lifts** for the bridge/training side. Reuse `_derive_spawn_xy`; do not reimplement.
- [`coverage-capture-multiscene-correctness`](../../completed/harness/coverage-capture-multiscene-correctness.md) — **the closest precedent.** It fixed the identical per-scene class of bug on the coverage side (unconditional per-scene USD bind, per-scene floor via `_get_infinigen_active_scene_floor_top_z`, `point_in_any_room` + `seal_free_space_to_rooms` containment, and two fail-loud gates). Mirror its fixes on the bridge path.
- [`occupancy-interior-fidelity`](../../completed/harness/occupancy-interior-fidelity.md) — why the corpus occupancy sidecars are now trustworthy spawn sources (all 5 grids healthy after the bake-seam fix).
- [`bridge-scene-memory-budget-gb10`](../bridge-scene-memory-budget-gb10.md) — **owns the deterministic scene-selection knob + GB10 memory budget** (it proposes a `--scene`/lightest-scene picker over `_get_scene_usd_paths()[0]` and a GB10 scene budget). This brief only makes the config-default USD↔spawn *self-consistent*; it does NOT add the scene-picker. Coordinate so the two don't conflict on `_get_scene_usd_paths()[0]` (see Out of scope).
- [`scene-provider-floor-sampler-cli`](../../parked/harness/scene-provider-floor-sampler-cli.md) — the spawn-point **generation** tooling; keep **disjoint** (see Out of scope).
- [`single-room-couch-scene-supply`](../../active/harness/single-room-couch-scene-supply.md) — sibling Brief B; supplies the single-room+couch scene + occupancy that this brief's converged spawn ultimately runs on for goal-(a), and the `make sim-bridge` SCENE_USD passthrough. This brief is independently shippable against the existing corpus (see Sequencing); only the goal-(a) end-to-end bullet waits on B.
- Goal-(a) brief: [`strafer-nocam-subgoal-singleroom-sim-validation`](../../active/trained-policy/strafer-nocam-subgoal-singleroom-sim-validation.md) — the validation whose **spawn/scene precondition (Prereq 4)** this unblocks. Note Prereq 1 (Jetson backend-selection wiring) is a *separate* hard gate, not addressed here.
- `HARNESS_EPIC_PLAN.md` — the harness epic already converged on occupancy-derived spawn for the bulk (coverage) route; this is in-grain synergy, not a reopening of any epic-closing item.

## The mechanism to converge on (already built)

The per-scene occupancy sidecar (`occupancy.npy` + `occupancy.json`, `origin_xy` / `resolution_m`, in the scene-authored world frame) is produced once per scene at scene-gen by `validate_scene_connectivity.py` and consumed via the numpy-only, Kit-free seam in `source/strafer_lab/strafer_lab/tools/scene_connectivity.py`:

- `load_occupancy(scene_dir) -> CachedOccupancy` (`.grid`, `.origin_xy`, `.resolution_m`).
- `occupancy_to_free_space(grid, grid_res=, robot_radius_m=ROBOT_RADIUS_M)` — invert + robot-radius disc-inflate → passable grid (the same config-space inflation the GPU training grid uses; inscribed/half-width radius so doorways aren't sealed).
- `seal_free_space_to_rooms(free, rooms, origin_xy=, grid_res=)` — mask the porous exterior pad to room footprints.
- `_cell_to_xy` / `_xy_to_cell`, `point_in_any_room`, `room_representative_xy`.

The spawn-selection logic the bridge needs is `coverage_capture._derive_spawn_xy` (`source/strafer_lab/scripts/coverage_capture.py:838-941`): deterministic, picks a free + in-room + planner-reachable cell, raises rather than masking a degenerate grid. **Its current signature is `_derive_spawn_xy(free_space, plan, occupancy, *, rooms, plan_path, invalid_endpoint_errors)` and its ENTIRE preference order is plan-anchored** — preference 1 is "first plan viewpoint that plans," and even the fallback row-major scan requires reachability to the traversal's first free in-room viewpoint (`coverage_capture.py:863-869`). There is no plan-free path inside it today.

**Convergence design (commit to this API).** Factor the function into:
- a **plan-free shared core** `spawn_pool_from_occupancy(free_space, rooms, occupancy, *, n=...) -> list[[x, y]]` in `scene_connectivity.py` that returns free + in-room cells of the loaded scene (the generalization of `_derive_spawn_xy`'s row-major in-room scan, `coverage_capture.py:927-936`), and
- coverage **wraps** that core with its plan-anchored single-point preference (preference 1 + the leg-reachability probe) and `_validate_spawn_ready`, preserving its exact current behavior.

The bridge/training pool calls the plain core. **Be explicit about the guarantee gap:** the bridge has no coverage plan, so the bridge pool's guarantee is **"free on the robot-radius-inflated grid + inside a room"** — it does *not* carry coverage's per-leg planner-reachability guard (the bridge does not plan legs). This is weaker than coverage's guarantee and must be named as such; do not imply the bridge inherits coverage's reachability property.

**Frame:** clean and already proven. The Infinigen scene is a single global prim at `/World/Room` with envs co-located at `env_spacing=0`, so scene-authored frame == env-local frame; the occupancy `origin_xy` ties the grid to that same frame; `reset_robot_state_on_floor` (`mdp/events.py:143-148`) adds `env_origins` itself. A util emitting points in the occupancy frame hands them in as-is — no world-frame pre-add (coverage measured the authored-vs-settled delta at sub-mm).

## Scope

- **Fix the union bug.** Replace `_get_infinigen_spawn_points_xy()` (`strafer_env_cfg.py:1155-1164`) with a `(scene_stem)`-parameterized derivation that, for the **single loaded scene only**, loads that scene's occupancy sidecar and returns occupancy-derived free-space spawn points (mirror the existing per-scene `_get_infinigen_active_scene_floor_top_z`, `:1180-1192`). No cross-scene union.
- **Converge on one shared core.** Promote `_derive_spawn_xy` (+ `_scene_dir_for`) out of `coverage_capture.py` into `source/strafer_lab/strafer_lab/tools/scene_connectivity.py` (or a sibling DGX-lane `tools/` module importing from it), refactored into the plan-free `spawn_pool_from_occupancy` core + coverage's plan-anchored wrapper as described above. Coverage's selection is behaviorally unchanged (same cell chosen per scene). Spawn points are occupancy-derived (free, eroded by robot radius via `occupancy_to_free_space`, sealed to rooms), not floor-bbox samples.
- **Fix only the `spawn_z` pooled-max (not `lift_ground`).** In `_apply_infinigen_scene_setup` (`strafer_env_cfg.py:1229-1231`), drop the pooled-max `_get_infinigen_floor_top_z()` and pin `spawn_z` from the **loaded** scene's `floor_top_z` (the per-scene helper `_get_infinigen_active_scene_floor_top_z` already exists and is already used for `lift_ground.target_z` at `:1233-1237` — `lift_ground` is already per-scene and correct; do not "fix" it).
- **Make the config-default USD↔spawn self-consistent.** The config-default geometry bind uses `_get_scene_usd_paths()[0]` (`strafer_env_cfg.py:1218`); ensure the per-scene spawn derivation references the **same** scene stem (`scene_link.stem`, already in scope) so the default never binds scene[0]'s geometry to a different scene's spawn. (Replacing `[0]` with a deterministic/lightest-scene *picker* is owned by `bridge-scene-memory-budget-gb10`, not here — see Out of scope.)
- **Mandatory runtime hook (`--scene-usd`).** In `run_sim_in_the_loop.py:555-557`, after swapping `scene_geometry.spawn.usd_path`, **re-derive and re-write** `env_cfg.events.reset_robot.params["spawn_points_xy"]`, `goal_command.spawn_points_xy`, `spawn_z`, and `lift_ground.target_z` for the overridden scene (port the post-swap re-derivation block at `coverage_capture.py:454-496`). Without this, the config-time fix is invisible whenever `--scene-usd` is passed — which is the goal-(a) path.
- **Decide and state the derivation timing.** Prefer **derive-at-consume** (matches coverage; the bridge/coverage compute spawn from `occupancy.npy` at config/runtime, and `scenes_metadata.json` keeps only labels / `floor_top_z` / `floor_bbox_xy` for the spawn role). Persisting occupancy-derived spawn into the JSON is the alternative; if chosen, fix `generate_scenes_metadata.py` to source from occupancy and to read-merge-write per-scene (`:422`, `:441` rewrite the whole file today → clobber). Generator-side floor-sampler rework otherwise stays in `scene-provider-floor-sampler-cli` (see Out of scope).

## No-regression contract (every consumer recon found)

- **Reset event — `mdp/events.py:reset_robot_state_on_floor` (:87-159).** Invariant: `spawn_points_xy` is `[x, y]` in the **env-local frame**; the function adds `env_origins` itself (`:143-148`). The converged util must emit env-local (== occupancy/scene-authored) XY — no world-frame pre-add. **Keep the empty-list fallback** (`:122-123`, spawn at env origin) for fresh/pre-metadata scenes and smoke tests. Pass a **fresh list** (the event caches the points tensor by `id()`).
- **Goal command — `mdp/commands.py` (:50-52, :110-113, field :394).** Invariant: it samples goals from `cfg.spawn_points_xy`; a per-loaded-scene pool keeps goals in the loaded scene. The subgoal variant sets `spawn_points_xy = None` (`commands.py:431`) and must stay unaffected (rolling-subgoal path does not use the pool).
- **Teleop — `teleop_capture.py:_resolve_active_spawn_points` (:281-293, applied :806-812).** Already per-active-scene (reads only the active scene's block), so it is NOT the union bug. Invariant: teleop must keep spawning in the **active** scene. If derive-at-consume drops `scenes_metadata.json` spawn points, port teleop onto the same occupancy derivation (preferred — "same way for BOTH"); if the JSON spawn schema is preserved, teleop is untouched. Either way teleop must not regress to the union, and its scene-USD override (`:798-800`) stays. **Also update the stale-pool warning** (`teleop_capture.py:1131-1134`, "scenes_metadata.json is stale (regenerate …)") and its `spawn_bbox` source (`:808-816`) if the spawn source moves to occupancy — else teleop logs a misleading "regenerate scenes_metadata.json" message that no longer applies.
- **Coverage-capture — `coverage_capture.py` (`_derive_spawn_xy` :838-941, usage :454-496).** This is the reference impl being promoted. Invariant: coverage spawn behavior is **behaviorally unchanged** — same occupancy source, same deterministic single-point pick (same cell per scene), same fail-loud gates (`_validate_spawn_ready` :944-1008, scene-identity hash :1011-1072), same per-scene USD/floor bind from #116. Promoting the function to a shared module is a refactor for coverage (call shape may change; chosen cell does not); `test_coverage_capture.py` stays green.
- **Bridge env — `StraferNavCfg_BridgeAutonomy` (`composed_env_cfg.py:607`) via `_apply_infinigen_scene_setup`.** The fix target; ships the corrected per-loaded-scene spawn at both config-default and the `--scene-usd` runtime hook.
- **Other Infinigen-kind training cfgs — `StraferNavCfg_TeleopCapture` (:596), `StraferNavCfg_Coverage` (:620).** Inherit `_apply_infinigen_scene_setup`; must continue to receive valid **in-room** spawns for the loaded scene (they override per-scene downstream today; the config-default must not regress them).
- **Bridge smoke — `bridge_harness_smoke.py:170-175`** reads a per-scene `spawn_points_xy` block and sets the reset param. If the JSON spawn schema changes, update this reader in lockstep (it must keep spawning in its scene).
- **Procroom / plane sources — `composed_env_cfg.py:483-484` and the plane path.** Never enter `_apply_infinigen_scene_setup`; untouched.
- **Teleop STRAFER_CFG physics — `assets/strafer.py` (`STRAFER_CFG`) + `_apply_default_nav_runtime`'s `PhysxCfg` (`strafer_env_cfg.py:1147`).** Has zero spawn coupling. **DO NOT TOUCH.** No solver-iter / stabilization / restitution / damping change is in this brief's scope.

## Sequencing (A gates B gates goal-(a) — and where this brief sits)

- **This brief is independently shippable and testable against the EXISTING corpus.** All of its assertions (per-loaded-scene spawn, occupancy-derived, one shared core, `--scene-usd` re-derivation, per-scene `spawn_z`) run Kit-free against the shipped `scene_high_quality_dgx_*` (seed1/2/5/6/7) occupancy sidecars, which already exist and are healthy. It does **not** depend on Brief B to land or to be tested.
- **Goal-(a) end-to-end is gated A → (needs B's scene) → goal-(a).** The only acceptance bullet here that waits on Brief B is the live "robot spawns inside the single-room+couch scene" smoke; everything else closes against the existing corpus. Brief B in turn supplies the scene + occupancy that this brief's converged spawn consumes.
- **Goal-(a) needs MORE than A+B.** This brief unblocks the **spawn/scene precondition** (goal-(a) Prereq 4). Goal-(a) also has an independent hard gate — Prereq 1, the **Jetson backend-selection wiring** (`STRAFER_NAV_BACKEND → default_navigation_backend`, drop the `plan_compiler` `nav2` hardcode, expand `_RECOGNISED_BACKENDS`) — which is out of scope here (Jetson lane) and is not satisfied by A+B.

## Acceptance criteria

- [ ] **Per-loaded-scene spawn (no union).** A test asserts that loading scene X (from the existing corpus) yields `reset_robot.params["spawn_points_xy"]` (and `goal_command.spawn_points_xy`) drawn **only** from scene X — every point lies within scene X's room footprints, and none falls in another corpus scene's coordinate range. Constructing the Infinigen-kind env cfg for scene X never reads any other scene's spawn data.
- [ ] **Occupancy-derived, not floor-bbox.** Every emitted spawn point is a free cell of scene X's `occupancy.npy` after robot-radius disc inflation (`occupancy_to_free_space`) and room sealing — assertable without Kit against the on-disk sidecar.
- [ ] **One shared core.** `coverage_capture` and the bridge/training cfg call a **single** occupancy→spawn core (`spawn_pool_from_occupancy`) in `strafer_lab/tools/` (DGX-lane); no duplicated spawn-selection logic. Coverage's deterministic single-point, plan-anchored selection is behaviorally unchanged (same cell chosen per scene; `test_coverage_capture.py` green). The bridge pool's documented guarantee is "free on the inflated grid + in-room" (no per-leg reachability guard).
- [ ] **`--scene-usd` coherence.** `run_sim_in_the_loop.py --scene-usd <X>` re-derives spawn for X: a test (or instrumented dry-run of the override block) shows `reset_robot.params["spawn_points_xy"]`, `goal_command.spawn_points_xy`, `spawn_z`, and `lift_ground.target_z` all reflect scene X after the override, not the config-default union/pooled-max.
- [ ] **Per-loaded-scene `spawn_z`.** `spawn_z` is pinned from the loaded scene's `floor_top_z`, not the pooled max. (`lift_ground.target_z` is already per-scene; confirm it is not regressed.)
- [ ] **Degenerate grid is not masked.** An over-occupied/degenerate occupancy grid surfaces as a fail-loud error (no planner-snap widening, no fallback to the external/union read), preserving the coverage contract.
- [ ] **No regression** in the workflows the touched code supports — call out the smoke tests: the harness suite (`test_coverage_capture.py` and the `generate_scenes_metadata` test if the JSON schema is touched) stays green; the env_cfg import/construct smoke (Kit-free, LD_PRELOAD libgomp) constructs `StraferNavCfg_BridgeAutonomy` / `StraferNavCfg_TeleopCapture` / `StraferNavCfg_Coverage` with valid in-room spawns; teleop and coverage-capture spawn behavior is unchanged-or-improved (same occupancy source). Teleop STRAFER_CFG physics is untouched.
- [ ] **(Operator-Kit-smoke, satisfied jointly with Brief B — not unit-testable.)** On the single-room+couch scene (from Brief B), the bridge robot settles inside that room's floor (in the same room as the couch target), within the empirically-confirmed sub-mm authored-vs-settled delta — the spawn precondition goal-(a) Prereq 4 requires. This bullet is an operator Kit smoke (it needs a live sim), consistent with how `coverage-spawn-from-occupancy` separated its sub-mm smoke from its numpy assertions.

### Maintenance clause

- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/` (e.g. `docs/example_commands_cheatsheet.md`'s bridge runbook, `source/strafer_lab/README.md`, the path-planning-architecture seam note), update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance) for the surface list and trigger heuristics.

## Investigation pointers

- Union bug + caller: `source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py:1155-1164` (`_get_infinigen_spawn_points_xy`), `:1211-1237` (`_apply_infinigen_scene_setup`), `:1218` (`_get_scene_usd_paths()[0]` config-default bind), `:1229-1231` (pooled-max `spawn_z` via `_get_infinigen_floor_top_z`), `:1233-1237` (per-scene `lift_ground.target_z` — already correct), `:1180-1192` (per-scene floor helper to mirror).
- Bridge cfg + dispatch: `composed_env_cfg.py:481-482` (gate), `:607` (`StraferNavCfg_BridgeAutonomy`), `:596` / `:620` (the other Infinigen-kind cfgs).
- Runtime override (second hook): `source/strafer_lab/scripts/run_sim_in_the_loop.py:555-557`.
- Mechanism to promote: `source/strafer_lab/scripts/coverage_capture.py:838-941` (`_derive_spawn_xy`, signature + plan-anchored preference order at `:863-869`, in-room scan core at `:927-936`), `:944-1008` (`_validate_spawn_ready`), `:1011-1072` (scene-identity gate), `:454-496` (post-swap re-derivation template), `:270-276` (`_scene_dir_for`).
- Shared seam (core home): `source/strafer_lab/strafer_lab/tools/scene_connectivity.py` (`load_occupancy`, `occupancy_to_free_space`, `seal_free_space_to_rooms`, `point_in_any_room`, `room_representative_xy`, `_cell_to_xy`/`_xy_to_cell`).
- Frame contract: `source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py:87-159` (env-local; `env_origins` added at `:143-148`); scene-as-global-prim at `strafer_env_cfg.py:283-287`.
- Goal command: `source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py:50-52, 110-113, 394, 431`.
- No-regress readers: `teleop_capture.py:281-293, 798-816, 1131-1134`; `bridge_harness_smoke.py:170-175`.
- Floor-bbox generator (blocker 3): `source/strafer_lab/scripts/generate_scenes_metadata.py:218-270` (`_sample_floor_points`), `:422`/`:441` (full-file clobber).

## Out of scope

- **The single-room+couch scene supply, the `fast_singleroom` generator fix, and the `make sim-bridge` SCENE_USD passthrough** — sibling Brief B [`single-room-couch-scene-supply`](../../active/harness/single-room-couch-scene-supply.md). This brief is shippable against the existing corpus and only depends on B for the goal-(a) end-to-end Kit smoke.
- **The deterministic / lightest-scene config-default scene PICKER and the GB10 scene memory budget** — owned by [`bridge-scene-memory-budget-gb10`](../bridge-scene-memory-budget-gb10.md). This brief only makes the existing `_get_scene_usd_paths()[0]` config-default USD↔spawn self-consistent; it does NOT replace `[0]` with a picker. Coordinate so the two don't collide on that line.
- **The Jetson-side backend-selection wiring** (`STRAFER_NAV_BACKEND → default_navigation_backend`, plan-compiler `nav2` hardcode, `_RECOGNISED_BACKENDS`) — goal-(a) Prereq 1, a separate Jetson-lane gate; not a spawn/scene concern and not DGX-lane.
- **Generator-side floor-sampler rework / embedding spawn points into USD customData** — `scene-provider-floor-sampler-cli` (parked). This brief is spawn **consumption/derivation**; it only touches `generate_scenes_metadata.py` if the chosen design persists occupancy spawn into the JSON (the read-merge-write clobber fix), otherwise the generator is untouched.
- **Multi-env parallel spawn variety beyond per-reset sampling from the per-scene pool** — v1 derives a per-loaded-scene pool; richer parallel-env spawn distribution is a follow-up.
- **Teleop STRAFER_CFG / physics tuning** — untouchable by construction (no spawn coupling).
- **Re-tuning the policy or fixing degenerate occupancy grids** — degenerate grids surface fail-loud and are owned by `occupancy-interior-fidelity` / a regenerate-occupancy follow-up, not masked here.
