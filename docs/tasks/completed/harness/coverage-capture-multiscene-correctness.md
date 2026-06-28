# Coverage capture multi-scene correctness — load the requested scene, spawn in-room, gate per scene

**Status:** Shipped 2026-06-27 in `464728a` (DGX). The coverage driver now binds the spawned geometry to the requested `--scene` (was: every scene loaded the first pooled scene's USD, so the scene-frame spawn + plan landed in another floorplan), pins `spawn_z` / ground `target_z` to that scene's own floor, seals the planner free-space to the room footprints, and requires the spawn + room representatives to lie inside a room footprint. Two fail-loud gates were added: a pre-build grid gate (spawn in-room + first leg plannable) and a pre-traversal runtime gate (the loaded `/World/Room` geometry's embedded metadata hash must equal the requested scene's). CPU corroboration on all 5 corpus scenes + full pure-Python harness suite green; the `--video` Kit smokes are operator-run (PR test plan).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/116

**Type:** bugfix (capture scene binding) + hardening (containment + per-scene gate)
**Owner:** DGX agent (coverage capture driver)
**Priority:** P0 — coverage capture produced correct collection on only 1 of 5 corpus scenes; the corpus is unusable until every scene captures in its own geometry.
**Branch:** task/coverage-capture-multiscene-correctness

## Story

As **the coverage capture driver**, I want **the simulator to load the same scene the occupancy grid + plan + spawn were computed for, and to refuse to capture a scene it cannot spawn/traverse correctly**, so that **collection is correct on every corpus scene (not just seed1) and a genuinely-bad scene fails loud instead of silently recording garbage.**

## The problem

Two independent defects; the first is the root cause, the second is a latent robustness hole.

1. **Wrong-scene geometry (root cause).** The driver resolves the per-scene USD (`scene_usd_path = resolve_scene_usd_path(scene=args.scene, ...)`) and uses it for metadata / occupancy / plan / spawn, but bound it to the sim geometry only inside `if args.scene_usd is not None:`. With `--scene-usd` unset (the normal path — `capture.py` never forwards it), the geometry was set by the env cfg's `__post_init__` → `_apply_infinigen_scene_setup`, which hardcodes `_get_scene_usd_paths()[0]` (`sorted()[0]` = seed1) for **any** `--scene`. So for every scene except seed1, a seed-N free-cell spawn + seed-N plan were applied inside **seed1's** floorplan → spawn outside rooms / in walls, drive into walls, fail to traverse. seed1 worked only because its frame *is* the loaded geometry.

   CPU cross-frame corroboration (seed-N interior points evaluated against seed1's grid — what the sim loaded): seed1 100% in-room (control); seed2 49%; seed5 73%; seed6 41% (its derived spawn lands on a **blocked** seed1 cell); seed7 80% (its spawn coincidentally lands in-room but its plan targets do not → nose-to-wall). Confirmed live: the real Coverage env cfg loads seed1's `export_scene.usdc` with `spawn_z=0.7136` (pooled max) and `target_z=0.1464` (seed1 floor) for every scene.

2. **No room containment (latent).** The spawn path gated only on free + planner-reachable. The cached occupancy grid keeps a free ~1.5 m exterior pad and the thin-z-slab rasterizer leaves perimeter walls porous, so a flood-fill from the exterior corner `(0,0)` reaches the grid center on all 5 scenes (interior+exterior are one connected free component), and the row-major `np.argwhere` fallback returns cell `(0,0)` — an outside-the-house spawn — on all 5. Today the primary path (plan viewpoints) wins so the fallback is never hit, but nothing prevented an outside-the-house spawn if it had been.

Also latent and surfaced by fix #1: `_apply_infinigen_scene_setup` sets `spawn_z` to the pooled **max** floor height across scenes (seed5's 0.614 outlier → ~0.71 m for every scene) and the ground-lift `target_z` from the first pooled scene's floor — both wrong per scene once the right geometry loads.

## Scope (shipped)

- **Bind geometry to the requested scene.** Unconditional `env_cfg.scene.scene_geometry.spawn.usd_path = str(scene_usd_path.resolve())` after the last `__post_init__` (subsumes the `--scene-usd` override, which is already folded into `scene_usd_path`).
- **Per-scene floor.** Override `events.reset_robot.params['spawn_z']` (= floor + 0.1) and `events.lift_ground.params['target_z']` (= floor − 0.002) from `_get_infinigen_active_scene_floor_top_z(args.scene)`. Capture-only; the training env's single default is unchanged.
- **Containment + free-space seal.** New `scene_connectivity.point_in_any_room` + `seal_free_space_to_rooms` (blocks free cells outside the room-footprint AABB — drops the exterior pad without touching interior doorways; verified zero change to leg routing on all 5 scenes). `_derive_spawn_xy` now requires the spawn to be in a room on both primary + fallback paths; `room_representative_xy`'s ring search now requires footprint containment.
- **Two fail-loud gates.** `_validate_spawn_ready` (pre-build, grid frame: spawn in-room + first leg in-room and plannable, else `SystemExit`); `_assert_loaded_scene_identity` (pre-traversal, runtime: the live `/World/Room` geometry's embedded metadata hash must equal the requested scene's, via `scene_metadata_reader.metadata_from_prim`; `customData` composition onto the referencing prim verified with pxr; falls back to a cfg `usd_path` equality check with a printed reason only if the prim exposes no metadata).
- **Docstring fix.** `coverage_plan.VisitWaypoint.target_xy` is scene-local (env-local), not world-frame.

## Acceptance

- [x] The sim loads the requested `--scene`'s geometry (unconditional bind); the runtime scene-identity gate proves it per run and fails loud on mismatch.
- [x] The derived spawn and the room representatives lie inside a room footprint; the exterior-corner fallback is unreachable after the seal (`free[0,0]` False on all 5).
- [x] `spawn_z` / ground `target_z` are the requested scene's floor, not the pooled max / first scene.
- [x] A genuinely-bad scene (no free+in-room+plannable spawn, or an unhealthy grid) fails with a clear non-zero exit; a good scene passes and captures.
- [x] Pure-Python tests cover containment, the seal, both gates, and `metadata_from_prim` hash parity; full `strafer_lab` harness suite green (645 passed, 1 skipped; +18 new tests, 627 → 645).
- [ ] **Operator `--video` Kit smoke on all 5 scenes**: each good scene spawns inside a room and traverses sensibly; seed1 stays correct (no regression). Commands in the PR test plan.

## Context bundle

- [`coverage-capture-driver`](coverage-capture-driver.md) — the driver this fixes (shipped #111, reworked #112).
- [`coverage-spawn-from-occupancy`](coverage-spawn-from-occupancy.md) — moved the spawn to in-driver occupancy free-space (#113); this brief adds the room-containment + scene-binding correctness that derivation assumed.
- [`occupancy-interior-fidelity`](occupancy-interior-fidelity.md) — fixed the over-occupancy class (#114). This defect is **independent**: all 5 grids are blocked-fraction-healthy and still failed before this fix.

## Out of scope

- Regenerating any genuinely-degenerate occupancy grid (a generation-side concern; the gate surfaces it, it does not repair it).
- Multi-env capture (v1 is single-env).
- Changing the connectivity-graph generator or the training env spawn/floor behavior.
