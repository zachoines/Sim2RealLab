# Generate goal-(a) scenes: a true single-room + a two-room scene (any groundable target)

**Status:** Shipped 2026-06-30 in `27e7da8` (DGX) — a parametric scene generator along two dimensions: **layout** (`--rooms <types>` tiles an EXACT floor plan via the `_build_floor_plan` tiler + Infinigen's `PredefinedFloorPlanSolver`; duplicates allowed — e.g. `--rooms bedroom bedroom` for adversarial same-class; omit for an organic constraint-solved house) and **quality/resources** (`--quality {high,low}` bundling `--texture-res` + `--object-density`, plus the `--geometry-detail` displacement opt-in — available memory is the driver). The named-scene-preset surface (incl. the misnamed `fast_singleroom`/`windows_baseline`) was retired for flags; also non-clobbering `generate_scenes_metadata.py --merge`, `SCENE_USD`/`SCENE_NAME` passthrough on `make sim-bridge`, and the goal-(a) mission decoupled from "couch". Tiler geometry verified end-to-end (real `solidify()` + shapely, 1–5 rooms incl. duplicates). **Kit generation gate SATISFIED 2026-06-30:** generated `scene_singleroom_000_seed0` (customData `rooms[]` == 1 `living_room`, 46 objects) + `scene_tworoom_000_seed0` (`rooms[]` == 2 `living_room`+`kitchen`, connected — 1 reachable cross-room edge, 72 objects) via the flags above; registered both with `generate_scenes_metadata.py --merge` (the 5 heavy corpus entries preserved); removed the misnamed `scene_fast_singleroom_000_seed0` asset; recorded the goal-(a) target = **`sofa`** (uniquely present in the single room) into `strafer-nocam-subgoal-singleroom-sim-validation.md`. Remaining before this brief fully closes: the operator's live bridge smoke (spawn-in-room + mission grounds+reaches the sofa), joint with Brief A (#127).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/128

**Type:** task
**Owner:** DGX agent
**Priority:** P1 — gates the **spawn/scene precondition** of goal-(a) (goal-(a) Prereq 4: a genuine single-room scene so any spawn is co-room with the target, plus at least one groundable object to drive to); the only "single-room" scene today is misnamed (~5-room) and the heavy corpus scenes OOM the GB10. Also delivers a two-room scene for the eventual cross-room lane and removes the misleading `fast_singleroom` asset.
**Estimate:** M — Kit/Blender-bound generation + a gin authoring round; ~a day plus operator generation passes.
**Branch:** task/single-room-couch-scene-supply

## Story

As **the DGX agent validating the trained subgoal policy in the sim bridge**, I want **a genuine single-room Infinigen scene (and a two-room scene) — each containing at least one groundable object — registered with metadata + occupancy + customData and selectable from `make sim-bridge`**, so that **goal-(a)'s mission has a ground-truth target the robot can spawn inside and reach without OOMing the GB10 or fighting a 5-room floorplan.** Infinigen cannot be made to guarantee a *specific* object like a couch, so the mission targets whatever unique, groundable object the scene actually contains (see Scope 2).

## Context bundle

- [`branching-and-prs.md`](../../context/branching-and-prs.md) — ship-in-PR-before-merge, stamp uses the work commit.
- [`conventions.md`](../../context/conventions.md) — no transient doc refs / commit trailers / env names in source; user-facing-doc maintenance sweep.
- [`ownership-boundaries.md`](../../context/ownership-boundaries.md) — DGX owns `source/strafer_lab/`, `docs/`, and the repo-root `Makefile` DGX targets; do not broaden scope.
- [`repo-topology.md`](../../context/repo-topology.md) — where env names + generation tooling live; the DGX is the GB10 (unified memory).
- [`infinigen-scene-corpus.md`](../../completed/harness/infinigen-scene-corpus.md) — the corpus-supply precedent this extends: the `prep_room_usds.py generate` runbook (prep → embed → connectivity → index), GUARD 1/GUARD 2 index gates, the GB10 "one Kit USD-load at a time" memory rule, and the record that `scene_fast_singleroom_000_seed0` is `customData ABSENT` / not corpus-counted.
- [`occupancy-interior-fidelity.md`](../../completed/harness/occupancy-interior-fidelity.md) — the Infinigen bake-seam fix that made the corpus occupancy grids trustworthy; the new scene's occupancy must clear the same free/floor-bbox health bar.
- [`coverage-spawn-from-occupancy.md`](../../completed/harness/coverage-spawn-from-occupancy.md) — the occupancy free-space → spawn derivation; cross-referenced here only to show the new scene's occupancy sidecar is the spawn source for BOTH lanes.
- [`bridge-spawn-from-occupancy.md`](../trained-policy/bridge-spawn-from-occupancy.md) — **Brief A**, the paired ticket: it makes the bridge derive robot spawn from the occupancy sidecar of the LOADED scene and decides whether `spawn_points_xy` persist in `scenes_metadata.json`. This brief supplies the scene + occupancy sidecar that Brief A consumes; the goal-(a) end-to-end run needs both.
- [`bridge-scene-memory-budget-gb10.md`](../bridge-scene-memory-budget-gb10.md) — the GB10 unified-memory OOM brief; its light-scene workaround gestures at exactly the single-room scene this brief delivers. The deterministic scene-selection knob is owned there, not here.

## The problem

Goal-(a) (drive the trained NOCAM_SUBGOAL policy end-to-end in the Isaac Sim bridge) has no scene to run in:

1. **The "singleroom" scene is misnamed and has no couch.** `scene_fast_singleroom_000_seed0` is a ~5-room apartment with bathtub/toilet/sink — no couch — and ships with no embedded customData / occupancy. The `FAST_SINGLEROOM` preset (`prep_room_usds.py:147-154`, `gin_configs=("fast_solve","singleroom")`, `max_rooms=1`) emits `restrict_solving.solve_max_rooms=1` (`prep_room_usds.py:585`), but that override only limits **which rooms get furnished** — per its own docstring in `infinigen/.../generate_indoors_util.py:220` ("only place objects in at most this many rooms") it flows into `quantity_limits` (`:267-270`) and does NOT constrain the floorplan room count. The room-count floor lives in `infinigen_examples/constraints/home.py` `home_room_constraints` `node_constraint` (`:327-335`), which requires Entrance/LivingRoom/Kitchen/Bedroom/Bathroom each `>= 1` — a guaranteed ~5-room minimum even on the `has_fewer_rooms` branch. So "singleroom" never produced one room; it is a generator misnomer.

2. **A *specific* object (e.g. a couch) cannot be reliably guaranteed in Infinigen.** `SofaFactory` exists (`infinigen/assets/objects/seating/sofa.py:1429`) and is wired into living rooms (`home.py:1086,1100-1106`), but placement is `in_range(0, ...)` (`home.py:1100-1106`; the guaranteed-sofa term at `:1102` is commented out) and furniture selection is stochastic — forcing a *named* object is fragile. So goal-(a) must NOT depend on a couch specifically; it targets whatever unique, groundable object the generated scene actually contains (a furnished room yields several).

3. **The heavy corpus scenes are not an option.** `scene_high_quality_dgx_*` (seed1/2/5/6/7) are 6-11 GB (`HIGH_QUALITY_DGX`, `max_rooms=5`, `texture_resolution=1024`, `prep_room_usds.py:136-141`) — they OOM the GB10 unified memory at Kit USD-load time (the symptom `bridge-scene-memory-budget-gb10` documents) and are multi-room, so they don't fit the single-room goal-(a) target.

4. **`make sim-bridge` cannot select a scene.** `sim-bridge` (`Makefile:223-229`) and `sim-bridge-gui` (`:231-237`) pass only `--mode bridge` — no `SCENE_NAME`/`SCENE_USD`. Only `sim-harness` (`:239-253`) forwards `$${SCENE_NAME:+--scene-name ...} $${SCENE_USD:+--scene-usd ...}`. So even once a single-room+couch scene exists, the operator cannot pin the bridge to it from the shell.

Net: there is no genuine single-room scene, no two-room scene, and no way to point the bridge at a chosen scene — plus a misnamed `fast_singleroom` asset that misleads. This brief supplies the scenes, the passthrough, and the cleanup.

## Scope

1. **Fix room-count control so the floorplan count is actually constrained — deliver BOTH a 1-room and a 2-room scene.** Drive the floorplan to an exact room count rather than only limiting furnished rooms. Do this the repo-clean way — via gin (`prep_room_usds` only ever touches Infinigen through `-g`/`-p`; never edit vendored Infinigen source). Options, pick the one that produces a stable exact room count:
   - A new gin (e.g. `true_singleroom.gin`) that overrides `home_room_constraints` / the `node_constraint` minimums (`home.py:327-335`) so a single-room grammar survives, OR
   - a `PredefinedFloorPlanSolver` single-room contour (`infinigen .../room/predefined.py`), OR
   - `restrict_solving.restrict_parent_rooms` to confine solving to a single room type.
   - The single-room's surviving room should be a **furnishable room type** (e.g. LivingRoom) so it contains groundable objects. For the **two-room** scene, produce exactly two connected rooms (a furnishable room + one adjacent room, sharing a door) — the cross-room fixture for the eventual autonomy-stack cross-room lane.
   - Author NEW `prep_room_usds.py` presets (e.g. `TRUE_SINGLEROOM` + `TRUE_TWOROOM`) in `PRESETS` (`:167-169`) referencing the new gin(s); no launcher change needed (`-g`/`-p` plumbing at `:580-588` already supports it). The misnamed `FAST_SINGLEROOM`/`WINDOWS_BASELINE`/`singleroom` recipes are handled by the fast_singleroom removal item below (do not silently mutate them mid-brief; decide their fate there).

2. **Ensure ≥1 groundable target object (any object — do NOT force a specific one).** The furnished room will contain Infinigen furniture; after generation, inspect the embedded customData `objects[]` and pick a **uniquely-present, groundable** label (appears ×1 in the scene) as the goal-(a) mission target. **Record that label** and update the goal-(a) validation brief's mission string away from "go to the couch" to the chosen object (see the cross-ref in Out of scope). Do NOT try to force a `SofaFactory`/couch — selection is stochastic and a named-object guarantee is fragile; a normally-furnished room offers several unique groundable objects to choose from.

3. **Register the scene: customData + occupancy + index entry.** Run the existing 2-step + occupancy pipeline for the new scene so the bridge can load and spawn in it (the `infinigen-scene-corpus.md` runbook). The new scene needs three artifacts, in this order:
   - **(a) customData embed** (`extract_scene_metadata` / `embed_scene_metadata`) so `objects[]`/`rooms[]` are in the USD and the `generate_scenes_metadata.py` customData gate (`:291-326`, `has_capture_metadata`) admits the scene to the index.
   - **(b) occupancy sidecar** via `validate_scene_connectivity.py` (omap path; `--rasterize-fallback` if needed) → `occupancy.npy` + `occupancy.json` in the scene dir. **This sidecar is THE spawn source** — Brief A's converged bridge spawn derives from it. It must clear the corpus free/floor-bbox health bar from `occupancy-interior-fidelity.md`. The single-room USD is small (~hundreds of MB, not 11 GB) so it will NOT OOM the GB10 the way the heavy scenes do.
   - **(c) `scenes_metadata.json` entry** via `generate_scenes_metadata.py` — for scene **discovery + `floor_top_z` + room/object labels**. Use a per-scene / read-merge-write invocation so the existing `high_quality_dgx` corpus entries are NOT clobbered (the script globs `scene_*.usdc` and rewrites the whole file at `:422`/`:441`). **Whether `spawn_points_xy` persist in this JSON at all is Brief A's decision** (Brief A prefers derive-at-consume from the occupancy sidecar, in which case this entry carries only labels + `floor_top_z`, not spawn points). This brief does not require `generate_scenes_metadata.py` to produce spawn points; it only requires the discovery entry + the occupancy sidecar.

4. **`make sim-bridge` SCENE_USD/SCENE_NAME passthrough.** Add `$${SCENE_NAME:+--scene-name $$SCENE_NAME}` and `$${SCENE_USD:+--scene-usd $$SCENE_USD}` to `sim-bridge` (`Makefile:223-229`) and `sim-bridge-gui` (`:231-237`), mirroring `sim-harness` (`:249-250`). The CLI already supports both flags (`run_sim_in_the_loop.py:119-135`). Keep the no-scene default working (unset SCENE_USD/SCENE_NAME → current behavior unchanged).

5. **Remove the misleading `scene_fast_singleroom_000_seed0` and reconcile the misnomer.** The asset (`Assets/generated/scenes/scene_fast_singleroom_000_seed0`) is a ~5-room apartment named "singleroom" — misleading, `customData ABSENT`, not corpus-counted, and (per the operator) a transient generated artifact that gets wiped/regenerated. Delete it, and reconcile every reference so nothing regenerates a 5-room scene under a "singleroom" name:
   - **The `FAST_SINGLEROOM` recipe is the root of the misnomer** (`prep_room_usds.py:147-154`, `singleroom`/`max_rooms=1` that does not constrain rooms). Decide with the operator: **(a)** fix it to actually produce one room (fold it into the room-count fix from Scope 1, keeping its `fast_solve` speed) and keep the name honest, or **(b)** delete/rename it and repoint its consumers. Do the same for `WINDOWS_BASELINE` (`:158-165`) if it shares the non-constraining recipe. The earlier "leave them for CI parity" guidance is superseded by this cleanup — the point is that no artifact called "singleroom" is actually multi-room.
   - **Repoint the consumers** currently tied to the name: `docs/HelloRoom.md` + any CI smoke (via the preset comment `:144-145`), `test_build_mission_queue.py:89` (uses `scene_name="scene_fast_singleroom_000_seed0"` — repoint to a mock/name that survives, or the new `TRUE_SINGLEROOM` scene), and the doc mentions in `bridge-scene-memory-budget-gb10.md` / `validator-evaluation.md` (update in the same commit per the maintenance clause). `grep -rn "fast_singleroom" docs/ source/` after the change and confirm no stale/misleading reference remains.

6. **Register BOTH new scenes** (single-room + two-room) via the pipeline in Scope 3 (customData + occupancy + non-clobbering index entry), so each is loadable + spawn-able by Brief A's converged spawn. The two-room scene need not be goal-(a)-usable yet (cross-room is blocked by autonomy-stack); it is the fixture for that lane and a proof the room-count control does N=2.

## Investigation pointers

- Singleroom presets: `source/strafer_lab/scripts/prep_room_usds.py:147-154` (`FAST_SINGLEROOM`, comment `:144-145` ties it to `docs/HelloRoom.md`/CI), `:158-165` (`WINDOWS_BASELINE`, same recipe — leave alone), `:167-169` (`PRESETS`), `:580-588` (`-g`/`-p` command build), `:585` (`restrict_solving.solve_max_rooms`).
- Why the override doesn't constrain rooms: `infinigen/infinigen_examples/util/generate_indoors_util.py:198-290` (`restrict_solving`, `:220` docstring, `:267-270` `quantity_limits`); room-count floor `infinigen/infinigen_examples/constraints/home.py:327-335` (`node_constraint` minimums); `singleroom.gin` (`infinigen/infinigen_examples/configs_indoor/singleroom.gin`).
- Couch: `infinigen/infinigen/assets/objects/seating/sofa.py:1429` (`SofaFactory`); `home.py:1086,1100-1106` (sofa constraint, `:1102` disabled guaranteed-sofa term).
- Registration pipeline: `source/strafer_lab/scripts/generate_scenes_metadata.py:291-326` (customData gate), `:422`/`:441` (glob + full-file clobber to avoid), `:345-362` (per-scene entry); `source/strafer_lab/scripts/validate_scene_connectivity.py:446-496` (omap occupancy producer), `:624-630` (`occupancy.{npy,json}` written next to the scene USD).
- Bridge load + scene flags: `source/strafer_lab/scripts/run_sim_in_the_loop.py:119-135` (`--scene-name`/`--scene-usd`), `:555-557` (USD override).
- Makefile: `Makefile:223-229` (`sim-bridge`), `:231-237` (`sim-bridge-gui`), `:239-253` (`sim-harness` passthrough template, `:249-250`).

## Acceptance criteria

- [ ] A new preset/gin produces a scene whose floorplan has **exactly one room** (verified in customData `rooms[]`), and a second scene with **exactly two connected rooms** — NOT the ~5-room result the old `singleroom` recipe produces. The single-room's room is a furnishable type (contains objects).
- [ ] **Each** new scene contains **≥1 uniquely-present, groundable object** (any label — no forced couch), verified in the embedded customData `objects[]`. The chosen single-room target label is **recorded** and the goal-(a) validation brief's mission string is updated to it (away from "go to the couch").
- [ ] Both scenes are registered: customData embedded in the USD (clears `generate_scenes_metadata.py`'s `has_capture_metadata` gate), an `occupancy.npy` + `occupancy.json` sidecar next to each USD that clears the corpus free/floor-bbox health bar from `occupancy-interior-fidelity.md`, and a `scenes_metadata.json` discovery entry per scene with `floor_top_z` + room/object labels.
- [ ] `generate_scenes_metadata.py` was run without clobbering the existing `high_quality_dgx` corpus entries (all prior scenes still present after the new scenes are added).
- [ ] `SCENE_USD=<new-scene>.usdc make sim-bridge` (and `make sim-bridge-gui`) pins the bridge to a chosen scene; unset SCENE_USD/SCENE_NAME leaves the current default behavior unchanged. (`make sim-harness` unchanged.)
- [ ] **The misleading `scene_fast_singleroom_000_seed0` is removed and its references reconciled** — the asset is gone, no recipe regenerates a 5-room scene under a "singleroom" name, and `grep -rn "fast_singleroom" docs/ source/` shows no stale/misleading reference (`test_build_mission_queue.py:89`, `docs/HelloRoom.md`/CI, and the doc mentions are repointed/updated).
- [ ] **(Operator-Kit-smoke, satisfied jointly with Brief A.)** With Brief A merged, `SCENE_USD=<single-room-scene> make sim-bridge` spawns the robot **inside** the single room (occupancy-derived spawn from the loaded scene, not a foreign scene's coords) and the mission to the **recorded target object** grounds + reaches it in that single room. This brief's standalone scope is the scenes + occupancy sidecars + passthrough + cleanup; the spawn behavior itself is Brief A.
- [ ] No regression in the existing corpus or coverage: the `high_quality_dgx` scenes still load + capture, the `make sim-harness` scene-selection path is unchanged, and the coverage corpus is untouched (call out the relevant Kit smoke — `make harness-smoke` and a coverage capture on an existing corpus scene).

### Maintenance clause

- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics. In particular this
      change touches an operator copy-paste surface (`make sim-bridge`
      gains `SCENE_USD`/`SCENE_NAME`) and adds a scene preset — sweep
      the top-level `Readme.md`, `source/strafer_lab/README.md`,
      `docs/HelloRoom.md`, and `docs/example_commands_cheatsheet.md`.

## Out of scope

- **Bridge spawn derivation + the union-bug fix** — that is Brief A ([`bridge-spawn-from-occupancy.md`](../trained-policy/bridge-spawn-from-occupancy.md)). This brief supplies the scene + occupancy sidecar Brief A consumes; it does not change `_get_infinigen_spawn_points_xy` / `_apply_infinigen_scene_setup` / the `run_sim_in_the_loop.py` `--scene-usd` spawn re-derivation, and it defers the persist-vs-derive `spawn_points_xy` decision to Brief A.
- **The deterministic scene-selection knob + GB10 memory budget** — owned by [`bridge-scene-memory-budget-gb10.md`](../bridge-scene-memory-budget-gb10.md). This brief delivers the light single-room scene that brief's workaround needs, but does not add the `--scene`/lightest-scene picker over `_get_scene_usd_paths()[0]`.
- **The Jetson backend-selection wiring** (`STRAFER_NAV_BACKEND → default_navigation_backend`) — goal-(a) Prereq 1, a separate Jetson-lane gate; not a scene-supply concern. A+B unblock only the spawn/scene precondition of goal-(a), not the backend gate.
- **Generator-side floor-sampler tooling** (`generate_scenes_metadata.py` floor-sampler params / the area-weighted `_sample_floor_points` path) — owned by the parked `scene-provider-floor-sampler-cli.md`. This brief only *runs* the registration pipeline for one new scene + uses a non-clobbering invocation; it does not redesign the sampler.
- **Bulk corpus generation** — this brief delivers exactly two goal-(a) scenes (one 1-room, one 2-room), not a corpus sweep.
- **Running the cross-room goal-(a) mission** — the two-room scene is the *fixture* for the future cross-room lane (blocked by `autonomy-stack` today); exercising a cross-room mission is out of scope. (Handling of the misnamed `FAST_SINGLEROOM`/`WINDOWS_BASELINE`/`singleroom` recipes is now IN scope — see Scope 5 — because removing the misleading asset requires reconciling them.)
- **Teleop-perf physics / STRAFER_CFG** — no spawn or scene-gen coupling; untouched.
