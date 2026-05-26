# Teleop ergonomics: control modes + target visibility + scene polish

**Type:** new feature (operator UX) + bug-fix bundle
**Owner:** DGX agent
**Priority:** P1 — blocking productive teleop sessions on the harness driver shipping in PR #63.
**Estimate:** M overall, but split: a "ship now" core (S) + three follow-up briefs (each S).
**Branch:** `task/harness-writer-teleop` (continues — same branch as the Tier 1 driver)

## Story

As a **DGX operator running the harness teleop driver on Infinigen scenes**, I want **a usable set of viewport + control + visibility behaviors** so that **I can actually finish episodes ergonomically**. The driver as shipped today drives, but: I can't see where targets are, the robot sometimes spawns outside the room, ceilings occlude the top-down view, doors block paths, and the control mode is a single hard-coded "world-frame arcade" that doesn't suit every collection style.

## Context bundle

- [`harness-architecture.md` §Driver: teleop](harness-architecture.md#driver-teleop) — operator UX baseline; this brief expands it.
- [`harness-architecture.md` §Episode-end button mapping](harness-architecture.md#episode-end-button-mapping-teleop-only) — chord table unchanged.
- [`infinigen-scene-corpus.md`](infinigen-scene-corpus.md) — scene-side fixes (doors, room polygons) belong here; this brief cross-references.
- [`isaac-sim-rt-2-default-renderer.md`](../sim-performance/isaac-sim-rt-2-default-renderer.md) — DLSS / RT 2.0 / FPS Multiplier. Sim-perf scope, not harness scope; cross-reference only.
- [`mission-generator.md`](mission-generator.md) — emits `planned_path` rows. The full suggested-path overlay in `harness-architecture.md:286` waits on this.

## Survey: what's already planned, what's net-new

| Operator ask | Status | Where it lives |
|---|---|---|
| 1a — Top-down + world-arcade twin-stick (existing default) | **Already shipped** in PR #63's teleop driver. Keep as default; expose under `--control-mode world_arcade`. | This brief (rename + flag only). |
| 1b — Egocentric first-person + classic control | **Net-new.** Stick maps to body-frame; viewport follows the robot. | This brief (`--control-mode egocentric`). |
| 1c — Decouple render FPS from capture FPS | **Partially net-new.** Render-side perf (RT 2.0 / DLSS) is `isaac-sim-rt-2-default-renderer.md`'s scope. Decoupling the *capture cadence* from the env step rate (`--capture-rate-hz`) is net-new and small. | This brief (capture-rate flag). Sim-perf brief stays separate. |
| 2a — Operator-visible pathing | **Already planned.** Full `planned_path` debug-draw polyline is `harness-architecture.md:286` deferred to Tier 1.5, gated on `mission-generator` shipping. **Expanded here:** ship a cheap **target marker** (single colored sphere at `target_position_3d`) now — covers 90% of "where am I going?" without waiting on mission-generator. | This brief (target marker) + cross-ref to existing plan. |
| 2b — Pre-episode point-click target selection | **Net-new + heavy.** Real Kit-side interactive pick coroutine + side panel UI. Out of scope for the harness PR shipping window. | **Follow-up brief filed below.** |
| 3a — Hide ceilings in top-down view | **Net-new.** Runtime fix (toggle visibility on prims matching `*_ceiling_*`) is cheaper than re-baking USDC. | This brief (`--hide-ceilings`). |
| 3b — Open doors at preprocess | **Net-new — preprocess scope.** Folds into `infinigen-scene-corpus.md`. | **Adds an acceptance item to** `infinigen-scene-corpus.md`. |
| 3c — Robot spawns outside Infinigen room | **Real bug.** `_get_infinigen_spawn_points_xy()` at `strafer_env_cfg.py:1164` pools spawn points across ALL scenes in `scenes_metadata.json` instead of just the active one. With two scenes loaded, ~50% of resets place the robot at the wrong scene's coordinates. | This brief (fix). |
| 3c (alt) — Point-click robot placement / cycle saved spawns | **Net-new + heavy.** Same scope as 2b. | **Folded into the 2b follow-up brief.** |

## Acceptance (ship-now bundle, this brief)

All of these land in the same PR as the teleop driver (PR #63), since they all touch `teleop_capture.py` and the cheatsheet:

- [ ] **Active-scene-only spawn-point pool.** Teleop driver overrides `cfg.events.reset_robot.params["spawn_points_xy"]` to use only the entry for `--scene` from `scenes_metadata.json`. Diagnostic print at startup of pool size + first spawn-point sampled per episode. Same patch fixes `goal_command.spawn_points_xy`.
- [ ] **`--hide-ceilings`** runtime flag (default off). After env creation, walk the stage and set `visibility="invisible"` on every prim whose name matches a configurable regex (default catches `*_ceiling_*` per `generate_scenes_metadata.py:62`'s known pattern). Restored on close.
- [ ] **Target marker** — colored debug-draw sphere at the open episode's `target_position_3d`. Rendered to the editor viewport only, NOT into camera render products (same guarantee as the planned suggested-path overlay). Cleared when the episode closes; redrawn when the next episode opens.
- [ ] **`--control-mode {world_arcade, egocentric}`** (default `world_arcade`):
  - `world_arcade`: today's behavior. ViewerCfg overhead at `(0, 0, 12)` looking down. Left stick = world-frame velocity; right stick X = yaw rate.
  - `egocentric`: ViewerCfg `origin_type="asset_root"`, `asset_name="robot"`, eye behind + above the robot looking forward. Left stick = body-frame velocity (no heading rotation); right stick X = yaw rate. Useful for CLIP-style coverage where the operator needs to see what the robot sees.
- [ ] **`--capture-rate-hz`** (default 8) decoupled from env step rate. Env steps every sim tick (smoothness preserved); writer samples every `round(env_step_rate_hz / capture_rate_hz)` ticks. The existing `--fps` flag is renamed to `--capture-rate-hz` (alias `--fps` retained for back-compat in this PR).
- [ ] Cheatsheet's "Harness data capture" section documents all four new flags + the spawn-point fix.
- [ ] `make test-harness` stays at 116+ tests, all green. Pure-Python additions get unit tests.

## Acceptance (follow-up briefs, filed by this brief)

Two follow-up briefs split off:

### `teleop-in-sim-target-picker.md` (file under `parked/harness/`)

Pre-episode point-click target selection (your 2b) + point-click / cycle-saved-spawn robot placement (your 3c alternative). One brief covers both — they share the same Kit-side interactive-pick infrastructure (a coroutine that yields until the operator clicks in the viewport, then resolves the clicked prim path back to the closest scene_metadata object). Estimate M–L. Parked because:

- Console picker + target marker (this brief) together close most of the UX gap.
- The interactive-pick scaffold is genuinely new code (no existing precedent in the repo).
- Whichever pattern wins ("click an object" vs "cycle a side panel") is a UX call best made *after* operators have used the console-picker + target-marker pattern for a session or two.

### Update to `infinigen-scene-corpus.md` (acceptance item)

Add: "Doors in generated scenes are open by default, OR a postprocess pass rotates door prims to their open transform. Validated by spawning the robot at one room's spawn point and confirming a clear path to an adjacent room's spawn point exists at sim startup (no closed-door collision)." Single bullet adds to the existing brief's acceptance.

## Out of scope (this brief)

- **Full suggested-path overlay** — the `planned_path` polyline over LLM-emitted waypoints. Stays in `harness-architecture.md:286`'s Tier 1.5 list, gated on `mission-generator`.
- **DLSS / RT 2.0 perf** — `isaac-sim-rt-2-default-renderer.md`. The capture-rate decouple this brief ships is orthogonal — it's a writer-cadence change, not a renderer change.
- **Multi-camera live view** (PIP via cv2 was already attempted; degraded to no-op against `opencv-python-headless`). Re-attempt via Isaac Sim secondary viewport would be a sibling brief if the egocentric mode doesn't close the UX gap.
- **Anything Jetson-side.** All scope is DGX in-process Isaac Lab.

## Implementation outline

1. **Spawn-point fix** in `teleop_capture.py`:
   ```python
   scene_block = scenes_metadata.get("scenes", {}).get(args.scene, {})
   pool = scene_block.get("spawn_points_xy", [])
   if not pool:
       raise SystemExit(f"no spawn_points_xy in scenes_metadata.json for scene {args.scene!r}")
   env_cfg.events.reset_robot.params["spawn_points_xy"] = pool
   env_cfg.commands.goal_command.spawn_points_xy = pool
   print(f"[teleop_capture] using {len(pool)} spawn points from active scene only")
   ```

2. **`--hide-ceilings`** — after env creation, traverse the USD stage, find prims matching `r"_ceiling(_\d+)?$"`, call `UsdGeom.Imageable(prim).MakeInvisible()`. Restore on shutdown (not strictly needed for a one-shot teleop session but cheap to track).

3. **Target marker** — use `isaacsim.util.debug_draw.get_debug_draw_interface()` to render a colored sphere at `(target_position_3d[0], target_position_3d[1], target_position_3d[2] + 0.3)`. Persist across ticks until `end_episode`, then clear. Debug-draw lives outside Replicator's render product capture path, so the marker is operator-only — same guarantee as the planned suggested-path overlay.

4. **`--control-mode`** — add to argparse. In `world_arcade`, keep `ViewerCfg(eye=(0,0,12), lookat=(0,0,0), origin_type="env")` and `_stick_to_body_action` as today. In `egocentric`, set `ViewerCfg(eye=(-2.0, 0, 1.2), lookat=(2.0, 0, 0.5), origin_type="asset_root", asset_name="robot")` and short-circuit the world→body rotation in the stick mapper (stick already body-frame; no `cos_h/sin_h` rotation needed).

5. **`--capture-rate-hz`** — argparse adds the flag. Compute `ticks_per_capture = max(1, round(env_step_rate_hz / capture_rate_hz))`. In the loop, increment a `tick_counter` per `env.step`; only call `writer.add_frame` when `tick_counter % ticks_per_capture == 0`. The existing `--fps` becomes an alias.

6. **Tests** — pure-Python unit tests for the new `_resolve_active_spawn_points`, the ceiling-name regex matcher, and the control-mode stick math.

## Investigation pointers

- Existing teleop driver: [`source/strafer_lab/scripts/teleop_capture.py`](../../../../source/strafer_lab/scripts/teleop_capture.py).
- Spawn-point pool helper: [`strafer_env_cfg.py:1164`](../../../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py#L1164) (`_get_infinigen_spawn_points_xy`).
- ViewerCfg follow-cam: `isaaclab.envs.common.ViewerCfg` supports `origin_type="asset_root"` with `asset_name="robot"` — the same mechanism this brief uses for egocentric.
- Debug-draw API: `from isaacsim.util.debug_draw import _debug_draw; debug_draw_interface = _debug_draw.acquire_debug_draw_interface()`.
- Ceiling prim naming: `generate_scenes_metadata.py:62` documents the regex `^[a-z]+(?:_[a-z]+)*(?:_\d+)+_(floor|ceiling|wall|exterior|staircase)$`.
- Cheatsheet: [`docs/example_commands_cheatsheet.md`](../../../example_commands_cheatsheet.md) — running ops reference; harness section gets the new flags.
