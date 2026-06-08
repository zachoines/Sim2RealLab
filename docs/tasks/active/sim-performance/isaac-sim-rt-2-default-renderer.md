# Default Isaac Sim/Lab to Real-Time 2.0 renderer with 4x FPS multiplier

**Type:** task / config
**Owner:** DGX agent (`source/strafer_lab/`, likely `bridge/` or scene init)
**Priority:** P2
**Estimate:** S (~half-day; renderer setting + FPS multiplier + perf-mode toggle, plus retest)
**Branch:** task/isaac-sim-rt-2-default-renderer

## Story

As a **sim-in-the-loop operator running the bridge headed or headless**, I want **Isaac Sim/Lab to default to the Real-Time 2.0 renderer at 4× FPS multiplier in Performance mode (not the Interactive Path-Tracing renderer)**, so that **the camera-publish OmniGraph and the editor viewport both produce frames at the faster RT 2.0 rate — closing some of the sub-unity RTF gap that current headless runs sit in (~0.04–0.1 RTF per [bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md))**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/bridge-runtime-invariants.md](../../context/bridge-runtime-invariants.md)
  — "Phase-level profiler" reference numbers we'll want to re-measure after the change.

## Context

Isaac Sim ships several render modes: Real-Time 1.0 (the legacy RTX path), Real-Time 2.0 (the newer RTX path with improved DLSS / denoiser integration), and Interactive (full path tracing). The bridge currently runs whatever the Kit app config defaults to — most likely RT 1.0 in a headless context, and Interactive when the viewport is active.

The DGX bridge's mainloop is bottlenecked on rendering pipeline cost (`sim.render` + `simulation_app.update`) per the bridge-runtime-invariants table — 88.6 ms `sim.render` (headed) and 74.2 ms `simulation_app.update` (headless). RT 2.0 + the FPS Multiplier (DLSS-style frame generation) can reportedly close some of that gap on supported GPUs (RTX 30/40-series). Performance mode trades shader quality for throughput.

Three knobs to flip:
1. **Renderer:** `RaytracedLighting` (RT 1.0) / `RealTime2` (RT 2.0) / `Interactive` (path tracing). Default to `RealTime2`.
2. **FPS Multiplier:** Kit's DLSS-FG-equivalent. Default to 4×.
3. **Render mode:** `Performance` vs `Quality`. Default to `Performance`.

These can be set via Kit `.kit` app config, via Isaac Sim Python API in the bridge init, or via Hydra render settings. The bridge already constructs `PinholeCameraCfg` and `CameraStreamConfig` — there's a place to inject renderer settings near there.

## Approach

### A. Locate the renderer setting

Identify where Kit's renderer is currently chosen in our stack:
- Is it a `.kit` app file we ship?
- Is it set via `simulation_app.update(carb.settings)` calls in the bridge init?
- Is it a Hydra render config in the env build?

### B. Add the three settings

Flip:
- `/rtx/rendermode` → `RealTime2`
- `/rtx/post/fpsMultiplier` (or equivalent Kit setting) → `4`
- `/rtx/rendermode/quality` → `Performance`

Gate on a launch arg or env var so an operator can revert to the legacy renderer for visual debugging (some shaders behave differently between RT 1.0 and RT 2.0, and a debugging session benefits from determinism).

### C. Re-measure perf

Run the bridge's `--profile` mode and capture a new row in the bridge-runtime-invariants doc's perf table:

| Configuration | `sim.step` p50 | `sim.render` p50 | `simulation_app.update` p50 | Loop total p50 | Throughput |
|---|---|---|---|---|---|
| `make sim-bridge` headless RT 2.0 + FPS×4 + Perf | TBD | TBD | TBD | TBD | TBD |
| `make sim-bridge-gui` RT 2.0 + FPS×4 + Perf | TBD | TBD | TBD | TBD | TBD |

Update the doc with the new numbers in the same commit.

## Acceptance criteria

- [ ] Bridge launch on default settings produces renders via the Real-Time 2.0 renderer (verifiable via a log line or a Kit setting dump on startup).
- [ ] FPS Multiplier is set to 4× and Performance mode is enabled.
- [ ] An operator can revert to the legacy renderer with a single launch-arg override (document the override in the bridge runbook).
- [ ] New perf numbers recorded in `bridge-runtime-invariants.md`'s reference table; if RT 2.0 closes the perf gap meaningfully, update the "Headless vs `--viz kit` defaults" section's narrative.
- [ ] No regression in the camera-publish contract: `/d555/color/image_raw`, `/d555/depth/image_rect_raw`, and their `camera_info` companions still publish at the same resolution + same intrinsics. RT 2.0 must not silently change resolution or aspect.
- [ ] No regression in scene visual correctness for the operator's standard debugging scenes (run an Infinigen scene through `make sim-bridge-gui` and confirm the viewport looks sane).
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance) for the surface list and trigger heuristics.

## Investigation pointers

- `source/strafer_lab/strafer_lab/bridge/` — bridge init.
- `source/strafer_lab/strafer_lab/bridge/graph.py` — OmniGraph render product nodes; renderer settings often live near here.
- `source/strafer_lab/scripts/run_sim_in_the_loop.py` — bridge entry point; might be the natural place for a renderer launch arg.
- Isaac Sim renderer settings reference: NVIDIA's RTX render mode docs (look for `rtx.rendermode`, `rtx.post.fpsMultiplier`).

## Out of scope

- **Real-time policy training renderer.** Training default in `source/strafer_lab/scripts/train_strafer_navigation.py` is its own decision (Path-Tracing isn't used there anyway; just don't touch).
- **DLSS quality / sharpness tuning.** Default presets are fine for v1; deeper DLSS tuning is a separate brief if it surfaces as a perception-side issue.
- **Eliminating the bridge perf bottleneck entirely.** Camera-publish OmniGraph cost is tracked separately under `async-camera-publishers.md`. RT 2.0 chips at one piece of the loop; doesn't replace that work.
