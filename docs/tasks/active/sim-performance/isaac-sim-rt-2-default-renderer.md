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

## Findings & decisions (implementation)

The three knobs were verified against the installed Isaac Sim 6.0.0 + IsaacLab
source (adversarially cross-checked). The brief's setting names predate this
build; the corrected knobs:

| Brief's guess | Actual (verified) |
|---|---|
| renderer `RealTime2` | `/rtx/rendermode = "RealTimePathTracing"` (the UI "Real-Time 2.0"; already the platform SimulationApp default). RT 1.0 is `RaytracedLighting`. |
| `/rtx/post/fpsMultiplier → 4` | DLSS-G frame generation `/rtx-transient/dlssg/enabled` (on/off). The x2/x3/x4 factor is an opaque internal *hashed* Kit setting — no stable carb path, no `RenderCfg` field, so "4×" is not cleanly expressible. |
| `/rtx/rendermode/quality → Performance` | `--rendering_mode performance` → `apps/rendering_modes/performance.kit` (DLSS execMode 0, RT2 path-tracer tuning). |

Settings are injected at **app-launch** (Kit CLI args), not via post-boot
`RenderCfg.carb_settings`: RT 2.0's renderer *registration*
(`/rtx-transient/rt2Enabled`, `/persistent/rtx/modes/rt2/enabled`) is a boot-time
persistent preference — headless IsaacLab logs show it absent by default, so the
mode must be forced at startup. `RenderCfg.carb_settings` stays the home for
dynamic render settings (e.g. the RTX auto-exposure histogram).

**Depth-integrity decision (Constraint 1): DLSS frame generation is deliberately
NOT enabled — ships at 1×.** The depth stream is a live policy input. DLSS-G is a
swapchain/present-path feature: it never writes the offscreen annotator AOVs
(`rgb`, `distance_to_image_plane`) the bridge reads, is a no-op headless (present
thread disabled), and cannot inject/duplicate frames into the env-step-driven
publish cadence (worst case a DROP). Isaac Sim's own Performance preset disables
it ("does not yet support tiled camera well") and the base app disables it as
synthetic-data-incompatible. This is Constraint 1's sanctioned "drop to 1×, ship
RT 2.0 + Performance only" path — scoping FG off the sensor path was possible, so
no STOP. The one residual risk — RT1↔RT2 *geometric* depth parity — is not
doc-guaranteed and is the sole runtime gate:
[`probe_rt2_depth_integrity.py`](../../../source/strafer_lab/scripts/probe_rt2_depth_integrity.py)
checks (a) cadence, (b) frame-diff under motion, (c) RT1-vs-RT2 static depth
≤ 1e-3 m.

**Merge gate.** Per the scheduling rule this stays unmerged and the rig stays on
the current renderer until (1) the operator's v1 depth-subgoal live mission runs
on the current config (the depth-chain baseline), (2) the depth-integrity probe
passes on RT 2.0, and (3) the `--profile` perf rows are filled. Stamp + `git mv`
this brief to `completed/` in the merge PR once those land.

## Acceptance criteria

Code + tests + docs ship in this PR; the GPU-dependent rows are gated on the
operator's bridge sessions (see **Merge gate** above). Legend: `[x]` done, `[~]`
partial/revised, `[ ]` gated on a GPU session.

- [x] Bridge launch on default settings selects RTX Real-Time 2.0 — startup log line `[sim_in_the_loop] active renderer: /rtx/rendermode=...` reads the live setting back (proof once a session runs).
- [~] FPS Multiplier + Performance: Performance preset enabled (`--rendering_mode performance`); the FPS multiplier is **deliberately off (1×)** for depth integrity — revised from the brief with rationale in Findings.
- [x] Single launch-arg revert: `--renderer legacy` (`RENDERER=legacy make sim-bridge`), documented in the cheatsheet + `bridge-runtime-invariants.md`.
- [ ] New perf numbers in `bridge-runtime-invariants.md` — rows added (TBD); fill with `--profile` on both make targets in a GPU session.
- [~] Camera-publish contract: unit-pinned (the renderer override touches only RTX launch settings, never resolution/intrinsics); runtime confirmation via `bridge_harness_smoke.py`'s depth-shape assert + the depth-integrity probe (gated).
- [ ] Scene visual correctness via `make sim-bridge-gui` on an Infinigen scene (operator visual check, gated).
- [x] Doc maintenance sweep: cheatsheet, package README, tools-and-scripts-map, `bridge-runtime-invariants.md` updated (interim `INTEGRATION_SIM_IN_THE_LOOP.md` left untouched — exempt).

## Investigation pointers

- `source/strafer_lab/strafer_lab/bridge/` — bridge init.
- `source/strafer_lab/strafer_lab/bridge/graph.py` — OmniGraph render product nodes; renderer settings often live near here.
- `source/strafer_lab/scripts/run_sim_in_the_loop.py` — bridge entry point; might be the natural place for a renderer launch arg.
- Isaac Sim renderer settings reference: NVIDIA's RTX render mode docs (look for `rtx.rendermode`, `rtx.post.fpsMultiplier`).

## Out of scope

- **Real-time policy training renderer.** Training default in `source/strafer_lab/scripts/train_strafer_navigation.py` is its own decision (Path-Tracing isn't used there anyway; just don't touch).
- **DLSS quality / sharpness tuning.** Default presets are fine for v1; deeper DLSS tuning is a separate brief if it surfaces as a perception-side issue.
- **Eliminating the bridge perf bottleneck entirely.** Camera-publish OmniGraph cost is tracked separately under `async-camera-publishers.md`. RT 2.0 chips at one piece of the loop; doesn't replace that work.
