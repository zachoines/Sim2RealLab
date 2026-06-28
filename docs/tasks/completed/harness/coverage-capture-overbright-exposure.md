# Coverage capture over-bright exposure — correct the blown-to-white d555 perception RGB corpus-wide

**Status:** Shipped 2026-06-28 in `8cd3bb4` (DGX). Diagnosed the blown-to-white d555 perception RGB to the render layer — the scene's bright baked emitters render with RTX auto-exposure OFF — and enabled RTX histogram auto-exposure corpus-wide via `RenderCfg.carb_settings` (re-bake-free), plus a `coverage_capture --render-carb` probe seam + carb readback and a reusable `measure_perception_exposure.py` QA gate. **Validated** on the operator's fresh captures: the catastrophic blowout is gone (seed6/7 clipped fraction 88%/75% → 1.5%/2.3%, fully-white frames 25/38 → 0, mean luma 245/239 → 129/138 in band; seed2 passes outright), and the auto-exposure carb settings reach the TiledCamera render product (the PR's #1 risk, resolved). Remaining work is **fine-tuning only** (marginal crushed-black on the dim/shadowed scenes + a few bright in-frame fixtures) and is deferred to [`parked/harness/perception-exposure-finetune`](../../parked/harness/perception-exposure-finetune.md), triggered by the planned Infinigen corpus regen.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/117
**Type:** bug (perception data quality) + diagnosis
**Owner:** DGX agent (scene-provider postprocess + strafer_lab camera/render config)
**Priority:** P0 — the recorded d555 perception RGB was clipped to white on the over-bright scenes; the corpus was unusable for perception training.
**Branch:** task/coverage-capture-overbright-exposure

## Story

As **the perception data collector**, I want **the recorded d555 RGB frames to be correctly exposed (not clipped to white) on every corpus scene**, so that **the LeRobot corpus is usable training data instead of blown-out frames.**

## Context bundle

- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- Predecessor: [coverage-capture-multiscene-correctness.md](coverage-capture-multiscene-correctness.md) (#116 — geometry now correct on all 5 scenes; exposure was the last blocker to a usable corpus).

## Diagnosis (CPU-measured)

The defect is **render-layer**, not the baked SphereLights.

**Measured clipping (on-disk `--video` smoke runs, via `measure_perception_exposure.py`):**

| scene | mean luma | clip (≥250) | crush (≤5) | fully-white frames | row profile |
|---|---|---|---|---|---|
| seed5 | 62.4 | 7.3% | 42.2% | 0 | MID-peaked (bimodal: dark voids + blown edges) |
| seed6 | 245.2 | 87.7% | 0.3% | 25/138 | BOTTOM-peaked (0.75→0.96) |
| seed7 | 239.0 | 74.7% | 0.7% | 38/399 | BOTTOM-peaked (0.49→0.87) |

The BOTTOM/MID-peaked row signatures + the fully-white horizontal frames are
whole-scene over-exposure (floor/mid), **not** ceiling bloom (which would peak at
top) — so the `--video` global ceiling-hide confound is small and these frames
are production-relevant. (The numeric exposure target still must be set on a
ceiling-on capture — see Acceptance.)

**Light inventory (per-scene, baked into each `export_scene.usdc`, via pxr):**

| layer | type | n | intensity | summed |
|---|---|---|---|---|
| Infinigen point lamps (DOMINANT) | `SphereLight` `normalize=True` | 12–14 | 1.1e8–6.1e8 | **3.7e9–6.3e9** |
| postprocess fill (`inject_ceiling_light_emitters`) | `SphereLight` `normalize=False` | 8–10 | 1e5 | ~1e6 (**~0.02%**) |
| Infinigen area | `RectLight` `normalize=True` | 19–23 | 3.18 | ~60–73 |
| Infinigen env | `DomeLight` | 1 | 0.25 | 0.25 |
| env-cfg fill (runtime) | DomeLight 2000 + DistantLight 3000 | — | — | additive |

The dominant illumination is the scene's own bright baked emitters
(`PointLampFactory_*` lamps + `RectLight`s; physically-based, ~orders of
magnitude above the env dome/distant fill). Kit's RTX renderer applies a default
ACES tonemap but leaves **auto-exposure OFF** (`/rtx/post/histogram/enabled` = 0),
so the HDR radiance sits above the tonemap shoulder and the recorded LDR RGB (the
perception camera's `rgb` = post-tonemap `LdrColor` AOV) clips to white. Per-scene
brightness does not track baked-emitter power (seed5 has the most yet is darkest),
so no single fixed intensity/scale fixes both the blown and dim scenes — the
trigger for an adaptive render-layer fix.

Note: `PointLampFactory_*` lamps and `RectLight`s are a **separate** prim set from
the `CeilingLightFactory_*` *fixtures* that `postprocess_scene_usd.py`'s
`inject_ceiling_light_emitters` adds a SphereLight under (those fixtures carry no
Infinigen emitter — verified: 0 per scene). The ceiling injection is a distinct,
still-needed fill, not the over-exposure driver; whether the `PointLampFactory`/
`RectLight` emitters render in Isaac Sim well enough to drop the injection is a
separate follow-up (operator Kit check), out of scope here.

## The fix (shipped here)

Enable **RTX histogram auto-exposure** corpus-wide on the sim's
`RenderCfg.carb_settings` for Infinigen scenes — the one re-bake-free,
scene-count-invariant lever that adapts to the per-scene spread, and what the
real D555 does (its ROS driver runs auto-exposure). Affects only RGB (depth is
geometric), so it is safe on every Infinigen env, not just captures.

- `strafer_env_cfg.py`: `_apply_infinigen_render_exposure(cfg)` sets
  `cfg.sim.render.carb_settings = {rtx.post.histogram.enabled: True,
  rtx.post.histogram.whiteScale: 7.0}`, called from `_apply_infinigen_scene_setup`.
- `coverage_capture.py`: `--render-carb KEY=VALUE` (repeatable) to sweep
  exposure knobs on the production probe without source edits, + a post-init
  carb readback log (confirms the setting reached the live RTX renderer).
- `measure_perception_exposure.py`: a re-runnable CPU QA gate (per-file
  clip/luma/crush + PASS/FAIL vs the bands; takes MP4 paths or `--lerobot-root`).

**Why not the alternatives:** (B) lowering `--light-intensity` touches only the
ceiling-fixture fill lights — does not move the dominant emitters; (C) trimming
the env dome/distant removes additive fill and would deepen crush; (D) scaling
the baked emitters cannot fix the seed5↔seed6/7 spread (anti-correlated power vs
brightness).

## OPEN — resolved by the operator probe (auto vs fixed exposure)

Shipped default is **auto-exposure**. It cannot be verified CPU-side that RTX
histogram auto-exposure reaches the offscreen **TiledCamera render product** (vs
the GUI viewport), or that it meters cleanly. `num_envs==1` (enforced by the
capture driver) removes the tiled multi-env averaging risk, and the `rgb` AOV is
post-tonemap (so exposure settings are in-path), but the histogram behavior on
the tiled RP is the residual risk. If the probe shows AE mis-converges or
flickers, switch to the documented **fixed-exposure fallback**
(`--render-carb rtx.post.histogram.enabled=false rtx.post.tonemap.filmIso=<low>`)
and commit those values. seed5's crushed-black may additionally need an ambient
fill (`rtx.sceneDb.ambientLightIntensity`) — re-measure its baseline on the
ceiling-on capture first, since `--video` removed its ceiling fill.

## Acceptance criteria

Achieved (this PR — the catastrophic data-loss fix):

- [x] Catastrophic blowout eliminated: clipped fraction collapses (seed6/7
      88%/75% → 1.5%/2.3%) and fully-white frames go to 0 (from 25/38); mean
      luma lands in [90,150] on every measured scene; seed2 passes the full gate.
- [x] Auto-exposure carb settings confirmed to reach the live RTX renderer /
      TiledCamera render product (carb readback + the clip collapse on the
      recorded MP4 — not a viewport-only observation).
- [x] No regression in the coverage-capture workflow (harness + navigation
      suites green; `--render-carb` parsing + the exposure cfg unit-tested).
- [x] Measured on the operator's fresh captures (still `--video` ceiling-hidden,
      which inflates crushed-black — the remaining gap is partly that confound).

Deferred to [`parked/harness/perception-exposure-finetune`](../../parked/harness/perception-exposure-finetune.md)
(operator chose to take the moderate win now and finalize after the planned
Infinigen corpus regen, rather than tune against soon-to-change data):

- [ ] Final ceiling-on (non-`--video`) acceptance across all 5 scenes against the
      bands (clipped ≤ band, mean luma in band, crushed-black ≤ band).
- [ ] Pull crushed-black under band on the dim/shadowed scenes via an ambient
      fill sweep (`rtx.sceneDb.ambientLightIntensity`); commit the value.
- [ ] Resolve seed5's residual in-frame bright-fixture clipping (accept as real,
      or `whiteScale`/clamp tweak, or recalibrate the bands to realistic indoor).

## Investigation pointers

- `source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py` — `_INFINIGEN_RENDER_EXPOSURE_CARB`, `_apply_infinigen_render_exposure`, the env-cfg dome/distant fill.
- `source/strafer_lab/scripts/coverage_capture.py` — `--render-carb`, `_parse_render_carb_overrides`, `_log_render_carb_readback`.
- `source/strafer_lab/scripts/measure_perception_exposure.py` — the QA gate.
- RTX carb keys verified in installed Kit schema (`omni.usd.schema.render_settings.rtx`): `rtx.post.histogram.{enabled,whiteScale,tau,minEV,maxEV,useExposureClamping}`, `rtx.post.tonemap.{op,filmIso,exposureTime,fNumber}`, `rtx.sceneDb.ambientLightIntensity`. `cameraExposure` does NOT exist.
- IsaacLab `RenderCfg.carb_settings` application: `IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py` `_apply_render_cfg_settings`.

## Out of scope

- Re-baking emitter intensities / authoring `normalize` (rejected — see "Why not the alternatives").
- Changing the postprocess light injection behavior (only its docstring/help are corrected).
- Exposure domain randomization (`enable_exposure_variation`) — a separate post-pixel augmentation, not an un-clip.
- The 80x60 policy training envs' RGB (policy observes depth; auto-exposure touches only RGB).
