# Coverage capture over-bright exposure — correct the blown-to-white d555 perception RGB corpus-wide

**Type:** bug (perception data quality) + diagnosis
**Owner:** DGX agent (scene-provider postprocess + strafer_lab camera/render config)
**Priority:** P0 — the recorded d555 perception RGB is clipped to white on the over-bright scenes; the corpus is unusable for perception training until exposure is correct on all 5 scenes.
**Branch:** task/coverage-capture-overbright-exposure
**PR:** _(pending)_
**Status:** code + diagnosis in review; numeric acceptance is OPERATOR-GATED on a production ceiling-on capture (see Acceptance).

## Story

As **the perception data collector**, I want **the recorded d555 RGB frames to be correctly exposed (not clipped to white) on every corpus scene**, so that **the LeRobot corpus is usable training data instead of blown-out frames.**

## Context bundle

- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [SCENE_PROVIDER_CONTRACT.md](../../../SCENE_PROVIDER_CONTRACT.md) — light inventory + render-exposure section (updated here)
- Predecessor: [completed/coverage-capture-multiscene-correctness.md](../../completed/coverage-capture-multiscene-correctness.md) (#116 — geometry now correct on all 5 scenes; exposure was the last blocker to a usable corpus).

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

The dominant illumination is **Infinigen's own physically-based HDR emitters**
(point lamps, ~4 orders of magnitude above everything else). The postprocess
`--light-intensity` 100000 lights and the env dome/distant are radiometrically
negligible / additive fill. Kit's RTX renderer applies a default ACES tonemap
but leaves **auto-exposure OFF** (`/rtx/post/histogram/enabled` = 0), so the
billions-of-nits HDR sits above the tonemap shoulder and the recorded LDR RGB
(the perception camera's `rgb` = post-tonemap `LdrColor` AOV) clips to white.

Per-scene brightness does **not** track baked-emitter power (seed5 has the most
power yet is the darkest) — the point-HDR-emitter framing signature. So no single
fixed intensity/scale satisfies both the blown (seed6/7) and dim (seed5) scenes:
the empirical trigger for an adaptive render-layer fix.

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
- `measure_perception_exposure.py`: the re-runnable CPU QA gate (per-scene
  clip/luma/crush + PASS/FAIL vs the bands).
- Docs: SCENE_PROVIDER_CONTRACT light-inventory + render-exposure section;
  corrected `postprocess_scene_usd.py` docstring/help (it claimed Infinigen
  exports no emitters and that `--light-intensity` fixes blowout — both false).

**Why not the alternatives:** (B) lowering `--light-intensity` touches only the
~0.02% fill lights — a no-op; (C) trimming the env dome/distant removes additive
fill and would deepen seed5's crush; (D) scaling the baked emitters cannot fix
the seed5↔seed6/7 spread (anti-correlated power vs brightness).

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

## Acceptance criteria (measured on a PRODUCTION ceiling-on, non-`--video` capture)

- [ ] **Operator:** run `coverage_capture` WITHOUT `--video` (or with
      `--video-keep-ceiling`) on at least seed6/seed7 (worst) + seed5 (dim);
      re-run `measure_perception_exposure.py` on the recorded perception MP4s.
- [ ] Clipped (any channel ≥250) ≤ 2% per scene AND 0 fully-white frames.
- [ ] Mean luma in [90, 150] per scene.
- [ ] Crushed-black (any channel ≤5) ≤ 10% per scene; **seed5 passes clipped AND
      crushed simultaneously**.
- [ ] `--video` smoke on seed1 (known-good) + seed2/5/6/7 reads correctly.
- [ ] Confirm via the capture log's carb readback that the exposure settings
      reached the live RTX renderer (render-product check — a viewport-only
      "looks fine" is not acceptance).
- [ ] If your work invalidates a fact in any referenced context module, package
      README, or guide under `docs/`, update those in the same commit.
- [ ] No regression in the coverage-capture workflow (harness + navigation
      suites green; `--video` smoke unchanged).

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
