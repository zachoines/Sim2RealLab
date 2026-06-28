# Perception exposure fine-tuning — close the marginal crush/clip gap and finalize the bands

**Type:** task (perception data quality, fine-tuning)
**Owner:** DGX agent (strafer_lab camera/render config)
**Priority:** P2 — the catastrophic blowout is already fixed and the corpus is usable; this closes the remaining marginal misses and locks the acceptance bands.
**Estimate:** S (one carb knob + a band decision; tools already in place)
**Branch:** `task/perception-exposure-finetune`

**Blocked on / trigger:** Filed-on-trigger. Un-park after the **next Infinigen
corpus regeneration** (operator deferred this to avoid tuning exposure against
soon-to-change scenes). The over-bright fix shipped in
[`coverage-capture-overbright-exposure.md`](../../completed/harness/coverage-capture-overbright-exposure.md)
(#117) eliminated the catastrophic blowout corpus-wide; this is the remaining
fine-tuning, gated on the regenerated corpus.

## Story

As **the perception data collector**, I want **the recorded RGB to also clear the
crushed-black and clip bands on a ceiling-on capture of every scene**, so that
**the corpus has no marginal shadow/highlight loss, not just no catastrophic blowout.**

## Context

The shipped fix enables RTX histogram auto-exposure corpus-wide; it took the
worst scenes from 75–88% clipped (25–38 fully-white frames) to ~1.5–2.3% clipped
(0 white frames), mean luma in [90,150]. Remaining, measured on the operator's
`--video` (ceiling-hidden) captures — so crushed-black is **inflated** by the
ceiling-hide removing fill:

| scene | mean luma | clip (≥250) | crush (≤5) |
|---|---|---|---|
| seed2 | 119.9 | 0.5% | 5.2% (PASS) |
| seed5 | 110.6 | 7.1% | 8.7% |
| seed6 | 128.9 | 1.5% | 12.3% |
| seed7 | 138.1 | 2.3% | 14.2% |

Two residual issues: (1) crushed-black > 10% on the shadowed scenes (seed6/7) —
expected to drop on a ceiling-on capture before any change, and further with an
ambient fill; (2) seed5's ~7% clip is a few genuinely bright in-frame fixtures,
which a global exposure cut cannot fix without re-crushing the others.

## Scope

- **Ambient fill sweep.** On a ceiling-on (non-`--video`) capture, sweep
  `rtx.sceneDb.ambientLightIntensity` (default 1.0) up via
  `coverage_capture --render-carb rtx.sceneDb.ambientLightIntensity=<N>` until
  crushed-black is in band without pushing clip over; commit the value into
  `_INFINIGEN_RENDER_EXPOSURE_CARB` (`strafer_env_cfg.py`). Runbook: capture
  per value into a distinct `--output`, then
  `measure_perception_exposure.py --lerobot-root <dir>`.
- **seed5 clip.** Decide: accept as real in-frame fixtures, apply a `whiteScale`
  / exposure-clamp tweak, or recalibrate the acceptance bands to realistic
  auto-exposed-indoor values (a real camera with windows + hard shadows exceeds
  clip ≤ 2% / crush ≤ 10%).
- **Final acceptance.** Ceiling-on capture of all 5 scenes (seed1/2/5/6/7)
  passing the (possibly recalibrated) bands; report per-scene numbers.

## Related follow-up (separate, can fold in or split)

- **Drop `inject_ceiling_light_emitters`?** Verify in Kit whether Infinigen's own
  `PointLampFactory_*` SphereLights + `RectLight`s render brightly enough that
  the postprocess ceiling-fixture injection is redundant. CPU-confirmed: 0
  Infinigen emitters are authored under any `CeilingLightFactory_*` fixture, so
  the injection lights the ceiling fixtures specifically — dropping it could
  darken rooms with no point lamps. Needs a per-room GPU check; re-bake if dropped.

## Out of scope

- The render-layer auto-exposure mechanism itself (shipped + validated in #117).
- Re-architecting the light bake.

## Investigation pointers

- `source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py` — `_INFINIGEN_RENDER_EXPOSURE_CARB` (add the ambient value here).
- `source/strafer_lab/scripts/coverage_capture.py` — `--render-carb` (sweep seam) + the carb readback log.
- `source/strafer_lab/scripts/measure_perception_exposure.py` — the QA gate (configurable `--max-clip` / `--max-crush` / `--luma-min/max` for band recalibration).
