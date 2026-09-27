# Decide what the training depth noise does about the structure the real D555 capture measured

**Type:** investigation (training noise model — a decision, then possibly a change)
**Owner:** DGX
**Priority:** P2 — the shipped `disparity_noise_px = 0.08` passes gate (B), so nothing is broken.
The capture also measured real noise structure that the training term does not model. It is P2
rather than P3 because the robust tier v3 trained on draws mostly below the passing interval, and
the next retrain decision should see that.
**Estimate:** M (a matched sim capture, gate (C), then a decision per item)
**Branch:** `task/depth-noise-real-structure`

## Story

As the **training depth noise model**, I need **the real-sensor structure that
[`real-d555-depth-texture-2026-09-26`](../../../measurements/real-d555-depth-texture-2026-09-26/README.md)
measured compared, item by item, with what training injects**, so that **any change to the noise
term rests on a measured gap and a passed gate, not on the amplitude check alone.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [`real-d555-depth-texture-2026-09-26`](../../../measurements/real-d555-depth-texture-2026-09-26/README.md)
  — every number below.
- [`noise-texture-parity-2026-09-17`](../../../measurements/noise-texture-parity-2026-09-17/README.md)
  §8 — the gates (Gate 0, (B), (C), (G)) any change is accepted against.

## Context

`DepthNoiseModel` injects σ_z = z²·σ_d/(f·B) i.i.d. per 80×45 cell after the reduction. It has no
surface, angle or spatial-correlation term. The capture was taken at robot mount height on a glossy
hardwood floor and found:

1. **Amplitude: passes.** Gate (B) passes at σ_d 0.08 in every band. The sensor is the noisier side
   by 1.11–1.88×. The worst band is 2.5–3.5 m (0.531, one box face), close to §8(G)'s 2×.
2. **The robust tier sits mostly below the passing interval.** v3 trained on σ_d drawn
   log-uniformly over [0.002, 0.16]. Only [0.075, 0.16] passes every band, about 17 % of draws. The
   median draw injects 5–8× less than the sensor delivers.
3. **The floor is noisier than walls at the same depth.**
   - The registered cell rule excludes the floor, because it drops out intermittently at grazing
     angles.
   - On the deployed output, floor σ_post is 1.5–2.2× that of other surfaces, depth-normalised.
   - Part of that is validity-mask flicker through the 6.0 m substitution. Removing it still leaves
     1.3–1.7×.
4. **Specular floor reflections.** 2–2.5 % of the frame's cells read phantom depth beyond the floor
   plane.
5. **Noise correlated across cells.** Adjacent 80×45 cells' temporal residuals correlate at
   0.32–0.61, rising with band. Training's per-cell noise is independent, so gate (C)'s
   |d − median3×3| statistic will expose this.
6. **Quantisation structure.** The real depth sits on a disparity lattice (C ≈ 1000 m). Some Z16
   codes never occur.

Gate (C) and any training-side change need §8 Gate 0's matched sim capture through the 640×360
perception camera. It was not taken.

## Acceptance criteria

- [ ] A matched sim capture: same poses, 640×360 perception camera, the tier's noise, and the
      fitted geometry of the real capture. Lens 0.278 m, pitch −0.9°, and the real intrinsics (see
      below).
- [ ] Gate (C) evaluated on real vs sim frames, per band, with the registered statistic.
- [ ] A per-item decision for items 2–6: model it, deliberately not model it, or defer, with the
      reason. Any change to a noise field is gated by §8, including (G)'s direction rule.
- [ ] If a change is made, it ships with its own record and gate readings. No change is also a
      valid outcome and says so.

## Investigation pointers

- `source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py`, `DepthNoiseModel`.
- `source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py`, the tiers'
  `disparity_noise_px` and its range.
- The deposit's `analysis/secondary/` and `verification/verify_floor/`: per-cell floor
  decomposition.
- **Adjacent: real depth intrinsics.** This unit's camera_info is fx = fy = 321.522 px (VFOV
  58.48°), against the 335.65 px (56.41°) sim models. That belongs to camera parity
  ([`depth-camera-vfov-parity`](../../completed/depth-camera-vfov-parity.md), note 2026-09-26), but
  a matched sim capture must pick one.

## Out of scope

- Re-measuring the sensor. Add one still pose with several surfaces at 2.5–3.5 m only if item 1's
  margin matters to the decision.
- The stream defects on the Jetson
  ([`d555-l4t-stream-integrity`](../reliability/d555-l4t-stream-integrity.md)).
