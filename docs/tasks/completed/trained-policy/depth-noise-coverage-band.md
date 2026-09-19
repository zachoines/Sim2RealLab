# Put per-pixel-featureless depth in the training distribution, and measure it

**Status:** Shipped 2026-09-18 in `d27e17b` (gx10-d1d8).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/221

**Type:** task (training-side mechanism plus the measurement that sizes it)
**Owner:** DGX
**Priority:** P1 — the next rig gate is sim-bridge again, and until the training
distribution contains depth as featureless as the bridge's, that gate cannot
distinguish navigation from a habit of reading texture presence.
**Estimate:** M (one measurement sitting plus a small noise-model change)
**Branch:** `task/depth-noise-coverage-band`

## Story

As the **depth-subgoal retrain**, I need **a training distribution that contains
depth with as little per-pixel texture as the deploy path delivers**, so that
**a policy trained on it cannot pass or fail the next gate on whether texture is
present rather than on where the obstacles are.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [context/env-composition-contract.md](../../context/env-composition-contract.md)
  — the composition goldens a new noise-model field moves.
- [measurements/noise-texture-parity-2026-09-17](../../../measurements/noise-texture-parity-2026-09-17/README.md)
  — establishes that every depth inference on record is bridge depth, and that
  the v2 artifact's rig-class command is its response to a featureless far field.

## Context

Training injects σ_z = z²·σ_d/(f·B) i.i.d. **per 80×45 policy pixel** with no
reduction. The deploy path applies the sensor's error at 640×360 and then
collapses every 8×8 block with a median, so a deploy frame arrives with next to
no per-pixel texture: measured over the 1 799-frame bridge capture, its
high-pass p95 against a pose-matched clean render is 0.000524 (normalised),
against 0.00565 for the shipped realistic tier — an order of magnitude.

That gap is not, by itself, a calibration error. Whether the *real* sensor is
noisier or smoother than training after the block median turns on the
within-block correlation ρ, which
[`real-d555-depth-texture-capture`](../../active/trained-policy/real-d555-depth-texture-capture.md) exists
to measure and which no capture has measured yet. **This brief changes no
shipped σ_d.**

It addresses a narrower and separable problem: **coverage**. The next rig gate
runs against the bridge again, so its depth field is exactly the capture's. If
the training distribution contains no frames that featureless, a retrained
policy can key on texture presence exactly as v2 did, and the gate measures the
habit rather than the navigation. A per-env band on σ_d reaching down to zero
puts that region in the distribution without moving the tier's centre.

The comparison is a texture statistic measured on both sides with no policy in
the loop — the p95 of `|d − median3×3(d)|` with the near-field-fill class
excluded, which is the statistic the 2026-09-17 record's structure table already
reports.

## Acceptance criteria

- [x] One numpy-only texture statistic in `strafer_shared`, importable by both
      lanes, reproducing the 2026-09-17 deposit's `texture_structure.json`
      figures for the bridge capture and the realistic tier.
- [x] A CLI that reads a node capture, a gym capture or an `.npz` and prints the
      statistic per depth band with n per band.
- [x] Tests that are mutation-proven: a swapped percentile, a dropped near-field
      exclusion and a wrong border rule each fail a named test.
- [x] The σ_d coverage curve measured through the production `DepthNoiseModel`
      on the 30 anchor frames, with σ_d\* reported against the capture.
- [x] The statistic at σ_d = 0 with the rest of a tier active, reported, with the
      term carrying any residue named.
- [x] `disparity_noise_px_range` on `DepthNoiseModelCfg` and the contract-level
      depth camera cfg, drawn per env at reset, `None` restoring current
      behaviour bit-identically and leaving `torch.get_rng_state()` unchanged.
- [x] A band pinned to either end reproduces the fixed-σ output for that σ_d.
- [x] A partial reset redraws only the envs it names.
- [x] The composition-contract preimage diff names exactly the new field; the
      layout goldens hold.
- [x] The retrain's DR spec puts ≥ 10 % of drawn envs within 2× of the capture's
      statistic. A uniform draw missed it — 6.05 % on the robust tier's
      [0, 0.16] — so the **law** changed rather than the mechanism: the band is
      drawn log-uniformly, which reads **30.96 %** on [0.002, 0.16] and 37.21 %
      on [0.002, 0.08]. Both ends must be positive.
- [x] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [x] No regression in the workflows the touched code supports: the pure suite
      (1308 passed, 1 skipped) and the four-target contract gate (222 passed).

## Investigation pointers

- `source/strafer_shared/strafer_shared/depth_texture.py` — the statistic.
- `source/strafer_lab/scripts/measure_depth_texture.py` — the CLI.
- `source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py:522-545`
  — where the per-env coefficient is drawn, alongside `DelayBuffer`'s latency
  band, which is the pattern it follows.
- `source/strafer_lab/tests/contracts/test_depth_texture_statistic.py` and
  `source/strafer_lab/tests/navigation/test_depth_disparity_band_dr.py`.
- [`depth-noise-coverage-2026-09-18`](../../../measurements/depth-noise-coverage-2026-09-18/README.md)
  — the coverage curve, the band's numbers, and the DR-spec proposal.

## Out of scope

- The retrain itself, and turning the band on in any shipped contract tier.
- Any change to the tiers' `disparity_noise_px`, `hole_probability`, or the
  near-field conventions reconciled on 2026-09-13.
- Calibrating σ_d against the real sensor — that needs ρ, and is owned by
  [`real-d555-depth-texture-capture`](../../active/trained-policy/real-d555-depth-texture-capture.md).
- Rendering the policy camera at 640×360 and sharing the deploy reduction; its
  cost is re-measured in the 2026-09-18 record, not implemented.
- Using any trained artifact as a pass/fail gate on a training distribution.
