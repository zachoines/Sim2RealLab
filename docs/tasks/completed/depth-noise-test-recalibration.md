# Re-derive the depth-noise variance tests' analytic yardstick

**Status:** Shipped 2026-07-18 in `1fd41ac` (DGX). Kit `depth_noise` 6/6 +
`noise_models` 55/55 green on the branch; pure `tests/` 770 passed / 1 skipped
(+5 new). Archaeology on `41bfa6d^` confirms both tests passed at 80×60 on the
old mega-df — the yardstick moved (RNG realization / pixel count), not the model.
**PR:** https://github.com/zachoines/Sim2RealLab/pull/153

**Type:** task / test-correctness (certifier for the depth retrain)
**Owner:** DGX (`source/strafer_lab/test_sim/sensors/depth_noise/`)
**Priority:** P2
**Estimate:** S — pure statistics + comments; the Kit gate + archaeology run are the bulk of the wall-clock.
**Branch:** task/depth-noise-test-recal

## Story

As the **owner of the depth sim-to-real pipeline about to launch a depth
retrain**, I want **the depth-noise integration tests to pass on a
derivation-backed yardstick rather than a fixed-seed realization**, so that
**the noise-model certifier is green for the right reason and recalibrates by
arithmetic on the next resolution change instead of flaking on a hair-width.**

## Context bundle

Read these before starting:
- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

**The fault.** Two seeded Kit integration tests failed on `main` by
hair-widths: `test_hole_variance_on_wall_pixels` (variance ratio 0.9985, ~0.03%
below its 95% CI) and `test_fresh_frame_variance` (ratio 1.000413, ~0.007%
above). The sibling `test_gaussian_variance_at_wall_distance` passed on the same
machinery. The noise models themselves behave — the rate / detection /
zero-mean tests all pass.

**Root cause — the CI's effective sample size, not the model.** Both tests build
their chi-squared variance-ratio CI in `variance_ratio_test` purely from
`n_samples`, and both passed `n_samples = pooled (pixels x timesteps)
first-difference count` — 5.1M (holes) / 63.3M (frame drops). At that df the 95%
CI half-width is ±0.12% / ±0.035%, so a small, *systematic* (not random)
render-vs-analytic residual reads as a model mismatch. The pooled count wildly
over-counts independent evidence: (a) temporal first-differences
`d[t]=y[t]-y[t-1]` have lag-1 autocorrelation −0.5, so timesteps are not
independent draws, and (b) the residual under test is a per-pixel-structured
term (fixed geometry + RTX render precision) that extra frames re-measure rather
than average away. `test_gaussian` survived the same over-precise df only
because its `expected_var` is computed per-sample from the identical formula the
model applies, pinning its ratio at ~1.0000 — it sat on the same fragile df and
was the next hair-width failure on any reseed.

**Why it surfaced now (the archaeology).** The depth policy resolution changed
80×60 → 80×45 (16:9) in `41bfa6d` (#143). The depth-noise test scene reads
`distance_to_image_plane` (perpendicular depth) off a fronto-parallel wall at a
fixed 2.0 m, so **every wall pixel reads 2.0 m regardless of vertical FOV** — the
analytic expected variances (holes `2·p·(1−p)·(1−1/3)² = 0.042222`; fresh
`(1−p_drop)·2·(z²·σ_d/(f·B)/max_range)² = 1.2525e-6`) are **resolution-invariant**.
What actually moved 80×60 → 80×45 was the wall-pixel count and the fixed-seed
RNG realization:

- **`test_fresh_frame_variance` (uncapped):** wall pixels/env 2960 → 2400, so
  `n_samples` and the CI width both shifted; the systematic residual exceeded
  even the (slightly wider) 80×45 CI.
- **`test_hole_variance` (capped at 200 wall px/env):** `n_samples` is
  *identical* at both resolutions (the cap binds — 2400 and 2960 both > 200), so
  its CI never moved. Its red-flip was **purely the RNG realization**: the
  per-frame element count (4800 → 3600 px) changes the seed-42 draw sequence and
  the `randperm` subsample, nudging the ratio from just-inside to 0.9985. (This
  refines the coordinator's "the CI moved with `n_samples`" story — true for the
  fresh test, but the hole test's CI was already resolution-stable.)

The pre-#143 tree (`41bfa6d^` = `f05da0f`) was run as an archaeology check: both
failing tests **passed** there, on the same code — confirming the yardstick moved
(realization / pixel count), not the model. See the PR body for that run's
output.

**The fix — spatial degrees of freedom.** The honest independent unit is the
wall PIXEL, not the pooled pixel×timestep diff: holes and stereo gaussian are
drawn IID per pixel, so the pixel count is a deliberately **conservative ceiling
on independent information**. A new thin wrapper `variance_ratio_test_spatial`
(in `test_sim/sensors/depth_noise/utils.py`) forwards to the shared
`variance_ratio_test` with `df = total_wall_pixels − 1`; each of the three
variance tests passes its live `total_wall_pixels` instead of `n_samples`. At
current geometry that is ±2.45% (holes, capped 12 800 px) / ±0.79% (fresh /
gaussian, ~158 720 px) — both failing ratios land well inside, with the CI a
closed-form function of the live pixel count so a future resolution change
recalibrates by arithmetic, no seed or tolerance edit.

**What deliberately did NOT change.** `variance_ratio_test` and
`noise_models.py` are untouched (the noise_models suite's callers are
unaffected — the wrapper only forwards). Seeds are untouched. The
`MAX_WALL_PIXELS_PER_ENV=200` subsample in `test_holes` is **kept** so
`measured_var` is byte-identical to the pre-fix realization (the ratio stays the
validated 0.9985) — its comment is corrected (it now bounds point-estimate cost,
not the CI). Removing the cap would tighten the hole CI to ±0.79% (≈3× the
regression-detection power) but changes the analyzed pixel set / `measured_var`,
so it is left as an optional future tightening that needs its own Kit re-verify;
it was not taken here because the brief scopes the change to "only the
expectations/CIs become derivation-backed; determinism stays."

**Honest caveat (in the derivation comment, per the brief).** `df = N_pixels` is
a conservative independent-info ceiling, not a claim that it is exactly the
estimator's sampling df. If the wall pixels are near-perfectly homoscedastic the
residual is a small fixed render-vs-analytic bias; the wide spatial-df CI
correctly declines to flag a sub-0.2% gap between a stochastic RTX pipeline and
an idealized analytic formula — which is what an integration test of that
pipeline should do, while the rate / detection / zero-mean tests guard the model
behaviour itself.

**A Kit-free certifier.** New pure test
`tests/navigation/test_depth_noise_resolution_sensitivity.py` (imports only
installed constants + numpy/scipy, no Isaac Sim) pins the two invariants by
arithmetic: (1) the expected variances are resolution-invariant while the VFOV
genuinely moves (56.4° @80×45 vs 71.1° @80×60, reproducing the #143 probe), and
(2) the pixel-df CI admits both observed residuals while the old pixel×timestep
CI rejects them — so the next resolution change recalibrates by formula, and a
regression back to temporal df is caught here without a Kit boot.

## Acceptance criteria

- [x] `test_hole_variance_on_wall_pixels`, `test_fresh_frame_variance`, and
  `test_gaussian_variance_at_wall_distance` green on the branch via the Kit
  recipe; ratios reported in the PR body with their new spatial-df CIs.
- [x] The chi-squared CI is built from `total_wall_pixels`, with a derivation
  comment stating `df = N_wall_pixels − 1`, `CI ~ 1.96·sqrt(2/df)`, and the
  resolution-invariance of the expectation — so recalibration is arithmetic.
- [x] `variance_ratio_test` and `noise_models.py` untouched; seeds untouched; the
  noise_models Kit suite unaffected (wrapper forwards only) — confirmed 55/55
  green on the branch.
- [x] Kit-free resolution-sensitivity test reproduces the current AND pre-change
  expected variances (identical) and VFOVs (56.4° / 71.1°) from their
  geometries, and asserts pixel-df admits / temporal-df rejects the residuals.
- [x] Pure `tests/` suite untouched-and-green (counts in the PR body).
- [x] Archaeology: the two failing tests pass on `41bfa6d^` (pre-80×45),
  evidencing a moved yardstick; result in the PR body.
- [x] Lands BEFORE any depth retrain — it is that work's certifier.
