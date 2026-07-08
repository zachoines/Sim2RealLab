# Fix the policy-camera vertical FOV to match the real D555 sensor

**Type:** task / bugfix (train↔deploy geometry parity)
**Owner:** DGX (`strafer_shared` + `strafer_lab`). The deploy-side downsample
change is a parallel Jetson follow-up (see **Operator hand-off**).
**Priority:** P1 — the whole DEPTH policy family trains on a vertically
magnified depth image versus what deployment feeds it; this is the suspected
root cause of the observed retreat/wiggle. Gates any DEPTH retrain.
**Estimate:** M (implementation + Kit probe gate; the ~16 h retrain is the
operator's).
**Branch:** task/depth-camera-vfov-parity

## Story

As a **DEPTH-family policy about to be trained and deployed against a real
D555**, I need **the sim policy camera to render the same vertical field of
view the real sensor spans**, so that **the depth image I learn from in
training is the same one I am given at deployment — not a 1.26× vertically
magnified version that makes obstacles look closer and taller than they are.**

## The bug (diagnosed by the Jetson handoff, verified by the probe)

The sim **policy camera** rendered at **80×60 (4:3)** and the **perception
camera** (which mirrors the real D555 stream) at **640×360 (16:9)**. Isaac Sim
derives a camera's vertical FOV from its render **resolution aspect ratio**
(square pixels): 80×60 → **71.1°**, 640×360 → **56.4°**. Deployment
block-averages the 640×360 depth down to the policy grid before the policy
reads it, so the policy **trained on 71.1° depth but deployed on 56.4° depth**
— a **1.26× vertical magnification** (71.1/56.4). Phantom obstruction →
retreat/wiggle. Affected the whole DEPTH family.

## The design decision: 80×45, not `vertical_aperture`

The first attempt (per the original brief) was to set
`PinholeCameraCfg.vertical_aperture` on the policy camera to the real sensor's
implied value, keeping 80×60. **The Kit probe disproved this:** Isaac Sim / RTX
(and Isaac Lab's own `Camera.data.intrinsic_matrices`) derive vertical FOV /
`fy` from the render **resolution**, and **ignore the authored
`vertical_aperture`**. Evidence (probe, GB10, headless):

- The USD `verticalAperture` on two 80×60 policy prims differed (2.07 vs 2.76
  mm) but their sensor `fy` was **identical** (41.96 = 80·1.93/3.68, the
  resolution-derived value) and their **rendered depth was byte-identical**.
- A 16:9 perception camera with the *same* `vertical_aperture` rendered 56°
  while an 80×60 rendered 71° — so VFOV tracks **resolution**, not aperture.

So the only lever on the sim policy camera's vertical FOV is its **resolution**.
The fix makes the policy camera **80×45 (16:9)**, whose resolution-derived VFOV
is `2·atan((3.68·45/80)/(2·1.93))` = **56.4°** — the real sensor's. Both
cameras are now 16:9, so the deploy downsample 640×360 → 80×45 is a clean 8×
block-average that preserves the vertical FOV (it was a 16:9→4:3 *squash*
before).

Full probe evidence + the disproven-`vertical_aperture` writeup lives in
`DEPTH_VFOV_FIX_PROBE_FINDINGS.md` (repo-external coordinator doc).

## What changed (this PR — `strafer_shared` + `strafer_lab`)

- `strafer_shared/constants.py`: `DEPTH_HEIGHT 60 → 45` (comment carries the
  probe finding + the 16:9/56.4° derivation).
- Obs dims are **derived**, never re-literal: `policy_interface._DEPTH_FIELDS`
  depth dim = `DEPTH_WIDTH*DEPTH_HEIGHT` (3600); `PolicyVariant.DEPTH` /
  `.DEPTH_SUBGOAL` `obs_dim` = 3619 (was 4819) via the existing prefix-slice
  construction — one edit, both variants.
- Encoders resolution-follow: `depth_rnn_model` (`_DEFAULT_DEPTH_OBS_DIM`, the
  ONNX-preprocess reshape), `depth_encoders` (both `view()`s + the CNN
  `SpatialSoftArgmax` grid, now derived from the conv stack via
  `_conv_stack_feature_hw` → 6×10 at 45 rows vs 8×10 at 60). **The DeFM
  backbone is unaffected** — it resizes any input to 224×224, so it never sees
  45 rows; only the reshape changes.
- Consumers derived: `lerobot_writer` (`_POLICY_RES`), `collect_demos`,
  `bridge_harness_smoke`; all stale "80x60" policy-cam descriptions updated.
- `d555_cfg`: no aperture is set — the 16:9 resolution does the work (the
  perception camera is untouched).

## The h264 even-dimension wrinkle (flagged for review)

80×45 has an **odd height**, and h264/libx264 requires even frame dims. The
depth-policy obs rides as a 16UC1 PNG sidecar (no even-dim constraint, stays
exactly 80×45), but the **optional rgb-policy debug video** must encode. This
PR bottom-pads that debug video one row to 80×46 (`_POLICY_VIDEO_RES` in
`lerobot_writer`); **the obs the policy consumes is untouched**. The
alternative — dropping the rgb-policy video entirely — is a bigger cross-file
change; the pad keeps the harness feature-complete. Easy to swap to "drop it"
if preferred.

## Gate

**Probe gate — satisfied.** The one-time Kit probe (retired in this PR;
methodology folded into the appendix below) characterized the fix and PASSed:
measured VFOV **fixed (80×45) = 56.41°** vs **control (80×60) = 71.13°** — the
resolution change moved the geometry to the real sensor's ~56.4° — with
fixed-vs-reference row-parity **max 0.099 m / mean 0.006 m** (in-range rows
match to <0.01 m). PASS evidence is in the PR body.

**Standing regression guard:** the Kit-free aspect-parity test
`tests/navigation/test_depth_camera_aspect.py` — the policy and perception
cameras must share an aspect ratio, because Isaac renders resolution-derived
square-pixel FOV (the invariant the probe discovered). That assert IS the
geometry gate going forward; the probe script was a fossil (its control arm
fabricates the retired 80×60 camera, and an un-run Kit script is false comfort).

Unit gate: `make test-lab` (pure suite green + the depth-reward geometry
recalibrated for 45 rows + all obs-dim fixtures derived).

## Operator hand-off

1. Retrain DEPTH_SUBGOAL on the 80×45 camera (same env otherwise — one
   variable) → export `strafer_depth_subgoal_v1`
   (`--env Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0`).
2. **Jetson downsample follow-up** (parallel, `DEPTH_8045_JETSON_PROMPT.md`):
   the deploy path must downsample 640×360 → 80×**45**. The shared
   `depth_downsampler` already derives from `DEPTH_WIDTH/DEPTH_HEIGHT`, so it
   auto-adapts; the follow-up covers the re-export + any strafer_ros tests that
   still assert 4819/4800. **Merge order: this PR first** (it publishes the
   constants the follow-up consumes). Training needs only this PR; **v1
   deployment** needs both.
3. The live-mission attempt runs with the RC2 depth-freshness gate in place
   (separate Jetson PR).

## Artifact staleness (expected)

The existing DEPTH-direct artifact and `strafer_depth_subgoal_v0` are now BOTH
geometry- *and* dimension-stale: v0's 4819-dim / 60-row obs won't load against
the 3619-dim / 45-row contract (it becomes unloadable — that's what the sidecar
variant/dim check is for; versioning exists for exactly this). Retrain when the
lane next needs it.

## Ship discipline

No trailers / transient refs / env names. Resolve operator inline comments
before merge. Report test counts + the probe's before/after evidence in the PR
body. Brief stays **active** — it closes when the operator's retrain + live
validation land.

## Appendix — probe methodology (for re-characterizing on an Isaac version bump)

The probe that produced the gate evidence was deleted (its control arm
fabricates the retired 80×60 camera; an un-run Kit script is false comfort). The
durable protection is the aspect-parity invariant it found. If a future Isaac
Sim / RTX version changes how camera apertures are honored, re-characterize in
~an hour by reconstructing it:

- **Setup:** three cameras at one identical pose (level, forward) over a flat
  ground plane. `ref` = perception 640×360 block-averaged onto the policy grid
  (the deploy input); `fixed` = the policy camera; `control` = an 80×60 camera
  (the pre-fix 4:3 resolution, before/after evidence).
- **Signal:** `distance_to_image_plane` over a floor (a surface *parallel* to
  the optical axis) varies row-to-row purely with the vertical FOV, so a
  row-wise depth profile is the VFOV fingerprint. A frontal wall does NOT work —
  a plane *perpendicular* to the optical axis has one constant image-plane
  distance, so its profile is flat regardless of VFOV.
- **Clip before comparing:** clip depth to `DEPTH_MAX` (6 m — the D555
  saturation the obs pipeline enforces) *before* the row comparison. Near the
  horizon, floor depth → ∞, where any sub-pixel sampling difference explodes
  into tens of metres and swamps the signal; clipping restores the in-range
  comparison the policy actually sees.
- **Tolerance:** row-tol 0.10 m — within the real D555's depth noise at range
  (~2 % at 6 m) and ~10× tighter than a real VFOV mismatch would produce
  (meter-scale across many rows). Also gate the *mean* delta (≤0.02 m) to catch
  a systematic shift no single row trips.
- **Measured VFOV (self-proving):** for a level camera at height `h`, the bottom
  floor row (`y_ndc = (H-1)/H`) reads depth `d = h / (y_ndc · tan(vfov/2))`, so
  `vfov = 2·atan(h / (d · y_ndc))` — read the effective FOV directly off the
  rendered profile (the probe measured 56.41° for 80×45, 71.13° for 80×60).

**Disproven — do not re-propose `vertical_aperture`.** Two 80×60 prims with
different authored `verticalAperture` (2.07 vs 2.76 mm) had identical sensor
`fy` (41.96) and byte-identical rendered depth; a 16:9 camera with the *same*
aperture rendered 56° while an 80×60 rendered 71°. VFOV tracks **resolution**,
not the authored aperture — which is why the fix is a resolution change (80×45,
16:9) and the standing guard is aspect parity.
