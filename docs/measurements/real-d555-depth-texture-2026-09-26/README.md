# Real D555 depth texture either side of the block reduction, at robot mount height, 2026-09-26

This record measures the real D555's per-pixel temporal noise before and after the 8×8 block
median the policy's observation goes through. It gives the within-block correlation ρ per range
band, which
[`noise-texture-parity-2026-09-17`](../noise-texture-parity-2026-09-17/README.md) §6 needed and
nothing had measured.

The capture: five static recordings from one robot placement on a hardwood dining-room floor,
lens 0.278 m above the floor and level within 1°. Filters were read back off before every
recording.

**ρ.**
- In the headline set (poses 1, 2, 3), ρ is 0.64–0.90 in every band, on every estimator. The repeats
  (set B) read 0.63–0.88.
- Both ranges are above both the 2026-09-17 crossovers and this capture's own. The one reading below a
  crossover anywhere is set C's 2.5–3.5 m ratio of band medians: 0.351 on 20 cells (§2).
- So the sign that §6 left undetermined is settled for the surfaces measured. At the shipped
  `disparity_noise_px = 0.08`, the real post-reduction σ is larger than the σ training injects, in
  every band, by 1.11–1.88×.

**Gate (B)** (§8) **passes in all five bands at 0.08**, so the brief's outcome is **no code
change**. It comes with three limits:
- **2.5–3.5 m is a marginal pass,** 0.531 against a 0.5 floor. It rests on one flat box face, in
  recordings that fail the capture's own stillness rule.
- **The registered survival rule keeps almost none of the glossy floor.** On the floor the deployed
  output is noisier than on walls at the same depth. Its descriptive ratios fall to 0.33–0.49, or
  0.55–0.63 once validity-mask flicker is removed.
- **The robust training tier's σ_d range mostly sits below the passing interval.**

**Revisit trigger.** The trigger of the parked 640×360-render option is **met**. Its render half
has already shipped; the native-resolution noise half stays parked, now on the measured ρ.

**Brief.** This closes
[`real-d555-depth-texture-capture`](../../tasks/completed/real-d555-depth-texture-capture.md),
with one method deviation, set out below: two of the five recordings count as still under the rule
registered during the capture.

## Setup

- **Rig.** For this capture the Jetson was moved next to the robot on longer Ethernet runs, so the
  robot could stand on the floor with the camera on its mount. The camera stayed on its 6-inch USB
  cord at 5000 Mbit/s.
  - During poses 1 and 2 the Jetson lay on top of the robot with its fan running, and the house
    HVAC was on.
  - For poses 1b, 3 and 2b it was on the floor behind the camera, and the HVAC was off.
- **Camera.** D555 `409122301816`, firmware 7.56.19918.835, `realsense2_camera` 4.58.4, images
  `de3a865e5810`.
- **Read-back.** `ros2 param get /d555` was run immediately before every bag:
  - the four post-processing filters (plus the disparity filter and HDR merge) read `False`;
  - depth auto-exposure reads `True`;
  - the profile is `640x360x30` Z16.

  The five read-back files are byte-identical (sha256 `3ba667ef…`), as are the pilot's and the dry
  run's, because `pose_capture.sh` writes no timestamp into them. Per-bag timing rests on the
  script's step order and each recording's `t_start_utc.txt`, `t_end_utc.txt` and `bag_info.txt`.
- **Geometry, measured from the data.** A plane fit of 1/z on the bottom 60 rows of each
  recording's temporal median gives the lens 0.2781–0.2791 m above the floor, pitch −0.83° to
  −0.92° (axis up) and roll +0.18° to +0.19°, in every pose.
  - Training's camera sits at about 0.298 m, level, with ±3° mount jitter on the robust tier.
  - The real depth intrinsics are fx = fy = 321.522 px, cx = 318.441, cy = 180.322 (camera_info).
- **Scenes.** Five recordings, all from the same placement looking along the dining room into the
  living room:

  | recording | scene | floor visible |
  |---|---|---|
  | pose 1 | the long view | 0.51 m out to 5.5 m, through the doorway |
  | pose 2 | pose 1 plus a large flat cardboard box at 3.01 m, square-on | |
  | pose 3 | the box at 0.709 m, filling most of the frame | |
  | pose 1b | repeat of pose 1 under the changed conditions | |
  | pose 2b | repeat of pose 2 under the changed conditions | |

  Screens were off, and it was night, so there was no sun.
- **Stream handling.** On this host every depth frame is published twice (see
  [`real-d555-hardware-readback-2026-09-25`](../real-d555-hardware-readback-2026-09-25/README.md)).
  The tools therefore drop each frame that is bit-identical to the one before it, then skip 60.
  Timing is on bag receive time, because the header stamps do not advance.
  - Each recording keeps 1062–1131 distinct frames over 37.4–37.7 s at 30.00 fps.
  - Pose 2 lost 69 frames to rosbag2's cache overflowing (its `record.log`: 137 lost messages).
    That is the recorder, not the sensor.
- **Analysis.**
  - **Primary:** `d555_texture_analysis.py` version 3, unchanged, the bytes the plan fixes.
  - **Secondary:** `d555_secondary_analysis.py`, reported separately.
  - Both import the production `decode_depth_image` and `downsample_depth` rather than
    reimplementing them.
  - Before this record was written, an independent recomputation reproduced every headline number
    to the last printed digit: frame counts, σ both sides, ρ, gate (B), and the invalid-block
    shares.

## What was registered, and when

`PREANALYSIS.md` is stamped 04:33:53Z, before the first bag. It fixes:
- the primary tool;
- the two secondary analyses;
- depth-only stillness (this host has no IMU).

Addenda were appended as the capture went and, by the plan's own account, never edited back:
1. **After pose 1:** a lattice-corrected stillness rule plus a coherent-motion check. A pose counts
   as still only if both pass.
2. **After pose 2:** its failure, the decision to re-record, and the changed conditions.
3. **After everything:** corrections, and what was not pre-registered.

**How the chronology is anchored.**
- **The plan.** Its times are the file's own stamps: 04:33:53Z, 04:48:45Z, 05:03:35Z, 05:08:05Z,
  05:11:48Z and 08:11:07Z. `PREANALYSIS.md` has one evidence-repo commit, `465ab9b` at 08:12:26Z,
  after every bag, so git cannot corroborate them.
- **The primary tool** is anchored in git: it is byte-identical to the copy committed in `7ff3f70`
  at 2026-09-26T03:24:51Z, before the first of the five recordings.
- **Addendum 2's last note** is stamped 05:11:48Z, the same second as pose 1b's `t_start_utc.txt`.
  The capture script writes that file before its read-back and 15 s wait, and the bag's first
  message follows 58.75 s later. At one-second resolution the note cannot be ordered against the
  script's start.

**Not pre-registered:** pose 3, pose 2b, the three pose sets, and the choice of set A as the
headline. Set A mixes the before and after conditions.
- set A: poses 1, 2, 3;
- set B: the repeats, 1b, 2b, 3;
- set C: poses 1, 3, the two that count as still.

**All three sets give the same gate (B) verdict and the same side of every crossover,** so no
conclusion here depends on the choice. The one exception is a 20-cell estimator in set C, noted
below.

## Stillness

| pose | primary rule (1 mm step) | lattice-corrected (≤ 1 %) | coherent motion | counts as still |
|---|---|---|---|---|
| pose 1 | 4.75 % DRIFTED | 0.91 % | pass | yes (under addendum 1, written after it) |
| pose 2 | 6.56 % DRIFTED | **1.77 %** | pass | no |
| pose 3 | 0.92 % STILL | 0.40 % | pass | yes |
| pose 1b | 5.71 % DRIFTED | **1.22 %** | pass | no |
| pose 2b | 6.61 % DRIFTED | **1.79 %** | pass | no |

**How the rules work.**
- The primary rule compares each always-valid pixel's median over the first and last thirds of the
  recording. It allows one 1 mm Z16 step.
- The real depth sits on a disparity lattice, C = 997–1030 m, whose step is z²/C: about 34 mm at 5.8 m (33.7 / 32.7 mm at C = 997 / 1030 m).
  The lattice-corrected rule allows one lattice step instead.

**Poses 2, 1b and 2b stay "not still" under the rule as registered.** Their failures still carry no
motion signature:
- **No camera motion.** Between thirds the floor plane moves by at most 0.012 mrad and 0.006 mm.
  Per-frame floor fluctuations are uncorrelated between the left and right halves of the frame,
  which bounds rigid per-frame pitch at about 0.01 mrad.
- **Common mode.** Frame-global common mode is 0.002 px, Jetson on the robot or off.
- **Mechanism.**
  - About 90 % of the flags outside the box, and 78 % on it, are pixels whose median hops across a
    Z16 code the camera never emits. The median moves about 6 mm while the pixel's mean moves about
    0.6 mm.
  - A separate slow component appears only on the cardboard faces: autocorrelation +0.04 to +0.07
    out to about 2 s, in patches up to about 16 px.
  - For poses 2 and 2b the 1 % threshold sits at their own false-flag level: a 3 s
    block-permutation null exceeds it in 97 % and 95 % of random splits. For poses 1, 1b and 3 the
    null never exceeds it (medians 0.80 / 0.81 / 0.20 %).
  - In all five poses the first-vs-last split beats every random split, so slow structure beyond
    3 s is common to all of them.
  - Pose 1b crosses 1 % because it has slightly more of that very-slow, spatially incoherent wander
    than pose 1 (30 s structure function 0.0795 against 0.0730 σ²). The skipped codes turn it into
    median hops: 4.3 % of the pixels beside a skipped code flag, against 3.0 % in pose 1.
  - No coherent component appears.
- **Effect on the numbers.** If all of the slow component were motion, post-reduction σ would be
  overstated by at most 3.5 % in any band. That would raise, not lower, the one marginal gate (B)
  ratio.

## 1. σ either side of the reduction

**Survivorship.**
- A raw 640×360 pixel counts if it is nonzero in every analysed frame. It is binned by its temporal
  median, and σ is its standard deviation over frames (ddof 1).
- An 80×45 cell counts if all 64 of its pixels are nonzero in every frame and its production
  output is never the near fill. It is binned by the temporal median of that output.
- Poses pool by concatenation.

Set A:

| band m | raw n | raw σ p50 / p90 mm | 2026-08-04 raw p50 mm | cells n | post σ p50 / p90 mm | median cell depth m |
|---|---:|---|---:|---:|---|---:|
| 0.4–1.0 | 270 301 | 0.84 / 1.77 | 1.1 | 2342 | 0.700 / 1.087 | 0.709 |
| 1.0–1.5 | 104 227 | 2.98 / 5.86 | 2.7 | 842 | 2.396 / 4.628 | 1.296 |
| 1.5–2.5 | 97 790 | 6.36 / 12.28 | 10.1 | 1137 | 5.397 / 10.205 | 1.926 |
| 2.5–3.5 | 19 090 | 23.51 / 34.16 | 15.5 | 249 | 21.282 / 28.769 | 3.004 |
| 3.5–5.5 | 22 737 | 44.75 / 85.87 | 87.1 | 319 | 35.986 / 77.886 | 4.628 |

**Where the cells come from.**
- 0.4–1.0 m is mostly pose 3's box face: 1901 of 2342 cells. Without pose 3 it reads 1.038 at
  gate (B), still a pass.
- 2.5–3.5 m is 229 cells from pose 2, of which 206 lie within 3.01 ± 0.08 m (the box face's depth),
  plus 20 from pose 1.
- Pose 3 contributes nothing beyond 1.5 m.

**Sets B and C** agree to within a few percent. The exception is set C's 2.5–3.5 m band: without
the box it holds 20 cells, reading 10.4 mm.

**Against 2026-08-04.** This capture reads lower raw σ than the 2026-08-04 table in three of five
bands, with the filters verified off. The scenes and heights differ, so it does not show what the
2026-08-04 filters were, and it cannot bear on that capture's own true raw σ. What it does not
support is carrying that table to other scenes as a lower bound on the sensor's per-band raw σ. The
2026-09-18 per-capture bound stands.

## 2. ρ per band

ρ = (r² − π/128) / (1 − π/128), where r = σ_post / σ_raw, π/128 = 0.02454 and
1 − π/128 = 0.97546.

| band m | ratio of band medians: r → ρ | flat cells only, per-cell ratio (n) | 2026-09-17 crossover | this capture's crossover | side |
|---|---|---|---:|---:|---|
| 0.4–1.0 | 0.8343 → (0.6960 − 0.0245)/0.9755 = **0.688** | 0.789 (1687) | 0.293 | 0.522 | above both |
| 1.0–1.5 | 0.8040 → **0.637** | 0.795 (30) | 0.512 | 0.416 | above both |
| 1.5–2.5 | 0.8488 → **0.713** | 0.821 (401) | 0.227 | 0.610 | above both |
| 2.5–3.5 | 0.9051 → **0.815** | 0.831 (191) | 0.516 | 0.210 | above both |
| 3.5–5.5 | 0.8041 → **0.638** | 0.897 (36) | 0.062 | 0.303 | above both |

**Robustness.**
- The median of per-cell ratios (0.81–0.89) and the member-pixel ratio (0.68–0.84) are also above
  both crossovers in every band.
- Set B gives 0.691 / 0.629 / 0.769 / 0.862 / 0.700.
- Set C agrees, except for its 2.5–3.5 m ratio of band medians: 0.351 on 20 cells.

**Caveats.**
- **0.4–1.0 m is quantisation-limited.** Z16's Δ²/12 is 12 % of σ²; the Z16-corrected ρ is
  0.66–0.81.
- **Flat-cell counts are small** at 1.0–1.5 m (30) and 3.5–5.5 m (36).
- **Read ρ as the median's effective attenuation,** not as a pure within-block correlation. The
  real noise is also correlated *across* block edges, and anisotropically. Adjacent cells' temporal
  residuals correlate at a median of 0.35–0.69 horizontally, rising with range (about 0.68 at
  3.5–5.5 m). Vertically they sit at about 0.33–0.40 with no trend on pose-bands with at least 100
  pairs, and 0.26 on pose 3's near box.
  Per pose and band: `verification/verify_briefboxes/check_<pose>.json`. Training injects noise
  independently per cell.

## 3. The 80×45 texture statistic

These are per-frame values, median over all frames of set A, on the deployed output. The whole-frame
statistic is high-pass p95 = |d − median3×3(d)|.

| band m | cells/frame | frame hp p95 mm | exact-zero share | residual vs temporal median: hp p95 mm | exact-zero share |
|---|---:|---:|---:|---:|---:|
| 0.4–1.0 | 1632 | 6.5 | 0.545 | 3.5 | 0.293 |
| 1.0–1.5 | 638 | 45.5 | 0.477 | 10.0 | 0.224 |
| 1.5–2.5 | 576 | 54.3 | 0.477 | 19.0 | 0.245 |
| 2.5–3.5 | 103 | 222 | 0.400 | 48.5 | 0.200 |
| 3.5–5.5 | 133 | 383 | 0.565 | 134.6 | 0.224 |

- The whole-frame column carries scene edges. The residual column isolates temporal noise.
- Exact-zero shares include Z16's tie rate.
- **No training-side frames were captured,** so gate (C) is not evaluated here.

## 4. Invalid-block outcome

Since #232, the deploy path far-clamps **13.1 %** of cells per frame (p05–p95 9.6–16.6 %) and
near-fills **0.00 %**. That is identical, cell for cell, to the training convention.

**Before #232.** The old path would have written the 0.2 m near fill on **9.3 %** of cells, every
one of them a block with at least 32 of its 64 pixels invalid. None came from a genuine sub-0.4 m
return: the nearest floor is 0.51 m.

**At mount height, 9.4 % of blocks are at least half invalid,** against 33.9 % in the 2026-08-04
benchtop capture. The far clamp is 13.1 %, of which 9.3 % comes from those invalid blocks. The rest,
about 3.8 %, is genuine returns at or beyond 6 m, mostly in pose 1's long view (its far clamp is
16.4 %).

## 5. Gate (B) and the recommendation

The ratio is σ_z ÷ σ_post p50, with σ_z = z²·σ_d / (673 px × 0.095 m) evaluated at the band's
median cell depth. The window is [0.5, 2.0] below 3.5 m and [0.33, 3.0] above. Set A:

| band m | σ_d 0.08 (default) | 0.002 (robust low) | 0.16 (robust high) | 0.0179 (robust geometric mean) | σ_d that passes |
|---|---|---|---|---|---|
| 0.4–1.0 | **0.899 PASS** | 0.022 FAIL | 1.797 PASS | 0.201 FAIL | 0.045–0.178 |
| 1.0–1.5 | **0.876 PASS** | 0.022 FAIL | 1.753 PASS | 0.196 FAIL | 0.046–0.183 |
| 1.5–2.5 | **0.860 PASS** | 0.022 FAIL | 1.720 PASS | 0.192 FAIL | 0.047–0.186 |
| 2.5–3.5 | **0.531 PASS** | 0.013 FAIL | 1.061 PASS | 0.119 FAIL | 0.075–0.302 |
| 3.5–5.5 | **0.745 PASS** | 0.019 FAIL | 1.489 PASS | 0.167 FAIL | 0.035–0.322 |

The band-midpoint convention of §6 gives 0.876 / 0.816 / 0.927 / 0.529 / 0.704, and set C gives
0.906 / 0.884 / 0.864 / 0.847 / 0.749. Set B's figures are in the next paragraph.

**Recommendation: `disparity_noise_px = 0.08` passes gate (B); no code change.**
- **All five bands are inside the window,** in sets A, B (0.899 / 0.894 / 0.836 / 0.521 / 0.717)
  and C, and in both depth conventions.
- **The real sensor is the noisier side everywhere:** 1.11 / 1.14 / 1.16 / 1.88 / 1.34× training.
  None reaches the 2× at which §8(G) would call for an increase.
- **2.5–3.5 m is the closest** at 1.88×.
  - 229 of its 249 cells come from pose 2, which does not count as still. 206 of those lie within
    3.01 ± 0.08 m, the box face's depth.
  - The ratio holds across time windows (0.537–0.549) and on the re-record (0.521).
  - Resampling 32-px tiles puts its 2.5 % quantile at 0.50.
  - If that band matters to a decision, the one capture worth adding is a still pose with several
    surfaces at 2.5–3.5 m.

**The robust tier.** It is stated here for the noise decision, not as a recommendation. v3 trained
on the robust tier, which draws σ_d log-uniformly over [0.002, 0.16]. Against this capture:
- only draws in [0.075, 0.16] pass every band, about 17 % of the range;
- two thirds of draws fall below the window in every band;
- the median draw, 0.018, injects 5–8× less than the sensor delivers.

## 6. The floor, which the registered rule excludes

**Survivorship.** The glossy floor fills about 25 % of the frame: 951–974 cells per pose, or 515
where pose 3's box covers it. The registered cell rule keeps only 15–38 of them, because
grazing-angle floor pixels drop out intermittently. So everything above describes walls, cabinets
and the box, not the floor.

The secondary analysis's deployed-output arm keeps a cell whose deployed output is never
far-clamped and never near-filled. It keeps 94.8–95.6 % of floor cells.

**Floor versus other surfaces.** From the deployed output (secondary, descriptive, not a gate
reading), in set A:
- σ_post p50 on the floor is 1.159 / 3.683 / 9.218 mm at 0.4–1.0 / 1.0–1.5 / 1.5–2.5 m, on
  1976 / 290 / 48 cells.
- On other surfaces it is 0.723 / 2.841 / 5.957 mm.
- Normalised for depth, the floor is 1.9 / 1.5–1.6 / 2.2× the other surfaces.

**Where the excess comes from.** It has two sources.
- **Validity-mask flicker.** An invalid pixel enters the median as 6.0 m, so a changing count moves
  the block median along the floor's in-block gradient. That is 35–47 % of the floor's variance at
  0.4–1.0 m and up to about 30 % beyond.
- **The floor's own sensor noise.** Floor pixels valid in every frame have 1.4–1.6× the per-pixel σ
  of other surfaces at 0.65–1.0 m.

**With the flicker removed on both sides**, the depth-normalised floor excess is 1.45 / 1.32 / 1.65× by
band in set A, and 1.46 / 1.32 / 1.60 in set B. These are the floor row's `eq p50 within` over the
other row's, within-mask-state σ on both sides, from `verification/verify_floor/decomp.py`. That
script's summary line divides by the reflection-cleaned other surfaces' deployed σ instead.
- Across the deposited estimators it spans 1.4–1.6 / 1.3–1.5 / 1.6–1.9×, lowest at 1.0–1.5 m.
- The 1.5–2.5 m figure rests on 48–57 floor cells bunched near 1.55 m.

**What gate (B) would read on the floor.**
- With flicker included, 0.08 would read 0.455 / 0.489 / 0.327.
- On the flicker-free σ it would read 0.58–0.63 and 0.55–0.59 in the first two bands. That is
  inside the window, at its low end.
- Only the 48 floor cells near 1.55 m stay below 0.5 on sensor noise alone.

**Reflections.** The floor is glossy, and 78–93 cells per pose (2–2.5 % of the frame) see the
mirror image instead. They read 2 % or more beyond the floor plane: phantom extra depth in front of
furniture. Training's noise model, independent per cell and surface-blind, has no counterpart for
either effect.

## 7. The parked 640×360-render option's revisit trigger: met

The trigger, from
[`d555-invalid-pixel-statistics`](../../tasks/completed/d555-invalid-pixel-statistics.md), Out of
scope, says the option re-enters "if … the real-sensor reduction residual is materially larger than
sim's".

The residual is the per-band p95 of |block median − centre-2×2 ray| ÷ DEPTH_MAX. On the
temporal-median frame it is **5733 / 317 / 449 / 33 / 7.1×** the
[`deploy-resolution-depth-2026-09-19`](../deploy-resolution-depth-2026-09-19/README.md) §4 sim
figure, and at least 7.3× in every band of sets B and C.

"Materially" is undefined at source, but no reading of it survives 7× in every band. Two limits:
- the two figures are not scene-matched;
- the ratios below 2.5 m divide by near-zero sim values.

So the 33× and 7× at range carry the weight.

**Consequences.**
- **The render half has already shipped.** It rendered the policy camera at 640×360 and shared the
  deploy median, in `deploy-resolution-depth-2026-09-19`.
- **The native-resolution noise half stays parked, now on a measured reason.** The median
  attenuates i.i.d. native noise to 0.157 of its raw σ, where the real field keeps 0.80–0.91. So
  i.i.d. native injection would deliver 5–6× too little.
- **It re-enters only as a spatially correlated noise model,** and only if gate (C) fails once a
  matched sim capture exists.

## Adjacent findings

- **Depth intrinsics.**
  - This unit's factory depth calibration at 640×360 is fx = fy = 321.522 px, which makes the
    vertical FOV 58.48° and the horizontal FOV 89.73°.
  - Sim models 335.65 px (56.41° / 87.27°) from 1.93 / 3.68 mm and calls that the real VFOV.
  - The real focal length is 4.2 % shorter, and the principal point is off-centre by −1.56 / +0.32
    px.
  - This is one unit, one firmware, one profile.
- **Disparity lattice.** The depth values sit on a lattice with C = 997–1030 m (1/32 px of
  disparity at f·B ≈ 31 px·m). Z16's 1 mm is therefore not the only quantisation step: the lattice
  step is 1.7 / 3.6 / 9 / 21 mm at 1.3 / 1.9 / 3.0 / 4.6 m.
- **A trap in the stillness test.** Z16 codes the camera never emits make a pixel's median jump
  about two Z16 steps while its mean barely moves, so median-based stillness tests over-flag at
  range.

## What this record does not claim

- **A floor-inclusive gate (B).** The registered pass covers surfaces valid in every frame. The
  floor readings in §6 are descriptive.
- **Gate (C), or any texture parity.** No matched sim capture through the 640×360 perception camera
  was taken; §8 Gate 0 requires one before gate (C) or any training-side change.
- **Location diversity.** Every recording is from one placement in one room, on one floor material,
  at night.
- **A cause of the stillness failures beyond the mechanism in the stillness section,** or a
  re-classification of any pose.
- **Any change to a training parameter.** None was made.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `real-d555-depth-texture-2026-09-26/record-files/` |
| deposit commit | `465ab9b689ef6489f923477a40f00b748b4241da` |

The deposit holds:
- **`capture/`:**
  - `PREANALYSIS.md` with its addenda;
  - per recording: the read-back, notes, recorder logs, frame-number check and single-pose report;
  - the five depth bags, gzip-compressed and split. `DEPOSIT.md` gives both the stored and the
    uncompressed digests.
- **`analysis/`:** the primary tool on sets A, B and C, and the secondary tool.
- **`diagnostics/`:** the stillness and jitter checks run during the capture.
- **`verification/`:** the independent recomputation and adversarial checks, with their scripts and
  a JSON of every verdict.
- **`dryrun_20260925/`:** an unrecorded 2026-09-25 pipeline dry run, indicative only.
- **`hw/` and `tools/`:** node logs and stamps, and every tool with its self-test log.

The colour clips and every image derived from them show a private home. They stay on the host and
are listed in `DEPOSIT.md` by digest only; nothing here depends on them.

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/real-d555-depth-texture-2026-09-26/record-files/. \
      docs/measurements/real-d555-depth-texture-2026-09-26/
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/real-d555-depth-texture-2026-09-26/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
a089546c06d3401778296fccd26835adf7dfbae85034032b009ee466b28904d1  analysis/secondary.log
22966455e76c292d1f147f72e8eff61486e3ba01a5955cc52dea29a6ddbfa418  analysis/secondary/secondary_report.md
1f6e9c09eb1b4912c2c4da5698a7d54ffc5b4e03c287ce1b673b2504672c6317  analysis/secondary/secondary_summary.json
a38d931ff4470003c64e4d9cc2a28b19c453c5502cad92d6dfe73f4f5159a53b  analysis/setA_pose1_2_3/d555_texture_report.md
8826e85f5e8088e53416698ae52ab7436cdb407f3f2e671adac5ddf76deeee24  analysis/setA_pose1_2_3/d555_texture_summary.json
5ac16f8b3579f96eaca71ff639e34112266cdda4fcedcf47b3ebe4f29010bf3d  analysis/setA_pose1_2_3.log
ed17670caffed7b97466ccb360ed30882c33166f3f249ca2cdf40a96253e3003  analysis/setB_pose1b_2b_3/d555_texture_report.md
7b2bc6c18a87216accdbf08f38d71fe50422e29da364c1a8cec38956c31fb9fc  analysis/setB_pose1b_2b_3/d555_texture_summary.json
33485131aadd68abfb192b6f39e4dfd357b589a919603fca48e04da4e464b62b  analysis/setB_pose1b_2b_3.log
301e4d28bb835d8cb1f08a265f8c0b7d25e09545a162003af8bb19ebfa4a30e9  analysis/setC_pose1_3/d555_texture_report.md
9a5e7116119dfaed0b3927c0534d25de951d46c55db16c8f045a5214c13809ac  analysis/setC_pose1_3/d555_texture_summary.json
29a819343bea1e1e93563cbedd623acfa7e785ca68adf75c54d1446497d402fd  analysis/setC_pose1_3.log
d9503a0c5cbbc3045a30dd1f9ecc82e7937746259943e94121becb567e34330a  capture/check_pose1b/d555_texture_report.md
75fa7a09309c2c95db8064f145b35f98329d1a5c939899b5927d184de4b0804e  capture/check_pose1b/d555_texture_summary.json
8bb16f7e565267a8fc846a99f6df04c26813a8b8db68788546863d44114e4d25  capture/check_pose1b.log
00796c45dd09646f32e9e09da9c6a1f8671f444cb1d2ddc57e0db5c8fecbb1b9  capture/check_pose1/d555_texture_report.md
aa9b59a386427dfa4963387e8853ab2d75e104d76f9d84919637140678f5592c  capture/check_pose1/d555_texture_summary.json
8d343e98b2d0d0d5979d7304495a7f524442d9798bd3f496074f5ff7717a87a7  capture/check_pose1.log
979bdcd9d830f75f4634039f6e5662fca56da95d81636660beac900630ff7521  capture/check_pose2b/d555_texture_report.md
837c3455c1ca941ade460bfe5d1923bdfc2fcf8e6b1013bbea5be1819e40159d  capture/check_pose2b/d555_texture_summary.json
21e9fc86904576d7d31b1bc28d06b5393247105f3466cd6471fe4416b2b2be95  capture/check_pose2b.log
67cd5cab30a81f9830d7bd861d54bee99ba0b2e220f26d293ee31e53459560be  capture/check_pose2/d555_texture_report.md
44441c113c9f9a192edb46ce3a2601f7a17b63c9c183f49a526457ceee736736  capture/check_pose2/d555_texture_summary.json
d1c1d6a7359e88b916b7d1f971ee6a94a9fce58b3651c1af00d4044adf56225f  capture/check_pose2.log
0edace58acec934e913b2398b9ef80dd3fd4171364a02c79a68f7c6e73ba2493  capture/check_pose3/d555_texture_report.md
0b8374b8c5d37117ecdaea921b30d993ddb2fdc75ebf4b69d239c67040acdb85  capture/check_pose3/d555_texture_summary.json
6d8365bd13f32dddcd4b0f2bd1204bc730f7efcdcadb9a3db3c314c36585b08e  capture/check_pose3.log
b31cf47076c11f0e158b523b381212ce6e49aee7950a0782a1465ac27a13af81  capture/pose1/bag_info.txt
6fb369a0656d3c46a9fd379c8911881ae3003dfb9da7fd2de7529c67295de85c  capture/pose1/bag/metadata.yaml
93ac1b5eca889cb8b1d4d6f03f280df5f03ec2eb78ff229864f2748f97bf82c4  capture/pose1/bag/pose1_0.db3.gz.part-00
55795eacd53dde70777d4aff686f34c8eb965d8d2be37565bc741ea3e55e67e2  capture/pose1/bag/pose1_0.db3.gz.part-01
fdef7db99e0c1ff053085ef0a676687f01328fe4192bc9857e4abccb33a918ce  capture/pose1/bag/pose1_0.db3.gz.part-02
331ebef3e42d32c20acc03ef881effcf9d1ac229f6898ac75e0dd4eea46b2b74  capture/pose1b/bag_info.txt
10150296a633103af64c6a1cf7cbede516a2d747c92f957176bbddd5219183b1  capture/pose1b/bag/metadata.yaml
1ec0e1614dee197c5c70000aa53152b53f251d7a609ec55696af603b043bb278  capture/pose1b/bag/pose1b_0.db3.gz.part-00
1ffbf296f7a3f17d7a77dac952e3dec18c1cf1be33622c369f72a12ecf70068b  capture/pose1b/bag/pose1b_0.db3.gz.part-01
8cc4dc67978a1caeace774111c7566bf791e280f80016a8573a01378a07b9502  capture/pose1b/bag/pose1b_0.db3.gz.part-02
48eb9438c97fef2c19e7275944a122b7c6ce4c765bfacfa452ba618198ec17f9  capture/pose1b/frame_numbers.json
1afbbe3a318deb7e1f98a8ef34931a52444cfcfa8badd7aba0cf4ba4b919c3cb  capture/pose1b/notes.txt
9b62c71d87bd86d784d0d50ef2c172561decc31134e34c3d00485547f72ed067  capture/pose1b/params_dump.yaml
3ba667efb74aa05ecb5a6ced481dbc22a0f15b6405e0b424ea67b62b5ccf9220  capture/pose1b/params_readback.tsv
cc19f1317879d95f1bd6ccad0e430168f3c0918e8d45de91d876d27a95556cd8  capture/pose1b/record_color.log
54a69e6103eefbfab9f6271a022d7d01d8171f8977d65252efd7f9675df4e844  capture/pose1b/record.log
1bdb20d97db86e6fd8b5ca6d94192607d60e8482937dd5e91f8f97da2ab73058  capture/pose1b/t_end_utc.txt
6bde2b71e56393c00cf3feffdca3e298e6e1228717fe10753fd77cbf1e67eed4  capture/pose1b/t_start_utc.txt
7553921f4869bab8b4263c9d8d7972c29dbb9b071f09293ba1694502eda253a6  capture/pose1/frame_numbers.json
5a7e0f960c7bddaf9499d0191aaf50ac9491b7e75b65ab40edb0e9ebe96e47ef  capture/pose1/notes.txt
9b62c71d87bd86d784d0d50ef2c172561decc31134e34c3d00485547f72ed067  capture/pose1/params_dump.yaml
3ba667efb74aa05ecb5a6ced481dbc22a0f15b6405e0b424ea67b62b5ccf9220  capture/pose1/params_readback.tsv
0f3a10163d8cfdde2472b2529782fa31e587123ed2ff9c83bb13d58f85a69f85  capture/pose1/record_color.log
f95117a04d77caca89380d1bec611876247f13cdd932748888955b76c8abe04e  capture/pose1/record.log
7a9220652592bbc712b6da9d13ba3f42498e3958c03ad10a2f2ad6e0d6e094ae  capture/pose1/t_end_utc.txt
3e8eaeb62fa2d6cd34a4680a55088a9869c33959ab72235cccd9ba0635eb6681  capture/pose1/t_start_utc.txt
4e98c050522d99f1eb14d82137ff496fd3acd8e5b786a330111da716190e4c4a  capture/pose2/bag_info.txt
f8088c1f57103d17c3c59df96a6b6295f90524ec4e5d7f9ebe5f391295ad4783  capture/pose2/bag/metadata.yaml
99e17014b3f52d026a416dcea934c291e01eebc944ddf213c79eb78575aa701c  capture/pose2/bag/pose2_0.db3.gz.part-00
36b4babae5f811a646f03b6b9406474b04e4a068c5c063ebccd9f7c67d98c055  capture/pose2/bag/pose2_0.db3.gz.part-01
de90db1f0c903925966d41dce008a9183c55c554c2f96ba4efae03ecf4949556  capture/pose2/bag/pose2_0.db3.gz.part-02
292243034053945a58879d606337d0e8b9d40b5f8a1fc5e4d3119716625d72f7  capture/pose2b/bag_info.txt
beab4f1c737a42ce1856bd523e06565de466c7c9fdd09f5081d191bb472135fa  capture/pose2b/bag/metadata.yaml
ff66624e16d5622dd6ae21e72a833d2d9b6cd086c2b8d88966310777deebe919  capture/pose2b/bag/pose2b_0.db3.gz.part-00
19ffb07f5420e7712fcfb567569fae2f4c7380f4c71d154b6e4d0f27d0774888  capture/pose2b/bag/pose2b_0.db3.gz.part-01
debdeaf9a086b4ae742986ce01c18378a80c47b1a30b8fd0cdc7882967f64358  capture/pose2b/bag/pose2b_0.db3.gz.part-02
d2f7d13b8089dc7c1ccfff39e51d423edb65da96f193ae5893cd38ce9132dca2  capture/pose2b/frame_numbers.json
82c95b8820ddb41c2bc05c1130a3162dc6eef2a002071552d869968a14c6532e  capture/pose2b/notes.txt
9b62c71d87bd86d784d0d50ef2c172561decc31134e34c3d00485547f72ed067  capture/pose2b/params_dump.yaml
3ba667efb74aa05ecb5a6ced481dbc22a0f15b6405e0b424ea67b62b5ccf9220  capture/pose2b/params_readback.tsv
50708dcf2fe23c48b8273889d63386f3b2aa98d6409285ae177c6b2cc593c272  capture/pose2b/record_color.log
3d67568a9262c146d7d63113da62b1bd4f80d2f3b1cdf7a53987da5a5cf320a2  capture/pose2b/record.log
bb0c68737879e4f8b7f29e3635cc5f6191ba205e0c6f86a869d795fd3bf51543  capture/pose2b/t_end_utc.txt
44e97f89d604ae027351cc3c5d7c518a137dacd4475b3dce5c8748827a261876  capture/pose2b/t_start_utc.txt
632682e309b7f25c9d621aaeef7f8e273ebe62d95d42755fa744884341799bde  capture/pose2/frame_numbers.json
8510191b6ca72660514ab766f1b9be5bc1058ffdac6831ac6fdc3524d85d1396  capture/pose2/notes.txt
9b62c71d87bd86d784d0d50ef2c172561decc31134e34c3d00485547f72ed067  capture/pose2/params_dump.yaml
3ba667efb74aa05ecb5a6ced481dbc22a0f15b6405e0b424ea67b62b5ccf9220  capture/pose2/params_readback.tsv
e15de83855459c4defaa610d01063c58afa98e4ee4f1692f08ff00d292c51811  capture/pose2/record_color.log
ab3f80a64f7c1c5135f372f066714e121e77af277027a70531537ae25bf40d97  capture/pose2/record.log
4322566783ac6f040be2a78267fd500dcdbdff9d2d9e4180e5e15b9a224ce6ab  capture/pose2/t_end_utc.txt
6d93f8d2c36c8a1e23165a1cac380b01ac5aa4bf7e0649b843523dfa310ba070  capture/pose2/t_start_utc.txt
c0557b76dd7d486d09cacfc8cd760bae35a9086584e96553489d163c0d0909f3  capture/pose3/bag_info.txt
afdba35f7a08b98ec117edf5f8f840ca24846582860382122719978e77fb6c34  capture/pose3/bag/metadata.yaml
b05c9f031881ed6c80e29bd1b81aa99fb0c6875f33dca9eb32734d9815095c7f  capture/pose3/bag/pose3_0.db3.gz.part-00
7898fd2515b3fe693883e13fdbaf12b109426745d3dd510c3b7cd04a7c591602  capture/pose3/bag/pose3_0.db3.gz.part-01
9f394c161e88546d8df9dc3b7946aaa6f0174e8bbdd4de8e4d66caddc60e2a64  capture/pose3/frame_numbers.json
f9ac0832637b4ce09ef541656bb19654647ec4d4c00339e5493305108c2f194a  capture/pose3/notes.txt
9b62c71d87bd86d784d0d50ef2c172561decc31134e34c3d00485547f72ed067  capture/pose3/params_dump.yaml
3ba667efb74aa05ecb5a6ced481dbc22a0f15b6405e0b424ea67b62b5ccf9220  capture/pose3/params_readback.tsv
91c78069f213dd6ad922c7f0341d6f95cd52a4be052e732878a3800ae285e0c9  capture/pose3/record_color.log
d80882e7cb3ebf13fdd308c106557a8efcef8ab576c34b32d5814dc5c91baea4  capture/pose3/record.log
6dc70dac5a96d298df40415134be5b706c3a4e456dd65e8b780bc67c7520a6c5  capture/pose3/t_end_utc.txt
09d63db381dac526d44b16d7fed627bf26361b036011c4c2a44c704fcbf3ade9  capture/pose3/t_start_utc.txt
1dc041f1e8898ebc81854a0adf59ea0a0b0eb4d489eb8dfa302931e3879c6c8f  capture/PREANALYSIS.md
dd4ec478d487dfa69beb5eaf45d706ed1430abdf115ca4f03ed5ff8a21ec0992  diagnostics/floorfit.py
c30d8a3a5896b8aa84dbefaa1e5ca6b1d45b8d33de27290f8e0a565710e475b1  diagnostics/jitter_all.txt
b26223a1ac5caeae8ad638d9d60d8afe63461f00952e60d79a649d49f6b1d66d  diagnostics/jitter.py
20b80c8596eb95518eec83228da67a876c08d41f6484ef203237688cf315dcfc  diagnostics/still3.py
238eae2c1f03fa3619397b5bfd799617319d8af8878d129eb678c4847e7410bb  diagnostics/stillness_pose1b.txt
5f68900b14618ee6e7af77bf5269a6d9f25770cddb35fb8658703afeb922f613  diagnostics/stillness_pose1.txt
fbce342298c257618edbc24568bece5f71295a34096af3c234a0034eb0bacf07  diagnostics/stillness_pose2b.txt
b5adb6d72a51d7d25dab2c8d1966cd4f874d9063e99ed99c1c4541bb1642cb89  diagnostics/stillness_pose2.txt
6fa7104f0e5811e416de714d9c0408e3fa461c0c9668aae318f1c9d319135478  diagnostics/stillness_pose3.txt
cfaef74e0ab1f24fca742e9e6470772f46bc0926de1ed7067e7a55ba62e06ba5  dryrun_20260925/bag_info.txt
6b15c84431648a5ba8d0105759865c6539adac9b7aef28a072500433cd6b8925  dryrun_20260925/check_desk_dryrun.log
d5fd64f98a9db39d557c81431394ffa77da4925cbea3cdeef90ffacf60fe3bad  dryrun_20260925/d555_texture_report.md
02ec11e8bd792fbcbaeb3c196cec6bd4fbc945bc9a45844db6a2a1abb0a1ce42  dryrun_20260925/d555_texture_summary.json
5a46eeb94d0f107bce64ed51f64a7c076fca2153e795be7aa56f22f2ebabfcbc  dryrun_20260925/frame_numbers.json
3ba667efb74aa05ecb5a6ced481dbc22a0f15b6405e0b424ea67b62b5ccf9220  dryrun_20260925/params_readback.tsv
48b805d5b397ae8c72f90d4c69526e3db652c72e8dd32a5f167ee659ca6fa77b  hw/camera_usb_state_20260926.txt
9f5ad4c9c76e7186f89f1b283dc78595a9dcad13d338a2be60edeab5bcb011e2  hw/image_realsense_pkgs.txt
dea45276b44623641cd42ecd621f0c95cbe459a16812194e8137103be6e71e9c  hw/image_stamps.txt
ee9bdd69322553f8876650891c141fc89ee404afc9bfe79629c433b505d1d0c5  hw/perception_node_20260926.log
082427b60aa6ed833a82274d2d38ce7e35a88da52dacc5e6f10e8715cf86febf  hw/perception_up5_utc.txt
0476639daa021c0c67f3d66d28d2d12f2d9911dcf5ea469302a883d2f3953665  tools/analyse_bags.sh
c1bd7b5d21ffbb5c725d8ade99d42d721f4e0ab4c7933a8c00a6a33ca0a36a58  tools/bag_frame_numbers.py
4001fd2dbfa4560806b4719c0bb9be45a8e99df71e40a861d97e8db413776176  tools/bag_to_npz.py
1765daa9a25a7675d0d9f97df92790f74f45a23480eb655160d9f4a30b730aff  tools/CAPTURE.md
80154af66d61a35a59673655b5cf648971988cfe3ee0d39dd580d95c6594eb53  tools/d555_secondary_analysis.py
65a28fb178c0ce89739603c84dd8cfbe0109f0b082608d1b7242bce50efce605  tools/d555_secondary_analysis.selftest.log
39c75cdc6a4e4ab972ea243bb8106cbbbcadfb8d501ae8482eed31a46bc723a3  tools/d555_texture_analysis.py
b9cf9014f70b9ea304d4fa1f0878e038b86403500b25486f808aadd2e9e8ac01  tools/d555_texture_analysis.selftest.log
4663491584a3b9a742f950c968a8bef2ebbe67221ebe78904b5f656dcfa30be0  tools/pose_capture.sh
8fb1d776f7d799c46e1d04ace3c81bfd1b06bc59661fa6cf4eb66f5167fc5a2a  verification/stillness_panels_depth_only.png
f9246128b81913c4e420090454855269e87b6bef0bdbc20ec5b88a20ee211926  verification/verification_verdicts.json
3d101e7dea9e11ff4020c0ab13c2743ebfa675a43155026be578c07b7495f166  verification/verify_briefboxes/check_pose1b.json
7e9d91f9389538906a65cc968556e46c036cb50f8b9a44bca49322f796241c5c  verification/verify_briefboxes/check_pose1.json
2c2a777725bd2e68fd14815c68dc6c0cc8dc2b6aa98987de18baacb8ffb13d4d  verification/verify_briefboxes/check_pose2b.json
a654c293f233aa770646fca8e23dc584371a18fe631361abc4c97396276332c7  verification/verify_briefboxes/check_pose2.json
8d5b24a47a37ef0ff7b682fa5e136299ab0b30fe73f33ca0c4106c368eec3051  verification/verify_briefboxes/check_pose3.json
05e0fe649260f58aca3b8b6517e02b8c7b9104e3ed262b0e86948c5b9c0d34ad  verification/verify_briefboxes/check.py
27e1aa23a7af5f1a5fb6bcfcee4b0c268fdf2c8bfb38d8c16f675654f8924a1c  verification/verify_briefboxes/cm2.py
3cc076f1f276c2da8c56dc0fb1ef8c232766a82f55a5af761507d78a23f0f77a  verification/verify_briefboxes/cm.py
79b1da3e7ea2a2ab2f4627cb4c1ae21345e21246800e3b307953a3f4c6391385  verification/verify_briefboxes/flags_pose1b.npy
2166a3df17603a97c549bf7196596532345513bae16ef9eb2a0f32340b695b75  verification/verify_briefboxes/flags_pose1.npy
1b210fa86148f3dff765ba390d22e3d8a70325bac3777bd4915f84ba7b61712b  verification/verify_briefboxes/flags_pose2b.npy
8598f35051f4a9097a0f6952a7ba7e1344e36c9f6ebdddf4547ab08e64ca6783  verification/verify_briefboxes/flags_pose2.npy
7fbdd7801ed93ac5ebd2008cf6c7502224e6989eb0ec61a50e40896da8195ad1  verification/verify_briefboxes/flags_pose3.npy
dbe747de4b6858792d7c39f590e51969425451f2d05de1418a6f94eca4a9f289  verification/verify_briefboxes/floorcm.py
d80c591d43e35196025253f1fcc9b48bdf40399cba49d482eef206b85c3d99b7  verification/verify_briefboxes/log_pose1b.txt
9bd4686a222906395c9087d8299d0a0b107f45d80eddc8ba33a4e920261b7072  verification/verify_briefboxes/log_pose1.txt
7f0b412b735c9c2b84f2a81922d7fbc0a22670a3c774a666dd3a16f7074288f2  verification/verify_briefboxes/log_pose2b.txt
48060be6132ebf835e8d0043e7d6c491d3fd559b9b2e12e8497bee951e01af10  verification/verify_briefboxes/log_pose3.txt
f80fe0ce801a1cd77edc5ebfdc32fff5b68eb862858907cdcb1d87cc197feb66  verification/verify_briefboxes/okmask_pose1b.npy
2debf08377a43edd08bebbe92b03076ceacdba92054d5e8efd00f447da6c361f  verification/verify_briefboxes/okmask_pose1.npy
04038b8099d025c24d8231554bf499335708c6a7a96061ca680dcdd33a0fab80  verification/verify_briefboxes/okmask_pose2b.npy
eb4c78b2c5c0f12c5804f394a886c68483d1df94e42f8cc2cb7fe1ccf58d6310  verification/verify_briefboxes/okmask_pose2.npy
fb4efe4d56d6f9d3ddfc7409b1d5a2920d2d2bb84aee481947d79c3165ae5c24  verification/verify_briefboxes/okmask_pose3.npy
ee35a10f5818ffbf979841169305b3b8412b9c2a8b3a7cfa190a16859e269cd0  verification/verify_floor/decomp.py
2f2559a6bdc46b14132770ab27c23b8c6e2f4ca61e1c0abbbf2818727a516533  verification/verify_floor/floor_pose.py
8d0a495412cc488db1b5547e224a03a01aa2945d42b975e19cfedf6aad78c674  verification/verify_floor/pool2.py
3d51f4e558912c1b7d0ec34da041b82497b20dddbd39a55b184636fde81439bb  verification/verify_floor/pool.py
b009f6c1ffec0a0493a6beeb2595a2b3e1f3b7161782bdd6f3ec4ac55f0df6f0  verification/verify_floor/pose1b_cells.npz
eb7f8bf2cdd4fe791750ac38efbd40e5deb533030e5a76d348341ec424d26a68  verification/verify_floor/pose1b_pix.npz
8dd83c761973f10dc130a3239297a2edd01dcff9c7540b5536b6a6d7170f3294  verification/verify_floor/pose1b_series.npz
d011ec87a5aa6a283ae59445adcaa90379e7fced809274a895692475b291f57f  verification/verify_floor/pose1_cells.npz
5d0890bfa98ef9f3d021ef3109ff279e30e1a13ece46619a4efe482a73f2666b  verification/verify_floor/pose1_pix.npz
1c1d4542d13922850570a242b112de40ac017ed77dd26d4811fa7a378d41a503  verification/verify_floor/pose1_series.npz
4f2bfe459f6180526566e93892e0e3abebbc49a9ba8d100118d05b886e8fa9b5  verification/verify_floor/pose2b_cells.npz
8f538fca74e5083692c596fb8cc881f5b24837db6cc586bf541f60469d286033  verification/verify_floor/pose2b_pix.npz
83d2651ee5845ea161840013da2dfdc95efd7867a18f58534bbe50eff1b1891a  verification/verify_floor/pose2b_series.npz
655ca9bcccee0ac580c15250c16a72e31f4848cb15687a596f20889743a4faa0  verification/verify_floor/pose2_cells.npz
308c5ea9101a9c9fa30c5c25aeed00147930086fe0138a6c84bc8b4712865316  verification/verify_floor/pose2_pix.npz
143c07709c7a50812bafec125fee28765e6ba560369ff41a6b5f22e6f77ecab4  verification/verify_floor/pose2_series.npz
01a5c52db1cd7e35d21d432f6a8e111db5b5c58660230cd07848fd2a35a3c627  verification/verify_floor/pose3_cells.npz
378d1a2fb9e75f5f6bd9aeec8781c0c9f58dcc4df2973b5516b0d470bb0b8637  verification/verify_floor/pose3_pix.npz
6eb7bd30f3855c14291009d6f929719ab2655a94b9fbe1dabb4b26ac44abf786  verification/verify_floor/pose3_series.npz
b4971b64227082516a16ff97bc582f2dd3004fe46e1124e5fce350d85bf5bfa7  verification/verify_gateB/comp.py
0aadae90ff8b7e24b48187b15fcc2e80a760ce3656a8594ee73e4ec49efbf78c  verification/verify_gateB/extract.py
1df4e5666de0d1c8c7cdb1175b30bfa15a45b5e2bdd689e550335d724eb99f5a  verification/verify_gateB/flicker.py
53ff90fdbdd4507857641d0d1d4aea0976d961dd0c11b960e7176d2b40a912c2  verification/verify_gateB/floor2.py
cfc5b0c860253192c184ec00288c65c93fdc354ab6f758df40d16d17cc83ddef  verification/verify_gateB/floor.py
f3a4e2864be32ddcafa0751ed9af4968a0945f0cece378e1ffe5e32db9a63c9c  verification/verify_gateB/gateb.py
7de63ecb96b084e9ddc3be3336cef8af55bf417e45d5538d5d0bd7dc438b57cf  verification/verify_gateB/robust2.py
beb28b8deb8d97c14c558dc0b886357c2d933fe1669474545462565e254f0434  verification/verify_gateB/robust.py
de57756538d90dba9e175c5c19a8e5cea208ace45af9b2ab6d7e7ba23f7ffe8b  verification/verify_recompute/agg.py
cadf532a2788b98aaa27e3d23e8ac3588118b06911f57e94acaa51445ee50c88  verification/verify_recompute/floor.py
d93dcbf74a9659330f9fff14cb4e96ec4ff35066448894ef62dd837f79011914  verification/verify_recompute/mmload.py
eefbee4f86b2a6a64842df1b3b6cbba4513c170f3cc60704a9371b3547b62da5  verification/verify_recompute/nzmed.py
35710db4864c2e00503e5ad9ab52056cd3c26ef7796f3a9d5c33ab03cecef7a8  verification/verify_recompute/perpose.py
7adcd0a01017ce10199313a11e5ad0956060cf78e3d381692a9206da27f48649  verification/verify_recompute/pose1_floor_invz.npy
053b3bad263d8013c2f3216bc1ef0b0475c41fb3a94841ed49cd6cd269d98121  verification/verify_recompute/pose1_floor_svd.npy
56cd8457aa22a3b579ac562c20eabec8a75724a08f017e006f3f42952e0ea916  verification/verify_recompute/pose1_nzmed.npy
06cf2aec96437aafe29a76460676777cad0f962aa0a5e0b3942e519378e8f911  verification/verify_recompute/pose2_floor_invz.npy
916f51a829d0cf97e1c741eecbed816490089129d6efef1f95e950cb4686257f  verification/verify_recompute/pose2_floor_svd.npy
6ce99a7c9fdb83de79583a164d44a7db525b84e507e3af61e5bf9ba5888c3eb8  verification/verify_recompute/pose2_nzmed.npy
3f758225bc112f078eb62b99fb1f3b89447e766047af9525449b41cb86bc55f1  verification/verify_recompute/pose3_floor_invz.npy
ac60932662fe6ae3282ced1292b874fd3841bbe74371ad149f463079d34ea2cf  verification/verify_recompute/pose3_floor_svd.npy
fc6945d9a69ae776e9e7e2d834a9eaac507e4e18ce9638c586cd066fae239a4e  verification/verify_recompute/pose3_nzmed.npy
3afe52924d531a4997e9a0c84b7b11c94f792e7581da968803c2e4bef30ee894  verification/verify_stillness/boxdetail.py
af4dcf0f7051d9917d96e7e31478e832bb6d7efe62063f502b62a432d5fbb9d8  verification/verify_stillness/box_pose2b.json
71708e9a8cc6d67f680827d5f125e01be1a0373a09571cbebd25c2e2f4ceffbb  verification/verify_stillness/box_pose2.json
b8549ed7d81bbae7bb334d03f5c0b0317078b10b0663cb543cfa069117a65988  verification/verify_stillness/cell_pose1b.json
34df3f7c505f23806b643e6687de13e1c03cc107b654687da11ce243b9e4151e  verification/verify_stillness/cell_pose1.json
11cc520d86434cf34cad10a65f2aebb968cbd0ea05d872fa6967bc233cd7d0d2  verification/verify_stillness/cell_pose2b.json
fdd07bfdbe0ebd105923f811a8805d88d5264bbb46f10059c5dc79f2d61d6ff3  verification/verify_stillness/cell_pose2.json
428b5698289fcf3524498da49bd05a0663dba253ca2a6105ba69348fcda79edf  verification/verify_stillness/cell_pose3.json
10207416ee49a7efb78248af697da430a852dc56d0c601f486d6edec0eeaeab8  verification/verify_stillness/cell_slow.py
9eb96b4a839f7837aaa38b4087d6a9341e7d96acaa57b80cf2aac0f0dff1ff9d  verification/verify_stillness/cm_pose1b.json
66ff99eeeed7acf22eb008b0f779ac5021a4b64de7117ba0f593d8fbbc4a3f82  verification/verify_stillness/cm_pose1.json
6d79b311da66950d3ff458910c68181875af3905c0b7bf641654809dda90df0d  verification/verify_stillness/cm_pose2b.json
37cd2f1eddead1b470be632dbc3e14d3fbf62b4dc196e19ef6933c094b7d5e5a  verification/verify_stillness/cm_pose2.json
b442f0d1df91488cd62405e176eb96abda982308994c613daf4fc209685fe9c4  verification/verify_stillness/cm_pose3.json
48c29e16dfa286894aba691c2447eeff78eb21e836942c33a6ac38267e5df6eb  verification/verify_stillness/comb_pose2.json
d2f797239778afe77c709f4068c3b2d5242141a12e4086d288a6525b254331f2  verification/verify_stillness/comb.py
8687e32297ee2c797e5246775d046e39942c20f2ef26ad6a5f2b380b1157a434  verification/verify_stillness/commonmode.py
70d02c1a2f94d06783efa2ab9cb2b1afb67098ec2bb2d039bc52cda101984e68  verification/verify_stillness/figs.py
d6c666273c5a8d4eed2b03733fe44685755572b75c8e7a8ad374c7c9bd998a8a  verification/verify_stillness/gaps_pose1b.json
be3b86351633eb1a92c93e57e974b6687c9294f0bdff1c34b55743ac4a1741c6  verification/verify_stillness/gaps_pose1.json
9aa1db364c0be8f1cd4e345e3ec33a1a81b2b6781e88e7c8e5290f669255a3af  verification/verify_stillness/gaps_pose2b.json
2c560008d091ff20d33abf456a647dae7c434a1e86f2d07a063f2918763ff06f  verification/verify_stillness/gaps_pose2.json
e5f08525ad649d6bc3ac209609c6101265e0686e101fca55af1d3059ab06550a  verification/verify_stillness/gaps.py
3f934fb4f1b212391a63541c3a88a0464a1b168a5ba3960a0787ab71fc729d4d  verification/verify_stillness/maps_pose1b.npz
bc991473ce1e133e45ea6d0180b5b708c03cb5338017e9f9278c8a5ca52d1916  verification/verify_stillness/maps_pose1.npz
125b24a08fc7262ef11a79eaef65ebf7d454d5b40b2d9a7d0b2b2a98a7525559  verification/verify_stillness/maps_pose2b.npz
c35ad654bb93d78ae831796d9f8b9f7271db864d168a602c906e5e41f9b828ee  verification/verify_stillness/maps_pose2.npz
12c87808cb973ccf10e58ba8d03a3f7604884a651bb47cd70ae7a2b2ffe20c28  verification/verify_stillness/maps_pose3.npz
5b5cb532185cf7ee0efa2575d9ffbc5672d24f598340be9769db123d16ee3590  verification/verify_stillness/meanmed_pose1b.json
294a3689670f1cd8b9d6b9cb32fe33d63afa30404a8966b1eb0bc646cc7ce06d  verification/verify_stillness/meanmed_pose1.json
7839fa18314767dc08fa9241dce2092123db5432db659608b65785cdd466a9ca  verification/verify_stillness/meanmed_pose2b.json
f3e29129f371d69a6c4a412a5238afb0d56f7414437ea19004321a2151358c25  verification/verify_stillness/meanmed_pose2.json
4c8d12283a3048b8bf17eef91c0f5cde681ade839044907a3325eb1150680faf  verification/verify_stillness/meanmed_pose3.json
37167766e699bf41c59ff5c90758ff9659b25770e222acb47269565a59a0484d  verification/verify_stillness/meanmed.py
60c9f6e4ecfae4b9903d3d439125d7e5e48b947ed7292786d2af0c180ea25cb1  verification/verify_stillness/overlap.json
c312bda05e8e09766fad3f65204ec2db38d7312302c73e8b9595e6158d406cca  verification/verify_stillness/overlap.py
601ba3c7311641347276fdc8b1872a721134f17ccd00efafb0307769583fbe19  verification/verify_stillness/res_pose1b.json
ee1257543eb14ae64f33768ddbada8917f4ff0ea49ddcf99885fe6e13603a27a  verification/verify_stillness/res_pose1.json
eeea9586101f5458cadd073fcb752cf847988ddea1a3009950f1847084632856  verification/verify_stillness/res_pose2b.json
86e64887d7c0957be3f4a322c3dbfaf58ac62b1d5824a15a854ea16bf3efdebf  verification/verify_stillness/res_pose2.json
1418b46e0095edc81284d4cc0bbfcd7f57c3596c11cb99838301562316dd4468  verification/verify_stillness/res_pose3.json
113ae06c4d50e5dbd1690c023ea6873522396ef3244a0cd3cb8429875a253488  verification/verify_stillness/rigid_pose1b.json
ff972e0ce1b928bc0cfdbb651486e592e8ed9ac0839a3c8904371174f84eedcb  verification/verify_stillness/rigid_pose1.json
453e60169f1b0631bc402eb06e492e7b6f4b011abc589a88b3156c7334e2823a  verification/verify_stillness/rigid_pose2b.json
b5d4c1520fcd81458172d50a6d6055e0ea8041a587c4802fb7a75a102f644119  verification/verify_stillness/rigid_pose2.json
23c23d6d8d9337c496b21487bbc49d326da3f6a5f264df54f263a857fa59e28a  verification/verify_stillness/rigid_pose3.json
2c0780465de0787e52838c1bb781df30ab8c285f6e9bf4842ef21700840ba164  verification/verify_stillness/rigid.py
6b9165090b91b29fd949ae95c710f4dbd6d38cbe9fa275a14ee6b194f97f2f86  verification/verify_stillness/sim_pose1b_tau120.json
e9d397f2c694ea41a1f38df65e821446540af71ea80011208b5fc337e7048ee4  verification/verify_stillness/sim_pose1_tau120.json
c064e7348cc9affc4f80e3f27526b110d1dca4526511dc2985c9b28f956d2d63  verification/verify_stillness/sim_pose2b_tau120.json
22ee0443bc0c8f2a8598cf4851e3482b6d8aa428b3477a1cb66778daec92dd1b  verification/verify_stillness/sim_pose2_tau120.json
ef081af7509be3f413d6b151d1de42932f60a0d3ec117f18079e0beee5854fb7  verification/verify_stillness/sim_pose2_tau600.json
7f7a65bda429cd92f4fb9162a31908366e75425d302a488b99e66e892cc6e3bf  verification/verify_stillness/sim_pose3_tau120.json
c5930cfc5d61641890bcebd1573e9ccb376b93fa99e1d5936fda4ebdf043c61b  verification/verify_stillness/sim_rule.py
ee71cc78c23b3bc692e00b30aab4ffe9ba1df7b082ad55ab44a7c307074735d6  verification/verify_stillness/still_refute.py
2c4bed2c27a9f1ce7797fe915a35eeb0eb2b743ca35d1265737092a788450563  verification/verify_stillness/summ.py
```
