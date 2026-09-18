# Measure the real D555's depth texture either side of the block reduction

**Type:** investigation (bench measurement, feeds a training decision)
**Owner:** Jetson
**Priority:** P1 — it is the only blocker on deciding whether the training depth
noise term is mis-calibrated, and every depth sim-to-real claim in the
depth-subgoal line currently rests on a quantity nobody has measured.
**Estimate:** S–M (one bagging sitting plus an offline analysis; M only if the
filter pinning in `perception.launch.py` has to land first)
**Branch:** `task/real-d555-depth-texture-capture`

## Story

As the **training depth noise model**, I need **the real sensor's per-pixel
noise measured both before and after the 8×8 block reduction the policy's
observation actually goes through**, so that **the injected σ can be compared
with what the policy would receive on hardware instead of with a raw-pixel
figure that is not the same quantity.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [completed/d555-invalid-pixel-statistics.md](../../completed/d555-invalid-pixel-statistics.md)
  — the 2026-08-04 capture this extends, including the parked option whose
  revisit trigger this measurement evaluates.

## Context

`DepthNoiseModel` injects σ_z = z²·σ_d/(f·B) **i.i.d. per policy pixel at
80×45**, with no reduction stage. The deploy path applies the same physical
per-raw-pixel error at 640×360 and then collapses every 8×8 block with
`np.median` ([`obs_pipeline.py`](../../../../source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py)).
So training's σ and the deploy path's delivered σ are different quantities, and
which is larger depends on how correlated the noise is **inside** a block.

Call that within-block correlation ρ. For equicorrelated noise the median of 64
samples attenuates σ by √(ρ + (1−ρ)·π/128) — a factor of 6.46× at ρ = 0 and 1×
at ρ = 1. ρ is measured nowhere.

The consequence is that the existing measurement cannot settle the question it
looks like it settles. [`d555-invalid-pixel-statistics.md`](../../completed/d555-invalid-pixel-statistics.md)
§4 reports per-pixel temporal σ at **raw** 640×360 — p50 1.1 / 2.7 / 10.1 / 15.5
/ 87.1 mm over its five range bands. Put through the reduction, the 3.5–5.5 m
band lands anywhere between 13.6 mm (ρ = 0) and 87.1 mm (ρ = 1) against
training's 25.3 mm at that range. The band-by-band ρ at which the two are equal
is 0.293 / 0.512 / 0.227 / 0.516 / 0.062, and the shipped `disparity_noise_px =
0.08` sits inside every band's [ρ = 0, ρ = 1] equivalent interval. **The sign of
the training-versus-real noise inequality is undetermined**, so no σ change can
be justified in either direction yet. That is measured in
[`noise-texture-parity-2026-09-17`](../../../measurements/noise-texture-parity-2026-09-17/README.md).

A second reason this is P1 rather than P2: the reduction residual is exactly the
statistic the parked training-lane option needs.
[`d555-invalid-pixel-statistics.md`](../../completed/d555-invalid-pixel-statistics.md)
parks "render the policy camera at 640×360 in training and share
`downsample_depth`" with the revisit trigger "if this brief's measurement shows
the real-sensor reduction residual is materially larger than sim's, the option
re-enters the retrain conversation." That measurement reported raw σ only, so
the trigger was neither met nor refuted and has been dormant since.

Note the sensor has never fed the depth policy: the node requires `32FC1` and
the driver publishes `16UC1`
([`d555-depth-decode-validity`](d555-depth-decode-validity.md)). This brief does
**not** need that fixed — it bags the camera topic directly and runs
`downsample_depth` offline.

## Method

1. **Pin the filter state first, or record that it is unpinned.**
   `strafer_perception/config/d555_params.yaml` is never loaded by any launch
   path, so the four post-processing filters and depth auto-exposure are at
   `realsense2_camera` defaults rather than at the values that file names.
   Either land [`d555-params-file-inert`](../reliability/d555-params-file-inert.md)
   first, or read the running values back off the node and record them. Spatial
   or temporal filtering would bias every number here, and a temporal filter
   would bias it **down**, in the direction that makes the sensor look quieter
   than it is.
2. **Bag.** 640×360 Z16 on `/d555/depth/image_rect_raw`, ≥600 frames (≥20 s) at
   each of ≥3 **static** poses in a real deployment room, at **robot mount
   height** — not benchtop height, which is what put a desk surface below the
   stereo minimum through the bottom third of the 2026-08-04 frame. Nothing in
   the scene may move; a person in frame invalidates the temporal statistic.
3. **Compute both sides of the reduction, per range band.** Use the existing
   bands (0.4–1.0, 1.0–1.5, 1.5–2.5, 2.5–3.5, 3.5–5.5 m) so the results line up
   with 2026-08-04:
   - per-pixel temporal σ of the **raw** 640×360 stream, and
   - per-policy-pixel temporal σ of the same stream after the production
     `strafer_inference.obs_pipeline.downsample_depth`.
   Report **n per band** and the validity-survivorship rule, both of which the
   2026-08-04 table omits.
4. **Derive ρ** from the ratio r of the two: ρ = (r² − π/128)/(1 − π/128).
   This is the deliverable.
5. **Report the texture statistic, not just σ.** On the 80×45 frames, p95 of
   |d − median3x3(d)| and the **exact-zero share** of that high-pass, per band.
   A plain σ or std is not enough: on frames that are mostly exactly their own
   local median, a std is dominated by depth-discontinuity geometry and reads as
   "matching" when the per-pixel texture differs by orders of magnitude.
6. **Report the invalid-block outcome.** The share of 80×45 pixels the deploy
   path writes the near fill at, given Z16's invalid-is-`0`. On the 2026-08-04
   capture 33.9 % of blocks had ≥32/64 invalid, and the near-fill rule turns
   those into a 0.2 m reading where the training convention is the far clamp.

## Acceptance criteria

- [ ] Raw and post-`downsample_depth` per-pixel temporal σ, tabulated per range
      band, with n per band and the survivorship rule stated.
- [ ] **ρ reported per band**, with the arithmetic shown, plus a plain statement
      of which side of each band's crossover value it falls on.
- [ ] The 80×45 texture statistic (high-pass p95 and its exact-zero share) per
      band, on the same frames.
- [ ] The share of 80×45 pixels the deploy path near-fills, and the far-clamp
      share for comparison with training.
- [ ] The filter and auto-exposure state during the capture recorded as read
      back off the running node, not as read out of the unloaded params file.
- [ ] A recommendation against
      [`noise-texture-parity-2026-09-17`](../../../measurements/noise-texture-parity-2026-09-17/README.md) §8
      gate (B): the training term's injected σ at 80×45 is within [0.5×, 2.0×] of
      the measured post-reduction σ below 3.5 m and [0.33×, 3.0×] above it, or it
      is not. **`disparity_noise_px = 0.08` may already pass, in which case the
      correct outcome is no code change** and the recommendation says so.
- [ ] The parked 640×360-render option's revisit trigger explicitly evaluated —
      met or not met — so it stops being dormant either way.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- `source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py:76-84`
  — the reduction this brief measures either side of: the `isfinite` rescue, the
  8×8 block median over `reshape(45,8,80,8)`, the near-field fill and the clip.
  Run it offline on the bagged frames rather than reimplementing it.
- [`completed/d555-invalid-pixel-statistics.md`](../../completed/d555-invalid-pixel-statistics.md)
  §4 — the raw per-pixel σ table this extends, and §5 for the sub-0.4 m
  behaviour that makes the near-fill contract load-bearing. Its "Out of scope"
  section holds the parked 640×360-render option and its revisit trigger.
- `source/strafer_ros/strafer_perception/launch/perception.launch.py` — the
  `rs_launch.py` include whose explicit argument dict carries no
  `--params-file`, which is why the filter state has to be read off the running
  node.
- `source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py`,
  `DepthNoiseModel.__call__` — where σ_z is injected, at `cfg.height`×`cfg.width`
  = 45×80, with `torch.randn_like` and no reduction stage.
- `source/strafer_shared/strafer_shared/constants.py` — `DEPTH_*` and
  `PERCEPTION_*`, for the 8×8 ratio and the fill/clip constants both sides share.
- [`noise-texture-parity-2026-09-17`](../../../measurements/noise-texture-parity-2026-09-17/README.md)
  §6 for the commensurability algebra and the crossover-ρ table, and its
  deposited `probes/sensor_commensurability.py` for the arithmetic in runnable
  form.

## Out of scope

- **Changing `disparity_noise_px` or any other noise field.** This brief
  produces the number the decision needs; the decision and any training-side
  change are separate. A change made before ρ is known is unjustified in either
  direction.
- **The 16UC1 decode.** Owned by
  [`d555-depth-decode-validity`](d555-depth-decode-validity.md). This capture
  reads the camera topic directly and does not need the policy path working.
- **Re-surveying the reliable depth range.** That is
  [`real-d555-depth-range-survey`](../investigations/real-d555-depth-range-survey.md);
  if one sitting can serve both, good, but the range question is not this
  brief's acceptance.
- **Anything about the retrain.** The retrain is not held on this
  ([`noise-texture-parity-2026-09-17`](../../../measurements/noise-texture-parity-2026-09-17/README.md)
  §11 records why).
