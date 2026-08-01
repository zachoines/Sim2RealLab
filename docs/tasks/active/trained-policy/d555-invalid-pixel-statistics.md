# Measure the real D555's invalid-pixel statistics on the bench

**Type:** investigation (bench measurement)
**Owner:** Jetson
**Priority:** P2 — nothing is blocked on it; it decides one contained follow-up.
**Estimate:** S (~20 min of capture, riding an existing bench sitting)
**Branch:** `task/d555-invalid-pixel-statistics`

## Story

As the **deploy depth pipeline**, I need **to know how the real D555's invalid
pixels are distributed in space**, so that **the block reduction feeding the
policy is chosen on the sensor it actually runs against rather than on the
renderer it was tuned against.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

`downsample_depth` reduces each 8×8 block of the 640×360 sensor depth to one
80×45 policy pixel. It ships a block **median**. A validity-majority +
centre-2×2 variant measured better against the training render — overall
4.92e-04 against the median's 6.19e-04 — and was **held** rather than shipped,
because its advantage rests on an assumption about invalid pixels that the
renderer guarantees and the real sensor may not:

- **Sim:** invalid depth is a coherent `+inf` region from the 50 m frustum cull —
  26.5% of pixels, all in image rows 0–22, with a clean boundary.
- **Real D555:** NaN from specular surfaces, occlusion, and sub-0.4 m range.
  Spatially scattered, and plausibly clustered on dark or glossy surfaces.

The held variant reads **4 of 64** source pixels, so it is only safe if invalid
pixels are rare *inside* that 4-pixel window. The median reads all 64 and
degrades either way. That is the whole of the open question.

Ruled 2026-08-01: ship the median, hold the variant behind this measurement.
The absolute stake is small — 6.19e-04 vs 4.92e-04 normalized is ~0.8 mm at 6 m,
both an order below the depth-noise envelope the policy trained under — so
robustness dominates optimality and there is no urgency.

## Acceptance criteria

- [ ] Captured in the **actual deployment room**, not a bench corner —
      specular/dark clustering is environment-dependent and a clean-wall capture
      would answer the wrong question.
- [ ] Fraction of invalid (NaN / zero) pixels per frame, over N frames.
- [ ] **The decisive statistic: whether invalids cluster at 8×8-block scale.**
      Report the connected-component size distribution of invalid regions (or
      run-length distribution), binned so that "smaller than a block" and
      "comparable to or larger than a block" are distinguishable.
- [ ] Per-pixel temporal σ on a static flat surface at ~2 m and ~5 m.
- [ ] The D555's **sub-0.4 m behaviour** characterised in the same sitting —
      what the sensor actually returns below its stereo minimum (NaN, zero, or a
      wrong-but-finite value). The `nearfield_clip` / `nearfield_fill` contract
      in `downsample_depth` consumes this and is currently assumed, not measured.
- [ ] **Decision recorded against the rule below**, and the follow-up either
      filed or the median confirmed permanent.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No code change in this brief — it is a measurement.

## Decision rule (pre-registered)

| measurement | outcome |
|---|---|
| invalids are isolated speckle, components mostly smaller than an 8×8 block | switch to validity-majority + centre-2×2 as a contained follow-up |
| invalids form block-scale or larger patches | **the block median stands permanently**; record it and close the question |

## Scheduling

Rides the existing bench sitting alongside the 16UC1 encoding and IMU QoS chain
items — **not a standalone session**. (Those two items are tracked
coordinator-side; there is no brief for them in this repo to cross-link.)

## Investigation pointers

- The reduction under test: `strafer_inference.obs_pipeline.downsample_depth`.
- The sim-side comparison that produced the numbers above joins the preserved
  raw 640×360 bag frames against the preserved native 80×45 gym render; both
  live under `~/strafer_v2_validation/`.
- Sim invalid structure for contrast: 26.5% of pixels, all in rows 0–22,
  coherent boundary; straddling blocks are 2.42% of the image.

## Out of scope

- Changing the reduction in this brief. The switch, if the numbers call for it,
  is its own PR.
- Re-opening the depth *geometry* question — settled, see
  [`depth-camera-vfov-parity`](../../completed/depth-camera-vfov-parity.md).
- The training-lane zero-drift option (render the policy camera at 640×360 in
  training and share `downsample_depth`). Rejected on budget 2026-08-01 — it
  removes a drift term already an order below the trained noise envelope, at 64×
  the policy-camera render cost against an env-count ceiling far below what
  training uses. **Revisit trigger:** if this brief's measurement shows the
  real-sensor reduction residual is materially larger than sim's, the option
  re-enters the retrain conversation.
