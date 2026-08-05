# Measure the real D555's invalid-pixel statistics on the bench

**Status:** MEASURED 2026-08-04 on the real D555 + NX in a deployment room.
**Ruling: the block median stands permanently.** Not because the variant is
dangerous — its true discriminating failure rate is 0.45% — but because its
entire claimed advantage is **two orders of magnitude below the sensor's own
temporal noise**, while it adds a failure mode the median structurally cannot
have. See "Measurement" below. The question is closed; the follow-up is not filed.

The capture was **uncalibrated** (camera on a desk in a deployment room). The
ruling deliberately rests only on the sensor-intrinsic measurements, which that
does not compromise — see "What this capture does and does not establish". One
earlier sim-versus-real claim drawn from the pose-dependent numbers has been
**retracted** there.

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

- [context/repo-topology.md](../context/repo-topology.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

`downsample_depth` reduces each 8×8 block of the 640×360 sensor depth to one
80×45 policy pixel. It ships a block **median**. A validity-majority +
centre-2×2 variant measured better against the training render — overall
4.92e-04 against the median's 6.19e-04 — and was **held** rather than shipped,
because its advantage rests on an assumption about invalid pixels that the
renderer guarantees and the real sensor may not:

- **Sim:** invalid depth is a coherent `+inf` region from the 50 m frustum cull —
  26.5% of pixels, all in image rows 0–22, with a clean boundary.
- **Real D555:** ~~NaN~~ **`0`** from specular surfaces, occlusion, and sub-0.4 m
  range — **corrected 2026-08-04 by measurement: the stream is Z16 and invalid is
  zero, not NaN.** Spatially scattered, and clustered on dark or glossy surfaces
  (confirmed). The NaN assumption has consequences beyond this brief — see "Two
  hard findings for the sibling 16UC1 item".

The held variant reads **4 of 64** source pixels, so it is only safe if invalid
pixels are rare *inside* that 4-pixel window. The median reads all 64 and
degrades either way. That is the whole of the open question.

Ruled 2026-08-01: ship the median, hold the variant behind this measurement.
The absolute stake is small — 6.19e-04 vs 4.92e-04 normalized is ~0.8 mm at 6 m,
both an order below the depth-noise envelope the policy trained under — so
robustness dominates optimality and there is no urgency.

## Acceptance criteria

- [x] Captured in the **actual deployment room**, not a bench corner —
      specular/dark clustering is environment-dependent and a clean-wall capture
      would answer the wrong question.
- [x] Fraction of invalid (NaN / zero) pixels per frame, over N frames.
- [x] **The decisive statistic: whether invalids cluster at 8×8-block scale.**
      Report the connected-component size distribution of invalid regions (or
      run-length distribution), binned so that "smaller than a block" and
      "comparable to or larger than a block" are distinguishable.
- [x] Per-pixel temporal σ, range-binned (binning by local range answered this from the scene as-is, without staging a flat-surface pair).
- [x] The D555's **sub-0.4 m behaviour** characterised in the same sitting —
      what the sensor actually returns below its stereo minimum (NaN, zero, or a
      wrong-but-finite value). The `nearfield_clip` / `nearfield_fill` contract
      in `downsample_depth` consumes this and is currently assumed, not measured.
- [x] **Decision recorded against the rule below**, and the follow-up either
      filed or the median confirmed permanent.
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. (This brief's own Context asserted the real sensor returns
      NaN; corrected in place. No other surface asserts the convention.) See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [x] No code change in this brief — it is a measurement.

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
  [`depth-camera-vfov-parity`](depth-camera-vfov-parity.md).
- The training-lane zero-drift option (render the policy camera at 640×360 in
  training and share `downsample_depth`). Rejected on budget 2026-08-01 — it
  removes a drift term already an order below the trained noise envelope, at 64×
  the policy-camera render cost against an env-count ceiling far below what
  training uses. **Revisit trigger:** if this brief's measurement shows the
  real-sensor reduction residual is materially larger than sim's, the option
  re-enters the retrain conversation.

---

## Measurement 2026-08-04 — and the ruling

**Setup.** Real D555 on the NX (`8086:0b56`, USB, `/dev/video0-5`), streaming
Z16 640×360 @30 through the shipped `perception.launch.py`. Post-processing
filters are disabled in `d555_params.yaml` (decimation / spatial / temporal /
hole-filling), so this is the sensor's **raw** invalid structure — hole filling
in particular would have masked exactly what this brief asks about.

**Pose caveat, up front.** The camera sat on a benchtop in a dining room looking
through a doorway into a living room: a real deployment room (the acceptance
criterion that matters), but **higher than a robot mount**, with bench clutter
and a seated person filling the near field. That is measured rather than
hand-waved — every result below is reported both whole-frame and conditioned on
the robot-relevant 1.5–6.0 m band, and the two differ substantially. A sunlit
window is present but does **not** dominate: it is the third-largest persistent
invalid region at 0.77% of frame, against 19.7% for the near workbench and 9.3%
for the seated person.

**Encoding (feeds the sibling 16UC1 item):** the stream is **Z16 — invalid is
`0`, not NaN.** A consumer testing `isnan` alone would read every invalid pixel
as a valid 0 m return.

### 1. Invalid fraction

**36.06%** of pixels (min 35.52, max 36.44, sd 0.18 pp over 120 frames), with the
heaviest concentration in the **bottom** rows: the per-band invalid *rate* is
65.7% across rows 330–360 against 23.9% across rows 0–30.

> **DO NOT COMPARE THIS TO THE SIM'S 26.5%.** An earlier draft did, and called
> the renderer's invalid structure "close to the inverse of the sensor's". That
> claim is **retracted**: it is an artefact of where the camera sat, not a
> sensor-versus-renderer difference.
>
> The camera was on a desk. The bottom rows are the **desk surface itself**,
> centimetres from the lens and below the stereo minimum — this capture's own
> numbers say so: 4.16% of pixels return finite values *inside* 0.4 m, and the
> largest persistent invalid region (19.7% of frame) is the near workbench. A
> robot-mounted camera would have floor there at a metre or more.
>
> Both the **fraction** and the **spatial distribution** of invalids are
> dominated by scene content and camera pose. Comparing them across two
> completely different scenes measures the scenes, not the sensors. See
> "What this capture does and does not establish" below.

### 2. Component sizes — the pre-registered statistic, and why it is ambiguous

| bin | components | share of invalid AREA |
|---|---|---|
| 1 px | 18 746 | 0.2% |
| 2–8 px | 17 904 | 0.6% |
| 9–63 px | 5 528 | 1.2% |
| 64–255 px | 1 454 | 1.7% |
| 256–1023 px | 340 | 1.6% |
| ≥1024 px | 448 | **94.7%** |

**By count, 95.0% of components are smaller than an 8×8 block. By area, only
2.0% of invalid pixels live in those components.** Read by count the rule says
SWITCH; read by area the same data says block-scale patches dominate and the
median stands.

**That is a defect in the rule, not the data**, and worth carrying forward: a
distribution this heavy-tailed cannot be summarised by "mostly". What a
reduction encounters is where the invalid *pixels* are, not where the
*components* are.

### 3. What actually discriminates the two reductions

Conditioned on the block's local range:

| range band | blocks with ≥1 invalid | centre-2×2 hit \| invalid | centre-2×2 ALL invalid \| invalid |
|---|---|---|---|
| 0.0–1.0 m | 62.9% | 56.7% | 34.3% |
| 1.0–1.5 m | 52.6% | 45.5% | 19.4% |
| 1.5–2.5 m | 20.9% | 40.3% | 19.0% |
| 2.5–4.0 m | 33.8% | 41.7% | 15.0% |
| 4.0–6.0 m | 30.2% | 42.7% | 14.4% |

But most blocks with a fully-invalid centre are blocks that are *mostly*
invalid, where the median degrades too — those do not discriminate. The case
that does: **block majority VALID (so the variant calls it valid) while the
centre-2×2 is entirely invalid.** The variant then emits a confident
wrong-but-finite depth; the median structurally cannot.

| | rate |
|---|---|
| **discriminating failure, all blocks** | **0.45%** |
| discriminating failure, robot band 1.5–6.0 m | **0.43%** |
| discriminating failure, near field <1.5 m | 0.81% |
| blocks with ≥32 of 64 invalid (median degrades too) | 33.90% |
| blocks with no valid pixel at all | 23.9% |

The variant is therefore **not** catastrophic — roughly 1 block in 230. (An
intermediate reading of this capture reported "32.7% garbage"; that was wrong,
having counted blocks hopeless for both reductions.)

### 4. Temporal σ — and why the question is moot

Per-pixel σ over 120 frames, on pixels valid in ≥80% of them:

| range | σ p50 | σ p90 |
|---|---|---|
| 0.4–1.0 m | 1.1 mm | 2.7 mm |
| 1.0–1.5 m | 2.7 mm | 5.8 mm |
| 1.5–2.5 m | 10.1 mm | 19.1 mm |
| 2.5–3.5 m | 15.5 mm | 32.8 mm |
| **3.5–5.5 m** | **87.1 mm** | **227.1 mm** |

**The variant's whole claimed advantage is ~0.8 mm at 6 m. The sensor's own
temporal noise is 87 mm p50 in the 3.5–5.5 m band** — the furthest band with
enough stable pixels to measure. Extrapolating that band's z² trend to 6 m lands
higher still, so quoting the measured band is the conservative form of the
argument: the margin is ~100×, and it widens with range rather than narrowing. The optimisation is invisible
against the noise it would have to beat.

### 5. Sub-0.4 m behaviour

The sensor **does** return finite sub-clip values: 4.16% of pixels fell in
(0, 0.4) m, spanning 0.244–0.399 m. `downsample_depth`'s `nearfield_clip` /
`nearfield_fill` contract is therefore **load-bearing, not defensive** — without
it those readings would pass through as genuine near-obstacle depth. Previously
assumed; now measured.

### Ruling against the pre-registered rule

**The block median stands permanently**, on three independent grounds:

1. **The advantage is unmeasurable** — 0.8 mm against 87 mm of temporal σ at the
   ranges where it would apply.
2. **It buys a failure mode the median cannot have** — 0.45% of blocks would get
   a confident wrong-but-finite depth. Small, but strictly worse than graceful
   degradation for no measurable gain.
3. **The pre-registered proxy, read by area, agrees** — 94.7%
   of invalid area sits in components of a block or larger.

Ground 1 is **tier 1** (sensor-intrinsic) and carries the ruling on its own.
Ground 2 is **tier 2** — right in direction, scene-dependent in magnitude.
Ground 3 is weaker still: its by-area dominance is driven by the same large
near-field regions this capture classes as **tier 3**, so it corroborates only
loosely. Both corroborate rather than decide. That split is deliberate:
this was an uncalibrated capture, and a ruling resting on tier-3 numbers would
not survive re-pointing the camera.

The follow-up switch is **not** filed. The question is closed.

### What this capture does and does not establish

An uncalibrated capture cannot support every kind of claim, and the difference
matters more than any single number here. Three tiers:

**Tier 1 — sensor-intrinsic. Robust to the pose; safe to build decisions on.**

- **Temporal σ versus range** (§4). σ grows as roughly the z² a stereo triangulator
  predicts — 1.1 mm at 0.7 m to 87 mm at ~4.5 m — because disparity quantisation
  scales that way regardless of what is in frame. The far bin degrades somewhat
  faster than z² alone, plausibly texture-limited. Scene affects the constant,
  not the law, and **the ruling rests on this tier**: a 100× margin over the
  variant's 0.8 mm advantage is not closeable by re-pointing the camera.
- **Sub-0.4 m behaviour** (§5): the sensor returns *finite* values below its
  stereo minimum. A contract fact about the device.
- **Z16 encoding: invalid is `0`, not NaN.** A device fact.

**Tier 2 — sensor × scene interaction. Direction is meaningful, magnitude is not.**

- Invalids form **coherent patches on textureless, dark and specular surfaces**
  rather than i.i.d. speckle. Which surfaces, and how much of the frame they
  occupy, is scene-dependent — but that stereo fails *in patches* is a property
  of stereo matching, and it is what makes a 4-pixel read window riskier than a
  64-pixel one. The 0.45% discriminating-failure rate is a **sample from one
  scene**, not a device constant.

**Tier 3 — scene/pose artefact. Cite for context only; do not compare across setups.**

- The **36.06% invalid fraction**, the **row/column profile**, and the
  **33.9% of blocks with ≥32/64 invalid** are all products of a cluttered desk,
  a seated person and a doorway. Change the pose and every one of them moves.
  Nothing sim-versus-real can be concluded from them.

**A calibrated capture would be needed** to make any depth-realism claim against
the renderer — matched geometry (a flat target at surveyed ranges, plus a
known-specular and a known-textureless sample), the same target replicated in
sim, and accuracy/bias reported alongside σ. That is a different brief. It is
**not** needed for the ruling above, and it is not on any current critical path:
the sim-bridge lane's depth comes from Isaac's renderer, so the real D555 is not
in that loop at all. File it if and when real-robot deployment becomes live.

### Also observed during the sitting

- **The IMU never published** — `imu_filter` logged `Still waiting for data on
  topic imu/data_raw` throughout. That is the separately-tracked IMU-QoS bench
  item, recorded here so the sitting need not be repeated to confirm it.

### Artifacts — deliberately not committed

The capture scripts and the scene/invalid-mask images stay on the NX under
`~/strafer_v2_validation/` and are **not** committed, for two reasons.

**The images are not publishable.** This repository is public, and the colour
frame and invalid mask are of a private home with a person in frame. Nothing in
the ruling depends on seeing them: what they establish — that the largest
persistent invalid regions are near clutter at ~2.35 m and a seated person at
~0.67 m, and that the sunlit window is only the third-largest at 0.77% of frame
— is stated numerically above and is what the argument actually rests on.

**The scripts were written for one capture, not as tooling.** They hard-code
this scene's geometry in their reasoning, so committing them would present
throwaway code as a reusable bench harness. The method is specified precisely
enough above to rebuild against a different setup: subscribe to
`/d555/depth/image_rect_raw`, treat Z16 `0` as invalid, take 8×8-connected
components of the invalid mask and report their size distribution **by area as
well as by count**, score the discriminating case (block majority valid while
the centre-2×2 is wholly invalid) conditioned on the block's local range, and
bin per-pixel temporal σ by range over pixels valid in ≥80% of frames.

A re-run at robot mount height, or the calibrated capture described above, wants
purpose-built tooling against its own geometry rather than a copy of this.

### Two hard findings for the sibling 16UC1 item (measured, not inferred)

The capture surfaced a deploy blocker that is more consequential than the
reduction question this brief exists to settle. Both facts are measured.

**1. The inference node cannot consume the real D555's depth topic at all.**

- The node subscribes to `/d555/depth/image_rect_raw` and **hard-requires
  `32FC1`**, incrementing `depth_bad_encoding` and `return`ing on anything else
  (`inference_node.py:688-694`).
- The real driver publishes that topic as **`16UC1`** — measured directly today.
- In the sim-bridge lane the Isaac publisher emits `32FC1`, which is why this has
  never bitten: arm 1's counters read `bad_encoding=0` against
  `inferences=4387`.

So on hardware the node would drop **100%** of depth frames and never infer.
Nothing bridges the gap: `depth_downsampler` does convert 16UC1→32FC1, but
publishes to `/d555/depth/downsampled` at 80×45, whereas the node needs
640×360 and runs its own `downsample_depth` (which asserts the full-res shape).

**2. Adding a 16UC1 decode path is not sufficient — the invalid convention
inverts.** `downsample_depth` rescues only **non-finite** values:

```python
depth = np.where(np.isfinite(depth), depth, max_depth)   # 0 is FINITE -> untouched
depth = np.median(...)                                    # block median
depth = np.where(depth < nearfield_clip, nearfield_fill, depth)   # 0 < 0.4 -> 0.2 m
```

Z16 invalid is `0`, which is finite, so it survives to the nearfield rule and
becomes **`DEPTH_NEARFIELD_FILL = 0.2 m`**. Sim invalid (`+inf`) becomes
`max_depth = 6 m`. **The same reduction maps invalid depth to "6 m away" in sim
and "0.2 m away" — an obstacle in the robot's face — on hardware.**

Scale, from this capture: **33.9% of blocks have ≥32/64 invalid**, so roughly a
third of the policy's depth input would read as near-obstacles. A policy trained
where invalid means "far" would meet an input where invalid means "blocked".

This is uncalibrated on magnitude (that 33.9% is a tier-3, pose-dependent
number) but the **mechanism is tier 1** — it follows from the encoding and the
code, not from where the camera pointed.

**Ask:** the 16UC1 item should cover both — a decode path *and* an explicit
invalid-mask contract carried through the reduction, rather than relying on
`isfinite`. Recommend the node take an explicit validity mask alongside depth,
so neither sentinel convention is load-bearing.
