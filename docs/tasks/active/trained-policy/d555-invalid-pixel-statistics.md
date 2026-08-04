# Measure the real D555's invalid-pixel statistics on the bench

**Status:** MEASURED 2026-08-04 on the real D555 + NX in a deployment room.
**Ruling: the block median stands permanently.** Not because the variant is
dangerous — its true discriminating failure rate is 0.45% — but because its
entire claimed advantage is **two orders of magnitude below the sensor's own
temporal noise**, while it adds a failure mode the median structurally cannot
have. See "Measurement" below. The question is closed; the follow-up is not filed.

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
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
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
  [`depth-camera-vfov-parity`](../../completed/depth-camera-vfov-parity.md).
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

**36.06%** of pixels (min 35.52, max 36.44, sd 0.18 pp over 120 frames), against
the sim's 26.5%. The *structure* differs more than the fraction: the renderer's
invalids are a coherent frustum-cull band in rows 0–22, while the sensor's are
scattered and concentrated at the **bottom** — 65.7% in rows 330–360 vs 23.9% in
rows 0–30. **The renderer's invalid structure is close to the inverse of the
sensor's.**

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
temporal noise there is 87 mm — 100× larger.** The optimisation is invisible
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
3. **The pre-registered proxy, read the honest way (by area), agrees** — 94.7%
   of invalid area sits in components of a block or larger.

The follow-up switch is **not** filed. The question is closed.

### Observations outside this brief's scope

- **The renderer's invalid structure is nearly the inverse of the sensor's**
  (sim 26.5%, coherent, rows 0–22; real 36.1%, scattered, bottom-heavy). Any
  real-robot deployment of a depth policy trained on this renderer inherits that
  gap — relevant to the v2 joint-distribution-brittleness synthesis.
- **A third of blocks are unmeasurable** in this room at this pose (33.9% with
  ≥32/64 invalid; 23.9% with none valid). That dwarfs the reduction question.
- **The IMU is silent** — `imu_filter` logged `Still waiting for data on topic
  imu/data_raw` throughout. That is the sibling IMU-QoS bench item presenting
  itself, captured here so the sitting need not be repeated.

Tools, kept for a re-run at robot mount height:
`~/strafer_v2_validation/tools/{d555_invalid_stats,d555_invalid_where,d555_variant_robustness}.py`.
Scene and mask images:
`~/strafer_v2_validation/logs/d555_benchtop_{color,invalidmask}.png`.
