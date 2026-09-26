# Depth noise texture, training against the deploy path — 2026-09-17

This record set out to measure the robot's depth texture, match the training
distribution to it, and verify the match with the deployed v2 artifact as a
discriminator. It establishes instead that **the input it was to measure the
target from is not sensor depth**, that **the discriminator does not test what it
was being used to test**, and that consequently **no training-side noise change
is currently justified in either direction**.

One defect it found along the way is independent of all of that and is fixed
here: the observation delay buffer emitted exact `0.0` during its warm-up.

Everything measured is CPU — the production noise model, the production deploy
reduction and the v2 artifact under `onnxruntime` — except the gates in §9.
Setup, digests and machine-local inputs: [`provenance.md`](provenance.md).

---

## 1. The capture the target was to be read from is renderer depth

`goal-a-attribution-2026-08-22`'s robot-side observation capture
(`arm3-obs-capture/node_obs.jsonl`) was written by the inference node on
`strafer-nx`, but the depth in it came from the Isaac Sim ROS 2 bridge, not from
the D555. Three independent strands agree.

**Its own manifest.** `scene/task = Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`,
`scene/environment_seed = 42`, `scene/kit_log = kit_20260822_093054.log`,
`scene/bridge_pid = 830008`, `scene/cadence_contract = "publish 30.00 Hz sim,
frame_skip=3 (derived), bridge tick 120 Hz"`. The attribution record's own prose
calls it "noise-bearing sim depth" rather than sensor depth.

**An arithmetic exclusion that needs no metadata.** The real driver publishes
Z16 — integer millimetres — and `downsample_depth` reduces each block with a
median over 64 of them, so every non-sentinel output value must land on a 0.5 mm
grid. Measured over 60 frames spanning the capture, 216 000 pixels:

| | |
|---|---:|
| interior (non-sentinel) pixels | 206 846 |
| interior values on the 0.5 mm grid | **0.0019** |
| interior values on the 1.0 mm grid | 0.0020 |
| distinct interior values in the sample | 187 949 |
| near-field-fill share | 0.0318 |
| far-clamp share | 0.0106 |

0.19 % is the chance rate for continuous floats. A 16UC1-derived stream would
give 100 %. `probes/capture_provenance.py`.

**The encoding gate makes the alternative impossible.** `inference_node.py`
drops any depth frame whose encoding is not `32FC1`, and the driver publishes
`16UC1`, so on hardware the node drops every frame and never infers — the
standing P1 item
[`d555-depth-decode-validity`](../../tasks/active/trained-policy/d555-depth-decode-validity.md).
No converter bridges the gap in the deployed graph: `depth_downsampler` converts
but publishes 80×45 to a topic no policy-path node subscribes to, and
`downsample_depth` raises unless the frame is exactly 360×640. The node's depth
topic is never remapped.

**The same holds for the mission gate this line is chasing.**
[`goal-a-rig-gate-2026-08-17`](../goal-a-rig-gate-2026-08-17/README.md) ran on
the same bridge task and seed, at RTF 0.106, with `bad_encoding` 0. Its depth,
RGB, IMU, wheel encoders, odometry and actuation were all sim; the real hardware
in that loop was the Jetson compute, the TRT engine and the cable. That record
already attributes its command-tracking deficit to the **modelled** actuation
parameters rather than to the chassis.

**Every depth inference on record is bridge depth**: 4 387 (2026-08-02), 24 892
(the mission gate), 1 799 (the capture). There are no real-camera runs in either
repository.

The consequence for this record's own question is direct: **"the robot's depth"
as used across the depth-subgoal line means the node's pipeline run on renderer
depth.** That is a statement about the capture path, not about the sensor.

## 2. The deploy depth path has exactly one spatial operator

Re-traced from the subscription to the observation vector:

| # | stage | where |
|---|---|---|
| 1 | subscribe `/d555/depth/image_rect_raw` | `inference_node.py:182`, `:507` |
| 2 | encoding gate, `32FC1` or drop | `:801-806` |
| 3 | `np.frombuffer` + reshape, no filtering | `:808-810` |
| 4 | `isfinite` rescue → `max_depth` | `obs_pipeline.py:76` |
| 5 | **8×8 block median**, `reshape(45,8,80,8)` | `:77-80` |
| 6 | near-field fill below the near clip | `:81-83` |
| 7 | clip to `[0, max_depth]` | `:84` |

The block median at `:77-80` is the only spatial operation between the sensor
and the policy, and `:1355` is its sole call site. There is **no 3×3 median and
no smoothing** on the deploy side. The 3×3 median that exists is
`DepthNoiseModel._neighbourhood_median` — a **training**-side invalid-pixel fill
gated on `hole_probability`, landed by
[`depth-nearfield-convention-mismatch`](../../tasks/completed/depth-nearfield-convention-mismatch.md).
The two sit on opposite sides of the boundary.

One correction to the repository's own reasoning about this.
`strafer_perception/config/d555_params.yaml` disables the four RealSense
post-processing filters, and is cited elsewhere as why no driver-side smoothing
reaches the policy — but **no launch file, compose service, Dockerfile or
entrypoint loads it**. `perception.launch.py` includes `rs_launch.py` with an
explicit argument dict and no `--params-file`. The filters are off by
`realsense2_camera` default, not by that file. Same conclusion, void reason, and
a wrapper version bump could change it silently:
[`d555-params-file-inert`](../../tasks/completed/d555-params-file-inert.md).

## 3. What separates the capture's depth from the training noise

Both compared against the same clean observation-term output at the anchor pose,
scaled units, one frame. The near-field-fill class is 39.44 % of this frame and
is a **constant**, so each arm is given whole-frame and with that class excluded.

| residual | std | p95 \|Δ\| | high-pass p95 | high-pass ==0 | lag-1 x | lag-1 y | low band |
|---|---:|---:|---:|---:|---:|---:|---:|
| **whole frame** | | | | | | | |
| capture, tick 0 | 0.057722 | 0.027569 | **0.000499** | **0.887** | +0.863 | +0.542 | **0.8162** |
| training stereo σ_d 0.08 | 0.002019 | 0.004577 | 0.004617 | 0.208 | +0.010 | +0.020 | 0.0974 |
| training realistic tier | 0.002016 | 0.004622 | 0.004555 | 0.210 | +0.012 | +0.039 | 0.1003 |
| i.i.d. control, matched p95 | 0.013898 | 0.027752 | 0.027920 | 0.114 | +0.013 | −0.023 | 0.0881 |
| **near-field class excluded** | | | | | | | |
| capture, tick 0 | 0.008775 | 0.012597 | **0.000524** | **0.845** | +0.863 | +0.542 | 0.8162 |
| training stereo σ_d 0.08 | 0.002594 | 0.005684 | 0.005649 | 0.114 | +0.010 | +0.020 | 0.0974 |
| training realistic tier | 0.002591 | 0.005684 | 0.005603 | 0.116 | +0.012 | +0.039 | 0.1003 |
| i.i.d. control, matched p95 | 0.013886 | 0.027860 | 0.027711 | 0.119 | +0.013 | −0.023 | 0.0881 |

High-pass is `residual − median3x3(residual)`; the low band is the lowest radial
quarter's share of Hann-windowed power with DC excluded.
`probes/texture_structure.py`.

On valid pixels the capture's residual is **3.4× larger in total** and **10.8×
smaller in per-pixel texture**. It is 84.5 % exactly-on-its-own-median where
training sits at 11.4 %, which is the i.i.d. chance rate the matched control
measures at 11.9 %; its lag-1 autocorrelation is +0.86/+0.54 against +0.01/+0.02;
and 81.6 % of its power sits in the lowest spectral quarter against 9.7 %. So the
capture's residual is a smooth geometric field and the training term's is
statistically indistinguishable from white noise. **Amplitude is not what
separates them.**

### 3.1 Three statistics that mislead on these frames

- **The constant fill class dilutes any frame-level figure, and in opposite
  directions.** Left in, it *inflates* the capture's residual p95 — 0.012597 on
  valid pixels against 0.027569 whole-frame — while *deflating* training's,
  0.005684 against 0.004577, because training carries no constant mass of its
  own. Excluding the class therefore moves the two figures toward each other:
  the "capture residual is 4.2× the stereo Gaussian's" figure published earlier
  in this line is **2.2× on valid pixels**.
- **A plain `std` is meaningless on a mostly-exactly-zero distribution.** On mid
  and far pixels the static roughness std gives train/capture ratios of 1.11×
  and 1.05× — reading as agreement where the per-pixel texture differs by three
  orders of magnitude. Only the exact-zero share and upper quantiles on smooth
  pixels are texture statistics here.
- **A `std` that exceeds the p95 of the same absolute values** signals that a
  handful of class-boundary pixels own the variance, which is the capture's
  whole-frame case (0.0577 against 0.0276). Read it as a warning, not a number.

## 4. Candidate pipelines against the discriminator

Each arm's depth is spliced into the node's own tick-0 observation and run
through v2@998 on CPU, which is the design
[`depth-convention-fix-2026-09-13`](../depth-convention-fix-2026-09-13/README.md)
used. Production noise model, production deploy reduction. 7 seeds — 7, then
1 through 6; `mean off-goal` is the range across them and `rig` the min–max of 30. `rhp95` is the
p95 of the residual's high-pass. Rig class is off-goal in [−83°, −79°].

| candidate | p95 \|res\| | rhp95 | mean off-goal | rig |
|---|---:|---:|---:|---:|
| clean (reference) | 0.000000 | 0.000000 | −78.83° | **29–29** |
| stereo σ_d 0.16 @80×45 | 0.009577 | 0.009346 | −22.76 … −20.38° | 0–0 |
| stereo σ_d 0.08 @80×45 | 0.004788 | 0.004673 | −26.87 … −24.75° | 0–0 |
| stereo σ_d 0.04 | 0.002394 | 0.002337 | −31.45 … −29.52° | 0–0 |
| stereo σ_d 0.02 | 0.001197 | 0.001168 | −41.00 … −39.63° | 0–0 |
| stereo σ_d 0.0124 | 0.000742 | 0.000724 | −51.69 … −50.28° | 0–0 |
| stereo σ_d 0.008 | 0.000479 | 0.000467 | −61.16 … −60.01° | 0–0 |
| stereo σ_d 0.004 | 0.000239 | 0.000234 | −72.47 … −71.46° | 0–0 |
| stereo σ_d 0.002 | 0.000120 | 0.000117 | −77.64 … −77.11° | 1–4 |
| stereo σ_d 0.001 | 0.000060 | 0.000058 | −78.69 … −78.48° | 8–18 |
| realistic tier, shipped default | 0.004804 | 0.004680 | −26.79 … −24.72° | 0–0 |
| robust tier, shipped (v2's tier) | 0.009586 | 0.009298 | −23.40 … −21.27° | 0–0 |
| realistic + post-noise 3×3 median | 0.005319 | 0.005251 | −38.30 … −35.32° | 0–1 |
| reduction-emulated σ_d 0.08, block upsample | 0.000749 | 0.000723 | −51.67 … −50.45° | 0–0 |
| reduction-emulated σ_d 0.16, block upsample | 0.001499 | 0.001446 | −36.71 … −35.63° | 0–0 |
| reduction-emulated σ_d 0.08, bilinear | 0.004068 | 0.004097 | −72.26 … −70.84° | 0–1 |
| reduction-emulated σ_d 0.16, bilinear | 0.005225 | 0.005307 | −54.85 … −52.30° | 0–0 |
| bilinear upsample bias control, **no noise** | 0.002941 | 0.003012 | −93.12° | 0–0 |

No arm's mean off-goal moves more than 3.2° across the 7 seeds. Four rows have
a seed-dependent rig count: the two smallest-σ rows, which straddle the
boundary, and the post-noise 3×3 median and bilinear σ_d 0.08 rows, which reach
1 of 30 on some seeds. `probes/candidate_sweep.py`.

**A post-noise 3×3 median is harmful, not neutral.** It lowers per-pixel texture
as intended, but it *raises* the residual from clean — p95 0.004804 → 0.005319,
std 0.002110 → 0.007486, a 3.5× increase — because a median relocates genuine
depth-edge pixels a long way. It moves the training distribution away from clean
sim *and* away from any sensor model, and the deploy-equivalence that would
motivate it does not exist (§2).

**A zero-floored noise-scale randomization is settled by the ladder, not by a
separate arm.** Drawing σ_d per env or per episode from a range reaching down to
zero makes each frame one row of the ladder above, so the discriminator's verdict
on the mixture is the mixture of the rows' verdicts. Every row at σ_d ≥ 0.004 is
0/30 on every seed, and σ_d ≤ 0.002 is below any plausible sensor by more than an
order of magnitude, so any range wide enough to be a randomization contains
draws that score 0. The same argument covers pairing that randomization with the
post-noise median. Neither is measured as its own arm, and neither needs to be.

**Emulating the reduction is dominated by how the raw field is built, and the
6.45× figure is an artefact of the easy case.** The 64-sample median attenuates
i.i.d. noise by a std factor of **0.154796** measured over 4×10⁵ draws (1/k =
6.46; the asymptotic √(π/2n) = 0.156664 overstates the survivor by 1.21 %). That
factor is recovered by a block-constant upsample, which has no within-block
depth variation. Give the raw field realistic within-block gradients and the
median **selects** a surface rather than averaging. Measured through the
production reduction, with the upsample's own bias cancelled by differencing
against the same field reduced without noise:

| raw field | mean within-block std | surviving noise std | attenuation | upsample bias |
|---|---:|---:|---:|---:|
| block-constant upsample | 0.000000 m | 0.00192540 m | **6.446×** | 0.000000 m |
| bilinear upsample | 0.034101 m | 0.00311467 m | **3.985×** | 0.029528 m |

`probes/reduction_attenuation.py`. The block-constant row reproduces the
sampling law to 0.2 %, as it must. So the honest equivalence between emulating
the reduction and injecting directly at 80×45 is σ_d 0.08 ↔ **≈0.020**, not
0.0124 — and correspondingly the bilinear arm at σ_d 0.08 lands at −72° where
the block-constant arm lands at −51°. The bilinear upsample also carries its own
bias — with **zero noise** it already displaces the frame by 0.0295 m std and
sits at −93.12°, past the rig class on the far side — so that arm is not a clean
measurement of the reduction at all. Neither variant reaches the rig class.

## 5. The discriminator does not test data realism

The pre-registered acceptance this record was commissioned under asked that
training-pipeline depth put v2 in the rig class at 29/30 or better. That is
reachable, but only by perturbations carrying no per-pixel texture, and the
reference arms show why.

| perturbation | p95 \|res\| | rhp95 | mean off-goal | rig |
|---|---:|---:|---:|---:|
| per-frame depth scale σ = 1 % | 0.009489 | 0.000004 | −78.86 … −78.83° | **29–29** |
| per-frame depth scale σ = 2 % | 0.018978 | 0.000007 | −78.88 … −78.79° | 28–29 |
| per-frame depth scale σ = 5 % | 0.047445 | 0.000018 | −78.96 … −78.58° | 21–28 |
| per-frame depth bias σ = 12 mm | 0.003802 | 0.000000 | −78.92 … −78.63° | 15–21 |
| per-frame depth bias σ = 30 mm | 0.009506 | 0.000000 | −79.00 … −78.30° | 11–18 |
| per-frame depth bias σ = 60 mm | 0.019012 | 0.000000 | −78.96 … −77.63° | 10–18 |

A 1 % depth-scale error scores the clean maximum of 29/30 on every seed while
carrying **2.0× the shipped realistic tier's** residual amplitude. Uniform
featureless fields place the behaviour:

| uniform depth field | p95 \|res\| | mean off-goal | rig |
|---|---:|---:|---:|
| 0.2 m | 0.754528 | +13.03° | 0/30 |
| 1.0 m | 0.621195 | −143.02° | 0/30 |
| 3.0 m | 0.466667 | −71.85° | 0/30 |
| **5.0 m** | 0.800000 | **−79.13°** | **30/30** |
| **5.5 m** | 0.883333 | **−79.44°** | **30/30** |
| **6.0 m** | 0.966667 | **−80.45°** | **30/30** |

**A uniform far wall scores 30/30 — a perfect score the clean scene itself does
not reach.** The rig-class command is v2's response to a depth field that reads
as far and featureless: "no near obstacle detected". It is not a signature of
robot-like data.

The response is two-sided. Per-pixel content drives the command toward the
referent; low-frequency spatial structure drives it past the rig class the other
way (§4's noiseless bilinear arm, at −93.12°, is an instance). The rig class is a
narrow band between two opposite failure directions, which is why no single
amplitude law describes it — and why a single scalar does not either: the uniform
arms carry `rhp95` 0.005222, three orders above the value that separates the
scene-preserving arms, and still score 30/30.

So the criterion answers "does this depth field read as a featureless far
surface", and its answer is pre-determined for any per-pixel sensor model.
Since the rig-class response is itself the behaviour a retrain exists to remove,
requiring a new training distribution to reproduce it would conserve it.
**§8 replaces the criterion rather than reporting a pass or fail against it.**

## 6. What the one real-sensor measurement can and cannot settle

[`d555-invalid-pixel-statistics`](../../tasks/completed/d555-invalid-pixel-statistics.md)
§4 holds the only real-D555 noise measurement in the repository: per-pixel
temporal σ over 120 frames, Z16 640×360, bench pose in a deployment room. It
**cannot** be compared directly with the training term, because it is a
**raw-pixel** quantity and the training term injects σ_z i.i.d. **per policy
pixel at 80×45** with no reduction stage, while the deploy path puts a median of
64 raw pixels between them.

Made commensurable — for equicorrelated within-block noise, post-median σ =
raw σ · √(ρ + (1−ρ)·π/128), where ρ is the within-block spatial correlation:

| band | real raw σ p50 | post-reduction, ρ=0 → ρ=1 | training σ_z (f=673, σ_d=0.08) | crossover ρ | σ_d equivalent, ρ=0 → ρ=1 |
|---|---:|---:|---:|---:|---:|
| 0.4–1.0 m | 1.1 mm | 0.17 → 1.10 mm | 0.61 mm | 0.293 | 0.0225 → 0.1435 |
| 1.0–1.5 m | 2.7 mm | 0.42 → 2.70 mm | 1.96 mm | 0.512 | 0.0173 → 0.1105 |
| 1.5–2.5 m | 10.1 mm | 1.58 → 10.10 mm | 5.00 mm | 0.227 | 0.0253 → 0.1614 |
| 2.5–3.5 m | 15.5 mm | 2.43 → 15.50 mm | 11.26 mm | 0.516 | 0.0173 → 0.1101 |
| 3.5–5.5 m | 87.1 mm | 13.64 → 87.10 mm | 25.34 mm | 0.062 | 0.0431 → 0.2750 |

`probes/sensor_commensurability.py`. The attenuation here uses the **asymptotic**
π/128 rather than the measured k² = 0.154796², which is the closed form the
crossover algebra inverts. The difference is immaterial to every reading this
table supports: the crossover ρ move by ≤ 0.001 and the ρ = 0 column by about
1 %, both far inside the interval the argument turns on. §4 uses the measured
factor, where it is the quantity under test rather than an algebraic convenience.

**ρ is measured nowhere, and it decides the sign.** The σ_d that would reproduce
the real sensor's post-reduction noise spans 0.017–0.043 px at ρ = 0 and
0.110–0.275 px at ρ = 1, and the shipped **0.08 px lies inside that interval in
every band**. Neither "training is smoother than the sensor" nor "the sensor is
noisier than training" is established. The measurement that closes it is
[`real-d555-depth-texture-capture`](../../tasks/active/trained-policy/real-d555-depth-texture-capture.md),
which also evaluates a revisit trigger the 2026-08-04 record pre-registered and
that has been dormant since, because that record reported raw σ only and the
trigger is phrased on the reduction residual.

Two adjacent readings that do **not** survive checking, recorded so they are not
re-derived:

- **The focal-length convention is a reparameterisation, not a defect with a
  fix.** σ_z depends only on σ_d/(f·B), so f = 336.5 at σ_d = 0.08 gives
  2.502542×10⁻³ — bit-identical to f = 673 at σ_d = 0.16, which is the shipped
  robust tier. (The repository's own intrinsics give 671.30 px at 1280 width,
  335.65 at 640 and 41.96 at 80; the shipped 673.0 is 0.25 % off the first.)
  What is left is a **documentation** gap: a 1280-native disparity σ is being
  injected at a post-reduction pixel and the convention is not written down.
- **The sensor does not measurably degrade faster than z².** OLS on the five
  band midpoints gives b = 2.262 ± 0.256 with the far bin and **1.916 ± 0.209
  without it** (R² 0.963 and 0.977). The implied σ_d is flat at 0.110–0.161 px
  across the inner four bands and jumps to 0.274 px only in the far bin — a
  regime break in one texture-limited bin, which is how that record words it,
  not a changed exponent.

Also worth carrying: that record justifies its σ table as the sensor's **raw**
structure on the grounds that the post-processing filters are disabled in
`d555_params.yaml` — the file §2 shows is never loaded. The filter state during
that capture is therefore unverified, and a temporal filter would have biased
those figures **down**.

## 7. The delay-buffer warm-up, and the fix

`DelayBuffer` allocated `torch.zeros` and `reset()` zeroed it, so a ring slot
read before it had been written returned exactly `0.0` across the whole
observation. Nothing on either side of the boundary emits that depth: the
observation term floors at `DEPTH_NEARFIELD_FILL`, and the deploy reduction
writes the same fill below its near clip. Measured, 64 envs, a 10-step
post-reset window, both shipped tiers:

| tier / tree | all-zero env-steps, full reset | with a partial reset at step 5 | per-step all-zero envs | min emitted |
|---|---:|---:|---|---:|
| realistic, before | 56 | 90 | 39, 17, 0, … | **0.000000** |
| robust, before | 119 | 179 | 64, 39, 16, 0, … | **0.000000** |
| realistic, after | **0** | **0** | all zero | 0.996339 |
| robust, after | **0** | **0** | all zero | 0.992087 |

`probes/delay_warmup.py`.

The shipped latency is a per-env **range** — realistic `1 / (0,2)`, robust
`2 / (1,3)` — so the warm-up length varies per env, and the convention A/B never
exercised it because that harness sets `latency_steps = 0`. **v2's own tier is
robust**, where at step 0 every one of 64 envs emitted an all-zero 3 600-dim
depth vector. Exposure at the measured mean episode length of 164.8 control
steps is **1.225 % of frames at the robust tier and 0.619 % at realistic**, with
**29.7 %** and **20.1 %** of 48-step windows containing at least one. Those
frames reach the learner: the observation manager resets the noise model inside
`_reset_idx`, and the environment computes the observation after that, as its own
source comment says it must.

The fix primes the ring with the first frame an env delivers after its own
reset. The per-env mask is load-bearing: only terminated envs are reset, so
priming all of them would overwrite the in-flight ring history of the envs that
kept running and shorten their latency by a frame. Because the four modalities
share the class, depth, IMU, encoder and RGB latency are all fixed at once.

Properties, measured rather than argued:

- **The RNG stream is unperturbed.** The prime consumes no randomness;
  `torch.get_rng_state()` is bit-equal across a full reset, a partial reset and
  50 calls, and the drawn per-env delays are bit-identical to a reference draw at
  the same seed. Stated precisely: the RNG stream and every steady-state
  emission are bit-identical; full training **rollouts** are not, because the
  warm-up observation changes deliberately and so do the actions taken on it.
- **No golden moves.** The composition snapshot hashes the configuration object
  and this change adds no configuration field; both policy-layout goldens drop
  `noise` regardless. The depth fingerprint is likewise not re-recorded.
- **The latency distribution is untouched**, so the mean-preserving bands the
  contract tiers ship are unchanged. The only statistic that moves is the
  warm-up, from a value nothing emits to a repeat of a live frame — the emission
  the drop and hold paths already produce.
- **Mutation-proven.** Four of the five new tests fail against the previous
  behaviour; the other two are the non-regression guards, which pass either way
  by design. The check ran in a separate worktree rather than by swapping files
  in the working tree.

### 7.1 Six existing tests pinned the old behaviour and had to move

`test_sim/noise_models/test_delay_buffer.py` is a dedicated suite for this class
and it asserted the zero fill directly, in three places. Naming which assertions
moved, and to what, so the change is auditable rather than merely green:

| test | asserted | now asserts |
|---|---|---|
| `test_delay_buffer_exact_delay` (4 params) | the first `delay_steps` outputs are **zeros** | they are the **first frame** |
| `test_delay_buffer_reset_clears_history` | the first post-reset output is **zero** | it is the post-reset frame, **and no step of the pre-reset history is readable at any point in the new episode** |
| `test_delay_buffer_per_env_reset` | the reset envs read **zeros** | they read their own post-reset frame |

The middle one is the substantive rewrite. "The output is zero" was a weaker
claim about a stronger behaviour: zero is also what an unwritten buffer returns,
so the assertion could not distinguish "the history is gone" from "nothing has
arrived yet". It now fills the pre-reset buffer with values that cannot recur and
checks every step of the new episode against every step of the old history.

The third one's second assertion — that the envs which did **not** reset still
read the frame they wrote a step ago — is unchanged and is the one that catches
the mistake this fix could have made. Its first assertion moved.

`test_delay_buffer_device_batch_size` gains two assertions rather than changing
one: it described a warm-up its assertions never reached, and now checks the
first two emissions as well as the third. That makes it discriminating too, so
the module fails **12** against the previous behaviour — the six above plus this
one's six parameter combinations — where it failed 6 before the addition.

`test_delay_buffer_preserves_signal_content` passes unaltered, as do both
zero-latency passthroughs and every test in `test_observation_latency.py`; two
comments there described the warm-up as zero-filled and are corrected. Nothing
else in the 222-test contract gate moved, which is the evidence for the
no-golden-movement claim above.

## 8. Pre-registered acceptance for a future noise-parity change

Replacing the criterion §5 retires. None of it uses v2 as a gate.

**Gate 0 — data prerequisite, before any training-side change is written.** A
real-D555 bag per
[`real-d555-depth-texture-capture`](../../tasks/active/trained-policy/real-d555-depth-texture-capture.md):
640×360 Z16, ≥600 frames at each of ≥3 static poses at robot mount height, the
filters and depth auto-exposure pinned by explicit launch arguments and their
values recorded, and a matched sim capture through the 640×360 perception camera.
Pixels binned by their own depth into the five existing bands, with n per band
reported and the survivorship rule stated.

- **(A) Commensurability, reported not gated.** Per band, the per-pixel temporal
  σ of the raw stream and of the same stream after the production
  `downsample_depth`. Their ratio r gives ρ = (r² − π/128)/(1 − π/128).
- **(B) Noise amplitude — gate.** The injected σ at 80×45 lies within
  [0.5×, 2.0×] of the measured post-reduction σ in the four bands below 3.5 m and
  [0.33×, 3.0×] in 3.5–5.5 m. **σ_d = 0.08 may already pass, in which case the
  correct outcome is no code change.**
- **(C) Texture — the gate that matters.** On 80×45 frames, p95 of
  |d − median3x3(d)| and the exact-zero share of that high-pass agree within 2.0×
  between the real bag after `downsample_depth` and training frames after the
  noise term, per band. Both sides are measurable with no policy in the loop.
- **(D) Invalid-convention precondition.** The deploy path must not map
  majority-invalid blocks to the near fill. On the 2026-08-04 capture 36.06 % of
  raw pixels are invalid, Z16 invalid is a finite `0` so the `isfinite` rescue
  does not catch it, and 33.9 % of blocks carry ≥32/64 invalid — a 5.8 m error on
  roughly a third of the policy's depth pixels in that room, some 400× the
  far-band σ. It dominates any amplitude question and is owned by
  [`d555-depth-decode-validity`](../../tasks/active/trained-policy/d555-depth-decode-validity.md).
- **(E) v2 is descriptive only.** Report the off-goal distribution on the new
  training depth and on the real-bag depth side by side, with the uniform-field
  reference curve. No threshold.
- **(F) Non-regression.** At neutral settings the term draws no random numbers
  and leaves `torch.get_rng_state()` bit-equal; contract goldens change only in
  fields the change names, diffed by field name.
- **(G) Falsifiability, stated up front.** If (A) returns a ρ high enough that
  the real post-reduction σ exceeds training's by more than 2×, **the change is
  an increase in σ_d, not a reduction.**

## 9. Gates

| gate | result | notes |
|---|---|---|
| contract gate — `test_sim/noise_models`, `test_sim/env/test_composition_contract.py`, `test_sim/env/test_obs_contract.py`, `tests/contracts/test_depth_nearfield_parity.py` | **222 passed**, 0 failed, 0 skipped, 180.6 s | boot watchdog, attempt 2 after a boot stall, 190 s wall |
| temporal texture — `tests/navigation/test_temporal_texture_dr.py` | **46 passed** (40 before), 5.3 s | pure, no Kit |
| delay buffer alone — `test_sim/noise_models/test_delay_buffer.py --noconftest` | **16 passed**, 2.7 s | no Kit; the module the six moved assertions live in. Not GPU-free — `test_sim.common.DEVICE` is `cuda:0` |
| mutation check — the new warm-up class against `9c4d674` | **4 failed, 2 passed** | the two passes are the non-regression guards |
| mutation check — `test_sim/noise_models/test_delay_buffer.py` against `9c4d674` | **12 failed, 4 passed** | the six rewritten assertions plus the six parameter combinations of the case that gained them (§7.1) |

Commands, from the repository root:

```
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 tools/kit_boot_watchdog.sh \
  --label contracts --log gate_contracts.log -- \
  <env_isaaclab3 python> -m pytest \
    source/strafer_lab/test_sim/noise_models \
    source/strafer_lab/test_sim/env/test_composition_contract.py \
    source/strafer_lab/test_sim/env/test_obs_contract.py \
    source/strafer_lab/tests/contracts/test_depth_nearfield_parity.py \
    -q --junitxml=gate_contracts.xml

LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 <env_isaaclab3 python> -m pytest \
  source/strafer_lab/tests/navigation/test_temporal_texture_dr.py -q
```

**Every contract golden holds**, which is the evidence for the claim in §7 that
this change moves none: the composition snapshot hashes the configuration object
and no configuration field is added.

The watchdog is not decoration here, and this record's own gate run shows why.
Run without it, the contract invocation stalled: the pytest tree accumulated
**0 s of CPU time over 32 minutes** and wrote no output past the launcher's
first Kit warnings — the 6.0.1.0 boot-stall signature the wrapper exists to
detect ([`kit-boot-hang-2026-09-11`](../kit-boot-hang-2026-09-11/README.md)).
Through the wrapper, the run this table reports caught the same stall in flight
and recovered from it:

```
attempt 1: STALLED during boot at 61s (rss=66724kB, no CPU or output for 60s) -> relaunching
attempt 2: exit=0 wall=190s
```

Anyone re-running §9 should use the wrapper rather than concluding the suite is
slow; the stall is intermittent, so a clean first attempt proves nothing either
way.

## 10. Hand-on

Filed from this record:

- [`real-d555-depth-texture-capture`](../../tasks/active/trained-policy/real-d555-depth-texture-capture.md)
  (Jetson, P1) — the measurement that settles ρ and unblocks §8.
- [`d555-params-file-inert`](../../tasks/completed/d555-params-file-inert.md)
  (Jetson, P2) — the unloaded params file.
- [`d555-depth-decode-validity`](../../tasks/active/trained-policy/d555-depth-decode-validity.md)
  amended with what the encoding gate blocks beyond deployment.

Recommended and not filed, for whoever owns the next step:

- Correct the 2026-08-04 record's justification for its σ table being raw
  structure (§6).
- Write down which resolution `disparity_noise_px` denotes (§6).
- A shared 80×45 texture statistic usable identically on a real bag and on
  training frames — §8(C) names it; it is the only depth property both lanes can
  measure today.
- Re-cost the parked 640×360-policy-camera option on the current Isaac Sim /
  Isaac Lab pair; its budget rejection predates the upgrade.
- Record in the attribution line that the rig-class command is a response to a
  featureless far field (§5), so the 0/6 is read as "no near obstacle detected".
- A `DEPOSIT.md` note that `same-pose-probe/gym_obs.jsonl` carries NaN at
  observation dims 10–13 and 16–18, so replaying it without splicing a real
  prefix returns a constant depth-insensitive action.

## 11. What this record does not establish

- **Whether today's training depth distribution is wrong at all**, and in which
  direction. §6 is explicit that ρ decides it and ρ is unmeasured. The retrain is
  not held on this, and nothing here predicts it will pass a mission gate.
- **Whether the real D555, once decoded, puts v2 near its rig class.** There are
  no real-camera inference runs to ask.
- **Whether the mission-gate failure has a depth-*distribution* cause.** Depth
  **content** is already established as causal, and this record does not disturb
  that: in
  [`goal-a-attribution-2026-08-22`](../goal-a-attribution-2026-08-22/README.md)
  depth is the only field whose substitution moves the command — every non-depth
  field swapped to its clean-sim value moves it by at most 0.570°, while depth
  moves it 95.8° and in both directions (node depth spliced into a
  noise-bearing sim row gives −82.58°, and noise-bearing depth spliced into the
  node row gives +13.51°, against the node's own −82.30°). What is open is
  whether the noise **distribution** is the part that matters, and that question
  is weakened from the other side too: the run had no real sensor and no real
  actuator in the loop, and its own record attributes the command-tracking
  deficit to modelled actuation parameters.
- **What the RealSense wrapper defaulted its filters to** during the 2026-08-04
  capture. The package is not installed on this host and the params file that
  claimed to set them was not loaded.

## Evidence deposit

Everything this record produced or read, other than this README and
[`provenance.md`](provenance.md), is deposited in the companion evidence
repository:

| | |
|---|---|
| repository | `https://github.com/zachoines/Sim2RealLab-Artifacts` (private) |
| deposit directory | `noise-texture-parity-2026-09-17/record-files/` |
| deposit commit | `c1323bd35d29b30a65f76d71449f65997720a03d` |

The deposit mirrors this record's own directory, so the paths this README names
resolve unchanged after restoring it:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/noise-texture-parity-2026-09-17/record-files/. \
    docs/measurements/noise-texture-parity-2026-09-17/
```

The restored files are not part of any commit. The digests check without
restoring anything:

```
cd Sim2RealLab-Artifacts/noise-texture-parity-2026-09-17/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
ab37d8003aa8c1fbd81cdf315fafd46b1ad75ba54f9a66ec62be3fa437006c12  gates/gate_contracts.xml
007b88b15dd96f604624047a966a06ee2a05da1dd64a8270d7a82a4a2c173359  gates/gate_delaybuffer.xml
7430441048eed0f1cd54bf333e94178677c6542fe08da24fc395dad2be4ecbdf  gates/gate_temporal.xml
e0d69ff0a75c2bc0835aa59d4327616ba640139fa69470feb271f43ccd8b4180  gates/mutation_delaybuffer.log
b98e9cfdafbe076754a4692df268ffa2d8fc9f51e19f40f429d4d7c315ed33fb  gates/mutation_premutation.log
394795e0711f18a890db2b2f9fe2bff054f90abd24ce740677651d5b3369b131  gates/watchdog_contracts.log
11e7888982b3f6b0b29f8233bbb51dc8d38099921e642e88364bbe254ab9278d  probes/candidate_sweep.py
5a1fe593b2b4488df7b650ba75b795176e056955cd5bc5303f9c65daa2dd4b57  probes/capture_provenance.py
050042b674fd41bcbb00d2478a7ccb1d835d154fc998da51cdb6aa39054286b3  probes/delay_warmup.py
8adc3dc911d2213b95be919ae1eff25f4026d798fc939087963d8b8ec1c669ba  probes/reduction_attenuation.py
fefb3fb64d8d487843c396f6c6dc458eba194459d59cd9f52bf8e628c14dba68  probes/sensor_commensurability.py
cb528b0b52cd87da89ac2840d1d0f3e774920916f0b32383554c2f2099e70daf  probes/texture_structure.py
8591105189bdd18abb40365fd982dd7609ee936c5cf9d97d22b84b627d78ba07  provenance/capture_provenance.json
74cd510ff8a1146ea98e3b77acebe19b96c12bf6fa74f9b2aabe914e3c2efddc  sensor/reduction_attenuation.json
3f3618473e68573546cbf6a41a7041bbd9c0dabbe054d9bdbb1f47f3d5748901  sensor/sensor_commensurability.json
97699a22de6314fbc3719aa33f0deef168f6dd9f833c6aeb663843c10220d57f  sweep/candidate_sweep.json
8434ac7c8b7d5baf7b4b6368882b9f534ec697b9d38689ab41cd47023172f362  texture/texture_structure.json
727f7d22c78c9222ab31ed6a6507a6b332c9a98a3004c3585d2267a5760b0b3f  warmup/delay_warmup_after.json
613de78b79971b537ba305e20e0cd24415a88a87d4538defa3f76026f28f5f60  warmup/delay_warmup_before.json
```

The inputs this record reads from other records are cited in
[`provenance.md`](provenance.md) by their own deposits, and are not re-deposited
here.
