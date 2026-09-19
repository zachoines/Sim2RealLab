# Depth texture coverage: does training contain the field the deploy path delivers? — 2026-09-18

The 2026-09-17 record established that the deployed depth policy has never seen
real sensor depth, and that the v2 artifact's off-goal command is its response to
a **far, per-pixel-featureless** field rather than to a noise amplitude
([`noise-texture-parity-2026-09-17`](../noise-texture-parity-2026-09-17/README.md)).
The next rig gate runs against the same bridge, so it will deliver that same
featureless field. This record asks a coverage question that needs no policy in
the loop: **does the training distribution contain depth that featureless, and if
not, what would put it there?**

It answers yes-with-a-gap, sizes the gap, and lands the mechanism that closes it
— a per-environment band on `disparity_noise_px`, neutral by default and wired
into no shipped tier. It also re-costs the rejected option of rendering the
policy camera at the deploy resolution, on the canonical Isaac Sim pair, because
that rejection was made in 2026-08-01 on a version that is no longer installed.

**No shipped σ_d, hole rate or near-field convention changes here.** Three of the
pre-registered readings did not survive measurement; §10 says which, with the
numbers, rather than quietly applying them.

Everything is CPU except §8's training arms and §9's contract gate. Setup,
digests and machine-local inputs: [`provenance.md`](provenance.md).

---

## 1. One statistic, and the three conventions inside it

`strafer_shared.depth_texture` computes, on an 80×45 frame:

- `p95_abs_highpass` — the 95th percentile of `|d − median3x3(d)|`
- `exact_zero_share` — the share of that high-pass that is exactly zero

with the near-field-fill class excluded, optionally per depth band, and
optionally against a `reference` frame so the statistic describes a residual
rather than the frame. It is numpy-only, so the deploy lane can import it; the
training lane reaches it the same way.

Three decisions inside it are load-bearing, and on each the brief's prose and
its acceptance criterion asked for different things. The acceptance criterion
won — it is the one with measurements behind it — and each decision is pinned by
a test that a plausible alternative fails.

| decision | what was implemented | the alternative, and what it costs |
|---|---|---|
| **border** | 3×3 median with the border **replicated**, every pixel counted | dropping the one-pixel border keeps 3354 of 3600 and reads **0.00050021** where the deposit reads **0.00052390** — a 4.5 % shift, so "interior pixels" and "reproduces the deposit to four significant figures" cannot both hold |
| **units** | the caller's; `fill` and the bands are declared in the same units | the deposit's figures are **normalised**, not metres: 0.00052390 normalised is 0.0031434 m |
| **field** | the frame, unless a `reference` is given | the deposit's figures are **residuals** against a pose-matched clean render; a frame also carries the scene's own edges, which are the larger signal (§3) |

With the deposit's conventions the function reproduces
`noise-texture-parity-2026-09-17`'s `texture/texture_structure.json` **exactly**,
not merely to four significant figures:

| scope | p95 high-pass | high-pass ==0 | pixels |
|---|---:|---:|---:|
| valid only — measured | 0.0005238950252532957 | 0.8454128440366973 | 2180 |
| valid only — deposit | 0.0005238950252532957 | 0.8454128440366973 | 2180 |
| whole frame — measured | 0.0004988133907318077 | 0.8875 | 3600 |
| whole frame — deposit | 0.0004988133907318077 | 0.8875 | 3600 |

`source/strafer_lab/scripts/measure_depth_texture.py` reads a node capture, a
gym capture or an `.npz` and prints the same numbers per band, with n per band.

## 2. The coverage curve

The σ_d ladder through the production `DepthNoiseModel` on the 30 anchor frames
of the 2026-09-13 harness, holes off, against the bridge capture's own residual
at the pose-matched tick. Normalised units; per-frame statistic, median over the
30 frames.

| σ_d | valid-only p95 | whole-frame p95 | ≤ capture (valid) | ≤ capture (whole) |
|---:|---:|---:|:---:|:---:|
| 0.16 | 0.01130859 | 0.00938458 | no | no |
| 0.08 | 0.00565431 | 0.00469231 | no | no |
| 0.04 | 0.00282713 | 0.00234613 | no | no |
| 0.02 | 0.00141357 | 0.00117308 | no | no |
| 0.0124 | 0.00087643 | 0.00072729 | no | no |
| 0.008 | 0.00056543 | 0.00046924 | no | **yes** |
| 0.004 | 0.00028272 | 0.00023462 | **yes** | yes |
| 0.002 | 0.00014135 | 0.00011731 | yes | yes |
| 0.001 | 0.00007068 | 0.00005864 | yes | yes |
| 0.0 | 0.00000000 | 0.00000000 | yes | yes |
| **capture** | **0.00052390** | **0.00049881** | — | — |

**σ_d\* = 0.004** on the valid-only scope — the scope §1 requires, with the
near-field class excluded on both sides. The pre-registered expectation was
0.008, and 0.008 is exactly the **whole-frame** answer. The pre-registration
compared a whole-frame training figure (the 2026-09-17 sweep reports
`residual_highpass_p95` over the whole frame) against a valid-only capture
figure. Scope-matched, the ladder answer moves one rung.

The statistic is linear in σ_d — `p95 = 0.0706789 · σ_d` over these 30 frames —
so the ladder rung understates the match. **The σ_d that reproduces the capture's
texture exactly is 0.00741**, which is what the pre-registration's 0.008 is a
good estimate of. Read off the 1 024-environment draw of §7 instead of the 30
anchor frames the slope is 0.0713080 and the match 0.00735; the two differ by
0.9 %, which is the frame-to-frame spread of the statistic.

## 3. A frame statistic cannot answer this question; a residual can

Over all 1 799 frames of the bridge capture, the **frame** statistic reads
0.02288 m (p05 0.01864, p95 0.02509) — roughly seven times the tick-0 **residual**
of 0.00314 m. The difference is the scene's own geometry, which a 3×3 median
removes wherever a surface is closer than its neighbours. At the pose-matched
tick the capture frame reads 0.03817 m and the clean training render 0.03764 m:
the two agree to 1.4 %, because the geometry is the same. The residual between
them is 0.00314 m, an order of magnitude smaller than either.

That is why §2's ladder is read off residuals. The frame statistic is dominated
by what the room contains, so comparing a training frame's to a deploy frame's
compares two scenes, not two noise models.

The frame statistic is still the right thing to report per band, and there it
carries the finding:

| band | capture, frame mode | what it means |
|---|---:|---|
| 0.4–1.0 m | 0.00000121 m | featureless |
| 1.0–1.5 m | 0.00000000 m | featureless |
| 1.5–2.5 m | 0.00000089 m | featureless |
| 2.5–3.5 m | 0.05389738 m | geometry |
| 3.5–5.5 m | 0.03117562 m | geometry |

**Wherever the deploy path has a surface inside 2.5 m, the frame it hands the
policy has no per-pixel variation at all.** That is the region obstacle
avoidance happens in, and it is the region training never renders featureless.

## 4. No single σ_d covers the capture across bands

Residual statistic per band at the anchor pose, normalised:

| band | n | capture | σ_d = 0.16 | σ_d = 0.08 | σ_d = 0.008 | σ_d matching the capture |
|---|---:|---:|---:|---:|---:|---:|
| 0.4–1.0 m | 190 | 0.00000106 | 0.00050617 | 0.00025308 | 0.00002531 | **0.000334** |
| 1.0–1.5 m | 76 | 0.00000025 | 0.00106130 | 0.00053065 | 0.00005307 | **0.0000377** |
| 1.5–2.5 m | 535 | 0.00007582 | 0.00366820 | 0.00183407 | 0.00018341 | **0.00331** |
| 2.5–3.5 m | 613 | 0.00025481 | 0.00846310 | 0.00423155 | 0.00042315 | **0.00482** |
| 3.5–5.5 m | 766 | 0.00145854 | 0.01566937 | 0.00783469 | 0.00078344 | **0.0149** |

The per-band matching σ_d spans a factor of **about 400** (398 unrounded). The training term's texture
follows z² by construction; the capture's does not follow it at all — it is near
zero out to 2.5 m and then rises. So a single σ_d, at any value, is either far
too loud in the near field or far too quiet in the far field. That is an
independent argument for a **distribution** over σ_d rather than a better point
estimate of it, and it does not depend on the calibration question the real
sensor still owns.

## 5. At σ_d = 0 nothing else textures a frame

With `disparity_noise_px = 0` and a tier's remaining terms configured — holes,
frame drops, stream holds, camera failures — the valid-only high-pass p95 is
**exactly 0.00000000 on every one of the 30 frames**, on both the realistic and
the robust tier. The ratio to the capture is **0.0000**, so the pre-registered
"within 2× of the capture's" check **fails**.

Configured is not fired, and the two halves of the claim rest on different
evidence. The probe makes one forward call over a 30-environment batch, so the
frame-drop and stream-hold terms *cannot* fire — both re-emit a previous frame
and there is none — and the 0.001 camera-failure term fired on none of the 30.
**The hole term is the one this measures**, and it contributes nothing.
**The temporal terms are excluded by reasoning rather than by this measurement**,
and the reasoning is short: a repeated frame is a frame that already passed
through this statistic, and a failed frame is uniform at the far clamp. Neither
adds per-pixel texture; both are visible only across time.

The finding the check was there to produce: **no term other than the stereo
Gaussian carries any spatial high-pass at the 95th percentile.** The hole term
is spatial but too sparse to reach one: at the shipped rates it touches 1 % and
3 % of pixels, and the p95 of 2180 valid pixels is the 109th largest, so a hole
rate below ~5 % cannot move it.

| hole rate | valid-only p95 | ratio to capture | frames with any non-zero |
|---:|---:|---:|---:|
| 0.01 (realistic) | 0.00000000 | 0.0000 | 0.00 |
| 0.03 (robust) | 0.00000000 | 0.0000 | 0.00 |
| 0.05 | 0.00003380 | 0.0645 | 0.53 |
| 0.08 | 0.00007742 | 0.1478 | 1.00 |
| 0.12 | 0.00013291 | 0.2537 | 1.00 |
| 0.20 | 0.00530688 | 10.13 | 1.00 |

Reaching the capture's texture through holes alone would take a rate near 0.20,
twenty times the shipped realistic value and a different sensor model entirely.
**The band is the only lever on this statistic**, and at σ_d = 0 it delivers a
field strictly more featureless than the capture's — so the featureless end is
covered, and over-covered, rather than approximated.

## 6. The per-environment band

`disparity_noise_px_range: tuple[float, float] | None = None` on
`DepthNoiseModelCfg` and on the contract-level depth camera cfg, drawn per
environment at reset as `DelayBuffer.delay_steps_range` draws the latency.
`None` restores current behaviour.

The draw is **log-uniform**, `low · (high/low)^u` with `u ~ U[0,1)`. σ_d is a
scale parameter and the band of interest spans two decades; a uniform draw over
[0.002, 0.16] puts 99 % of its environments above 0.002 and only 1 % in the
quiet decade the deploy path actually occupies. Both ends must be positive, and
a band naming zero is refused rather than clamped — clamping would make the low
end a different law. §5 shows σ_d = 0 is strictly *more* featureless than the
capture, so nothing is lost: a low end of 0.002 already reaches a field a
quarter as textured.

| pre-registered property | result |
|---|---|
| neutral output bit-identical to `main`, 6 frames, both tiers | **yes**, CPU and CUDA |
| `torch.get_rng_state()` bit-equal to `main`'s after construction and two resets | **yes**, CPU and CUDA |
| a band pinned to one value reproduces the fixed σ_d output | **yes**, `max\|diff\| = 0`, **temporal terms live**, both tiers and both devices |
| a partial reset redraws only the envs it names | **yes** |
| the preimage diff names exactly the new field | **yes** — see below |
| layout goldens hold | **yes** |

The coefficient is held in double and cast at use, so a draw landing on a band
end divides exactly as the scalar path's Python float does; without that the two
paths differ in the last bit and the band stops being a reparameterisation.

**Sampling order is part of the claim.** The band draws **last** — at the end of
construction and at the end of `reset`, after the hold process and the delay
buffer have taken their own per-env draws. Drawn earlier it would shift theirs,
and a pinned band would then differ from the fixed path by up to 0.25 m with the
temporal terms live, even though the noise term itself was identical. Drawn
last, the hold-process parameters are unchanged by the band's presence and a
pinned band reproduces the fixed path bit-for-bit **with the temporal terms
live**, on both tiers and both devices. A test pins the order.

One fact the same probe settled and that the ordering has to accommodate:
constructing a `DepthNoiseModel` already consumed CPU randomness before this
change, because the hold process draws its per-env parameters there. So two arms
are only comparable if the build is seeded as well as the run; the probe and the
suite both do that. On CUDA those draws land on the device generator, so
`torch.get_rng_state()` reads unchanged there whatever the configuration — the
CPU comparison is the one with teeth.

**Composition contract.** Diffed by field name against the stored preimages, the
complete pooled delta over all 25 goldens is one added key:

```
added  observations.policy.depth_image.noise.disparity_noise_px_range  None  x16
added  policy.depth_image.noise.disparity_noise_px_range               None  x1
```

17 goldens move — the 16 depth-bearing contracts and the depth-observation
golden. The 6 NoCam contracts hold. **Both layout goldens hold**, because the
layout serializer drops any attribute named `noise` at every depth. Nothing
else changed, so the pre-registered STOP did not fire. This is the same
17-golden shape the 2026-09-13 convention fix recorded.

## 7. The draw law, and the floor it now clears

Pre-registered spec: uniform on [0, tier σ_d], with a floor of **≥ 10 %** of
drawn environments landing within 2× of the capture's statistic. The within-2×
window is σ_d ∈ [0.00367, 0.01469], from §2's linear law.

**Uniform misses that floor**, and by a margin that widens as the band widens,
because a uniform draw over a band spanning two decades puts almost nothing in
the quiet one. Measured over 1024 draws through the production model:

| draw | measured | analytic | floor ≥ 10 % |
|---|---:|---:|:---:|
| uniform [0, 0.16] — the robust tier's, and v2's | 6.05 % | 6.89 % | **missed** |
| uniform [0, 0.08] — the realistic tier's | 11.72 % | 13.78 % | met |
| **log-uniform [0.002, 0.16] — shipped** | **30.96 %** | 31.64 % | **met** |
| **log-uniform [0.002, 0.08] — shipped** | **37.21 %** | 37.58 % | **met** |

**The remedy pre-registered for a uniform miss — a point mass at 0.0 — moves the
number the wrong way.** The window's lower edge is σ_d = 0.00367, and a point
mass at zero adds weight strictly *below* it, where 2.34 % of uniform draws
already sit. A point mass of weight w multiplies the share by (1 − w): at
w = 0.10 the robust tier's 6.05 % becomes 5.45 %. It improves coverage of the
featureless end, which §5 shows is already over-covered, at the cost of the
quantity the floor is about.

The change actually made is the **law**, not a second mechanism: one line at the
sampler, the same single `torch.rand` per environment, no new configuration
field, and therefore no golden movement beyond §6's. A pinned band is exact
under the closed form, so the reparameterisation property is unaffected.

One pre-registered figure needed correcting rather than confirming: the analytic
share was given as ln 4 / ln 80 = 31.64 % for **both** shipped bands. That is
right for [0.002, 0.16], whose ends differ by a factor of 80; [0.002, 0.08]
differs by a factor of 40, so its analytic share is ln 4 / ln 40 = **37.58 %**,
and the measurement agrees at 37.21 %. Both clear the floor either way.

## 8. Rendering at the deploy resolution costs about 1.1x at 96 environments

On 2026-08-01 the option of rendering the policy camera at 640x360 and sharing
the deploy path's block reduction was rejected on budget, at "64x the
policy-camera render cost", on Isaac Sim 6.0.0.0 with the old Isaac Lab pin.
That pin is retired. Re-costed on the canonical pair, on a scratch branch that
is deposited and never merged: the camera at 640x360, a torch 8x8 block median
inside `depth_image` ordered exactly as `downsample_depth` has it, against the
shipped 80x45. The median is byte-exact against the deploy function over 20
random fields carrying frustum-cull infinities and a sub-near patch.

### What was actually benched

The arm injects the tier's stereo noise **after** the reduction, at 80x45 —
`ObsTerm(noise=...)` wraps the term's output, and `DepthNoiseModel` asserts an
80x45 frame. So the number below is the cost of **"render at 640x360,
block-median in the term, inject noise at 80x45 as today"**. That is the
configuration a retrain would use, and it makes the *clean* training field
identical to bridge depth by construction. It is **not** the form §8's
pre-registered decision was written against — raw-resolution noise synthesised
at 640x360 with the right within-block correlation and then reduced — which was
not run, and which is the form that needs ρ.

The arm also renders a 640x360 RGB channel the policy never reads, because the
composed scene's camera carries `rgb` alongside depth. That inflates it; the
shipped 80x45 arm renders the same unused channel at 1/64 the pixels.

### The numbers, at v2's environment count

Both arms: 96 environments, 8 iterations, `--headless`, through the boot
watchdog, GPU idle before each boot, both booting on attempt 1. Steady state is
iterations 2-8; iteration 1 carries the warm-up.

| | shipped 80x45 | 640x360 + median in term | ratio |
|---|---:|---:|---:|
| collection time, steady state | 25.49 s | 35.48 s | **1.392x** |
| learning time, steady state | 64.55 s | 67.52 s | 1.046x |
| iteration time, steady state | 90.04 s | 103.00 s | 1.144x |
| peak process memory | 46 748 MiB | 54 993 MiB | 1.176x |
| peak system used | 73 339 MiB | 79 174 MiB | 1.080x |

**The firm number is the collection ratio, 1.39x** — collection is the phase the
change touches. The iteration ratio's third digit is not firm: the learning
phase does identical work in both arms (the observation is 3 600 wide either
way) yet drifts from 80 s to 56 s and rebounds within each run, a spread of
about 10 % of an iteration. Pairwise per-iteration ratios run **1.05-1.31**, and
holding the learning phase at the baseline mean gives **1.11x**. So the honest
statement is **1.11-1.14x, call it about 1.1x**, for a 64x pixel count.

For a v2-equivalent 1 000-iteration run that is 25.0 h against 28.6 h at the
measured 1.144x, or 27.8 h at 1.11x. v2 itself ran 998 iterations in about
26.1 h at this env count, so the baseline arm reproduces the run it stands in
for.

**This is established at 96 environments only.** It should not be carried to a
larger environment count without measuring there — see below.

### Above 96 environments the two arms do not scale together

| env count | shipped 80x45 | 640x360 + median in term |
|---:|---|---|
| 96 | 90.0 s/iteration | 103.0 s/iteration |
| 192 | completes, **246.8 s** (2.74x its 96-env time) | completes, **1 168.3 s** (11.3x) |
| 384 | killed in scene construction | killed in scene construction |

At 192 the gap is **4.7x**, and it is **entirely in the learning phase**:
collection is 48.2 s against 69.7 s (1.45x, in line with 96), while learning is
198.6 s against 1 098.6 s. The learning phase does identical work in the two
arms. Worse for a memory explanation, the 640x360 arm's sampled peak system
memory at 192 is **lower** than the baseline's (114 183 against 122 223 MiB).

**So the cause of the 192-environment slowdown is not established by this
measurement**, and the earlier reading — that the ceiling is identical and set
by host memory the camera barely moves — is not supported. What can be said:
both arms complete one iteration at 192 and neither at 384; nothing between 96
and 192 was measured; and the 384 failures are not a camera fact — both die
during USD scene construction 49-51 s into Kit, with no render product ever
created and no training loop entered, killed at the host's memory limit.
A scene-memory ceiling, reached before the camera matters.

### Pre-registered decision

Direction A enters the retrain only if the wall-clock ratio is ≤ 2x **and** the
within-block correlation ρ has been measured. The ratio is about 1.1x at 96
environments, so the budget condition is met and the 2026-08-01 rejection does
not survive on its own terms. ρ has **not** been measured — that is
[`real-d555-depth-texture-capture`](../../tasks/active/trained-policy/real-d555-depth-texture-capture.md),
unexecuted — and without it raw-resolution noise cannot be synthesised
correctly: injecting i.i.d. noise at 640x360 and then medianing attenuates it
6.46x, under-injecting by that factor if the real field is correlated.
**So direction A as pre-registered does not enter the retrain, on the ρ
condition alone.** Nothing is implemented here and the scratch branch is
deposited rather than merged.

## 9. Gates

| gate | result | notes |
|---|---|---|
| pure suite — `source/strafer_lab/tests/` | **1308 passed, 1 skipped**, 104.5 s | 1265 passed, 1 skipped before; +21 statistic, +22 band |
| contract gate — `test_sim/noise_models`, `test_sim/env/test_composition_contract.py`, `test_sim/env/test_obs_contract.py`, `tests/contracts/test_depth_nearfield_parity.py` | **222 passed**, 0 failed, 0 skipped, 176.9 s | boot watchdog, **attempt 1, no relaunch**, 189 s wall |
| temporal texture — `tests/navigation/test_temporal_texture_dr.py` | **46 passed** | unchanged by this work |
| composition goldens, recomputed Kit-free | **25 of 25 reproduce** | the draw law and the sampling order are not configuration fields, so no golden moves beyond §6's re-freeze |

Mutation checks. Each was applied to the shipped module, the suite run, and the
module restored; the restored baselines are 21 and 22 passed.

| mutation | result | the named test that caught it |
|---|---|---|
| percentile 95 → 90 | 3 failed | `TestThePercentile::test_the_statistic_is_the_95th_percentile_of_the_absolute_highpass` |
| near-field exclusion dropped | 5 failed | `TestTheNearFieldExclusion::test_the_fill_class_is_excluded_by_default` |
| border replicate → zero pad | 4 failed | `TestTheBorderRule::test_the_border_is_replicated_not_shrunk` |
| border replicate → reflect | 2 failed | the same named test |
| draw law log-uniform → uniform | 2 failed | `TestTheDrawLaw::test_the_draw_is_log_uniform_not_uniform` |
| positivity guard removed | 3 failed | `TestTheBandIsAReparameterisation::test_a_non_positive_end_is_refused` |
| band drawn before the hold process | 2 failed | `TestTheBandIsAReparameterisation::test_a_pinned_band_matches_the_fixed_path_with_the_temporal_terms_live` |

**Watchdog accounting.** Every gate and both 96-environment training arms booted
on attempt 1. Two boot stalls occurred in the environment-count search: one on a
first version of that harness which passed `--attempts 1` and so reported a
stall as a failure to fit — the harness was corrected and the run repeated — and
one on the corrected baseline 384-environment probe, which relaunched and
completed on attempt 2. Every attempt line is deposited.

## 10. What the pre-registration got wrong

Five pre-registered readings did not survive measurement, and are recorded here
rather than quietly applied. Two of the five were this record's own, corrected
in its fix round rather than by a later reader.

| pre-registered | measured | why |
|---|---|---|
| the statistic is read over **interior** pixels, and reproduces the deposit to four significant figures | cannot both hold | the deposit replicates the border and counts every pixel; dropping the border reads 0.00050021 against 0.00052390 (§1) |
| σ_d\* ≈ 0.008 | **0.004** on the mandated scope | 0.008 is the whole-frame answer; the pre-registration compared a whole-frame training figure with a valid-only capture figure. The *matching* σ_d is 0.00741 (§2) |
| if uniform misses the 10 % floor, a point mass at 0.0 is the smallest fix | it lowers the number | the point mass adds weight strictly below the window; changing the *law* to log-uniform raises it from 6.89 % to 31.64 % and is what shipped (§7) |
| log-uniform's analytic share is ln 4 / ln 80 for both shipped bands | right for one of them | [0.002, 0.16] spans a factor of 80 and reads 31.64 %; [0.002, 0.08] spans 40 and reads 37.58 % (§7) |
| the 640×360 arm costs 1.14× and the env ceiling is camera-independent | **1.11–1.14× at 96 envs**, and the ceiling claim is withdrawn | the two arms do not scale together above 96 envs, the 192-env gap is entirely in a phase doing identical work, and the 384-env kills happen in scene construction before any render (§8) |

Two held: the σ_d = 0 check produced the finding it was there to produce (§5),
and the preimage diff named exactly the new field — at two paths, ×16 and ×1 —
with the layout goldens holding (§6).

Three of the pre-registered document references did not resolve.
`depth-camera-vfov-parity.md` has no "Out of scope" section and never uses the
phrase "direction A"; the 640×360 rejection and its revisit trigger are in
`d555-invalid-pixel-statistics.md` §Out of scope, which is where §8's amendment
went. The attribution record spells its figure "0 of 6", not "0/6", and never
frames the bisection in terms of noise *amplitude* — its language was already
presence-versus-clean, so the amendment sharpens rather than corrects it. And
the same-pose probe lives under `goal-a-attribution-2026-08-22`, not under
`depth-convention-fix-2026-09-13`; its note is a sibling deposit rather than a
file added inside an immutable one.

## Evidence deposit

Everything this record produced or read, other than this README and
[`provenance.md`](provenance.md), is deposited in the companion evidence
repository:

| | |
|---|---|
| repository | `https://github.com/zachoines/Sim2RealLab-Artifacts` (private) |
| deposit directory | `depth-noise-coverage-2026-09-18/record-files/` |
| deposit commit | `6e1d523ccc4ff8a1962ba8f3197b2037fb1bf0a8` |

A second, smaller deposit at
`goal-a-attribution-2026-08-22/same-pose-probe-notes-2026-09-18/` carries the
note on that capture's NaN prefix. It is a sibling of the 2026-08-22 deposit
rather than a file inside it, because that deposit is referenced by a merged
record and is immutable.

The deposit mirrors this record's own directory, so the paths this README names
resolve unchanged after restoring it:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/depth-noise-coverage-2026-09-18/record-files/. \
    docs/measurements/depth-noise-coverage-2026-09-18/
```

The restored files are not part of any commit. The digests check without
restoring anything:

```
cd Sim2RealLab-Artifacts/depth-noise-coverage-2026-09-18/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
bb8d30c4ba542b998286d9648a7f5835e901e2a947d3a7c0ac6754fbb659cb83  bench/bench_direction_a.sh
aac2125965671c7bec62fca25c2c4438debad1875efd8bdb0d97bd6464b500a8  bench/ceiling_baseline_192.log
f28ffe04b9d0d332667cccf4cfb78797883a0b48431497ce9ac3a4dd9d3e80ca  bench/ceiling_baseline_384.log
79e2ad168bf94ddd95be329c3f7f008de05849cf2e0b8fb3059cafa5bc0d84e4  bench/ceiling_baseline.out
c3b838419d8374eca513fb348f693cbdd3c729e8b23bbeb47b0fe949486dd243  bench/ceiling_directionA_192.log
c0b1913bfae907cbe97870f997796df5231a82817c0a6991fecda56d64f98cb4  bench/ceiling_directionA_384.log
76e0469ee1976120b1033350f7547bf5297b8d9f14d5be5fb5ba181035e7e9c5  bench/ceiling_directionA.out
664f2c7fadefd24dacc216238f9e4e4c2a7c26e4553b9fc8ce23e4f0340a1a21  bench/ceiling_direction_a.sh
e0c644ec703423cf20738365b2a17d38fc36b8d975e09d369f34269f985982a4  bench/ceilmem_baseline_192.txt
7e07d9bba9e8ad613a64a27a924257205b0aed1bf5b8acf81360f271f338c4cb  bench/ceilmem_baseline_384.txt
07ee623195e5a6a90eeac006a70829783b9d1385e9580f2c80175cdaa8104319  bench/ceilmem_directionA_192.txt
17966ca3b95f8f577b6acecf3ed55d3c5ccfcb33f7acf39077d7410d79be4961  bench/ceilmem_directionA_384.txt
10e73652e6808037301d5cadf46e93f001c41cfd90354842dcacfbc976eeb41b  bench/ceilwd_baseline_192.log
4f217f57d931349092a716db534b731c3844e004e73cc8fb7d42b141cae3a87f  bench/ceilwd_baseline_384.log
17e0770ed9b29d636e600e6f6af73cd9d299a15bc878641deb59d045ff9095a4  bench/ceilwd_directionA_192.log
5163dac8439c40a60488ec57d2486e10a8a5e46d3f9013815e4c0e782cb6fb35  bench/ceilwd_directionA_384.log
69a1e614b86b44395be6b4d2966ce2a8d61c77ee598eb63639b95a560a0f96fd  bench/direction_a.patch
11b7b4881e1898d5a4a9583f1aa83f1faadc37fee948265147856a1c63007552  bench/mem_baseline80x45.txt
d7c57eed6625004b04578df3cc44041ad8898d62307a1e4cd3bfa1a8827f99f5  bench/mem_directionA640x360.txt
03dd8a1159b0085e6827f6694b4d371f27bb71c861307c6f8f97a9bcec1a5695  bench/summary_baseline80x45.txt
c459384baff2e88464c08b8b06c49833d74798c8c95ed329d918eef63f44ab6f  bench/summary_directionA640x360.txt
1294277c25f38fa4038aefc2a7cba122ed5e5dc632d627b2f2aee90e9a0ce0cb  bench/train_baseline80x45.log
7f80a9ebf345c18ec73f950be8ecca3dd99b1a482c5215116203a1a41f193b5e  bench/train_directionA640x360.log
563e03a97e6b3d2163f1c3966ec3f107a733bde7f1e9fc067e99c5d9c0705e47  bench/watchdog_baseline80x45.log
d78901095158ada16940830cb11ebac33be5189cb565deb6510a232effa6cb5e  bench/watchdog_directionA640x360.log
2068e742cb3bdda28f6fdf9f9576287f3cb11f4ec589400ee90bea376e507884  coverage/band/branch_cpu.npz
bbe963c1dbb0e52f889b14cc537d7231b4459150e4eb7228c97c8ad026d5fa02  coverage/band/branch_cuda.npz
2068e742cb3bdda28f6fdf9f9576287f3cb11f4ec589400ee90bea376e507884  coverage/band/main_cpu.npz
bbe963c1dbb0e52f889b14cc537d7231b4459150e4eb7228c97c8ad026d5fa02  coverage/band/main_cuda.npz
1bd87f0399baa1c375f3c34344d4e0feb84e0d9433a91386869c3f8db7d996cd  coverage/capture_frame_texture.json
c87f70b7aabd956b213a4f3b01f7a5f66a56e9aec71c8a21b4e21ad1e7013517  coverage/coverage_curve.json
a1c8274caac1449773f002f95ec840d3ae6009f306ed036b0e0b318dedc2963f  coverage/dr_share.json
267abc06b06c743db8f004ee41616f1cd319d0075f3b68c801b045b248061b2b  coverage/median_equivalence.json
127f2fd839ce464e67b2d21febf75c7e5dfc9cb8aa8a5f47b22cd8696b834a0b  coverage/samepose_replay.json
36dc5b6dc0f1c9e8b6d0caab47462085c08a22d8099ee1779dde082ceb3402a6  coverage/texture_statistic_acceptance.json
be6b40216ee3bb22e1d7eb1c9f0b5b2c76c2d51ee139a2deabae564c9b823a03  gates/gate_contracts_kit_console.log
c603fcea526112411fa78294111fcaa78a02505b28392485b6056cf9822cfb2f  gates/gate_contracts_stdout.log
ea051b81d476f007d84c87650ebe394d63b37a2ebd1b0c9b1528cddff3f8ba4d  gates/gate_contracts.xml
cb2c36ed96b5dfaef8f5d5d06cd0b746baedd8553e1df52abebf4c455d8f4353  gates/gate_new_suites.log
53337e8733ef6bc32fdce9998eaf7f6a77b63b2a8cdfc37fbc41d6f98aba84d6  gates/gate_new_suites.xml
bf1ccd7d50ca962842bbc4e0b68a582771f1adbbe54082449fe7be7702a7fdd2  gates/gate_pure.log
d2c54dd30ad01743c87648287c64ee0ce56b657f275a85ef215fc0e9e0b94a8c  gates/gate_pure.xml
abb575e230bc8c2edc87235eeb9988b7a0ec1d7a095f270149bb9b5df6e695cd  gates/gate_temporal.log
c3577940cf2755d0989ebfad53fa207eebc2214050315f8a714a6a99116e6de8  gates/gate_temporal.xml
3d8dc83d5515d1685be269b943d3ef2999e89e0c6851f7a17f922e92bcb87b42  gates/mutation_disparity_band.log
7aa3e4aeee3a22f9cdb359f7c4192009ceaa298e0b8d9397ca2fa3264e653b05  gates/mutation_texture_statistic.log
8712ee4207ce68192557b154420caa7dfb40991f25c8fad42cb4221af7a6a7ab  goldens/after/hashes.json
022a41e4807c2876400735c71219a6b69ae4eabc4988472e9e6d41502c9c7e31  goldens/after/preimages/contract-RLDepthEnriched_Real.json
57029c366aa3400627e06ad530168d12ba8376bc14650147109408b5a3895220  goldens/after/preimages/contract-RLDepthEnriched_Real_PLAY.json
871e7e9a4b4fe5d3e1bc036c259d2776f7fa9100b74df829e8bf00e8a475848d  goldens/after/preimages/contract-RLDepthEnriched_Robust.json
28544ca002419d4776d44e987833b19923926755669a62eb5a9721fc059547ff  goldens/after/preimages/contract-RLDepthEnriched_Robust_PLAY.json
e9452ec81d0272ed78b515c403bf2b2340f77052d2419219ac4251dbddf79ab7  goldens/after/preimages/contract-RLDepth_Real.json
51eced1aa04aa5bd5b2f4139d12a65368b50e2d1ae6ea3b22a0c6d7be93fd3a8  goldens/after/preimages/contract-RLDepth_Real_PLAY.json
49230e20e1e7b9f1c44df54b41e99b79ac12e3523963cdcae721d1585044e75b  goldens/after/preimages/contract-RLDepth_Robust.json
e314270a2cdadafcda584d67cdf3daca4c961a3033622d4ae6109cc2aec09ad4  goldens/after/preimages/contract-RLDepth_Robust_PLAY.json
09516c10984f3ad2d9c7993074634240ae030e540d16e2b67f7e4b6b5b35abab  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Real.json
75ce3c17354028d6fba7b383383ebcdd401941c5f02e67433374fa6f04906d18  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Real_PLAY.json
e4116b1af3db3c77fe8f561b229aeb3d9aa9e38a441bbb42b2e1690833163084  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Robust.json
91b12897e5c51b46a0ec100ae4c0ad4f5e2fd7798b5ac4d7657b0fe9f09ade1b  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Robust_PLAY.json
f8dbf1f4f88a74e7c3840b27eaa97803c2513ef978398b26a49e650ce0ad5504  goldens/after/preimages/contract-RLDepthSubgoal_Real.json
1480bd411e52d895697af1dacf567a85f9b3296a6d2a653ae8a40043478da6ab  goldens/after/preimages/contract-RLDepthSubgoal_Real_PLAY.json
349ace8f366c57641d05c415ca840b3cb9a2e1d7581a164115dd2895180d4b4a  goldens/after/preimages/contract-RLDepthSubgoal_Robust.json
f8adff1b8ba5a12cfd02b7953abec77e11c7b1b3d60070b62247f309457f40a5  goldens/after/preimages/contract-RLDepthSubgoal_Robust_PLAY.json
9ec306286b72e3db5b365559f791daada842c0ab54d694b02e216038aaf8c402  goldens/after/preimages/contract-RLNoCam.json
dfb9e45be05cc08391eb640204a4e64fbef9b2259ebef6785ac315bca1146902  goldens/after/preimages/contract-RLNoCam_PLAY.json
b17399b3e4506e5ea2496d9663eb5390042f325378688465de620c30910b6875  goldens/after/preimages/contract-RLNoCamSubgoal_Real.json
c88e515ea9fda493ab9ef81040561e499d26865d143f7e51cfcc42e24cd0be8d  goldens/after/preimages/contract-RLNoCamSubgoal_Real_PLAY.json
132be1a681068dbcfc777090bda66d280c04c3d868e334d9998d5c38f8e2bab5  goldens/after/preimages/contract-RLNoCamSubgoal_Robust.json
58d8b760c10aacaef63b455d75ec65cdc0e3682b0983e7061738cac5c1c0abba  goldens/after/preimages/contract-RLNoCamSubgoal_Robust_PLAY.json
88b7c0e4a1ac221feb5db9f245b3e8a20e04400d90a32ab706bf0f191a762bdc  goldens/after/preimages/depth_obs-RLDepth_Real.json
6c4cd4326c0d1ad5efb0737a708c39eadc40ea39802f005841317201c02a8e36  goldens/after/preimages/layout-depth.json
2a7e51cd9c4e0d6e2feee951ce2494c757f5df26e51c6dcdc76cb0134e5f4170  goldens/after/preimages/layout-nocam.json
c5cb7c7f3750ffb8a26be44d703c442d808e71e05d56ce62933936111bfadb86  goldens/before/hashes.json
0f1ec0458956793bd69028a9cf74bad77a806bc0585add2998b9d8d0a4aa3e86  goldens/before/preimages/contract-RLDepthEnriched_Real.json
6615b0a3021cce935cb6199be9a6b9d404001138ff0cfbff4504810f73b472ae  goldens/before/preimages/contract-RLDepthEnriched_Real_PLAY.json
d9109417089ccad39ae0f099d3de54d18d973e0f4068fe2196894cf9a01ec6bf  goldens/before/preimages/contract-RLDepthEnriched_Robust.json
2d1325f67d737398d4056729f215c48df87c1f804c8bfeec20cce8bf1befa0ed  goldens/before/preimages/contract-RLDepthEnriched_Robust_PLAY.json
59cfc1fce4a93095ff99f004836272bcb6b43225524181a97dd3db1c5b822246  goldens/before/preimages/contract-RLDepth_Real.json
2d141166c220bbf91876e2ce3a6e530290754749bad04ffe28c6c439a42c12c8  goldens/before/preimages/contract-RLDepth_Real_PLAY.json
dbebf9616149094c3112bb4b74ed7d54035375a643e9f2baae81156ebec210ad  goldens/before/preimages/contract-RLDepth_Robust.json
ecd30523975af7fd291acc407d29c3bcf4d28bfc7d37d88e48aed35fa685a3db  goldens/before/preimages/contract-RLDepth_Robust_PLAY.json
56e53425b0e922b95b304ef6b5aaec80a9de3c067824ca48b26a52642fb226e4  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Real.json
2a3085ba188c7e55dd54a541f47cecb75f0e36e541b1e1421919070f1b139a5e  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Real_PLAY.json
5fbbc180ea7d47527582f306cb1e1de163ca2a2656e79df871b3ac136eab87ab  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Robust.json
9bb384aa0ce9248045600f573e6ac985bd0bea70f6dfa11d72121f40ca8761db  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Robust_PLAY.json
d19a5d600bd759aa9cacdaaf9876b74e91c3be729cebda6fe6b16c013b4bfc3a  goldens/before/preimages/contract-RLDepthSubgoal_Real.json
8ad659d04bc6a120e0af87d38cb42c05b96b48e16887416d99d4d07cecbc342e  goldens/before/preimages/contract-RLDepthSubgoal_Real_PLAY.json
b0708d162ffa3d7d690f264a1fb0f2ffb3c66de32b3e30e5eda38bbff48adc2b  goldens/before/preimages/contract-RLDepthSubgoal_Robust.json
c18bcd56b998bb0396b615a9bbfadbf1cb0d4df84d5d97312d5c0fe764bc5e8c  goldens/before/preimages/contract-RLDepthSubgoal_Robust_PLAY.json
9ec306286b72e3db5b365559f791daada842c0ab54d694b02e216038aaf8c402  goldens/before/preimages/contract-RLNoCam.json
dfb9e45be05cc08391eb640204a4e64fbef9b2259ebef6785ac315bca1146902  goldens/before/preimages/contract-RLNoCam_PLAY.json
b17399b3e4506e5ea2496d9663eb5390042f325378688465de620c30910b6875  goldens/before/preimages/contract-RLNoCamSubgoal_Real.json
c88e515ea9fda493ab9ef81040561e499d26865d143f7e51cfcc42e24cd0be8d  goldens/before/preimages/contract-RLNoCamSubgoal_Real_PLAY.json
132be1a681068dbcfc777090bda66d280c04c3d868e334d9998d5c38f8e2bab5  goldens/before/preimages/contract-RLNoCamSubgoal_Robust.json
58d8b760c10aacaef63b455d75ec65cdc0e3682b0983e7061738cac5c1c0abba  goldens/before/preimages/contract-RLNoCamSubgoal_Robust_PLAY.json
b605f785f5a4ec301034ce73e5279e274f3b86faf05952c8005f2b5103097e65  goldens/before/preimages/depth_obs-RLDepth_Real.json
6c4cd4326c0d1ad5efb0737a708c39eadc40ea39802f005841317201c02a8e36  goldens/before/preimages/layout-depth.json
2a7e51cd9c4e0d6e2feee951ce2494c757f5df26e51c6dcdc76cb0134e5f4170  goldens/before/preimages/layout-nocam.json
a6b45a2e1ba521fc430359693eafd2707fad07216ed2a8ec9d8cc4b6334be437  probes/band_equivalence.py
53724edbbabf77d1bcc8a502950c726c36e5c634c2ccb6b366453b14d0eb736f  probes/capture_frame_texture.py
4ee96c8cc6029be47f44d3d2bf12ae694e945fb5a1cb90b091cf2e337f712dd0  probes/coverage_curve.py
6249f84d3fd61649071eb6460d3402b2089a4ba13f23565a11ad9ab1c1cf5b90  probes/dr_share.py
3b49094679c88771c1088e33a6cd3891da8570d7d589fa8d61db2b95f2fb050b  probes/golden_attribution.py
a2438b687164db12bb5766e12636b3c0f291300037ed37e5d358898c5c1c8088  probes/median_equivalence.py
d44ca113e9cc3f8d6cc9a72694ef1830501bbe2318f99ab62bc9b1f5ff55ba95  probes/samepose_replay.py
517130f1d7d14e7a47d65e94fe8d54f9782422ace1c8a322949cea3f1017b178  probes/texture_statistic_acceptance.py
```

The inputs this record reads from other records are cited in
[`provenance.md`](provenance.md) by their own deposits, and are not re-deposited
here.
