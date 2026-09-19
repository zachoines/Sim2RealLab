# Deploy-resolution depth in training: the reduction, what it moves, and what it costs — 2026-09-19

The policy camera rendered 80×45 directly. The deploy node renders 640×360 and
reduces it with an 8×8 block median. The next rig gate is sim-bridge, so the
artifact is gated on the deploy field — and training on a separately rendered
one leaves the gate measuring a field the policy never saw.

This record ships the reduction: the policy camera renders the deploy resolution
and `mdp.observations.depth_image` applies the deploy node's block median, gated
on the rendered shape. It also sets the robust tier's
`disparity_noise_px_range` to (0.002, 0.16).

**No σ_d value, hole rate, drop rate, hold term or near-field convention changes
here.** The noise is injected after the reduction, at 80×45, exactly as before.
Nothing here calibrates σ_d against the real sensor; the within-block
correlation ρ is still unmeasured and this record does not need it. Hosts,
trees, interpreter and digests are in `provenance.md`.

---

## 1. What the change is

`make_d555_camera_cfg` binds `PERCEPTION_HEIGHT` × `PERCEPTION_WIDTH`, and
`depth_image` reduces with the deploy node's block median — after the
non-finite rescue, before the near-field fill, matching
`obs_pipeline.downsample_depth`'s ordering. The reduction is gated on the
rendered shape, so a field already on the policy grid passes through untouched
and the pre-change render stays reachable from a scratch config without a cfg
field selecting it.

There is no switch. 640×360 and 80×45 are both 16:9, so RTX derives the same
vertical FOV for either and the 8× block ratio is exact in both axes.

The 2026-08-01 rejection of the higher-resolution render in
`depth-camera-vfov-parity` is amended by §5 below: its throughput premise is
dead, and §4 shows its accuracy premise was wrong in the opposite direction from
the one it assumed.

## 2. The two reductions are the same operator, bit for bit

The training reduction and the deploy reduction are written in different array
libraries against different signatures, so their agreement is pinned
byte-for-byte rather than to a tolerance. Byte-equality is the right bar: the
reduction is exact arithmetic over the same 64 inputs, so any difference is a
different operator, not a rounding difference.

Over 120 independent random 640×360 float32 fields — plain, and carrying +inf,
NaN, −inf, a block straddling the near clip, and all of those at once — the
training term and `downsample_depth` agree on every one of 3600 policy pixels,
on every field. The dispatch's expectation was 20 trials; the probe runs six
edge classes × 20.

The divergence the test exists to catch is the even-count median. numpy averages
the two middle of 64; `torch.median` returns the lower-middle. Substituting
`torch.median` in the shipped term, in a detached worktree, fails the
byte-equality test on **3414 of 3600** policy pixels with a largest gap of
0.65 m — so the test is demonstrably sensitive to the one choice it pins.

`tests/contracts/test_depth_deploy_reduction.py` carries the byte-equality test,
the pass-through shape test, the mutation guard, and a pin on the camera
resolution and block ratio. §3 says why that last one has to be a test.

## 3. Golden movement: the camera moves nothing, and that is a gap

Pre-registration expected the camera's `height`/`width` to move on every
contract with a policy camera, and the depth-observation golden with it. **That
is wrong, and the direction matters.**

The composition contract's serializer hashes ten manager/scalar fields plus
`sim.dt`, `sim.render_interval`, `scene.num_envs` and `scene.env_spacing`. The
scene is never walked as a tree, so no sensor cfg — and therefore no camera
resolution — reaches any hash. The depth-observation golden hashes
`cfg.observations`, which carries the term's function, params, scale and noise,
none of which encode a resolution.

Measured with the #221 attribution walker, before and after:

| golden | moved |
|---|---|
| the 8 robust depth contracts | yes |
| the 8 realistic depth contracts | no |
| the 6 camera-less contracts | no |
| depth-observation golden | no |
| both layout goldens | no |

and the complete pooled per-field delta, over every golden, is one field:

```
added    observations.policy.depth_image.noise.disparity_noise_px_range.0 '0.002'   x8
added    observations.policy.depth_image.noise.disparity_noise_px_range.1 '0.16'    x8
removed  observations.policy.depth_image.noise.disparity_noise_px_range   None      x8
```

So §6's one-line tier change accounts for the whole movement, and the camera
change is invisible to the suite. Nothing unexpected moved, and both layout
goldens hold — the tensor a deployed checkpoint consumes keeps its shape, order
and scales.

The gap this leaves is real: the composition goldens cannot catch a silent
revert of the render resolution, and the term would then quietly stop reducing.
That is why the camera resolution and the exact 8× ratio are pinned by an
assertion in the new contract test rather than left to the hashes.

## 4. The reduction drift is not small — it is the dominant term

The 2026-08-01 rejection estimated the drift between the two renders as "an
order below the noise envelope". It is not.

Both fields are read from the same environment at the same tick, from two
cameras on the same body link with the same intrinsics, offset and clipping,
differing in resolution and prim path alone — so no scene, pose or seed
difference is in the comparison. Both then go through the observation term's own
stages, so the comparison is between observations. Figures are normalised
(metres ÷ `DEPTH_MAX`), and residuals bin each pixel by the surface it is on.

| band (m) | new vs old, median p95 | pixels | the capture's own residual, for scale |
|---|---|---|---|
| 0.4–1.0 | 0.000100 | 800 | 0.00000106 |
| 1.0–1.5 | 0.005874 | 338 | 0.00000025 |
| 1.5–2.5 | 0.041863 | 704 | 0.00007583 |
| 2.5–3.5 | 0.017535 | 793 | 0.00025480 |
| 3.5–5.5 | 0.019509 | 921 | 0.00145850 |

Whole frame, 0.018343. The two fields are **not** close: only 0.56 % of policy
pixels are bit-identical, and the largest single-pixel difference over 30 frames
is 0.351 normalised — 2.1 m. The per-band figures are stable across all 30
frames (the maximum and the median agree to three significant figures), so this
is structure, not noise: it is concentrated at depth discontinuities, where a
single ray per policy pixel lands on whichever surface it happens to hit while a
median of 64 returns the majority one. 81 % of the residual's high-pass is
exactly zero, which is the same statement.

The pre-change field also carried **less** per-pixel texture than the reduced
one: 0.000432 against 0.002389 on the same frame. The old render was not a
smoothed version of the deploy field; it was a differently-aliased one.

Two pre-registrations fail here and are reported rather than met:

- *New-vs-capture at the matched pose* was not obtainable. The 2026-08-22 anchor
  pose cannot be re-rendered — seed-42 room generation moved across the Isaac
  Sim pair flip — so new-vs-old is reported alone, as the dispatch allowed.
- *"≤ 0.000524 in every band"* is not a coherent threshold. 0.000524 is a
  whole-frame valid-only figure over 2180 pixels; the capture's own per-band
  residuals span 0.00000025 to 0.00145854, so the far band's baseline already
  exceeds it. The per-band figures above are reported descriptively, against the
  per-band capture column, and no pass/fail is claimed from them.

None of this is a defect in the change. It is the measurement of what the change
does, and it is larger than the noise question that motivated looking.

## 5. Cost: 1.128× at 96 environments

Twenty iterations on `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0`,
96 environments, seed 42, headless, through the boot watchdog. Following #221,
the figure is the mean of iterations 2–8; iteration 1 is warm-up.

| | collection | learning | iteration |
|---|---|---|---|
| #221 baseline, 80×45 | 25.485 s | 64.553 s | 90.039 s |
| #221 benched arm, 640×360 | 35.482 s | 67.521 s | 103.003 s |
| this change | 35.690 s | 65.896 s | **101.584 s** |

**1.128×** the baseline, and 0.986× the benched arm — so gating the reduction on
the rendered shape costs nothing against the ungated patch the 103.0 s figure
was measured on. Over iterations 2–20 the mean is 96.858 s and the spread is
89.68–116.96 s, which is the learning phase's own drift that #221 recorded; the
2–8 window is kept for comparability.

**This is established at 96 environments only.** The two arms scale differently
above it and nothing between 96 and 192 was measured.

No NaN appears anywhere in the run. Peak system memory was 87810 MiB of 124543
(#221's arm: 79174 MiB). The per-process figure is not comparable to #221's — it
was sampled by a different method here and undercounts — so only the system
figure is offered.

The first boot stalled and the watchdog relaunched it:

```
[kit-boot-watchdog smoke] attempt 1: STALLED during boot at 61s (rss=48344kB, no CPU or output for 60s) -> relaunching
[kit-boot-watchdog smoke] attempt 2: exit=0 wall=1990s
```

Six further relaunches occurred across the Kit suites. The stall is
`kit-boot-hang-2026-09-11`, unchanged by this record.

The export path runs on the smoke checkpoint: `obs_dim` 3619, `action_dim` 3,
`is_recurrent` true, ONNX and TorchScript written with a sidecar. The exported
artifact then drives the play environment for 60 steps, reading 3619 dims (19
scalar + 3600 depth) and returning finite three-dimensional actions throughout.
None of this is a training result; it is the proof that the reduction holds end
to end at the resolution the retrain will use.

`play_strafer_navigation.py` itself could not be used: it forces a visualizer,
and no visualizer backend is configurable on this host. It raises at environment
construction, before the policy or the depth path is reached, and does so
identically on `main` — the forcing is byte-identical there. The rollout above
is that script's rollout with the one line's effect removed.

## 6. The band on the robust tier

`disparity_noise_px_range=(0.002, 0.16)` on the ROBUST tier's depth camera, and
nowhere else. The realistic tier stays on a fixed σ_d and its goldens hold. The
draw is log-uniform, per environment, at reset, and last — the properties #221
shipped and its suites still gate.

The band exists because the deploy field's texture rises with depth in a shape
no single σ_d matches: the per-band matching σ_d spans about 400×. It is a
coverage decision, not a calibration.

`depth-noise-coverage-2026-09-18` §8 pre-registered that "direction A" enters a
retrain only once ρ is measured. That pre-registration does not govern this
change, by the same record's own scoping at §8: the form that needs ρ is
raw-resolution noise synthesised at 640×360 and then reduced, which is not what
this is. The form shipped here — render at 640×360, block-median in the term,
noise injected at 80×45 as today — is the one that record calls "the
configuration a retrain would use", and it carries no ρ dependency because the
noise never sees the raw grid. Raw-resolution injection remains gated on the
real-D555 capture and is untouched.

## 7. What the change broke, and what that says

The policy camera's consumers assumed its render *was* the policy grid. Four
places did, and none were pre-registered:

- The sim-in-the-loop capture adapter fed the raw render to the LeRobot writer,
  which validates the policy streams at 80×45 — so any bridge or teleop capture
  session would have raised on its first recorded frame. It now reduces through
  the same operator, which was lifted out of the observation term so there is
  one definition rather than two. Non-finite values are left alone for the
  capture, which records raw metres, so a fully culled block still records as
  culled.
- `rgb_policy` has no reduction. At the deploy resolution it is the perception
  camera's image, so the adapter now says that rather than letting the writer
  fail on a shape.
- The depth-noise integration suite built its geometric wall mask from the
  camera's own resolution and indexed the depth observation with it — 230400
  against 3600. The mask belongs on the grid the observation lives on.
- The perception-camera contract asserted the two cameras differ in size. They
  now differ in prim path and channel set alone.

Two of these were caught only by the Kit suites, which the pure suite cannot
reach. That is the lesson worth keeping: a change invisible to every
composition golden still had four consumers, and the goldens said nothing about
any of them.

`depth_obstacle_proximity_penalty` reads the raw policy camera and so now
computes over 230400 pixels rather than 3600. It ships inert (`weight=0.0`) and
its cost is already inside the 101.6 s above, so nothing is changed for it here;
its referent does become finer, which matters only if it is ever re-enabled.

## 8. A gate that cannot fail

Running the composition contracts as raw `pytest test_sim/env` reports exit 0
**regardless of failures**. Measured: with one golden deliberately broken, the
suite recorded 9 failures in its junit XML and exited 0. The cause is
`test_sim/conftest.py`'s `os._exit` teardown, which also truncates the terminal
summary, so neither the exit code nor the printed tail shows the failure.

`run_tests.py` is immune — it reads the junit XML rather than the exit code,
which is what #214 hardened it to do, and it correctly reported this record's
`depth_noise` failures. The exposure is any gate invoked as raw pytest against
`test_sim/`, including through the boot watchdog, which faithfully propagates
the 0. Every contract figure in this record was read from the XML for that
reason. This is not introduced here and nothing is changed for it; it is
recorded because the dispatch's own gate instruction would have produced a false
green.

## 9. Gates

| suite | result |
|---|---|
| pure python (`tests/`) | 1312 passed, 1 skipped |
| navigation (band + temporal) | 339 passed |
| composition contracts (`test_sim/env`) | 269 passed |
| full Kit suite (`run_tests.py all`) | 499 passed, 6 relaunches |
| byte-equality mutation | fails on 3414/3600 pixels, as required |

The full Kit suite first reported 497/499; the two failures were the wall-mask
grid in §7, and `depth_noise` re-ran 6/6 after the fix.

## 10. What is not claimed

- Nothing about the real sensor. Every depth inference on record is bridge
  depth, and no real-D555 depth capture exists.
- Nothing about ρ, and no calibration of σ_d. The band is coverage.
- Nothing about environment counts above 96.
- No training result. Twenty iterations is a path check.
- No verdict on the drift in §4. It is reported against the capture's per-band
  column descriptively; the pre-registered threshold was not a coherent one.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `deploy-resolution-depth-2026-09-19/record-files/` |
| deposit commit | `171bcceb90d1bbf106b556c5b57a45f99f38673c` |

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/deploy-resolution-depth-2026-09-19/record-files/. \
      docs/measurements/deploy-resolution-depth-2026-09-19/
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/deploy-resolution-depth-2026-09-19/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

The deposit's `DEPOSIT.md` carries the sha256 of all 111 files. The inputs this
record reads from other records are cited in `provenance.md` by their own
deposits, and are not re-deposited here.
