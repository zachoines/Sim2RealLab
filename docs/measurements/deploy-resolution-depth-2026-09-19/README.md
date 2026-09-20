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

The two resolutions do get different post-processing, as a consequence of their
size rather than as a separate choice: Isaac Lab defaults `antialiasing_mode` to
DLSS at Performance, so the deploy-resolution product is configured for it while
an 80×45 product is skipped as below the 64×64 floor. **That does not reach the
depth annotator.** Rendering the same seeded scene at the same pose under
`sim.render.antialiasing_mode` "DLSS" and "Off" leaves `distance_to_image_plane`
bit-identical over five frames and 3600 policy pixels. The arms are
demonstrably separated: the "Off" arm emits no DLSS log lines against three, and
its colour mean shifts by about 2.8. So depth does not pass through the
upscaler, §4's comparison is unaffected by it, and the cost DLSS carries belongs
to the colour channel alone.

A first attempt is deposited and superseded. It set the knob on `SimulationCfg`,
where it is a stray attribute — the field lives on `RenderCfg` — so both arms
ran the default and the bit-identical depth was a same-setting repeat. Its
colour means differed by about 0.1, which is the run-to-run nondeterminism a
genuine separation has to clear.

The 2026-08-01 rejection of the higher-resolution render in
`depth-camera-vfov-parity` is amended by §5 below, on budget. §4 measures the
drift term that rejection estimated but never measured, and confirms it.

## 2. The two reductions are the same operator, bit for bit

The training reduction and the deploy reduction are written in different array
libraries against different signatures, so their agreement is pinned
byte-for-byte rather than to a tolerance. Byte-equality is the right bar: the
reduction is exact arithmetic over the same 64 inputs, so any difference is a
different operator, not a rounding difference.

Over 120 independent random 640×360 float32 fields — plain, and carrying +inf,
NaN, −inf, a block straddling the near clip, and all of those at once — the
training term and `downsample_depth` agree on every one of 3600 policy pixels,
on every field. The plan expected 20 trials; the probe runs six edge classes
× 20.

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

## 4. The reduction drift, measured with both cameras pointed the same way

The pre-change render and the reduced deploy-resolution render are compared from
the same environment at the same tick, over **30 distinct poses** — the robot is
driven with random actions and a frame pair taken every 8 steps.

**The first attempt at this measurement was confounded and its figures are
withdrawn.** The enriched tiers carry `jitter_d555_camera_prim`, which consumes
the sampled mount offset and writes it onto the shipped camera prim alone
(`sensor_name="d555_camera"`). The scratch comparison camera never received it,
so the two cameras differed by a rotation of up to the tier's mount band — 3° per
axis on robust — and not by resolution alone. Checking that the two cfgs matched
was not enough; an event rewrote one prim's pose after construction.

The probe now points the comparison prim with the same shipped function, and
carries a third camera left deliberately unpointed so the confound's size is
measured rather than asserted. Figures are normalised (metres ÷ `DEPTH_MAX`);
residuals bin each pixel by the surface it is on.

| band (m) | drift, median p95 | the capture's own residual | ratio | the confound, for scale |
|---|---|---|---|---|
| 0.4–1.0 | 2.98e-08 | 1.06e-06 | 0.028× | 6.13e-05 |
| 1.0–1.5 | 1.84e-06 | 2.48e-07 | 7.407× | 1.88e-03 |
| 1.5–2.5 | 2.18e-06 | 7.58e-05 | 0.029× | 2.21e-03 |
| 2.5–3.5 | 6.34e-05 | 2.55e-04 | 0.249× | 1.17e-02 |
| 3.5–5.5 | 3.46e-04 | 1.46e-03 | 0.238× | 3.00e-02 |

Whole frame, **1.69e-06** — about 310× below the 0.000524 the plan named, and
below the capture's own residual in four of the five bands. The deposited frame
pairs are rounded to 1e-07 normalised for size, so the two figures at or below
that scale — the 0.4–1.0 m band's 2.98e-08 and the bit-identical share — come
from the unrounded per-frame statistics in `reduction_drift_v2.json`, not from
recomputation over the pairs. The 1.0–1.5 m band
reads 7.4× the capture's, but that is the smallest capture figure of the five
(2.5e-07, essentially zero), so the ratio there carries little. Mean |Δ| across
the frame is **1.9 mm**.

**The confound was 67× the drift** on mean |Δ| (2.1e-02 against 3.2e-04
normalised), which is why the withdrawn figures read as large.

The drift is small in aggregate but **not uniformly small**: the median
worst-pixel difference in a frame is 0.70 m and the largest over 30 frames is
2.47 m, with 26 of 30 frames carrying at least one pixel past 6 cm. That is the
expected signature of a sampling change at depth discontinuities — a single ray
per policy pixel lands on whichever surface it hits, where a median of 64 returns
the majority one — and it is confined to edges: only 11.3 % of pixels are
bit-identical (unrounded), yet the typical difference is a millimetre or two.

The two fields also carry almost the same per-pixel texture on their own
(0.000194 reduced against 0.000168 direct), so the reduction is not adding
structure.

**This confirms the 2026-08-01 estimate rather than overturning it.** That
rejection called the reduction's drift contribution "a term already an order
below the trained noise envelope" without measuring it; measured, it is further
below than that. (Its separate "~0.8 mm at 6 m, an order below the depth-noise
envelope" sentence is about the gap between two *reduction variants*, not this
drift.) So this record fills a measurement gap; §5 is what amends the rejection,
on budget.

What that means for the change is not a weaker case but a different one. The
argument was never that the old field was badly wrong — it is that the gate runs
the artifact on the deploy field, and making training's clean field identical by
construction removes the question rather than bounding it. The measurement says
the change is faithful and safe, not that it was urgent.

Two further notes on what was not obtained:

- *New-vs-capture at the matched pose* was not obtainable. The 2026-08-22 anchor
  pose cannot be re-rendered — seed-42 room generation moved across the Isaac
  Sim pair flip — so new-vs-old is reported alone.
- *"≤ 0.000524 in every band"* was not a coherent threshold, and this is the
  plan's own scope error rather than a measurement problem: 0.000524 is a
  whole-frame valid-only texture statistic over 2180 pixels, used as a per-band
  drift ceiling. The per-band figures are reported descriptively against the
  capture's per-band column, and no pass or fail is claimed from them.

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
figure is offered. The sampler matched on `train_strafer_navigation.py`, a
pattern its own command line contains, so it never exited on its own; the series
runs about 700 s past the smoke and into the export boot, and the peaks above are
taken over the training window.

The first boot stalled and the watchdog relaunched it:

```
[kit-boot-watchdog smoke] attempt 1: STALLED during boot at 61s (rss=48344kB, no CPU or output for 60s) -> relaunching
[kit-boot-watchdog smoke] attempt 2: exit=0 wall=1990s
```

Seventeen relaunches occurred in all across this record's boots, every one the
same stall: `kit-boot-hang-2026-09-11`, unchanged here. Sixteen carry a resident
size of 48–66 MB; the export boot's stalled at **521 916 kB**, far past the
others, which is a different point in the boot reaching the same parked state
and is not otherwise accounted for. One suite (`commands`) exhausted all three
attempts and was re-run.

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

The policy camera's consumers assumed its render *was* the policy grid. Five
places did, and none were pre-registered:

- The **sim-in-the-loop capture** adapter fed the raw render to the LeRobot
  writer, which validates the policy streams at 80×45 — so any bridge capture
  session would have raised on its first recorded frame. It now reduces through
  the same operator, which was lifted out of the observation term so there is
  one definition rather than two. Non-finite values are left alone for the
  capture, which records raw metres, so a fully culled block still records as
  culled.
- The **teleop capture** did the same thing on its *default* invocation, which
  is worse: `--capture-policy-cam` defaults on, so a session with no flags
  requested the policy colour channel, launched Kit, loaded the scene and then
  died on frame one. It reduces the depth the same way.
- **`rgb_policy` has no reduction.** At one render resolution the policy
  camera's colour channel is the perception camera's image, so the token folds
  into `rgb_full` where stacks are normalised. Folding rather than refusing
  keeps the deprecated flag and any stored stack working. The LeRobot schema
  loses a colour column that would have been a byte-duplicate of the one it
  keeps.
- The **depth-noise integration suite** built its geometric wall mask from the
  camera's own resolution and indexed the depth observation with it — 230400
  against 3600. The mask belongs on the grid the observation lives on.
- The **perception-camera contract** asserted the two cameras differ in size.
  They differ in prim path and channel set alone. That contract also belonged to
  no `run_tests.py` suite and had never executed; its assertions need no Kit
  boot, so it moved to the pure tree.

Only the Kit suites could see two of these, and the pure suite could not see any
of them before the move. That is the lesson worth keeping: a change invisible to
every composition golden still had five consumers, and the goldens said nothing
about any of them.

`depth_obstacle_proximity_penalty` reads the camera directly and its docstring
is about what the policy senses, so it reduces too — restoring the referent it
had before the render changed. **It is not inside the cost in §5**: it ships
inert at `weight=0.0`, and Isaac Lab's reward manager skips zero-weight terms
before calling them, so it never ran during the smoke. What re-enabling it would
cost is unpriced, which the retrain brief records. The reduction is applied at
that call site rather than inside the shared function, whose contract is the
observation's: the penalty's geometry is resolution-general and is exercised on
grids of any size.

## 8. A gate that cannot fail

Running the Isaac Sim suites as raw `pytest test_sim/...` reports exit 0
**regardless of failures**. Measured in a detached worktree with one golden
deliberately broken, two arms differing in one kwarg:

| arm | teardown | junit failures | process exit |
|---|---|---|---|
| A | `simulation_app.close()` | 1 | **0** |
| B | `simulation_app.close(exit_code=exitstatus)` | 1 | **1** |

The attribution is the shutdown, not the teardown's own `os._exit`. That call
passes pytest's own status and is correct — a Kit-free mimic of the same
teardown exits 1. `SimulationApp.close` takes `exit_code: int = 0`, and with
`/app/fastShutdown` enabled Kit terminates the process before the `finally`
reaches `os._exit`. Isaac Sim documents exactly this: "Process exit status to
preserve when fast shutdown terminates the process. Nonzero values flush stdio
and exit with the supplied status before Kit's fast-shutdown path can replace it
with 0."

The truncated terminal summary is a **separate** effect of the same teardown and
survives the fix: the session hook runs before the terminal reporter's, so the
failure lines are never printed. Arm B exits 1 and still prints nothing after the
progress bar.

`run_tests.py` is immune and always has been — it decides from the junit XML,
which is what #214 hardened it to do, and it correctly reported this record's
`depth_noise` failures. The exposure is every invocation that is not
`run_tests.py`, including one wrapped in the boot watchdog, which propagates the
false 0 faithfully. Every contract figure in this record was read from the XML
for that reason.

The symptom has been on record since 2026-08-23: the Isaac Lab stage-3 record
logged a run showing 22 failures followed by `### pytest exit=0` and read that
exit code as evidence the bare invocation was healthy, listing "the invocation"
among the causes it excluded. Nothing is changed for it here; the fix and its
proof are filed as `kit-suite-exit-code-false-green`.

## 9. Gates

All at the measurement head, after the corrections in §7.

| suite | result |
|---|---|
| pure python (`tests/`) | 1356 passed, 1 skipped |
| full Kit suite (`run_tests.py`) | 499 tests, **1 failure**, 0 errors |
| composition contracts (`test_sim/env`) | 269 passed |
| byte-equality mutation | fails 3414/3600, as required |
| exit-code arms (§8) | A exits 0 on 1 failure, B exits 1 |

The one Kit failure is `test_collision_imu_mean_differs_from_free`, which is the
filed P3 flake `collision-imu-signal-flaky`: restitution-0 contact physics
leaves the collision mean riding the free-motion noise floor. It reproduced
twice here and it is not this change's — the deposit's preserved failure XMLs
carry the same test failing on 2026-09-13 and 2026-09-14, before this branch
existed, and nothing here touches the IMU path.

Two further first-run results were not this change's either and are recorded
rather than hidden. `commands` exhausted all three boot attempts and produced no
XML; re-run, 8/8. `noise_models` failed one hole-fill assertion once and passed
67/67 on re-run — the module is docstring-only in this branch's diff.

Every Kit figure is read from the junit XML rather than an exit code, for the
reason §8 gives.

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
| deposit commit | `5910c921464290c4d8501325e5a4a6f08d7fead7` |

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

sha256 of every file in the deposit:

```
7735c73d1beed89901152c5f4c246607acdcd66b4836978b6f3a85d509a7c6ed  dlss3/dlss_DLSS.json
7add20bc5856fb3fc48822e0c29c49024261eb6727d52960f043f8c8ec967808  dlss3/dlss_Off.json
e6818394c69e9ee382b4e2a821b744718f505f77ce995f5816823ba7ee3ca7d8  dlss3/stdout_DLSS.log
309bcc57d1ebc2e89e88f80ec8fa5722577ca5f86ae24a937afbd130f0b70ad2  dlss3/stdout_Off.log
c10a1784660bda97d5284f5d36b70147a53624cac76fc7c765f7a99bae30aa74  dlss3/wd_DLSS.log
c10a1784660bda97d5284f5d36b70147a53624cac76fc7c765f7a99bae30aa74  dlss3/wd_DLSS.log.attempt1
69bbdc54ccb04a68b50391b9f37659a1960ffbfec03186efac74df30ab497b85  dlss3/wd_Off.log
69bbdc54ccb04a68b50391b9f37659a1960ffbfec03186efac74df30ab497b85  dlss3/wd_Off.log.attempt1
968b338ac613662a1f8299b0a2484b61b8b718bc72d428dd86794065c54de790  dlss/dlss_DLSS.json
d6e3ddd51d59777a6999baaca02d09037fe8bb23a72a799b95c23e2ba1e7a600  dlss/dlss_Off.json
bcf3a3c3c346a85c8a3b737bf29e43ebe14990e08f547b2d9ae875bec54c39c0  dlss/stdout2_DLSS.log
e2a56d90534e61ef2e73af06540930a202fbf235066edfb302170069f17431b1  dlss/stdout2_Off.log
0a71f55f1995864d509dde699da55df1dfcadd1fbe37b90b7f3f29d3d8458976  dlss/stdout_DLSS.log
283531f84b951898d08358dea696755f6cbfd99cd39da10d889245fe1ce085fb  dlss/stdout_Off.log
2632b95cf59c507c975156992cdb6f64664141cf6cb98bf29125a31e9e5cd267  dlss/SUPERSEDED.md
8eb0130b4aeef7396ee710a83a0370bebece8e5cd4eeed0178305a046147f089  dlss/wd2_DLSS.log
8eb0130b4aeef7396ee710a83a0370bebece8e5cd4eeed0178305a046147f089  dlss/wd2_DLSS.log.attempt1
edfe72cb4b41b1c0ed83cbafb1d62949c44640ed46f1b3ee57159604d48d6203  dlss/wd2_Off.log
edfe72cb4b41b1c0ed83cbafb1d62949c44640ed46f1b3ee57159604d48d6203  dlss/wd2_Off.log.attempt1
4e5416b467b33fc43a45d8972ff59f1bf2c1a58d2d11982f9765e584b7a348b5  dlss/wd_DLSS.log
4e5416b467b33fc43a45d8972ff59f1bf2c1a58d2d11982f9765e584b7a348b5  dlss/wd_DLSS.log.attempt1
9bff26b2d54e0ef32602a60b4156e6dfed7be2216a0938e41eac31bfa1ca627a  dlss/wd_Off.log
9bff26b2d54e0ef32602a60b4156e6dfed7be2216a0938e41eac31bfa1ca627a  dlss/wd_Off.log.attempt1
ab6fe459ff3945f6d6a25f9406093c84ccca572bb4b6ac912a91604e30a15c25  drift2/drift2_stdout.log
1accaa0fe5ce2cd5cb224a9e9b3fffc380f3533aa756679690423634bf96430f  drift2/reduction_drift_v2_frames.json
790886abbb9b0d08e00815ab64f677157826d911ede64ee9b21fc2543dae9511  drift2/reduction_drift_v2.json
2935fc7d6660092389721f573dd5c8f327cd7d1f12dfcfb44c0594fc092f03f0  drift2/watchdog_drift2.log
2935fc7d6660092389721f573dd5c8f327cd7d1f12dfcfb44c0594fc092f03f0  drift2/watchdog_drift2.log.attempt1
ce76282fc1c22c2fd460e939e950f80213d0a49292b948e4305c515b9fdb20fb  drift/drift_stdout.log
9f520bf772c4bbe98aabe69bff1698ae26c438d663bef68beb353c55398f035a  drift/reduction_drift.json
0ac99f6788ec8d3c50cbc3bacf529825294d6cc8806f0cf00be1bfe4e56e451b  drift/watchdog_drift.log
0ac99f6788ec8d3c50cbc3bacf529825294d6cc8806f0cf00be1bfe4e56e451b  drift/watchdog_drift.log.attempt1
32802fe125cc17e10957ea5cbf79730935311141610c20324eb4a7b14411d76d  drift/WITHDRAWN.md
83cfaced794e74d2b5e8e190c0224fc9627fa3d1f22934a065041aa1ff08d3bd  gates/exitcode_armA.log
432d8fdc4fb03e2ddcdb7025c7a47c31137a65a75d7340ebd6e2204e8781b51c  gates/exitcode_armA.xml
202d19ae43d8dd7ecab34156b52031ccf55201fb9df89d875d57c705b0af59f7  gates/exitcode_armB.log
06923948b4759eca407376e2fab135005d624927e0ef8a29667a1d591ce20820  gates/exitcode_armB.xml
3d95877b6fc306bc8022744dccf85820dd5b5ff223fd3671a5a92e5fcad21567  gates/exitcode_probe_stdout.log
b84c027c775bbe60ed1bb91a2bdd75651dc3971df4a235b75efe55518d085d9e  gates/gate_contracts_kit.log
b84c027c775bbe60ed1bb91a2bdd75651dc3971df4a235b75efe55518d085d9e  gates/gate_contracts_kit.log.attempt1
ca32d821c7c5457c9a8c0c726df852033e7ae879a5af0d5fd85852713e01ac02  gates/gate_contracts_stdout.log
fedbd61d87b71cb1ed135ff0b9081dfc788166bbed0524d4096c7a50b5f8c82c  gates/gate_contracts.xml
8ab8f494bf9cc02435e97b46488d6dbb120f4eacc6bfa3df8f329e75473ddfa2  gates/gate_depth_noise.log
c0072299b5c34efdb27c7e192b60aa1a0b7a196d6c761aa0ea3126d7a897f38c  gates/gate_depth_noise_wd.log
c0072299b5c34efdb27c7e192b60aa1a0b7a196d6c761aa0ea3126d7a897f38c  gates/gate_depth_noise_wd.log.attempt1
b87f9f53552225dc0cf02b0c26de65b9d68030c33420c535967ed99ca9cdeaad  gates/gate_env_final.log
0a39999509661727762a0c3743d32f75c3326ad954b9cd111a6dd6678ca0283a  gates/gate_kit_all.log
7851fc33a7f38f084179611a50d7eb8e065ad0098babcfa136b946298607a3ed  gates/gate_kit_all_wd.log
7851fc33a7f38f084179611a50d7eb8e065ad0098babcfa136b946298607a3ed  gates/gate_kit_all_wd.log.attempt1
eb5de8f24dc65ecebcffc3c92ac5bc8253312897f764078990b8d2c21137d133  gates/gate_navigation.log
389dc162a46b2b9c42d741ea1aee45ed9aeb2eb0b03d7037f6658fa51cf55c65  gates/gate_pure.log
79ed8df8223a81a84d569d3cac3394d52cd0d011a7c4578764c020756694d17b  gates/gate_pure.xml
d6bc8cf7f63f441c44ff08b7b1d9cf996a59ebdd854fc2d91da7a467575bbc89  gates/gate_rerun3.log
f826b3a0d8d49bab15aa7ae661a69e1511ddad9713048fa67426b1cd1604f4b3  gates/gate_rerun3_wd.log
f826b3a0d8d49bab15aa7ae661a69e1511ddad9713048fa67426b1cd1604f4b3  gates/gate_rerun3_wd.log.attempt1
210948681c268d67c0c4c1736000954f16ae1c239c58cb5347499344a7704a88  gates/kit_xml/test_results_actions.xml
17c7366850ff58745c0d0e386ce336bce81369fbdddf648bf47ad1d238a7ec69  gates/kit_xml/test_results_camera_jitter.xml
aa2a09d0cf62a5a1eb42a4a944242041079c332e0b8e7acb3cf1a58cec4484d6  gates/kit_xml/test_results_commands.xml
2e9c01b4d11c0886602e718763fe381447f109a1fc52c1804835ac7eaedeb00d  gates/kit_xml/test_results_curriculums.xml
0703e06ff99923691e238c755b234a880fca5d30453a5618bf0dfce7d6419f90  gates/kit_xml/test_results_depth_noise_test_frame_drops.xml
a93357c53c9e0bd7414c0df289338da1e14a3d96851fdfde7dc2865934cc5b2f  gates/kit_xml/test_results_depth_noise_test_gaussian.xml
22b978dfde9c8e81a1bea3b20e3b517fb6ffb70d65dbe37f56eac34be9a2df79  gates/kit_xml/test_results_depth_noise_test_holes-FAILRUN-20260919-172048.xml
5894ea55cf1fd171563b8041cc28dc124dd1e27b601e4594efdc042f61cfa5fa  gates/kit_xml/test_results_depth_noise_test_holes.xml
5cb4db80d35d07b1db7d50b630297607717f9afb25cbb95caacb51fccd3120bd  gates/kit_xml/test_results_env.xml
b238b5acf1c1c8071e499ae06b54020d60ce88c37cd5738f5287c773f7c5c63a  gates/kit_xml/test_results_events.xml
098982decef8e85580beeb1d3fd6d58e33bb883dcc4b843483c942f2cae4a2ae  gates/kit_xml/test_results_imu_test_imu_collision-FAILRUN-20260913-212832.xml
a1912a53d8d392fa0578b15caa1789ab2d0c071e2babe9e47bd2ce751e569956  gates/kit_xml/test_results_imu_test_imu_collision-FAILRUN-20260913-213747.xml
76ae4049267391facb4b44f40eb986c041de7fdbe443cd6ebf540d0cfd802d5f  gates/kit_xml/test_results_imu_test_imu_collision-FAILRUN-20260914-210059.xml
3967cbc07c46a619436c5b33d487b4de45bcc040900e3e1baf7f6ede23d145c3  gates/kit_xml/test_results_imu_test_imu_collision-FAILRUN-20260919-201604.xml
087aa2ca2f7a7eca166953322922d09289c83a17f2f2b1cb4f8753830734bcca  gates/kit_xml/test_results_imu_test_imu_collision-FAILRUN-20260919-202529.xml
087aa2ca2f7a7eca166953322922d09289c83a17f2f2b1cb4f8753830734bcca  gates/kit_xml/test_results_imu_test_imu_collision.xml
7dd290dcd96ca8b2b727a796b27119df3c852b07e12785618a6be878f0b2624f  gates/kit_xml/test_results_imu_test_imu.xml
dd501d135c3c581432fe01d17338b5677990bdc19a79e464b685598588678bbc  gates/kit_xml/test_results_noise_models-FAILRUN-20260919-200306.xml
3c653951f623c7e0efb50402b3a1494e8aafed2078cb59136fc629a829f8c358  gates/kit_xml/test_results_noise_models.xml
5287e1f4958dc5a7ccc3d02083285b311a7f96deab44539375eff8f90f6ef6fe  gates/kit_xml/test_results_obs_dump.xml
78565fd3a1b460efc0da42baec0f633bc5963c67cf8d59db218efd16003fba02  gates/kit_xml/test_results_observations.xml
3caa8a39b8eb564e1f9e9b9dbab71b70fab5ec146fa4b1d831a9a5bec9dfd8d7  gates/kit_xml/test_results_rewards_test_collision_rewards.xml
0ccfa5ccccefade3cfe3251e2d8ccc5bfac03a3fd2104ae1fd0e4b6ca9e30d06  gates/kit_xml/test_results_rewards_test_rewards.xml
b464fb44ffb7a3fc4068e5ab426a1308c631531d7e611c5c28ec4055f32bd263  gates/kit_xml/test_results_sensors.xml
b7e77337dfed0272e22a6f45db87541df2eff65b820dcff96f0e1287b34073e3  gates/kit_xml/test_results_terminations.xml
380e6991a25693692facda79038efaf0d92b411e3227720b262376c5d819861c  gates/mutation_block_median.log
9f01a6271554bdf18a62c32d4eeb0ffbf0a6f34a8813527d15049f5b65ceaf2a  goldens/after/hashes.json
022a41e4807c2876400735c71219a6b69ae4eabc4988472e9e6d41502c9c7e31  goldens/after/preimages/contract-RLDepthEnriched_Real.json
57029c366aa3400627e06ad530168d12ba8376bc14650147109408b5a3895220  goldens/after/preimages/contract-RLDepthEnriched_Real_PLAY.json
1a8fb607245b2c3b41e0cc11c6b3a8d7f288d7f7d1b2c6a1cc8e146f288b62e0  goldens/after/preimages/contract-RLDepthEnriched_Robust.json
fa852dac0e54b6db1dc4184c22ddf25771feead4efc91ddfeacc31b6fff9fdf2  goldens/after/preimages/contract-RLDepthEnriched_Robust_PLAY.json
e9452ec81d0272ed78b515c403bf2b2340f77052d2419219ac4251dbddf79ab7  goldens/after/preimages/contract-RLDepth_Real.json
51eced1aa04aa5bd5b2f4139d12a65368b50e2d1ae6ea3b22a0c6d7be93fd3a8  goldens/after/preimages/contract-RLDepth_Real_PLAY.json
44d40b0db19b942daa597cdf952002dfe54549d016f31df0bfb06323f3edc0aa  goldens/after/preimages/contract-RLDepth_Robust.json
4a9c1ba4ac54104d66032bee65e4c9a9107ec0cf9f3126bdc09d2ec0e233c5d4  goldens/after/preimages/contract-RLDepth_Robust_PLAY.json
09516c10984f3ad2d9c7993074634240ae030e540d16e2b67f7e4b6b5b35abab  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Real.json
75ce3c17354028d6fba7b383383ebcdd401941c5f02e67433374fa6f04906d18  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Real_PLAY.json
86fe46d070f0e86adc487284246720b83cb42feabb563196bae37bf0b3a8852c  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Robust.json
4e29289981d1126f4da670e7ba046db84e1559ebc6a32a5643bc641460cf1858  goldens/after/preimages/contract-RLDepthSubgoalEnriched_Robust_PLAY.json
f8dbf1f4f88a74e7c3840b27eaa97803c2513ef978398b26a49e650ce0ad5504  goldens/after/preimages/contract-RLDepthSubgoal_Real.json
1480bd411e52d895697af1dacf567a85f9b3296a6d2a653ae8a40043478da6ab  goldens/after/preimages/contract-RLDepthSubgoal_Real_PLAY.json
cc18d913bd3d1294f7d27327c5b1e6cbdfde76ea767889184ea2e68f966f54e0  goldens/after/preimages/contract-RLDepthSubgoal_Robust.json
f6d9d59a8ff13ec395f3eba921f5f5223bd0971e3a6e51c9cd1dec9057022e0f  goldens/after/preimages/contract-RLDepthSubgoal_Robust_PLAY.json
9ec306286b72e3db5b365559f791daada842c0ab54d694b02e216038aaf8c402  goldens/after/preimages/contract-RLNoCam.json
dfb9e45be05cc08391eb640204a4e64fbef9b2259ebef6785ac315bca1146902  goldens/after/preimages/contract-RLNoCam_PLAY.json
b17399b3e4506e5ea2496d9663eb5390042f325378688465de620c30910b6875  goldens/after/preimages/contract-RLNoCamSubgoal_Real.json
c88e515ea9fda493ab9ef81040561e499d26865d143f7e51cfcc42e24cd0be8d  goldens/after/preimages/contract-RLNoCamSubgoal_Real_PLAY.json
132be1a681068dbcfc777090bda66d280c04c3d868e334d9998d5c38f8e2bab5  goldens/after/preimages/contract-RLNoCamSubgoal_Robust.json
58d8b760c10aacaef63b455d75ec65cdc0e3682b0983e7061738cac5c1c0abba  goldens/after/preimages/contract-RLNoCamSubgoal_Robust_PLAY.json
88b7c0e4a1ac221feb5db9f245b3e8a20e04400d90a32ab706bf0f191a762bdc  goldens/after/preimages/depth_obs-RLDepth_Real.json
6c4cd4326c0d1ad5efb0737a708c39eadc40ea39802f005841317201c02a8e36  goldens/after/preimages/layout-depth.json
2a7e51cd9c4e0d6e2feee951ce2494c757f5df26e51c6dcdc76cb0134e5f4170  goldens/after/preimages/layout-nocam.json
9b370d4ca9507a8a4c5d90901b2d7ae7d35440ade0373efbaac81b8fd19335ef  goldens/attribution.txt
8712ee4207ce68192557b154420caa7dfb40991f25c8fad42cb4221af7a6a7ab  goldens/before/hashes.json
022a41e4807c2876400735c71219a6b69ae4eabc4988472e9e6d41502c9c7e31  goldens/before/preimages/contract-RLDepthEnriched_Real.json
57029c366aa3400627e06ad530168d12ba8376bc14650147109408b5a3895220  goldens/before/preimages/contract-RLDepthEnriched_Real_PLAY.json
871e7e9a4b4fe5d3e1bc036c259d2776f7fa9100b74df829e8bf00e8a475848d  goldens/before/preimages/contract-RLDepthEnriched_Robust.json
28544ca002419d4776d44e987833b19923926755669a62eb5a9721fc059547ff  goldens/before/preimages/contract-RLDepthEnriched_Robust_PLAY.json
e9452ec81d0272ed78b515c403bf2b2340f77052d2419219ac4251dbddf79ab7  goldens/before/preimages/contract-RLDepth_Real.json
51eced1aa04aa5bd5b2f4139d12a65368b50e2d1ae6ea3b22a0c6d7be93fd3a8  goldens/before/preimages/contract-RLDepth_Real_PLAY.json
49230e20e1e7b9f1c44df54b41e99b79ac12e3523963cdcae721d1585044e75b  goldens/before/preimages/contract-RLDepth_Robust.json
e314270a2cdadafcda584d67cdf3daca4c961a3033622d4ae6109cc2aec09ad4  goldens/before/preimages/contract-RLDepth_Robust_PLAY.json
09516c10984f3ad2d9c7993074634240ae030e540d16e2b67f7e4b6b5b35abab  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Real.json
75ce3c17354028d6fba7b383383ebcdd401941c5f02e67433374fa6f04906d18  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Real_PLAY.json
e4116b1af3db3c77fe8f561b229aeb3d9aa9e38a441bbb42b2e1690833163084  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Robust.json
91b12897e5c51b46a0ec100ae4c0ad4f5e2fd7798b5ac4d7657b0fe9f09ade1b  goldens/before/preimages/contract-RLDepthSubgoalEnriched_Robust_PLAY.json
f8dbf1f4f88a74e7c3840b27eaa97803c2513ef978398b26a49e650ce0ad5504  goldens/before/preimages/contract-RLDepthSubgoal_Real.json
1480bd411e52d895697af1dacf567a85f9b3296a6d2a653ae8a40043478da6ab  goldens/before/preimages/contract-RLDepthSubgoal_Real_PLAY.json
349ace8f366c57641d05c415ca840b3cb9a2e1d7581a164115dd2895180d4b4a  goldens/before/preimages/contract-RLDepthSubgoal_Robust.json
f8adff1b8ba5a12cfd02b7953abec77e11c7b1b3d60070b62247f309457f40a5  goldens/before/preimages/contract-RLDepthSubgoal_Robust_PLAY.json
9ec306286b72e3db5b365559f791daada842c0ab54d694b02e216038aaf8c402  goldens/before/preimages/contract-RLNoCam.json
dfb9e45be05cc08391eb640204a4e64fbef9b2259ebef6785ac315bca1146902  goldens/before/preimages/contract-RLNoCam_PLAY.json
b17399b3e4506e5ea2496d9663eb5390042f325378688465de620c30910b6875  goldens/before/preimages/contract-RLNoCamSubgoal_Real.json
c88e515ea9fda493ab9ef81040561e499d26865d143f7e51cfcc42e24cd0be8d  goldens/before/preimages/contract-RLNoCamSubgoal_Real_PLAY.json
132be1a681068dbcfc777090bda66d280c04c3d868e334d9998d5c38f8e2bab5  goldens/before/preimages/contract-RLNoCamSubgoal_Robust.json
58d8b760c10aacaef63b455d75ec65cdc0e3682b0983e7061738cac5c1c0abba  goldens/before/preimages/contract-RLNoCamSubgoal_Robust_PLAY.json
88b7c0e4a1ac221feb5db9f245b3e8a20e04400d90a32ab706bf0f191a762bdc  goldens/before/preimages/depth_obs-RLDepth_Real.json
6c4cd4326c0d1ad5efb0737a708c39eadc40ea39802f005841317201c02a8e36  goldens/before/preimages/layout-depth.json
2a7e51cd9c4e0d6e2feee951ce2494c757f5df26e51c6dcdc76cb0134e5f4170  goldens/before/preimages/layout-nocam.json
470635387e358f882172426eb1ab590d9ce647a49c4ceffab1875c3068e0081f  probes/byte_equality.py
3f77aa7b684c50e01de6a8075664d876c830491fac0f122f1ba380c3198ca2cf  probes/dlss_depth_passthrough.py
48df3cba9f2d2451acff9f29293a27ec0b3167d74f675732ff583e604fc40487  probes/exitcode_arms.sh
eff0c5488c07265d65e502166cd89a4641fe2ecfc09110c64e806a15ef679568  probes/exitcode_conftest.py
1e35089d24254e2720a6cc09d23318aea5873b83fcef748082d0df9684096e23  probes/exitcode_test_probe.py
8fd3da8a37ccf8e357e16f20a74da1a54b4e746fa11c5ce8c555385c568e7ed5  probes/exported_policy_rollout.py
3b49094679c88771c1088e33a6cd3891da8570d7d589fa8d61db2b95f2fb050b  probes/golden_attribution.py
d0ec0f62947c1295b012a8c06581b326b21e61561807f3df37c5ada009825480  probes/reduction_drift.py
dace4f0b26df7c0613457ff00febc7237ce59056b5cb7ed8d11f592895d20515  probes/reduction_drift_v2.py
0d776455914382b1de99144fbc43f42bb3f596df958c03d3e1b6354ba7576ca6  probes/sample_memory.sh
ab6252ed3fae354d0ad92b8c6578e44c3d1c7cc5a60fa0bae22fabaef1e5e13a  smoke/ckpt_path.txt
535185a1b55f3904e58f4a3a49d0e13a6cd1ecae29459e7bff0f487ba34e28e3  smoke/export/depth_subgoal_v3_smoke.json
0bbdccb470de43290e51d793b2d6a397341a9d59ba13cc57c4eafbb4c20c354e  smoke/export/depth_subgoal_v3_smoke.onnx
8e61fe33f4f0c260b48e36bec71884eef05ba21ab8fffb6ef769b293a7f9002b  smoke/export/depth_subgoal_v3_smoke.pt
72fc786712833556daa0eede631135aa21711fb0849fac4dc15cf0af5b168858  smoke/exported_policy_rollout.json
a3f6ab944bab873b0efb5cec944d507a7505f531ec360e9849cf083dcb880baf  smoke/export.log
cd1bc53e3f1bff900f98b3271e1d21b92bc2be1ea97a2b13a5c7edbf9b1f9b3f  smoke/memory_smoke.txt
1cc34154ea520249e780bc05940d6aeb85ce8b3106452f988bee09689f990271  smoke/play.log
7dbd1fcd1c765c2d40c7e76caeb09e373a2396760378f5a5fbd1e6b8cb4aaf58  smoke/play_rerun.log
d83421889a89d617883c584b298605ae0ec459096fca430da6feffb0993ac732  smoke/play_viser.log
7f86ce278ba4bcc28fb0d2dcb7c0838553856bc79cf870e59075ae6efd15dcb8  smoke/rollout.log
f94161c3d225b79b7a2170aee083a1656663ba8d48de5eeec2cc864408dc14df  smoke/runs/run_20260919_174852/events.out.tfevents.1789858133.gx10-d1d8.555112.0
fb46a983e9bb3b98d7144ec68470d08196963075285f9685ab18a0eff705644f  smoke/runs/run_20260919_174852/model_0.pt
869c190bfad9c25bfb03fc70f19fd0377296ea4b458c7c551c04dbc794d248bd  smoke/runs/run_20260919_174852/model_19.pt
00603d7d90ebb2a81e45da9897529079ee5fd2bd0be73909e3554a646060f734  smoke/summary_smoke.txt
cf8689641e28316c6b597a4b6ed2648587f24f987e56a1d76a21c1f9fbc96041  smoke/train_smoke.log
173e14c7143b201f48db9c0afc7f4364b8ceaa44317dac68112c507f8b6324ae  smoke/watchdog_export.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  smoke/watchdog_export.log.attempt1
173e14c7143b201f48db9c0afc7f4364b8ceaa44317dac68112c507f8b6324ae  smoke/watchdog_export.log.attempt2
73bb2cebbd36b0f7241013de0d1a86ce1dac50a82965791faeed526d571ba1d7  smoke/watchdog_play.log
73bb2cebbd36b0f7241013de0d1a86ce1dac50a82965791faeed526d571ba1d7  smoke/watchdog_play.log.attempt1
a9732e50a33ce3daa2f02466ce4f2b632000e1089913be0f6b8a754790d54e96  smoke/watchdog_play_rerun.log
a9732e50a33ce3daa2f02466ce4f2b632000e1089913be0f6b8a754790d54e96  smoke/watchdog_play_rerun.log.attempt1
a42b9874f48df741d3e7cfafa5e12945fe6c980c63990bb0222063cf4203e21e  smoke/watchdog_play_viser.log
a42b9874f48df741d3e7cfafa5e12945fe6c980c63990bb0222063cf4203e21e  smoke/watchdog_play_viser.log.attempt1
8b7c13e55f8e2c11d9361583122ae5a65cc8843cb6107d88be5b09700b2edf34  smoke/watchdog_rollout.log
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  smoke/watchdog_rollout.log.attempt1
8b7c13e55f8e2c11d9361583122ae5a65cc8843cb6107d88be5b09700b2edf34  smoke/watchdog_rollout.log.attempt2
e6ae94155447a4d4fdf269077b8f5b2c9b752045e6cdcb2008edf5defbcc025c  smoke/watchdog_smoke.log
59813208d168f9e417f3baec0e869167106b7bdd99897307a37f0bfd1038e9b3  smoke/watchdog_smoke.log.attempt1
e6ae94155447a4d4fdf269077b8f5b2c9b752045e6cdcb2008edf5defbcc025c  smoke/watchdog_smoke.log.attempt2
```

The inputs this record reads from other records are cited in `provenance.md` by
their own deposits, and are not re-deposited here.
