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
`antialiasing_mode` "DLSS" and "Off" leaves `distance_to_image_plane`
bit-identical over five frames and 3600 policy pixels, while the colour
channel's mean moves — so the knob took effect and depth does not pass through
the upscaler. §4's comparison is unaffected by it, and the cost DLSS carries
belongs to the colour channel alone.

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
below the capture's own residual in four of the five bands. The 1.0–1.5 m band
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
bit-identical, yet the typical difference is a millimetre or two.

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
| pure python (`tests/`) | 1355 passed, 1 skipped |
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
| deposit commit | `b836a47c21bdf8704a6e35b9eb37ba9b0f76f392` |

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

The deposit's `DEPOSIT.md` carries the sha256 of all 163 files. The inputs this
record reads from other records are cited in `provenance.md` by their own
deposits, and are not re-deposited here.
