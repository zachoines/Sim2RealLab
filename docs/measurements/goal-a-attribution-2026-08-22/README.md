# Attributing the depth-subgoal v2 off-goal command — 2026-08-22

The six-mission gate of 2026-08-17 reached 0 of 6 goals with a stack that met
every contract it owns ([`goal-a-rig-gate-2026-08-17`](../goal-a-rig-gate-2026-08-17/README.md)).
This record closes the attribution. The failure is the artifact's, it is present
in the first inference of a mission, and the field that controls it is **depth**:
fed the depth statistics its training environment produced, v2 commands toward
its referent; fed the depth the deploy pipeline assembles, it commands 80° away
from it. The divergence between those two is a **convention mismatch in the
near field**, and it is measurable in the source before any policy is involved.

Nothing here is a fix. The defect and the fix directions are carried by
[`depth-nearfield-convention-mismatch`](../../tasks/completed/depth-nearfield-convention-mismatch.md).

Setup, digests, interpreter and the machine-local inputs: [`provenance.md`](provenance.md).

---

## 1. The mechanism, in the source

Training assembles the policy's depth in two stages. The observation term
writes the near field first
([`observations.py:613-678`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py#L613)):
every pixel closer than `nearfield_clip` = 0.4 m becomes `nearfield_fill` =
0.2 m, then the image is clamped to [0, `max_depth` = 6.0]. The realism noise
model then runs on that output
([`noise_models.py:532-566`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L532)),
in this order: stereo Gaussian, holes → `max_range`
([`:555-557`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L555)),
clamp, then

```python
too_close = noisy_data < self.cfg.min_range
noisy_data = torch.where(too_close, torch.full_like(noisy_data, max_range), noisy_data)
```

([`:561-562`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L561)).
`min_range` defaults to **0.2 m** and `max_range` to 6.0 m
([`:644-645`](../../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L644)),
and no config overrides either: both flow from `DepthCameraNoiseCfg`'s defaults
([`sim_real_cfg.py:258-262`](../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L258))
through `get_depth_noise` ([`:789-824`](../../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L789))
unchanged on both realism tiers.

**The `too_close` threshold and the value the term writes are the same number.**
By the time the noise model runs, no pixel is genuinely below 0.2 m — the term
has already replaced every sub-0.4 m pixel with exactly 0.2 m. The rule
therefore fires on the fill constant itself, dithered by the stereo term, whose
standard deviation at that depth is

```
σ_z(0.2 m) = z² · σ_d / (f · B) = 0.2² × 0.08 / (673 × 0.095) = 5.005e-5 m
```

Symmetric dither about a threshold placed exactly on the population's value
sends **half** of it across: 2 000 000 draws give 0.5004 at the realistic tier
(`disparity_noise_px` 0.08) and 0.4999 at the robust tier (0.16). The slam rate
is a property of the threshold's position, not of the noise magnitude — raising
or lowering the realism tier does not move it.

The deploy pipeline writes the other convention. `downsample_depth`
([`obs_pipeline.py:44-85`](../../../source/strafer_ros/strafer_inference/strafer_inference/obs_pipeline.py#L44))
rescues non-finite values to `max_depth`, takes the 8×8 block median, then
applies the same nearfield rule as the training term — `DEPTH_MIN` 0.4 →
`DEPTH_NEARFIELD_FILL` 0.2 ([`constants.py:99-134`](../../../source/strafer_shared/strafer_shared/constants.py#L99)) —
and clamps to [0, 6.0]. It has no `too_close` stage.

So the deploy node agrees with the **clean** training observation term and
disagrees with what training's policy actually saw. The agreement with the
clean term is exact, not approximate: at the probe pose both write the identical
float32 value `0.03333334` at every nearfield pixel, and neither writes the far
clamp at all — the node's tick-0 depth spans `0.0333 … 0.8723` scaled
(0.200 … 5.234 m) with a far-clamp share of **0.0000**, the clean sim image
`0.0333 … 0.8661` (0.200 … 5.197 m). The 6.0 m readings in training are
produced by the noise model and by nothing else.

---

## 2. Files

| path | holds |
|---|---|
| `replay/replay.py` | the offline replay: 1 799 captured node observations through both artifacts on CPU |
| `replay/actions_v2_cpu.jsonl`, `replay/actions_v1_cpu.jsonl` | one `{t_sim, action}` record per tick |
| `replay/analysis.json` | signature, agreement against the on-robot commands, referent timeline, v1 probe |
| `replay/verify_replay.py` | the independent re-derivation run at the time |
| `same-pose-probe/probe.py`, `same-pose-probe/probe_stdout.log` | the Kit probe that reproduced the room and placed the robot at the captured pose, and its full output |
| `same-pose-probe/meta.json` | scene, seed, decimation, disabled terminations, anchor and achieved pose, clean quartet, term layout |
| `same-pose-probe/gym_obs.jsonl` | 30 **clean** observation-term outputs (3 619 dims) |
| `same-pose-probe/env_obsbuf.jsonl` | the same 30 ticks as the ObservationManager delivers them, **after** the realism noise |
| `same-pose-probe/closed_loop_v2.jsonl`, `same-pose-probe/closed_loop_v1.jsonl`, `same-pose-probe/closed_loop_summary.json` | 450 closed-loop steps per artifact from the identical pose |
| `same-pose-probe/field_diff_and_patch.py`, `same-pose-probe/field_diff_and_patch_results.json` | field-by-field node-vs-clean diffs against the pre-registered bands |
| `same-pose-probe/bisect_and_motion.py`, `same-pose-probe/bisect_and_motion_results.json` | the capture's motion onset, the tick-0 scene guard, the bisection ticks and both sweeps |
| `same-pose-probe/patch_replays_consolidated.json` | all 14 patch replays and the two 30-record sweeps in one file |
| `same-pose-probe/analyze.py`, `same-pose-probe/analysis.json` | the probe's own summary pass over its outputs |
| `same-pose-probe/launch_ts.txt` | wall stamps bracketing the probe run |
| `same-pose-probe/verify1_frames_and_diffs.py`, `same-pose-probe/verify1_out.json` | the independent frame-arithmetic and field-diff re-derivation run at the time |
| `same-pose-probe/*.npy` | mean node image, mean clean image, and their difference, 45×80 |
| `verify/` | re-derivations written for this record — §3, §6 and §7 below |

Analysis scripts and outputs from the attribution sessions are kept verbatim,
following the `analyze_tf.py` precedent in the gate record; the one
substitution made to them is described at the end of [`provenance.md`](provenance.md).
The 131 MB observation capture the replay consumes is deposited separately,
under the same record name in `arm3-obs-capture/`; its digest is in
[`provenance.md`](provenance.md).

### Evidence deposit

None of the paths above are in this directory. They are in the companion
evidence repository `https://github.com/zachoines/Sim2RealLab-Artifacts`, a
private repository holding the evidence behind these records, under
`goal-a-attribution-2026-08-22/record-files/`, deposited at commit
`ae3e8c0ef50c80ca7832df3dec8d1c2831a5bb12`. That directory mirrors this one,
so a path above names the same file there. The scripts resolve their inputs as
`docs/measurements/goal-a-attribution-2026-08-22/…` relative to the
working directory, so restoring the deposit into that path is what makes the
commands below run:

```
cp -a <clone>/goal-a-attribution-2026-08-22/record-files/. \
    docs/measurements/goal-a-attribution-2026-08-22/
```

The deposit's own `DEPOSIT.md` comes across with the files and is not part
of the record.

These 31 digests are the payload rows of the record's former `MANIFEST.sha256`,
which is not deposited: its other two rows digested this README and
`provenance.md`, and this edit changes both.

sha256 of every deposited file, paths relative to this directory:

```
010a120826b5e5c5c8f518b6effb2963443cb0556feeb1f0f282481dd5db3c0e  replay/actions_v1_cpu.jsonl
730d218a6956a298134228882570a3c40acf7f31112f0935eede483622f868ff  replay/actions_v2_cpu.jsonl
c4a72d3ccc62988507d3aa5d50dab5f6c6c2ba1deb530a0f15176ce075df2c27  replay/analysis.json
45711db995eb53a6387e2e6251c8f2d09d66f5123b076d5d7bbbf57d2f794b00  replay/replay.py
0deb2ad2e8f6d1564e26bb989da0705d8396c682299be29f52e39b62a8d5d26a  replay/verify_replay.py
176a9183a5ee50dc60526e103b24917fa91d1c8b3ff6e413325cf4d992b1c1ac  same-pose-probe/analysis.json
c2069292cfac4f499eb5972e35648d80e1cd3f2fe47e2e54f43fda459e093ddd  same-pose-probe/analyze.py
d8fc6d060c833da5f4bd5515ffb704f86908388e5c3bfd690e0faa11a47cc4a1  same-pose-probe/bisect_and_motion.py
8ac9134ec6e54fa177ef9586563c67954fca96fab48dd5e5891f44a8247d91b9  same-pose-probe/bisect_and_motion_results.json
8a452d03178009615bb16aaa4aad095f395daa3d3b6cc1da09d6401abee79303  same-pose-probe/closed_loop_summary.json
a65d3ea59507429178d733b6f35d905d27fd12d6ab86d4bf5713c61757bd5437  same-pose-probe/closed_loop_v1.jsonl
6b225e90db491a6563b93577c3492e40a1ced26d4fcc1bfd7a204a6266fb4f3f  same-pose-probe/closed_loop_v2.jsonl
237d6e6bb449bd0fb9f99baed8f1f452759d995c651b36ba4020f8debd4269f7  same-pose-probe/depth_diff_mean_img.npy
101b18ed32e43680b3aa3d1a1496f5b2d2c3298f4dbc7b23bce65cfa6e861e7a  same-pose-probe/env_obsbuf.jsonl
29f7b603777c8c2cb7b6ccb6ac67f1a90d9ac49d2710446ba25a4c014d791290  same-pose-probe/field_diff_and_patch.py
f32d213f34dacdd7d9d19f2ff73d3d2a30ee93ae4db0a45b3511a66f0c48cc67  same-pose-probe/field_diff_and_patch_results.json
448b6335cfec24f36ce16538c5519b48ae52d3df95a47c037c35b16b18c947b4  same-pose-probe/gym_depth_mean.npy
821fe832e37952766088a60d6bb77fd8f4a21c0cdcb83195b9db4d2067cb0cbf  same-pose-probe/gym_obs.jsonl
90e97683c82c2ce1fc894cdbffeff491af63b70f1b5600aacaa30a0652bc28a0  same-pose-probe/launch_ts.txt
7dbc36289051656f355fb00ed6cd661f5574d474864c0ab9208127c167af0768  same-pose-probe/meta.json
0c9cd2c7f3b13ff1a873adc5dc106f97768351127d9ee1fdc99d0739bbb9f839  same-pose-probe/node_depth_mean.npy
eef734090ab1811246d1647af151d6f5a045e0c535abce46476a9e988dc66f0a  same-pose-probe/patch_replays_consolidated.json
8cf92884db6591cb7994fa154a7b08058882f1c748b4827138d9e715f2bafd0d  same-pose-probe/probe.py
1109377167a53e8f3c7a426da086ee2ddb8c38bc4bf46fd8d7b88602e15a7dc0  same-pose-probe/probe_stdout.log
f61da6544d0da1a6cd7ac01495c59d0ad28118bb40c675797481d8904dd6100b  same-pose-probe/verify1_frames_and_diffs.py
06a67055faf196edae0348e18dbbe784b6f560b2ab6b67c293affd064af9e7b7  same-pose-probe/verify1_out.json
9a820fd5eb255ef719bd736f1d85679d0fda4500f2c853cc10ea9bce194e415a  verify/recheck_corruption_stats.py
47e182216916b05fa62c9dc99b0f775cace158754d55b487c6cfab66f4fe9af9  verify/recheck_patch_replays.py
f9509742eb5ff45e2563d06c40015edeaee06d707c845fbbb1cb4f5f4b8ed8fb  verify/recheck_sweeps.py
0bd2e1842c97827a29d38b093ba6cf7f9db5c508f730f752deeb8f1c5a18fc9b  verify/specificity_probe_1.py
f511b7ac8533aa824bab93ebade1401e0b4c452b7b3bb6c27de8164a8b9829b6  verify/specificity_probe_2.py
```

---

## 3. The convention divergence, measured

Both observation streams for the same 30 ticks are in the deposit, so the
divergence re-derives from the record's own files.
`verify/recheck_corruption_stats.py` reads nothing else and needs only numpy:

```
$ python3 docs/measurements/goal-a-attribution-2026-08-22/verify/recheck_corruption_stats.py
records paired                        30
clean nearfield-fill share of frame   0.3786
clean far-clamp share of frame        0.0000
noisy far-clamp share of frame        0.1957
slammed (clean 0.2 m -> noisy 6.0 m)  0.1889 of frame
  as a share of the nearfield class   0.4990
rows 0-21   mean 0.0002   max 0.0050
rows 22-44  mean 0.3694   min 0.2483   max 0.4088
```

**18.89% of the frame reads exactly 6.0 m in training where the deploy pipeline
reads 0.2 m.** The share of the nearfield class that crosses, **0.4990**, is the
predicted 0.5 to within 0.1 pp — the threshold collision of §1, measured rather
than argued. The residual 0.68 pp between the 19.57% far-clamp share and the
18.89% slam is the hole channel (`hole_probability` 0.01 acting on the 62% of
pixels the slam does not touch ≈ 0.62 pp).

The divergence is a floor band, not a scatter: rows 0–21 carry **0.0002**, rows
22–44 carry **0.3694**. The camera looks down onto floor within 0.4 m, so the
bottom half of every frame is the affected region, and it is contiguous.

The same signature appears in the published paired difference:
`env_noise_vs_clean_depth.paired_m` has p95 = max = **5.79999 m**, which is
6.0 − 0.2 exactly, at mean 1.12276 m; correlation between the noisy and clean
depth at record 15 is **0.257**.

This is measured at one pose in one room on the enriched bridge scene, which
runs the **realistic** tier. It is not a distribution over training. v2 trained
on the **robust** tier (§6), where holes are 3× and camera failure is enabled,
so its training corruption was strictly larger than the 18.89% reproduced here;
the slam component is identical because it is tier-independent.

---

## 4. The offline replay — the artifact's, and present at tick 0

`replay/replay.py` pushes all 1 799 captured node
observations through both ONNX artifacts on `CPUExecutionProvider`, one tick at
a time from a zeroed hidden state.

**The CPU replay reproduces what the robot actually commanded**, so the
TensorRT path is not the cause and the capture is faithful. Against the
`last_action` the node recorded on the next tick, over all 1 799 ticks:

| | vx | vy | wz |
|---|---:|---:|---:|
| mean abs difference | 3.574e-4 | 2.152e-4 | 4.693e-4 |
| max abs difference | 2.283e-3 | 1.154e-3 | 1.476e-3 |
| correlation | 0.999885 | 0.999980 | 0.999986 |

Means agree to the third decimal: on-robot `vx −0.04506 / vy −0.02566 /
wz −0.01929` with positive-`vy` duty 0.30590, against the CPU replay's
`−0.04494 / −0.02580 / −0.01975` at duty 0.30461.

**The first inference is already the failure.** Tick 0 of the mission returns
`[−0.00294, −0.41903, −0.63968]` — no forward component, a large lateral one,
a large yaw one — against a goal bearing of −8.13° in the body frame. Over the
first 8 s sim the commanded direction sits **−173.45°** off the goal direction.
Nothing accumulates into this; it is there before the policy has integrated a
single step of recurrent state.

**The referent freeze follows the failure, it does not precede it.** The
subgoal referent occupies 19 distinct map positions across the mission, and all
18 transitions fall in `t_sim` 361.200 → 361.967 — the first 1.13 s of a 60 s
mission, which begins at 360.833. It is then static for the remaining 58.8 s.
The off-goal command is present at 360.833, before the first transition. The
freeze is downstream: the robot barely advances, so the generator's cursor
never rolls.

**v1 drives on the same frames.** Given byte-identical observations, the v1
artifact returns `vx` mean **+0.65875** with positive duty **0.99944**, first
tick `[+0.17561, −0.09146, −0.53942]`, against v2's `vx` mean −0.04494. The
observations are not unusable; this artifact cannot use them.

For the magnitudes, measured from `replay/actions_v2_cpu.jsonl`:
`vx ∈ [−0.1467, +0.2988]`, `vy ∈ [−0.4190, +0.0602]`, `wz ∈ [−0.6397, +0.0611]`
over the whole mission. The last 20 s sim is one-signed in both translation
axes — positive-`vy` duty 0.000 with zero sign changes, at `vy` mean −0.0568 —
which is the sustained strafe the mission shows, at a magnitude the record
should be read from rather than from elsewhere (§9).

---

## 5. The same-pose probe — the room reproduces, and v2 drives in sim

`same-pose-probe/probe.py` rebuilds
`Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0` at environment seed 42
— the capture's scene, reproduced from seed determinism — disables `time_out`,
`robot_flipped`, `sustained_collision` and `goal_reached`, and teleports the
robot to the pose the capture recorded. Requested `(-0.14900, -2.201)` at yaw
2.22; achieved `(-0.14678, -2.17217)` at yaw **2.21370**, i.e. 2.9 cm and 0.36° out.

The analytic quartet closes at that pose. The clean observation terms read
relative `(0.39222, −0.75196)`, distance `0.84844` m, heading `−1.09007` rad;
computed analytically from the placed pose and the referent they read
`(0.39234, −0.75228)`, `0.84844`, `−1.09007`, agreeing to `1.2e-5 / 3.1e-5 /
3.9e-9 / 2.2e-8` scaled. The node's own tick-0 quartet is inside the 0.01
pre-registered band on all four (`1.219e-3`, `2.406e-3`, `2.697e-3`,
`1.151e-4`). The referent bearing is −7.476° here against −8.1° on the rig.

**Closed loop from that pose, 450 steps / 15.0 s sim each:**

| | first command | off referent | net displacement | net toward referent | mean action vx |
|---|---|---:|---:|---:|---:|
| v2 `model_998` | `[+0.49752, +0.01687, −0.54257]` | **+9.42°** | 0.16159 m | 0.15572 m | +0.25031 |
| v1 `model_500` | `[−0.22987, +0.19735, −0.43888]` | +146.77° | 0.93009 m | 0.41380 m | +0.09112 |

v2's first command in sim is 9.4° off its referent — against −82.3° on the
robot's own observation at a referent bearing within 0.7° of it. That reversal, on one
artifact between two observation sources, is the result this probe exists to
produce.

Read the rest of the row honestly. v2 commands toward the referent and averages
`vx` +0.250, but nets 0.156 m of the 0.848 m to it in 15 s sim; v1's first
command is 146.8° off yet it covers 0.930 m. "v2 drives in sim" is a statement
about the commanded direction at the tick the bisection tests, not a statement
that v2 arrives. The two artifacts' closed loops also ran back to back in one
episode (v2 over `t_sim` 1.133–16.133, v1 over 16.200–31.200) after a
re-teleport, not in independent episodes.

---

## 6. The bisection — depth is the controlling field

Every patch replay is a single tick from a zeroed hidden state through
`strafer_depth_subgoal_v2_998.onnx` on CPU. "Off referent" is
`atan2(vy, vx)` minus the −8.1° referent bearing; the pre-registered
toward-referent criterion is `|off| ≤ 45°`.

The harness first reproduces the robot's own command from the captured
observation to `3.9e-5` (`a_node0_unmodified`), so the patched cells differ
from the rig only by the patch.

| patch | off referent | toward |
|---|---:|:--:|
| node tick-0, unmodified | −82.30° | no |
| node tick-0, depth ← clean sim mean image | −81.45° | no |
| node tick-0, imu_accel ← clean | −82.77° | no |
| node tick-0, imu_gyro ← clean | −82.25° | no |
| node tick-0, encoders ← clean | −81.73° | no |
| node tick-0, quartet ← clean | −82.70° | no |
| node tick-0, body velocity ← clean | −82.37° | no |
| clean sim row 15 + node bookkeeping | −79.28° | no |
| **node tick-0, depth ← noise-bearing sim depth** | **+13.51°** | **yes** |
| **clean cell, depth ← noise-bearing sim depth** | **+13.61°** | **yes** |
| noise-bearing sim row 0, unmodified | +12.18° | yes |
| noise-bearing sim row 15, unmodified | +14.30° | yes |
| **noise-bearing row 15, depth ← clean sim depth** | **−80.05°** | **no** |
| **noise-bearing row 15, depth ← node depth** | **−82.58°** | **no** |

Every non-depth field can be replaced with its clean-sim value and the command moves by at
most 0.570°. Depth alone moves it by 95.8°, and it moves it in both
directions: substituting corruption-bearing depth into an otherwise-node
observation flips it toward the referent, and substituting either clean sim
depth or node depth into an otherwise-corruption-bearing observation flips it
back. That is the four-way bidirectional result.

Swept across all 30 records rather than one:

| sweep arm | toward referent |
|---|---|
| noise-bearing rows, unmodified | **29 / 30** |
| whole clean-sim rows + node bookkeeping | **1 / 30** |
| node observation, depth ← clean sim record *i* (single field) | **0 / 30** |

The **whole-clean-row** arm converges hard: records 2–29 all land in
−79.11° … −81.94°, a band 3° wide. The two exceptions are adjacent early records
— record 0 in that arm (−10.36°) and record 1 in the noise-bearing arm (−99.06°)
— both inside the probe's settle transient.

The third row is a re-derivation added for this record: substituting only the
3 600 depth dimensions of each clean sim record into the node's own observation
never produces a toward-referent command, which is the tighter single-field form
of the same result.

**Reproduction.** `verify/recheck_patch_replays.py`
and `verify/recheck_sweeps.py` were written against
the published method rather than by re-executing the analysis, and re-run the
eight cells and both sweeps from the shipped `gym_obs.jsonl` / `env_obsbuf.jsonl`
(sha256 verified equal to the originals) plus the machine-local capture. Worst
absolute deviation across every re-derived command component and angle:
**0.000e+00**. Both sweep counts reproduce.

---

## 7. Specificity probe — run for this record, and it does not isolate the class

§6 establishes that corruption-bearing depth and clean depth put the command in
different classes. It does not establish that the *slammed pixels* are the
operative part of the corruption. That was tested here, and the controls do not
separate. Both probes are single-tick from a zeroed state, off referent in
degrees; the source column names which one each row came from
(`verify/specificity_probe_1.py`,
`verify/specificity_probe_2.py`):

| manipulation of the node's own tick-0 depth | off referent | toward | from |
|---|---:|:--:|:--|
| unmodified | −82.30 | no | 1, 2 |
| every nearfield-fill pixel → far clamp (the training convention) | **+37.70** | yes | 1, 2 |
| the same, checkerboard half of them | +26.55 | yes | 1 |
| **first** 1 288 non-nearfield pixels in raster order → far clamp | **−4.53** | **yes** | 1 |
| random **non**-nearfield pixels, same count → far clamp | +48.40 | no | 2 |
| random pixels anywhere, same count → far clamp | +51.77 | no | 2 |
| nearfield pixels → 0.5 / 1.0 / 2.0 / 4.0 m instead | −101.40 / −115.60 / −119.18 / −18.20 | no / no / no / yes | 2 |
| whole image rescaled to the same mean, no class structure | −25.18 | yes | 2 |

Applying the training convention to the robot's own frame swings the command
120° toward the referent, which is the predicted direction. But equal-area
far-clamps that are *not* the slam move it comparably. The deterministic
raster-order control — the first 1 288 non-nearfield pixels, which is the top of
the image — lands at **−4.53°** and crosses the criterion outright; the randomized
versions of the same control swing ~130° and land at 48–52°, outside the
criterion but near enough that the verdict turns on where the threshold sits. A
uniform rescale with no pixel-class structure crosses as well. The two controls
disagreeing with each other is itself the finding: the response tracks *how much
of the image reads far* and *where*, and neither probe holds those fixed while
moving only class membership. The inverse behaves the same way: undoing only the 671 slam-created
pixels on the corruption-bearing record 15 moves it from +14.30° to −27.30°, in
the predicted direction but well short of the −80.05° that replacing the whole
depth field with clean sim depth produces.

The honest reading is that **v2's command direction is acutely sensitive to
near-field depth content**, that the train-versus-deploy convention divergence
is the instance of that sensitivity which is actually present in deployment, and
that these probes do not license the narrower claim that the slammed pixel class
is specifically what the policy keys on. Isolating it needs a design that holds
mean depth and affected area fixed while moving only the class membership.
Recorded as an open question on the brief, not as a result.

---

## 8. Deviations from the pre-registered protocol

**The scene guard tripped and the anchor moved.** The strict guard compares the
node's records 0–29 against the clean sim images and requires a Pearson
correlation ≥ 0.9 over the 3 600 pixels. Pooled, it reads **0.8951** and fails.
The cause is in the capture, not the reproduction: the robot is stationary only
at record 0. `last_action` is `[0, 0, 0]` at record 0 and non-zero from record 1;
maximum absolute encoder velocity climbs 0.018 → 0.022 → 0.310 → 0.499 → 0.612 →
0.696 across records 0–5; correlation against the capture's own record 0 decays
1.000 → 0.9945 (record 3) → 0.8592 (record 9) → 0.8154 (record 11). Pooling
therefore averages a moving camera against a fixed one.

Re-anchored to the stationary tick, the guard passes: correlation **0.9829**
against the mean clean image and 0.9779 against clean record 0, with mean depth
1.94500 m against 1.94707 m, signed offset −0.0021 m, per-pixel difference p50
**0.0026 m** and p95 0.1161 m. The single localized block reaching 3.673 m is
the residual §9 records. Every bisection cell in §6 uses node tick 0, so it
runs on the passing anchor; the diagnosis is in
`same-pose-probe/bisect_and_motion_results.json`
under `node_motion_onset` and `scene_guard_tick0`.

**Decimation 4, against 1 on the rig.** `sim_dt` 1/120 s with `decimation` 4
gives `step_dt` 1/30 s, so the control cadence is the 30 Hz sim the rig ran; the
physics substep count per control step differs.

**The probe's realism tier is not v2's training tier.** The enriched bridge
scene runs `level="real"` ([`composed_env_cfg.py:844-857`](../../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py#L844));
v2 trained on `robust` (§9 of the brief). The slam is identical across tiers;
holes, frame drops, camera failure and latency are not.

---

## 9. What this record does not establish

- **Field agreement is not what exonerates the node.** Four of the compared
  fields are outside their pre-registered bands: `imu_accel` (band 0.005,
  settled p50 0.0088), `imu_gyro` (band 0.005, p95 0.0330), `encoders` (band
  0.01, p50 0.1856) and `body_velocity` (band 0.005, p50 0.0627). They are out
  of band because the capture's robot was in motion — its encoder means are
  `[−0.391, −0.286, −0.101, +0.011]` against a stationary sim's ≈ 0. What
  exonerates them is §6: replacing each with its clean value moves the command
  by at most 0.570°. Causal inertness, not numerical agreement.
- **The exoneration is one tick.** Every patch replay is tick 0 of one mission
  from a zeroed hidden state. It establishes first-command causality on a
  recurrent policy, not closed-loop causality, and it is not a whole-mission
  statement about the node.
- **A localized depth residual remains** between the node and clean sim at the
  anchor: p95 0.1161 m with a single block reaching 3.673 m, and the pooled
  30-record paired difference reads p50 0.4269 m / p95 2.5138 m. It is shown
  causally inert by the same patch swap (−82.30° → −81.45° with the whole clean
  mean image substituted), not shown to be small.
- **"v1 tolerates both conventions" is supported asymmetrically.** On the
  robot's own frames v1 drives (`vx` mean +0.659, duty 0.999). In sim from the
  identical pose it advances further than v2 (0.930 m against 0.162 m) but its
  first command is 146.8° off the referent. v1 was not tested against a
  corruption-bearing single-tick sweep the way v2 was.
- **The command magnitudes quoted elsewhere are not reproduced here.** Figures
  of `vy` +0.145 and +0.224 at duty 1.00 have circulated for this failure; the
  capture's raw policy output never exceeds `vy` +0.0602 in the positive
  direction — its whole range is −0.4190 … +0.0602 — and the node publishes the
  action as m/s capped at `NAV_LINEAR_VEL` 0.7841, so no scaling in
  `strafer_shared.constants` maps one onto the other. The one-signed, zero-sign-change shape those figures describe does appear —
  in the last 20 s sim, at `vy` mean −0.0568. Which log, window and unit
  produced the quoted magnitudes is unresolved and is carried as an open item
  on the brief.
- **The one gate success is not explained.** `PILOT_uncontrolled_heading`
  reached 0.299 m of a 3.029 m goal in 53.6 s sim. It is the only run in the
  gate record whose start heading differs: every scored mission and every
  2026-08-19 repeat started at heading error −2.6° to −2.8°, and the pilot at
  **−115.5°**. Depth is body-fixed, so the pilot is also the only run whose
  camera faced a materially different part of the room, which is consistent with
  a content-dependent nearfield share — but no depth was captured for it, and
  goal bearing alone does not separate the set (M2 failed at −107.2°, M5 and M6
  at +89.5° and +90.6°). Consistent, unproven.
- **One artifact, one scene, one pose.** Nothing here characterizes the
  affected pixel share over a training distribution, and nothing here tests a
  fix.

---

## 10. Training provenance — what differs between v1 and v2

Neither run recorded a configuration. `logs/rsl_rl/strafer_navigation/<run>/git/`
is empty for every run involved, the checkpoints carry `iter` and state dicts
and nothing else, and the TensorBoard files carry no text or hyperparameter
records. The per-run stdout logs named in [`provenance.md`](provenance.md) are
the only surviving statement of what was run.

**The mechanism is older than both artifacts, and it is two-sided.** The
`too_close` slam and the `min_range = 0.2` default entered in `52e1bd5` on
**2026-01-12**; the `nearfield_fill = 0.2` the threshold lands on entered in
`c50a76a` on **2026-03-23**, which is when the collision was created. Neither
line has been modified since (`git log -S` on both). Across the two export
commits —
`eeacccc` (2026-07-08) and `69014c6` (2026-07-26, the merge of #168) —
`noise_models.py`, `sim_real_cfg.py`, `observations.py` and `d555_cfg.py` are
**byte-identical**. Neither run's depth pipeline differs from the other's.

**Two candidate changes, neither of which separates the two runs.** #153 merged
2026-07-18, which is *inside* the window, but it re-derives a variance test's
confidence interval from the wall-pixel count and nothing else: it states in its
own body that the noise models are not touched, and the empty tree diff over
those files confirms it. #143, the 80×60 → 80×45 policy camera that took the
rendered vertical FOV from ~71° to 56.4°, lands *before* the window — its last
commit is at `2026-07-08T05:40:27Z` and v1's run begins at `05:59:23Z`, nineteen
minutes later, and v1's own stdout log is named `depth_subgoal_vfov8045`. Both
artifacts trained on the 80×45 camera.

**What did change is the environment.** From the stdout logs:

| | v1 | v2 |
|---|---|---|
| run | `run_20260708_005923`, iterations 0 → 578 | `run_20260726_221955` 0 → 499, then `run_20260727_171735` 500 → 998 resuming leg 1's `model_499.pt` |
| task | `Isaac-Strafer-Nav-RLDepth-Subgoal-Real-v0` | `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0` |
| deployed checkpoint | `model_500.pt` | `model_998.pt` |
| seed | 42 | 42 |

v2 is a fresh run, not a continuation of v1 — leg 1 has a `model_0.pt` and no
resume line. **Its task ID is absent from v1's tree**, which is the load-bearing
fact; the enriched variants enter at `3054af1` (#156, 2026-07-20), between the
two runs. `StraferNavCfg_RLDepthSubgoal_Real`, the class v1 used, is
byte-identical between the two trees, and enrichment is gated on `enrich_depth`
in `_ComposedStraferNavEnvCfg`, so the change reached v2 only through the task
ID.

The iteration counts come from TensorBoard; v1's stdout log truncates at its
iteration-574 block.

The two config fields that differ are `enrich_depth` False → True and
`level` `real` → `robust`. Both move the affected share, in the same direction:

- **`enrich_depth`** raises near-camera surface across the board — enclosing
  walls 1.0 → 2.7 m, a per-episode ceiling at probability 0.7 and 2.2–2.9 m,
  shelf/cabinet/tall-cylinder heights 0.8/0.6/0.7 → 2.0/2.1/1.8 m, two mid-room
  columns at probability 0.5, difficulty un-pinned from 7 to U[4, 7], and robot
  spawn inflation 2 → 1, which starts the robot closer to obstacles
  (`strafer_env_cfg.py`, the `_ENRICH_*` block).
- **`robust`** triples `hole_probability` (0.01 → 0.03), doubles
  `disparity_noise_px` (0.08 → 0.16), enables camera failure at 0.001, and
  takes depth latency 1 → 2 steps. It leaves `min_range` and `max_range` alone,
  so it does not move the slam rate — §1.

**The training curves do not add to this, and are recorded so they are not read
as if they did.** v2 leg 1 starts from `model_0.pt`, so its first logged
iteration is a fresh policy rather than a resume: `Train/mean_episode_length`
32.111 and `Train/mean_reward` −0.174 at step 0, against v1's 20.857 / −1.029 at
its own step 0. By the end of each run leg 1 reads 200.54 / 3.315 at iteration
499 and v1 reads 149.90 / 4.145 at 578. Leg 2 resumes leg 1 and its first logged
iteration, step 499, reads 33.091 / −0.330 — the discontinuity there is the
resume, not a change of scene. Two runs of different lengths under different
task IDs cannot be compared this way to say which environment was harder; the
stdout logs settle that directly, and nothing here is offered as corroboration
of it.

**What is missing, exactly.** No measurement of the affected pixel share under
v1's environment exists, and none can be made from the artifacts on either
host — it needs a Kit run on the pre-enrichment tree. So the direction of the
change is argued from the enrichment parameters above, and its magnitude is not
measured. `training-run-provenance-manifest` is the brief that would have made
this recoverable rather than reconstructed.
