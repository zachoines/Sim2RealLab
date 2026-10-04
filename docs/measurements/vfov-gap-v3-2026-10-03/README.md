# The real D555's field of view in v3's closed loop, 2026-10-03

The real unit's depth stream reads fx = fy = 321.522 px at 640×360, an HFOV of 89.73° and a VFOV
of 58.48° ([`real-d555-depth-texture-2026-09-26`](../real-d555-depth-texture-2026-09-26/README.md)).
Training renders the policy camera from 1.93 mm over a 3.68 mm aperture, 335.652 px: 87.27° and
56.41°. This record measures what that difference does to v3's commands. It uses the closed-loop
protocol of [`debug-marker-checks-2026-09-23`](../debug-marker-checks-2026-09-23/README.md) check 4,
which is G7's eval, in two arms that differ only in the policy camera's focal length, and scores
them against a reading written down and committed before the first scored launch.

**By the registered rule the gap matters. It does so on tick 0, and the shift is small.**
- At 128 start poses that both arms share, v3 at the real intrinsics steers 0.146° to the right of
  v3 at the shipped ones on its first tick. The standard error across seeds is 0.045, so the shift
  is 3.2 standard errors from zero, and it has the same sign at all eight seeds.
- That rightward mean is not a fixed bias. B's tick-0 commands sit slightly closer to the subgoal
  bearing than A's, and 106 of arm A's 128 point left of it, so the pull reads as rightward on
  average (§3).
- Over the first second, the registered statistic is 0.250° the other way, against a standard
  error of 0.324: inside the spread. That holds only for the construction registered. Ten tick
  pairs straddle the ±180° seam, and the two other ways of building the same difference put it
  beyond one standard error, in the same direction (§3).
- The rule called the gap material if either difference exceeded its standard error, so it reads
  **matters**. Tick 0 decides it under any construction.

**What tick 0 sees.** Tick 0's depth does not show the start pose. In every first episode of every
launch it is the same floor-only frame, rendered before the reset: the far clamp above the horizon
and a bare floor below. Room geometry first reaches the policy at tick 2–6 (median 3; 272 of the 288
first episodes at tick 2–4). So the shift that fires the rule is v3's response to that frame, drawn
at two focal lengths (§4).

**The camera's effect, against the eval's own noise.**
- **Size.** Per pose, the camera moves the tick-0 command by 0.61° on average. That is about 40
  and 120 times the difference between two launches of the same arm at the same poses (0.016° and
  0.005°).
- **Direction.** The shift is positive over ticks 0–3: +0.15°, +0.12°, +0.14° and +0.10° (A − B;
  3.2, 3.3, 1.6 and 1.8 standard errors).
  - What the frames show: ticks 0–1 are the pre-reset floor frame at every shared pose. By tick 2,
    36 of the 128 show the room, and by tick 3, 72 do. These counts are summed from the seed 42–49
    rows of `tick0_frames.txt`.
  - From tick 4, when 121 do, the sign is mixed. Tick 4 gives −0.26° (1.9 standard errors), and
    the first second −0.25°, as registered.
  - At each pose's first frame with room geometry the mean is +0.16° (1.0 standard error), and the
    per-pose change is 1.46°.

  Everything in this bullet except tick 0 and the first second was computed after the runs.
- **The first second.** At seed 42, two launches of one arm differ by 1.88° and 1.56°. The two
  arms differ there by 2.44°, the same order.
- **Speed.** At the shared poses, the mean speed shift stays inside its standard error: −0.0001 at
  tick 0 and −0.0010 over the first second, in normalized command units, against standard errors of
  0.0004 and 0.0012. Per pose, tick-0 speed still moves by 0.005, against a replicate floor of
  0.00004 and 0.0001.
- **Speed over all episodes.** Tick-0 speed is 0.020 higher at the real intrinsics (2.1 / 3.1
  standard errors, Welch / paired). In most of those episodes, all but each launch's first 16, the
  stale tick-0 frame is the previous episode's view, which differs between the arms.

**Against check 4's marker shift (+3.66°, to the left).**
- The real intrinsics move v3 the opposite way at tick 0 (0.15° right) and over whole episodes
  (0.51° right).
- They move it toward the marker direction over the first second: 0.25° left, or 0.29–0.44° under
  the other constructions.

**Whole episodes, G7's own metrics** (reported beside the reading, not scored by it):
- **Offset median.** The direction-offset median goes from 1.08° to 0.57°, a change of 0.51°. That
  is beyond G7's own standard error (0.28° at eight launches per arm), and 2.2 standard errors
  across these seeds.
- **Absolute offset.** The median absolute offset goes from 15.06° to 14.18°.
- **Completion** goes from 0.895 to 0.906 (A − B −0.011, standard error 0.015).
- **Other outcomes.** Off-path divergence is the only outcome metric past its standard error: 0.88
  % of episodes against 0.50 %, 1.1 / 1.2 standard errors. Collisions, near-arrival and progress
  stay inside theirs.

**What follows.**
- v3 is not changed.
- The next depth-policy training renders its camera at the measured intrinsics. It ships with
  whatever the noise items in
  [`depth-noise-real-structure`](../../tasks/active/trained-policy/depth-noise-real-structure.md)
  decide, rather than as a change of its own; the item is recorded there.
- The rig-class rate is not reported: the protocol has none. The rig class is a heading band
  around one bridge pose's goal bearing and has no definition at these poses.

## 1. The two arms, as rendered

| arm | focal length, cfg / USD | Isaac Lab K, fx = fy | rendered fx, fy | rendered cx, cy | HFOV / VFOV, rendered |
|---|---|---|---|---|---|
| A, as shipped | 1.93 / 1.93 mm | 335.652 px | 335.652, 335.652 px | 320.000, 180.000 | 87.265° / 56.407° |
| B, real fx | 1.8487515 / 1.8487515 mm | 321.522 px | 321.522, 321.522 px | 320.000, 180.000 | 89.728° / 58.483° |

**How arm B is set.** Its focal length is 321.522 × 3.68 / 640 mm, i.e. 1.93 × 321.522 / 335.652. It
goes through the camera cfg (`scene.d555_camera.spawn.focal_length`), set before the env is built,
with the aperture unchanged. Both arms render at 640×360, and the renderer derives fy = fx (square
pixels), so B's VFOV follows from its fx.

**"Rendered" is measured from the pixels, not read from the prim.** A probe gave the policy camera
a second channel, the range to the optical centre. It then fitted fx, fy, cx and cy from the
per-pixel ratio of that range to the image-plane depth, over two environments' frames. The fits
leave an rms residual of 1.2e-7 to 8.1e-7 on r² − 1 (largest single pixel 8.8e-6), so the renderer
is an exact pinhole at the authored focal length in both arms.

**Read back in every launch.**
- **Asserted.** Before its first step, each launch refuses to roll out unless the running camera's
  intrinsic matrix gives environment 0 its arm's fx within 0.01 px, and the 16 environments agree
  to within 0.01 px.
- **Recorded, not asserted, by the launch.** The USD focal length on every prim, and a second
  read-back after the rollout (`policy_camera_after_arm` in the results JSONL). The scorer then
  voids a launch whose post-rollout fx or USD focal differs from the pre-rollout read.
- All 18 launches carry their arm's values in all three.

**What stays the same in both arms.**
- The principal point stays at (320, 180), so the real unit's (318.44, 180.32) is not reproduced.
  Isaac Lab's camera cfg does carry aperture offsets, but its spawner ignores them, warning that
  Omniverse does not support them, and its camera forces the principal point to the image centre.
  The pre-registration's "takes no aperture offset" was imprecise on that point.
- The depth noise model's stereo coefficient uses its own constant (673 px at 1280 wide), not the
  camera, so the injected noise is the same in both arms.
- The env has no perception camera.
- The 2 cm mount-height difference is in neither arm.

**The artifact.** The eval loads `model_999.pt`, the checkpoint the exported artifact
`strafer_depth_subgoal_v3_999.onnx` (`c866bfd5…`) was exported from
([`depth-subgoal-v3-retrain-2026-09-21`](../depth-subgoal-v3-retrain-2026-09-21/README.md)).
Every first episode's dumped observations (ticks 0–7, 16 environments, 16 launches) were replayed
through the ONNX artifact. It reproduces the eval's actions to 1.3e-3 in normalized command units,
or 0.31° in commanded direction. That is a CPU-against-GPU float difference, and the model is the
same in both arms.

## 2. Protocol, and what was written down first

**Protocol.** G7's eval, `eval_cadence_emulation.py`, on
`Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0` with `--profile clean --num_envs 16
--episodes 100 --headless`, observation corruption on. Each launch is one Kit boot, with nothing
else on the GPU.

**Launches.** Eight seeds, 42–49, one launch per arm per seed. Arm A went first on even seeds and
arm B first on odd ones. One more launch per arm at seed 42 then measured the eval's run-to-run
floor at identical poses.

**The tree.** `ed0d5af` with one scratch commit that is not merged. It adds three things to the
eval:
- the focal-length override;
- a check that refuses to roll out unless the running camera's intrinsic matrix carries the
  expected fx;
- read-only records: each episode's first 30 ticks (command, observed bearing, the depth block's
  sum and the start state), and the policy observation of every environment's first episode for
  ticks 0–7.

None of it draws a random number.

**The poses.** These are the start states of each environment's first episode, drawn in the initial
reset from seeded streams before any policy step. All 128 (8 seeds × 16 environments) are identical
in both arms, with a recorded difference of exactly 0. Later episodes start where each arm's resets
lead and share none: 0 of 603 later starts match.

**Definitions.**
- **Steering** is the signed direction offset, the quantity check 4's +3.66° is a shift of: the
  angle of the commanded (vx, vy) from the observed subgoal bearing, positive counter-clockwise
  (left), defined above a commanded speed of 0.05.
- **Speed** is the planar magnitude of the normalized command (1.0 is 1.568 m/s).
- **Tick 0** is an episode's first policy tick. **The first second** is ticks 0–29 at 30 Hz.
- **Per pose, at tick 0**, A − B is wrapped to ±180° and needs both arms' offsets defined.
- **Per pose, over the first second**, the value is the mean of the wrapped per-tick A − B. It is
  taken over the ticks before min(len A, len B, 30) where both arms' offsets are defined.
  - Two of the 128 first episodes end before tick 30: seed 46 env 12 at tick 25 in arm A, and
    seed 47 env 3 at tick 22 in arm A and 24 in arm B.
  - So 3 743 of the 3 840 tick pairs enter: 13 are cut by those endings, and 84 have a command
    below 0.05 in one arm.
  - These counts were taken from the onset records in the deposited eval files.

**The reading.**
- Per pose: A − B in steering at tick 0, and A − B in mean steering over the first second.
- Per seed: the mean over its poses.
- |A − B| is the mean of those over the eight seeds, and the yardstick is its standard error across
  seeds.
- The rule, as registered: "The gap matters if |A − B| in steering exceeds that standard error at
  tick 0 or over the first second; otherwise it is recorded as inside the spread."

The plan noted before the runs that a one-standard-error rule flags a true null "about a third of
the time per statistic, and about half the time with two". It accepted that, because a false
"matters" costs only that the next training uses the measured intrinsics. The z-scores below let a
stricter reading be applied to the same numbers.

**Why this yardstick and not G7's.** G7's own spread is run-to-run at one seed. At tick 0, at a
fixed pose, that spread is nearly zero (§3), which would make any difference "matter". The standard
error across seeds is the larger yardstick, and the one the reading names.

**When it was written.** The reading, its scoring script and the eval patch were committed to the
evidence repository at `ffd841b` (2026-10-03 18:51 CDT). The first scored launch started at 18:52.

## 3. The reading

Per-pose A − B at the 128 shared starts. Each per-seed value is the mean over that seed's 16 poses.

| statistic | A − B | SE across seeds | z | per-seed values | per pose: mean \|A − B\|, sd of signed A − B |
|---|---|---|---|---|---|
| **steering, tick 0** | **+0.146°** | 0.045 | **+3.24** | +0.04 +0.20 +0.04 +0.06 +0.13 +0.43 +0.17 +0.10 | 0.61°, 0.89° |
| **steering, first second** | **−0.250°** | 0.324 | −0.77 | +0.53 −0.82 −0.05 −0.15 +0.12 −2.22 +0.68 −0.10 | 1.93°, 3.68° |
| speed, tick 0 | −0.0001 | 0.0004 | −0.32 | | 0.0052, 0.0067 |
| speed, first second | −0.0010 | 0.0012 | −0.82 | | 0.0069, 0.0107 |

A − B is positive when arm B, the real intrinsics, steers further right than arm A. So B steers
0.15° right at tick 0 and 0.25° left over the first second.

**What the tick-0 shift is.** It is the mean of a small pull toward the subgoal bearing, not a fixed
rightward bias. These figures were computed after the runs, from the tick-0 rows of the deposited
eval files.
- B's tick-0 offsets are 1.9 % less dispersed than A's (sd 18.72° against 19.08°).
- B's mean absolute offset is 0.17° lower: 3.7 standard errors across seeds, positive at all eight.
- Across poses, A − B rises with the pose's mean offset (slope +0.019, Spearman p 4e-4).
- 106 of arm A's 128 tick-0 commands point left of the subgoal (arm B: 105). B sits right of A at
  63 of those 106 poses, but at only 10 of the 22 where A points right.

**The first-second statistic and the ±180° seam.** A command nearly opposite the subgoal sits near
±180°, where offsets of −179° and +176° are 5° apart, not 355°. Ten tick pairs where both arms'
offsets are defined straddle the seam. The registered statistic wraps each tick's difference before
averaging. The two other constructions of the same per-pose difference read:

| first-second steering, per pose | A − B | SE across seeds | z |
|---|---|---|---|
| mean of per-tick wrapped differences (registered) | −0.250° | 0.324 | −0.77 |
| difference of the two arms' mean offsets (computed after the runs) | −0.441° | 0.302 | −1.46 |
| difference of the two arms' circular means (computed after the runs) | −0.286° | 0.173 | −1.65 |

So "inside the spread over the first second" holds for the registered construction only. The sign
is the same in all three, and the verdict does not depend on which is used.

**Same arm, same poses: the run-to-run floor.** One more launch per arm at seed 42, in its own boot,
compared pose by pose with that arm's first seed-42 launch:

| per pose, mean \|difference\| | A vs A | B vs B | A vs B, seed 42 | A vs B, all seeds |
|---|---|---|---|---|
| steering, tick 0 | 0.005° | 0.016° | | 0.610° |
| steering, first second | 1.883° | 1.563° | 2.435° | 1.932° |
| steering at the first frame with room geometry (computed after the runs) | 0.017° | 0.098° | | 1.463° |

The two launches of one arm see a byte-identical tick-0 observation, depth and scalars, at all 16
poses, with the recurrent state at zero. Their tick-0 commands still differ, by up to 5e-4, so the
difference arises in the policy's forward pass on the GPU; it is not isolated further. By the first
second, closed-loop divergence at identical poses is of the same order as the camera's effect.

## 4. What tick 0 shows, and when the room arrives

This env never sets `num_rerenders_on_reset`, which Isaac Lab defaults to 0. So no frame is rendered
between a reset and the observation that follows it, and the depth delay buffer repeats its first
frame. The policy therefore starts every episode on a frame rendered before its reset.

**In first episodes**, that frame is the default scene's: the camera at its spawn height over a bare
floor, with the far clamp above the horizon.

A frame counts as floor-only here when the rows above the horizon read the far clamp, and every row
below is flat across the image and no nearer than the row beneath it. A wall or object anywhere in
view breaks that. The policy-grid frames from all 18 launches' first-episode dumps show the
following.
- **Ticks 0 and 1** are floor-only in all 288 first episodes, whatever the start pose. Within a
  launch, every environment's tick-0 frame is the same floor render up to its own noise draw.
- **The bottom row** of the frame reads 0.6675 m in arm A and 0.6394 m in arm B in every launch.
  That ratio, 0.9579, is 321.522 / 335.652, as a pinhole over a level floor at 0.35 m predicts.
- **The first change** in the depth block comes at tick 1 in 144 first episodes, at 2 in 130 and at
  3 in 10. One pose changes first at 5 and one at 6, in both arms. In 270 of the 288 the changed
  frame is still the pre-reset floor render, with a fresh noise draw: its floor rows move by
  0.43 mm in the median.
- **Room geometry** first appears at tick 2 in 80 first episodes, at 3 in 82, at 4 in 110, at 5 in
  14 and at 6 in 2. Every first episode shows it by tick 6.
- **The timing** is the same in both arms at every pose.

**In later episodes**, tick 0 also shows a frame from before the reset: the previous episode's last
view. The depth block first changes at tick 2–8 (median 3), in all 1 515 later episodes.

**What the tick-0 statistics measure.**
- The registered tick-0 statistic measures v3's response to the floor-only frame at the two focal
  lengths, under the start pose's scalars.
- The registered "first live tick", the first tick whose depth block differs from tick 0, mostly
  lands on the re-drawn floor-only frame:

| per pose | A − B | SE across seeds | z |
|---|---|---|---|
| steering, first live tick (registered) | +0.017° | 0.071 | +0.24 |
| steering, first frame with room geometry (computed after the runs) | +0.162° | 0.156 | +1.04 |

**Training is the same cfg, so it carries the same stale start.**
- The training env is this cfg without the play settings. It also leaves `num_rerenders_on_reset`
  at 0 and carries the same delay buffer, so v3 trained on stale episode starts that show the
  previous episode's last view.
- The floor-only frame itself occurs in training only at a run's initial reset, once per
  environment.
- On the robot there is no scene reset. The inference node assembles an observation only from the
  newest depth frame the camera has delivered (`inference_node.py`), a view of where the robot
  stands.
- What that difference does is not measured here.

## 5. Beside the reading: per-launch values

Arm means over the eight seeds, with sample sd. A − B is given with its standard error across seeds,
unpaired (Welch, as check 4 computed it) / paired. "All episodes" means each launch's ~100 scored
episodes, most of which the two arms do not share.

| metric | A | B | A − B | SE (Welch / paired) | z (Welch / paired) |
|---|---|---|---|---|---|
| steering, tick 0, median over all episodes | 17.70° ± 3.23 | 14.00° ± 3.57 | +3.70° | 1.70 / 2.17 | +2.2 / +1.7 |
| steering, first second, median over all first-second ticks pooled across episodes | 3.17° ± 3.77 | 0.79° ± 2.51 | +2.39° | 1.60 / 1.54 | +1.5 / +1.6 |
| speed, tick 0, mean over all episodes | 0.349 ± 0.019 | 0.369 ± 0.019 | −0.020 | 0.010 / 0.006 | −2.1 / −3.1 |
| speed, first second, mean over all first-second ticks pooled across episodes | 0.507 ± 0.014 | 0.514 ± 0.029 | −0.007 | 0.011 / 0.013 | −0.6 / −0.5 |
| direction-offset median, whole episode (G7) | 1.08° ± 0.32 | 0.57° ± 0.58 | +0.51° | 0.23 / 0.18 | +2.2 / +2.8 |
| direction offset, median of the absolute | 15.06° ± 0.61 | 14.18° ± 0.65 | +0.88° | 0.32 / 0.30 | +2.8 / +2.9 |
| fraction of commands left of the subgoal | 0.525 ± 0.008 | 0.514 ± 0.015 | +0.011 | 0.006 / 0.005 | +1.9 / +2.4 |
| completion | 0.895 ± 0.023 | 0.906 ± 0.037 | −0.011 | 0.015 / 0.019 | −0.7 / −0.6 |
| sustained collision | 0.095 ± 0.024 | 0.089 ± 0.036 | +0.006 | 0.015 / 0.020 | +0.4 / +0.3 |
| off-path divergence | 0.0088 ± 0.0083 | 0.0050 ± 0.0053 | +0.0038 | 0.0035 / 0.0032 | +1.1 / +1.2 |
| near-arrival | 0.504 ± 0.043 | 0.495 ± 0.049 | +0.009 | 0.023 / 0.023 | +0.4 / +0.4 |
| progress, mean | 0.890 ± 0.014 | 0.892 ± 0.023 | −0.002 | 0.010 / 0.011 | −0.2 / −0.1 |

- **Against G7's yardstick.** The whole-episode offset median moves by 0.51°. G7 ran four launches
  per pin at seed 42
  ([`isaac-lab-upgrade-stage3-2026-08-23`](../isaac-lab-upgrade-stage3-2026-08-23/README.md)),
  and its per-pin sds were 0.685° and 0.407° in offset and 0.0435 and 0.0289 in completion.
  - At eight launches per arm, G7's standard error is √(0.685²/8 + 0.407²/8) = 0.28°, so the 0.51°
    is beyond it.
  - Check 4 quotes the same sds at four launches per arm, which gives about 0.40°.
  - Completion's G7 standard error at eight is 0.018, against a −0.011 difference.
- **Tick 0 over all episodes.** It moves more than the shared-pose tick 0: 3.70° against 0.15°. In
  a later episode, tick 0's stale frame is the previous episode's view, which differs between the
  arms in both pose and focal length.
- **A cross-check against check 4.** Arm A reproduces check 4's marker-free arm, run on a different
  tree with a visualizer. Its offset median is 1.08° ± 0.32 here against 1.29° ± 0.40 there, and
  completion 0.895 against 0.905.

## What is not claimed

- That the tick-0 shift changes an outcome. Of the outcome metrics only off-path divergence moves
  past its standard error, by 1.1–1.2 of them.
- Why v3 steers right on the wider floor-only frame.
- Anything about the real D555 itself, its principal point, the 2 cm mount-height difference, or
  the noise model's 673 px constant, which matches neither the sim prim nor the real unit.
- What the stale frames at an episode's start do to the trained policy, or to deployment, where
  the first frame is live.
- A rig-class rate. The protocol has none.

## Deviations from the protocol as first specified

- **Two plumbing launches came first.** Both ran at seed 7, not a scored seed, before the reading
  was committed, to check the flags and records. Their numbers enter nothing. The patch changed
  between them:
  - the start state now records the subgoal in the environment frame (it had been world-frame);
  - the onset depth sum and the observation dump read what the policy consumed (the same thing
    under `clean`);
  - the eval refuses `--onset-obs-ticks` greater than `--onset-ticks`;
  - a refused intrinsics check exits non-zero.

  The scoring script gained the void rule for identical A/B lists and the handling of an undefined
  statistic. Every scored launch runs the final patch.
- **`--headless` replaces check 4's `--viz kit`.** Check 4 needed a visualizer to position its
  markers, and no tree since #224 renders one.
- **The first A_s42 launch was stopped by hand during its boot**, before any rollout, so the series
  could be restarted detached from the shell (started 18:51:15; `logs/aborted_launch/`). The
  "stopped 18:52:07" stamp in that sampling log is when the line was written, not when the process
  was killed.
- **Two launches gave up and were relaunched.** A_s46 and B_s49 each stalled at boot on all three
  watchdog attempts and wrote nothing. Each was relaunched with the same command in the registered
  order (`vfov_series_resume.sh`). Their files are kept as `logs/launch1_stalled.*`.
- **Boot stalls.** 17 of the 40 Kit boots on 2026-10-03 are confirmed boot stalls (about 75 MB
  resident, no CPU, no output for 60 s). One more, the hand-stopped first A_s42 boot, matched the
  signature. It had written no Kit output when stopped, whereas the 11 launches that booted first
  time printed their first Kit line within 1 s of starting (`logs/sampling.log` against each log).
  - The watchdog relaunched 15 of the 17. The other two were the third attempts of A_s46 and
    B_s49, relaunched as above.
  - The rate is entered in
    [`kit-boot-hang-2026-09-11`](../kit-boot-hang-2026-09-11/README.md).
- **The reading understated how long the stale start lasts.** It said the first second carries
  "one to four stale ticks before live ones". The first depth change is usually a fresh noise draw
  on the same pre-reset frame, and room geometry arrives at tick 2–6.
- **The first-second statistic depends on the seam.** The reading's wording, "A − B in the mean
  steering", also fits a difference of the two arms' means. The committed script wraps each tick's
  difference instead. All three constructions are reported (§3).
- **Statistics computed after the runs are labelled as such:**
  - steering at ticks 1–7 and at the first frame with room geometry;
  - the first-second statistic's two other constructions;
  - seed 42's arm-to-arm first-second difference.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `vfov-gap-v3-2026-10-03/record-files/` |
| deposit commit | `b4c70ae93a0ef7ba843eaeec9a9234ad12ff39ef` |

The reading, its scoring script, the eval patch, the rendered-intrinsics probe and the two plumbing
launches are also in the deposit's first commit, `ffd841bc32b846f7629dc44baf9d9a414f8d6900`, made
before the first scored launch.

The deposit holds:
- the launch inputs:
  - `PREREGISTRATION.md`, the reading as committed before the runs;
  - `vfov_eval_flag.patch`, the scratch commit the launches ran;
  - `vfov_run.sh`, `vfov_series.sh` and `vfov_series_resume.sh`, the launch commands;
- the raw results:
  - `eval/`, each launch's results JSONL and first-episode observations;
  - `logs/`, the console logs, package bindings, GPU checks, watchdog attempts and sampling log;
- the analysis:
  - `vfov_summary.py` with `summary.txt` and `summary.json`, the reading and everything beside it;
  - `render_probe/`, the rendered-intrinsics probe;
  - `onnx_parity.py`, the artifact check;
  - `tick0_frames.py`, `stale_start.py`, `posthoc_first_structure.py` and `first_second_seam.py`,
    with their outputs;
  - `frames/`, both arms' policy depth at seed 42's first four poses;
- `plumbing/`, the two launches that came before the reading.

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/vfov-gap-v3-2026-10-03/record-files/. \
      docs/measurements/vfov-gap-v3-2026-10-03/
```

Re-derive the reading with:

```
python docs/measurements/vfov-gap-v3-2026-10-03/vfov_summary.py docs/measurements/vfov-gap-v3-2026-10-03
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/vfov-gap-v3-2026-10-03/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
ddd0d063d14232fdb014640336e9ae384162bca49dd75f0cc3c187df6f6d7a40  PREREGISTRATION.md
858409304e7a4802fa9e0ac8500114ae0fa37369aee0995fb6ee34a9976b2a82  checkpoints_sha256.txt
3bb893210c4d74e823a98f49f0b2c9a6362e755d77ea15075e147d0bbf5dae38  eval/A_s42/cadence_20261003_185555.jsonl
7c6e5d481d189c2159c0128d229f4dfe4ff7646d2229a547b83a4a00b093f666  eval/A_s42/cadence_20261003_185555_first_episode_obs.npz
413681c9c9cd662d2fae2aa480515051117c158c76a434512a3b03b3234ec0da  eval/A_s42_r2/cadence_20261003_195454.jsonl
3e965111908661941f98fb95bf0664b731446a1976cc4f4643165e481ee315f9  eval/A_s42_r2/cadence_20261003_195454_first_episode_obs.npz
97d50c82496025acba0dd0cc09c067ac12ac8fa1bc9d067b203403c575a6fcae  eval/A_s43/cadence_20261003_190438.jsonl
660aa514483d2815579849120495e67aecc8cecd5ec7b589fe2ab647186ed939  eval/A_s43/cadence_20261003_190438_first_episode_obs.npz
254392d95012f03ea3a2fe328cc42e7e9b847a5ab05d427b877dd49396cec19b  eval/A_s44/cadence_20261003_190738.jsonl
b023d730afe79ad05ea4f417f86d5157e0b1879a9357682c7a167f4ebcd37929  eval/A_s44/cadence_20261003_190738_first_episode_obs.npz
41ab9934e71ce0d2d2a2370e6ca2ea9ed653c52fbd2c4fa5cfea6c475936c6d6  eval/A_s45/cadence_20261003_191707.jsonl
077285a0a3f50bc3b94f634f2b458f7eca5edc0ef5b32d54d63299e8a4267d68  eval/A_s45/cadence_20261003_191707_first_episode_obs.npz
090a69f38bdfa06cf472a5ca6726b2da2ca1f48f0c56a620bae3ee74223ff493  eval/A_s46/cadence_20261003_192336.jsonl
079eac48d599f7d2cb2f95720d6c32298bfe58744d3642e789f1379ac17bc29f  eval/A_s46/cadence_20261003_192336_first_episode_obs.npz
c7200bc6d892a9e5a4caaf5eb5248d553cc0bc4abd861336ed0c975da8ecd5e0  eval/A_s47/cadence_20261003_193301.jsonl
35f95a219fa79e7a88db264c46ddab46ab0bbf0153ca97841daaf81077b23408  eval/A_s47/cadence_20261003_193301_first_episode_obs.npz
91c7393976671bed72f5690662c29087f475841c9c3feeeb015bd5fc76d1d6b4  eval/A_s48/cadence_20261003_193543.jsonl
26fe893d802a84c42cd1ba17ad54d3897488c4367abfc56679afac5e16a4910e  eval/A_s48/cadence_20261003_193543_first_episode_obs.npz
c64fc730700b43e653913e08c7d95f9e5437f5f6712d7a8841b19b5941d850bf  eval/A_s49/cadence_20261003_195007.jsonl
bbe438869a476997212a6ec0384abfeb8ef60c39a15fe25389ac7be53444751a  eval/A_s49/cadence_20261003_195007_first_episode_obs.npz
413c227878c97cfdd1301c2e4af3495db7fc1b19b46e2d62ff930666a7eb68ec  eval/B_s42/cadence_20261003_185846.jsonl
2e5d688dcaaf9ea7f37822090779367e1950b9266a1f1a611ea5ba5ab6f3a70f  eval/B_s42/cadence_20261003_185846_first_episode_obs.npz
88580796d9eea51c5f8d959c0c76bbd0f6560ec805b7c0035ed364f2c92f2f7f  eval/B_s42_r2/cadence_20261003_195847.jsonl
30caac08a50c3540775b95f0158629d0102af8d7f87ed0c2afdde99d464a866f  eval/B_s42_r2/cadence_20261003_195847_first_episode_obs.npz
9be2b67dde5b6cdcd4f48db4ef3ff8408a385d9272ee12ecb0ba1373e8f1b270  eval/B_s43/cadence_20261003_190134.jsonl
e18cbaa81bcabf715f18e19468eb7e38d5247852a486851d81ad48f6474c2d35  eval/B_s43/cadence_20261003_190134_first_episode_obs.npz
0ae32d69e33fa4eae1f2e7088c3a8633c6dc2573793fd03ddade9807acda3a57  eval/B_s44/cadence_20261003_191029.jsonl
90b02b63505081a9169dfac9bc411400a195606acb206ae093d3ec70cddbf2ad  eval/B_s44/cadence_20261003_191029_first_episode_obs.npz
c7beddbd57722c8b43b12776e81a1ed10c01d7c6b79e18a240c484d72943652c  eval/B_s45/cadence_20261003_191417.jsonl
cd826b12f5b5abec4fe5b626edd4acb45d5e86c01225bd4fe00ed52b7b90aa7c  eval/B_s45/cadence_20261003_191417_first_episode_obs.npz
f5ff660715ee5864a022a13e580a76d1f84be39ce6feaac948a2ab513e529833  eval/B_s46/cadence_20261003_192626.jsonl
6ceb23a82d76ee7b02f79687b202f746955100bca922812a36f6968952f568b0  eval/B_s46/cadence_20261003_192626_first_episode_obs.npz
ff462ee4e3dc05b61672a4b502e38736285f94eb24e110c9c35142038a0776c5  eval/B_s47/cadence_20261003_192912.jsonl
ad6938f68980be37c09b7f6321337dce4aeebff5c845858ed7d64301d736fdfc  eval/B_s47/cadence_20261003_192912_first_episode_obs.npz
8ee6a5f10dca9844ee335ae44884e3a3221b6d94ac9d1202a9f0ecd5ccd9bd6a  eval/B_s48/cadence_20261003_193940.jsonl
04084b039cde124303f8af69691e110946645407e8cef0aeded56a3f34a3a2e0  eval/B_s48/cadence_20261003_193940_first_episode_obs.npz
b22749f68a9b1570a146d28fec6c09231492e9e4daab2fac9585be0a3f25c331  eval/B_s49/cadence_20261003_194709.jsonl
2d9015583ef2e6846e03ae830b8bd6f1bff66e5c93de9f7caee9a504afbac84f  eval/B_s49/cadence_20261003_194709_first_episode_obs.npz
d8921aee1d0414d0cb9153096f40043c60547c2d45baa7c747d530a6a2e2e790  first_second_seam.py
b589394e4ba818c2effcd0e608f28aa3f66c95ae34dc6a9990217bb04821e0a8  first_second_seam.txt
aa1d8228501fef62cf5643442ea12da41f91ed449e01d2e0ee636058fb64d201  frames.py
06bad198690a65105ec2060435218536ea92271d9b0ee7082eb23a127bdda644  frames/s42_env0.png
30973ede010e7c6a108ed107391b72991d10d91762ad5358ee47a47ccddd91b2  frames/s42_env1.png
f0317786100df33dea1a4b927cd32379ab3620983a933471f0fbe7b8e8677fd8  frames/s42_env2.png
d03527007bb49e514bbd2abf0800cb91c95fe919eeb64dc36ab911c93f23549b  frames/s42_env3.png
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/A_s42.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/A_s42.gpu_before.txt
46d34357f417a62fdce905234e6b7ed0e301837e5d0b48326d6b5f9c9e70dc10  logs/A_s42.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/A_s42_r2.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/A_s42_r2.gpu_before.txt
b40209cfbb6e24f3df6f3a82376ff0fc60e9f65d213c8c429c9377f9f449457e  logs/A_s42_r2.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/A_s43.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/A_s43.gpu_before.txt
23e9c6e7841b5c8f7dbc2de173655e3c9638de41a43d3596a15b43a2aa8bf610  logs/A_s43.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/A_s44.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/A_s44.gpu_before.txt
b43e05c721c42b58223e3cbd0de2a00ba4d9d80eaed2c2ebf74c524e34f14200  logs/A_s44.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/A_s45.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/A_s45.gpu_before.txt
6ca4061f0e107c998918be927c320171cedf936af6b84f47bc6c37fb7f80f4c1  logs/A_s45.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/A_s46.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/A_s46.gpu_before.txt
8c402ae480d8c27522ebf0ff05a44e22693426561f2c8bf362eda2243daa9cad  logs/A_s46.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/A_s47.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/A_s47.gpu_before.txt
a81c4b7aaf6a17582aa85495d3b023b864b20acb799258d65f52ffcb67215b85  logs/A_s47.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/A_s48.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/A_s48.gpu_before.txt
06cdb94571b063dad882ec21eb2f20c59a47857d4f3180788a6c2881514f575b  logs/A_s48.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/A_s49.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/A_s49.gpu_before.txt
9d3411e573e7038267481a14c7e2ad0048275bdc16e11f293ec1952e86d0fcd3  logs/A_s49.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/B_s42.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/B_s42.gpu_before.txt
187f88c2739d3b2bc451522cc4b35b1d594480c5b46bf30af86b59fce3b42ebf  logs/B_s42.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/B_s42_r2.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/B_s42_r2.gpu_before.txt
bc54cd69744dbc41dc367884f14e617ecbc4b02abba265a9ad4175a4571e0ae7  logs/B_s42_r2.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/B_s43.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/B_s43.gpu_before.txt
7bfdcaea4293b8fab949fa688fe1f1cf2e722f473d48cd30eb756f52900820a7  logs/B_s43.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/B_s44.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/B_s44.gpu_before.txt
3fd7ce29ceb65af4e7e20262a812c7dfbb783d3671cf56079c9afd348273ce66  logs/B_s44.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/B_s45.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/B_s45.gpu_before.txt
9d91ae1812e345e1184a15446356c5f167bd1866d05e1fd5b129ae32f43ec9ae  logs/B_s45.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/B_s46.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/B_s46.gpu_before.txt
97cafd2f39a2c71ea2f9aeb902085e22858a25c3b20512957145cbf86fb7da39  logs/B_s46.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/B_s47.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/B_s47.gpu_before.txt
392e596f588cf3e9a46a9bddb2760afe8f1f62fb2182e134b440fd0a377d0b17  logs/B_s47.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/B_s48.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/B_s48.gpu_before.txt
671cea525988176b57e34c502c667b805adb07e8b8f1bce1308bcfa65706140b  logs/B_s48.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/B_s49.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/B_s49.gpu_before.txt
6766fd21fcd36399c4e1fdce6f7ba43e13b3b03176fa9f5f7f1c9208cdd3a8c7  logs/B_s49.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/aborted_launch/A_s42.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/aborted_launch/A_s42.gpu_before.txt
121d386566a93278fa4f766987be682e44ae4f5d215dcfb5474cc537d7a48e99  logs/aborted_launch/A_s42.log
fcdfea92f4de2fd8a4c5a526ab10bd6b1d4b0a99a5faadca030e0e87978b72be  logs/aborted_launch/sampling.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/aborted_launch/watchdog_A_s42.log.attempt1
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/launch1_stalled.A_s46.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/launch1_stalled.A_s46.gpu_before.txt
d4a64d19c141f6e6be0f42c09b57e1e97c4460ac220b40eac73280a0492f7214  logs/launch1_stalled.A_s46.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  logs/launch1_stalled.B_s49.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  logs/launch1_stalled.B_s49.gpu_before.txt
8874ce4a23ae1d0703213f833d6ca608215ba4ecb9d671388e441ac81024e19e  logs/launch1_stalled.B_s49.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/launch1_stalled.watchdog_A_s46.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/launch1_stalled.watchdog_A_s46.log.attempt1
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/launch1_stalled.watchdog_A_s46.log.attempt2
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/launch1_stalled.watchdog_A_s46.log.attempt3
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/launch1_stalled.watchdog_B_s49.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/launch1_stalled.watchdog_B_s49.log.attempt1
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/launch1_stalled.watchdog_B_s49.log.attempt2
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/launch1_stalled.watchdog_B_s49.log.attempt3
3b01b3009a7b4b3829e413226c02098c22d9bec8c2e7da8985986084cf8fcf77  logs/sampling.log
9fd15a23aaed0cc90712aba9e65d55017379287f33aa87a11f9f9f28b0dddd3f  logs/series_console.out
e65dd832cca4403bf476fe257f1b1e8c419c5cbb52869c16f0f6b576ae373046  logs/series_resume2_console.out
0ceb7679b51cb8cf2b5c9530f63a8c8de89769706615981d0b34d598e61045d4  logs/series_resume_console.out
f661536b9c4281a6a3459d0f03e8a0878f52482fe9cf22e9872571a775ca9595  logs/watchdog_A_s42.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/watchdog_A_s42.log.attempt1
f661536b9c4281a6a3459d0f03e8a0878f52482fe9cf22e9872571a775ca9595  logs/watchdog_A_s42.log.attempt2
31549af1d7a90de813fd1edcc23624d225a350c95059f707c6eeb7fe63e89bc1  logs/watchdog_A_s42_r2.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/watchdog_A_s42_r2.log.attempt1
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/watchdog_A_s42_r2.log.attempt2
31549af1d7a90de813fd1edcc23624d225a350c95059f707c6eeb7fe63e89bc1  logs/watchdog_A_s42_r2.log.attempt3
4c7040bd100583734c7b021de8b50c79ff80634ce76028ac4d5c8b3671e2bc4c  logs/watchdog_A_s43.log
4c7040bd100583734c7b021de8b50c79ff80634ce76028ac4d5c8b3671e2bc4c  logs/watchdog_A_s43.log.attempt1
08a20182720220c004f937ed2f825268a00bdfb5cc0a1a4e8a852237b55b666c  logs/watchdog_A_s44.log
08a20182720220c004f937ed2f825268a00bdfb5cc0a1a4e8a852237b55b666c  logs/watchdog_A_s44.log.attempt1
204972756e15725aa3addd38eedcb7bfd10b48aa66c59f5da24f529a75668c6d  logs/watchdog_A_s45.log
204972756e15725aa3addd38eedcb7bfd10b48aa66c59f5da24f529a75668c6d  logs/watchdog_A_s45.log.attempt1
657c2f206992dfecafd4d27f78ac620b51783045b7cf107bcfd16b3de6907132  logs/watchdog_A_s46.log
657c2f206992dfecafd4d27f78ac620b51783045b7cf107bcfd16b3de6907132  logs/watchdog_A_s46.log.attempt1
1fd473b43772931b86073f999cff95f9553c8a22888bee786e419580d515dc0a  logs/watchdog_A_s47.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/watchdog_A_s47.log.attempt1
1fd473b43772931b86073f999cff95f9553c8a22888bee786e419580d515dc0a  logs/watchdog_A_s47.log.attempt2
3d61e411ca02194ad6dc95fe7d2fde3713f2b5c2742d06bc5b7d15e01aee6d12  logs/watchdog_A_s48.log
3d61e411ca02194ad6dc95fe7d2fde3713f2b5c2742d06bc5b7d15e01aee6d12  logs/watchdog_A_s48.log.attempt1
4ec91330b60fa3e6556ef3e22c8b7ad61750f708ae21cb393c47a4dcf80e6bfd  logs/watchdog_A_s49.log
4ec91330b60fa3e6556ef3e22c8b7ad61750f708ae21cb393c47a4dcf80e6bfd  logs/watchdog_A_s49.log.attempt1
a017899e125b6646387e64df9aeb9af7dd450c0e40b0af75368a822b1cd790b7  logs/watchdog_B_s42.log
a017899e125b6646387e64df9aeb9af7dd450c0e40b0af75368a822b1cd790b7  logs/watchdog_B_s42.log.attempt1
53960fd738b4c12b7756b6220fed2fd496cf21c5516b0bdfbf15bf11d29f9f32  logs/watchdog_B_s42_r2.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/watchdog_B_s42_r2.log.attempt1
53960fd738b4c12b7756b6220fed2fd496cf21c5516b0bdfbf15bf11d29f9f32  logs/watchdog_B_s42_r2.log.attempt2
6b88253381b1d98a5598fc1dfc99b47035d65fce264d13f9e1c36ad4492a3d3a  logs/watchdog_B_s43.log
6b88253381b1d98a5598fc1dfc99b47035d65fce264d13f9e1c36ad4492a3d3a  logs/watchdog_B_s43.log.attempt1
0cbc924dca8983f86f2522b6b669cf6bd3da2c95304a18282d73affefdf6d915  logs/watchdog_B_s44.log
0cbc924dca8983f86f2522b6b669cf6bd3da2c95304a18282d73affefdf6d915  logs/watchdog_B_s44.log.attempt1
ab9d716b0eb7243219d3b756eb9b571e4a49e16309e56f5b51e90309a6418fb3  logs/watchdog_B_s45.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/watchdog_B_s45.log.attempt1
ab9d716b0eb7243219d3b756eb9b571e4a49e16309e56f5b51e90309a6418fb3  logs/watchdog_B_s45.log.attempt2
f317dc5e0dea5fc7a0a7e7ca54cb34cd59f59bef97a8f5a8597b0e4eaa602631  logs/watchdog_B_s46.log
f317dc5e0dea5fc7a0a7e7ca54cb34cd59f59bef97a8f5a8597b0e4eaa602631  logs/watchdog_B_s46.log.attempt1
9665506d4dea8d785fa65513e1d9fe4eb1bf354050841de6df2ebc13cdc673f4  logs/watchdog_B_s47.log
9665506d4dea8d785fa65513e1d9fe4eb1bf354050841de6df2ebc13cdc673f4  logs/watchdog_B_s47.log.attempt1
d71bbfa518e1fb7a36865a5fc7911170debb6ca3b482f4f5287bd2699843d66c  logs/watchdog_B_s48.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/watchdog_B_s48.log.attempt1
d71bbfa518e1fb7a36865a5fc7911170debb6ca3b482f4f5287bd2699843d66c  logs/watchdog_B_s48.log.attempt2
ecdc8c0a1197b3164df6246320f29045a150dc92158f0ee14d07d806b4a79bc4  logs/watchdog_B_s49.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  logs/watchdog_B_s49.log.attempt1
ecdc8c0a1197b3164df6246320f29045a150dc92158f0ee14d07d806b4a79bc4  logs/watchdog_B_s49.log.attempt2
ba349d58ee4405ce5193213266caaee14d208bec44980874c4e26b6dac5383f0  onnx_parity.py
c9063b43e9fbc8d222f87c1a613ea4785bd7f94cc49e3eda0802c749360771df  onnx_parity.txt
3b5dec5d0713fef868d09b670c6da5fcad43e851dccf2cd0d4cb92a23e362bad  plumbing/eval/smoke_A_s7/cadence_20261003_185006.jsonl
9278dd8bd432f4487c74883dee409c8c36e30993a7d80e81a592f4a5ca37680b  plumbing/eval/smoke_A_s7/cadence_20261003_185006_first_episode_obs.npz
16119ad5f47676c042c23b1ef0a647eea4b9878707f232fe2c93194d93f2e0c8  plumbing/eval/smoke_B_s7/cadence_20261003_183112.jsonl
0d89fe40b86f136ad0f36333d0a8a0557f7a3c2abfacfed9af2a874d196eaf06  plumbing/eval/smoke_B_s7/cadence_20261003_183112_first_episode_obs.npz
abbdc44a20a610e49cd668b7761f3236bdcb79592b3e4e6dfb1dde589f6c4440  plumbing/logs/sampling.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  plumbing/logs/smoke_A_s7.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  plumbing/logs/smoke_A_s7.gpu_before.txt
f074dc3a06939bcf24388bebd8738a3108f8c447bf1a4a3664027cd431436b2f  plumbing/logs/smoke_A_s7.log
af6e011b793f7e716ba1f4b17269a6f5d641e695f0f595c02fad7ca053d19d54  plumbing/logs/smoke_B_s7.binding.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  plumbing/logs/smoke_B_s7.gpu_before.txt
e1ee1061b32965fc9738ccc4f8fd2feb25c08258c5f1ba74efbca428309435bb  plumbing/logs/smoke_B_s7.log
f10bdd9d67d2cbc896c15176f8688d151b8df08bdc521ea551825ea002691538  plumbing/logs/watchdog_smoke_A_s7.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  plumbing/logs/watchdog_smoke_A_s7.log.attempt1
f10bdd9d67d2cbc896c15176f8688d151b8df08bdc521ea551825ea002691538  plumbing/logs/watchdog_smoke_A_s7.log.attempt2
f0d1fad1d0b4c7a2f1b62542aa8b09a176482bacded3649323103583aac32fd4  plumbing/logs/watchdog_smoke_B_s7.log
f0d1fad1d0b4c7a2f1b62542aa8b09a176482bacded3649323103583aac32fd4  plumbing/logs/watchdog_smoke_B_s7.log.attempt1
c882a47f6ab2e55fd0cc0e0999461d71eff9d65eb1da50a8e62dd2d723c9746c  plumbing/vfov_eval_flag.first.patch
f8068aa522f06d3a7ece00a12f53585d62e5a52a1d9a9bb64fe9ebd999a47ee5  posthoc_first_structure.py
5547b5ba555de31e4127891485b9f2643d857e6c10680965a773bbdbe8f10db0  posthoc_first_structure.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  render_probe/probe_A.gpu_before.txt
5fd326531647f1ed6771251a139ec19aef57474e373e54e3fddeb96486bb8b9b  render_probe/probe_A.json
4106e00da2026bf87b0c37a74ee056c061361654ff8a9946f3c70305e14032e4  render_probe/probe_A.log
95b5d8d13f589f2de078f5f18ab83622af51438d94f2d535d00eaf4c059d9521  render_probe/probe_A_env0.npz
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  render_probe/probe_B.gpu_before.txt
eb17a47167aa146d78d71850d65a85df7000c9abfe087b64ae592699f99390eb  render_probe/probe_B.json
a211cb9dab00e9f5ce3cdc5c81ce3cce8babd09662d8d0973c7dff2cd61bd6d7  render_probe/probe_B.log
f777a2f5ddbec39d94eea371f0deec83409ae23cfac662aceaa0098fecfb2247  render_probe/probe_B_env0.npz
8aa6e6527243de7790c238007ae48048ddcad56205b2143cba00f7d8b57f0d9a  render_probe/vfov_render_probe.py
381fd86bf6633a05e7c22034d76622502e3a0f3bb332748baf81c1261ce208e3  render_probe/vfov_render_probe_cmd.sh
2b2375cc08f1bafbec1b505cd406dbd6a8b111ea0ac9b816ab38750208a87c63  render_probe/watchdog_probe_A.log
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  render_probe/watchdog_probe_A.log.attempt1
18838466c6a102de9fd75d4c0d2a55fe3af05e0fbb9d27fdeb35d73e104bbd4e  render_probe/watchdog_probe_A.log.attempt2
2b2375cc08f1bafbec1b505cd406dbd6a8b111ea0ac9b816ab38750208a87c63  render_probe/watchdog_probe_A.log.attempt3
4364e14347ed81588d5a911620a149501204faa7c21ebeca3e5b5b732e9a3b2e  render_probe/watchdog_probe_B.log
4364e14347ed81588d5a911620a149501204faa7c21ebeca3e5b5b732e9a3b2e  render_probe/watchdog_probe_B.log.attempt1
2b8e0f6d98fcd9992ccde9be0d4e9fb688c0dad6362c48d4ed737b236b5b119a  stale_start.py
cfd62dee2ec9e146c6a04f3c4b2bb9fbdae2c206c5b3623ef23fe5177eaa2fe5  stale_start.txt
f4fd6216e7409d5c084ca72cf624f19c78f55171306b7f4ba8c6eb9abc3422cd  summary.json
1f3e03dff2bfc8faccff5c9f9e8826532cb76937ebdec3d0c447767c6081dabd  summary.txt
6d3241030782366d188172593cc14ee2c3c7e55d284ece1fc435e69287391f6f  tick0_frames.py
dca37f4ae6e1ece6dc429d986528ad1c9d3bf62f6c50e619cdab46fcc7edfcd6  tick0_frames.txt
466401d97edf24e292d5938b714c26d0de0ada9e525393252976c4268e4d9f4a  vfov_eval_flag.patch
97b6f84f4fe5d5145d2e0ac9218a84708d93affb58ef15b40d5a996458697898  vfov_run.sh
3628a8466117064bd0ae77827bcd1f4e0b04a1e1c60059c19c3934a1038f0962  vfov_series.sh
8b2fe71ce811f7f551edaf383087beb9744da21df847c19213aace5b5ece4d8e  vfov_series_resume.sh
f7a87da56a7dfa8e755ae66acce4751284a560b251aa2f5fee3da275f7c6664b  vfov_summary.py
```
