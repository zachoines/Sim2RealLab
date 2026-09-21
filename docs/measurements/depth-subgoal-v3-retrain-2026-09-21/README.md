# The depth-subgoal v3 retrain: the run, the exported artifact, and how it reads the bridge capture — 2026-09-21

v3 is the depth-subgoal policy retrained on the depth path the deploy node produces. The
policy camera renders 640×360 and the observation term reduces it with the deploy node's
8×8 block median; the near-field class saturates instead of inverting to the far clamp
(#219); the observation delay buffer warms up on the first frame (#220); and the robust
tier draws its stereo σ_d log-uniformly per environment over (0.002, 0.16) (#221, #222).
This record covers the training run, the exported artifact and its play smoke, and the
three descriptive tables the brief
[`depth-subgoal-v3-retrain`](../../tasks/active/trained-policy/depth-subgoal-v3-retrain.md)
asks for: v2 and v3 side by side on the bridge capture's tick-0 frame set, the per-band
texture of the training depth against the bridge capture, and the two runs' training
curves.

None of it is the acceptance gate. The brief is accepted against the sim-bridge rig-gate
protocol of [`goal-a-rig-gate-2026-08-17`](../goal-a-rig-gate-2026-08-17/README.md), which
is not run here, and no row below carries a threshold.

- The run completed 1000 iterations in one leg at 95.25 s per iteration, 6.2 % under the
  101.584 s the deploy-resolution record measured at 96 environments, with no NaN and one
  boot attempt.
- The exported artifact declares `obs_dim` 3619, `action_dim` 3, `is_recurrent`, and the
  enriched robust play env. Driven 60 steps on that env, it returns 60 finite, distinct
  actions.
- On the bridge capture's tick-0 frame set v3 gives no rig-class frame in any row: 0 of 30
  at every uniform-field distance and every σ_d, where v2 gives 30 of 30 at 5, 5.5 and
  6 m. v3's heading moves 7.2° across the robust tier's σ_d ladder against v2's 48.7°; v3
  answers the depth field through yaw rate and speed instead.
- The training depth's texture is louder than the bridge capture in the two near bands at
  every draw of the band, and brackets the bridge capture in the three far bands.

## 1. The run

| item | value |
|---|---|
| tree | `main` at `615fc14904967ebe377e10ea3a51b863c5a5a245` (the #222 merge), clean |
| task | `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0` |
| environments / seed / iterations | 96 / 42 / 1000 (0–999), one leg, no resume |
| depth noise band | `disparity_noise_px_range = (0.002, 0.16)`, log-uniform per environment — the shipped robust tier, unchanged |
| composition contract hash | `faf867563f76d01078faf9d6fe0dbd93523bc36734e75edaaf66cf1afaad001d` |
| runner | `num_steps_per_env` 48, `save_interval` 100, learning rate 3.0e-4 `"fixed"`, `empirical_normalization` false, `check_for_nan` true |
| boot | one attempt, exit 0, wall 95 324 s (26.48 h) |
| NaN, Inf, CUDA error, OOM | none in the log |

The contract hash is recomputed without Kit by `probes/launch_provenance.py`, which
mirrors the serializer in `test_sim/env/test_composition_contract.py`. It is taken at the
cfg's default `num_envs` (64), where the golden was frozen, and equals the frozen
`RLDepthSubgoalEnriched_Robust` golden; the snapshot includes `scene_num_envs`, so the
run's command-line 96 is recorded separately rather than hashed. The composed cfg renders
the policy camera at 360×640 and gives the depth term an 80×45 noise grid.

Per-iteration time, seconds:

| | collection | learning | iteration |
|---|---:|---:|---:|
| mean, iterations 1–999 | 35.66 | 59.59 | **95.25** |
| median | 35.62 | 59.15 | 94.68 |
| p95 | 36.78 | 65.94 | 102.00 |
| max | 37.92 | 89.09 | 124.99 |
| iteration 0 | 37.54 | 81.99 | 119.53 |

Against the 101.584 s the deploy-resolution record measured on the shipped path at 96
environments, the mean is 6.2 % lower, and the run took 26.48 h against the brief's
≈28 h. Iteration time rose from 92.95 s over iterations 1–499 to 97.56 s over 500–999.
All of the rise is learning time; collection held at 35.65 and 35.68 s.

Memory, over the run's 45 649 samples at 2 s: `sys_used` peaked at 98 773 MiB of
124 543; GPU compute use at 74 851 MiB; trainer RSS at 19 378 MiB. GPU memory is unified
on GB10, so `sys_used` includes the GPU figure.

## 2. The training curves

`tables/curves/training_curves.png` overlays the two runs on one update axis: the
path_complete termination share, mean reward, mean episode length and the policy's
action std, each as a 20-update trailing mean. `tables/curves/curves_binned.csv` is the
same data in 50-update bins.

Final 100 updates, v3 over 900–999 and v2 over 899–998:

| metric | v3 | v2 |
|---|---:|---:|
| path_complete termination share | 0.870 | 0.883 |
| mean reward | 4.884 | 5.093 |
| mean episode length, steps | 163.9 | 153.1 |
| off_path_divergence share | 0.0227 | 0.0153 |
| sustained_collision share | 0.1067 | 0.1011 |
| time_out share | 0.0010 | 0.0005 |
| cross-track error | 0.0874 | 0.0853 |
| policy action std | 0.307 | 0.301 |

| | v3 | v2 |
|---|---:|---:|
| first update where the 20-update mean path_complete share exceeds 0.5 | 520 | 398 |
| … exceeds 0.8 | 717 | 542 |
| peak policy action std (update) | 0.653 (432) | 0.642 (354) |

Both runs start at the initial 0.30 action std, rise to a peak, and settle back to about
0.30 as the path_complete share climbs. v3 does each later: 122 updates later to a 0.5
share and 175 to 0.8.

Reading the curves:

- `Episode_Termination/path_complete` is the share of environments whose most recently
  ended episode ended on that term, averaged over the rollout. It is not a per-episode
  success rate. Reward and episode length are trailing means over the last 100 completed
  episodes.
- v2's two legs are joined on the absolute update axis. Leg 2 re-logs update 499 as the
  first iteration after reloading `model_499.pt`, on freshly reset environments: reward
  −0.330, episode length 33.1, path_complete 0.000, against leg 1's 3.315, 200.5 and 0.759
  at the same update. The join keeps leg 1's 499; the crossing figures do not move either
  way. Both final checkpoints carry 1000 PPO updates.
- The two runs trained on different distributions, and not only in the depth path. At
  `615fc14` the robust tier also carries command holds, depth-stream holds and
  referent-frame drift, none of which exist at v2's export head `69014c6`, and v2 trained
  before the #218 Isaac Sim / Isaac Lab upgrade. Each curve describes its own run's
  training distribution; the gap between them is not a measure of policy quality, and the
  per-iteration times are not comparable either.

## 3. The exported artifact and its play smoke

```
tools/kit_boot_watchdog.sh --label v3-export --log export/watchdog_export.log -- \
  $ISAACLAB -p source/strafer_lab/scripts/export_policy.py \
    --checkpoint logs/rsl_rl/strafer_navigation/run_20260919_234233/model_999.pt \
    --output models/strafer_depth_subgoal_v3_999 \
    --variant DEPTH_SUBGOAL \
    --env Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0 \
    --formats pt,onnx --headless
```

One boot attempt, 17 s. The sidecar, `export/strafer_depth_subgoal_v3_999.json`:

| field | value |
|---|---|
| `obs_dim` | 3619 |
| `action_dim` | 3 |
| `is_recurrent` | true |
| `env_id` | `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0` |
| `git_commit` | `615fc14904967ebe377e10ea3a51b863c5a5a245` |
| `source_checkpoint` | `logs/rsl_rl/strafer_navigation/run_20260919_234233/model_999.pt` |
| `onnx_opset` | 18 |
| `trained_period_s` | 0.0333 |
| `policy_variant` | `DEPTH_SUBGOAL` |

| file | sha256 |
|---|---|
| `strafer_depth_subgoal_v3_999.onnx` | `c866bfd54ec1a8352159e33d7875d41e3f07a442ff8301ba3700867932e2eb91` |
| `strafer_depth_subgoal_v3_999.pt` | `bcfccf65009b3945ed40a171fa345188879a4b277e3930a76770f2329e03b896` |
| `strafer_depth_subgoal_v3_999.json` | `d22e3504e6aa2f0cefbcf2af41561c8acbab4ecf87a8c2e3a4b494294bf33ab5` |
| `model_999.pt`, the source checkpoint | `725fc6bfcf32ee756f70a459e45f2a62f17e14289a40e458a349f1a086c21484` |

The env is named because `export_policy.py` defaults the `DEPTH_SUBGOAL` variant to
`Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0` and `PolicyVariant` has no enriched
member, so an export that omits `--env` records the realistic, non-enriched env. v2's
sidecar, `strafer_depth_subgoal_v2_998.json`, records exactly that. The field is metadata.
The export builds the named env only for its shapes and for `trained_period_s`
(dt × decimation, 1/30 s for both envs); the weights and the observation normalizer come
from the checkpoint, because `empirical_normalization` is off at the runner level and the
model's `obs_normalizer` is module state that `runner.load` restores. Both ONNX files take
`obs [1, 3619]` and `h_in [1, 1, 128]` and return `actions [1, 3]` and `h_out [1, 1, 128]`,
so §4 compares the two checkpoints whatever env each sidecar names.

**Play smoke.** `exported_policy_rollout.py` from the deploy-resolution-depth-2026-09-19
deposit, run unmodified, drives the TorchScript artifact on
`Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0`, batch 1, for 60 steps. One
boot attempt, 19 s; the result is `smoke/exported_policy_rollout.json`.

| | |
|---|---|
| observation width from the env | 3619 (3600 depth + 19 scalar) |
| steps | 60 of 60 |
| `action_dim` | 3 |
| all actions finite | yes |
| distinct actions | 60 |
| first action | (0.1768, 0.2878, 0.8045) |
| last action | (0.8131, −0.0245, −0.0018) |

**Video.** `video/videos/rl-video-step-0.mp4` is the same rollout recorded overhead by
`probes/exported_policy_rollout_video.py`, in a separate boot (one attempt, 25 s) so the
recording could not touch the smoke's result: enabling video sets
`/isaaclab/video/enabled`, which changes the render loop. 1280×720, 30 fps, 60 frames.
Frame 0 is black, because the render product is created on the first capture; frames 1–59
show the room through its one-way ceiling, with the robot moving just right of centre.
The subgoal and path markers stay frozen at the world origin, the centre of the frame,
because the probe registers no visualizer (§6). The room's floor renders dark and the
robot is a dark shape on it; `video/v3_smoke_overhead_graded.mp4` is a copy cropped to
the room with the midtones lifted, for viewing.

**Video with the markers.** `markers/play_videos/play_20260921_105717/rl-video-step-0.mp4`
is the stock `play_strafer_navigation.py` on the same artifact and env, 300 steps
(10 s), with headless set by `HEADLESS=1` rather than `--headless` so that its Kit
visualizer registers (§6). One boot attempt, 53 s. The subgoal env draws no goal sphere:
the cyan sphere is the rolling subgoal, 0.7–1.3 m ahead along the path, and the white
dots are the planned path at 0.2 m spacing. The first episode runs the path to its end in
about 3 s; the later cuts are new episodes in regenerated rooms. The run is not
byte-identical to the smoke — a registered visualizer adds a physics forward and its own
app update to every render — so the clip is for looking at, not for numbers.

Whether the positioned markers reach the D555 depth image is not established. Isaac Lab
flags every marker prototype `primvars:invisibleToSecondaryRays` so that depth images
skip them, and while the robot drives the markers are in the D555's view: body +X, the
camera axis, stays within 19° of the subgoal and the velocity within 4° of body +X
(`markers/leak_run3/depth_marker_leak.json`). The in-place test in
`markers/depth_marker_leak.py` cannot answer it — hiding the whole scene leaves the depth
buffer unchanged too, because the camera does not refresh outside the env step — so it is
recorded as inconclusive.

## 4. v2 and v3 on the bridge capture's tick-0 frame set

The harness is `candidate_sweep.py` from the noise-texture-parity-2026-09-17 deposit, run
unmodified, and `tables/samepose/table_a_samepose.py`, which imports that harness and adds
three rows: the node's own observation, the shipped robust tier with σ_d pinned, and the
band drawn live. Each row puts a 3600-dim depth block under the bridge capture's own
19-dim scalar prefix from tick 0 (IMU, encoders, the subgoal quartet, body velocity, last
action zero), because the dumped gym prefix holds NaN in dims 10–13 and 16–18. Each ONNX
runs one tick per frame from a zero hidden state, on CPU.

Off-goal is the direction of the commanded (vx, vy) relative to the goal bearing of
−8.1°, wrapped into [−180°, 180°). The rig class is an off-goal in [−83°, −79°], counted
over a row's frames. It is v2's response to a far, featureless field and is retired as a
criterion (noise-texture-parity-2026-09-17); it appears here descriptively. The noise
rows use the production `DepthNoiseModel`, injected at 80×45 after the reduction — the
injection point v3 trained with — at seed 7. Brackets give the range of the row's mean
over seeds 7 and 1–6, and the rig count's range over the same seeds.

The deposited `candidate_sweep.py` and `convention_ab.py` re-run unmodified with v2@998 on
`615fc14` reproduce their deposited JSON byte for byte (`97699a22…` and `3ef84e86…`). The
v3 rows come from the same harness with only the ONNX changed.

**Table (a1).** Mean off-goal in degrees, and rig-class frames over the row's n.
Source: `tables/samepose/table_a_samepose.json`.

| row | n | v2 off-goal | v2 rig | v3 off-goal | v3 rig |
|---|---:|---:|---:|---:|---:|
| node depth — the bridge capture's tick-0 observation | 1 | −82.30 | 1/1 | −30.60 | 0/1 |
| clean sim — 80×45 direct render at the capture pose, 2026-08-22 | 30 | −78.83 | 29/30 | −29.49 | 0/30 |
| uniform field 0.2 m | 30 | +13.03 | 0/30 | +23.92 | 0/30 |
| uniform field 1 m | 30 | −143.02 | 0/30 | −2.70 | 0/30 |
| uniform field 3 m | 30 | −71.85 | 0/30 | −10.84 | 0/30 |
| uniform field 5 m | 30 | −79.13 | 30/30 | −11.91 | 0/30 |
| uniform field 5.5 m | 30 | −79.44 | 30/30 | −12.02 | 0/30 |
| uniform field 6 m | 30 | −80.45 | 30/30 | −11.80 | 0/30 |
| stereo only, σ_d 0.002 | 30 | −77.50 (−77.64..−77.11) | 1–4/30 | −29.17 (−29.21..−29.14) | 0/30 |
| stereo only, σ_d 0.008 | 30 | −60.30 (−61.16..−60.01) | 0/30 | −27.43 (−27.56..−27.43) | 0/30 |
| stereo only, σ_d 0.08 | 30 | −26.64 (−26.87..−24.75) | 0/30 | −22.10 (−22.20..−21.96) | 0/30 |
| stereo only, σ_d 0.16 | 30 | −22.76 (−22.76..−20.38) | 0/30 | −21.56 (−21.79..−21.48) | 0/30 |
| robust tier, σ_d pinned 0.002 | 30 | −72.06 (−72.06..−69.59) | 0–6/30 | −28.68 (−28.68..−28.53) | 0/30 |
| robust tier, σ_d pinned 0.008 | 30 | −57.43 (−57.43..−55.91) | 0/30 | −27.19 (−27.22..−27.12) | 0/30 |
| robust tier, σ_d pinned 0.08 | 30 | −27.08 (−27.32..−24.98) | 0/30 | −22.09 (−22.17..−21.95) | 0/30 |
| robust tier, σ_d pinned 0.16 | 30 | −23.40 (−23.40..−21.27) | 0/30 | −21.52 (−21.71..−21.46) | 0/30 |
| robust tier, band (0.002, 0.16) drawn live | 30 | −44.33 (−46.69..−43.95) | 0–2/30 | −25.07 (−25.72..−24.93) | 0/30 |

v3 gives no rig-class frame in any row. Its off-goal moves 7.2° across the robust tier's
σ_d ladder and 7.6° across the stereo-only ladder, against v2's 48.7° and 54.7°, and the
node row and the clean-sim row are 1.1° apart for v3 against 3.5° for v2.

Off-goal ignores yaw rate and speed, and that is where v3's response to depth shows.
**Table (a2)**, the mean commanded action at seed 7, (vx, vy, wz) and |v_xy|, same source:

| row | v2 action | v2 \|v_xy\| | v3 action | v3 \|v_xy\| |
|---|---|---:|---|---:|
| node depth | (−0.003, −0.419, −0.640) | 0.419 | (0.223, −0.179, −0.681) | 0.286 |
| clean sim | (0.024, −0.464, −0.760) | 0.465 | (0.227, −0.175, −0.684) | 0.287 |
| uniform field 0.2 m | (0.098, 0.008, −0.629) | 0.098 | (0.475, 0.135, 0.366) | 0.494 |
| uniform field 1 m | (−0.218, −0.120, −0.655) | 0.249 | (0.681, −0.130, 0.214) | 0.693 |
| uniform field 3 m | (0.041, −0.230, −0.089) | 0.233 | (0.531, −0.182, 0.020) | 0.562 |
| uniform field 6 m | (0.006, −0.242, −0.060) | 0.242 | (0.458, −0.166, −0.041) | 0.487 |
| robust tier, σ_d pinned 0.002 | (0.071, −0.418, −0.793) | 0.427 | (0.224, −0.168, −0.686) | 0.280 |
| robust tier, σ_d pinned 0.16 | (0.234, −0.143, −0.785) | 0.275 | (0.405, −0.230, −0.673) | 0.466 |
| robust tier, band drawn live | (0.194, −0.271, −0.800) | 0.350 | (0.277, −0.178, −0.700) | 0.329 |

On the room, node and clean sim alike, v3 yaws at wz ≈ −0.68 and moves at |v_xy| ≈ 0.29.
On a uniform field at 3–6 m its yaw falls to between −0.04 and +0.02 and its speed rises
to 0.49–0.56. v3's command therefore does depend on the depth field; what does not move
much is its heading. v2's answer to the same far fields is the nearly pure lateral command
(vx ≈ 0.01, vy ≈ −0.24) that the rig class scores. Heavier σ_d speeds v3 up, from
|v_xy| 0.28 at 0.002 to 0.47 at 0.16, with wz between −0.67 and −0.71 throughout.

**Table (a3).** The same noise arms applied to the node's own tick-0 depth, 30 independent
draws each. The node's depth is the bridge's 640×360 render through the deploy block
median, so it is the only clean field at the capture pose on v3's training path. Source:
`tables/samepose/table_a_node_base.json`.

| row | v2 off-goal | v2 rig | v3 off-goal | v3 rig |
|---|---:|---:|---:|---:|
| node depth, no noise | −82.30 | 30/30 | −30.60 | 0/30 |
| + stereo only, σ_d 0.002 | −83.61 (−83.74..−83.45) | 3–8/30 | −30.38 (−30.39..−30.36) | 0/30 |
| + stereo only, σ_d 0.008 | −67.88 (−68.30..−67.23) | 0/30 | −29.33 (−29.40..−29.32) | 0/30 |
| + stereo only, σ_d 0.08 | −23.31 (−23.31..−21.91) | 0/30 | −23.85 (−24.07..−23.84) | 0/30 |
| + stereo only, σ_d 0.16 | −19.32 (−19.32..−17.69) | 0/30 | −23.26 (−23.69..−23.26) | 0/30 |
| + robust tier, σ_d pinned 0.002 | −78.27 (−78.27..−75.26) | 3–8/30 | −29.88 (−29.88..−29.68) | 0/30 |
| + robust tier, σ_d pinned 0.008 | −64.17 (−64.17..−61.39) | 0–2/30 | −28.95 (−28.95..−28.79) | 0/30 |
| + robust tier, σ_d pinned 0.08 | −23.93 (−23.93..−22.53) | 0/30 | −23.81 (−23.93..−23.70) | 0/30 |
| + robust tier, σ_d pinned 0.16 | −19.75 (−19.75..−18.60) | 0/30 | −23.20 (−23.55..−23.20) | 0/30 |
| + robust tier, band drawn live | −46.67 (−50.45..−46.08) | 0–3/30 | −26.82 (−27.58..−26.56) | 0/30 |

What bounds these tables:

- n = 30 is one pose observed 30 times, with frame 0 inside the settle transient, not 30
  poses.
- The clean-sim row is the 2026-08-22 80×45 direct render made before the #218 upgrade, a
  depth path v3 did not train on. No clean frame at the capture pose exists on v3's path,
  and none can be rendered: seed-42 room generation moved across the upgrade
  (deploy-resolution-depth-2026-09-19 §4). The node row is the only clean field at this
  pose on v3's path, and it is one frame.
- The uniform fields have no floor structure. They show what a policy does with a
  featureless field, not with a wall.
- Every number is one tick from a zero hidden state; none describes closed-loop behaviour.
- σ_d 0.28 is not a row: the band's top stays at the shipped 0.16.

## 5. The training depth's texture against the bridge capture

The statistic is `strafer_shared.depth_texture.texture_stats_by_band` in residual mode —
the frame minus its clean reference — with the near-field class excluded, over the bands
0.4–1.0, 1.0–1.5, 1.5–2.5, 2.5–3.5 and 3.5–5.5 m, in normalised depth (metres / 6). The
**bridge capture** column in every table is the bridge capture's tick-0 residual against
the pose-matched clean frame, from depth-noise-coverage-2026-09-18 §4
(`coverage/coverage_curve.json`, `bands[].capture`). The three draws are the band's two
ends and its log-uniform median: σ_d 0.002, 0.0178885 (√(0.002 × 0.16)) and 0.16. The
stereo term runs alone, holes off, at seed 7, which is §4's own convention.
`coverage_curve.py` from that deposit re-runs unmodified to byte-identical JSON
(`c87f70b7…`), reproducing the bridge capture column and all three of its σ_d columns.
Source for every table here: `tables/texture/texture_band_draws.json`.

**Table (b1)**, on the pose-matched anchor frame, p95 of the residual:

| band (m) | pixels | bridge capture | σ_d 0.002 | σ_d 0.0178885 | σ_d 0.16 |
|---|---:|---:|---:|---:|---:|
| 0.4–1.0 | 190 | 0.00000106 | 0.00000633 | 0.00005659 | 0.00050617 |
| 1.0–1.5 | 76 | 0.00000025 | 0.00001327 | 0.00011867 | 0.00106130 |
| 1.5–2.5 | 535 | 0.00007583 | 0.00004585 | 0.00041012 | 0.00366820 |
| 2.5–3.5 | 613 | 0.00025481 | 0.00010580 | 0.00094620 | 0.00846310 |
| 3.5–5.5 | 766 | 0.00145854 | 0.00019588 | 0.00175188 | 0.01566937 |

**Table (b2)**, on the 30 deploy-resolution frames of the deploy-resolution-depth-2026-09-19
deposit (640×360 reduced by the shipped term, `Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0`,
seed 42), median p95 over the frames. These frames are not pose-matched to the bridge
capture.

| band (m) | median pixels (frames) | σ_d 0.002 | σ_d 0.0178885 | σ_d 0.16 |
|---|---:|---:|---:|---:|
| 0.4–1.0 | 720 (30) | 0.00000660 | 0.00005909 | 0.00052848 |
| 1.0–1.5 | 573 (30) | 0.00001669 | 0.00014930 | 0.00133548 |
| 1.5–2.5 | 1084 (30) | 0.00004173 | 0.00037317 | 0.00333765 |
| 2.5–3.5 | 540 (30) | 0.00009557 | 0.00085485 | 0.00764606 |
| 3.5–5.5 | 333 (28) | 0.00016732 | 0.00149660 | 0.01338611 |

**Table (b3)**, the σ_d whose texture equals the bridge capture's in each band, and the
share of the band's log-uniform draws at or below it:

| band (m) | anchor σ_d | share at or below | deploy-resolution σ_d | share at or below |
|---|---:|---:|---:|---:|
| 0.4–1.0 | 0.000334 | 0.0 % | 0.000320 | 0.0 % |
| 1.0–1.5 | 0.0000374 | 0.0 % | 0.0000298 | 0.0 % |
| 1.5–2.5 | 0.00331 | 11.5 % | 0.00364 | 13.6 % |
| 2.5–3.5 | 0.00482 | 20.1 % | 0.00533 | 22.4 % |
| 3.5–5.5 | 0.0149 | 45.8 % | 0.0174 | 49.4 % |

In the two near bands every draw of the band is louder than the bridge capture — the
band's low end is 6.0× and 53× the bridge capture there — so no environment trains on a
near field as quiet as the bridge's. In the three far bands the bridge capture lies inside
the band: between its low end and its median at 1.5–3.5 m, and at the median at
3.5–5.5 m (1.20× on the anchor, 1.03× on the deploy-resolution frames).

The render path does not move the statistic. At the same 30 poses, the deploy-resolution
frames and the 80×45 direct frames agree within 0.32 % with the stereo term alone, and
within 3.7 % with the robust tier's holes added. The statistic is linear in σ_d — each
band's slope is constant to four significant figures across the three draws — so any other
draw is the slope times σ_d.

What bounds these tables:

- The bridge capture column is one frame at one tick. The single-frame spread over 20
  seeds at σ_d 0.16 is roughly ±15–35 % per band (`seed_spread_anchor_frame0`).
- The bridge capture's residual was taken against the direct render that preceded #222,
  so it carries the render-path drift deploy-resolution-depth-2026-09-19 §4 measured:
  0.028×, 7.4×, 0.029×, 0.249× and 0.238× of the column, band by band. It cannot be
  re-based, because the anchor pose cannot be re-rendered.

## 6. Found along the way

- **`--headless` on the command line disables every visualizer, and the stock scripts
  then raise.** `play_strafer_navigation.py` requests a Kit visualizer unconditionally, and
  `train_strafer_navigation.py` whenever `--video` is passed (its headless exemption was
  removed at `c4873e7`). On Isaac Lab v3.0.0-beta2.patch1 the deprecated `--headless` flag
  sets the launcher's disable-all switch while the requested `['kit']` still reaches the
  settings, so `SimulationContext` resolves no visualizer and raises
  `Explicitly requested visualizer(s) ['kit'] could not be configured`
  (`app_launcher.py:825-836`, `simulation_context.py:549-576`). Nothing is missing from
  the install; the `isaaclab_visualizers` `extension.toml` warning is unrelated.
  `HEADLESS=1` — or `args.headless = True` set in code — keeps the run headless without the
  switch, and the same script then records with the markers positioned
  (`markers/play_headless_env_cmd.sh`; `markers/play_headless_flag_unwatched.log` is the
  `--headless` control raising). The markers need a registered visualizer because this
  Isaac Lab version dispatches their callbacks only from `update_visualizers()`, which
  returns early without one. `rgb_array` capture itself needs no visualizer — `VideoRecorder`
  resolves a Kit capture of `/OmniverseKit_Persp` from the physics backend — and of
  `env_cfg.viewer` only `eye` and `lookat` reach the recorder, untranslated to world
  coordinates.
- **The run left no cfg record of its own.** The training script writes no params
  directory, and rsl-rl's `git/` directory in the run came out empty.
  `provenance/launch_provenance.json` stands in for it; the open brief
  `training-run-provenance-manifest` covers the gap.
- **`sample_memory.sh` never exits on its own.** `pgrep -f` matches the sampler's own
  command line, which carries the pattern, so `train/memory.txt` ends with 12 270 rows of
  the sampler observing itself after the trainer exited.
- **`tools/kit_boot_watchdog.sh --log` is the attempt's output copied at exit**, not an
  attempt ledger; the attempt line goes to the watchdog's stdout, here the last line of
  `train/train.log`.

## 7. What is not claimed

- Nothing about the real D555. Every depth frame here is simulated or Isaac Sim ROS 2
  bridge depth. There is no real-sensor behaviour, no value of ρ (the within-block
  correlation of real D555 depth noise), and no calibration of σ_d against the real
  sensor.
- None of these tables is the acceptance gate. The brief's gate — the sim-bridge rig-gate
  protocol with the fixed-goal leg — is not run here, and no threshold is applied to any
  row.
- Table (a) is one tick from a zero hidden state at one pose; it says nothing about
  closed-loop navigation.
- The training curves are each run's own training-distribution metrics; the gap between
  them is not a measure of policy quality.
- The uniform-field rows are featureless fields, not walls.

## Evidence — deposits

Two deposits in one commit: this record's file set, and the training run's checkpoints
and event file beside it. The paths this record names under `train/`, `export/`,
`smoke/`, `video/`, `tables/`, `probes/` and `provenance/` resolve in the first.

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directories | `depth-subgoal-v3-retrain-2026-09-21/record-files/` (this record's file set); `depth-subgoal-v3-retrain-2026-09-21/run_20260919_234233/` (the training run: every checkpoint and the event file) |
| deposit commit | `630d0799a7a8ed9cfe8dfcd75cf7093a493b09f7` |

Restore this record's file set into its directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/depth-subgoal-v3-retrain-2026-09-21/record-files/. \
      docs/measurements/depth-subgoal-v3-retrain-2026-09-21/
```

Verify either deposit's digests with:

```
cd Sim2RealLab-Artifacts/depth-subgoal-v3-retrain-2026-09-21/record-files     # or run_20260919_234233
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

The inputs this record reads from earlier deposits are listed with their digests in
[`provenance.md`](provenance.md).

sha256 of every file in `record-files/`:

```
5874a9d1c9325e6805192ad110e0e9a89a282c71ea6ab757378b5ce51b572592  export/export.log
efa78a0f2ee8a9852d1344ec720125ffc4dbee812ac3873c8ac5c2ae734ac2a1  export/export_cmd.sh
d22e3504e6aa2f0cefbcf2af41561c8acbab4ecf87a8c2e3a4b494294bf33ab5  export/strafer_depth_subgoal_v3_999.json
c866bfd54ec1a8352159e33d7875d41e3f07a442ff8301ba3700867932e2eb91  export/strafer_depth_subgoal_v3_999.onnx
bcfccf65009b3945ed40a171fa345188879a4b277e3930a76770f2329e03b896  export/strafer_depth_subgoal_v3_999.pt
bd11789619897558f810413b4ff80e7f3ad4445e1e83ed80b42a96822f3db6bf  export/watchdog_export.log
bd11789619897558f810413b4ff80e7f3ad4445e1e83ed80b42a96822f3db6bf  export/watchdog_export.log.attempt1
78a2350010aaf715d2416e7b9fd8f072bb5a54cfea8108bda8d3818c6abeb710  markers/depth_marker_leak.py
d6481d928af066f3b030ba3bd28e646135f3251f92191a223e661333e7ee3827  markers/depth_marker_leak_cmd.sh
d907bee0966a22c8ce7ec79013c847360e93ca92adc74b4d05219418c4cccd71  markers/depth_marker_leak_run1.log
1c74bfa73f6ef3b74002847eb846140e8bcfe5b192566c967583b63f2d192d89  markers/depth_marker_leak_run2.log
41870227a7fa34049e23bab751b21bd56fa45c6dcb30d697e4375bf3cde2cbab  markers/depth_marker_leak_run3.log
6accd3b49a5bc45e05a54f02e81197a67af1868cd08f05423ceccd00da72cdc4  markers/leak_run1/depth_marker_leak.json
10baf1eed455fb66e4d9ada71d68d8e615cc59dd7f1f56de9e2ebcb52a449a5f  markers/leak_run2/depth_marker_leak.json
820a21942460b8101343f36650e94e6f20d11a78e943c3aa8ace79de0c09d4fc  markers/leak_run3/depth_marker_leak.json
ffba1b203b74d412230edb40cc072cdae3dd5fd88ac9ec584786079f924aeb9b  markers/play_headless_env.log
9011501f66d65ae7818dfceaad71fd7a3dcaf3047e5996ceb34771fffc566238  markers/play_headless_env_cmd.sh
1339cb52ee1790b8fa99d6c7134863effba33c816b70fb9d24ba69afe357e985  markers/play_headless_flag.log
182c27b085b955060a62c5e88e22826c223b2ba9874e2916e9ea55ba0413940c  markers/play_headless_flag_cmd.sh
7e455f8fccad3d315f2492f27ba2925a0cab74d2bdbb367deb6a286b9304f1f8  markers/play_headless_flag_unwatched.log
089e810ee493d7ee5bdc7f74890078f4f34ce0f4d220e66b9e27406397a1b948  markers/play_videos/play_20260921_105717/rl-video-step-0.mp4
384c770425947b4dcbaa244129f7f3470f7e78c544426f3b0e9954cc1bdc65e7  markers/watchdog_depth_marker_leak_run1.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  markers/watchdog_depth_marker_leak_run1.log.attempt1
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  markers/watchdog_depth_marker_leak_run1.log.attempt2
384c770425947b4dcbaa244129f7f3470f7e78c544426f3b0e9954cc1bdc65e7  markers/watchdog_depth_marker_leak_run1.log.attempt3
7da3b348a4372096bca1a1790f1c87b847645b7f08ffd514f8a7b50c1318a7c3  markers/watchdog_depth_marker_leak_run2.log
7da3b348a4372096bca1a1790f1c87b847645b7f08ffd514f8a7b50c1318a7c3  markers/watchdog_depth_marker_leak_run2.log.attempt1
a900b162cdcd5cc43c8765cc7506c484150866e4b5cea35a71bd8fb44f6b1cea  markers/watchdog_depth_marker_leak_run3.log
1c8a5627a5334186b9b5e02d7717ed8816d70ac68008769ca2cd9c6908b7bed1  markers/watchdog_depth_marker_leak_run3.log.attempt1
a900b162cdcd5cc43c8765cc7506c484150866e4b5cea35a71bd8fb44f6b1cea  markers/watchdog_depth_marker_leak_run3.log.attempt2
1df3ace4f78224ab0a21f93a9bfdab82af1d20cc84b9be5b810eb21b71ecf8bd  markers/watchdog_play_headless_env.log
1df3ace4f78224ab0a21f93a9bfdab82af1d20cc84b9be5b810eb21b71ecf8bd  markers/watchdog_play_headless_env.log.attempt1
817f433582ba4dae44c69130a0ba7dd8bb707cd9902ea4c697010f7e1391048b  markers/watchdog_play_headless_flag.log
817f433582ba4dae44c69130a0ba7dd8bb707cd9902ea4c697010f7e1391048b  markers/watchdog_play_headless_flag.log.attempt1
817f433582ba4dae44c69130a0ba7dd8bb707cd9902ea4c697010f7e1391048b  markers/watchdog_play_headless_flag.log.attempt2
817f433582ba4dae44c69130a0ba7dd8bb707cd9902ea4c697010f7e1391048b  markers/watchdog_play_headless_flag.log.attempt3
b9e013c0f9a29f7e9e0aa460bb6f4c4f1dce7d524d7c7e755e5fd753c90dd3f0  probes/exported_policy_rollout_video.py
fea15ebed64193c53fbbf729fcfcafb5c38a3a15beef6f3008b6240ec58dd87d  probes/launch_provenance.py
0d776455914382b1de99144fbc43f42bb3f596df958c03d3e1b6354ba7576ca6  probes/sample_memory.sh
e51467c8d80ee800466fc662fd221bae1e2e1f7b8e2bb399ce3c989845126ef3  provenance/launch_provenance.json
60963cef4d794f9828377fa5b24697a33fa2a4d55a1b04830d9a53db0360e305  smoke/exported_policy_rollout.json
492b5494f7f84e768857a7f1e81cf2ed1ce44d1401e3fbe7e1b48ee9089c14aa  smoke/probe_sha256.txt
e772b9c2d7661f65b15dc681519bf4d60cef5bda93767dd494ae7c91b6721110  smoke/smoke.log
49384cd6ce0619d3b940210890833825a1a612ef5dcff0b7aa30b83e89bd79bc  smoke/smoke_cmd.sh
e44b4514b052f5df9ef1f455e2e34e43badf8c2b710cf0356eac04ecd5e2f75a  smoke/watchdog_smoke.log
e44b4514b052f5df9ef1f455e2e34e43badf8c2b710cf0356eac04ecd5e2f75a  smoke/watchdog_smoke.log.attempt1
8916afc2e600126f884af7e72c52939074b2164f258e4a206a22928cc51f22e1  tables/curves/curves.json
06091a9713af0648739e77eb04c1c0b14927c7d1d66f477cea8a38004cb58001  tables/curves/curves_binned.csv
702fb611c3619a5f5820345d575a06a84ce52d76b3303d76055e8a5256999eca  tables/curves/curves_summary.json
80ec1e18f14f2262031e88cfd276f864c8a342d775e82a73087872acd2600bfc  tables/curves/extract_curves.py
8de6f2a1b4e1d07a99634f4c90c01ad1e6463704bfd9a4c13803400b7230de7d  tables/curves/plot_curves.py
1a7b575bab956faf854e5208de0bf3b4c419ae588e892b14b57dba16ab86d687  tables/curves/training_curves.png
9bd0136a2cf0aca6560648482805dd68d8285d62d49e721d91d76f1de3f2c434  tables/samepose/candidate_sweep_v3/candidate_sweep.json
ea66411cf4bc9f29d33fe17b18afcc08aae24932dc134934146da7bdff5237a5  tables/samepose/candidate_sweep_v3/stdout.log
ce553701d1c63f44fad18cd0afe86a1cc8a6d78306e3ae25419f80b73ec4aee9  tables/samepose/table_a_node_base.json
49928a72d8053006eead5fe3e74b2f274accd439919c47490fedf4ec45d5b605  tables/samepose/table_a_node_base.py
2776bf83a34302b8ac6b20764a7c9961b3b2d12e35f1ea01a6224d0fbca0c9ea  tables/samepose/table_a_node_base.stdout.log
a848bd42184efbdaff3458cac1028a8963c768b844386a64cd098d89e013bb8a  tables/samepose/table_a_samepose.json
2b1f083aae9daa430947032b63a7d302f15089a2032b477fe0e93b83092c0cbc  tables/samepose/table_a_samepose.py
16368f30febe39880b05412a3419ce4ab837b1d3a199634197db01b1a6f385ec  tables/samepose/table_a_samepose.stdout.log
97699a22de6314fbc3719aa33f0deef168f6dd9f833c6aeb663843c10220d57f  tables/samepose/validate/candidate_sweep_v2/candidate_sweep.json
522f99df890027149848c46e6fcb429d032af96c5b9f888311284333309d64c0  tables/samepose/validate/candidate_sweep_v2/stdout.log
709e73679f10901f9eb3127f5b956fa9173a10177e89f6342f9c56731bc1bfb8  tables/samepose/validate/convention_ab_v2.stdout.log
3ef84e86e64cb5ef0f5f52a59097987222d814223c666e622dac07247d6d3c26  tables/samepose/validate/convention_ab_v2/convention_ab.json
9552b86cc3deab61d53aa2cc680ce6132b4024ea8cae9f0a08287f20ee2e313f  tables/texture/texture_band_draws.json
f40f2a9c85ba43f320adabca8fdee6d594824170b7c55c4ad03ead2db4cc6bdb  tables/texture/texture_band_draws.py
2b9740134066b65d9e4072efdb057ba87c41784ad109566f2f7cda050b261fc9  tables/texture/texture_band_draws_stdout.log
c87f70b7aabd956b213a4f3b01f7a5f66a56e9aec71c8a21b4e21ad1e7013517  tables/texture/validate/coverage_curve.json
4ee96c8cc6029be47f44d3d2bf12ae694e945fb5a1cb90b091cf2e337f712dd0  tables/texture/validate/coverage_curve.py
c565acbe41a08aa703b86639b9ff4bf32e1670c6aeecfc4429cbf87d52b5aafc  tables/texture/validate/coverage_curve_stdout.log
68504d0658134bab4811ef96cb15de0c90f331ca556593a066d0314851a9f3b3  train/launch_cmd.sh
ddc60b63257829595bd6e486426b5cd8955968f5759e3be65d33515b44b114b3  train/memory.txt
40b2c682d0644c936936a1dc8340f53ba66460e74879e699057c88a209ab0206  train/train.log
b15d6178b5d50c472d1f00c8692b73942dcc4ec1c9499fa7c57657c72355e976  train/watchdog.log
9da62bf1a195d9275be57b71ed54b8397a1da12a55f0c786e95dbe85891ddefc  train/watchdog.log.attempt1
eaf9ba7e0980bf58a89240d6f4eb89688ddae2ad9e35c92ce9b17852d7ee4228  video/exported_policy_rollout_video.json
96cff074ac8b798f5a30d46374ce1b73532efa5baf920284f333d3eb85395108  video/probe_sha256.txt
31c179f9afc3d43540bcd57fb1c4ed73fc93e16c2ee35384da2eb0e1696e195f  video/v3_smoke_overhead_graded.mp4
bbf65ca4d74cb30efec61a65811a8739242fb4c08dba1d854bfdcb4dfe9e1135  video/video.log
878edb3c06d9b3b30ebad5da98da4a65334ef63297780d62042ca76a23f519c3  video/video_cmd.sh
a9321db32280f17438e9d9b08a93faec980991703ee2e300e12fffb7c4fbf40a  video/videos/rl-video-step-0.mp4
d8da53d9857325d78c883d763e0f02f3cb5a19fd4971f3790c486c32fc5223b6  video/watchdog_video.log
d8da53d9857325d78c883d763e0f02f3cb5a19fd4971f3790c486c32fc5223b6  video/watchdog_video.log.attempt1
```

sha256 of every file in `run_20260919_234233/`:

```
be77aee14db15db6a988ddd724344f30acbac1b3de6df5804a22ae367724ace3  events.out.tfevents.1789879354.gx10-d1d8.667274.0
e9d6a69fd982e259ae3dea5df67f7577e65b437496d2b946f4786a03af704f5d  model_0.pt
547f49d0eb2d0a4586e64791d82a71592ff5ad9632a6dfeb3ff3992ca79c1024  model_100.pt
5caccc1e05c194cea5937b5481b290779c0ae293f2cf285f3c18d04464e64921  model_200.pt
720f138b2b4eb84c5aba82b879f9385f8b4ab76d1595d26f917892642fd11168  model_300.pt
df0f4f20ae8f9e1e108a15933d59170af862936e88b2e83d64bc1c3532fb8895  model_400.pt
d3a81cc02f7863a7b7ac9098a6b4517fbe7b38de928efa0eed424066c672eb08  model_500.pt
b8107b2d09b51e78c9a6d0e6aa7b7a1dec3138f0dbe390f0ea44c2ec9c7b3021  model_600.pt
a8531c7eb0a0b37cd42aea74fbd827fd9a588b8c1ca0ac227ee7f2a307db91c5  model_700.pt
c0ebf592331908d5569ea54eaf66e822185ab0163df722173edb95a574be4e68  model_800.pt
0196ca94ce8bb2d06df9b002134e01e3907c53f8fb9d83cdde84806e879cfbcd  model_900.pt
725fc6bfcf32ee756f70a459e45f2a62f17e14289a40e458a349f1a086c21484  model_999.pt
```
