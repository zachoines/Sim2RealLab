# Isaac Lab / Isaac Sim pre-bump baseline — 2026-08-14

Numbers produced by the current sim stack, recorded before it is bumped. A
"no regression" claim after a stack change is only checkable against values
captured before it, and the classes that matter most here — a PhysX or RTX
re-default, an observation-manager behaviour change — move no config-hash
golden and no test count, so they are invisible unless the behaviour itself was
measured first.

Every number below states the exact command that produced it. Nothing in the
sim stack, the env configs, the goldens, the probes, or the test harness was
modified to obtain them.

## Provenance

| item | value |
|---|---|
| Sim2RealLab | `e7ea7bd62f850e23b6e8250395fc85a8b356a6b0` (`main`, merge of #201, 2026-08-13) |
| Isaac Lab checkout | `ae41e2aca68bcf06cb6ea02dd6618e4ddb16e1da` (`develop`, 2026-04-23), `VERSION` = 3.0.0 |
| `isaaclab.__version__` | 4.6.12 |
| conda env | `env_isaaclab3`, python 3.12.13 |
| isaacsim | 6.0.0.0 |
| torch | 2.10.0+cu130 |
| rsl-rl-lib | 5.0.1 |
| onnxscript | 0.6.2 |
| onnx / onnxruntime | 1.21.0 / 1.25.1 |
| numpy / warp-lang | 2.3.1 / 1.12.0 |
| driver / CUDA | 580.82.09 / 13.0 |
| GPU | NVIDIA GB10 |
| captured | 2026-08-14, 15:45–17:55 local |

Full interpreter state: [`pip-freeze-env_isaaclab3.txt`](pip-freeze-env_isaaclab3.txt),
[`provenance.md`](provenance.md). Machine-local conditions the migration
inherits — duplicate `isaaclab` dist-info, an editable install pointing at a
deleted worktree, the two local Kit modifications, the standing warning set, and
the warp-array shape of `Articulation.data` — are written up in
[`env-facts.md`](env-facts.md).

---

## 1. Determinism floor (D0)

**Verdict: this stack is bit-reproducible run to run. Physics comparisons after
the bump can be hash gates, not band gates.** No same-pin, same-seed
rerun-identity evidence existed anywhere in the repo before this; without it a
post-bump physics difference could not be read as a regression, because the
spread between two runs of the *same* build was unknown.

| probe | runs | result |
|---|---|---|
| roller probe, PGS | 4 independent Kit boots | all four CSVs byte-identical (`8a40c79d2a5b63de…`) |
| roller probe, TGS | 2 independent Kit boots | both CSVs byte-identical (`2302cdc945252e47…`) |
| pose trace, 16 envs / 300 steps / DR active | 2 independent Kit boots | `dr_hash`, `trace_hash`, `action_hash` and the `.npz` all identical |
| export trajectories, 3 artifacts × 2 formats × 2 sequences | 2 evaluations each | max action delta `0.0`, max hidden delta `0.0` |

The pose trace is the strong form: 16 parallel envs, domain randomization live,
episode resets occurring inside the rollout, and the post-reset draw hashed
separately from the trajectory so an RNG-order change would be attributable
apart from a physics change. Both hashes matched.

```
$ISAACLAB -p docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/pose_trace_probe.py \
    --headless --num_envs 16 --seed 42 --steps 300 \
    --out docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/physics/pose_trace_run1.npz
```

| field | value |
|---|---|
| env | `Isaac-Strafer-Nav-RLNoCam-v0`, `env_cfg.seed = 42`, 16 envs, 300 steps |
| `dr_hash` | `194fe656d3974568a297a5fad35b349c9f52a10c2123fb5cc301b0c679b118f6` |
| `trace_hash` | `c6632a4e63c9e19198802dfc2c8c7adc0eb53c79fc26420d4b98e29fae934aaa` |
| `action_hash` | `8dce355fb1188a4066db77c7c0955708c91bd65f1dbdcc651f54d4a94e2bf3fe` |
| DR fields captured | `root_state_w`, `joint_pos`, `joint_vel`, `masses`, `inertias`, `material_properties` |
| DR fields unavailable | none |

The trace is not degenerate, which matters because a probe that recorded nothing
would also hash identically. Over the 300 steps: every one of env 0's 300 poses
is distinct, per-env planar path length ranges 1.87–10.36 m, the 16 envs are
spread across a standard deviation of 11.4 m in x, all quaternions are unit-norm,
the action stream is uniform on [−1, 1] with 4800/4800 distinct rows, and the
domain-randomized body masses take 160 distinct values across 16 envs × 105
bodies. The reproduced quantity is rich, varied, DR-driven motion.

Raw: [`physics/pose_trace_run1.npz`](physics/pose_trace_run1.npz) +
[`.json`](physics/pose_trace_run1.json), and the `run2` pair.

---

## 2. Physics — roller probe

`--solver-type 0` is passed explicitly on every PGS leg: the probe's own default
is `1` (TGS), which is **not** what the nav envs ship.

```
$ISAACLAB -p source/strafer_lab/scripts/roller_bounce_probe.py --headless \
    --solver-type 0 --omega-fracs 0.25,0.5,0.75,1.0 --duration 5.0 \
    --csv docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/physics/roller_z_pgs_run1.csv
```

### PGS — the shipped configuration

| ω frac | late p2p (mm) | growth | dominant f (Hz) | roller (rad/s) | max z (mm) |
|---|---|---|---|---|---|
| 0.25 | 0.266 | 0.53 | 12.80 | 66.4 | 48.1 |
| 0.50 | 0.945 | 0.72 | 25.60 | 118.5 | 48.1 |
| 0.75 | 1.598 | 0.81 | 38.60 | 154.8 | 48.2 |
| 1.00 | 2.108 | 0.48 | 9.80 | 185.4 | 50.1 |

Ride height, from the `--inspect` leg (`--solver-type 0`): **rest 47.99 mm**,
spin-mean 47.56 mm, min 47.31 mm. The completed roller brief's recorded value is
47.98 mm, so the ride height has not moved.

### TGS — positive control

Run so the baseline can state that the probe still detects the fault class it
exists to detect, rather than assuming it.

| ω frac | late p2p (mm) | growth | roller (rad/s) | max z (mm) |
|---|---|---|---|---|
| 0.25 | 0.434 | 0.68 | 102.5 | 48.6 |
| 0.50 | 2.581 | 0.54 | 183.6 | 52.0 |
| 0.75 | 6.820 | 0.71 | 261.9 | 56.4 |
| 1.00 | **25.121** | 1.02 | 287.4 | 72.6 |

At full wheel speed TGS separates from PGS by **11.9×** on late peak-to-peak
(25.121 mm vs 2.108 mm) and by 22.5 mm on peak chassis height. The probe sees the
fault class.

### One candidate post-bump band does not hold at the baseline

Four bands have been floated for a post-bump roller gate. Checked against the
*current, healthy* stack, three hold and one does not, so it cannot be used as
written:

| candidate band | baseline (PGS) | holds? |
|---|---|---|
| late p2p ≤ 3 mm at every frac | max 2.108 mm at frac 1.0 | yes, 1.4× margin |
| growth ≤ 1.3 | max 0.81 | yes |
| roller ≤ 100 rad/s at frac 1.0 | **185.4 rad/s** | **no — exceeded at baseline** |
| ride height 48 ± 1 mm | 47.99 mm | yes |

The roller-speed band appears to have been carried over from a "healthy 55 rad/s"
figure that this configuration does not reproduce; the measured PGS/TGS split on
that quantity is 185.4 vs 287.4 rad/s, a 1.55× separation, far weaker than the
11.9× that late peak-to-peak gives. **Late peak-to-peak and ride height are the
discriminating quantities; roller angular velocity is not.** The band needs
re-deriving from this table before it gates anything.

Reproduce the tables from the committed CSVs, and diff runs pairwise:

```
python3 docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/analyze_roller_csv.py \
    docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/physics/roller_z_*.csv \
    --out docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/physics/roller-metrics.json
```

The probe prints these numbers itself, but Isaac Sim tears the process down with
`os._exit` and the buffered table does not survive it — the first pass produced
correct CSVs and no console summary. Re-running under `PYTHONUNBUFFERED=1`
recovers it. The CSVs are the artifact; the recomputation script is how the
numbers come back from them.

---

## 3. Behavioural evaluation

```
$ISAACLAB -p source/strafer_lab/scripts/eval_cadence_emulation.py \
    --env Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0 \
    --checkpoint logs/rsl_rl/strafer_navigation/run_20260727_171735/model_998.pt \
    --profile clean --num_envs 16 --episodes 100 --seed 42 --headless \
    --out-dir docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/eval
```

| metric | value |
|---|---|
| episodes | 101 |
| **completion rate** | **0.8119** (82 / 101) |
| `path_complete` | 0.8119 |
| `sustained_collision` | 0.1584 |
| `off_path_divergence` | 0.0297 |
| progress fraction, mean / median | 0.8680 / 0.9570 |
| near-arrival rate | 0.5743 |
| steps, mean | 123.04 |
| direction offset, median | 4.510° |
| direction offset, median abs | 13.815° |
| direction offset, p10 / p90 | −71.490° / 42.837° |
| **fraction left** | **0.6216** |
| realized profile | 13344 ticks, all fresh, 0 held, 0 stale, 30.0 Hz |

Raw: [`eval/cadence_20260814_160942.jsonl`](eval/cadence_20260814_160942.jsonl)
(one object; `episodes[]` carries per-episode cause, progress fraction and tick
accounting).

**This number supersedes the 0.720 recorded for the same checkpoint on
2026-08-10**, and the difference is attributable rather than noise. The two runs
share checkpoint, env id, profile, seed, env count, tick rate and warm-up; the
entire gap sits in one cause bucket:

| cause | 2026-08-10 | 2026-08-14 |
|---|---|---|
| `path_complete` | 0.72 | 0.8119 |
| `sustained_collision` | 0.15 | 0.1584 |
| `off_path_divergence` | **0.13** | **0.0297** |

Between those dates the referent-drift work landed — `4eb8eeb` bounds the
referent-drift band by the off-path corridor and `e7b5146` gives a drift arm its
own displacement, both ancestors of `e7ea7bd` via #200 and #201 — and the newer
JSONL carries `env_drift_active` / `harness_drift_gain` fields the older one does
not. Off-path divergence is exactly the quantity that work targeted. The
post-bump comparison must be made against **0.8119**, not against the older
figure, and if more tree movement lands before the bump the reference should be
re-measured again rather than carried forward.

At p ≈ 0.81 and n = 101, one standard error is 0.039, so a ±2·SE band is
**0.734 – 0.890**. `--episodes 400` would halve it.

---

## 4. Depth observation statistics

The config-hash goldens cover the depth term's declaration, never the tensor it
produces, so a renderer change that alters the picture a deployed checkpoint is
fed moves no golden. Actions are held at zero and the observation is sampled from
the reset pose, so the numbers track the renderer rather than the physics.

```
$ISAACLAB -p docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/depth_obs_stats.py \
    --headless --env Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0 \
    --num_envs 8 --seed 42 --frames 30 \
    --out docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/render/depth_obs_enriched_robust_play.json
```

30 frames × 8 envs × 45 × 80, values already scaled by `1/DEPTH_MAX`.

| statistic | Enriched-Robust-Play | Subgoal-Real-Play |
|---|---|---|
| mean | 0.351818 | 0.283900 |
| std | 0.315225 | 0.303940 |
| min / max | 0.0 / 1.0 | 0.0 / 1.0 |
| p01 / p50 / p99 | 0.0 / 0.210356 / 1.0 | 0.0 / 0.159985 / 1.0 |
| fraction at min | 0.058333 | 0.041667 |
| fraction at max | 0.098672 | 0.080546 |
| row-band mean, top 3 rows | 0.4757, 0.4797, 0.4795 | 0.5617, 0.5471, 0.5333 |
| row-band mean, bottom 3 rows | 0.1365, 0.1326, 0.1266 | 0.1201, 0.1166, 0.1122 |
| sha256 (float64 stack) | `f4c3375ad9c6565a1d0f0d6db14193f3c8dd747ac4f7ce2beb3037e154b44b09` | `a6529378732e3c5a1d66fde436d86b651ed68f1decf5bb7562ade710aa8bbb2e` |

The full 45-entry row-band and 80-entry column-band profiles are in the JSON.
The row profile is the vertical-FOV fingerprint. On the Enriched-Robust tier it
holds a plateau near 0.48 across the top third (peaking at 0.4815 on row 4) and
then falls to 0.1266 at the bottom row; the Real tier starts at its maximum
0.5617 and falls throughout. Neither is strictly monotonic — 35 of 44 row-to-row
steps are non-increasing on the robust tier, 42 of 44 on the real tier — so the
comparison quantity is the *shape* of the profile, which a renderer that starts
honouring the authored vertical aperture would change and a single mean would
hide.

**Frame 0 of the Enriched-Robust tier is all zeros** (`per_frame_mean[0] = 0.0`);
the Real tier's is 0.1687. The robust tier carries frame-drop depth noise and the
draw is seeded, so this reproduces — but it means 1/30 of the robust aggregate is
a degenerate frame. Compare frames 1–29 if the aggregate ever needs to be read
without it; `per_frame_mean` in the JSON makes that possible without a re-run.

---

## 5. Config-hash goldens

All 26 stored goldens recompute byte-exact at `e7ea7bd`, and the full canonical
JSON preimage of every hashed object is dumped so the post-bump comparison names
the moved fields instead of just reporting that a hash moved.

```
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
$STRAFER_ISAACLAB_PYTHON \
  docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/dump_golden_preimages.py \
  --out docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/goldens
```

| golden class | count | status |
|---|---|---|
| contract hashes (composed RL variants) | 22 | 22/22 byte-exact |
| depth-obs golden | 1 | byte-exact |
| policy-obs layout goldens (`depth`, `nocam`) | 2 | byte-exact |
| palette golden | 1 | byte-exact |

Preimages: [`goldens/preimages/`](goldens/preimages/) — one JSON per hashed
object, `contract-<variant>.json`, `depthobs-RLDepth_Real.json`,
`layout-depth.json`, `layout-nocam.json`, `palette-pre_enrichment.json`.
Recomputed values and the stored constants sit side by side in
[`goldens/golden-hashes.json`](goldens/golden-hashes.json), which also records
which observation profile each of the 22 variants resolves to.

The dump script imports the contract test module by path and calls its own
`_canon` / `_contract` / `_hash`, so a preimage is byte-identical to the one that
produced the stored golden rather than a reimplementation of it.

---

## 6. Export anchors

Comparison anchors for the export-equivalence gate. `models/` is untracked and
the source checkpoint lives under gitignored `logs/`, so both were copied to a
preserved location outside the repo before anything else; the frozen trajectories
are committed here.

```
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 CUDA_VISIBLE_DEVICES= \
$STRAFER_ISAACLAB_PYTHON \
  docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/dump_export_trajectories.py \
  --models-dir models \
  --out docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/export-anchors
```

128 steps per sequence, two sequences per artifact (`normal` = standard normal
over every dim; `indist` = the 19 scalar dims N(0,1) and the 3600 depth dims
U[0,1]). ONNX threads `h_out → h_in` through raw `sess.run`; TorchScript carries
state in the `hidden_state` buffer, zeroed once at sequence start. The
observation sequences are stored **inside** the `.npz` and are replayed with
`--replay`, so the post-bump comparison never depends on a numpy or torch RNG
stream staying stable. The replay path was exercised against this dump: all 22
arrays reproduce with max delta 0.0 and the re-written `.npz` carries the same
sha256, so `--replay` is known to work rather than assumed to.

| artifact | obs dim | recurrent | formats |
|---|---|---|---|
| `strafer_depth_subgoal_v2_998` | 3619 | yes | `.pt` + `.onnx` |
| `strafer_nocam_subgoal_v0` | 19 | no | `.pt` + `.onnx` |
| `strafer_nocam_subgoal_gru_smoke` | 19 | yes | `.pt` only |

### Re-evaluation determinism

Every artifact × format × sequence evaluated twice: **max action delta 0.0, max
hidden delta 0.0**. The anchor reproduces exactly, so any post-bump delta is
signal.

### Cross-format agreement, measured rather than assumed

| artifact | sequence | max action delta | max hidden delta |
|---|---|---|---|
| `strafer_depth_subgoal_v2_998` | normal | 2.921e−06 | 1.492e−05 |
| `strafer_depth_subgoal_v2_998` | indist | 2.623e−06 | 1.676e−05 |
| `strafer_nocam_subgoal_v0` | normal | 2.861e−06 | — |
| `strafer_nocam_subgoal_v0` | indist | 2.384e−06 | — |

**These are the pre-bump `.pt`-vs-`.onnx` agreement levels, and they are far
above the 7.5e−8 float-noise figure that has been used to calibrate export
tolerances** — 39× on actions and 224× on hidden state. That figure measures
eager PyTorch against legacy ONNX; this measures TorchScript against ONNX over a
128-step threaded GRU rollout, where the hidden state accumulates.

The candidate export tolerances still clear at baseline, but with far less room
than a 7.5e−8 anchor suggests: **3.4× on the ≤1e−5 action bound and 6.0× on the
≤1e−4 hidden bound.** What actually carries the gate is the distance to the
known-wrongness class (0.064 action / 0.193 hidden), which is still 4.3 and 4.1
decades away. Set the tolerance from 1.7e−5, not from 7.5e−8, and state the
margin as the separation from the wrongness class rather than from float noise.

Raw: [`export-anchors/export-trajectories.npz`](export-anchors/export-trajectories.npz)
(3.78 MB) + [`export-trajectories-manifest.json`](export-anchors/export-trajectories-manifest.json)
(per-sequence moments, artifact SHA256s, sidecar contents, and the torch / ORT /
onnx / onnxscript / rsl-rl / numpy versions the numbers were produced under).

### Preserved binaries

Copied outside the repo, byte-verified against the live tree, and **not
committed**:

| file | sha256 |
|---|---|
| `model_998.pt` (the v2 source checkpoint) | `effaf5de095da1313309a50c94e9d49080ece03b9153caae8eb0c8a03c4a1f17` |
| `strafer_depth_subgoal_v2_998.pt` | `03871cb5a09bffd2383994bd00f8245ea68a4aec1cac33f98a5ac6ae5d637b37` |
| `strafer_depth_subgoal_v2_998.onnx` | `855e1df7d0dac3be7229f933b59546b26f18959f97966b9e2f2f22e752bf5165` |
| `strafer_depth_subgoal_v2_998.json` | `3b5d3e909eeb6f9e36fb466c2cd979468bbc29014290e278bba5a32a446017e2` |
| `strafer_nocam_subgoal_v0.pt` | `4168a655a50dc928ff697e5e9bf3277999199a2bd8f30e5e1ff9dcabc0e3ef56` |
| `strafer_nocam_subgoal_v0.onnx` | `95f06bc7c27f25139145f78d8c103a339f130933985424410f66477a3e6c60d2` |
| `strafer_nocam_subgoal_v0.json` | `e812c46251c6d7b9a9cf50f434cf611d882a6b4dc821b50c7d2a5b9cc56fa892` |
| `strafer_nocam_subgoal_gru_smoke.pt` | `3cac49dcec9b81c7c3fb9cce98b17b73a677513fda7c2786365f1bd1e962bfeb` |
| `strafer_nocam_subgoal_gru_smoke.json` | `6d8fa1f1cfcc7729105a910c3f1055c92eed99b18a4afba038e81c4de1872b4b` |

The full 20-entry manifest, covering the `v0` / `v1` depth artifacts and the
other five checkpoints of the same training run, travels with the preserved copy
as `SHA256SUMS.txt`. The `v2_998` sidecar predates the `trained_period_s` field,
so any re-export from this checkpoint adds it.

---

## 7. Test-suite counts

| suite | command | result |
|---|---|---|
| Kit (`run_tests.py all`) | `source env_setup.sh && make test-lab` | **486 tests / 14 suites**, 485 pass, 1 fail (the known flake) |
| pure (`test-lab` second half) | same invocation | **1230 passed, 1 skipped**, 150.11 s |
| contract subset, no Kit | the two-file `--noconftest` command below | **147 passed** (132 + 15), 16.51 s, exit 0 |
| `test-autonomy` (`.venv_vlm`) | `make test-autonomy AUTONOMY_PY=$PWD/.venv_vlm/bin/python` | **701 passed**, 79 deselected |
| `test-vlm` | `make test-vlm` | **143 passed** |
| `env-check` | `make env-check` | PASS, exit 0 |
| `test-ros` / `test-driver` | `tools/run_ros_tests.sh ros` | **unknown — needs a Jetson run or a built container** |

```
LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \
$STRAFER_ISAACLAB_PYTHON -m pytest --noconftest -q \
  source/strafer_lab/test_sim/env/test_composition_contract.py \
  source/strafer_lab/test_sim/env/test_obs_contract.py
```

### Per-suite Kit counts

| suite | tests | suite | tests |
|---|---|---|---|
| terminations | 13 | env | 268 |
| events | 14 | noise_models | 55 |
| commands | 8 | depth_noise | 6 |
| observations | 13 | imu | 4 |
| curriculums | 4 | obs_dump | 4 |
| rewards | 46 | camera_jitter | 3 |
| sensors | 1 | **total** | **486** |
| actions | 47 | | |

The 486/14 total is unchanged from the 2026-08-09 run; the tree moved but the Kit
test count did not.

### The IMU flake, dispositioned

`test_collision_imu_mean_differs_from_free` **failed** in the full run:

```
AssertionError: IMU distribution not significantly different during collision.
Free mean: 15.60 m/s², Collision mean: 15…
```

The suite was re-run once — `$ISAACLAB -p source/strafer_lab/run_tests.py imu` —
and **passed 4/4**. Both outcomes are the baseline: this suite is a coin flip and
a post-bump comparison that treats a single imu failure as a regression will be
wrong about half the time.

The junit XMLs in [`suites/kit-junit-xml/`](suites/kit-junit-xml/) carry the
**rerun** (green) imu result, because `run_tests.py` writes to a fixed path and
the rerun overwrote it. The failure above is from the full-run console. Those
files are copies of the run's output renamed from `test_results_*.xml` to
`kit-suite-*.xml`, because the original name is gitignored as live test output.

### Tests the baseline does not cover

Two Kit test files are absent from every `run_tests.py` SUITES entry and are
therefore never executed by `make test-lab`:

| file | test functions |
|---|---|
| `source/strafer_lab/test_sim/sensors/test_d555_perception_cfg.py` | 27 |
| `source/strafer_lab/test_sim/sensors/depth_noise/test_scene_cfg.py` | 2 |

37 of the 39 files under `test_sim/` are reachable through SUITES; these two are
not, so **29 test functions of assumed protection are not running** and are not
in the 486. Recorded, not fixed — changing the SUITES map is a change to the
instrument.

### `test-ros`

```
$ tools/run_ros_tests.sh ros
ERROR: no native ROS toolchain and image 'strafer-cpu:humble' is missing.
       Build it first:  make images
```

`colcon` is absent, `/opt/ros` is absent, and no `strafer-cpu:humble` image is
built on this host. The count is unknown here, and the post-bump Jetson-lane
comparison has no pre-bump anchor until one is taken on a Jetson or a built
container.

---
## 8. Render — training video exposure

One 200-frame overhead clip from a short training launch, measured with the
in-tree exposure tool. Numbers, not the verdict: the tool's default bands are
calibrated for perception-camera frames, and this is a 1280×720 overhead viewer
render, a different image entirely.

```
$ISAACLAB -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0 \
    --num_envs 64 --max_iterations 6 --seed 42 --headless \
    --video --video_length 200 --video_interval 2000 --log_dir <scratch>

$STRAFER_ISAACLAB_PYTHON \
  source/strafer_lab/scripts/measure_perception_exposure.py <clip>.mp4
```

| metric | value |
|---|---|
| frames | 201 |
| resolution (H×W) | 720 × 1280 |
| mean_rgb | 87.6, 86.4, 84.8 |
| **mean_luma** | **86.6** |
| clip_frac | 0.0 |
| crush_frac | 0.0056 |
| white_frames | 0 |
| row_bands_top_to_bottom | all ten bands 0.0 |
| tool verdict | FAIL — `mean_luma 86.6 outside [90.0, 150.0]` |

Two things the post-bump comparison needs to know about this row:

- **The FAIL is a band-domain mismatch, not a render regression.** The
  `[90, 150]` luma window and the 2 % clip ceiling were set for the capture
  lane's RGB perception frames. The overhead training view sits at 86.6, just
  under the floor, and does so on a healthy stack. Compare `mean_luma` against
  86.6, not against 90.
- **The row-band profile is degenerate here and carries no information.**
  `row_bands_top_to_bottom` reports the *clipped-pixel* fraction per band, and
  `clip_frac` is exactly 0.0, so all ten bands read 0.0 by construction. It
  cannot detect a change until something starts clipping. The depth row-band
  profile in section 4 is the render fingerprint with actual shape in it.

The clip itself (428740 bytes, sha256
`d329671c459fa895d1978664f993dc59189ad5250705aada234aa4a56f412301`) is preserved
outside the repo with the other binaries. Raw tool output:
[`render/exposure-train-video.txt`](render/exposure-train-video.txt).

---
## 9. Training-curve leg

A baseline *shape* for `train_strafer_navigation.py`, the one Tier-1 entry point
with no automated coverage at all. No recipe knob is touched — this is not an
experiment, and the curve is not a result to be improved on. What it establishes
is that the training path runs end to end, at what throughput, and with what
early trajectory, so a post-bump run can be overlaid on it.

```
$ISAACLAB -p source/strafer_lab/scripts/train_strafer_navigation.py \
    --env Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-v0 \
    --num_envs 64 --max_iterations 100 --seed 42 --headless \
    --log_dir <scratch>
```

`--max_iterations 100` runs iterations **0 through 99**; there is no iteration
100. The last row below is the final iteration, not a 101st.

| iteration | steps/s | mean reward | mean episode length | collection s | learning s | value loss | entropy | action std |
|---|---|---|---|---|---|---|---|---|
| 0 | 39 | −0.28 | 35.20 | 20.192 | 57.648 | 0.0109 | 0.6220 | 0.30 |
| 10 | 52 | −3.82 | 179.29 | 18.328 | 40.371 | 0.0375 | 0.5940 | 0.30 |
| 25 | 53 | −4.81 | 278.72 | 19.501 | 37.891 | 0.0215 | 0.5514 | 0.29 |
| 50 | 53 | −5.06 | 319.09 | 18.037 | 38.938 | 0.0141 | 0.5611 | 0.29 |
| 75 | 54 | −4.60 | 334.22 | 17.559 | 38.387 | 0.0173 | 0.5825 | 0.30 |
| **99** | **55** | **−5.09** | **385.86** | 18.297 | 37.069 | 0.0184 | 0.5641 | 0.30 |

Throughput across all 100 iterations: **min 35, max 58, mean 53.4 steps/s** at
64 envs × 48 rollout steps (3072 steps per iteration). Wall clock 16:18:23 →
17:54:22, about 96 minutes. **Iteration 0 is not representative** — it carries
first-iteration warm-up (39 steps/s, 57.6 s of learning time against a ~38 s
steady state) and should be excluded from a throughput comparison. Use the mean
of 53.4, or the iteration-50 value of 53.

The shape to overlay, not to judge: episode length climbs steadily (35 → 386)
while mean reward falls and then partly recovers (−0.28 → −5.06 at iteration 50
→ −4.60 at 75 → −5.09 at the end). A fresh policy surviving longer accumulates
more per-episode penalty before it learns to complete, so a falling reward
alongside a rising episode length is the expected early trajectory here, and
100 iterations is far too short for completion behaviour to appear. The ±15 %
throughput band around 53.4 steps/s is **45.4 – 61.4**.

Full per-iteration data for all 100 iterations, ten columns:
[`training/training-curve.csv`](training/training-curve.csv). The five sampled
rows above, machine-readable:
[`training/curve-iters.json`](training/curve-iters.json).

---

## 10. File manifest

SHA256 of every committed file in this directory:
[`MANIFEST.sha256`](MANIFEST.sha256). Verify with:

```
cd docs/measurements/isaac-lab-upgrade-baseline-2026-08-14 && \
  sha256sum -c MANIFEST.sha256
```

Binaries deliberately kept outside the repo, with hashes recorded above:
the v2 source checkpoint and the other five of its training run, every
`models/` artifact, the two roller `--inspect` MP4s, the training video clip,
and the two depth-observation `.npz` stacks. They live together with a
20-entry `SHA256SUMS.txt` covering the model artifacts and checkpoints.
