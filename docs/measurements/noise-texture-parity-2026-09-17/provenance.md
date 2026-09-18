# Provenance — depth noise texture, training against the deploy path, 2026-09-17

captured_date: 2026-09-17

## Host

| role | host | kernel | driver |
|---|---|---|---|
| probes, gates | `gx10-d1d8` (DGX Spark, aarch64 GB10) | 7.0.0-1019-nvidia | 580.173.02 |

Everything in [`README.md`](README.md) was produced on this host. No robot was
involved, and that is one of the record's findings rather than a limit of the
measurement: there is no real-sensor depth in either repository to read.

## Trees

| item | value |
|---|---|
| Sim2RealLab, base of the work | `9c4d674` (`main`, merge of #219) |
| branch | `task/depth-delay-buffer-warmup` |
| Isaac Lab clone | `/home/zachoines/Documents/repos/IsaacLab` @ `v3.0.0-beta2.patch1` |

The canonical pair is Isaac Sim 6.0.1.0 with Isaac Lab `v3.0.0-beta2.patch1`.
The retired pair is not used anywhere in this record.

## Interpreter

`env_isaaclab3` on `gx10-d1d8`: python 3.12.13, numpy 2.3.1, torch 2.11.0+cu130,
onnxruntime 1.25.1, isaacsim 6.0.1.0. Every inference ran on
`CPUExecutionProvider`.

`LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` is set for the Kit-free runs, as
`env_setup.sh` does for this host. Every probe in the deposit is Kit-free and
runs from the repository root in a few seconds; only §9's gates boot Kit.

Each probe imports the production modules rather than reimplementing them:
`probes/candidate_sweep.py` and `probes/texture_structure.py` import
`DepthNoiseModel`/`DepthNoiseModelCfg`, and `candidate_sweep.py` and
`capture_provenance.py` reach `strafer_inference.obs_pipeline` by inserting the
ROS package's directory on `sys.path` (the package is not installed in this
environment).

## Artifacts

| artifact | sha256 |
|---|---|
| `models/strafer_depth_subgoal_v2_998.onnx` | `855e1df7d0dac3be7229f933b59546b26f18959f97966b9e2f2f22e752bf5165` |

The digest matches the one the attribution and convention-fix records published,
so §4 and §5 run against the same bytes those records measured. `models/` is
untracked repository content.

## Inputs this record reads from another record

Nothing here is re-derived; each input is cited by digest in the record it
belongs to.

| input | source |
|---|---|
| `same-pose-probe/gym_obs.jsonl` — the clean anchor frames | `goal-a-attribution-2026-08-22`, deposit `ae3e8c0ef50c80ca7832df3dec8d1c2831a5bb12` |
| `arm3-obs-capture/node_obs.jsonl` — the 1 799-record capture | same record's sibling deposit, `arm3-obs-capture/` |
| `ab/node_obs_rec0.json` — the capture's tick-0 row | `depth-convention-fix-2026-09-13`, deposit `a03ebd7` |
| the real-D555 per-pixel σ table | `docs/tasks/completed/d555-invalid-pixel-statistics.md` §4, transcribed into `probes/sensor_commensurability.py` as a literal |

The capture is stored compressed. Restoring and verifying it is the first step of
reproducing §1 and §3:

```
gunzip -k node_obs.jsonl.gz
sha256sum -c <<< "e480766eb33bc38728c7d24ab99f5428fe574b7063c6dc32faa2c97a58641224  node_obs.jsonl"
```

Restoring the 2026-08-22 record from its deposit is what makes §3 and §4
re-runnable from the paths their probes name:

```
cp -a <clone>/goal-a-attribution-2026-08-22/record-files/. \
    docs/measurements/goal-a-attribution-2026-08-22/
```

The restored copy is not part of any commit. Both probes read those files and
leave them byte-identical; nothing in this record writes into another record's
directory.

The real-sensor σ table is **transcribed**, not read from a file: the 2026-08-04
measurement lives only as a markdown table in its own brief, with no deposited
array behind it. §6 of the record says what that costs — no n per band, no
confidence intervals, and an unverified filter state.

## The before/after pair in §7

The delay-buffer warm-up figures are a two-tree measurement. The `after` arm ran
in the working tree on `task/depth-delay-buffer-warmup`; the `before` arm ran in
a **separate `git worktree` checked out at `9c4d674`**, with `PYTHONPATH` pointed
at that worktree's `source/strafer_lab`, so the pre-change tree was never
reconstructed by editing files in place. The same `probes/delay_warmup.py` ran in
both, differing only in its `--label`.

Both arms use `SEED = 31`, 64 envs and a 10-step post-reset window, and read the
shipped tier settings through `get_depth_noise(REAL_ROBOT_CONTRACT)` and
`get_depth_noise(ROBUST_TRAINING_CONTRACT)` rather than restating them, so the
latency ranges the record quotes are the ones the contracts actually carry.

## Seeds and what depends on them

`probes/candidate_sweep.py` runs every arm over **7** seeds: 7 first, then 1
through 6. `--seeds 8` is the bound on the generator `[7] + [s for s in
range(1, 8) if s != 7]`, which never yields 8, so the count is one fewer than
the flag reads; `sweep/candidate_sweep.json` records the list it actually used.
Seed 7 is the convention A/B's seed, kept first so the overlapping rows are
directly comparable with that record. Four rows have a seed-dependent rig
count — the two smallest-σ rows and two that reach 1 of 30 on one seed each —
which is why §4 quotes every row as a range.

`probes/texture_structure.py` and `probes/sensor_commensurability.py` use fixed
seeds (7 and 0). The median-of-64 attenuation factor is a 4×10⁵-draw Monte Carlo
and reproduces to the digits quoted.

## Evidence deposit

See the record's evidence section for the deposit directory, its commit and the
digest of every file.
