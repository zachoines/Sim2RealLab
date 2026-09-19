# Provenance — depth texture coverage, 2026-09-18

captured_date: 2026-09-18

## Host

| role | host | kernel | driver |
|---|---|---|---|
| probes, gates, training arms | `gx10-d1d8` (DGX Spark, aarch64 GB10) | 7.0.0-1019-nvidia | 580.173.02 |

No robot was involved. As the 2026-09-17 record established, there is no
real-sensor depth in either repository to read, so every depth field here is
either a training render or the Isaac Sim bridge capture of 2026-08-22.

## Trees

| item | value |
|---|---|
| Sim2RealLab, base of the work | `0c2f01f` (`main`, merge of #220) |
| branch | `task/depth-noise-coverage-band` |
| direction-A scratch branch (§9, never merged) | `scratch/direction-a-recost`, from the same base |
| Isaac Lab clone | `/home/zachoines/Documents/repos/IsaacLab` @ `v3.0.0-beta2.patch1` |

The canonical pair is Isaac Sim 6.0.1.0 with Isaac Lab `v3.0.0-beta2.patch1`.
The 2026-08-01 cost rejection §9 revisits was made on the retired pair, which is
the reason the re-cost was worth running at all.

Both training arms and the before/after golden states ran in separate
`git worktree`s, never by editing a working tree in place.

## Interpreter

`env_isaaclab3` on `gx10-d1d8`: python 3.12.13, numpy 2.3.1, torch 2.11.0+cu130,
onnxruntime 1.25.1, isaacsim 6.0.1.0.

`LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1` is set for every run, as
`env_setup.sh` does for this host. Every probe outside §9's training arms and
§8's contract gate is Kit-free and runs from the repository root in seconds.

The editable installs in this environment point at the **main checkout**, so
each run sets `PYTHONPATH` to the worktree's `source/strafer_lab` and
`source/strafer_shared` explicitly; `strafer_lab.__file__` was confirmed to
resolve inside the worktree before any measurement.

Probes import the production modules rather than reimplementing them:
`DepthNoiseModel` / `DepthNoiseModelCfg` for every injection, and
`strafer_shared.depth_texture` for every statistic. The composition-golden
attribution imports the test module's own `_canon` / `_contract` / `_hash` by
path, so a preimage is byte-identical in form to the one that produced the
stored golden — the probe is the 2026-09-13 deposit's, carried forward with a
`--repo` flag so it can be pointed at a worktree.

## Inputs read from other records

| input | from | what it is |
|---|---|---|
| `same-pose-probe/gym_obs.jsonl` | `goal-a-attribution-2026-08-22` | 30 clean observation-term frames at the defect pose — the anchor set |
| `arm3-obs-capture/node_obs.jsonl.gz` | `goal-a-attribution-2026-08-22` | the 1 799-frame bridge capture through the node pipeline |
| `ab/node_obs_rec0.json` | `depth-convention-fix-2026-09-13` | the pose-matched capture tick the residual statistic is read against |
| `texture/texture_structure.json` | `noise-texture-parity-2026-09-17` | the figures §1's statistic is accepted against |
| `probes/golden_attribution.py` | `depth-convention-fix-2026-09-13` | the by-field-name preimage diff |

`gym_obs.jsonl` carries `NaN` in dimensions 10–13 and 16–18 in every row. This
record reads only its depth tail (dimensions 19 onward), which is sound. A note
deposited beside it records what replaying the whole vector does, and why no one
should: `goal-a-attribution-2026-08-22/same-pose-probe-notes-2026-09-18/`.

## Artifact

`models/strafer_depth_subgoal_v2_998.onnx`, sha256
`855e1df7d0dac3be7229f933b59546b26f18959f97966b9e2f2f22e752bf5165`, is used in
exactly one place — the deposited note above, to show what the NaN prefix does.
**No trained artifact is a gate anywhere in this record**; the 2026-09-17 record
retired that use, and the coverage question here is answered with no policy in
the loop.

## The training arms

Both arms ran through `tools/kit_boot_watchdog.sh` with the GPU confirmed idle
(`nvidia-smi --query-compute-apps` empty) before each boot, one Kit-booting
actor at a time. Every boot succeeded on attempt 1; no watchdog relaunch fired
anywhere in this record.

`nvidia-smi` reports no GPU memory total on this host — GB10 is unified memory —
so peak memory is taken from two series sampled at 2 s: the per-process figure
`nvidia-smi --query-compute-apps=used_memory` does report, and system used
(`MemTotal − MemAvailable`). `MemTotal` is 124 543 MiB.
