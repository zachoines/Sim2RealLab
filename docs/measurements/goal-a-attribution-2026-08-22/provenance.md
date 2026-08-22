# Provenance — depth-subgoal v2 off-goal attribution, 2026-08-22

captured_date: 2026-08-22

## Hosts

| role | host | kernel |
|---|---|---|
| robot, deploy stack, observation capture | `strafer-nx` | 5.15.148-tegra |
| sim, replay, probe, bisection, verification | `gx10-d1d8` | 6.11.0-1014-nvidia |

The two are joined by the direct cable (`docs/tasks/context/repo-topology.md`).
Every figure in [`README.md`](README.md) was produced on `gx10-d1d8` except the
observation capture itself, which the robot wrote.

## Trees

| item | value |
|---|---|
| Sim2RealLab, analysis | `4f144bad1d3bcd00604dc7cc213804422818a6a9` (`main`, merge of #208) |
| Sim2RealLab, sim host at probe time | `78bc2f9f488b04c8fc3fc195445778bb1cec8d5e` |
| deploy image revision under test | `e7ea7bd62f85` (both `strafer-cpu:humble` and `strafer-gpu:humble`) |

## Interpreter

`env_isaaclab3` on `gx10-d1d8`: python 3.12.13, numpy 2.3.1, torch 2.10.0+cu130,
onnxruntime 1.25.1, isaaclab 4.6.12. Every inference in this record ran on
`CPUExecutionProvider`.

The corruption statistic in [`README.md`](README.md) §3 re-derives on any host
with numpy — [`verify/recheck_corruption_stats.py`](verify/recheck_corruption_stats.py)
reads only files in this directory. The replay re-checks need onnxruntime and
the two machine-local inputs below.

## Artifacts

| artifact | sha256 | source checkpoint |
|---|---|---|
| `models/strafer_depth_subgoal_v2_998.onnx` | `855e1df7d0dac3be7229f933b59546b26f18959f97966b9e2f2f22e752bf5165` | `run_20260727_171735/model_998.pt` |
| `models/strafer_depth_subgoal_v1.onnx` | `4c70e1257cadfff12a28fc22b6a47a4c75918cde56f341abf1f0491f53b6d6ef` | `run_20260708_005923/model_500.pt` |

Both declare `policy_variant: DEPTH_SUBGOAL`, `obs_dim 3619`, `is_recurrent: true`,
`onnx_opset 18`, `env_id Isaac-Strafer-Nav-RLDepth-Subgoal-Real-Play-v0`. Export
stamps and tree commits, from the sidecars beside each artifact: v1
2026-07-08T21:16:43Z at `eeacccc1598f5aafd0cf4b3ab5abd2d6d024804c`; v2
2026-07-28T18:37:39Z at `69014c6f0621f62e9ea672e4c2d666d143fd64b0`.

The on-robot run replayed here bound `TensorrtExecutionProvider` first and used
two persisted, unrebuilt engines: `…11017259586141949358_0_0_sm87.engine`
sha256 `07682894947d36b956867a3bfc55026c3a12ad58c33946d3e8cf303b6e22ec78`
(dated Jul 22) and `…619207121200783073_0_0_sm87.engine` sha256
`320dff7c9084530e58fe11414c788a34cf18b6bc190d0c3412d784ad6addaaa1` (Jul 28).

## Machine-local inputs, not in git

The node-assembled observation capture is 131 MB and stays on the sim host at
`gx10-d1d8:~/arm3_obs_capture_20260822/`:

| file | bytes | sha256 |
|---|---|---:|
| `node_obs.jsonl` | 131 149 352 | `e480766eb33bc38728c7d24ab99f5428fe574b7063c6dc32faa2c97a58641224` |
| `MANIFEST.json` | 3 843 | `360df87b5d4ed2e951df3f38555519b96e4cdf2e45df8a2116c34a493d8bddbe` |
| `generator_and_node.log` | 67 335 | `e5f69b8f96ce7a140a0f908a14c513c7361155943f9615ab8fd08bc8b9801c90` |
| `mission_record.json.log` | 593 722 | `b9e2934fa2ce711029cc21b6c9ba6efdcb746b09311ca32f27145ef53f698341` |

`node_obs.jsonl` holds 1 799 records spanning `t_sim` 360.833 → 420.767 in the
`PARITY_SCHEMA.md` shape. Field order for `DEPTH_SUBGOAL`, with the scale each
carries: `0-2` imu_accel, `3-5` imu_gyro, `6-9` encoder velocities in ticks,
`10-11` subgoal relative (1/10), `12` subgoal distance (1/10), `13` subgoal
heading (1/π), `14-15` body velocity xy (1/2), `16-18` last action (1.0),
`19-3618` depth (1/6).

The capture came from one mission on
`Isaac-Strafer-Nav-Capture-Bridge-ProcRoom-Enriched-v0`, environment seed 42,
Kit log `kit_20260822_093054.log`, SLAM key `enrich_isolate1`, anchoring
`mission`, `mission_timeout_s` 60.0, fixed start heading 130.0°: start
`(-0.499, -0.451, 2.22)` in `map` at 3.09 m from goal `(-2.0, 2.25)`, goal
bearing −8.1° relative to heading; final `(-0.464, -0.524, 1.351)` at 3.171 m,
`ABORTED` at 60.0 s sim, net −0.081 m.

Two training-run stdout logs are likewise machine-local, on `gx10-d1d8` under
`~/Workspace/Sim2RealLab/logs/rsl_rl/strafer_navigation/`:
`depth_subgoal_vfov8045_stdout.log` (v1), `depth_subgoal_v2_stdout.log` and
`depth_subgoal_v2b_stdout.log` (v2's two legs). They are the only surviving
record of which environment each run trained under — see [`README.md`](README.md) §6.

## What was edited in the copied artifacts

The scripts and outputs in this directory are reproduced as they ran, with one
substitution: the transient directory they were written into has been rewritten
to this record's path, so the scripts resolve their inputs from a repository
checkout. No numeric field, log line, or array was altered. The `.npy` and
`.jsonl` payloads are byte-identical to what the runs produced —
[`MANIFEST.sha256`](MANIFEST.sha256) digests what is here.
