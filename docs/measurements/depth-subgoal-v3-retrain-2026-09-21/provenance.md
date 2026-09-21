# Provenance — the depth-subgoal v3 retrain, 2026-09-21

captured_date: 2026-09-21

## Host

| role | host | kernel | driver |
|---|---|---|---|
| sim / training | gx10-d1d8 (DGX Spark, GB10, aarch64, 124 543 MiB unified) | 7.0.0-1019-nvidia | 580.173.02 |

`nvidia-smi` reports no GPU memory total on this part, so memory is sampled from
`--query-compute-apps=used_memory` and `/proc/meminfo`.

## Tree

Every boot and every CPU computation ran on Sim2RealLab `main` at
`615fc14904967ebe377e10ea3a51b863c5a5a245` (the #222 merge), clean apart from the
untracked `models/` directory the exports write to. Nothing in the tree changed for this
record. #219 (`9c4d674`), #220 (`0c2f01f`) and #221 (`b6975f5`) are ancestors of that
commit.

## Interpreter

| item | value |
|---|---|
| conda env | `env_isaaclab3` |
| python | 3.12.13 |
| numpy | 2.3.1 |
| torch | 2.11.0+cu130 |
| onnxruntime | 1.25.1 |
| gymnasium | 1.2.1 |
| rsl-rl-lib | 5.4.2 |
| tensorboard | 2.21.0 |
| moviepy | 1.0.3 |
| Isaac Sim | 6.0.1.0 |
| Isaac Lab | v3.0.0-beta2.patch1 (`ffff603ea`) |
| LD_PRELOAD | `/lib/aarch64-linux-gnu/libgomp.so.1` |

`isaaclab.sh` takes its interpreter from the active conda environment, so each boot
script activates `env_isaaclab3` and sources `env_setup.sh` before calling it; a detached
shell inherits no active environment.

## Boots

One Kit-booting process at a time, each through `tools/kit_boot_watchdog.sh`, with
`nvidia-smi --query-compute-apps` confirmed empty before each boot and after the last.

| boot | command | attempts | result |
|---|---|---:|---|
| training, first launch | `train/launch_cmd.sh` before the environment activation was added | 1 | exit 1 after 1 s, before Kit: `isaaclab.sh` ran the base conda python and failed on `ModuleNotFoundError: No module named 'lazy_loader'`, and the watchdog passed the exit through without relaunching — `[kit-boot-watchdog v3-retrain] attempt 1: exit=1 wall=1s`. That line is kept here; the launch's log file was overwritten by the second launch. |
| training | `train/launch_cmd.sh` | 1 | exit 0, wall 95 324 s, 2026-09-20T04:42:09Z → 2026-09-21T07:10Z |
| export | `export/export_cmd.sh` | 1 | exit 0, 17 s |
| play smoke | `smoke/smoke_cmd.sh` | 1 | exit 0, 19 s |
| video | `video/video_cmd.sh` | 1 | exit 0, 25 s |

The memory series ran beside the training boot as
`probes/sample_memory.sh 'train_strafer_navigation.py' train/memory.txt`.

## CPU work

The three tables ran without Kit, on the CPU, with ONNX Runtime's CPU provider:
`tables/samepose/`, `tables/texture/` and `tables/curves/`. Each directory's `.py` files
are the scripts that produced it; each `validate/` directory re-runs an earlier deposit's
probe unmodified.

## Inputs this record reads from earlier deposits

In the evidence repository, at the commit this record's deposit sits on:

```
821fe832e37952766088a60d6bb77fd8f4a21c0cdcb83195b9db4d2067cb0cbf  goal-a-attribution-2026-08-22/record-files/same-pose-probe/gym_obs.jsonl
073cbb8593609f870eba0cc2f5e183e62f147f701850ef4f49553c4c6a987c7b  depth-convention-fix-2026-09-13/record-files/ab/node_obs_rec0.json
1accaa0fe5ce2cd5cb224a9e9b3fffc380f3533aa756679690423634bf96430f  deploy-resolution-depth-2026-09-19/record-files/drift2/reduction_drift_v2_frames.json
11e7888982b3f6b0b29f8233bbb51dc8d38099921e642e88364bbe254ab9278d  noise-texture-parity-2026-09-17/record-files/probes/candidate_sweep.py
e3103106608a6e4bf2c7cb4c3d4a5ed96228ab2925b031ced8221fd9a5fe9c4d  depth-convention-fix-2026-09-13/record-files/probes/convention_ab.py
4ee96c8cc6029be47f44d3d2bf12ae694e945fb5a1cb90b091cf2e337f712dd0  depth-noise-coverage-2026-09-18/record-files/probes/coverage_curve.py
8fd3da8a37ccf8e357e16f20a74da1a54b4e746fa11c5ce8c555385c568e7ed5  deploy-resolution-depth-2026-09-19/record-files/probes/exported_policy_rollout.py
0d776455914382b1de99144fbc43f42bb3f596df958c03d3e1b6354ba7576ca6  deploy-resolution-depth-2026-09-19/record-files/probes/sample_memory.sh
```

## Inputs held on the host

Not carried by either repository.

```
855e1df7d0dac3be7229f933b59546b26f18959f97966b9e2f2f22e752bf5165  models/strafer_depth_subgoal_v2_998.onnx
cad44d68c25c2ca2fae1f15a11d81ebfa3c572a117da5d519d2ea3fce353fcf9  logs/rsl_rl/strafer_navigation/run_20260726_221955/events.out.tfevents.1785122398.gx10-d1d8.958158.0
625c8e5672000be6590eb1da8a702dede4b0416634dd92c1807056395ed4edd4  logs/rsl_rl/strafer_navigation/run_20260727_171735/events.out.tfevents.1785190658.gx10-d1d8.980608.0
```

The first is v2@998, the artifact the earlier depth records score; the other two are
v2's two training legs, 0–499 and 500–998, read for the training curves.
