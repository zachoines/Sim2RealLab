# Provenance — the real D555's field of view in v3's loop, 2026-10-03

captured_date: 2026-10-03

## Host

| role | host | kernel | driver |
|---|---|---|---|
| sim | gx10-d1d8 (DGX Spark, GB10, aarch64) | 7.0.0-1019-nvidia | 580.173.02 |

## Tree and stack

| Sim2RealLab tree | stack |
|---|---|
| `ed0d5af` (the #233 merge), in a worktree, plus one scratch commit applying `vfov_eval_flag.patch` | Isaac Sim 6.0.1.0 + Isaac Lab v3.0.0-beta2.patch1 (`ffff603ea`), conda `env_isaaclab3`, torch 2.11.0+cu130 |

The worktree ran with `PYTHONPATH` naming its own `source/strafer_lab` and
`source/strafer_shared`, because the environment's editable install points `strafer_lab` at
another checkout. Every launch recorded where the packages resolved before booting
(`logs/*.binding.txt`); all name the worktree.

## Checkpoint and artifact

| file | sha256 | matches |
|---|---|---|
| `depth-subgoal-v3-retrain-2026-09-21/run_20260919_234233/model_999.pt` in Sim2RealLab-Artifacts | `725fc6bfcf32ee756f70a459e45f2a62f17e14289a40e458a349f1a086c21484` | the source-checkpoint row of `depth-subgoal-v3-retrain-2026-09-21` |
| `depth-subgoal-v3-retrain-2026-09-21/record-files/export/strafer_depth_subgoal_v3_999.onnx` in Sim2RealLab-Artifacts | `c866bfd54ec1a8352159e33d7875d41e3f07a442ff8301ba3700867932e2eb91` | the exported-artifact row of the same record |

The eval loads the checkpoint. `onnx_parity.py` replays every env's first-episode observations
through the ONNX artifact and compares its actions with the ones the eval recorded.

## GPU

Every Kit boot went through `tools/kit_boot_watchdog.sh` with its default attempts, one at a time,
with `nvidia-smi --query-compute-apps` empty before it (`logs/*.gpu_before.txt`,
`render_probe/*.gpu_before.txt`, `plumbing/logs/*.gpu_before.txt`).

The rendered-intrinsics probe and the two plumbing launches ran from 18:27 to 18:50 CDT on
2026-10-03. The scored launches and the replicates ran back to back from 18:52 to 19:59.
