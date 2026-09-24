# Provenance — marker checks 4 and 2, 2026-09-23

captured_date: 2026-09-23

## Host

| role | host | kernel | driver |
|---|---|---|---|
| sim | gx10-d1d8 (DGX Spark, GB10, aarch64) | 7.0.0-1019-nvidia | 580.173.02 |

## Trees and stacks

| check | Sim2RealLab tree | stack |
|---|---|---|
| 4 | `2575c65` (the #223 merge; code-identical to `fa4cb93`, the last `main` before #224 removed the marker geometry), in a worktree, plus one scratch commit applying `check4/check4_eval_flag.patch` | Isaac Sim 6.0.1.0 + Isaac Lab v3.0.0-beta2.patch1 (`ffff603ea`), conda `env_isaaclab3`, torch 2.11.0+cu130 |
| 2 | `66c01a1` (the first parent of the #218 merge, the last pre-flip tree), in a worktree, unmodified | Isaac Sim 6.0.0.0 + `IsaacLab-retired` (`ae41e2a`), conda `env_isaaclab3-retired`, torch 2.10.0+cu130 |

Both worktrees ran with `PYTHONPATH` naming their own `source/strafer_lab` and
`source/strafer_shared`: each environment's editable install points `strafer_lab` at the main
checkout, whose command terms carry no marker code since #224. Every run recorded where the
packages resolved before booting (`check4/logs/*.binding.txt`, `check2/run/preboot_resolution.txt`).
The retired pair was entered by activating its conda environment and calling its own
`isaaclab.sh`; it was not renamed, and nothing was installed into it. All 14 of its `isaaclab*`
editable finder shims name `IsaacLab-retired`.

## Checkpoints

| policy | file | sha256 | matches |
|---|---|---|---|
| v3 | `depth-subgoal-v3-retrain-2026-09-21/run_20260919_234233/model_999.pt` in Sim2RealLab-Artifacts | `725fc6bfcf32ee756f70a459e45f2a62f17e14289a40e458a349f1a086c21484` | the source-checkpoint row of `depth-subgoal-v3-retrain-2026-09-21` |
| v2 | `isaac-lab-upgrade-baseline-2026-08-14/upgrade_baseline_artifacts/checkpoints/run_20260727_171735/model_998.pt` in Sim2RealLab-Artifacts | `effaf5de095da1313309a50c94e9d49080ece03b9153caae8eb0c8a03c4a1f17` | §6 of `isaac-lab-upgrade-baseline-2026-08-14`, the copy G7 used |

Check 2 needs no checkpoint: it places the commands and measures in place.

## GPU

Every Kit boot went through `tools/kit_boot_watchdog.sh` with its default attempts, one at a time,
with `nvidia-smi --query-compute-apps` empty before it (`check4/logs/*.gpu_before.txt`,
`check2/run/gpu_before.txt`) and again after the last one (`check2/run/gpu_after.txt`). The
launches ran back to back from 22:31 to 23:46.
