# Provenance — v3's time to goal in its training env, 2026-10-11

captured_date: 2026-10-11

## Host

| role | host | kernel | driver |
|---|---|---|---|
| sim | gx10-d1d8 (DGX Spark, GB10, aarch64) | 7.0.0-1019-nvidia | 580.173.02 |

The kernel and driver are as each launch's Kit log prints them (`scored/logs/*.log`).

## Tree and stack

| Sim2RealLab tree | stack |
|---|---|
| `9f7d207` (`main`, the #238 merge), in a detached worktree, no tree change | Isaac Sim 6.0.1.0 + Isaac Lab v3.0.0-beta2.patch1 (`ffff603ea`), conda `env_isaaclab3`, torch 2.11.0+cu130, rsl-rl-lib 5.4.2 |

- The worktree ran with `PYTHONPATH` naming its own `source/strafer_lab` and
  `source/strafer_shared`, because the environment's editable install points `strafer_lab` at
  another checkout.
- Every launch recorded where the packages resolved before booting (`scored/logs/*.binding.txt`,
  `plumbing/logs/*.binding.txt`). All name the worktree for `strafer_lab` and `strafer_shared`,
  and the Isaac Lab clone at `v3.0.0-beta2.patch1`.
- `run_ttg.sh` refuses to launch unless the worktree is at `9f7d207` with no tracked changes, and
  the eval's sha256 matches.
- Each `meta.json` records the tree commit, the resolved package paths and the versions again from
  inside the run.

The eval is `time_to_goal_eval.py` in the deposit, sha256
`877d158bcea76324ddb251e7d1ea2dae21ce631eafeeb1dfcca827c27b68d5c5`. The launches ran it from a
staging copy of the deposit's files. Every binding file records that sha256, plumbing and scored
alike.

Scoring ran under `env_isaaclab3`'s Python 3.12 (its `lib/python3.12` paths in
`scored/logs/*.binding.txt`) with numpy 2.3.1 (`summary/summary.txt`, second line).

## Checkpoint

| file | sha256 | matches |
|---|---|---|
| `depth-subgoal-v3-retrain-2026-09-21/run_20260919_234233/model_999.pt` in Sim2RealLab-Artifacts | `725fc6bfcf32ee756f70a459e45f2a62f17e14289a40e458a349f1a086c21484` | the source checkpoint of `strafer_depth_subgoal_v3_999.onnx` (`c866bfd5…`), `depth-subgoal-v3-retrain-2026-09-21` |

The eval checks the checkpoint's sha256 before Kit boots and prints the match
(`scored/logs/watchdog_*.log*`). Each `meta.json` records it again.

The deploy-cap launch loaded the inference node's `obs_pipeline.py` from the worktree and checked
that its `l1_clamp_velocity` is `strafer_shared`'s `l1_clamp_twist`. It hashed the worktree's
`inference_node.py` without importing it. That sha256,
`44a2950a03419ed4d7023476656a871eef23bdf69d093888774e682d69cb35d8`, is recorded in
`scored/eval/deploy-cap/meta.json` and equals `9f7d207`'s
`source/strafer_ros/strafer_inference/strafer_inference/inference_node.py`.

## GPU

- Every Kit boot went through `tools/kit_boot_watchdog.sh` with its default attempts, one launch at
  a time.
- `nvidia-smi --query-compute-apps` was empty before each launch (`scored/logs/*.gpu_before.txt`,
  `plumbing/logs/*.gpu_before.txt`, all empty). `scored/chain.log` also reads `gpu_apps=0` at each
  scored start.

## Timeline (CDT, 2026-10-11)

| step | start | end | boot | eval wall / rollout | vector steps |
|---|---|---|---|---|---|
| plumbing, `plumb-baseline` (seed 7, 8 envs) | 19:08:20 | 19:09:00 | 1st attempt | 38 s / 25 s | 178 |
| plumbing, `plumb-open-dwell` | 19:09:10 | 19:11:51 | 1st attempt | 158 s / 146 s | 1 200 |
| plumbing, `plumb-deploy-cap` | 19:11:51 | 19:12:40 | 1st attempt | 46 s / 33 s | 258 |
| pre-registration written | 19:13:08 | | | | |
| pre-registration committed, `675c0c5` | 19:13:20 | | | | |
| run 1 `baseline` | 19:13:33 | 19:26:07 | 3rd attempt; attempts 1–2 stalled at 61 s | 626 s / 602 s | 1 333 |
| run 2 `open-dwell` | 19:26:07 | 19:44:28 | 1st attempt | 1 097 s / 1 073 s | 2 400 |
| run 3 `deploy-cap` | 19:44:28 | 20:08:50 | 3rd attempt; attempts 1–2 stalled at 61 s | 1 334 s / 1 310 s | 2 923 |

- **Sources.** Plumbing times are from `plumbing/logs/sampling.log` and scored times from
  `scored/chain.log`; `scored/logs/sampling.log` agrees within 1 s (its start stamps are 1 s
  later, its end stamps equal). Boot attempts are from the watchdog lines in each console log.
- **Wall times.** The eval's own wall time is `wall_s` in each `meta.json`, and its rollout time is
  `readback.rollout_wall_s`. The watchdog's wall for the winning attempt is 630, 1 100 and 1 338 s.
- **Stalls.** Each stalled attempt reached about 80 MB resident with no CPU or output for 60 s. It
  left only the launcher's first lines (`scored/logs/watchdog_*.log.attempt1`, `.attempt2`).
