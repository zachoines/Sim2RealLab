# Provenance — the command debug markers reach the robot's cameras, 2026-09-21

captured_date: 2026-09-21 → 2026-09-22

## Host

| role | host | kernel | driver |
|---|---|---|---|
| sim | gx10-d1d8 (DGX Spark, GB10, aarch64, 124 543 MiB unified) | 7.0.0-1019-nvidia | 580.173.02 |

## Trees

| work | tree |
|---|---|
| visibility test, on/off capture, in-place probe, marker play run, `--headless` controls, check 1 | the main checkout on a docs-only branch; every source file equal to `615fc14` (the #222 merge) |
| exposure survey on `main`, golden attribution "before", the `main`-code depth measurement | a detached worktree of `615fc14` |
| golden attribution "after", the gates, the post-fix recordings | branch `task/debug-markers-out-of-cameras` |
| mutation arms | detached worktrees of that branch, one change per arm, restored after each |

Kit-free work ran with `PYTHONPATH` naming the tree under test, since the editable install
points at the main checkout; `strafer_lab.__file__` was checked to resolve into the worktree.
Worktrees had the host's untracked scene corpus linked in, so the Infinigen variants construct.

## Interpreter

| item | value |
|---|---|
| conda env | `env_isaaclab3` |
| python | 3.12.13 |
| numpy / torch / onnxruntime | 2.3.1 / 2.11.0+cu130 / 1.25.1 |
| gymnasium / moviepy / opencv | 1.2.1 / 1.0.3 / 4.13.0 |
| Isaac Sim / Isaac Lab | 6.0.1.0 / v3.0.0-beta2.patch1 (`ffff603ea`); the previous pair read at `IsaacLab-retired` `ae41e2a` |
| LD_PRELOAD | `/lib/aarch64-linux-gnu/libgomp.so.1` |

Kit boots went through `tools/kit_boot_watchdog.sh`, one at a time, except one unwatched
`--headless` control (`play_markers/play_headless_flag_unwatched.log`) run to see past three
consecutive boot stalls under the watchdog. Kit suite verdicts are read from the JUnit XML.

## Inputs this record reads

| input | where |
|---|---|
| v3 exported policy | `models/strafer_depth_subgoal_v3_999.pt` / `.onnx`, deposited by `depth-subgoal-v3-retrain-2026-09-21` |
| v2 exported policy | `models/strafer_depth_subgoal_v2_998.onnx` (sha256 `855e1df7…5165`) |
| the 2026-08-22 bridge capture | Sim2RealLab-Artifacts `goal-a-attribution-2026-08-22/arm3-obs-capture/node_obs.jsonl.gz` |
| the noise tier and the golden walker | `sim_real_cfg.py` at each tree; `golden_attribution.py` carried forward from the `deploy-resolution-depth-2026-09-19` deposit, unchanged |
| the v2 training video frame | `logs/rsl_rl/strafer_navigation/run_20260726_221955/videos/rl-video-step-9000.mp4` at 3 s, on the host |
| the previous stack's training videos | `logs/rsl_rl/strafer_navigation/**/videos/` on the host, the 13 runs that recorded video before 2026-08-23, listed in `exposure/training_video_sample/sources.json` |
