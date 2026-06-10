# Repo topology

The end-to-end stack runs across two hosts on the same LAN, sharing
one git repository. Both hosts must agree on `ROS_DOMAIN_ID` and
`RMW_IMPLEMENTATION` to discover each other's ROS topics.

## Hosts

| Role | Hostname | IP | What runs here |
|------|----------|----|----------------|
| **DGX Spark** | `dgx-spark` | 192.168.50.196 | Isaac Sim, ROS 2 sim bridge, VLM service, LLM planner, RL training |
| **Jetson Orin Nano** | `jetson-desktop` | 192.168.50.24 | RTAB-Map, Nav2, executor (`strafer-executor`), goal-projection service, on-robot ROS bringup |

ROS 2 distro: **Humble** on both hosts.
DDS: `rmw_cyclonedds_cpp` (cross-host discovery; FastDDS shared-memory
default doesn't span machines).
ROS domain ID: **42** (any value works as long as both hosts agree).

## Repository

Single git remote. `main` is the working line; per-task branches
(one brief → one branch → one PR, see
[`branching-and-prs.md`](branching-and-prs.md)) merge into it.

| Host | Repo path |
|------|-----------|
| DGX | `~/Workspace/Sim2RealLab/` |
| Jetson | `~/workspaces/Sim2RealLab/` |

Verify from inside the repo with `git remote -v` + `git rev-parse --show-toplevel`.

## Python environments (DGX)

Three live environments partition the DGX stack — two conda envs and one
venv. Each is forced apart by a hard constraint, not convenience, so this
table names *what each is for and why it is separate*; the build recipe
for each lives in exactly one place (linked under **Recreate** below).

| Env | Kind | Python | For | Key contents |
|-----|------|--------|-----|--------------|
| `env_isaaclab3` | conda | 3.12 | Training, the sim bridge, **and all `strafer_lab` tests** (Kit + pure-Python) | Isaac Sim 6 + Isaac Lab develop, `pxr`, CUDA torch 2.10 (`+cu130`), lerobot 0.5.1, warp, onnx |
| `.venv_vlm` | venv | 3.12 | The VLM + LLM-planner services and their test suites | CUDA torch 2.11 (`+cu128`, with the NVRTC swap), transformers 5.x, `strafer_vlm`, `strafer_autonomy` |
| `env_infinigen` | conda | 3.11 | Infinigen procedural scene generation only | source-built `bpy==4.2.0`, Infinigen 1.19.x (editable, `--no-deps`) |

**Why three — both splits are forced, not incidental:**

- **`.venv_vlm` is kept by design — cadence isolation.** Isaac Sim's
  *compiled* torch is a hard floor: `env_isaaclab3` cannot move off torch
  2.10 without risking the sim. The VLM / LLM stack wants the fast-moving
  ceiling (newer `transformers` / torch per newer models — currently torch
  2.11 + transformers 5.x). One env can't satisfy both, so the services
  keep their own venv.
- **`env_infinigen` is pinned to 3.11** because Infinigen's deps don't all
  support 3.12 yet.

**Recreate** (each recipe is documented once — link, don't duplicate):

- `env_isaaclab3` — Isaac Sim 6 + Isaac Lab develop build:
  [`source/strafer_lab/README.md` → Install (DGX Spark)](../../../source/strafer_lab/README.md#install).
- `.venv_vlm` — venv + CUDA-torch + NVRTC-swap bootstrap:
  [`Readme.md` → Install (DGX Spark)](../../../Readme.md#dgx-spark-grace--blackwell-aarch64-ubuntu).
- `env_infinigen` — aarch64 `bpy` wheel + Infinigen: the `README.md` in the
  sibling `~/Workspace/blender-build/` directory (machine-specific, outside
  this repo).

The Jetson uses system Python 3.10 (Ubuntu 22.04 / ROS 2 Humble default)
+ a colcon workspace; it uses none of the DGX envs above.

`env_setup.sh` sources `.env` (operator-tuned) and exports
`STRAFER_ISAACLAB_PYTHON`, `STRAFER_INFINIGEN_PYTHON`, `ISAACLAB`,
`COLCON_WS`, `CONDA_ROOT`, `CONDA_ENV`. Always `source env_setup.sh`
before running any DGX-side command. The `$ISAACLAB` symbol pins to
`isaaclab.sh -p` in the bundled Isaac Sim install.

## Workspace layout

```
Sim2RealLab/
├── source/
│   ├── strafer_lab/         # Isaac Sim envs, RL policies, ROS 2 sim bridge
│   ├── strafer_autonomy/    # planner + executor + clients (Python-only)
│   ├── strafer_ros/         # all ROS 2 packages: bringup, slam, nav, perception, msgs
│   ├── strafer_vlm/         # VLM service (DGX-side HTTP)
│   └── strafer_shared/      # cross-host shared constants + utilities
├── docs/                    # design + task briefs + perf doc + cheatsheet
├── env_setup.sh             # source me first
├── .env / .env.example      # operator-tuned host paths
├── Makefile                 # targets: test-dgx, sim-bridge, sim-bridge-gui, serve-vlm, serve-planner
└── logs/                    # rsl_rl runs, scene-gen output, etc. (gitignored)
```

Ownership boundaries are spelled out in
[`ownership-boundaries.md`](ownership-boundaries.md).

## Build / install

- **Python-only** (`strafer_autonomy`, `strafer_vlm`, `strafer_lab`,
  `strafer_shared`): `pip install -e source/<pkg>` once. Edits take
  effect on next interpreter start.
- **ROS 2 packages** (`strafer_ros/*`): `colcon build` from the
  Jetson's `~/strafer_ws`. Re-source `install/setup.bash` after.
- Both Jetson and DGX assume `pip install -e .` is already done for
  the Python-only packages they consume.

## Key entry-point scripts

| Path | Purpose |
|------|---------|
| `source/strafer_lab/scripts/train_strafer_navigation.py` | RL training (rsl_rl PPO, optional DAPG/GAIL aux losses, optional video) |
| `source/strafer_lab/scripts/play_strafer_navigation.py` | Inference rollout from a checkpoint or an exported `.pt` (headed or headless+MP4) |
| `source/strafer_lab/scripts/export_policy.py` | Export an rsl_rl checkpoint to a deployable `.pt` / `.onnx` (+ JSON sidecar) consumed by `strafer_shared.policy_interface.load_policy()` |
| `source/strafer_lab/scripts/benchmark_policy.py` | Inference-latency stats on an exported artifact, with ONNX execution-provider preference for the Jetson TRT-EP path |
| `source/strafer_lab/scripts/test_strafer_env.py` | Env smoke tests (no policy; predefined motion patterns) |
| `source/strafer_lab/scripts/run_sim_in_the_loop.py` | The sim bridge — `--mode bridge` (default) drives env from `/cmd_vel`; `--mode harness` walks scene metadata as missions |
| `source/strafer_lab/scripts/collect_demos.py` | Gamepad-driven demo collection for DAPG/GAIL aux losses |
| `source/strafer_lab/scripts/postprocess_scene_usd.py` | Bake colliders + lights into Infinigen-exported USDC |
| `source/strafer_lab/scripts/prep_room_usds.py` | Run Infinigen scene generation + invoke postprocess |

## Cheatsheet

The operator-facing one-liners (training, smoke, demo collection,
fine-tunes, headed inference, full-stack autonomy bringup, sim
bridge + DDS bench) live in
[`docs/example_commands_cheatsheet.md`](../../example_commands_cheatsheet.md).
That file is the canonical place for "exactly which command to run."
This module describes the **shape** of the system; the cheatsheet
tells operators **how to invoke** it.
