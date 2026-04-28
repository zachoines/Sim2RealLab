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

Single git remote, `phase_15-isaaclab3` branch is the active working
line (Isaac Lab 3.0 / Isaac Sim 6 migration; the project hasn't
returned to a stable Isaac Lab release yet).

| Host | Repo path |
|------|-----------|
| DGX | `~/Workspace/Sim2RealLab/` |
| Jetson | `~/workspaces/Sim2RealLab/` |

Verify from inside the repo with `git remote -v` + `git rev-parse --show-toplevel`.

## Conda environments (DGX)

| Env name | Python | Purpose |
|----------|--------|---------|
| `env_isaaclab3` | 3.12 | Isaac Sim 6 + Isaac Lab develop, the main DGX env. Used for training, bridge, smoke tests, demo collection. |
| `env_infinigen` | 3.11 | Infinigen scene generation only. Pinned to 3.11 because Infinigen's deps don't all support 3.12 yet. |

The Jetson uses system Python 3.12 + a colcon workspace.

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
├── Scripts/                 # operator-facing entry points (training, play, smoke tests)
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
| `Scripts/train_strafer_navigation.py` | RL training (rsl_rl PPO, optional DAPG/GAIL aux losses, optional video) |
| `Scripts/play_strafer_navigation.py` | Inference rollout from a checkpoint (headed or headless+MP4) |
| `Scripts/test_strafer_env.py` | Env smoke tests (no policy; predefined motion patterns) |
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
