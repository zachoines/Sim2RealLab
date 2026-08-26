# Repo topology

The end-to-end stack runs across two hosts on the same LAN, sharing
one git repository. Both hosts must agree on `ROS_DOMAIN_ID` and
`RMW_IMPLEMENTATION` to discover each other's ROS topics.

## Hosts

| Role | Hostname | IP | What runs here |
|------|----------|----|----------------|
| **DGX Spark** | `gx10-d1d8` | 192.168.50.196 | Isaac Sim, ROS 2 sim bridge, VLM service, LLM planner, RL training |
| **Jetson Orin Nano** | `strafer-nx` | 192.168.50.161 | RTAB-Map, Nav2, executor (`strafer-executor`), goal-projection service, on-robot ROS bringup |

ROS 2 distro: **Humble** on both hosts.
DDS: `rmw_cyclonedds_cpp` (cross-host discovery; FastDDS shared-memory
default doesn't span machines).
ROS domain ID: **42** (any value works as long as both hosts agree).

**The sim-bridge camera streams cross a bandwidth-constrained link between the
two hosts, and that link is a design constraint rather than a detail.** DDS
data on it is carried **unicast per subscribing process**, so a camera topic's
wire cost multiplies with the number of remote subscribers instead of being
shared — adding a remote subscriber to a camera topic spends real budget. The
measured budget and the current transport topology are owned by
[`sim-bridge-link-transport-capacity`](../active/reliability/sim-bridge-link-transport-capacity.md);
the dated measurement record is
[`depth-receiver-host-capacity`](../completed/depth-receiver-host-capacity.md).
Consult the brief for the figure before sizing anything against it — the value
moves, which is why it does not live here.

## Repository

Single git remote. `main` is the working line; per-task branches
(one brief → one branch → one PR, see
[`branching-and-prs.md`](branching-and-prs.md)) merge into it.

| Host | Repo path |
|------|-----------|
| DGX | `~/Workspace/Sim2RealLab/` |
| Jetson | `~/Sim2RealLab/` |

Verify from inside the repo with `git remote -v` + `git rev-parse --show-toplevel`.

## Python environments (DGX)

Four environments live on the DGX — three conda envs and one venv. Each split
is forced by a hard constraint, not convenience, so this table names *what each
is for and why it is separate*; the build recipe for each lives in exactly one
place (linked under **Recreate** below). Two of them are Isaac Lab environments:
`.env` and the Makefile defaults name which one the tooling uses, and
`CONDA_ENV` in `.env` is the single place that answers "which".

| Env | Kind | Python | For | Key contents |
|-----|------|--------|-----|--------------|
| `env_isaaclab3` | conda | 3.12 | Training, the sim bridge, **and all `strafer_lab` tests** (Kit + pure-Python) | Isaac Sim 6.0.0.0 + Isaac Lab develop, `pxr`, CUDA torch 2.10 (`+cu130`), lerobot 0.5.1, warp, onnx |
| `env_isaaclab3beta2` | conda | 3.12 | The same, on the tagged Isaac Lab pin | Isaac Sim 6.0.1.0 + Isaac Lab `v3.0.0-beta2.patch1`, CUDA torch 2.11 (`+cu130`), torchcodec 0.16.0, rsl-rl-lib 5.4.2, lerobot 0.5.1 |
| `.venv_vlm` | venv | 3.12 | The VLM + LLM-planner services and their test suites | CUDA torch 2.11 (`+cu128`, with the NVRTC swap), transformers 5.x, `strafer_vlm`, `strafer_autonomy` |
| `env_infinigen` | conda | 3.11 | Infinigen procedural scene generation only | source-built `bpy==4.2.0`, Infinigen 1.19.x (editable, `--no-deps`) |

**Why separate — each split is forced, not incidental:**

- **`.venv_vlm` is kept by design — CUDA-build isolation.** Isaac Sim is
  compiled against a specific torch build, so the Isaac Lab env's build tag is
  not free to move; the VLM / LLM stack tracks whatever build its
  `transformers` line wants. The two run the same torch minor and differ in the
  CUDA build — `+cu130` against `+cu128` with the NVRTC swap — so the split
  rests on the build tag and on release cadence, not on a version floor.
- **`env_infinigen` is pinned to 3.11** because Infinigen's deps don't all
  support 3.12 yet.

**Recreate** (each recipe is documented once — link, don't duplicate):

- `env_isaaclab3beta2` — Isaac Sim 6 + tagged Isaac Lab build:
  [`source/strafer_lab/README.md` → Install (DGX Spark)](../../../source/strafer_lab/README.md#install).
  **This is the recipe a host builds from scratch**, and the only Isaac Lab
  recipe maintained here. `env_isaaclab3` predates it, is kept as a rollback
  artifact, is never written to, and is not rebuilt.
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
| `source/strafer_lab/scripts/train_strafer_navigation.py` | RL training (rsl_rl PPO, optional video) |
| `source/strafer_lab/scripts/play_strafer_navigation.py` | Inference rollout from a checkpoint or an exported `.pt` (headed or headless+MP4) |
| `source/strafer_lab/scripts/export_policy.py` | Export an rsl_rl checkpoint to a deployable `.pt` / `.onnx` (+ JSON sidecar) consumed by `strafer_shared.policy_interface.load_policy()` |
| `source/strafer_lab/scripts/benchmark_policy.py` | Inference-latency stats on an exported artifact, with ONNX execution-provider preference for the Jetson TRT-EP path |
| `source/strafer_lab/scripts/test_strafer_env.py` | Env smoke tests (no policy; predefined motion patterns) |
| `source/strafer_lab/scripts/run_sim_in_the_loop.py` | The sim bridge — `--mode bridge` (default) drives env from `/cmd_vel`; `--mode harness` walks scene-metadata targets or a `mission_queue.yaml` through the Jetson stack and records a LeRobot v3 dataset (dispatched via `capture.py --driver bridge`) |
| `source/strafer_lab/scripts/collect_demos.py` | Gamepad teleop recorder — writes per-episode obs / action / reward arrays to HDF5 |
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
