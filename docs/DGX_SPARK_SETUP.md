# DGX SPARK — Isaac Lab Setup for `strafer_lab`

Guide for bringing up the Isaac Lab RL training stack on an NVIDIA DGX SPARK (aarch64).

## System Snapshot

| Item | Value | Status |
|---|---|---|
| OS | Ubuntu 24.04.3 LTS (Noble) | Supported by Isaac Sim 5.1.0 |
| Architecture | aarch64 (ARM64) | Supported (DGX Spark explicitly listed) |
| GPU | NVIDIA GB10 (Blackwell) | Supported (RT cores present) |
| Driver | 580.82.09 | Close — 580.95.05 recommended for Spark |
| CUDA | 13.0 | Required for aarch64 |
| RAM | 128 GB | Exceeds 64 GB recommendation |
| Disk | ~800 GB free | Plenty |
| GLIBC | 2.39 | Exceeds 2.35 minimum |
| Python | 3.12.3 (system) | Need 3.11 for Isaac Sim 5.x |
| Git branch | `phase_14` | Correct |

## Blockers and Unknowns

1. **Driver version mismatch** — NVIDIA docs recommend `580.95.05` for Spark; current is `580.82.09`. May work but could cause subtle PhysX/rendering issues. Upgrade if GPU faults appear.

2. **Python 3.11 not installed** — Isaac Sim 5.x hard-requires Python 3.11. System has 3.12. Install via Miniconda.

3. **PyTorch aarch64 + cu13** — Isaac Lab pip install page has a separate "Linux (aarch64)" tab. The exact wheel URL differs from x86_64.

4. **Vulkan tools not installed** — `vulkaninfo` missing. Install `vulkan-tools` for diagnostics (not required for headless).

5. **`nvidia-smi` reports VRAM as `[N/A]`** — GB10 uses unified memory with the CPU. VRAM tracking may behave differently; watch for OOM at high `--num_envs`.

6. **Known DGX Spark Isaac Lab limitations** — No SkillGen, no OpenXR, no JAX GPU, no Livestream. None affect `strafer_lab`.

7. **Repo is Windows-centric** — All docs use `.\isaaclab.bat` and Windows paths. Linux equivalent is `./isaaclab.sh`.

## Setup Steps

### Step 0: Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
# restart shell or source ~/.bashrc
```

### Step 1: Create Python 3.11 environment

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
pip install --upgrade pip
```

### Step 2: Install Isaac Sim 5.1.0

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### Step 3: Install PyTorch for aarch64 + CUDA 13

```bash
# Use the aarch64 cu13 build (check Isaac Lab pip install page "Linux (aarch64)" tab).
# May resolve automatically via Isaac Sim deps. If not:
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Step 4: Verify Isaac Sim

```bash
isaacsim  # first run pulls extensions (~10 min), accept EULA with "Yes"
# Ctrl+C to exit after the window appears
```

### Step 5: Clone and install Isaac Lab

```bash
cd ~/Documents/repos
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab

# Install Linux build deps (already present on this machine)
sudo apt install cmake build-essential -y

# Install Isaac Lab + rsl_rl framework
./isaaclab.sh --install rsl_rl
```

### Step 6: Verify Isaac Lab

```bash
cd ~/Documents/repos/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
# Should show a black viewport window. Ctrl+C to exit.
```

### Step 7: Install `strafer_shared` and `strafer_lab`

```bash
cd ~/Documents/repos/Sim2RealLab
pip install -e source/strafer_shared
pip install -e source/strafer_lab
```

### Step 8: List registered Strafer envs

```bash
python -c "import strafer_lab; import gymnasium as gym; print([e for e in gym.envs.registry.keys() if 'Strafer' in e])"
```

Expected: 30 environment IDs (15 train + 15 play).

## Smoke Test

### First: NoCam headless (fastest, safest)

```bash
cd ~/Documents/repos/IsaacLab
./isaaclab.sh -p ../Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-NoCam-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --headless
```

Validates: Isaac Sim physics, Isaac Lab RL pipeline, strafer_lab env registration, strafer_shared kinematics, RSL-RL training loop, GB10 GPU compute.

### Second: Scale up NoCam

```bash
./isaaclab.sh -p ../Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-NoCam-v0 \
  --num_envs 512 \
  --max_iterations 100 \
  --headless
```

### Third: Test suite

```bash
cd ~/Documents/repos/IsaacLab
./isaaclab.sh -p ../Sim2RealLab/source/strafer_lab/run_tests.py rewards observations terminations
```

### Fourth: Depth variant

```bash
./isaaclab.sh -p ../Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-Depth-v0 \
  --num_envs 16 \
  --max_iterations 10 \
  --headless
```

### Fifth: ProcRoom NoCam

```bash
./isaaclab.sh -p ../Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --headless
```

## Windows-to-Linux Translation Table

| Windows | Linux |
|---|---|
| `.\isaaclab.bat` | `./isaaclab.sh` |
| `C:\Worspace\IsaacLab` | `~/Documents/repos/IsaacLab` |
| `C:\Worspace\venv_isaac\Scripts\Activate.ps1` | `conda activate env_isaaclab` |
| `..\Scripts\train_strafer_navigation.py` | `../Sim2RealLab/Scripts/train_strafer_navigation.py` |
| `..\source\strafer_lab\run_tests.py` | `../Sim2RealLab/source/strafer_lab/run_tests.py` |
| Backslash paths `\` | Forward slash `/` |

## Training Commands (Linux)

Every training session on DGX SPARK requires this preamble:

```bash
cd ~/Documents/repos/IsaacLab
conda activate env_isaaclab3
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
```

### NoCam Realistic (recommended first)

```bash
./isaaclab.sh -p ../Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 512
```

### Depth Realistic

```bash
./isaaclab.sh -p ../Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-Depth-v0 --num_envs 32
```

### Headless large-scale NoCam

```bash
./isaaclab.sh -p ../Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 4096 --headless
```

### ProcRoom Depth with video + LR schedule

```bash
./isaaclab.sh -p ../Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0 \
  --num_envs 64 \
  --num_steps 48 \
  --headless \
  --video --video_length 800 --video_interval 1000 \
  --depth_encoder cnn \
  --lr_schedule linear --lr_min 3e-5
```

### TensorBoard monitoring (separate terminal)

```bash
conda activate env_isaaclab
tensorboard --logdir /home/zachoines/Documents/repos/IsaacLab/logs/rsl_rl/strafer_navigation
```

## System Resource Monitoring

Monitor GPU, CPU, and memory usage in a separate terminal during training:

```bash
watch -n 2 'echo "=== GPU ===" && nvidia-smi && echo && echo "=== CPU + MEM ===" && top -bn1 | head -5 && echo && free -h'
```

> **Note:** On DGX SPARK (GB10), `nvidia-smi` reports VRAM as `[N/A]` because
> the GPU uses unified memory shared with the CPU. Watch `free -h` for overall
> memory pressure instead.

## Issues Found and Fixed

### 1. `LD_PRELOAD` required on aarch64

Isaac Lab's installer prints a warning: `libgomp.so.1` must be preloaded.
Always set this before running any Isaac Lab command:

```bash
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
```

### 2. RSL-RL v5.0.1 config API change (`KeyError: 'class_name'`)

`rsl_rl` v5.0.1 expects `cfg["actor"]["class_name"]` / `cfg["critic"]["class_name"]`
instead of the old `cfg["policy"]["class_name"]`. Isaac Lab provides
`handle_deprecated_rsl_rl_cfg()` to migrate configs, but
`Scripts/train_strafer_navigation.py` did not call it.

**Fix applied:** Added `handle_deprecated_rsl_rl_cfg(agent_cfg, _rsl_rl_version)`
before `agent_cfg.to_dict()` in the training script.

### 3. PyTorch CUDA capability warning

PyTorch warns about GB10 (CUDA capability 12.1) exceeding its tested max (12.0).
This is a non-fatal warning and training proceeds normally.

### 4. Omniverse EULA acceptance

First run of Isaac Sim requires accepting the EULA (`Yes` at prompt). This was
done interactively and is now cached.

## Smoke Test Results

- **Environment:** `Isaac-Strafer-Nav-Real-NoCam-v0`
- **Num envs:** 64
- **Iterations:** 10
- **Mode:** headless
- **Iteration time:** ~7.3s per iteration
- **Total time:** ~1 min 13 sec
- **Result:** SUCCESS — all reward, termination, curriculum, and metric channels active
- **Checkpoint saved to:** `logs/rsl_rl/strafer_navigation/run_20260401_223559`

---

## Isaac Lab 3.0 (Develop Branch) — Build from Source

The `feature/isaaclab-3.0-migration` branch targets Isaac Lab's `develop` branch
(version 4.5.24, pre-3.0 release). This requires Isaac Sim 6.0.0 and Python 3.12.
Below are the complete setup steps.

### System Requirements (additional)

```
libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxkbcommon-dev libgl1-mesa-dev
```

These are required to build `imgui-bundle` (a transitive Isaac Sim 6.0 dependency)
from source on aarch64.

### Step 0: Install X11 / GL dev libraries

```bash
sudo apt-get install -y libx11-dev libxrandr-dev libxinerama-dev \
  libxcursor-dev libxi-dev libxkbcommon-dev libgl1-mesa-dev
```

### Step 1: Create Python 3.12 environment

Isaac Sim 6.0.0 requires `Python ==3.12.*`.

```bash
conda create -n env_isaaclab3 python=3.12 -y
conda activate env_isaaclab3
```

### Step 2: Install PyTorch 2.10.0 + CUDA 13.0

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

### Step 3: Install Isaac Sim 6.0.0

```bash
pip install "isaacsim[all]>=6.0.0" --extra-index-url https://pypi.nvidia.com
```

> **Note:** This takes 5-10 min on first install. `imgui-bundle` compiles from
> source on aarch64 (~2 min).

### Step 4: Reinstall PyTorch with CUDA (critical!)

Isaac Sim's dependency resolver pulls in `torch==2.10.0` (CPU-only). Force
reinstall the CUDA build:

```bash
pip install --force-reinstall --no-deps \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.10.0+cu130 True
```

### Step 5: Switch Isaac Lab to develop branch

```bash
cd ~/Documents/repos/IsaacLab
git fetch origin develop
git checkout -b develop origin/develop
```

### Step 6: Install Isaac Lab from source

```bash
./isaaclab.sh --install rsl_rl
```

This installs all Isaac Lab submodules (isaaclab, isaaclab_physx, isaaclab_rl,
isaaclab_assets, isaaclab_tasks, etc.) plus the rsl_rl framework in editable mode.

### Step 7: Install strafer packages

```bash
cd ~/Documents/repos/Sim2RealLab
pip install -e source/strafer_shared
pip install -e source/strafer_lab
```

### Step 8: Smoke test

```bash
cd ~/Documents/repos/IsaacLab
conda activate env_isaaclab3
export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"

./isaaclab.sh -p ../Sim2RealLab/Scripts/train_strafer_navigation.py \
  --env Isaac-Strafer-Nav-Real-NoCam-v0 \
  --num_envs 64 \
  --max_iterations 10 \
  --headless
```


#### Step 9: Check if Unit Tests pass
```bash
cd ~/Documents/repos/IsaacLab && conda activate env_isaaclab3 && export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" && ./isaaclab.sh -p ~/Documents/repos/Sim2RealLab/source/strafer_lab/run_tests.py all
```

### Key API Changes (develop branch vs main)

| Old API (main / 2.x) | New API (develop / 3.0) |
|---|---|
| `write_root_state_to_sim(state, env_ids)` | `write_root_pose_to_sim_index(root_pose=..., env_ids=...)` + `write_root_velocity_to_sim_index(root_velocity=..., env_ids=...)` |
| `write_joint_state_to_sim(pos, vel, env_ids=...)` | `write_joint_position_to_sim_index(position=..., env_ids=...)` + `write_joint_velocity_to_sim_index(velocity=..., env_ids=...)` |
| `data.default_root_state` | `data.default_root_pose` + `data.default_root_vel` |
| `data.root_state_w` | `data.root_link_pose_w` + `data.root_link_vel_w` |
| `data.body_pos_w` / `data.body_quat_w` | `data.body_link_pos_w` / `data.body_link_quat_w` |
| `root_physx_view` | `root_view` |
| Quaternion WXYZ `(w,x,y,z)` | Quaternion XYZW `(x,y,z,w)` |
| Returns `torch.Tensor` | Returns `wp.array` — wrap with `wp.to_torch()` |
| Positional args in write methods | Keyword-only args (`root_pose=`, `env_ids=`, etc.) |
| `RslRlMLPModelCfg(init_noise_std=0.5)` | `RslRlMLPModelCfg(distribution_cfg=GaussianDistributionCfg(init_std=0.5))` |

> **Note:** `root_pos_w`, `root_quat_w`, `root_lin_vel_b` etc. still exist as
> shorthand aliases on the develop branch — no need to rename observation code.

### root_view getter/setter pattern (warp conversion)

The `root_view.get_*()` methods return warp arrays. Convert to torch for
manipulation, then convert back before writing:

```python
materials = robot.root_view.get_material_properties()
materials = wp.to_torch(materials)   # convert for torch indexing/math
# ... modify materials ...
robot.root_view.set_material_properties(
    wp.from_torch(materials),
    wp.from_torch(env_ids.cpu().int())  # indices MUST be on CPU
)
```

### Smoke Test Results (develop branch)

- **Environment:** `Isaac-Strafer-Nav-Real-NoCam-v0`
- **Stack:** Isaac Sim 6.0.0, Isaac Lab develop (v4.5.24), Python 3.12.13, PyTorch 2.10.0+cu130, rsl_rl 5.0.1
- **Num envs:** 64
- **Iterations:** 10
- **Iteration time:** ~7.2–7.6s per iteration
- **Total time:** ~1:15
- **Mean reward:** -0.02 → 0.50 (increasing over 10 iterations)
- **Goal reached:** 39.6%
- **Result:** SUCCESS

## Remaining Migration TODOs

Items that still need attention to complete the Isaac Lab 2.x → 3.0 migration.

### 1. Refactor `STRAFER_PPO_DEPTH_RUNNER_CFG` (deprecated `policy` field)

**File:** `strafer_lab/tasks/navigation/agents/rsl_rl_ppo_cfg.py` line 131

The depth-training runner config still uses the deprecated `policy` field with
`handle_deprecated_rsl_rl_cfg()`.  Refactor `StraferActorCritic` into separate
actor and critic models and migrate to `actor=RslRlCNNModelCfg` /
`critic=RslRlMLPModelCfg`.

### 2. Eliminate `compat.py` — dead-code removal and inline migration

**File:** `strafer_lab/compat.py`

Most compat wrappers are **dead code** — never imported or called anywhere:

| Dead symbol | Line | Purpose |
|---|---|---|
| `write_root_state` | 111 | Callers already use `write_root_pose_to_sim_index` / `write_root_velocity_to_sim_index` directly |
| `write_joint_state` | 123 | Callers already use `write_joint_position_to_sim_index` / `write_joint_velocity_to_sim_index` directly |
| `get_root_view` | 143 | Callers already use `asset.root_view` directly |
| `get_body_link_pos_w` | 155 | Callers already use `data.body_link_pos_w` directly |
| `get_body_link_quat_w` | 160 | Callers already use `data.body_link_quat_w` directly |
| `write_body_link_pose` | 165 | `proc_room.py:807` calls `.write_body_link_pose_to_sim_index` directly |
| `set_identity_quat` | 75 | Never called |
| `set_yaw_quat` | 81 | Never called |
| `extract_yaw` | 58 | Never called (manual inline used instead) |
| `yaw_to_quat_tensor` | 67 | Never called |
| `make_quat_tuple` | 51 | Never called |
| `QX`, `QY`, `QZ` | 41/43 | Never imported (only `QW` is used) |

**Still in use** (3 symbols — inline these, then delete compat.py):

| Symbol | Call sites |
|---|---|
| `ensure_torch` | `observations.py:535, 595` — replace with direct `wp.to_torch()` |
| `IDENTITY_QUAT` | `strafer_env_cfg.py:227, 1368` — replace with `(0.0, 0.0, 0.0, 1.0)` |
| `QW` | `events.py:463`, `test_events.py:362` — replace with literal `3` or inline constant |

### 3. Replace `AppLauncher` with `--viz none`

**File:** `test/conftest.py` line 17

```python
app_launcher = AppLauncher(headless=True, enable_cameras=True)  # TODO: replace with --viz none in 3.0 stable
```

When Isaac Lab 3.0 reaches stable release, replace `AppLauncher(headless=True,
enable_cameras=True)` with the new `--viz none` CLI flag.

### 4. Add electronics mass meshes to robot USD

**File:** `README.md` line 331

Add USD meshes with rigid bodies for RoboClaw motor controllers (×2), Jetson
Orin Nano, buck converter, and Intel RealSense D555 camera.  Each should be
positioned in the USD and given its catalog mass in `setup_physics.py`.

### 5. Hardcoded XYZW quaternion indices (fragile but correct)

These sites use raw `[:, 0]`..`[:, 3]` to unpack XYZW quaternions.  They work
on the develop branch but will break if the convention changes again (or if
backported to 2.x).  Consider using `QW`/`QX`/`QY`/`QZ` from compat (or
hardcoded constants) for clarity.

| File | Lines |
|---|---|
| `mdp/rewards.py` | 123, 160, 547–550 |
| `mdp/observations.py` | 385, 415, 475 |
| `mdp/events.py` | 55–56, 120–121, 510–511 |
| `mdp/proc_room.py` | 440–446 |
| `test/actions/conftest.py` | 386 |
| `test/rewards/test_rewards.py` | 305 |
| `test/rewards/test_collision_rewards.py` | 140 |
| `test/sensors/test_imu_collision.py` | 143 |
