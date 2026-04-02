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
conda activate env_isaaclab
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
