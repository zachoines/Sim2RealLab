# Strafer Lab

Isaac Lab extension for the GoBilda Strafer mecanum wheel robot. Two roles:

1. **RL policy training** — manager-based Isaac Lab environment, PPO training with
   RSL-RL or SKRL, realism / robust / ideal sim-to-real variants.
2. **Synthetic data pipeline** — Infinigen procedural scene generation, scene
   metadata extraction, 4-stage description generation, CLIP contrastive +
   VLM LoRA fine-tune data prep. See `tools/` and `scripts/` below.

The package's top-level `__init__.py` imports Isaac Sim Kit-dependent
subpackages (`.tasks`, `.assets`) behind a graceful `ModuleNotFoundError`
fallback, so `strafer_lab.tools.*` remains importable from plain-Python
environments (DGX batch scripts, pytest, CI) that do not launch Isaac Sim via
`AppLauncher`. When Isaac Sim IS active, the imports behave identically to
before.

## Installation

### DGX Spark (aarch64, preferred)

The DGX path installs `strafer_lab` into an Isaac-Lab conda env. Detailed
setup — Isaac Sim source build, Isaac Lab v2.3.2 checkout, `pxr` wiring — is
in `docs/PHASE_15_DGX.md`. Short version:

```bash
# 1. Source the shell wrapper to load .env (STRAFER_BLENDER_BIN, ISAACSIM_PATH, …)
#    and set LD_PRELOAD for aarch64.
cd /home/you/Workspace/Sim2RealLab
source env_setup.sh

# 2. Activate the conda env that was created via `isaaclab.sh -c env_phase15`
conda activate env_phase15

# 3. Install strafer_lab editable. `--no-build-isolation` is required because
#    flatdict (an upstream Isaac Lab dep) still uses legacy pkg_resources.
pip install --no-build-isolation -e source/strafer_lab
```

After install, smoke-test:

```bash
python -c "
import strafer_lab
from strafer_lab.tools.scene_labels import get_scene_label_set
from strafer_lab.tools.spatial_description import SpatialDescriptionBuilder
from strafer_lab.tools.dataset_export import run_export
print('strafer_lab tools OK')
"
```

### Windows workstation

```bash
# From the workspace root, with Isaac Lab Python environment active
cd C:\Workspace
python -m pip install -e source/strafer_lab
```

## Quick Start

### List Available Environments

```bash
python -c "import strafer_lab; import gymnasium as gym; print([e for e in gym.envs.registry.keys() if 'Strafer' in e])"
```

### Test Environment with Motion Patterns

```bash
# Default: Realistic Full (motor dynamics + sensor noise)
.\IsaacLab\isaaclab.bat -p Scripts\test_strafer_env.py --pattern forward

# Test with Ideal (no noise) for debugging
.\IsaacLab\isaaclab.bat -p Scripts\test_strafer_env.py --env Isaac-Strafer-Nav-v0

# Test all motion patterns
.\IsaacLab\isaaclab.bat -p Scripts\test_strafer_env.py --pattern all --duration 5.0
```

### Train with RSL-RL

All commands run from `C:\Worspace\IsaacLab` with the Isaac Lab venv activated
(`source C:/Worspace/venv_isaac/Scripts/activate` or `& C:\Worspace\venv_isaac\Scripts\Activate.ps1`).

```bash
cd C:\Worspace\IsaacLab

# NoCam Realistic (proprioceptive only, 19 obs dims — recommended first)
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 512

# Depth Realistic (adds depth camera, 4819 obs dims — cameras auto-enabled)
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-Depth-v0 --num_envs 32

# Full Realistic (RGB + Depth, 19219 obs dims)
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-v0 --num_envs 32

# Headless (no GUI, faster)
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 4096 --headless

# Stress-test with Robust (aggressive noise + dynamics)
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Robust-NoCam-v0 --num_envs 4096 --headless

# Resume from checkpoint
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --resume logs\rsl_rl\strafer_navigation\run_XXXXX\model_500.pt
```

### Monitor Training

```bash
# In a separate terminal (with venv activated)
tensorboard --logdir logs/rsl_rl/strafer_navigation
# Open http://localhost:6006
```

Key metrics to watch:
- **Train/mean_reward** — should trend upward
- **Train/mean_episode_length** — should increase (robot surviving longer)
- **Loss/policy** — oscillating but bounded (not diverging)
- **Policy/std** — slowly decreasing (not collapsing to 0)

### Evaluate Trained Policy

```bash
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Strafer-Nav-Real-NoCam-Play-v0 --num_envs 50
```

### Executing Tests

Tests require the Isaac Lab Python environment and a GPU. All test suites
launch Isaac Sim headlessly via the root `test/conftest.py`.

**Recommended: use `run_tests.py`** for clean output (suppresses Isaac Sim
initialization noise and captures results via junit-xml):

```bash
cd C:\Worspace\IsaacLab

# Run all suites
.\isaaclab.bat -p ..\source\strafer_lab\run_tests.py all

# Run specific suites
.\isaaclab.bat -p ..\source\strafer_lab\run_tests.py rewards observations

# Run a single suite
.\isaaclab.bat -p ..\source\strafer_lab\run_tests.py terminations
```

Available suites: `terminations`, `events`, `commands`, `observations`,
`curriculums`, `rewards`, `sensors`, `actions`, `env`, `noise_models`,
`depth_noise`, `imu`, `all`

**Direct pytest** (verbose Isaac Sim output, but useful for debugging):

```bash
cd C:\Worspace\IsaacLab

# Single suite
.\isaaclab.bat -p -m pytest ..\source\strafer_lab\test\rewards\test_rewards.py -v

# All tests (slow — each suite restarts Isaac Sim)
.\isaaclab.bat -p -m pytest ..\source\strafer_lab\test\ -v
```

**Notes:**
- Each suite runs in its own subprocess (Isaac Sim's `SimulationContext`
  is a singleton — only one environment per process).
- The root conftest calls `os._exit()` to avoid PhysX teardown hangs on
  Windows, which kills pytest before it prints its summary.
  `run_tests.py` works around this by capturing results via `--junit-xml`.
- `depth_noise` tests each create their own environment, so
  `run_tests.py` runs each file in a separate subprocess automatically.

## Project Structure

```
source/strafer_lab/
├── config/
│   └── extension.toml           # Package metadata
├── run_tests.py                 # Test runner (clean output, junit-xml)
├── scripts/                     # Synthetic data pipeline entry points
│   ├── collect_demos.py           # existing: gamepad teleop demo collection
│   ├── prep_room_usds.py          # Infinigen scene generation orchestrator
│   ├── extract_scene_metadata.py  # Serialize Blender State → scene_metadata.json
│   ├── generate_descriptions.py   # 4-stage scene description pipeline
│   ├── finetune_clip.py           # OpenCLIP ViT-B/32 contrastive fine-tune + ONNX export
│   └── prepare_vlm_finetune_data.py  # Qwen VLM LoRA SFT data prep
├── strafer_lab/
│   ├── __init__.py              # Kit-runtime fallback: tries tasks/assets,
│   │                              swallows ModuleNotFoundError from omni/pxr
│   │                              so tools/* stays importable without Isaac Sim
│   ├── assets/                  # Robot configurations (Isaac Sim-dependent)
│   │   ├── __init__.py
│   │   └── strafer.py           # ArticulationCfg for Strafer robot
│   ├── tools/                   # Pure-Python batch processing helpers
│   │   ├── __init__.py
│   │   ├── scene_labels.py        # RoomEntry / ObjectEntry accessors + shapely lookup
│   │   ├── spatial_description.py # SpatialDescriptionBuilder (Stage 1)
│   │   └── dataset_export.py      # CLIP CSV + basic VLM JSONL exporters
│   └── tasks/                   # RL environments (Isaac Sim-dependent)
│       ├── __init__.py
│       └── navigation/          # Navigation task
│           ├── __init__.py      # gym.register()
│           ├── strafer_env_cfg.py  # Environment configuration
│           ├── sim_real_cfg.py  # Sim-to-real noise contracts
│           ├── agents/          # RL algorithm configs
│           │   ├── rsl_rl_ppo_cfg.py
│           │   └── skrl_ppo_cfg.yaml
│           └── mdp/             # MDP components
│               ├── actions.py   # Mecanum wheel actions
│               ├── observations.py
│               ├── rewards.py
│               ├── terminations.py
│               ├── events.py
│               ├── curriculums.py
│               └── commands.py  # Goal command generator
├── test/
│   ├── conftest.py              # Root: AppLauncher + os._exit() teardown
│   ├── common/                  # Shared fixtures, stats, robot helpers
│   ├── actions/                 # Motor dynamics, kinematics tests
│   ├── commands/                # Goal command generation tests
│   ├── curriculums/             # Curriculum progression tests
│   ├── env/                     # Gym registration tests
│   ├── events/                  # Domain randomization tests
│   ├── noise_models/            # Sensor noise model tests
│   ├── observations/            # Observation function tests
│   ├── rewards/                 # Reward function tests
│   ├── sensors/                 # Observation structure + depth noise
│   └── terminations/            # Termination condition tests
├── pyproject.toml
└── setup.py
```

Note: `tools/*` unit tests live in `source/strafer_autonomy/tests/`
(`test_scene_labels.py`, `test_spatial_description.py`, `test_dataset_export.py`)
because the `strafer_autonomy` conftest installs a namespace stub that lets
those tests import `strafer_lab.tools.*` without loading the Isaac Sim-
dependent `strafer_lab/__init__.py`. This keeps the batch-processing tests
runnable in any Python env without an Isaac Lab install.

## Scripts

### RL training / evaluation (requires Isaac Sim + `AppLauncher`)

| Script | Purpose |
|--------|---------|
| `Scripts/test_strafer_env.py` | Test motion patterns (forward, strafe, rotate, etc.) |
| `Scripts/train_strafer_navigation.py` | Train navigation policy with RSL-RL PPO |

### Synthetic data pipeline (plain Python, no Isaac Sim required)

These scripts run from a plain Python environment — they shell out to
Blender for Infinigen work and do not launch Isaac Sim. Most of them depend
on `STRAFER_BLENDER_BIN` being set via `source env_setup.sh` at the repo
root; `prep_room_usds.py` fails loud with an actionable error if it is not.

| Script | Purpose | Depends on |
|---|---|---|
| `scripts/prep_room_usds.py` | Orchestrate Infinigen scene generation with DGX / Windows / low-memory presets. Subcommands: `generate`, `ingest`, `presets`. | `STRAFER_BLENDER_BIN`, Infinigen repo |
| `scripts/extract_scene_metadata.py` | Serialize Infinigen's Blender `State` (room types, polygons, semantic tags, relations) into `scene_metadata.json`. Also labels USD prims with `semanticLabel` / `instanceId` so Replicator produces labeled bboxes. | `bpy` (inside Blender subprocess), optional `pxr` for USD labeling |
| `scripts/generate_descriptions.py` | 4-stage description pipeline (Stages 1-3 + spot-check bookkeeping): programmatic spatial analysis → VLM description generation (Qwen2.5-VL-7B standalone) → ground-truth validation filter → reservoir sampling for human spot-check. | `scene_metadata.json` from Task 8 |
| `scripts/finetune_clip.py` | Phase 1 CLIP image-text contrastive fine-tuning (OpenCLIP ViT-B/32, symmetric InfoNCE). Exports both towers as `clip_visual.onnx` + `clip_text.onnx` for Jetson semantic map. | `open_clip_torch`, CSV from `dataset_export.py` |
| `scripts/prepare_vlm_finetune_data.py` | Comprehensive VLM LoRA SFT data prep: single-object grounding, 1:3 negatives, ~20% multi-object detection, ~10% description preservation (to protect `describe_scene`). | Perception data, `scene_metadata.json`, Task 9 descriptions |

### Batch processing Python helpers (`strafer_lab.tools`)

Pure-Python runtime helpers with no Isaac Sim dependency:

| Module | Exports |
|---|---|
| `strafer_lab.tools.scene_labels` | `get_scene_metadata`, `iter_rooms`, `iter_objects`, `get_scene_label_set`, `get_room_at_position`, `get_objects_in_room` — typed accessors over `scene_metadata.json` |
| `strafer_lab.tools.spatial_description` | `SpatialDescriptionBuilder`, `quat_to_yaw`, `classify_region`, `classify_bearing` — Stage-1 factual spatial relations (room, distance, bearing, near/mid/far) from simulation ground truth |
| `strafer_lab.tools.dataset_export` | `export_clip_csv`, `export_vlm_grounding_jsonl`, `run_export`, `pixel_bbox_to_qwen`, `format_qwen_grounding_answer` — convert perception + descriptions into CLIP CSV and basic VLM SFT JSONL (ProcRoom frames excluded) |

All three modules are importable from any Python env that has `shapely`,
`numpy`, and `Pillow` — they do NOT require Isaac Sim / Isaac Lab /
Omniverse Kit.

### Script Arguments

```bash
# test_strafer_env.py
--env          Environment ID (default: Isaac-Strafer-Nav-Real-v0)
--pattern      Motion pattern: forward, strafe, rotate, circle, figure8, all
--duration     Duration per pattern in seconds
--num_envs     Number of parallel environments

# train_strafer_navigation.py
--env            Environment ID (default: Isaac-Strafer-Nav-Real-v0)
--num_envs       Number of parallel environments (default: 512)
--max_iterations Training iterations (default: 1000)
--seed           Random seed (default: 42)
--log_dir        Log directory (default: logs/rsl_rl/strafer_navigation)
--resume         Path to checkpoint to resume from
--headless       Run without GUI (AppLauncher flag)
--device         Device to run on, e.g. cuda:0 (AppLauncher flag)
--enable_cameras Enable camera sensors (auto-enabled for non-NoCam variants)
```

## Robot Configuration

The Strafer robot is configured in `assets/strafer.py` with:

- **Wheel drives**: 4 mecanum wheels with velocity control
- **Roller bearings**: 40 passive rollers (10 per wheel)
- **Action space**: `[vx, vy, omega]` - forward, strafe, rotation velocities

### Sensors

| Sensor | Configuration | Update Rate |
|--------|--------------|-------------|
| D555 Camera | 80×60 RGB-D, 87°×58° FOV | 30 Hz |
| D555 IMU | BMI055, ±16g accel, ±2000°/s gyro | 200 Hz |
| Encoders | 537.7 PPR (GoBilda 5203 motors) | Physics rate |

### Encoder Model

The motor encoders model GoBilda 5203 series specifications:
- **Resolution**: 537.7 PPR at output shaft
- **Formula**: `((1 + 46/17) × (1 + 46/11)) × 28 = 537.7`
- **Conversion**: 85.57 ticks per radian

## Environments

Environments are organized by **realism level** (noise/dynamics) and **sensor configuration**:

### Realism Levels

| Level | Motor Dynamics | Sensor Noise | Use Case |
|-------|---------------|--------------|----------|
| **Ideal** | None | None | Debugging, baselines |
| **Realistic** | 50ms time constant, 1-3 step delay | IMU/encoder/camera noise | Sim-to-real target |
| **Robust** | 60ms time constant, 1-5 step delay | 1.5-2.5× noise | Stress-testing |

### Environment IDs

| Realism | Sensors | Train ID | Play ID | Obs Dims |
|---------|---------|----------|---------|----------|
| Ideal | Full (RGB+Depth) | `Isaac-Strafer-Nav-v0` | `Isaac-Strafer-Nav-Play-v0` | 19,219 |
| Ideal | Depth-only | `Isaac-Strafer-Nav-Depth-v0` | `Isaac-Strafer-Nav-Depth-Play-v0` | 4,819 |
| Ideal | NoCam | `Isaac-Strafer-Nav-NoCam-v0` | `Isaac-Strafer-Nav-NoCam-Play-v0` | 19 |
| Realistic | Full (RGB+Depth) | `Isaac-Strafer-Nav-Real-v0` | `Isaac-Strafer-Nav-Real-Play-v0` | 19,219 |
| Realistic | Depth-only | `Isaac-Strafer-Nav-Real-Depth-v0` | `Isaac-Strafer-Nav-Real-Depth-Play-v0` | 4,819 |
| Robust | Full (RGB+Depth) | `Isaac-Strafer-Nav-Robust-v0` | `Isaac-Strafer-Nav-Robust-Play-v0` | 19,219 |

### Observation Space

| Component | Dims | Source |
|-----------|------|--------|
| IMU Linear Acceleration | 3 | D555 BMI055 accelerometer (±16g) |
| IMU Angular Velocity | 3 | D555 BMI055 gyroscope (±2000°/s) |
| Wheel Encoder Velocities | 4 | GoBilda 537.7 PPR encoders |
| Goal Position | 2 | Relative (x, y) in robot frame |
| Goal Distance | 1 | Euclidean distance to goal |
| Goal Heading | 1 | Angular error to goal direction |
| Body Velocity | 2 | Body-frame linear velocity (vx, vy) |
| Last Action | 3 | Previous [vx, vy, omega] |
| Depth Image | 4,800 | D555 depth (80×60, normalized) |
| RGB Image | 14,400 | D555 RGB (80×60×3, normalized) |

## Sim-to-Real Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Asset Prep     │────▶│  RL Training    │────▶│  Gym Eval       │────▶│  ROS Deployment │
│  (USD scripts)  │     │  (Isaac Lab)    │     │  (fast, Python)  │     │  (Jetson/ROS2)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Policy Interface (`strafer_shared.policy_interface`)

The policy contract -- observation spec, action spec, normalization -- lives in `strafer_shared.policy_interface`. This module is the **single source of truth** for how observations are assembled and how actions are interpreted. Both the gym environment and the ROS2 inference node reference it.

Key components:
- **`PolicyVariant.NOCAM`** (19 dims), **`PolicyVariant.DEPTH`** (4819 dims) -- define obs field ordering and scales
- **`assemble_observation(raw, variant)`** -- normalizes and concatenates raw sensor values
- **`interpret_action(action)`** -- denormalizes `[-1,1]` to physical `(vx, vy, omega)`
- **`load_policy(path, variant)`** -- loads `.pt` or `.onnx`, returns a callable

### Gym Eval Path (Path 1)

For fast iteration during training, the gym environment handles observation assembly internally via `env.step()`. The policy is evaluated directly in Python with no ROS overhead:

```python
from strafer_shared.policy_interface import load_policy, PolicyVariant
import gymnasium as gym
import strafer_lab

env = gym.make("Isaac-Strafer-Nav-Real-NoCam-v0", num_envs=1)
policy = load_policy("logs/best_model.pt", PolicyVariant.NOCAM)

obs, info = env.reset()
done = False
while not done:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
```

The ROS eval path (Path 2) runs the same model on real hardware or Isaac Sim via ROS2 bridge. See `docs/STRAFER_AUTONOMY_ROS.md` and `docs/SIM_TO_REAL_PLAN.md`.

### Sim-to-Real Contract

The `SimRealContractCfg` in `sim_real_cfg.py` defines all sim-to-real parameters:

```python
from strafer_lab.tasks.navigation.sim_real_cfg import (
    IDEAL_SIM_CONTRACT,      # No noise/delays
    REAL_ROBOT_CONTRACT,     # Matches real hardware
    ROBUST_TRAINING_CONTRACT # Stress-test training
)
```

| Parameter | Ideal | Realistic | Robust |
|-----------|-------|-----------|--------|
| Motor time constant | - | 50ms | 60ms |
| Command delay | 0 steps | 1-3 steps | 1-5 steps |
| IMU accel noise (σ) | 0 | 0.01 | 0.015 |
| Encoder noise (σ) | 0 | 0.02 | 0.05 |
| Depth noise (σ) | 0 | 0.01 | 0.02 |

### 1. Asset Preparation (Complete)

USD processing scripts in `Scripts/`:
- `add_additional_components.py` - Import top-rack hardware into collapsed USD
- `collapse_redundant_xforms.py` - Clean USD hierarchy
- `setup_physics.py` - Add articulation, joints, colliders (BOM-based masses)

**TODO — Electronics masses**: Add USD meshes with rigid bodies for RoboClaw motor controllers (x2), Jetson Orin Nano, buck converter, and Intel RealSense D555 camera. Each should be positioned in the USD and given its catalog mass in `setup_physics.py`.

### 2. RL Training (This Extension)

- Manager-based RL environment
- PPO training with RSL-RL or SKRL
- Domain randomization for sim-to-real

### 3. Deployment

- [ ] **Export trained policy** to `.pt` (TorchScript) via `export_policy_as_jit()` — **current action item** (gates Phase 4)
  - Call after training: `python Scripts/export_policy.py --checkpoint logs/best_model/model_*.pt --output model.pt`
  - Validate with `benchmark_policy()` from `strafer_shared.policy_interface`; target <5ms on Jetson
- [ ] Measure inference latency on both platforms
- [ ] Later: export to ONNX, optimize with TensorRT on Jetson

**Phase 5 training note**: Before exporting the checkpoint intended for VLM deployment, train with goal position noise (`goal_position_noise_std: 0.2–0.3 m` in `commands.py`) to match the ±0.2–0.5m localization error of Qwen2.5-VL-3B visual grounding. Without this, the policy may oscillate at deployment when given imprecise VLM-generated goals.

## Customization

### Adding New Tasks

1. Create new task folder under `tasks/`
2. Define environment config (`*_env_cfg.py`)
3. Add custom MDP functions in `mdp/`
4. Register with `gym.register()` in `__init__.py`

### Modifying Rewards

Edit `tasks/navigation/mdp/rewards.py` to add/modify reward terms.

### Domain Randomization

Edit `tasks/navigation/mdp/events.py` to add randomization events.

## Dependencies

**Core (RL training, required):**

- Isaac Lab (`isaaclab`, `isaaclab_tasks`, `isaaclab_rl`) — v2.3.2 is the
  Windows-compatible target. Isaac Lab `v3.0.0-beta` drops Windows support
  and lives on a separate `feature/isaaclab-3.0-migration` branch.
- Isaac Sim 5.1.x — built from source on DGX Spark aarch64 (per
  `docs/PHASE_15_DGX.md`) or installed from binaries on Windows.
- Python 3.11 (DGX env_phase15) or 3.10+ (Windows)

**Synthetic data pipeline (plain Python, no Isaac Sim):**

- `shapely` — polygon point-in-polygon tests in `scene_labels` / `spatial_description`
- `numpy`, `Pillow` — standard image / array ops
- `bpy==4.2.0` — required by Infinigen, only needed in `env_infinigen`.
  PyPI has no Linux aarch64 wheel for Blender 4.2, so DGX Spark uses a
  source build documented in the `README.md` inside the sibling
  `~/Workspace/blender-build/` directory (outside this repo because the
  build artifacts are machine-specific).
- `open_clip_torch` — CLIP fine-tuning (Task 11 only)
- `transformers` + Qwen2.5-VL-7B — Stage 2 description generation (Task 9)
- `mlflow` — optional experiment tracking for `finetune_clip.py`

## License

MIT
