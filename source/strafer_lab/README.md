# Strafer Lab

Isaac Lab extension for training and deploying the Gobilda Strafer mecanum wheel robot.

## Installation

```bash
# From the workspace root, with Isaac Lab Python environment active
cd c:\Worspace
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

```bash
# Default: Realistic Full (recommended for sim-to-real)
.\IsaacLab\isaaclab.bat -p Scripts\train_strafer_navigation.py --num_envs 512

# Fast iteration with NoCam (proprioceptive only)
.\IsaacLab\isaaclab.bat -p Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-NoCam-v0 --num_envs 4096

# Stress-test training with Robust (aggressive noise)
.\IsaacLab\isaaclab.bat -p Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Robust-v0 --headless

# Or use Isaac Lab's built-in training script
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Strafer-Nav-Real-v0 --num_envs 4096
```

### Evaluate Trained Policy

```bash
python IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Strafer-Nav-Real-Play-v0 --num_envs 50
```

### Executing Tests

```bash
# From the IsaacLab directory
cd c:\Worspace\IsaacLab
.\isaaclab.bat -p -m pytest ..\source\strafer_lab\test\ -v

# Unit tests (no simulation needed for most)
.\isaaclab.bat -p -m pytest ..\source\strafer_lab\test\unit\ -v

# Integration tests (requires Isaac Sim)
.\isaaclab.bat -p -m pytest ..\source\strafer_lab\test\integration\test_motor_dynamics.py -v
```

## Project Structure

```
source/strafer_lab/
├── config/
│   └── extension.toml           # Package metadata
├── strafer_lab/
│   ├── __init__.py              # Package init, registers gym envs
│   ├── assets/                  # Robot configurations
│   │   ├── __init__.py
│   │   └── strafer.py           # ArticulationCfg for Strafer robot
│   └── tasks/                   # RL environments
│       ├── __init__.py
│       └── navigation/          # Navigation task
│           ├── __init__.py      # gym.register()
│           ├── strafer_env_cfg.py  # Environment configuration
│           ├── agents/          # RL algorithm configs
│           │   ├── rsl_rl_ppo_cfg.py
│           │   └── skrl_ppo_cfg.yaml
│           └── mdp/             # MDP components
│               ├── actions.py   # Mecanum wheel actions
│               ├── observations.py
│               ├── rewards.py
│               ├── terminations.py
│               ├── events.py
│               └── commands.py  # Goal command generator
├── pyproject.toml
└── setup.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `Scripts/test_strafer_env.py` | Test motion patterns (forward, strafe, rotate, etc.) |
| `Scripts/train_strafer_navigation.py` | Train navigation policy with RSL-RL PPO |

### Script Arguments

```bash
# test_strafer_env.py
--env          Environment ID (default: Isaac-Strafer-Nav-Real-v0)
--pattern      Motion pattern: forward, strafe, rotate, circle, figure8, all
--duration     Duration per pattern in seconds
--num_envs     Number of parallel environments

# train_strafer_navigation.py
--env          Environment ID (default: Isaac-Strafer-Nav-Real-v0)
--num_envs     Number of parallel environments
--max_iterations  Training iterations
--headless     Run without GUI
--resume       Path to checkpoint to resume from
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
| Ideal | Full (RGB+Depth) | `Isaac-Strafer-Nav-v0` | `Isaac-Strafer-Nav-Play-v0` | 19,215 |
| Ideal | Depth-only | `Isaac-Strafer-Nav-Depth-v0` | `Isaac-Strafer-Nav-Depth-Play-v0` | 4,815 |
| Ideal | NoCam | `Isaac-Strafer-Nav-NoCam-v0` | `Isaac-Strafer-Nav-NoCam-Play-v0` | 15 |
| Realistic | Full (RGB+Depth) | `Isaac-Strafer-Nav-Real-v0` | `Isaac-Strafer-Nav-Real-Play-v0` | 19,215 |
| Realistic | Depth-only | `Isaac-Strafer-Nav-Real-Depth-v0` | `Isaac-Strafer-Nav-Real-Depth-Play-v0` | 4,815 |
| Robust | Full (RGB+Depth) | `Isaac-Strafer-Nav-Robust-v0` | `Isaac-Strafer-Nav-Robust-Play-v0` | 19,215 |

### Observation Space

| Component | Dims | Source |
|-----------|------|--------|
| IMU Linear Acceleration | 3 | D555 BMI055 accelerometer (±16g) |
| IMU Angular Velocity | 3 | D555 BMI055 gyroscope (±2000°/s) |
| Wheel Encoder Velocities | 4 | GoBilda 537.7 PPR encoders |
| Goal Position | 2 | Relative (x, y) in robot frame |
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
- **`PolicyVariant.NOCAM`** (15 dims), **`PolicyVariant.DEPTH`** (4815 dims) -- define obs field ordering and scales
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

The ROS eval path (Path 2) runs the same model on real hardware or Isaac Sim via ROS2 bridge. See `source/strafer_ros/CLAUDE.md` and `docs/SIM_TO_REAL_PLAN.md` Section 0.

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
- `collapse_redundant_xforms.py` - Clean USD hierarchy
- `setup_physics.py` - Add articulation, joints, colliders

### 2. RL Training (This Extension)

- Manager-based RL environment
- PPO training with RSL-RL or SKRL
- Domain randomization for sim-to-real

### 3. Deployment

- [ ] Export trained policy to `.pt` (TorchScript) via `export_policy_as_jit()`
- [ ] Measure inference latency on both platforms
- [ ] Later: export to ONNX, optimize with TensorRT on Jetson

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

- Isaac Lab (isaaclab, isaaclab_tasks, isaaclab_rl)
- Isaac Sim 4.5+
- Python 3.10+

## License

MIT
