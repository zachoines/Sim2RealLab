# Strafer Robot Sim-to-Real Project

A complete sim-to-real pipeline for the Gobilda Strafer mecanum wheel robot using NVIDIA Isaac Lab.

## Overview

This project provides:
- **Asset processing pipeline** - Convert CAD → USD → Physics-rigged robot
- **Isaac Lab extension** - RL training environments for navigation
- **Sim-to-real deployment** - ROS2 bridge and ONNX export (planned)

## Project Structure

```
c:\Worspace\
├── Assets/                              # Robot USD assets
│   └── 3209-0001-0006-v6/
│       ├── 3209-0001-0006.usd           # Original imported USD
│       ├── 3209-0001-0006-collapsed.usd # Cleaned hierarchy
│       └── 3209-0001-0006-physics.usd   # Physics-rigged for simulation
│
├── Scripts/                             # Asset processing pipeline
│   ├── collapse_redundant_xforms.py    # Clean USD hierarchy
│   ├── setup_physics.py                # Add articulation, joints, colliders
│   ├── launch_isaac_sim.ps1            # Launch Isaac Sim
│   └── launch_isaac_lab.ps1            # Launch Isaac Lab
│
├── source/                              # Isaac Lab Extension
│   └── strafer_lab/                     # Custom extension package
│       ├── config/extension.toml        # Package metadata
│       ├── strafer_lab/
│       │   ├── assets/                  # Robot ArticulationCfg
│       │   │   └── strafer.py
│       │   └── tasks/                   # RL environments
│       │       └── navigation/
│       │           ├── strafer_env_cfg.py
│       │           ├── agents/          # RL algorithm configs
│       │           └── mdp/             # Observations, rewards, actions
│       ├── pyproject.toml
│       └── README.md
│
├── scripts/                             # Training & deployment (TODO)
│   ├── train.py
│   ├── play.py
│   └── ros2/
│
├── policies/                            # Trained checkpoints (TODO)
│
├── IsaacLab/                            # Isaac Lab (submodule/install)
│
└── README.md                            # This file
```

## Quick Start

### 1. Environment Setup

```powershell
# Activate the virtual environment
& C:\Worspace\venv_isaac\Scripts\Activate.ps1

# Install the strafer_lab extension
python -m pip install -e source/strafer_lab
```

### 2. Asset Processing Pipeline

```powershell
# Step 1: Consolidate redundant Xforms
python Scripts/collapse_redundant_xforms.py `
  --stage Assets/3209-0001-0006-v6/3209-0001-0006.usd `
  --root /World/strafer `
  --output ./collapse_log.txt `
  --tree-output ./collapsed_tree.txt `
  --output-usd ./Assets/3209-0001-0006-v6/3209-0001-0006-collapsed.usd

# Step 2: Add physics (articulation, joints, colliders)
python Scripts/setup_physics.py `
  --stage Assets/3209-0001-0006-v6/3209-0001-0006-collapsed.usd `
  --output-usd Assets/3209-0001-0006-v6/3209-0001-0006-physics.usd `
  --log ./setup_physics_log.txt `
  --delete-excluded
```

### 3. RL Training

```powershell
# List available environments
python -c "import strafer_lab; import gymnasium as gym; print([e for e in gym.envs.registry.keys() if 'Strafer' in e])"

# Train with RSL-RL
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py `
  --task Isaac-Strafer-Navigation-v0 `
  --num_envs 4096

# Evaluate trained policy
python IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py `
  --task Isaac-Strafer-Navigation-Play-v0 `
  --num_envs 50
```

## Sim-to-Real Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Asset Prep  │────▶│  2. RL Training │────▶│  3. Deployment  │
│  (USD scripts)  │     │  (Isaac Lab)    │     │  (ROS2 + ONNX)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
     ▲                         │                        │
     │                         ▼                        ▼
 CAD → USD              Domain Randomization     Real Robot
 collapse_xforms.py     PPO Training             ros2_bridge
 setup_physics.py       Reward Shaping           Policy Export
```

### Pipeline Status

| Stage | Status | Description |
|-------|--------|-------------|
| Asset Import | ✅ Done | Convert CAD to USD |
| Hierarchy Cleanup | ✅ Done | collapse_redundant_xforms.py |
| Physics Rigging | ✅ Done | setup_physics.py |
| Isaac Lab Extension | ✅ Done | strafer_lab package |
| RL Environment | ✅ Done | Navigation task |
| D555 Camera | ✅ Done | RGB + Depth observations |
| D555 IMU | ✅ Done | Accelerometer + Gyroscope |
| Motor Encoders | ✅ Done | 537.7 PPR encoder model |
| Training Scripts | 🔲 TODO | Custom train.py |
| Policy Export | 🔲 TODO | ONNX export |
| ROS2 Bridge | 🔲 TODO | omni.isaac.ros2_bridge |
| Real Robot Deploy | 🔲 TODO | Hardware integration |

## Robot Configuration

The Strafer robot is a 4-wheel mecanum drive platform:

- **Wheels**: 4 mecanum wheels with velocity control
- **Rollers**: 40 passive rollers (10 per wheel, free spinning)
- **Sensors**: Intel RealSense D555 (RGB-D camera + IMU)
- **Motors**: GoBilda 5203 gear motors with 537.7 PPR encoders

### Sensors

| Sensor | Model | Observations |
|--------|-------|-------------|
| Camera | D555 RGB-D | 80×60 RGB (14400) + Depth (4800) |
| IMU | D555 BMI055 | Accelerometer (3) + Gyroscope (3) |
| Encoders | GoBilda 537.7 PPR | Wheel velocities (4) |

### Actuator Groups

| Group | Joints | Control Mode |
|-------|--------|--------------|
| `wheel_drives` | 4 wheel cores | Velocity |
| `roller_bearings` | 40 rollers | Passive |

### Action Space

| Dimension | Description | Range |
|-----------|-------------|-------|
| vx | Forward velocity | -10 to 10 m/s |
| vy | Strafe velocity | -10 to 10 m/s |
| omega | Rotation rate | -5 to 5 rad/s |

## Development

### Adding New Tasks

1. Create folder under `source/strafer_lab/strafer_lab/tasks/`
2. Define `*_env_cfg.py` with `ManagerBasedRLEnvCfg`
3. Add MDP components in `mdp/` subfolder
4. Register with `gym.register()` in `__init__.py`

### Modifying Rewards

Edit `source/strafer_lab/strafer_lab/tasks/navigation/mdp/rewards.py`

### Domain Randomization

Edit `source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py`

## References

- [Gobilda Strafer Chassis](https://www.gobilda.com/strafer-chassis-kit-v4/)
- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim Documentation](https://docs.omniverse.nvidia.com/isaacsim/)

## License

MIT