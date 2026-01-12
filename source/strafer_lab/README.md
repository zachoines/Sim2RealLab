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

### Train with RSL-RL

```bash
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Strafer-Navigation-v0 --num_envs 4096
```

### Evaluate Trained Policy

```bash
python IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Strafer-Navigation-Play-v0 --num_envs 50
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

| Environment ID | Sensors | Obs Dims | Description |
|---------------|---------|----------|-------------|
| `Isaac-Strafer-Navigation-v0` | IMU + Enc + RGB + Depth | 19,215 | Full sensors |
| `Isaac-Strafer-Navigation-Depth-v0` | IMU + Enc + Depth | 4,815 | Depth only |
| `Isaac-Strafer-Navigation-RGB-v0` | IMU + Enc + RGB | 14,415 | RGB only |
| `Isaac-Strafer-Navigation-NoCam-v0` | IMU + Enc | 15 | Proprioceptive |

All environments have corresponding `-Play-v0` variants for evaluation (50 envs, no noise).

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
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Asset Prep     │────▶│  RL Training    │────▶│  Deployment     │
│  (USD scripts)  │     │  (Isaac Lab)    │     │  (ROS2 + ONNX)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 1. Asset Preparation (Complete)

USD processing scripts in `Scripts/`:
- `collapse_redundant_xforms.py` - Clean USD hierarchy
- `setup_physics.py` - Add articulation, joints, colliders

### 2. RL Training (This Extension)

- Manager-based RL environment
- PPO training with RSL-RL or SKRL
- Domain randomization for sim-to-real

### 3. Deployment (TODO)

- [ ] Export trained policy to ONNX
- [ ] ROS2 bridge integration
- [ ] Real robot deployment scripts

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
