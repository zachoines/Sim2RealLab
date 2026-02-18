# Strafer Robot -- Sim-to-Real

A complete sim-to-real pipeline for the GoBilda Strafer mecanum-wheel robot. Train navigation policies in NVIDIA Isaac Lab, deploy on a Jetson Orin Nano via ROS2.

## Vision

**End state**: A user gives a natural language command ("go to the kitchen, wait, then come back"). A VLM interprets the command alongside the robot's current state (Nav2 map, RGB/depth, IMU) and outputs a sequence of goal poses or skills. The robot executes each via trained low-level RL controllers, repeating until the command is fulfilled.

**MVP**: An RL policy trained in Isaac Lab navigates to a hardcoded goal pose while avoiding obstacles. The same model runs in simulation (via gym) and on the real robot (via ROS2). No VLM, no natural language, no skills -- just "here's a (x, y) goal, go there."

### Two Evaluation Paths

```
PATH 1: Gym Eval (fast, during training)
  Isaac Lab env.step(action) → obs tensor directly
  No ROS. Pure Python. Thousands of envs in parallel.
  Used for: training, hyperparameter search, quick policy checks

PATH 2: ROS Eval (realistic, deployment validation)
  Inference node reads sensor topics → assembles obs → runs policy → publishes cmd_vel
  Works against EITHER:
    (a) Real hardware (Jetson + RoboClaw + RealSense)
    (b) Isaac Sim via ROS2 bridge (sim acting as hardware)
  Used for: final validation, integration testing, real deployment
```

Both paths reference the same **policy contract** (observation spec, action spec, normalization) defined in `strafer_shared.policy_interface`. This guarantees sim-real parity.

### Building Beyond MVP

| Phase | Goal Interface | Policy | Platform |
|-------|---------------|--------|----------|
| **MVP** | Hardcoded goal pose | RL nav-to-goal (.pt) | Gym eval + ROS eval |
| **Phase 2** | Nav2 waypoints from map | RL nav + obstacle avoidance | ROS + Nav2 |
| **Phase 3** | VLM decodes NL commands → poses/skills | Multi-task RL (behavorial cloning from Nav2 data) | ROS + VLM |

## Hardware

| Component | Model | Purpose |
|-----------|-------|---------|
| Compute (sim) | Windows workstation + NVIDIA GPU | Isaac Lab training |
| Compute (real) | Jetson Orin Nano (JetPack 6.2) | On-robot inference |
| Camera | Intel RealSense D555 | RGB + depth + IMU (BMI055) |
| Motors | 4x GoBilda 5203 Yellow Jacket (19.2:1) | Mecanum wheel drive, 537.7 PPR encoders |
| Motor Controllers | 2x RoboClaw ST 2x45A | USB serial, velocity PID, encoder input |
| Chassis | GoBilda Strafer v4 | 4-wheel mecanum platform |

## Repository Structure

```
├── source/
│   ├── strafer_lab/             # Isaac Lab simulation (Windows)
│   │   ├── strafer_lab/
│   │   │   ├── assets/          # Robot ArticulationCfg
│   │   │   └── tasks/navigation/
│   │   │       ├── strafer_env_cfg.py   # 18 environment variants
│   │   │       ├── sim_real_cfg.py      # Sim-to-real contracts
│   │   │       ├── agents/              # PPO configs (RSL-RL, SKRL)
│   │   │       └── mdp/                 # Actions, observations, rewards
│   │   └── test/                # Simulation tests
│   │
│   ├── strafer_ros/             # ROS2 packages (Jetson Orin Nano)
│   │   ├── CLAUDE.md            # Agent context for Jetson-side development
│   │   ├── strafer_msgs/        # Custom message definitions
│   │   ├── strafer_driver/      # RoboClaw hardware interface
│   │   ├── strafer_perception/  # RealSense integration
│   │   ├── strafer_inference/   # ONNX/TensorRT policy inference
│   │   ├── strafer_slam/        # RTAB-Map SLAM
│   │   ├── strafer_navigation/  # Nav2 integration
│   │   └── strafer_bringup/     # Launch files
│   │
│   └── strafer_shared/          # Shared Python module (both machines)
│       └── strafer_shared/
│           ├── constants.py           # Single source of truth for all robot params
│           ├── mecanum_kinematics.py  # Forward/inverse kinematics (NumPy)
│           └── policy_interface.py    # Policy contract: obs/action specs, assembly, inference
│
├── Assets/                      # Robot USD assets
├── Scripts/                     # Asset processing & training scripts
├── IsaacLab/                    # NVIDIA Isaac Lab (submodule)
├── docs/
│   └── SIM_TO_REAL_PLAN.md      # Detailed deployment plan
└── README.md
```

## Pipeline Status

| Stage | Status | Details |
|-------|--------|---------|
| CAD to USD | Done | Asset import, hierarchy cleanup, physics rigging |
| Isaac Lab Extension | Done | 18 env variants (3 realism x 3 sensor configs x train/play) |
| Sensor Models | Done | D555 camera (RGB+depth), D555 IMU (BMI055), motor encoders (537.7 PPR) |
| Noise Models | Done | Per-sensor noise with 3 presets: ideal, realistic, robust |
| Sim-to-Real Contract | Done | Motor dynamics, command delay, sensor latency |
| Shared Kinematics | Done | `strafer_shared` module with constants + mecanum math |
| Policy Interface | Done | `policy_interface.py` -- obs/action contract, model loading |
| Jetson Setup | Done | JetPack 6.2 flashed, SSH accessible |
| ROS2 Packages | In Progress | Package structure created, implementation pending |
| Policy Export | Planned | PyTorch (.pt) initially, ONNX/TensorRT later |
| SLAM + Nav2 | Planned | RTAB-Map + robot_localization EKF + Nav2 |

## Quick Start

### Simulation (Windows)

```powershell
# Activate virtual environment
& C:\Worspace\venv_isaac\Scripts\Activate.ps1

# Install extensions
python -m pip install -e source/strafer_lab
python -m pip install -e source/strafer_shared

# List environments
python -c "import strafer_lab; import gymnasium as gym; print([e for e in gym.envs.registry.keys() if 'Strafer' in e])"

# Train (proprioceptive-only, fastest iteration)
.\IsaacLab\isaaclab.bat -p Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 4096

# Train (with depth camera, sim-to-real target)
.\IsaacLab\isaaclab.bat -p Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-Depth-v0 --num_envs 512

# Evaluate
python IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Strafer-Nav-Real-Play-v0 --num_envs 50

# Run tests
cd IsaacLab && .\isaaclab.bat -p -m pytest ..\source\strafer_lab\test\ -v
```

### Robot (Jetson Orin Nano)

```bash
# Clone repo on Jetson
git clone <repo-url> ~/strafer
cd ~/strafer

# Install shared module
pip install -e source/strafer_shared

# Set up ROS2 workspace
mkdir -p ~/strafer_ws/src
ln -s ~/strafer/source/strafer_ros/* ~/strafer_ws/src/
ln -s ~/strafer/source/strafer_shared ~/strafer_ws/src/

cd ~/strafer_ws
colcon build --symlink-install
source install/setup.bash
```

## Environments

| Realism | Sensors | Train ID | Obs Dims |
|---------|---------|----------|----------|
| Ideal | Full (RGB+Depth) | `Isaac-Strafer-Nav-v0` | 19,215 |
| Ideal | Depth-only | `Isaac-Strafer-Nav-Depth-v0` | 4,815 |
| Ideal | NoCam | `Isaac-Strafer-Nav-NoCam-v0` | 15 |
| Realistic | Full | `Isaac-Strafer-Nav-Real-v0` | 19,215 |
| Realistic | Depth-only | `Isaac-Strafer-Nav-Real-Depth-v0` | 4,815 |
| Realistic | NoCam | `Isaac-Strafer-Nav-Real-NoCam-v0` | 15 |
| Robust | Full | `Isaac-Strafer-Nav-Robust-v0` | 19,215 |

## Key Design Decisions

**Monorepo**: Sim and real code share a single repo so that the observation ordering, normalization constants, and kinematics are always in sync. The `strafer_shared` module is the single source of truth.

**Two-contract architecture**: The policy contract (obs spec, action spec, normalization) lives in `strafer_shared.policy_interface` and is used by both the gym environment and the ROS2 inference node. The ROS topic contract (`cmd_vel`, `joint_states`, `odom`, etc.) is the deployment interface -- identical whether the backend is real hardware or Isaac Sim via the ROS2 bridge.

**Custom driver over ros2_control**: The `strafer_driver` uses a custom Python node rather than ros2_control's `mecanum_drive_controller`. This keeps kinematics in `strafer_shared` (matching sim exactly) and avoids requiring a C++ hardware interface for RoboClaw serial communication.

**Sim-to-Real Contract**: Three realism presets define motor dynamics (time constant, slew rate), command delay, and per-sensor noise. The policy trained with `Realistic` or `Robust` presets transfers to real hardware.

**No Arduino**: RoboClaw ST 2x45A controllers connect directly to the Jetson via USB. They handle motor PID, encoder counting, and provide a Python API.

## Documentation

- [Deployment Plan](docs/SIM_TO_REAL_PLAN.md) -- project goals, MVP path, ROS2 architecture, implementation phases
- [Wiring Guide](docs/WIRING_GUIDE.md) -- motor, encoder, RoboClaw, and Jetson connections
- [Simulation Extension](source/strafer_lab/README.md) -- Isaac Lab environments, training, gym eval path
- [ROS2 Development Guide](source/strafer_ros/CLAUDE.md) -- Jetson-side driver/inference development

## References

- [GoBilda Strafer Chassis](https://www.gobilda.com/strafer-chassis-kit-v4/)
- [GoBilda 5203 Motor](https://www.gobilda.com/5203-series-yellow-jacket-planetary-gear-motor-19-2-1-ratio-24mm-length-8mm-rex-shaft-312-rpm-3-3-5v-encoder/)
- [RoboClaw ST 2x45A](https://www.gobilda.com/roboclaw-st-2x45a-motor-controller/)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [ROS2 Humble](https://docs.ros.org/en/humble/)

## License

MIT
