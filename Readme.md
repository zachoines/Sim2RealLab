# Strafer Robot — Sim-to-Real

A complete sim-to-real pipeline for the GoBilda Strafer mecanum-wheel robot. Train RL navigation policies in NVIDIA Isaac Lab, deploy on a Jetson Orin Nano via ROS2.

<p align="center">
  <img src="docs/artifacts/strafer_top.jpeg" alt="Strafer robot, top view" width="45%"/>
  <img src="docs/artifacts/strafer_side.jpeg" alt="Strafer robot, side view" width="45%"/>
</p>

---

## Introduction

This project builds a full autonomy stack for a mecanum-wheeled mobile robot — from simulation-trained reinforcement learning policies to real-world deployment with SLAM, navigation, and vision-language understanding. The robot platform is the [GoBilda Strafer Chassis v4](https://www.gobilda.com/strafer-chassis-kit-v4/), a 4-wheel mecanum platform capable of omnidirectional motion. An NVIDIA Jetson Orin Nano runs all onboard inference, perception, and control.

The sim and real code live in a single monorepo. A shared Python module (`strafer_shared`) defines all physical constants, kinematics, and the policy observation/action contract — guaranteeing that the model trained in simulation sees identical inputs on the real robot.

## Goals

**End state**: Natural language command → VLM reasoning → goal/skill sequence → RL low-level controller → hardware execution.

A user says *"go to the kitchen, wait, then come back."* A vision-language model (Qwen2.5-VL-3B) interprets the command alongside the robot's camera feed and map, producing a sequence of goal poses. The robot executes each via a trained RL navigation policy, repeating until the command is fulfilled.

**MVP**: An RL policy trained in Isaac Lab navigates to a hardcoded goal pose while avoiding obstacles. The same model runs in simulation (via gym) and on the real robot (via ROS2). No VLM, no natural language — just *"here's an (x, y) goal, go there."*

### Roadmap

| Phase | Goal Interface | What It Proves |
|-------|---------------|----------------|
| **MVP** | Hardcoded goal pose | Sim-to-real transfer works end-to-end |
| **Phase 2** | Nav2 waypoints from map | SLAM + autonomous navigation in mapped environments |
| **Phase 3** | NL command → VLM → goal pose | Full autonomy: language → perception → planning → control |

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

Both paths reference the same **policy contract** (observation spec, action spec, normalization) defined in `strafer_shared.policy_interface`.

## Current State

### Simulation

The Isaac Lab environment is feature-complete: 18 gym environment variants (3 realism presets x 3 sensor configs x train/play), mecanum kinematics, realistic sensor and actuator noise models, and a PPO training pipeline.

<p align="center">
  <img src="docs/artifacts/strafer_usd.png" alt="Strafer USD model in Isaac Sim editor" width="70%"/>
  <br/>
  <em>Robot USD model in the Isaac Sim editor</em>
</p>

<p align="center">
  <a href="docs/artifacts/strafer_isaac_lab_test_drive.mp4">
    <img src="docs/artifacts/strafer_usd.png" alt="Click to view test drive video" width="50%"/>
  </a>
  <br/>
  <em>Test drive in Isaac Lab — <a href="docs/artifacts/strafer_isaac_lab_test_drive.mp4">watch video</a></em>
</p>

### Hardware

All hardware is wired, tested, and operational. The ROS2 driver, perception pipeline, URDF, SLAM config, and Nav2 config are complete. The robot can be teleoperated and produces odometry, IMU data, and depth images.

### Pipeline Status

| Stage | Status | Details |
|-------|--------|---------|
| CAD to USD | Done | Asset import, hierarchy cleanup, physics rigging |
| Isaac Lab Extension | Done | 18 env variants (3 realism x 3 sensor configs x train/play) |
| Sensor & Noise Models | Done | D555 camera (RGB+depth), BMI055 IMU, motor encoders (537.7 PPR) |
| Sim-to-Real Contract | Done | Motor dynamics, command delay, sensor latency; 3 presets (Ideal/Realistic/Robust) |
| Shared Kinematics | Done | `strafer_shared` module — constants, mecanum math, policy interface |
| Jetson Setup | Done | JetPack 6.2, ROS2 Humble, SSH, udev rules |
| ROS2 Driver | Done | Auto-detect, auto-PID, cmd_vel → motors, encoders → odom/joint_states/TF |
| ROS2 Perception | Done | Depth downsampler, D555 HW clock sync, Madgwick IMU filter |
| URDF + TF Tree | Done | Full TF: map → odom → base_link → chassis/camera/wheels |
| SLAM Config | Done | RTAB-Map tuned for Jetson (2 Hz loop closure, 0.05m grid) |
| Nav2 Config | Done | MPPI controller with OmniMotionModel for mecanum |
| Bringup Launch | Done | Layered launch files + ValidateDrive smoke test |
| RL Policy Training | **In Progress** | Training with `Isaac-Strafer-Nav-Real-NoCam-v0` |
| Policy Inference Node | **Planned** | `strafer_inference` — assemble obs, run model, publish cmd_vel |
| SLAM + Nav2 End-to-End | **TODO** | Build map, localize, send Nav2 goal with obstacle avoidance |
| VLM Integration | **Planned** | `strafer_vlm` — Qwen2.5-VL-3B visual grounding, NL → goal pose |

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
│   │   ├── strafer_msgs/        # Custom message definitions
│   │   ├── strafer_driver/      # RoboClaw hardware interface
│   │   ├── strafer_perception/  # RealSense integration + depth downsampler
│   │   ├── strafer_description/ # URDF/xacro, TF frames
│   │   ├── strafer_inference/   # ONNX/TensorRT policy inference (planned)
│   │   ├── strafer_slam/        # RTAB-Map SLAM
│   │   ├── strafer_navigation/  # Nav2 integration
│   │   ├── strafer_vlm/         # VLM visual grounding (planned)
│   │   └── strafer_bringup/     # Launch files
│   │
│   └── strafer_shared/          # Shared Python module (both machines)
│       └── strafer_shared/
│           ├── constants.py           # Single source of truth for all robot params
│           ├── mecanum_kinematics.py  # Forward/inverse kinematics (NumPy)
│           └── policy_interface.py    # Policy contract: obs/action specs
│
├── Assets/                      # Robot USD assets
├── Scripts/                     # Asset processing & training scripts
├── IsaacLab/                    # NVIDIA Isaac Lab (submodule)
├── Makefile                     # Jetson helpers: make udev / make test / make lint
└── docs/
    ├── SIM_TO_REAL_PLAN.md          # Detailed deployment plan + phase status
    ├── SIM_TO_REAL_TUNING_GUIDE.md  # Actuator + sensor characterization guide
    ├── PHASE_5_VLM_INTEGRATION.md   # VLM (Qwen2.5-VL-3B) integration plan
    ├── WIRING_GUIDE.md              # Motor, encoder, RoboClaw, Jetson connections
    ├── D555_IMU_KERNEL_FIX.md       # RealSense HW clock drift fix for JetPack 6.x
    └── artifacts/                   # Images, videos
```

## Quick Start

### Simulation (Windows)

```powershell
# Activate virtual environment (required before every session)
& C:\Worspace\venv_isaac\Scripts\Activate.ps1

# Install extensions (one-time)
python -m pip install -e source/strafer_lab
python -m pip install -e source/strafer_shared

# List environments
python -c "import strafer_lab; import gymnasium as gym; print([e for e in gym.envs.registry.keys() if 'Strafer' in e])"
```

#### Training

All training commands run from `C:\Worspace\IsaacLab` with the venv activated.

```powershell
cd IsaacLab

# NoCam (proprioceptive-only, fastest iteration, recommended first)
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 512

# Depth (adds depth camera — cameras are auto-enabled)
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-Depth-v0 --num_envs 32

# Full (RGB + Depth, heaviest)
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-v0 --num_envs 32

# Headless (no GUI, faster training)
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 4096 --headless

# Resume from checkpoint
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --resume logs\rsl_rl\strafer_navigation\run_XXXXX\model_500.pt
```

#### Monitoring Training

```powershell
# In a separate terminal (with venv activated)
tensorboard --logdir logs\rsl_rl\strafer_navigation
# Open http://localhost:6006 in browser
```

#### Evaluate Trained Policy

```powershell
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Strafer-Nav-Real-NoCam-Play-v0 --num_envs 50
```

#### Run Tests

```powershell
cd IsaacLab
.\isaaclab.bat -p ..\source\strafer_lab\run_tests.py all
```

### Robot (Jetson Orin Nano)

```bash
# -- One-time setup --

# Install shared module and driver as editable packages
pip install -e source/strafer_shared
pip install -e source/strafer_ros/strafer_driver

# Set up ROS2 colcon workspace (symlinks into this repo)
mkdir -p ~/strafer_ws/src
ln -s $(pwd)/source/strafer_ros/strafer_driver ~/strafer_ws/src/
ln -s $(pwd)/source/strafer_ros/strafer_msgs  ~/strafer_ws/src/
cd ~/strafer_ws && colcon build --symlink-install && cd -

# Add to ~/.bashrc so every terminal finds the ROS2 packages
echo 'source ~/strafer_ws/install/setup.bash' >> ~/.bashrc
source ~/strafer_ws/install/setup.bash

# Install udev rules for RoboClaw auto-detection
make udev
```

```bash
# -- Running --

# Full bringup (driver + description + perception)
ros2 launch strafer_bringup base.launch.py

# Or start nodes individually:
# Terminal 1: Driver node
ros2 run strafer_driver roboclaw_node

# Terminal 2: Motion test
python3 source/strafer_ros/ros_test_motion.py --pattern forward --duration 3
python3 source/strafer_ros/ros_test_motion.py --pattern circle --speed 2.0

# Direct hardware test (no ROS2 needed)
python3 source/strafer_ros/test_motion_patterns.py --pattern forward --duration 3

# Run tests / lint
make test
make lint
```

## Key Design Decisions

**Monorepo**: Sim and real code share a single repo so that the observation ordering, normalization constants, and kinematics are always in sync. The `strafer_shared` module is the single source of truth.

**Two-contract architecture**: The policy contract (obs spec, action spec, normalization) lives in `strafer_shared.policy_interface` and is used by both the gym environment and the ROS2 inference node. The ROS topic contract (`cmd_vel`, `joint_states`, `odom`, etc.) is the deployment interface — identical whether the backend is real hardware or Isaac Sim via the ROS2 bridge.

**Custom driver over ros2_control**: The `strafer_driver` uses a custom Python node rather than ros2_control's `mecanum_drive_controller`. This keeps kinematics in `strafer_shared` (matching sim exactly) and avoids requiring a C++ hardware interface for RoboClaw serial communication.

**Sim-to-Real Contract**: Three realism presets (Ideal, Realistic, Robust) define motor dynamics, command delay, and per-sensor noise. The policy trained with `Realistic` or `Robust` presets transfers to real hardware without fine-tuning.

**No Arduino**: RoboClaw ST 2x45A controllers connect directly to the Jetson via USB. They handle motor PID, encoder counting, and provide a Python API.

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

## Documentation

- [Deployment Plan](docs/SIM_TO_REAL_PLAN.md) — Project goals, MVP path, ROS2 architecture, implementation phases
- [Sim-to-Real Tuning Guide](docs/SIM_TO_REAL_TUNING_GUIDE.md) — Actuator model mapping, sensor noise characterization, tuning procedure
- [VLM Integration Plan](docs/PHASE_5_VLM_INTEGRATION.md) — Qwen2.5-VL-3B architecture, fine-tuning, Jetson deployment
- [Wiring Guide](docs/WIRING_GUIDE.md) — Motor, encoder, RoboClaw, and Jetson connections
- [D555 IMU Kernel Fix](docs/D555_IMU_KERNEL_FIX.md) — RealSense HID sensor module build for JetPack 6.x
- [Simulation Extension](IsaacLab/README.md) — Isaac Lab framework documentation
- [ROS2 Development Guide](source/strafer_ros/CLAUDE.md) — Jetson-side driver/inference development context

## References

- [GoBilda Strafer Chassis](https://www.gobilda.com/strafer-chassis-kit-v4/)
- [GoBilda 5203 Motor](https://www.gobilda.com/5203-series-yellow-jacket-planetary-gear-motor-19-2-1-ratio-24mm-length-8mm-rex-shaft-312-rpm-3-3-5v-encoder/)
- [RoboClaw ST 2x45A](https://www.gobilda.com/roboclaw-st-2x45a-motor-controller/)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [ROS2 Humble](https://docs.ros.org/en/humble/)

## License

MIT
