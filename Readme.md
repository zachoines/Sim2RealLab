# Strafer Robot - Sim-to-Real

Sim-to-real development for the GoBilda Strafer mecanum robot.

This repository covers:

- Isaac Lab training on a Windows workstation
- Jetson ROS2 runtime for real-robot execution
- shared sim-to-real contracts in `strafer_shared`
- a workstation-hosted VLM grounding stack
- a planned workstation-hosted LLM planner with a Jetson-local executor

<p align="center">
  <img src="docs/artifacts/strafer_top.jpeg" alt="Strafer robot, top view" width="45%"/>
  <img src="docs/artifacts/strafer_side.jpeg" alt="Strafer robot, side view" width="45%"/>
</p>

## Project Direction

The current architecture is split into four layers:

1. `strafer_lab`
   - Isaac Lab environments, training, and evaluation
2. `strafer_shared`
   - shared physical constants, mecanum kinematics, and policy I/O contract
3. `strafer_ros`
   - Jetson-side robot runtime, sensing, navigation, TF, and safety-critical execution
4. `strafer_autonomy` and `strafer_vlm`
   - planner and VLM services hosted off-robot first, with the executor staying robot-local

The long-term end state is:

```text
user command
  -> planner service
  -> bounded MissionPlan
  -> Jetson executor
  -> grounding when needed
  -> robot-local projection and motion execution
  -> real hardware
```

The current MVP remains narrower:

```text
hardcoded or simple goal
  -> trained navigation policy
  -> robot-local execution
```

## Current State

### Simulation

The Isaac Lab environment is in good shape:

- 18 environment variants across realism and sensor modes
- mecanum kinematics and shared constants aligned with the robot
- sensor and actuator noise models for sim-to-real transfer
- PPO training pipeline for navigation policies

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
  <em>Test drive in Isaac Lab - <a href="docs/artifacts/strafer_isaac_lab_test_drive.mp4">watch video</a></em>
</p>

<p align="center">
  <a href="docs/artifacts/strafer_infinitygen_scene.mp4">
    <img src="docs/artifacts/strafer_infinitygen_thumbnail.png" alt="Click to view Infinigen scene video" width="50%"/>
  </a>
  <br/>
  <em>Robot in a procedurally generated Infinigen apartment - <a href="docs/artifacts/strafer_infinitygen_scene.mp4">watch video</a></em>
</p>

### Robot Runtime

The robot-side ROS stack already has substantial coverage:

- `strafer_driver`
  - RoboClaw interface, wheel commands, odometry, joint states, watchdog
- `strafer_perception`
  - RealSense integration, depth downsampling, timestamp fixing, IMU filtering
- `strafer_description`
  - URDF and TF tree
- `strafer_slam`
  - RTAB-Map launch and tuning
- `strafer_navigation`
  - Nav2 launch and parameterization
- `strafer_bringup`
  - layered bringup launch files and validation helpers

### Autonomy and VLM

The autonomy stack is now split intentionally:

- `strafer_autonomy`
  - owns planner-facing schemas, mission execution, and Jetson-side executor scaffolding
- `strafer_vlm`
  - owns Qwen grounding evaluation, training, and the future workstation grounding service

Current VLM status:

- workstation-first Qwen grounding is implemented and working on Windows
- live evaluation from a Windows-connected camera is working
- LoRA training and offline evaluation routines are implemented
- the LAN service boundary for the Jetson executor is planned but not yet implemented

### Pipeline Snapshot

| Area | Status | Notes |
|---|---|---|
| CAD to USD | Done | Asset import, rigging, hierarchy cleanup |
| Isaac Lab environments | Done | Realism presets and sensor variants are in place |
| Shared kinematics and policy contract | Done | `strafer_shared` is the sim-to-real boundary |
| Jetson ROS driver and perception | Done | Driver, perception, URDF, SLAM, and Nav2 config exist |
| RL policy training | In progress | Navigation training is active in Isaac Lab |
| `strafer_inference` runtime | Planned | Policy runtime on the Jetson is still to be implemented |
| `strafer_autonomy` executor | Scaffolded | Schemas, client stubs, command ingress, and mission runner exist |
| LLM planner service | Planned | Workstation-hosted planner service is the next major autonomy step |
| VLM grounding service | Planned | Current `strafer_vlm` tooling needs to be refactored into a service |

## Hardware

| Component | Model | Purpose |
|---|---|---|
| Workstation | Windows PC + NVIDIA GPU | Isaac Lab training, VLM grounding, planner hosting |
| Robot compute | Jetson Orin Nano | ROS2 runtime and robot-local execution |
| Camera | Intel RealSense D555 | RGB, depth, IMU |
| Motors | 4x GoBilda 5203 Yellow Jacket (19.2:1) | Mecanum drive |
| Motor controllers | 2x RoboClaw ST 2x45A | USB serial motor and encoder interface |
| Chassis | GoBilda Strafer v4 | 4-wheel mecanum platform |

## Repository Structure

```text
source/
  strafer_lab/         Isaac Lab simulation and training
  strafer_shared/      shared constants, kinematics, policy I/O
  strafer_ros/         Jetson ROS2 packages
  strafer_autonomy/    autonomy schemas, executor, planner/VLM clients
  strafer_vlm/         workstation grounding package and future grounding service

docs/
  SIM_TO_REAL_PLAN.md
  SIM_TO_REAL_TUNING_GUIDE.md
  STRAFER_AUTONOMY_ROADMAP.md
  STRAFER_AUTONOMY_ROS.md
  STRAFER_AUTONOMY_INTERFACES.md
  STRAFER_AUTONOMY_COMMAND_INGRESS.md
  STRAFER_AUTONOMY_LOCAL_DEVELOPMENT.md
  STRAFER_AUTONOMY_DEPLOYMENT_MODES.md
  STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md
  STRAFER_AUTONOMY_SYSTEMS_OVERVIEW.md
  STRAFER_AUTONOMY_VLM_GROUNDING.md
  STRAFER_AUTONOMY_LLM_PLANNER.md
```

## Quick Start

### Simulation on Windows

```powershell
# Activate the Isaac Lab environment
& C:\Workspace\venv_isaac\Scripts\Activate.ps1

# Install local packages
python -m pip install -e source/strafer_lab
python -m pip install -e source/strafer_shared

# List Strafer environments
python -c "import strafer_lab; import gymnasium as gym; print([e for e in gym.envs.registry.keys() if 'Strafer' in e])"
```

#### Training

Run training from `C:\Workspace\Sim2RealLab\IsaacLab`:

```powershell
cd IsaacLab

# Recommended first run: NoCam
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 512

# Depth
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-Depth-v0 --num_envs 32

# Full RGB + depth
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-v0 --num_envs 32

# Headless
.\isaaclab.bat -p ..\Scripts\train_strafer_navigation.py --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 4096 --headless
```

#### Monitoring

```powershell
tensorboard --logdir logs\rsl_rl\strafer_navigation
```

#### Test or play

```powershell
cd IsaacLab
.\isaaclab.bat -p ..\source\strafer_lab\run_tests.py all
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Strafer-Nav-Real-NoCam-Play-v0 --num_envs 50
```

### VLM Grounding on Windows

```powershell
python -m pip install -e source/strafer_vlm
python -m pip install -e "source/strafer_vlm[qwen,live]"
```

```powershell
# Single-image grounding smoke test
python -m strafer_vlm.test_qwen25vl_grounding `
  --image docs\artifacts\strafer_top.jpeg `
  --prompt "the robot chassis"
```

```powershell
# Live grounding against a local camera
python -m strafer_vlm.live_qwen25vl_grounding --source 0 --prompt "Locate: the robot chassis"
```

### Robot Runtime on Jetson

```bash
# Install shared code and Python-side ROS packages as needed
pip install -e source/strafer_shared
pip install -e source/strafer_ros/strafer_driver

# Create a ROS workspace
mkdir -p ~/strafer_ws/src

# Symlink the ROS packages you need from source/strafer_ros/ into ~/strafer_ws/src/
# Then build:
cd ~/strafer_ws && colcon build --symlink-install && cd -

source ~/strafer_ws/install/setup.bash
```

```bash
# Base bringup
ros2 launch strafer_bringup base.launch.py
```

For the current Jetson-side package inventory, expected topics, and implementation status, use:

- `docs/STRAFER_AUTONOMY_ROS.md`

## Key Design Decisions

### Monorepo with shared contracts

Sim, ROS, autonomy, and VLM code stay in one repository so the core contracts do not drift.

### `strafer_shared` is the sim-to-real boundary

The real robot and Isaac Lab must agree on:

- physical constants
- mecanum kinematics
- policy observation and action contract

### Keep execution local

Safety-critical execution remains on the robot.

Planner and VLM services may move between workstation and cloud without changing the robot-side execution boundary.

### Keep the VLM narrow

`strafer_vlm` does semantic grounding only.

Depth projection, TF transforms, reachability, and motion execution remain robot-local.

### Planner and executor stay separate

The LLM planner is a text-to-plan service.

The Jetson executor owns mission state, validation, retries, cancel, and robot-facing control.

## Environments

| Realism | Sensors | Train ID | Obs dims |
|---|---|---|---|
| Ideal | Full (RGB + depth) | `Isaac-Strafer-Nav-v0` | 19215 |
| Ideal | Depth-only | `Isaac-Strafer-Nav-Depth-v0` | 4815 |
| Ideal | NoCam | `Isaac-Strafer-Nav-NoCam-v0` | 15 |
| Realistic | Full | `Isaac-Strafer-Nav-Real-v0` | 19215 |
| Realistic | Depth-only | `Isaac-Strafer-Nav-Real-Depth-v0` | 4815 |
| Realistic | NoCam | `Isaac-Strafer-Nav-Real-NoCam-v0` | 15 |
| Robust | Full | `Isaac-Strafer-Nav-Robust-v0` | 19215 |
| Robust | Depth-only | `Isaac-Strafer-Nav-Robust-Depth-v0` | 4815 |
| Robust | NoCam | `Isaac-Strafer-Nav-Robust-NoCam-v0` | 15 |

## Documentation

Core docs:

- [Sim-to-Real Plan](docs/SIM_TO_REAL_PLAN.md)
- [Sim-to-Real Tuning Guide](docs/SIM_TO_REAL_TUNING_GUIDE.md)
- [Strafer Autonomy Roadmap](docs/STRAFER_AUTONOMY_ROADMAP.md)
- [Strafer Autonomy ROS](docs/STRAFER_AUTONOMY_ROS.md)
- [Strafer Autonomy Interfaces](docs/STRAFER_AUTONOMY_INTERFACES.md)
- [Strafer Autonomy Command Ingress](docs/STRAFER_AUTONOMY_COMMAND_INGRESS.md)
- [Strafer Autonomy Local Development](docs/STRAFER_AUTONOMY_LOCAL_DEVELOPMENT.md)
- [Strafer Autonomy Deployment Modes](docs/STRAFER_AUTONOMY_DEPLOYMENT_MODES.md)
- [Strafer Autonomy MVP Runtime Decision](docs/STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md)
- [Strafer Autonomy Systems Overview](docs/STRAFER_AUTONOMY_SYSTEMS_OVERVIEW.md)
- [Strafer Autonomy VLM Grounding](docs/STRAFER_AUTONOMY_VLM_GROUNDING.md)
- [Strafer Autonomy LLM Planner](docs/STRAFER_AUTONOMY_LLM_PLANNER.md)

Supporting docs:

- [Wiring Guide](docs/WIRING_GUIDE.md)
- [D555 IMU Kernel Fix](docs/D555_IMU_KERNEL_FIX.md)
- [Isaac Lab README](IsaacLab/README.md)

## References

- [GoBilda Strafer Chassis](https://www.gobilda.com/strafer-chassis-kit-v4/)
- [GoBilda 5203 Motor](https://www.gobilda.com/5203-series-yellow-jacket-planetary-gear-motor-19-2-1-ratio-24mm-length-8mm-rex-shaft-312-rpm-3-3-5v-encoder/)
- [RoboClaw ST 2x45A](https://www.gobilda.com/roboclaw-st-2x45a-motor-controller/)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [ROS 2 Humble](https://docs.ros.org/en/humble/)

## License

MIT
