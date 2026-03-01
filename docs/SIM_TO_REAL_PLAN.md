# Strafer Robot: Sim-to-Real Deployment Plan

---

## 0. Project Goals & MVP

### Vision

A human-robot interaction loop where a user gives natural language commands ("go to the kitchen, wait, then come back"). A VLM decodes the command alongside the robot's current state (Nav2 map, RGB/depth, IMU) and outputs a sequence of goal poses or skills. The robot executes each using trained low-level RL controllers until the command is fulfilled.

### MVP: RL Navigation to Goal

An RL policy trained in Isaac Lab navigates to a hardcoded goal pose while avoiding obstacles. The **same model** runs in both simulation and on the real robot. No VLM, no natural language, no skills.

**MVP success criteria**:
- Policy trained in Isaac Lab with `Realistic` noise preset
- Same `.pt` model loads and runs on both platforms
- Robot reaches goal within 0.3m accuracy
- Obstacle avoidance via depth or proprioceptive policy

### Two Evaluation Paths

The project has two distinct ways to evaluate a trained policy:

```
PATH 1: Gym Eval (fast, during training)                PATH 2: ROS Eval (deployment validation)
─────────────────────────────────────                   ──────────────────────────────────────
Isaac Lab env.step(action) → obs                        Inference node reads sensor topics
No ROS. Pure Python/PyTorch.                            Assembles obs → runs policy → cmd_vel
Thousands of envs in parallel.                          Works against real HW or Isaac Sim
                                                        via ROS2 bridge.
Used for: training, quick checks                        Used for: integration test, real deploy
```

Both paths reference the same **policy contract** defined in `strafer_shared.policy_interface`:
- **Observation spec**: field ordering, dimensions, normalization scales
- **Action spec**: dimensions, denormalization to physical velocities
- **Model loading**: `.pt` (TorchScript) now, `.onnx` (ONNX/TensorRT) later

### Two Contracts

| Contract | What it defines | Where it lives | Who uses it |
|----------|----------------|----------------|-------------|
| **Policy contract** | Obs fields, ordering, scales, action denorm | `strafer_shared.policy_interface` | Gym env config, ROS inference node |
| **ROS topic contract** | Topics, message types, rates, TF frames | `source/strafer_ros/CLAUDE.md`, launch files | Driver node, inference node, Nav2 |

The policy contract is the "inner" contract (model I/O). The ROS topic contract is the "outer" contract (system integration). The inference node bridges the two: it subscribes to ROS topics, calls `assemble_observation()` from the policy contract, runs the model, calls `interpret_action()`, and publishes `cmd_vel`.

### Roadmap Beyond MVP

| Phase | Goal Interface | Policy | Notes |
|-------|---------------|--------|-------|
| **MVP** | Hardcoded goal pose | RL nav-to-goal (.pt) | Current target |
| **Nav2** | Waypoints from map/planner | RL + obstacle avoidance | Nav2 global plan → RL local control |
| **VLM** | NL command → VLM → poses/skills | Multi-task RL | Train VLM on Nav2 trajectory data (behavioral cloning) |

---

## Context

The Strafer simulation environment in Isaac Lab is feature-complete: 18 Gym environments, mecanum kinematics, realistic sensor/noise models, and a PPO training pipeline. The next step is deploying trained policies onto real hardware. This plan covers hardware wiring, ROS2 software architecture, policy export/inference, and a full autonomy stack (SLAM + Nav2).

**Hardware**: Jetson Orin Nano, Intel RealSense D555, 4x GoBilda 5203 motors, 2x RoboClaw ST 2x45A. No Arduino needed -- RoboClaws connect directly to Jetson via USB.

---

## 1. Platform Recommendation

| Component | Version | Why |
|-----------|---------|-----|
| **JetPack** | 6.2 (L4T R36.x) | CUDA 12.6, TensorRT 10.x, Ubuntu 22.04. Already flashed. |
| **Board** | Jetson Orin Nano | 8GB RAM, 1024-core Ampere GPU, 40 TOPS AI performance |
| **ROS2** | Humble Hawksbill | LTS on Ubuntu 22.04, broad Nav2/RealSense support |
| **Python** | 3.10 | Ships with Ubuntu 22.04, onnxruntime-gpu compatible |

JetPack 6.2 is already flashed on the Orin Nano. Ubuntu 22.04 base pairs natively with ROS2 Humble.

---

## 2. Hardware Wiring

```
Jetson Orin Nano USB 3.1 #1  -->  RealSense D555 (RGB + Depth + IMU)
Jetson Orin Nano USB 3.1 #2  -->  RoboClaw #1 (addr 0x80): FL motor + FR motor + encoders
Jetson Orin Nano USB 2.0     -->  RoboClaw #2 (addr 0x81): RL motor + RR motor + encoders
```

**RoboClaw #1 (0x80 - Front Axle)**: M1=FL (wheel_1), M2=FR (wheel_2), EN1=FL encoder, EN2=FR encoder
**RoboClaw #2 (0x81 - Rear Axle)**: M1=RL (wheel_3), M2=RR (wheel_4), EN1=RL encoder, EN2=RR encoder

**Power**: 12V 3S/4S LiPo -> RoboClaws (motor power) + boost/buck converter -> Jetson Orin Nano (19V barrel jack). RealSense powered via USB 3.x.

**Udev rules**: `source/strafer_ros/99-strafer.rules` creates persistent symlinks `/dev/roboclaw0` (front, addr 0x80) and `/dev/roboclaw1` (rear, addr 0x81). Install with `make udev`.

**Motor direction**: `wheel_axis_signs = [-1, 1, -1, 1]` (FL, FR, RL, RR) from [strafer_env_cfg.py:183](source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py#L183). Apply sign correction in software, not by swapping wires.

---

## 3. Repository Structure (Monorepo)

```
<repo-root>/
├── source/
│   ├── strafer_lab/             # Isaac Lab simulation (Windows workstation)
│   ├── strafer_ros/             # ROS2 packages (Jetson Orin Nano)
│   │   ├── CLAUDE.md            # Agent prompt for Jetson-side Claude Code
│   │   ├── strafer_msgs/
│   │   ├── strafer_description/
│   │   ├── strafer_driver/
│   │   ├── strafer_perception/
│   │   ├── strafer_inference/
│   │   ├── strafer_slam/
│   │   ├── strafer_navigation/
│   │   └── strafer_bringup/
│   └── strafer_shared/          # Shared Python module (both machines)
│       └── strafer_shared/
│           ├── constants.py           # Single source of truth for all robot params
│           ├── mecanum_kinematics.py  # Forward/inverse kinematics (NumPy)
│           └── policy_interface.py    # Policy contract: obs/action specs, model loading
├── docs/
│   └── SIM_TO_REAL_PLAN.md
└── Assets/
```

On the Jetson, symlink into a colcon workspace:
```bash
mkdir -p ~/strafer_ws/src
ln -s ~/strafer/source/strafer_ros/* ~/strafer_ws/src/
ln -s ~/strafer/source/strafer_shared ~/strafer_ws/src/
```

### ROS2 Package Details

### strafer_driver: `roboclaw_node`

**Core node** -- bridges ROS2 and hardware.

| Direction | Topic | Type | Rate |
|-----------|-------|------|------|
| Sub | `/strafer/cmd_vel` | `geometry_msgs/Twist` | -- |
| Pub | `/strafer/joint_states` | `sensor_msgs/JointState` | 50 Hz |
| Pub | `/strafer/odom` | `nav_msgs/Odometry` | 50 Hz |
| TF | `odom` -> `base_link` | -- | 50 Hz |

Mecanum kinematics (Twist -> wheel angular velocities -> encoder ticks/sec -> RoboClaw `SpeedM1`/`SpeedM2`) uses a **shared `mecanum_kinematics.py`** module with constants matching the simulation exactly.

Safety: motor watchdog stops all wheels if no `cmd_vel` received within 500ms.

### strafer_perception: `depth_downsampler`, `timestamp_fixer`

| Direction | Topic | Type | Rate |
|-----------|-------|------|------|
| Sub | `/d555/depth/image_rect_raw` | `sensor_msgs/Image` (16UC1, mm) | 30 Hz |
| Pub | `/d555/depth/downsampled` | `sensor_msgs/Image` (32FC1, m, 80×60) | 30 Hz |

`timestamp_fixer` re-stamps all D555 streams from hardware clock to ROS system time, fixing a known JetPack 6.x sync issue (see `docs/D555_IMU_KERNEL_FIX.md`).

### strafer_inference: `policy_inference_node` *(planned)*

| Direction | Topic | Type | Rate |
|-----------|-------|------|------|
| Sub | `/d555/imu/filtered` | `sensor_msgs/Imu` | 200 Hz |
| Sub | `/strafer/joint_states` | `sensor_msgs/JointState` | 50 Hz |
| Sub | `/d555/depth/downsampled` | `sensor_msgs/Image` (32FC1, m, 80×60) | 30 Hz |
| Sub | `/strafer/goal` | `geometry_msgs/PoseStamped` | -- |
| Pub | `/strafer/cmd_vel` | `geometry_msgs/Twist` | 30 Hz |

Assembles observation vector via `assemble_observation(raw, variant)`, runs `.pt` or `.onnx` model, calls `interpret_action()`, publishes `cmd_vel`.

---

## 4. Controller Abstraction: Shared Kinematics

A pure-Python `mecanum_kinematics.py` module shared between sim and real, containing:

- Constants from simulation: `WHEEL_RADIUS=0.048`, `WHEEL_BASE=0.336`, `TRACK_WIDTH=0.4284`, `MAX_WHEEL_ANGULAR_VEL=32.67`, `ENCODER_PPR=537.7`, `WHEEL_AXIS_SIGNS=[-1,1,-1,1]`
- Forward kinematics: `[vx, vy, omega]` -> `[w_fl, w_fr, w_rl, w_rr]` (rad/s)
- Inverse kinematics: wheel velocities -> body velocity (for odometry)
- Unit conversions: rad/s <-> ticks/sec

The kinematic matrix replicates [actions.py:166-171](source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L166-L171) exactly.

**What the real robot does NOT replicate from simulation**: motor dynamics filter (real motor has physical inertia), command delay (real USB latency exists naturally), slew rate limiting (RoboClaw PID handles this). The policy was trained with these dynamics, so it produces compatible commands.

---

## 5. Observation Assembly (Critical for sim-to-real transfer)

The observation spec is defined in `strafer_shared.policy_interface` as the single source of truth. Both the Isaac Lab env config and the ROS2 inference node reference it. The `assemble_observation(raw, variant)` function normalizes and concatenates raw sensor values into the policy's expected input.

### NoCam variant (15 dims) -- `PolicyVariant.NOCAM`

| Index | Field | Key | Real Source | Scale |
|-------|-------|-----|-------------|-------|
| 0-2 | IMU linear acceleration | `imu_accel` | D555 IMU (m/s^2) | 1/156.96 |
| 3-5 | IMU angular velocity | `imu_gyro` | D555 IMU (rad/s) | 1/34.9 |
| 6-9 | Wheel encoder velocities | `encoder_vels_ticks` | RoboClaw `ReadSpeedM1/M2` (ticks/s) | 1/3000.0 |
| 10-11 | Goal position relative | `goal_relative` | TF base_link→goal (m) | 1.0 |
| 12-14 | Last action | `last_action` | Previous policy output | 1.0 |

### Depth variant (4815 dims) -- `PolicyVariant.DEPTH`

All NoCam fields (indices 0-14) plus:

| Index | Field | Key | Real Source | Scale |
|-------|-------|-----|-------------|-------|
| 15-4814 | Depth image | `depth_image` | D555 depth, 80x60, float32 meters | 1/6.0 |

### Usage

```python
from strafer_shared.policy_interface import assemble_observation, PolicyVariant

raw = {
    "imu_accel": imu_data.accel,               # (3,) m/s²
    "imu_gyro": imu_data.gyro,                  # (3,) rad/s
    "encoder_vels_ticks": [fl, fr, rl, rr],     # (4,) ticks/s -- raw from RoboClaw
    "goal_relative": [gx, gy],                  # (2,) meters in robot frame
    "last_action": prev_action,                 # (3,) normalized [-1, 1]
}
obs = assemble_observation(raw, PolicyVariant.NOCAM)  # → (15,) float32
```

Encoder velocities from `JointState` (rad/s) must be converted to ticks/sec via `RADIANS_TO_ENCODER_TICKS = 85.57` from [observations.py:32](source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py#L32) before passing to `assemble_observation`.

---

## 6. Policy Export Pipeline

### MVP: TorchScript (.pt)

1. **Train**: `Isaac-Strafer-Nav-Real-NoCam-v0` (15-dim obs, start without camera dependency)
2. **Export**: Use `export_policy_as_jit()` from Isaac Lab's `isaaclab_rl` module to produce a `.pt` (TorchScript) file
3. **Load**: `strafer_shared.policy_interface.load_policy("model.pt", PolicyVariant.NOCAM)` handles loading on both platforms
4. **Measure latency**: Run `benchmark_policy()` to measure inference time on both workstation and Jetson

### Later: ONNX + TensorRT

When model complexity increases (depth images, larger networks):
1. **Export**: `export_policy_as_onnx()` → `.onnx` file
2. **Optimize**: TensorRT FP16 on Jetson via `trtexec` or ONNX Runtime TensorRT EP
3. **Infer**: `load_policy("model.onnx", variant)` auto-detects format and uses ONNX Runtime

The `load_policy` function in `strafer_shared.policy_interface` handles both `.pt` and `.onnx` formats transparently. Switching from `.pt` to `.onnx` requires no code changes in either the gym eval script or the ROS inference node.

---

## 7. SLAM & Nav2 Integration

**SLAM**: RTAB-Map (`ros-humble-rtabmap-ros`) with D555 RGB-D. Tuned config in `strafer_slam/config/rtabmap_params.yaml`: 2 Hz loop closure, ORB features, 0.05m grid cells, g2o optimizer, 2D SLAM. Produces `/rtabmap/map` (`OccupancyGrid`) and `map→odom` TF.

**Odometry**: Wheel encoder odometry (`strafer_driver`) provides `odom→base_link` TF at 50 Hz. RTAB-Map uses this as the odometry source (no EKF needed for MVP).

**IMU filtering**: `imu_filter_madgwick` node fuses D555 IMU into `/d555/imu/filtered` (orientation-corrected). Used by inference node.

**Nav2 controller**: MPPI (`nav2_mppi_controller`) with the `OmniMotionModel` for full mecanum motion. Params in `strafer_navigation/config/nav2_params.yaml`.

**Nav2 operating modes**:
- **Mode 1 (MVP)**: Pure RL policy. `policy_inference_node` publishes directly to `/strafer/cmd_vel`. No Nav2. Goal is a hardcoded `PoseStamped`.
- **Mode 2**: Nav2 with RL as local controller plugin. Nav2 handles global planning + behavior trees; RL policy handles low-level velocity control.
- **Mode 3 (future)**: VLM decodes NL commands → Nav2 goal poses → RL local controller.

---

## 8. Implementation Phases

### Phase 1: Jetson Setup + Hardware Bring-Up ✅
- [x] Flash JetPack 6.2 -- Orin Nano on network, SSH accessible
- [x] VS Code Remote-SSH workspace (Windows ↔ Jetson)
- [x] Install ROS2 Humble
- [x] Udev rules: `99-strafer.rules` → `/dev/roboclaw0`, `/dev/roboclaw1`; install with `make udev`
- [x] RoboClaw serial communication verified, PID auto-set on startup
- [x] All 4 motors verified: spin direction, encoder counts, motion patterns
- [x] librealsense2 installed, D555 RGB + depth + IMU streams verified
- [x] D555 HW clock drift diagnosed and fixed (`timestamp_fixer`, see `docs/D555_IMU_KERNEL_FIX.md`)
- **Evidence**: `source/strafer_ros/test_motion_patterns.py`, `source/strafer_ros/test_d555_camera.py`, `source/strafer_ros/tune_pid.py`

### Phase 2: ROS2 Driver + Perception ✅
- [x] `strafer_msgs`: package scaffolding created
- [x] `strafer_driver`: `roboclaw_node` -- auto-detect ports, auto-PID, cmd_vel/odom/joint_states/TF, watchdog, diagnostics
- [x] `strafer_perception`: `depth_downsampler` (16UC1→32FC1 80×60), `timestamp_fixer` (D555 HW clock sync), `imu_filter_madgwick`
- [x] Shared `mecanum_kinematics.py` in `strafer_shared`
- **Evidence**: `source/strafer_ros/ros_test_motion.py`, `source/strafer_ros/ros_test_perception.py`

### Phase 3: URDF + TF Tree ✅
- [x] `strafer_description`: URDF/xacro with dimensions from `strafer_shared.constants`, `robot_state_publisher` launch
- [x] TF tree: `map → odom → base_link → {chassis, d555, wheel_1..4}`
- [ ] STL meshes from USD assets (visual-only, not blocking)
- **Test**: `colcon test --packages-select strafer_description` (URDF parse + TF validation)

### Phase 4: Policy Export + Inference ← **Current**
- [ ] Train policy on workstation: `Isaac-Strafer-Nav-Real-NoCam-v0` (15-dim obs, no camera)
- [ ] Export to TorchScript (`.pt`) via `export_policy_as_jit()`
- [ ] Create `strafer_inference` package with `policy_inference_node`:
  - Sub: `/d555/imu/filtered`, `/strafer/joint_states`, `/d555/depth/downsampled`, `/strafer/goal`
  - Pub: `/strafer/cmd_vel` at 30 Hz
  - Logic: `assemble_observation()` → model forward → `interpret_action()` → Twist
- [ ] Benchmark inference latency: target <5ms on Jetson
- **Test**: hardcoded goal → robot drives toward it and stops within 0.3m

### Phase 5: Sim-to-Real Tuning + Integration Testing
- [ ] Characterize all sensors (IMU noise, encoder noise) -- see [Sim-to-Real Tuning Guide](SIM_TO_REAL_TUNING_GUIDE.md)
- [x] RoboClaw PID tuned: P=15000, I=750, D=0, QPPS=2796 (auto-set on every startup)
- [ ] Verify measured motor τ falls within Robust envelope [20-100ms]
- [ ] Waypoint following accuracy (<0.3m error)
- [ ] Square test: compare real vs. sim trajectories
- **Test**: 10-waypoint course, 30-minute endurance run

### Phase 6: SLAM + Nav2 Integration
- [x] `strafer_slam`: RTAB-Map config tuned (2 Hz loop closure, 0.05m grid, g2o optimizer)
- [x] `strafer_navigation`: Nav2 + MPPI OmniMotionModel config
- [x] `strafer_bringup`: layered launch files (base / perception / slam / navigation) + `ValidateDrive` smoke test
- [ ] End-to-end test: build map of room, localize, send Nav2 goal
- [ ] Integrate RL policy as Nav2 local controller plugin
- **Test**: autonomous navigation in mapped environment with obstacle avoidance

---

## 9. Critical Source Files

| File | What to reference |
|------|-------------------|
| [policy_interface.py](source/strafer_shared/strafer_shared/policy_interface.py) | **Policy contract**: obs specs, action specs, `assemble_observation()`, `load_policy()` |
| [constants.py](source/strafer_shared/strafer_shared/constants.py) | All robot physical constants, PID gains, normalization scales |
| [mecanum_kinematics.py](source/strafer_shared/strafer_shared/mecanum_kinematics.py) | Kinematic matrix, forward/inverse kinematics |
| [roboclaw_node.py](source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py) | ROS2 driver: 50 Hz loop, watchdog, diagnostics, odom integration |
| [roboclaw_interface.py](source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py) | Packet serial protocol, CRC-16, retry logic |
| [depth_downsampler.py](source/strafer_ros/strafer_perception/strafer_perception/depth_downsampler.py) | 16UC1→32FC1 downsampler, clip [0.4, 6.0]m |
| [strafer.urdf.xacro](source/strafer_ros/strafer_description/urdf/strafer.urdf.xacro) | URDF: link dimensions, TF frames, wheel joints |
| [rtabmap_params.yaml](source/strafer_ros/strafer_slam/config/rtabmap_params.yaml) | SLAM: loop closure rate, grid resolution, ORB features |
| [nav2_params.yaml](source/strafer_ros/strafer_navigation/config/nav2_params.yaml) | Nav2: MPPI OmniMotionModel, velocity limits, costmaps |
| [actions.py](source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py) | Sim kinematic matrix (L166-171), motor dynamics filter |
| [strafer_env_cfg.py](source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py) | Sim obs order, normalization, camera config |
| [sim_real_cfg.py](source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py) | REAL_ROBOT_CONTRACT and ROBUST_TRAINING_CONTRACT |
| [strafer.py](source/strafer_lab/strafer_lab/assets/strafer.py) | Robot ArticulationCfg, DCMotorCfg, joint names |
