# Strafer Robot -- ROS2 Workspace

You are working on the real-hardware side of a sim-to-real robotics project. A mecanum-wheeled robot (GoBilda Strafer chassis) has been fully modeled in NVIDIA Isaac Lab on a separate Windows workstation. Your job is to build and maintain the ROS2 packages that run on the Jetson Orin Nano to control the physical robot.

## Project Architecture

This is a **monorepo**. The full repository lives at the root (one level above `source/`). Key paths:

```
<repo-root>/
├── source/strafer_lab/        # Isaac Lab simulation (Windows workstation -- DO NOT MODIFY from Jetson)
├── source/strafer_ros/        # ROS2 packages (THIS WORKSPACE -- runs on Jetson)
├── source/strafer_shared/     # Shared Python module (used by BOTH sim and real)
├── docs/                      # Project plans and documentation
│   └── SIM_TO_REAL_PLAN.md    # Master deployment plan -- READ THIS FIRST
└── Assets/                    # USD robot assets, STL meshes
```

**IMPORTANT**: The simulation code in `source/strafer_lab/` is the source of truth for physics parameters, observation ordering, and normalization constants. Never modify it from the Jetson side. If you need to change shared values, update `source/strafer_shared/` and coordinate.

## Hardware

- **Compute**: NVIDIA Jetson Orin Nano, JetPack 6.2, Ubuntu 22.04
- **Camera**: Intel RealSense D555 (RGB + depth + BMI055 IMU) via USB 3.x
- **Motors**: 4x GoBilda 5203 Yellow Jacket (19.2:1, 312 RPM, quadrature encoder 537.7 PPR)
- **Motor Controllers**: 2x RoboClaw ST 2x45A via USB serial
  - Front (addr 0x80): FL motor (M1) + FR motor (M2)
  - Rear  (addr 0x81): RL motor (M1) + RR motor (M2)

## ROS2 Stack (Humble)

```
source/strafer_ros/
├── strafer_msgs/          # Custom messages: WheelVelocities, EncoderTicks, PolicyAction
├── strafer_description/   # URDF/xacro, TF frames, robot_state_publisher
├── strafer_driver/        # RoboClaw node: cmd_vel -> motors, encoders -> JointState + odom
├── strafer_perception/    # RealSense launch wrapper + depth downsampler (80x60)
├── strafer_inference/     # ONNX/TensorRT policy inference node
├── strafer_slam/          # RTAB-Map configuration
├── strafer_navigation/    # Nav2 integration, behavior trees
└── strafer_bringup/       # Composed launch files
```

### Building

```bash
# On the Jetson, symlink into a colcon workspace:
mkdir -p ~/strafer_ws/src
ln -s ~/strafer/source/strafer_ros/* ~/strafer_ws/src/
ln -s ~/strafer/source/strafer_shared ~/strafer_ws/src/

cd ~/strafer_ws
colcon build --symlink-install
source install/setup.bash
```

## Shared Module: `strafer_shared`

`source/strafer_shared/` is a pip-installable Python package containing:

- **`constants.py`** -- All robot physical constants, encoder specs, normalization scales, RoboClaw addresses. These are the single source of truth; the simulation references identical values.
- **`mecanum_kinematics.py`** -- Forward/inverse kinematics for mecanum wheels. Pure NumPy, no PyTorch. The kinematic matrix exactly matches the simulation's `MecanumWheelAction`.
- **`policy_interface.py`** -- Policy contract: observation/action specs, `assemble_observation()`, `interpret_action()`, `action_to_wheel_ticks()`, `load_policy()`. This is the bridge between sensor data and the trained model.

Install it on the Jetson with: `pip install -e source/strafer_shared`

**Always import constants from `strafer_shared` rather than hardcoding values.** This prevents sim-real drift.

### Two Contracts

This project uses a two-contract architecture (see `docs/SIM_TO_REAL_PLAN.md` Section 0):

1. **Policy contract** (`strafer_shared.policy_interface`) -- defines obs field ordering, normalization scales, action denormalization. Used by the gym env config AND the ROS inference node. This is the "inner" contract.
2. **ROS topic contract** (this file + launch files) -- defines topics, message types, rates, TF frames. This is the "outer" contract. The inference node bridges the two.

## Critical Sim-to-Real Contract

The trained RL policy expects observations in a specific order and scale. The observation spec is defined in `strafer_shared.policy_interface.PolicyVariant` as the single source of truth. **Do NOT manually assemble or normalize observations** -- use `assemble_observation(raw, variant)` instead.

Summary for `PolicyVariant.NOCAM` (15 dims):

| Index | Field | Raw dict key | Scale |
|-------|-------|-------------|-------|
| 0-2 | IMU linear acceleration (m/s²) | `imu_accel` | × 1/156.96 |
| 3-5 | IMU angular velocity (rad/s) | `imu_gyro` | × 1/34.9 |
| 6-9 | Wheel encoder velocities (ticks/s) | `encoder_vels_ticks` | × 1/3000.0 |
| 10-11 | Goal position relative (m) | `goal_relative` | × 1.0 |
| 12-14 | Last action (normalized) | `last_action` | × 1.0 |

Encoder velocities from JointState (rad/s) must be converted to ticks/sec via `RADIANS_TO_ENCODER_TICKS ≈ 85.57` before passing to `assemble_observation()`.

For action output, use `interpret_action(action)` → `(vx, vy, omega)` or `action_to_wheel_ticks(action)` → `(4,)` ticks/sec array for direct RoboClaw commands.

## Motor Direction

`wheel_axis_signs = [-1, 1, -1, 1]` for [FL, FR, RL, RR]. These signs are applied **inside** `strafer_shared.mecanum_kinematics` functions (`twist_to_wheel_velocities`, `encoder_ticks_to_body_velocity`, etc.). **Do NOT apply signs again in the driver node** — doing so silently inverts two wheels.

Similarly, pass **raw** RoboClaw `ReadSpeedM1`/`ReadSpeedM2` tick rates directly to `encoder_ticks_to_body_velocity()` without any sign manipulation. The function handles sign correction internally. If a motor or encoder direction is physically wrong, fix it by swapping wires (see `docs/WIRING_GUIDE.md` Section 6), not in code.

## Key Topics

| Topic | Type | Publisher | Rate | Notes |
|-------|------|-----------|------|-------|
| `/strafer/cmd_vel` | `geometry_msgs/Twist` | inference / teleop | 30 Hz | input to driver |
| `/strafer/joint_states` | `sensor_msgs/JointState` | driver node | 50 Hz | wheel rad/s + cumulative rad |
| `/strafer/odom` | `nav_msgs/Odometry` | driver node | 50 Hz | odom→base_link |
| `/strafer/goal` | `geometry_msgs/PoseStamped` | user / Nav2 / VLM | -- | input to inference; frame: `map` |
| `/strafer/vlm_status` | `std_msgs/String` | vlm_goal_node | -- | `idle`/`detecting`/`navigating`/`goal_reached` |
| `/d555/imu/filtered` | `sensor_msgs/Imu` | madgwick node | 200 Hz | orientation-corrected |
| `/d555/depth/image_rect_raw` | `sensor_msgs/Image` | realsense node | 30 Hz | 16UC1, mm |
| `/d555/depth/downsampled` | `sensor_msgs/Image` | depth_downsampler | 30 Hz | 32FC1, m, 80×60 |
| `/d555/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | realsense node | 30 Hz | 16UC1, mm; depth in RGB frame; requires `align_depth.enable: true` |
| `/d555/color/image_sync` | `sensor_msgs/Image` | timestamp_fixer | 30 Hz | HW-clock-corrected |
| `/d555/color/camera_info_sync` | `sensor_msgs/CameraInfo` | timestamp_fixer | 30 Hz | HW-clock-corrected intrinsics |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | driver node | 1 Hz | RoboClaw connection state |

## Conventions

- **ROS2 Humble** -- use `rclpy` for Python nodes, `ament_python` or `ament_cmake` build type
- **Python 3.10** -- type hints encouraged, f-strings preferred
- Follow ROS2 naming: `snake_case` for topics/nodes/params, `CamelCase` for message types
- All physical constants come from `strafer_shared.constants` -- never hardcode
- Test with `colcon test`; use `launch_testing` for integration tests
- Udev rules go in `/etc/udev/rules.d/99-strafer.rules` for persistent device names

## Package Status

| Package | Status | Notes |
|---------|--------|-------|
| `strafer_msgs` | Done | Package scaffolding; add `SetCommand.srv` for Phase 5 |
| `strafer_driver` | Done | `roboclaw_node`: auto-detect, auto-PID, 50 Hz loop, watchdog, diagnostics |
| `strafer_perception` | Done | `depth_downsampler` + `timestamp_fixer` (D555 HW clock sync) + madgwick IMU filter |
| `strafer_description` | Done | URDF/xacro, all dims from `strafer_shared.constants`, `robot_state_publisher` |
| `strafer_slam` | Config done | RTAB-Map params tuned; end-to-end test pending |
| `strafer_navigation` | Config done | Nav2 + MPPI OmniMotionModel; end-to-end test pending |
| `strafer_bringup` | Done | Layered launch files + `ValidateDrive` smoke test |
| `strafer_inference` | **Planned** | Policy inference node -- current target (Phase 4) |
| `strafer_vlm` | **Planned** | Qwen2.5-VL-3B visual grounding + NL command → goal pose (Phase 5) |

## Current Phase: Phase 4 -- Policy Inference Node

Phases 1-3 are complete. Hardware is verified, the full driver + perception + SLAM/Nav2 config stack is in place. The next step is training an RL policy on the Windows workstation and creating the `strafer_inference` package to run it on the Jetson.

### Context

- RoboClaw PID is tuned and auto-applied on startup: **P=15000, I=750, D=0, QPPS=2796**
- Ports: `/dev/roboclaw0` (front, 0x80), `/dev/roboclaw1` (rear, 0x81) via udev (`make udev`)
- Perception pipeline is live: `/d555/imu/filtered` (200 Hz madgwick), `/d555/depth/downsampled` (30 Hz, 32FC1, 80×60)
- Full bringup: `ros2 launch strafer_bringup base.launch.py` → driver + description + perception
- Smoke test: `ros2 launch strafer_bringup navigation.launch.py` → full stack + ValidateDrive

### Task: Create `strafer_inference` package

Build the `strafer_inference` ROS2 package with a `policy_inference_node` that closes the loop between sensor data and motor commands via a trained RL policy.

#### Architecture

```
/d555/imu/filtered ────────────┐
/strafer/joint_states ─────────┤
/d555/depth/downsampled ───────┤  assemble_observation()  ┌──────────────┐
/strafer/goal ─────────────────┤ ───────────────────────► │  RL Policy   │ ──► interpret_action() ──► /strafer/cmd_vel
                                └──────────────────────    │  (.pt/.onnx) │
                                    last_action feedback   └──────────────┘
```

#### Implementation

1. **Subscribe**:
   - `/d555/imu/filtered` (`sensor_msgs/Imu`, 200 Hz) → cache latest `linear_acceleration` + `angular_velocity`
   - `/strafer/joint_states` (`sensor_msgs/JointState`, 50 Hz) → cache wheel velocities, convert rad/s → ticks/s via `RADIANS_TO_ENCODER_TICKS`
   - `/d555/depth/downsampled` (`sensor_msgs/Image`, 32FC1, 80×60, 30 Hz) → cache as flat float32 array (DEPTH variant only)
   - `/strafer/goal` (`geometry_msgs/PoseStamped`) → cache goal; compute `goal_relative` = goal in robot frame via TF

2. **Timer at 30 Hz**: assemble observation, run model, publish `cmd_vel`:
   ```python
   raw = {
       "imu_accel": [...],           # (3,) m/s²
       "imu_gyro": [...],            # (3,) rad/s
       "encoder_vels_ticks": [...],  # (4,) ticks/s -- raw from RoboClaw (via JointState × 85.57)
       "goal_relative": [...],       # (2,) meters in robot frame (from TF)
       "last_action": [...],         # (3,) normalized, cached from previous step
   }
   obs = assemble_observation(raw, PolicyVariant.NOCAM)  # → (15,) float32
   action = policy(obs)                                   # → (3,) float32
   vx, vy, omega = interpret_action(action)
   # publish Twist(linear.x=vx, linear.y=vy, angular.z=omega)
   last_action = action  # cache for next step
   ```

3. **Model loading**: `load_policy(path, variant)` from `strafer_shared.policy_interface` handles `.pt` and `.onnx` transparently. Pass `--model` path as ROS2 param or CLI arg.

4. **ROS2 parameters**:
   - `model_path` (string) -- path to `.pt` or `.onnx` file
   - `variant` (string, default: `"NOCAM"`) -- `"NOCAM"` or `"DEPTH"`
   - `publish_rate` (float, default: `30.0`)
   - `goal_timeout_s` (float, default: `5.0`) -- stop if no goal received within this window

5. **Safety**: If no goal received for `goal_timeout_s`, publish zero `cmd_vel`.

#### Package structure

```
strafer_inference/
├── package.xml
├── setup.py
├── strafer_inference/
│   ├── __init__.py
│   └── policy_inference_node.py
├── launch/
│   └── inference.launch.py
└── test/
    └── test_inference_node.py
```

#### Critical rules

- Use `assemble_observation()` and `interpret_action()` from `strafer_shared.policy_interface` -- do NOT manually normalize
- Encoder velocities: convert JointState rad/s → ticks/s using `RADIANS_TO_ENCODER_TICKS ≈ 85.57` before passing to `assemble_observation()`
- Goal in robot frame: use `tf2_ros.Buffer.lookup_transform("base_link", goal.header.frame_id, ...)` to transform goal pose
- All constants from `strafer_shared.constants` (do not hardcode)
- Single-threaded executor; all callbacks are cache-only; only the timer does work

### References

- `docs/SIM_TO_REAL_PLAN.md` -- Sections 3-4 for full ROS2 architecture
- `docs/SIM_TO_REAL_TUNING_GUIDE.md` -- Actuator model mapping, sensor characterization, tuning procedure for Phase 5
- `docs/WIRING_GUIDE.md` -- Physical connections, terminal layout
- `source/strafer_shared/strafer_shared/constants.py` -- All robot constants
- `source/strafer_shared/strafer_shared/mecanum_kinematics.py` -- Kinematics functions
