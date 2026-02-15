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

Install it on the Jetson with: `pip install -e source/strafer_shared`

**Always import constants from `strafer_shared` rather than hardcoding values.** This prevents sim-real drift.

## Critical Sim-to-Real Contract

The trained RL policy expects observations in a specific order and scale. Getting this wrong silently breaks the policy. Reference `docs/SIM_TO_REAL_PLAN.md` Section 5 for the full table. Summary for the NoCam variant (15 dims):

| Index | Field | Scale |
|-------|-------|-------|
| 0-2 | IMU linear acceleration (m/s²) | × 1/156.96 |
| 3-5 | IMU angular velocity (rad/s) | × 1/34.9 |
| 6-9 | Wheel encoder velocities (ticks/s) | × 1/3000.0 |
| 10-11 | Goal position relative (m) | × 1.0 |
| 12-14 | Last action (normalized) | × 1.0 |

Encoder velocities from JointState (rad/s) must be converted to ticks/sec via `RADIANS_TO_ENCODER_TICKS ≈ 85.57` before applying the scale.

## Motor Direction

`wheel_axis_signs = [-1, 1, -1, 1]` for [FL, FR, RL, RR]. Apply in software (in `strafer_shared.mecanum_kinematics`), not by swapping wires.

## Key Topics

| Topic | Type | Publisher | Rate |
|-------|------|-----------|------|
| `/strafer/cmd_vel` | `geometry_msgs/Twist` | inference node | 30 Hz |
| `/strafer/joint_states` | `sensor_msgs/JointState` | driver node | 50 Hz |
| `/strafer/odom` | `nav_msgs/Odometry` | driver node | 50 Hz |
| `/d555/imu` | `sensor_msgs/Imu` | realsense node | 200 Hz |
| `/d555/depth/image_rect_raw` | `sensor_msgs/Image` | realsense node | 30 Hz |
| `/d555/color/image_raw` | `sensor_msgs/Image` | realsense node | 30 Hz |

## Conventions

- **ROS2 Humble** -- use `rclpy` for Python nodes, `ament_python` or `ament_cmake` build type
- **Python 3.10** -- type hints encouraged, f-strings preferred
- Follow ROS2 naming: `snake_case` for topics/nodes/params, `CamelCase` for message types
- All physical constants come from `strafer_shared.constants` -- never hardcode
- Test with `colcon test`; use `launch_testing` for integration tests
- Udev rules go in `/etc/udev/rules.d/99-strafer.rules` for persistent device names

## Current Phase

Check `docs/SIM_TO_REAL_PLAN.md` Section 8 for the current implementation phase and what to work on next. Tasks are tracked with `- [ ]` / `- [x]` checkboxes.
