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

## Current Phase: Phase 2 -- ROS2 Driver Packages

Both RoboClaws are connected to the Jetson via USB and configured:
- **RoboClaw #1** (front): address 0x80, packet serial, 115200 baud
- **RoboClaw #2** (rear): address 0x81, packet serial, 115200 baud

Udev rules have NOT been created yet. RoboClaws will appear as `/dev/ttyACM0` and `/dev/ttyACM1` (order not guaranteed).

### Task: Build `strafer_driver` package

Create the `strafer_driver` ROS2 package with a `roboclaw_node` that:

1. **Subscribes** to `/strafer/cmd_vel` (`geometry_msgs/Twist`) and converts body velocity to per-wheel commands using `strafer_shared.mecanum_kinematics.twist_to_wheel_velocities(vx, vy, omega)`, then `wheel_vels_to_ticks_per_sec()`, then sends to RoboClaws via `SpeedM1`/`SpeedM2`. Twist field mapping: `linear.x` = vx (forward), `linear.y` = vy (strafe left), `angular.z` = omega (CCW yaw). The `cmd_vel` callback only stores the latest Twist — serial I/O happens in the timer.

2. **Publishes** `/strafer/joint_states` (`sensor_msgs/JointState`) at 50 Hz by reading encoder velocities from both RoboClaws (`ReadSpeedM1`/`ReadSpeedM2`), converting ticks/s to rad/s. Joint names must be `["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]` (matching simulation). Field population:
   - `velocity`: rad/s in URDF frame (sign correction already handled by kinematics)
   - `position`: cumulative radians from `ReadEncM1`/`ReadEncM2` counts (useful for `robot_state_publisher` and debugging)
   - `effort`: leave empty (RoboClaw does not report torque)

3. **Publishes** `/strafer/odom` (`nav_msgs/Odometry`) at 50 Hz by integrating wheel odometry using `strafer_shared.mecanum_kinematics.encoder_ticks_to_body_velocity()`. Also broadcasts `odom` -> `base_link` TF.

4. **Safety**: Stops all motors if no `cmd_vel` received within 500ms (watchdog).

5. **ROS2 parameters** (configurable via launch or YAML):
   - `front_port` (string, default: `/dev/ttyACM0`)
   - `rear_port` (string, default: `/dev/ttyACM1`)
   - `baud_rate` (int, default: 115200)
   - `publish_rate` (float, default: 50.0)

6. **Threading model**: Use a **single-threaded executor**. The 50 Hz timer callback handles all serial I/O (send motor commands from stored `cmd_vel`, read encoders, publish JointState/odom). The `cmd_vel` subscription callback only stores the latest Twist value — no serial I/O in the callback.

#### RoboClaw communication

**Vendor a minimal serial interface** in `roboclaw_interface.py` using `pyserial`. Do NOT depend on `roboclaw_3` or other third-party RoboClaw libraries — they are poorly maintained and untested on aarch64. The interface is ~100-150 lines wrapping the RoboClaw packet serial protocol (address byte + command byte + data + CRC16). Reference: [RoboClaw User Manual](https://downloads.basicmicro.com/docs/roboclaw_user_manual.pdf).

The key commands to implement:
- `SpeedM1(address, speed)` / `SpeedM2(address, speed)` -- velocity in ticks/sec (signed int32)
- `ReadSpeedM1(address)` / `ReadSpeedM2(address)` -- returns (speed, status) in ticks/sec
- `ReadEncM1(address)` / `ReadEncM2(address)` -- returns (count, status)
- `ForwardM1(address, 0)` / `ForwardM2(address, 0)` -- stop motors

#### Serial error handling

- **Read timeout**: 100ms per command
- **Retry**: 1 retry on checksum/timeout failure, then log warning and skip that cycle
- **Sustained failure**: After 10 consecutive failures, stop all motors, publish `diagnostic_msgs/DiagnosticStatus` with ERROR level, and attempt reconnect every 2 seconds
- **Diagnostics**: Publish `diagnostic_msgs/DiagnosticStatus` on `/diagnostics` with connection state, error counts, and last successful read timestamp

#### Package structure

```
strafer_driver/
├── package.xml
├── setup.py (or setup.cfg)
├── strafer_driver/
│   ├── __init__.py
│   ├── roboclaw_node.py      # Main ROS2 node
│   └── roboclaw_interface.py # Low-level serial interface to RoboClaw
├── config/
│   └── driver_params.yaml    # Default parameters
├── launch/
│   └── driver.launch.py      # Launch file
└── test/
    └── test_driver.py        # Basic tests
```

#### Critical rules

- **ALL constants** come from `strafer_shared.constants` -- never hardcode motor addresses, encoder PPR, wheel dimensions, etc.
- **ALL kinematics** come from `strafer_shared.mecanum_kinematics` -- do not reimplement
- **Do NOT apply `WHEEL_AXIS_SIGNS` in the driver** -- the kinematics functions handle signs internally
- Use `ament_python` build type
- Wheel order is always `[FL, FR, RL, RR]` = `[wheel_1, wheel_2, wheel_3, wheel_4]`
- RoboClaw #1 (0x80) controls M1=FL, M2=FR. RoboClaw #2 (0x81) controls M1=RL, M2=RR.
- `strafer_shared` is pip-installed (`pip install -e source/strafer_shared`), not a colcon package. Add `<exec_depend>strafer_shared</exec_depend>` in `package.xml` as documentation, but colcon will not manage this dependency. Add `pyserial` to `install_requires` in `setup.py`.

### First: Standalone wiring test script

Before building the full ROS2 driver, create a **standalone Python script** (no ROS2 dependency) that validates the physical wiring:
- Opens both RoboClaw serial ports
- Spins each motor briefly at low speed (~500 ticks/sec) and reads its encoder
- Verifies encoder counts increase in the expected direction
- Prints a pass/fail summary for each motor+encoder pair

This validates Phase 1 wiring before committing to the full ROS2 stack.

### Also needed (lower priority)

- **`strafer_msgs`**: Create the package with `package.xml` and `CMakeLists.txt`. Custom messages can wait until standard messages prove insufficient.
- **Udev rules**: Create `/etc/udev/rules.d/99-strafer.rules` for persistent `/dev/roboclaw_front` and `/dev/roboclaw_rear` symlinks. Run `udevadm info -a /dev/ttyACMx | grep serial` to find unique serial numbers.

### References

- `docs/SIM_TO_REAL_PLAN.md` -- Sections 3-4 for full ROS2 architecture
- `docs/WIRING_GUIDE.md` -- Physical connections, terminal layout
- `source/strafer_shared/strafer_shared/constants.py` -- All robot constants
- `source/strafer_shared/strafer_shared/mecanum_kinematics.py` -- Kinematics functions
