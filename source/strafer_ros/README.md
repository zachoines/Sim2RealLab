# strafer_ros

ROS 2 runtime for the Strafer robot on the Jetson Orin Nano — motor driver, perception, SLAM, navigation, URDF, shared ROS interfaces, and bringup launches.

`strafer_ros` is the robot-local execution layer. It owns every
safety-critical and real-time concern: wheel commands, odometry,
synchronized sensor streams, SLAM localization, Nav2 navigation, TF, and
the custom ROS interfaces the autonomy layer drives. It is a collection
of six runtime ROS 2 packages plus one interface package, built with
`colcon` on the Jetson. The autonomy executor in
[`strafer_autonomy`](../strafer_autonomy/README.md) is the primary
caller; it runs in a separate Python environment on the same Jetson and
talks to these packages through ROS topics, services, and actions.

## Role in the system

| Package | Language | Role |
|---|---|---|
| `strafer_msgs` | Interface (action/srv) | Shared ROS interface types: `ExecuteMission.action`, `GetMissionStatus.srv`, `ProjectDetectionToGoalPose.srv` |
| `strafer_driver` | Python | RoboClaw motor control, odometry, joint states, watchdog |
| `strafer_perception` | Python | RealSense D555 timestamp correction, depth downsampling, IMU filter, goal projection service |
| `strafer_description` | URDF + Python | Robot URDF, `robot_state_publisher`, TF frames |
| `strafer_slam` | Launch / config | RTAB-Map SLAM + `depthimage_to_laserscan` (no custom nodes) |
| `strafer_navigation` | Launch / config | Nav2 with MPPI holonomic controller (no custom nodes) |
| `strafer_bringup` | Launch-only | Layered composition: `base` → `perception` → `slam` → `navigation` → `autonomy` |

Sibling packages it interacts with:

- **`strafer_autonomy`** — executor calls `execute_mission` action + `get_mission_status` service + `project_detection_to_goal_pose` service. Executor also consumes streaming topics (`/d555/color/image_sync`, aligned depth, `/strafer/odom`, TF).
- **`strafer_shared`** — all robot constants (wheel radius, velocity limits, PID values, depth clip range) are imported from `strafer_shared.constants` by `strafer_driver` and `strafer_perception`. Mecanum inverse kinematics use `strafer_shared.mecanum_kinematics`.
- **`strafer_vlm`** — not a direct consumer. The executor retrieves grounding results from the VLM service and hands bboxes to `goal_projection_node` via the `ProjectDetectionToGoalPose` service.

`strafer_ros` does **not** own mission state, intent parsing, or HTTP transports. Those live in `strafer_autonomy`.

## What ships today

- **RoboClaw driver** (`strafer_driver/roboclaw_node.py`) — single-threaded executor, 50 Hz timer for all serial I/O, auto-writes PID/QPPS from `strafer_shared.constants` at startup, dual-controller addressing (0x80 front, 0x81 rear), `/cmd_vel` → per-wheel mecanum IK, 500 ms watchdog.
- **RealSense D555 stack** (`strafer_perception/`) — launch wiring for the RealSense ROS 2 node, `timestamp_fixer` (fixes Tegra USB clock drift), `depth_downsampler` (640×360 depth → 80×60 float32 meters for RL policy input), `imu_filter_madgwick` for filtered `/d555/imu/filtered`.
- **Goal projection service** (`strafer_perception/goal_projection_node.py`) — implements `ProjectDetectionToGoalPose.srv`: VLM bbox (Qwen normalized `[0, 1000]`) → pixel → depth lookup (5×5 median) → 3D camera frame → TF to map → standoff offset → reachability check.
- **URDF + TF tree** (`strafer_description/`) — URDF with chassis, 4 wheels, and `d555_link`; `robot_state_publisher` broadcasts `base_link → {chassis, wheels, d555_link}` from joint states and URDF.
- **RTAB-Map SLAM** (`strafer_slam/`) — launch composition for `depthimage_to_laserscan` (virtual 2D scan from aligned depth) + `rtabmap` (RGB-D SLAM with odom + IMU fusion, 2 Hz detection rate, 5 cm grid).
- **Nav2 navigation** (`strafer_navigation/`) — launch composition for the full Nav2 stack with MPPI holonomic controller, `Omni` motion model (produces `vx, vy, wz` for mecanum), costmaps patched from `strafer_shared` constants.
- **Bringup layers** (`strafer_bringup/launch/`) — 6 launch files that compose progressively: `base` (driver + URDF), `perception` (+ RealSense), `slam` (+ RTAB-Map), `navigation` (+ Nav2), `autonomy` (+ goal projection + executor), and `bringup_sim_in_the_loop` (no real hardware — consumes topics published by the DGX Isaac Sim ROS 2 bridge).
- **Diagnostic / tuning scripts** (top-level in `source/strafer_ros/`) — RoboClaw PID tuning, RoboClaw direct-drive diagnostics, D555 camera + IMU verification, perception-stack recording, SLAM + motion verification with map-building video output.

## Contracts

### ROS interfaces (defined in `strafer_msgs`)

| Interface | Type | Purpose |
|---|---|---|
| `strafer_msgs/action/ExecuteMission` | Action | Submit a mission, stream feedback, cancel mid-flight. Implemented by the autonomy executor. |
| `strafer_msgs/srv/GetMissionStatus` | Service | Query current-or-last mission snapshot. Implemented by the autonomy executor. |
| `strafer_msgs/srv/ProjectDetectionToGoalPose` | Service | Project a 2D VLM bbox into a map-frame goal pose. Implemented by `strafer_perception/goal_projection_node`. |

`ExecuteMission.action`:

```text
# Goal: string request_id, string raw_command, string source, bool replace_active_mission
# Result: bool accepted, string mission_id, string final_state, string error_code, string message
# Feedback: string mission_id, string state, string current_step_id, string current_skill, string message, float32 elapsed_s
```

`ProjectDetectionToGoalPose.srv`:

```text
# Request:  float32[4] bbox_normalized_1000, float64 image_stamp_sec, float32 standoff_m, string target_label
# Response: bool found, bool depth_valid, geometry_msgs/PoseStamped goal_pose, geometry_msgs/PoseStamped target_pose,
#           string[] quality_flags, string message
```

`bbox_normalized_1000` is in Qwen normalized `[0, 1000]` coordinates. The service converts to pixels using the synced `camera_info` topic.

### Topics published / consumed

Streaming inputs the autonomy layer and RL runtime consume:

| Topic | Type | Purpose |
|---|---|---|
| `/d555/color/image_raw` | `sensor_msgs/Image` | Raw RGB (hardware timestamp) |
| `/d555/color/image_sync` | `sensor_msgs/Image` | RGB with ROS-clock timestamps (fixed by `timestamp_fixer`) |
| `/d555/color/camera_info_sync` | `sensor_msgs/CameraInfo` | RGB intrinsics matching `image_sync` |
| `/d555/aligned_depth_to_color/image_sync` | `sensor_msgs/Image` | Timestamp-fixed aligned depth in RGB frame |
| `/d555/aligned_depth_to_color/camera_info_sync` | `sensor_msgs/CameraInfo` | Aligned-depth camera info |
| `/d555/depth/downsampled` | `sensor_msgs/Image` (32FC1, meters) | 80×60 policy input |
| `/d555/imu/filtered` | `sensor_msgs/Imu` | Madgwick-filtered IMU with orientation quaternion |
| `/strafer/odom` | `nav_msgs/Odometry` | Wheel odometry at 50 Hz |
| `/strafer/joint_states` | `sensor_msgs/JointState` | Wheel states at 50 Hz |
| `/scan` | `sensor_msgs/LaserScan` | Virtual 2D scan from depth (SLAM input) |
| `/tf`, `/tf_static` | TF streams | `map ← odom ← base_link ← {chassis, wheels, d555_link}` |

Commanded / driven:

| Topic | Type | Purpose |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | Nav2 / executor output → driver input |
| `/strafer/cmd_vel` | `geometry_msgs/Twist` | Driver subscribes here; Nav2 `cmd_vel` is remapped to this in the driver launch |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Driver connection state + error counts |

### Default interface names (match `strafer_autonomy` CLI defaults)

| Interface | Default name |
|---|---|
| `ExecuteMission` action | `execute_mission` |
| `GetMissionStatus` service | `get_mission_status` |
| `ProjectDetectionToGoalPose` service | `/strafer/project_detection_to_goal_pose` |

### TF tree

```text
map ← odom ← base_link ← {chassis, wheel_*, d555_link}
```

`map → odom` published by RTAB-Map. `odom → base_link` published by `strafer_driver` at 50 Hz. `base_link → d555_link` comes from the URDF via `robot_state_publisher`; `strafer_perception/perception.launch.py` currently also publishes a redundant static `base_link → d555_link` transform that should be cleaned up — treat the URDF transform as authoritative.

### Hardware / addressing

- Dual RoboClaw ST 2x45A over USB serial (`/dev/ttyACM*` or `/dev/roboclaw0` / `/dev/roboclaw1` via udev). Addresses `0x80` (front) + `0x81` (rear). Baud 115200, Packet Serial mode.
- 4× GoBilda 5203 motors at 19.2:1. Encoder resolution 537.7 PPR at output shaft.
- `wheel_axis_signs = [-1, 1, -1, 1]` for `[FL, FR, RL, RR]` is applied inside `strafer_shared.mecanum_kinematics`. **Do not** re-apply sign inversion inside this package.
- RoboClaw PID values (`P=15000`, `I=750`, `D=0`, `QPPS=2796`) live in `strafer_shared.constants` and are auto-written to RAM on every `roboclaw_node` startup.
- udev rules in [`99-strafer.rules`](99-strafer.rules) create the stable `/dev/roboclaw0` + `/dev/roboclaw1` symlinks and grant IIO permissions for the D555 IMU stack.

## Install

```bash
# Symlink packages into the colcon workspace
mkdir -p ~/strafer_ws/src
ln -s ~/Workspace/Sim2RealLab/source/strafer_ros/* ~/strafer_ws/src/

# Build
cd ~/strafer_ws
colcon build --symlink-install
source install/setup.bash
```

Prerequisites:

- Jetson Orin Nano with ROS 2 Humble.
- RealSense D555 on USB 3 with the IMU stack enabled (see [`docs/D555_IMU_KERNEL_FIX.md`](../../docs/D555_IMU_KERNEL_FIX.md) for the kernel module build required on Tegra kernels).
- Two RoboClaw ST 2x45A controllers wired per [`docs/WIRING_GUIDE.md`](../../docs/WIRING_GUIDE.md).
- `strafer_shared` and `strafer_autonomy` pip-installed into the ROS Python environment: `pip install -e source/strafer_shared source/strafer_autonomy`.
- udev rules: `sudo cp source/strafer_ros/99-strafer.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules`.

From the repo root, `make build` runs the colcon build and `make udev` installs the rules.

## Run

### Full stacks via `strafer_bringup`

| Launch file | Starts |
|---|---|
| `base.launch.py` | driver + description |
| `perception.launch.py` | base + RealSense + timestamp fixer + depth downsampler + IMU filter |
| `slam.launch.py` | perception + `depthimage_to_laserscan` + RTAB-Map |
| `navigation.launch.py` | slam + Nav2 (MPPI holonomic) |
| `autonomy.launch.py` | navigation + `goal_projection_node` + `strafer-executor` (needs `VLM_URL` + `PLANNER_URL`) |
| `bringup_sim_in_the_loop.launch.py` | perception + SLAM + Nav2 consuming topics from the DGX Isaac Sim ROS 2 bridge (no real hardware) |

Typical operator sequences:

```bash
# Navigation only (driver + perception + SLAM + Nav2):
ros2 launch strafer_bringup navigation.launch.py
# or from the repo root Makefile:
make launch

# Full autonomy (navigation + goal projection + executor → DGX services):
VLM_URL=http://192.168.50.196:8100 PLANNER_URL=http://192.168.50.196:8200 \
    ros2 launch strafer_bringup autonomy.launch.py
# or:
VLM_URL=http://192.168.50.196:8100 PLANNER_URL=http://192.168.50.196:8200 \
    make launch-autonomy
```

### Hardware validation scripts

These live at the top of `source/strafer_ros/` and are direct-hardware tools. Run outside the ROS launch graph (no SLAM / Nav2 running; they drive the motors directly).

```bash
# RoboClaw PID tuning and step-response test
python3 source/strafer_ros/tune_pid.py --read         # current PID
python3 source/strafer_ros/tune_pid.py --tune         # set + measure

# RoboClaw direct duty-cycle diagnostics (skip PID)
python3 source/strafer_ros/diagnose_roboclaw.py

# D555 camera + IMU standalone check
python3 source/strafer_ros/test_d555_camera.py

# SLAM + motion verification (requires navigation.launch.py running)
python3 source/strafer_ros/ros_test_slam.py                              # verify topics only
python3 source/strafer_ros/ros_test_slam.py --drive forward --duration 3  # drive + record
python3 source/strafer_ros/ros_test_slam.py --drive all --duration 3      # all patterns

# Perception stack record (requires perception.launch.py running)
python3 source/strafer_ros/ros_test_perception.py --record
```

### Cross-host config

The executor and SLAM topics use `rmw_cyclonedds_cpp` with `ROS_DOMAIN_ID=42` for LAN-wide visibility between the Jetson and DGX (used when running against the Isaac Sim ROS 2 bridge or for cross-host topic echo).

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
```

## Design

**Safety-critical execution stays local.** Wheel commands, watchdog, odometry, TF, and the SLAM / Nav2 loop all run on the Jetson. The workstation-hosted planner / VLM services are advisory: the executor can continue to refuse and cancel missions regardless of network state.

**All constants live in `strafer_shared`, not in ROS code.** Wheel radius, track width, velocity limits, depth clip range, PID values, RoboClaw addresses — every number that must stay consistent between sim and real is imported from `strafer_shared.constants`. Do not hardcode values in nodes, launch files, or config YAML.

**Motor sign inversion is applied exactly once, inside `strafer_shared.mecanum_kinematics`.** If a wheel spins backwards, fix the wiring per `docs/WIRING_GUIDE.md` — do not add a second sign flip in the driver, inference runtime, or autonomy code.

**The autonomy layer is not a ROS package.** `strafer_autonomy` is a pip-installed Python package that uses `rclpy` at runtime for the command server and ROS client. This keeps the mission runner, schemas, and HTTP clients testable in plain Python environments while still running inside the ROS 2 graph on the Jetson.

**Timestamp-fixed `*_sync` topics are authoritative for grounding and projection.** The RealSense D555 on Tegra USB produces hardware-clock timestamps that drift unpredictably (observed ~44× faster than system clock on Tegra). `timestamp_fixer` re-stamps all four camera topics with the current ROS clock at reception. `approximate_sync`-based consumers (RTAB-Map, `goal_projection_node`) must subscribe to the `*_sync` versions, never to the raw topics.

**URDF is the single source of truth for `base_link → d555_link`.** `strafer_perception` also publishes a static TF of the same transform — a historical wart to be removed. Consumers should treat the URDF transform as authoritative.

**`navigate_to_pose` stays backend-agnostic at the autonomy boundary.** The Nav2 path is the only backend that ships today. `strafer_direct` (pure-RL via `strafer_inference`) and `hybrid_nav2_strafer` (Nav2 global + RL local) are defined at the interface level but not yet implemented.

## Testing

`colcon` drives package-level tests; run after build.

```bash
cd ~/strafer_ws
colcon test --packages-select strafer_driver strafer_perception strafer_slam strafer_navigation strafer_bringup
colcon test-result --verbose
```

From the repo root:

```bash
make test         # all colcon tests (requires a prior `make build`)
make test-unit    # strafer_driver unit tests directly via pytest (no colcon)
```

End-to-end smoke checks with real hardware run through the validation scripts in `source/strafer_ros/` (see Run section).

## Deferred / known limitations

Tracked in [`docs/DEFERRED_WORK.md`](../../docs/DEFERRED_WORK.md). Items currently open:

- **`strafer_inference`** — planned Jetson package for Isaac-trained RL policy execution; not implemented. Once present, it becomes the backend for `execution_backend="strafer_direct"` and `"hybrid_nav2_strafer"` on the `navigate_to_pose` skill.
- **`orient_relative_to_target` action** — `strafer_msgs/action/OrientRelativeToTarget.action` is defined in the design docs but not shipped. Executor-side handler is drafted but commented out.
- **`rotate_in_place` PID on real hardware** — `JetsonRosClient.rotate_in_place()` uses open-loop `cmd_vel` with odom yaw feedback; tolerance tuning on hardware is pending.
- **Redundant static TF in `perception.launch.py`** — duplicates the URDF's `base_link → d555_link` transform and should be removed.
- **`MissionStatus.msg` topic** — optional topic-based status for dashboards; not yet added to `strafer_msgs`.

## References

- [`source/strafer_autonomy/README.md`](../strafer_autonomy/README.md) — executor, mission runner, CLI; consumes these ROS interfaces.
- [`source/strafer_shared/`](../strafer_shared/) — constants, mecanum kinematics, policy I/O contract. Authoritative for every shared value.
- [`source/strafer_lab/README.md`](../strafer_lab/README.md) — sim-side counterpart; uses the same shared contract so trained policies transfer unchanged.
- [`docs/WIRING_GUIDE.md`](../../docs/WIRING_GUIDE.md) — motor + encoder + RoboClaw + Jetson wiring, pinouts, address configuration.
- [`docs/D555_IMU_KERNEL_FIX.md`](../../docs/D555_IMU_KERNEL_FIX.md) — Tegra-kernel HID sensor module build (mandatory for D555 IMU).
- [`docs/SIM_TO_REAL_TUNING_GUIDE.md`](../../docs/SIM_TO_REAL_TUNING_GUIDE.md) — actuator / sensor alignment procedure pairing this package with `strafer_lab`.
- [`docs/VALIDATE_ISAAC_SIM_AND_INFINIGEN.md`](../../docs/VALIDATE_ISAAC_SIM_AND_INFINIGEN.md) — DGX-side validation (for sim-in-the-loop bringup).
- [`docs/DEFERRED_WORK.md`](../../docs/DEFERRED_WORK.md) — open items.
