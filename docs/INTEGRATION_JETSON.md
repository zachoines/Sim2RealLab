# Integration Context: Jetson Orin NX (Robot Computer)

## Role

The Jetson owns all hardware interaction, sensing, SLAM, navigation, and mission execution. It calls DGX Spark services over LAN HTTP for planning and visual grounding. Mission state, cancel, retry, and safety-critical behavior stay local on the robot.

## System Architecture

```
Jetson Orin NX (Robot)                       DGX Spark (Workstation)
========================                     ========================
strafer_ros                                  strafer_vlm (port 8100)
  - RoboClaw motor driver                      - Qwen2.5-VL-3B-Instruct
  - RealSense D555 camera+IMU                  - POST /ground
  - RTAB-Map SLAM                              - POST /describe
  - Nav2 navigation (MPPI holonomic)           - GET /health
  - goal_projection_node (service)
                                             strafer_autonomy.planner (port 8200)
strafer_autonomy.executor                      - Qwen3-4B LLM
  - AutonomyCommandServer (ROS action)         - POST /plan
  - MissionRunner (skill dispatch)             - GET /health
  - JetsonRosClient (local ROS2)
  - HttpPlannerClient → DGX:8200
  - HttpGroundingClient → DGX:8100

     ←——— LAN HTTP (service calls) ———→
```

## Data Flow: "go to the door"

1. Operator → `ExecuteMission` ROS action on Jetson
2. Jetson executor → `POST DGX:8200/plan` with `{"raw_command": "go to the door"}`
3. DGX planner returns `MissionPlan` with steps: `[scan_for_target, project_detection_to_goal_pose, navigate_to_pose]`
4. Executor runs `scan_for_target`: captures RGB from D555, calls `POST DGX:8100/ground` with JPEG base64, rotates via `cmd_vel` if not found
5. Executor runs `project_detection_to_goal_pose`: calls local ROS service with VLM bbox → map-frame goal pose
6. Executor runs `navigate_to_pose`: sends Nav2 action goal, blocks until done

## Relevant Packages

### `strafer_ros` — ROS2 packages for the physical robot

**`strafer_driver`**: RoboClaw motor control.
- Subscribes: `/strafer/cmd_vel` (Twist)
- Publishes: `/strafer/odom` (Odometry), `/strafer/joint_states` (JointState), `/diagnostics`
- Broadcasts TF: `odom → base_link` at 50 Hz
- Mecanum inverse kinematics, dual controllers (address 0x80 front, 0x81 rear)
- Watchdog: stops motors if no `cmd_vel` for 500 ms
- Config: `config/driver_params.yaml` (ports, baud, publish rate, auto-detect)

**`strafer_perception`**: RealSense D555 camera + IMU processing.
- `timestamp_fixer`: Re-stamps camera messages from hardware clock to ROS system time (needed for Jetson Tegra USB drift)
  - `/d555/color/image_raw` → `/d555/color/image_sync`
  - `/d555/aligned_depth_to_color/image_raw` → `/d555/aligned_depth_to_color/image_sync`
- `depth_downsampler`: 640×360 depth → 80×60 float32 meters → `/d555/depth/downsampled`
- `imu_filter_madgwick`: Fuses accel + gyro → `/d555/imu/filtered` (with orientation quaternion)
- `goal_projection_node`: ROS service `/strafer/project_detection_to_goal_pose`
  - Takes VLM bbox in Qwen [0,1000] normalized coords + standoff distance
  - Pipeline: bbox center → pixel coords → depth lookup (5×5 median) → 3D point → TF2 to map frame → standoff offset → reachability check
  - Returns: `PoseStamped goal_pose`, `PoseStamped target_pose`, quality flags

**`strafer_slam`**: RTAB-Map SLAM.
- Inputs: RGB sync, depth sync, laser scan, wheel odom, filtered IMU
- Outputs: `/rtabmap/map` (OccupancyGrid), `map → odom` TF
- Detection rate: 2 Hz, ORB features (500 max), 5 cm grid resolution

**`strafer_navigation`**: Nav2 with MPPI holonomic controller.
- Motion model: "Omni" (produces vx, vy, wz for mecanum drive)
- Footprint: 0.43 m × 0.43 m
- Max velocities: [0.5, 0.5, 1.9] m/s for [vx, vy, wz]
- Action: `/navigate_to_pose`

**`strafer_description`**: URDF + `robot_state_publisher`.
- TF tree: `map ← odom ← base_link ← {chassis, wheels, d555_link}`

**`strafer_msgs`**: Custom ROS interfaces.

### `strafer_autonomy` (executor portion) — Runs on Jetson

**`JetsonRosClient`** (`clients/ros_client.py`):
- Subscribes: `/d555/color/image_sync`, `/d555/aligned_depth_to_color/image_sync`, `/d555/color/camera_info_sync`, `/strafer/odom`
- Background `SingleThreadedExecutor` spin thread keeps sensor cache fresh
- All ROS imports deferred (module importable without ROS for testing)
- Methods:
  - `capture_scene_observation()` → `SceneObservation` (BGR, aligned depth meters, camera_info, robot pose). Raises if frames > 0.5s old.
  - `navigate_to_pose()` → `SkillResult`. Nav2 action client, blocks until completion/timeout/cancel.
  - `project_detection_to_goal_pose()` → `GoalPoseCandidate`. Calls local ROS service.
  - `rotate_in_place()` → `SkillResult`. Publishes `cmd_vel` angular-Z with odom feedback loop.
  - `cancel_active_navigation()` → bool. Cancels Nav2 goal.
  - `get_robot_state()` → dict (pose, velocity, navigation_active).

**`HttpPlannerClient`** (`clients/planner_client.py`):
- Config: `HttpPlannerClientConfig(base_url="http://<DGX_IP>:8200", timeout_s=10.0, max_retries=2)`
- `base_url` is a **required field with no default** — must be set at construction
- `plan_mission(PlannerRequest)` → `MissionPlan`
- Retry with exponential backoff (0.5s × 2^N) on connection errors and 5xx

**`HttpGroundingClient`** (`clients/vlm_client.py`):
- Config: `HttpGroundingClientConfig(base_url="http://<DGX_IP>:8100", timeout_s=15.0, max_retries=2)`
- `base_url` is a **required field with no default** — must be set at construction
- `locate_semantic_target(GroundingRequest)` → `GroundingResult`. Encodes image as JPEG base64, sends to `/ground`.
- `describe_scene(request_id, image_rgb_u8, prompt, max_image_side)` → `SceneDescription`. Sends to `/describe`.

**`MissionRunner`** (`executor/mission_runner.py`):
- Receives `MissionPlan`, dispatches skills sequentially in background thread
- Available skills: `capture_scene_observation`, `locate_semantic_target`, `scan_for_target`, `describe_scene`, `project_detection_to_goal_pose`, `navigate_to_pose`, `wait`, `report_status`, `cancel_mission`
- `scan_for_target`: loops up to `max_scan_steps`, each step = capture → ground → check cancel → rotate. On exhaustion, calls `describe_scene` and includes description in failure message.

**`AutonomyCommandServer`** (`executor/command_server.py`):
- ROS2 action server wrapping `MissionRunner`
- Action: `execute_mission` (ExecuteMission)
- Service: `get_mission_status` (GetMissionStatus)
- `build_command_server()`: Factory that wires planner/grounding/ros clients → runner → server

**CLI** (`cli.py`):
```bash
strafer-autonomy-cli submit "go to the door" --detach
strafer-autonomy-cli status
strafer-autonomy-cli cancel
```

## Custom ROS Interface Definitions

### ExecuteMission.action
```
# Goal
string request_id
string raw_command
string source
bool replace_active_mission
---
# Result
bool accepted
string mission_id
string final_state
string error_code
string message
---
# Feedback
string mission_id
string state
string current_step_id
string current_skill
string message
float32 elapsed_s
```

### ProjectDetectionToGoalPose.srv
```
# Request
float32[4] bbox_normalized_1000    # [x1, y1, x2, y2] in Qwen [0,1000] coords
float64 image_stamp_sec
float32 standoff_m
string target_label
---
# Response
bool found
bool depth_valid
geometry_msgs/PoseStamped goal_pose
geometry_msgs/PoseStamped target_pose
string[] quality_flags
string message
```

### GetMissionStatus.srv
```
---
bool active
string mission_id
string state
string raw_command
string current_step_id
string current_skill
string message
float32 elapsed_s
```

## Key Shared Constants (from `strafer_shared`)

```python
CHASSIS_LENGTH = 0.43        # meters
CHASSIS_WIDTH = 0.43
TRACK_WIDTH = 0.4132         # center-to-center wheel distance
WHEEL_BASE = 0.336           # front-to-back wheel distance
WHEEL_RADIUS = 0.048
NAV_LINEAR_VEL = 0.5         # m/s
NAV_ANGULAR_VEL = 1.9        # rad/s
MAP_RESOLUTION = 0.05        # meters
DEPTH_CLIP_NEAR = 0.3        # meters
DEPTH_CLIP_FAR = 6.0
```

## Setup & Launch Instructions

### Prerequisites
- Jetson Orin NX with ROS2 (Humble or later)
- RealSense D555 connected via USB 3.x
- Dual RoboClaw motor controllers on `/dev/roboclaw0`, `/dev/roboclaw1` (via udev rules)
- DGX Spark accessible on the LAN

### Build
```bash
cd ~/Sim2RealLab
colcon build --symlink-install --packages-up-to strafer_bringup strafer_autonomy
source install/setup.bash
```

### Launch Robot Stack
```bash
# Full stack: driver + perception + SLAM + Nav2
ros2 launch strafer_bringup navigation.launch.py
```

### Start Autonomy Executor
**No launch script exists yet.** A script is needed that:
1. Creates `HttpPlannerClient(config=HttpPlannerClientConfig(base_url="http://<DGX_IP>:8200"))`
2. Creates `HttpGroundingClient(config=HttpGroundingClientConfig(base_url="http://<DGX_IP>:8100"))`
3. Creates `JetsonRosClient()`
4. Calls `build_command_server(planner_client=..., grounding_client=..., ros_client=...)`
5. Spins the ROS node

### Submit a Mission
```bash
strafer-autonomy-cli submit "go to the door" --detach
strafer-autonomy-cli status
```

## What the DGX Spark Looks Like From Here

Two HTTP services reachable over the LAN:

| Service | URL | Purpose |
|---------|-----|---------|
| VLM | `http://<DGX_IP>:8100` | Visual grounding (`/ground`) and scene description (`/describe`) |
| Planner | `http://<DGX_IP>:8200` | Command-to-plan (`/plan`) |

Both have `/health` endpoints returning `{"status": "ok", "model_loaded": true, "model_name": "..."}`.

The DGX services are stateless. They accept JSON payloads (images as JPEG base64) and return structured results. All mission state lives on the Jetson.

## Known Integration Gaps

| # | Gap | Description |
|---|-----|-------------|
| 1 | **No executor launch script** | `build_command_server()` requires pre-configured clients with `base_url`. No launch file, entry point script, or wiring code exists to start the executor with DGX service URLs. |
| 2 | **No env-var support for client URLs** | `HttpPlannerClientConfig.base_url` and `HttpGroundingClientConfig.base_url` are required fields with no defaults and no env-var mapping. Need `PLANNER_URL` / `VLM_URL` env vars or launch arguments. |
| 3 | **Package installation** | `strafer_autonomy` must be pip-installed into the ROS Python environment or built as an ament_python package. |
| 4 | **`rotate_in_place` untested on hardware** | Uses open-loop `cmd_vel` angular-Z with odom yaw feedback. May need PID tuning or tolerance adjustment on real robot. |
| 5 | **`orient_relative_to_target` not implemented** | Raises `NotImplementedError`. Deferred from MVP. |
| 6 | **Network connectivity** | Need static IP or mDNS for the DGX Spark. Wired Ethernet preferred. |
| 7 | **Verify projection service** | `goal_projection_node` must be running alongside Nav2. Needs end-to-end verification with real VLM bbox → map-frame goal. |
| 8 | **Latency budget** | VLM grounding ~2-3s per call. Full scan (6 rotations × ground call) = ~18s before navigation begins. |
