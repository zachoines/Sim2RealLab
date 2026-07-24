# strafer_ros

ROS 2 runtime for the Strafer robot on the Jetson Orin Nano — motor driver, perception, SLAM, navigation, URDF, shared ROS interfaces, and bringup launches.

`strafer_ros` is the robot-local execution layer. It owns every
safety-critical and real-time concern: wheel commands, odometry,
synchronized sensor streams, SLAM localization, Nav2 navigation, TF, and
the custom ROS interfaces the autonomy layer drives. It is a collection
of seven runtime ROS 2 packages plus one interface package, built with
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
| `strafer_inference` | Python | Trained-policy execution: `inference_node` (obs assembly, ONNX/TorchScript load, `navigate_to_pose` action server, six-source watchdog, L1 velocity clamp) + `subgoal_generator_node` (rolling subgoal from Nav2-planned paths) for the `strafer_direct` / `hybrid_nav2_strafer` backends; diagnostic parity CLIs under `scripts/` |
| `strafer_bringup` | Launch-only | Layered composition: `base` → `perception` → `slam` → `navigation` → `autonomy` |

Sibling packages it interacts with:

- **`strafer_autonomy`** — executor calls `execute_mission` action + `get_mission_status` service + `project_detection_to_goal_pose` service. Executor also consumes streaming topics (`/d555/color/image_sync`, aligned depth, `/strafer/odom`, TF).
- **`strafer_shared`** — all robot constants (wheel radius, velocity limits, PID values, depth clip range) are imported from `strafer_shared.constants` by `strafer_driver` and `strafer_perception`. Mecanum inverse kinematics use `strafer_shared.mecanum_kinematics`.
- **`strafer_vlm`** — not a direct consumer. The executor retrieves grounding results from the VLM service and hands bboxes to `goal_projection_node` via the `ProjectDetectionToGoalPose` service.

`strafer_ros` does **not** own mission state, intent parsing, or HTTP transports. Those live in `strafer_autonomy`.

## What ships today

- **RoboClaw driver** (`strafer_driver/roboclaw_node.py`) — single-threaded executor, 50 Hz timer for all serial I/O, auto-writes PID/QPPS from `strafer_shared.constants` at startup, dual-controller addressing (0x80 front, 0x81 rear), `/cmd_vel` → per-wheel mecanum IK, 500 ms watchdog.
- **RealSense D555 stack** (`strafer_perception/`) — launch wiring for the RealSense ROS 2 node, `timestamp_fixer` (fixes Tegra USB clock drift), `depth_downsampler` (640×360 depth → 80×45 float32 meters; diagnostic, not the policy input), `imu_filter_madgwick` for filtered `/d555/imu/filtered`.
- **Goal projection service** (`strafer_perception/goal_projection_node.py`) — implements `ProjectDetectionToGoalPose.srv`: VLM bbox (Qwen normalized `[0, 1000]`) → pixel → depth lookup (5×5 median) → 3D camera frame → TF to map → standoff offset → reachability check.
- **URDF + TF tree** (`strafer_description/`) — URDF with chassis, 4 wheels, and `d555_link`; `robot_state_publisher` broadcasts `base_link → {chassis, wheels, d555_link}` from joint states and URDF.
- **RTAB-Map SLAM** (`strafer_slam/`) — launch composition for `depthimage_to_laserscan` (virtual 2D scan from aligned depth) + `rtabmap` (RGB-D SLAM with odom + IMU fusion, 2 Hz detection rate, 5 cm grid).
- **Nav2 navigation** (`strafer_navigation/`) — launch composition for the full Nav2 stack with MPPI holonomic controller, `Omni` motion model (produces `vx, vy, wz` for mecanum), costmaps patched from `strafer_shared` constants.
- **Bringup layers** (`strafer_bringup/launch/`) — 6 launch files that compose progressively: `base` (driver + URDF), `perception` (+ RealSense), `slam` (+ RTAB-Map), `navigation` (+ Nav2), `autonomy` (+ goal projection + executor), and `bringup_sim_in_the_loop` (no real hardware — consumes topics published by the DGX Isaac Sim ROS 2 bridge).
- **Diagnostic / tuning scripts** (top-level in `source/strafer_ros/`) — RoboClaw PID tuning, RoboClaw direct-drive diagnostics, D555 camera + IMU verification, perception-stack recording, SLAM + motion verification with map-building video output.
- **Trained-policy execution** (`strafer_inference/`) — `inference_node` loads an exported policy (ONNX/TorchScript) and drives `/strafer/cmd_vel` from a `navigate_to_pose` goal: variant-aware obs assembly (NOCAM/DEPTH ±SUBGOAL), a six-source freshness watchdog, recurrent hidden-state resets at mission boundaries, and an L1 velocity clamp. `subgoal_generator_node` follows the inference node's active-goal telemetry, replans via Nav2's `ComputePathToPose`, and rolls a subgoal along the planned path for the hybrid backend. The action server is advertised only when a policy loads (else the autonomy dispatcher falls back to Nav2).
- **Trained-policy parity tooling** (`strafer_inference/scripts/`) — `obs_parity.py` / `subgoal_parity.py` and the rclpy-free `strafer_inference.parity` library compare the deployed inference node's assembled observations and rolling-subgoal picks against the training env (or a rosbag self-check) on the sim-time axis; the node's `obs_dump_path` parameter emits the per-tick obs JSONL they consume. Diagnostic only — JSONL contract in `strafer_inference/scripts/PARITY_SCHEMA.md`.

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
| `/d555/depth/downsampled` | `sensor_msgs/Image` (32FC1, meters) | 80×45 (diagnostic, not the policy input) |
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
| `/d555/color/detections` | `vision_msgs/Detection2DArray` | Executor-published VLM grounding bbox (pixel coords, source-image stamp) — Foxglove RGB panel uses it as an annotation overlay. Empty array clears the previous overlay. **Latched (`TRANSIENT_LOCAL`)** so late subscribers (Foxglove attaching mid-mission, `ros2 topic echo` opened after the scan finished) get the most recent overlay state. To inspect from the CLI: `ros2 topic echo /d555/color/detections --qos-durability transient_local`. |
| `/d555/color/grounding_frame` | `sensor_msgs/Image` (`bgr8`) | The exact RGB frame the VLM grounded against, republished alongside `/d555/color/detections` so Foxglove can render a stable image+bbox overlay (the bbox is in pixel-space for *this* frame; the live camera has moved on by the time Foxglove draws). Refreshed only on accepted detections — empty / rejected grounding leaves the last-grounded view in place. **Latched (`TRANSIENT_LOCAL`)** for the same late-subscriber reason as the detections topic. |
| `/d555/color/detections_fg` | `foxglove_msgs/ImageAnnotations` | Same bbox data as `/d555/color/detections`, encoded as Foxglove's native image-annotation schema (one LINE_LOOP per bbox + a label TextAnnotation). Exists only because Foxglove Studio 2.x lists `Detection2DArray` topics on the graph but does not render them as image overlays. Use `/d555/color/detections` (`Detection2DArray`) for any non-Foxglove consumer (RViz, bag replay, downstream analytics). **Latched (`TRANSIENT_LOCAL`)**. |

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

## Containerized deployment (recommended)

The Jetson runs the stack in **containers** (the primary path) -- two images over
CycloneDDS, driven from the repo-root Makefile:

```bash
make images            # build strafer-cpu + strafer-gpu
make launch-sim        # sim-in-the-loop (consumes the DGX bridge; foxglove :8765)
make up                # full robot deploy (base+perception+slam+navigation+autonomy)
make launch-autonomy   # + GPU policy inference (--profile policy)
make submit CMD="go to the chair"   # submit a mission to the running stack
make down              # stop the containers
```

Full reference -- lanes, `policy`/`remote` profiles, host provisioning, the
policy fail-loud contract, the dev bind-mount overlay -- lives in
[`deploy/README.md`](deploy/README.md). New-Jetson provisioning: flash JP6.2 ->
`sudo apt install docker.io` -> `sudo bash deploy/host-setup/install-host-prereqs.sh`
-> `make images`. The container images bake everything the bare-metal sections below
describe (colcon build, the pip-editable + `onnxruntime-gpu` install, the
setuptools>=64 PEP-660 fix, the udev rules).

## Install (bare-metal -- advanced; direct single-node debug)

```bash
# Symlink packages into the colcon workspace
mkdir -p ~/strafer_ws/src
ln -s ~/workspaces/Sim2RealLab/source/strafer_ros/* ~/strafer_ws/src/

# Build (source the ROS environment first — it is not auto-sourced under
# `bash --noprofile`, CI, sudo, or cron)
cd ~/strafer_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Prerequisites:

- Jetson Orin Nano with ROS 2 Humble.
- RealSense D555 on USB 3 with the IMU stack enabled (see [`docs/D555_IMU_KERNEL_FIX.md`](../../docs/D555_IMU_KERNEL_FIX.md) for the kernel module build required on Tegra kernels).
- Two RoboClaw ST 2x45A controllers wired per [`docs/WIRING_GUIDE.md`](../../docs/WIRING_GUIDE.md).
- `strafer_shared` and `strafer_autonomy` pip-installed into the ROS Python environment: `pip install -e source/strafer_shared -e source/strafer_autonomy --no-build-isolation`. One `-e` per package (a lone `-e` makes only the first editable); `--no-build-isolation` reuses the host `setuptools`, since the stock Jetson pip 22.0.2 otherwise build-isolates a `setuptools` too old for PEP 660 and the editable install fails with a missing `build_editable` hook.
- `onnxruntime-gpu` for the `strafer_inference` node's TensorRT/CUDA execution providers — the stock CPU `onnxruntime` silently runs DEPTH inference on CPU (~84 ms, over the 33 ms budget). The CPU and GPU wheels share one install dir, so uninstall the CPU build first: `pip uninstall -y onnxruntime && pip install --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu==1.23.0` (JetPack 6.2 / CUDA 12.6 → `jp6/cu126`). Verify with `python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"` (lists `TensorrtExecutionProvider`). Do **not** `pip install` the `nvidia-*-cu12` / `tensorrt` wheels — the Jetson build links the JetPack system CUDA/cuDNN/TensorRT and pip copies conflict.
- udev rules: `sudo cp source/strafer_ros/99-strafer.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules`.
- `ros-humble-foxglove-bridge` for the headless visualizer in `bringup_sim_in_the_loop.launch.py`: `sudo apt install ros-humble-foxglove-bridge`. Skip if you always launch with `viewer:=false`.
- `ros-humble-vision-msgs` for the executor's `Detection2DArray` overlay publisher: `sudo apt install ros-humble-vision-msgs`. Required by `strafer_autonomy.clients.ros_client.JetsonRosClient.publish_detections()`.
- `ros-humble-foxglove-msgs` for the Foxglove-native `ImageAnnotations` companion publisher (renders the bbox overlay in Foxglove Studio's Image panel): `sudo apt install ros-humble-foxglove-msgs`. Optional — if missing, the executor logs a warning at startup and the canonical `/d555/color/detections` topic still publishes; only the Foxglove overlay is disabled.

From the repo root, `make build` runs the colcon build and `make udev` installs the rules.

## Run (bare-metal -- advanced; direct single-node debug)

> The `make launch*` / `make submit` targets now drive the **containers** (see
> *Containerized deployment* above). The `ros2 launch ...` commands below are the
> bare-metal equivalents, kept for direct single-node debugging.

### Full stacks via `strafer_bringup`

| Launch file | Starts |
|---|---|
| `base.launch.py` | driver + description |
| `perception.launch.py` | base + RealSense + timestamp fixer + depth downsampler + IMU filter |
| `slam.launch.py` | perception + `depthimage_to_laserscan` + RTAB-Map |
| `navigation.launch.py` | slam + Nav2 (MPPI holonomic) |
| `autonomy.launch.py` | navigation + `goal_projection_node` + `strafer-executor` (needs `VLM_URL` + `PLANNER_URL`) |
| `bringup_sim_in_the_loop.launch.py` | perception + SLAM + Nav2 + executor + `foxglove_bridge` (default :8765) consuming topics from the DGX Isaac Sim ROS 2 bridge (no real hardware) |

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

# Sim-in-the-loop (Jetson side; DGX must be running `make sim-bridge`):
make launch-sim
# foxglove_bridge comes up on :8765 by default; SSH-tunnel from the
# operator's workstation and connect Foxglove Studio to ws://localhost:8765
# (full walkthrough: docs/INTEGRATION_SIM_IN_THE_LOOP.md Stage 3.5).
# Disable the visualizer with `viewer:=false` if the dep isn't installed.
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

The perception depth stream is large (fragmented ~921 KB frames). Under load the kernel-default receive socket buffer overflows and whole frames are dropped at the receiver, so the deployment raises it: `strafer_bringup/config/cyclonedds.xml` requests a 16 MB receive buffer plus fragment-reassembly headroom, exported via `CYCLONEDDS_URI` from the bringup env files. That request is capped by `net.core.rmem_max` (default ~208 KB on the Jetson), so raise the kernel cap once per host — both the sim-in-the-loop and real-robot lanes — or the tuning is inert:

```bash
sudo cp source/strafer_ros/strafer_bringup/config/99-cyclonedds-rmem.conf /etc/sysctl.d/
sudo sysctl --system
cat /proc/sys/net/core/rmem_max   # expect 16777216
```

## Design

**Safety-critical execution stays local.** Wheel commands, watchdog, odometry, TF, and the SLAM / Nav2 loop all run on the Jetson. The workstation-hosted planner / VLM services are advisory: the executor can continue to refuse and cancel missions regardless of network state.

**All constants live in `strafer_shared`, not in ROS code.** Wheel radius, track width, velocity limits, depth clip range, PID values, RoboClaw addresses — every number that must stay consistent between sim and real is imported from `strafer_shared.constants`. Do not hardcode values in nodes, launch files, or config YAML.

**Motor sign inversion is applied exactly once, inside `strafer_shared.mecanum_kinematics`.** If a wheel spins backwards, fix the wiring per `docs/WIRING_GUIDE.md` — do not add a second sign flip in the driver, inference runtime, or autonomy code.

**The autonomy layer is not a ROS package.** `strafer_autonomy` is a pip-installed Python package that uses `rclpy` at runtime for the command server and ROS client. This keeps the mission runner, schemas, and HTTP clients testable in plain Python environments while still running inside the ROS 2 graph on the Jetson.

**Timestamp-fixed `*_sync` topics are authoritative for grounding and projection.** The RealSense D555 on Tegra USB produces hardware-clock timestamps that drift unpredictably (observed ~44× faster than system clock on Tegra). `timestamp_fixer` re-stamps all four camera topics with the current ROS clock at reception. `approximate_sync`-based consumers (RTAB-Map, `goal_projection_node`) must subscribe to the `*_sync` versions, never to the raw topics.

**URDF is the single source of truth for `base_link → d555_link`.** `strafer_perception` also publishes a static TF of the same transform — a historical wart to be removed. Consumers should treat the URDF transform as authoritative.

**`navigate_to_pose` stays backend-agnostic at the autonomy boundary.** Three backends dispatch: `nav2` (default), `strafer_direct` (pure-RL local control via `strafer_inference`), and `hybrid_nav2_strafer` (Nav2 global plan + RL local control following the subgoal generator). The inference node advertises its `navigate_to_pose` action server only when a policy loads; otherwise the dispatcher falls back to `nav2`.

## Testing

`colcon` drives package-level tests; run after build.

```bash
cd ~/strafer_ws
colcon test            # all packages — same set as `make test-ros`
colcon test-result --verbose
```

From the repo root:

```bash
make test-ros     # all colcon package tests (run `make build` first)
make test-driver  # strafer_driver unit tests directly via pytest (no colcon)
make test         # Jetson host: auto-dispatches to the test-jetson umbrella
                  # (test-autonomy + test-ros + test-driver)
```

End-to-end smoke checks with real hardware run through the validation scripts in `source/strafer_ros/` (see Run section).

## Deferred / known limitations

Tracked in [`docs/tasks/DEFERRED_WORK.md`](../../docs/tasks/DEFERRED_WORK.md). Items currently open:

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
- [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](../../docs/INTEGRATION_SIM_IN_THE_LOOP.md) — cross-host bringup runbook (Stage 3 covers `bringup_sim_in_the_loop.launch.py` consuming the DGX bridge).
- [`docs/tasks/DEFERRED_WORK.md`](../../docs/tasks/DEFERRED_WORK.md) — open items.
