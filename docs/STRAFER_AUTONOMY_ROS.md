# Strafer Autonomy ROS Runtime

This document is the canonical Jetson-side guide for the autonomy stack.

It is the source of truth for:
- what runs on the Jetson
- what remains in the ROS workspace
- what has moved to workstation-hosted services
- what work still needs to be done on the robot side

It is written for an agent or engineer continuing Jetson-side work.

## Scope

This document covers:
- `source/strafer_ros`
- Jetson-side installation of `source/strafer_shared`
- Jetson-side installation and execution of `source/strafer_autonomy`
- how the Jetson runtime interacts with remote planner and VLM services
- the remaining ROS-side work needed for the MVP and the end state

It does not cover:
- workstation-side Qwen training and evaluation details
- planner prompt engineering
- AWS or Databricks deployment details

Those are covered by the newer autonomy and deployment docs under `docs/`.

## Runtime Architecture

The Jetson-side autonomy architecture is:
- `strafer_autonomy.executor` runs on the Jetson
- planner service runs remotely, first on the Windows workstation
- grounding service runs remotely, first on the Windows workstation
- `strafer_ros` remains the robot-local execution layer
- `strafer_inference` remains the planned robot-local RL execution backend

## Runtime Split

### Robot-local on Jetson

- `strafer_driver`
- `strafer_perception`
- `strafer_description`
- `strafer_slam`
- `strafer_navigation`
- `strafer_msgs`
- `strafer_autonomy.executor`
- optional `strafer_inference` for RL policy execution

### Remote from the Jetson's point of view

- planner service
- `strafer_vlm` grounding service

### High-level MVP flow

```text
Operator -> ExecuteMission.action on Jetson
  -> strafer_autonomy.executor
  -> planner_client over LAN
  -> ros_client on Jetson
  -> vlm_client over LAN when grounding is needed
  -> strafer_ros skills and selectable local execution backends
```

## Monorepo Layout

Relevant paths:

```text
<repo-root>/
  source/
    strafer_lab/         # Isaac Lab simulation and RL training; do not modify from Jetson-side work
    strafer_ros/         # ROS2 packages and Jetson runtime assets
    strafer_shared/      # Shared constants, kinematics, and policy interface
    strafer_autonomy/    # Jetson-side executor + planner/VLM client layer
    strafer_vlm/         # Workstation/cloud grounding package
  docs/
    STRAFER_AUTONOMY_ROS.md
    STRAFER_AUTONOMY_COMMAND_INGRESS.md
    STRAFER_AUTONOMY_INTERFACES.md
    STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md
    STRAFER_AUTONOMY_ROADMAP.md
```

## Jetson Build And Install Model

The Jetson runtime uses two packaging styles:

1. ROS packages built with `colcon`
2. Python packages installed with `pip`

### ROS workspace

Build these from `source/strafer_ros`:
- `strafer_msgs`
- `strafer_driver`
- `strafer_perception`
- `strafer_description`
- `strafer_slam`
- `strafer_navigation`
- `strafer_bringup`

Recommended setup:

```bash
mkdir -p ~/strafer_ws/src
ln -s ~/strafer/source/strafer_ros/* ~/strafer_ws/src/

cd ~/strafer_ws
colcon build --symlink-install
source install/setup.bash
```

### Python packages

Install these into the same Jetson Python environment used for ROS nodes:
- `source/strafer_shared`
- `source/strafer_autonomy`

Recommended:

```bash
python -m pip install -e ~/strafer/source/strafer_shared
python -m pip install -e ~/strafer/source/strafer_autonomy
```

Important:
- `strafer_autonomy` is currently a Python package, not a ROS package
- it can still run ROS code because its `command_server` and CLI import `rclpy` lazily at runtime
- the ROS environment must be sourced before running those entry points

## Runtime Principles

These principles are part of the Jetson runtime contract and should not drift.

### 1. Simulation remains the source of truth

`source/strafer_lab` is still the reference for:
- physics parameters
- observation ordering
- normalization scales
- RL training assumptions

Do not fork Jetson-side constants away from the sim side.

### 2. Always use `strafer_shared`

Use `source/strafer_shared` for:
- robot constants
- mecanum kinematics
- policy loading
- observation assembly
- action interpretation

Do not hardcode these values in ROS or autonomy code.

### 3. The policy contract is still canonical

For RL inference work, the policy contract still lives in:
- `strafer_shared.policy_interface.PolicyVariant`
- `assemble_observation()`
- `interpret_action()`
- `load_policy()`

If Jetson-side code runs an Isaac-trained RL policy, it must use these helpers.

### 4. Motor direction handling

`wheel_axis_signs = [-1, 1, -1, 1]` for `[FL, FR, RL, RR]` are applied inside `strafer_shared.mecanum_kinematics`.

Do not apply wheel sign flips again in:
- `strafer_driver`
- `strafer_inference`
- autonomy code

If a motor direction is wrong physically, fix wiring rather than adding another sign correction layer.

### 5. Hardware and perception assumptions

Still true:
- Jetson Orin Nano on JetPack 6.2
- D555 RGB + aligned depth + BMI055 IMU
- two RoboClaw ST 2x45A controllers
- RoboClaw serial devices may appear as `/dev/ttyACM*` or optional udev aliases, depending on local setup
- timestamp-corrected D555 color stream
- aligned depth in the RGB frame

## Current Jetson-Side Components

| Component | Runtime location | Status | Purpose |
|----------|------------------|--------|---------|
| `strafer_msgs` | Jetson ROS workspace | Partial | custom ROS interfaces for autonomy and robot-side services |
| `strafer_driver` | Jetson ROS workspace | Done | RoboClaw control, odom, joint states, diagnostics |
| `strafer_perception` | Jetson ROS workspace | Done | D555 launch, timestamp fix, depth downsampling, IMU filtering |
| `strafer_description` | Jetson ROS workspace | Done | URDF, TF frames, robot_state_publisher |
| `strafer_slam` | Jetson ROS workspace | Launch/config only | RTAB-Map launch, depthimage_to_laserscan, and localization support |
| `strafer_navigation` | Jetson ROS workspace | Launch/config only | Nav2 bringup, parameter patching, and classical navigation backend config |
| `strafer_bringup` | Jetson ROS workspace | Done | layered launch files plus `validate_drive` smoke and validation tooling |
| `strafer_autonomy.executor` | Jetson Python runtime | Scaffolded | mission ingress, mission runner, robot-side orchestration |
| `strafer_inference` | Jetson ROS workspace | Planned | Isaac-trained RL policy execution backend on the Jetson |
| planner service | Workstation or cloud | Planned | natural-language command to bounded mission plan |
| `strafer_vlm` | Workstation or cloud | In progress | semantic grounding service |

Important current-state notes:
- the layered bringup launches still default to `/dev/ttyACM0` and `/dev/ttyACM1`, while the driver parameter file prefers `/dev/roboclaw0` and `/dev/roboclaw1` when udev rules are installed
- `strafer_slam` and `strafer_navigation` are currently launch/config packages; autonomy-facing projection and orientation services are still planned additions
- `strafer_inference` is not implemented yet, so RL execution remains an end-state backend rather than a present ROS capability

## ROS Topics And Interfaces That Still Matter

### Existing streaming inputs

These remain the main robot-local inputs that the Jetson autonomy layer should consume:

| Topic | Type | Purpose |
|------|------|---------|
| `/d555/color/image_sync` | `sensor_msgs/Image` | latest RGB frame for grounding |
| `/d555/color/camera_info_sync` | `sensor_msgs/CameraInfo` | RGB intrinsics |
| `/d555/aligned_depth_to_color/image_sync` | `sensor_msgs/Image` | timestamp-corrected aligned depth for 2D-to-3D projection |
| `/d555/aligned_depth_to_color/camera_info_sync` | `sensor_msgs/CameraInfo` | aligned-depth camera info in the RGB frame |
| `/d555/imu/filtered` | `sensor_msgs/Imu` | filtered IMU for robot state and RL inference |
| `/strafer/joint_states` | `sensor_msgs/JointState` | wheel states |
| `/strafer/odom` | `nav_msgs/Odometry` | robot-local pose and velocity |
| `/tf` and `/tf_static` | TF streams | frame transforms |

For projection and grounding handoff, prefer the timestamp-fixed `*_sync` camera topics rather than mixing synced RGB with raw aligned depth.

### New autonomy command ingress interfaces

These are the canonical robot-local command interfaces:

| Interface | Define in | Implement in | Status | Purpose |
|----------|-----------|--------------|--------|---------|
| `strafer_msgs/action/ExecuteMission.action` | `strafer_msgs` | `strafer_autonomy.command_server` | Added | submit, monitor, and cancel missions |
| `strafer_msgs/srv/GetMissionStatus.srv` | `strafer_msgs` | `strafer_autonomy.command_server` | Added | query active or last mission state |

### Remaining Jetson-side ROS interfaces to add

| Interface | Define in | Implement in | Purpose | Priority |
|----------|-----------|--------------|---------|----------|
| `strafer_msgs/srv/ProjectDetectionToGoalPose.srv` | `strafer_msgs` | `strafer_navigation` first | turn 2D grounding result into reachable `PoseStamped` goal | High |
| `strafer_msgs/msg/MissionStatus.msg` | `strafer_msgs` | publisher chosen later if needed | topic-based status for dashboards or logging | Medium |
| `strafer_msgs/action/OrientRelativeToTarget.action` | `strafer_msgs` | `strafer_navigation` first, or a future behavior package | face target / face away / lateral orientation behavior | Medium |

## Jetson-Side Package Responsibilities

### `strafer_msgs`

Owns shared ROS interfaces.

Current state:
- `ExecuteMission.action` exists
- `GetMissionStatus.srv` exists

Remaining work:
- add `ProjectDetectionToGoalPose.srv`
- optionally add `MissionStatus.msg`
- optionally add `OrientRelativeToTarget.action`

Clarification:
- `strafer_msgs` defines ROS interface types
- robot-local packages such as `strafer_navigation` implement the corresponding services and actions

### `strafer_driver`

Still owns:
- `/strafer/cmd_vel` to motors
- wheel state publication
- odom publication
- diagnostics
- watchdogs and basic safety at the driver boundary

This package should not know about:
- mission plans
- grounding
- planner output

### `strafer_perception`

Still owns:
- RealSense launch setup
- RGB timestamp correction
- aligned depth transport
- depth downsampling for RL if needed
- IMU filtering

Current state note:
- `strafer_perception/launch/perception.launch.py` still publishes a static `base_link -> d555_link` transform even though `strafer_description` already publishes the same transform from URDF
- treat the URDF transform as authoritative and the duplicate static transform as a current launch quirk to clean up later

This package should not own:
- semantic grounding
- mission planning
- goal generation

### `strafer_navigation`

This is the recommended first home for robot-side autonomy helpers that are not pure hardware drivers.

Current state:
- `strafer_navigation` currently provides Nav2 launch composition and parameter patching only
- it does not yet expose custom ROS nodes, services, or actions for autonomy

Recommended additions here:
- implement `strafer_msgs/srv/ProjectDetectionToGoalPose.srv`
- wrappers or helpers around the `nav2` execution backend
- future implementation of `strafer_msgs/action/OrientRelativeToTarget.action` if it stays close to navigation

Reason:
- it already owns Nav2 integration
- goal projection produces navigable target poses
- adding one service here is cheaper than creating a brand-new ROS package too early

### `strafer_inference`

This package is still relevant.

It is not replaced by `strafer_autonomy`.

Its role is different:
- `strafer_autonomy` decides what mission or goal should be executed
- `strafer_inference` would execute Isaac-trained RL navigation or behavior policies on the Jetson

Recommended interpretation:
- keep `strafer_inference` as the Jetson package for RL policy execution
- do not fold policy inference into `strafer_autonomy`
- do not treat Nav2 and RL as mutually exclusive forever

Practical recommendation:
- keep `navigate_to_pose` as the stable autonomy skill
- make the robot-side execution mode selectable from the start
- support three robot-local execution modes at the interface level:
  - `execution_backend="nav2"` — **MVP: only implemented backend**
  - `execution_backend="strafer_direct"` — post-MVP (requires `strafer_inference`)
  - `execution_backend="hybrid_nav2_strafer"` — post-MVP (requires `strafer_inference`)
- interpret the current `execution_backend` field as an execution-mode selector
- allow the default mode to remain `nav2` until the RL policy is ready, but do not make the interface Nav2-specific

Mode definitions:
- `nav2`
  - classical planning and control through Nav2
  - goal pose -> Nav2 planner/controller -> robot command path
- `strafer_direct`
  - direct RL execution through `strafer_inference`
  - goal pose or goal-relative target -> policy inference -> robot command path
  - intended for the path where RL replaces classical motion execution
- `hybrid_nav2_strafer`
  - Nav2 produces a higher-level path, waypoint sequence, or subgoals
  - `strafer_inference` executes the local motion policy against that guidance
  - intended for systems that keep classical global planning but use learned local control

### `strafer_autonomy`

This is now a Jetson-installed Python package, not a ROS workspace package.

Current scaffolded pieces:
- schemas
- planner/VLM/ROS client interfaces
- command ingress CLI
- command server around `ExecuteMission.action`
- mission runner

Jetson-side work still needed:
- implement `ros_client` against real ROS topics/services/actions
- add a runnable Jetson node entry point that builds the command server with concrete clients
- integrate launch and runtime docs

## MVP Jetson Architecture

```text
SSH CLI or later UI adapter
  -> ExecuteMission.action
  -> strafer_autonomy.command_server
  -> strafer_autonomy.mission_runner
  -> planner_client over LAN
  -> ros_client locally on Jetson
      -> capture_scene_observation from ROS topics
      -> project_detection_to_goal_pose via ROS service
      -> navigate_to_pose via selected local backend
  -> vlm_client over LAN when needed
```

### Key rule

The Jetson remains the owner of:
- mission state
- cancel semantics
- timeout behavior
- execution state
- final safety boundary

The workstation or cloud can plan and ground, but it does not own the control loop.

## Remaining Jetson Work

### Stage A: Finish robot-local ROS interfaces

Required:
1. define `ProjectDetectionToGoalPose.srv` in `strafer_msgs`
2. decide whether `MissionStatus.msg` is needed immediately
3. defer `OrientRelativeToTarget.action` unless navigation and projection are already working

### Stage B: Implement `strafer_autonomy.clients.ros_client`

Required behavior:
- cache RGB, aligned depth, camera info, odom, and TF locally
- expose `capture_scene_observation()`
- expose `get_robot_state()` — MVP returns `{"pose": ..., "nav_state": ...}` from odom + Nav2 status; battery and velocity are deferred until `DiagnosticStatus` aggregation is implemented
- call `ProjectDetectionToGoalPose.srv`
- dispatch `navigate_to_pose` to the selected execution mode
- support cancel for the currently active execution mode

This is the most important remaining Jetson-side Python task.

### Stage C: Implement the projection service

Recommended first placement:
- define `ProjectDetectionToGoalPose.srv` in `strafer_msgs`
- implement the first server in `strafer_navigation`

Required behavior:
- accept bounding box + image timestamp + standoff
- sample depth in the aligned RGB frame
- deproject to 3D using camera intrinsics
- transform target pose into `map`
- produce a reachable goal pose for navigation
- return quality flags when depth is sparse, invalid, or ambiguous

### Stage D: Run `strafer_autonomy` on the Jetson as a real runtime process

Needed pieces:
- concrete `planner_client`
- concrete `vlm_client`
- concrete `ros_client`
- command-server bootstrap entry point
- launch or service wrapper for robot startup

### Stage E: Integrate `strafer_inference` execution modes

This work should continue, but it is not blocked on the autonomy ingress scaffolding.

It is needed when:
- the Jetson should execute Isaac-trained RL navigation or skill policies directly
- the end state requires RL local controller behavior instead of or beneath Nav2

Once the policy is ready, the next Jetson task is to make `strafer_inference` a real `navigate_to_pose` execution mode in both:
- `strafer_direct`
- `hybrid_nav2_strafer`

### Stage F: Launch and bringup integration

Likely next additions after the executor works:
- new Jetson launch file or startup script for the autonomy runtime
- documentation for sourcing ROS + Python environment together
- operator CLI usage examples

## Recommended Jetson Implementation Order

1. `strafer_msgs/srv/ProjectDetectionToGoalPose.srv`
   - define in `strafer_msgs`
2. `strafer_autonomy.clients.ros_client`
3. projection service implementation in `strafer_navigation`
4. Jetson bootstrap for `strafer_autonomy.command_server`
5. LAN planner and VLM client implementations
6. launch integration
7. integrate `strafer_inference` as `strafer_direct`, then later `hybrid_nav2_strafer` if needed

## Testing Expectations On The Jetson

### Unit level

- `strafer_msgs` builds cleanly with `colcon`
- `ros_client` can be exercised with mocked message caches where possible
- projection logic should have deterministic tests for depth and TF edge cases

### Integration level

Bringup target:
- `strafer_driver`
- `strafer_perception`
- `strafer_description`
- `strafer_navigation`
- autonomy command server

Then validate:
1. `strafer_autonomy.cli submit ...` can send an `ExecuteMission` goal
2. mission status can be queried with `status`
3. a grounding response can be projected into a map-frame goal
4. the selected execution backend can receive and execute that goal
5. mission cancel works cleanly

### RL-path validation

When `strafer_inference` is implemented:
1. verify `assemble_observation()` and `interpret_action()` are used directly
2. verify sensor subscriptions and policy loading work on the Jetson
3. verify `/strafer/goal` or later goal interface drives policy behavior correctly

## Recommended Agent Assumptions For Future Jetson Work

An agent working on the Jetson side should assume:
- `strafer_vlm` is remote, not a Jetson ROS package
- `strafer_autonomy.executor` is robot-local and should stay robot-local
- `strafer_ros` remains the safety-critical execution layer
- `strafer_inference` is still part of the intended end state
- new ROS interfaces should be added only when they stabilize a boundary, not as speculative abstraction

## Execution Backend Decision

`navigate_to_pose` should be a stable autonomy skill with a swappable robot-local execution mode.

Decision:
- preserve one autonomy-level skill name: `navigate_to_pose`
- allow the robot side to execute that skill through:
  - `nav2`
  - `strafer_direct`
  - `hybrid_nav2_strafer`
- keep `nav2` as the initial default mode while the RL policy is still training
- do not bake Nav2 assumptions into the autonomy boundary

This keeps the current MVP practical while making it straightforward to:
- keep a classical mode
- add a direct RL mode
- add a hybrid classical-plus-RL mode

## Remaining Interface Question

One architecture question remains for the Jetson side:

What robot-local interface should `strafer_inference` expose when it becomes a robot-local execution mode provider?

Most likely options:
1. continue consuming a goal-like interface such as `/strafer/goal`
2. expose a more structured action or service so `ros_client` can manage status and cancellation uniformly across execution modes

My recommendation is option 2 in the long term, but it does not need to block the current Jetson-side scaffolding.

## Related Docs

- [STRAFER_AUTONOMY_COMMAND_INGRESS.md](STRAFER_AUTONOMY_COMMAND_INGRESS.md)
- [STRAFER_AUTONOMY_INTERFACES.md](STRAFER_AUTONOMY_INTERFACES.md)
- [STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md](STRAFER_AUTONOMY_MVP_RUNTIME_DECISION.md)
- [STRAFER_AUTONOMY_ROADMAP.md](STRAFER_AUTONOMY_ROADMAP.md)
- [SIM_TO_REAL_PLAN.md](SIM_TO_REAL_PLAN.md)
- [SIM_TO_REAL_TUNING_GUIDE.md](SIM_TO_REAL_TUNING_GUIDE.md)
- [WIRING_GUIDE.md](WIRING_GUIDE.md)
