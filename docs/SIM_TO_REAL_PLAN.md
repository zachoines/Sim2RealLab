# Strafer Robot: Sim-to-Real Deployment Plan

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

**Udev rules** for persistent device names (`/dev/roboclaw_front`, `/dev/roboclaw_rear`).

**Motor direction**: `wheel_axis_signs = [-1, 1, -1, 1]` (FL, FR, RL, RR) from [strafer_env_cfg.py:183](source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py#L183). Apply sign correction in software, not by swapping wires.

---

## 3. ROS2 Package Architecture

```
~/strafer_ws/src/
  strafer_msgs/            # Custom messages (WheelVelocities, EncoderTicks, PolicyAction)
  strafer_description/     # URDF/xacro, TF frames, robot_state_publisher
  strafer_driver/          # RoboClaw node: cmd_vel -> motors, encoders -> JointState + odom
  strafer_perception/      # RealSense launch wrapper + depth downsampler (80x60)
  strafer_inference/       # ONNX/TensorRT policy inference node
  strafer_slam/            # RTAB-Map configuration
  strafer_navigation/      # Nav2 integration, behavior trees
  strafer_bringup/         # Composed launch files
```

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

### strafer_inference: `policy_inference_node`

| Direction | Topic | Type | Rate |
|-----------|-------|------|------|
| Sub | `/d555/imu` | `sensor_msgs/Imu` | 200 Hz |
| Sub | `/strafer/joint_states` | `sensor_msgs/JointState` | 50 Hz |
| Sub | `/strafer/depth/policy_input` | `sensor_msgs/Image` | 30 Hz |
| Sub | `/strafer/goal` | `geometry_msgs/PoseStamped` | -- |
| Pub | `/strafer/cmd_vel` | `geometry_msgs/Twist` | 30 Hz |

Assembles observation vector in **exact simulation order**, applies normalization, runs ONNX inference, denormalizes action output.

---

## 4. Controller Abstraction: Shared Kinematics

A pure-Python `mecanum_kinematics.py` module shared between sim and real, containing:

- Constants from simulation: `WHEEL_RADIUS=0.048`, `WHEEL_BASE=0.304`, `TRACK_WIDTH=0.304`, `MAX_WHEEL_ANGULAR_VEL=32.67`, `ENCODER_PPR=537.7`, `WHEEL_AXIS_SIGNS=[-1,1,-1,1]`
- Forward kinematics: `[vx, vy, omega]` -> `[w_fl, w_fr, w_rl, w_rr]` (rad/s)
- Inverse kinematics: wheel velocities -> body velocity (for odometry)
- Unit conversions: rad/s <-> ticks/sec

The kinematic matrix replicates [actions.py:166-171](source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L166-L171) exactly.

**What the real robot does NOT replicate from simulation**: motor dynamics filter (real motor has physical inertia), command delay (real USB latency exists naturally), slew rate limiting (RoboClaw PID handles this). The policy was trained with these dynamics, so it produces compatible commands.

---

## 5. Observation Assembly (Critical for sim-to-real transfer)

The inference node must replicate the exact observation order and normalization from [strafer_env_cfg.py:249-257](source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py#L249-L257):

| Index | Field | Real Source | Scale (from [strafer_env_cfg.py:225-234](source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py#L225-L234)) |
|-------|-------|-------------|-------|
| 0-2 | `imu_linear_acceleration` | D555 IMU `linear_acceleration` (m/s^2) | 1/156.96 |
| 3-5 | `imu_angular_velocity` | D555 IMU `angular_velocity` (rad/s) | 1/34.9 |
| 6-9 | `wheel_encoder_velocities` | RoboClaw `ReadSpeedM1/M2` (ticks/s) | 1/3000.0 |
| 10-11 | `goal_position_relative` | TF transform base_link->goal (m) | 1.0 |
| 12-14 | `last_action` | Previous policy output | 1.0 |
| 15-4814 | `depth_image` (if Depth variant) | D555 depth, 80x60, float32 meters | 1/6.0 |

Encoder velocities from `JointState` (rad/s) must be converted to ticks/sec via `RADIANS_TO_ENCODER_TICKS = 85.57` from [observations.py:32](source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py#L32) before scaling.

---

## 6. Policy Export Pipeline

1. **Train**: `Isaac-Strafer-Nav-Real-NoCam-v0` (15-dim obs, start without camera dependency)
2. **Export**: Reconstruct actor MLP `[256, 256, 128]` with ELU activation (from [rsl_rl_ppo_cfg.py:14](source/strafer_lab/strafer_lab/tasks/navigation/agents/rsl_rl_ppo_cfg.py#L14)), load `actor.` weights from RSL-RL checkpoint, export via `torch.onnx.export`
3. **Optimize**: TensorRT FP16 on Jetson Orin Nano via `trtexec` or ONNX Runtime TensorRT EP
4. **Infer**: ONNX Runtime with `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']` fallback chain. Expected latency: <5ms for a 15-dim MLP.

---

## 7. SLAM & Nav2 Integration

**SLAM**: RTAB-Map (`ros-humble-rtabmap-ros`) with D555 RGB-D + IMU. Produces `/map` and `map->odom` TF.

**Sensor fusion**: `robot_localization` EKF fusing wheel odometry (50Hz) + D555 IMU (200Hz) for `odom->base_link` TF.

**Nav2 operating modes**:
- **Mode 1 (initial)**: Pure RL policy. `policy_inference_node` publishes directly to `cmd_vel`. No Nav2.
- **Mode 2**: Nav2 with RL as local controller plugin. Nav2 handles global planning + behavior trees; RL policy handles low-level control.
- **Mode 3 (future)**: Hybrid switching between traditional controller and RL based on context.

---

## 8. Implementation Phases

### Phase 1: Jetson Orin Nano Setup + Hardware Bring-Up
- [x] Flash JetPack 6.2 (done -- Orin Nano on network, SSH accessible)
- [ ] Set up VS Code Remote-SSH workspace (Windows <-> Jetson)
- [ ] Install ROS2 Humble
- [ ] Create udev rules, test RoboClaw serial communication
- [ ] Verify each motor spins correctly, encoders respond
- [ ] Install librealsense2 (from source with CUDA), verify D555 streams
- **Test**: standalone Python scripts for motor/encoder/camera validation

### Phase 2: ROS2 Driver Packages
- [ ] Create `strafer_msgs`, `strafer_driver` (roboclaw_node), `strafer_perception` (RealSense wrapper + depth downsampler)
- [ ] Implement shared `mecanum_kinematics.py`
- **Test**: keyboard teleop drives robot, odometry publishes, drive a 1m square

### Phase 3: URDF + TF Tree
- [ ] Create `strafer_description` with xacro matching sim dimensions
- [ ] Export STL meshes from USD assets in [Assets/3209-0001-0006-v6/](Assets/3209-0001-0006-v6/)
- **Test**: RViz2 shows correct robot model with live TF, camera frame matches RealSense intrinsics

### Phase 4: Policy Export + Inference
- [ ] Train policy on workstation with `Isaac-Strafer-Nav-Real-NoCam-v0`
- [ ] Export to ONNX, optimize with TensorRT on Jetson Orin Nano
- [ ] Create `strafer_inference` with `policy_inference_node`
- **Test**: fixed goal -> robot drives toward it, inference <5ms

### Phase 5: Integration Testing
- [ ] RoboClaw PID tuning to approximate sim's 50ms motor time constant
- [ ] Waypoint following accuracy (<0.3m error)
- [ ] Speed calibration, disturbance rejection
- [ ] Sim-real gap analysis: compare trajectories
- **Test**: 10-waypoint course, 30-minute endurance run

### Phase 6: SLAM + Nav2
- [ ] Install RTAB-Map, robot_localization, Nav2
- [ ] Create `strafer_slam`, `strafer_navigation`, `strafer_bringup`
- [ ] Build map, test localization, configure costmaps
- [ ] Integrate RL as Nav2 local controller
- **Test**: autonomous navigation in mapped environment with obstacle avoidance

---

## 9. Critical Source Files

| File | What to reference |
|------|-------------------|
| [actions.py](source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py) | Kinematic matrix (L166-171), velocity scaling (L173-179), motor dynamics |
| [strafer_env_cfg.py](source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py) | Observation order (L249-258), normalization constants (L225-234), camera config (L103-121) |
| [sim_real_cfg.py](source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py) | REAL_ROBOT_CONTRACT timing/noise params (L419-477) |
| [observations.py](source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py) | Encoder tick conversion (L28-33), sensor data extraction |
| [rsl_rl_ppo_cfg.py](source/strafer_lab/strafer_lab/tasks/navigation/agents/rsl_rl_ppo_cfg.py) | Network architecture: [256,256,128] ELU (L14-16) |
| [strafer.py](source/strafer_lab/strafer_lab/assets/strafer.py) | Robot ArticulationCfg, motor specs, joint names |
