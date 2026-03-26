# Sim-to-Real Tuning Guide

This guide is the deep-dive companion to `docs/SIM_TO_REAL_PLAN.md`.

Use:

- `SIM_TO_REAL_PLAN.md` for the training, export, and Jetson deployment path
- this guide for actuator, sensor, and timing alignment once policy transfer is being validated

How the Isaac Lab simulation models the real Strafer robot, and how to tune each component so the trained RL policy transfers cleanly.

**Key insight**: You don't need perfect alignment. The `Robust` training preset intentionally over-estimates noise and widens parameter ranges. If your real hardware falls within the Robust envelope, the policy should transfer. This guide helps you **verify** that the real hardware is within the trained distribution, and adjust the sim if it isn't.

**Current measured dimensions** (from `strafer_shared.constants`, updated after physical measurement):

| Constant | Value | Notes |
|----------|-------|-------|
| `WHEEL_RADIUS` | 0.048 m | 96mm mecanum wheel |
| `WHEEL_BASE` | 0.336 m | Front-to-rear axle, center-to-center |
| `TRACK_WIDTH` | 0.4284 m | Left-to-right axle, center-to-center (frame 360mm + 2x15.2mm axle + 38mm wheel) |
| `K` (lever arm) | 0.382 m | `WHEEL_BASE/2 + TRACK_WIDTH/2` -- used in kinematic matrix |
| `MAX_LINEAR_VEL` | 1.568 m/s | `WHEEL_RADIUS x MAX_WHEEL_ANGULAR_VEL` |
| `MAX_ANGULAR_VEL` | 4.10 rad/s | `MAX_LINEAR_VEL / K` |

---

## Table of Contents

1. [Architecture: Three Layers of Actuation](#1-architecture-three-layers-of-actuation)
2. [Layer 1: PhysX PD Controller -> RoboClaw Velocity PID](#2-layer-1-physx-pd-controller--roboclaw-velocity-pid)
3. [Layer 2: Motor Dynamics Filter](#3-layer-2-motor-dynamics-filter)
4. [Layer 3: Sensor Noise Models](#4-layer-3-sensor-noise-models)
5. [Tuning Procedure](#5-tuning-procedure)
6. [Sim-to-Real Validation Checklist](#6-sim-to-real-validation-checklist)
7. [Source File Reference](#7-source-file-reference)

---

## 1. Architecture: Three Layers of Actuation

Between a policy action and actual wheel motion, there are three distinct layers in simulation. Each has a real-world counterpart:

```text
Policy output: [-1, 1] normalized [vx, vy, omega]
    |
    v
+-----------------------------------+    <- USB serial latency
| LAYER 2a: Command Delay           |
| Circular buffer, 1-3 steps        |
+-----------------------------------+
| LAYER 2b: Slew Rate Limiting      |    <- Motor + gearbox inertia
| Max acceleration clamp            |
+-----------------------------------+
| LAYER 2c: Motor Dynamics Filter   |    <- Combined electrical + mechanical tau
| First-order low-pass, tau = 50ms  |
+-----------------------------------+
    |
    v  velocity target (rad/s)
+-----------------------------------+    <- RoboClaw velocity PID
| LAYER 1: PhysX PD Controller      |
| torque = D x (target - current)   |
| clamped by DC motor curve         |
+-----------------------------------+
    |
    v  torque -> PhysX rigid body dynamics
Wheel motion
```

The policy was trained with all three layers active (under the `Realistic` and `Robust` presets). On the real robot, Layers 2a-2c don't exist in software -- they emerge naturally from USB latency, motor inertia, and RoboClaw PID response. Your job is to verify the real-world values fall within the trained ranges.

---

## 2. Layer 1: PhysX PD Controller -> RoboClaw Velocity PID

### What Isaac Sim does

The robot's `DCMotorCfg` in [strafer.py](source/strafer_lab/strafer_lab/assets/strafer.py) defines:

```python
DCMotorCfg(
    stiffness=0.0,          # P gain (position) -- ZERO, no position control
    damping=10.0,            # D gain (velocity tracking)
    saturation_effort=2.38,  # Stall torque (N*m) -- hard limit
    velocity_limit=32.67,    # No-load speed (rad/s) -- back-EMF limit
)
```

With `stiffness=0.0`, this is a **D-only velocity controller**. Each physics step, PhysX computes:

```
torque = damping x (velocity_target - velocity_current)
       = 10.0 x (omega_target - omega_current)
```

This torque is then clamped by the DC motor torque-speed curve, which models back-EMF:

```
torque_max(omega) = stall_torque x (1 - |omega| / no_load_speed)
              = 2.38 x (1 - |omega| / 32.67)
final_torque  = clamp(torque, -torque_max, +torque_max)
```

At zero speed, full stall torque (2.38 N*m) is available. As speed approaches 32.67 rad/s (312 RPM), available torque drops linearly to zero. This is physically accurate for a brushed DC motor.

### What the real robot does

RoboClaw's `SpeedM1`/`SpeedM2` commands run a closed-loop velocity PID controller at ~300 Hz internally. The encoder provides feedback, and the RoboClaw adjusts PWM duty cycle to track the commanded velocity in ticks/sec.

### Mapping

| Aspect | Isaac Sim (PhysX) | RoboClaw (real) |
|--------|-------------------|-----------------|
| Controller type | D-only (damping x velocity error) | Full PID (QPPS-based) |
| Torque/current limit | DC motor curve model | Physical motor + RoboClaw current limit |
| Update rate | Physics dt (200 Hz typical) | ~300 Hz internal |
| Feedback source | Joint velocity from physics solver | Encoder ticks from hardware |
| Motor model | Linear torque-speed curve | Real electromagnetic + friction |

### Why exact gain matching isn't needed

The sim's `damping=10.0` determines how aggressively the *simulated* motor tracks velocity. On the real robot, RoboClaw's PID serves this role. What matters is that the **closed-loop step response** is similar -- specifically settling time, overshoot, and steady-state error. The policy doesn't "know" about the controller gains; it only observes the resulting wheel velocities through encoder readings.

### Tuned PID values (applied automatically on startup)

The RoboClaw PID has been tuned and is now written to RAM automatically on every `roboclaw_node` startup and by all test scripts. Current values from `strafer_shared.constants`:

```python
ROBOCLAW_PID_P = 15000
ROBOCLAW_PID_I = 750
ROBOCLAW_PID_D = 0
ROBOCLAW_QPPS  = 2796   # max ticks/sec at 312 RPM (537.7 PPR)
```

Use `source/strafer_ros/tune_pid.py` to re-run characterization and adjust if needed. If you retune, update these constants in `strafer_shared/constants.py` so all nodes and scripts stay synchronized.

**Target response characteristics**:
- **Settling time**: 50-80ms (matching sim's motor time constant range)
- **Overshoot**: <5% (policy wasn't trained with significant overshoot)
- **Steady-state error**: <2% of commanded velocity

---

## 3. Layer 2: Motor Dynamics Filter

The sim-to-real processing pipeline in [actions.py:309-326](source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L309-L326) applies three stages before the velocity target reaches PhysX. These model real-world imperfections that exist naturally in hardware.

### 2a. Command Delay

**Sim**: A circular buffer delays action commands by N physics steps.

```python
# From REAL_ROBOT_CONTRACT (sim_real_cfg.py:431-432)
action_latency_steps = 1      # 33ms at 30 Hz
action_latency_steps_range = (0, 2)  # 0-66ms random, sampled per reset
```

**Real-world source**: USB serial write -> RoboClaw command processing -> motor response. Typical round-trip: 2-5ms per command x 4 motors = 8-20ms total. At 50 Hz control rate (20ms period), this is roughly 0.5-1 step of delay.

**How to measure**: Timestamp the serial write and the first encoder velocity change. Use a tight Python loop:
```python
t0 = time.perf_counter()
roboclaw.speed_m1(address, 1000)  # command
while True:
    speed, status = roboclaw.read_speed_m1(address)
    if abs(speed) > 50:  # threshold for "motor started moving"
        latency = time.perf_counter() - t0
        break
```

**Tuning**: If real latency is ~10-20ms at your control rate, the sim's 0-66ms range already covers it with margin. No adjustment needed unless real latency exceeds 66ms.

### 2b. Slew Rate Limiting

**Sim**: Clamps the maximum velocity change per step.

```python
# From REAL_ROBOT_CONTRACT (sim_real_cfg.py:444)
max_acceleration_rad_s2 = 100.0  # rad/s^2 -> max_delta = 100 x dt per step
```

**Real-world source**: Motor + 19.2:1 gearbox inertia limits how fast velocity can change. The reflected inertia through the gearbox (J_reflected = J_motor x ratio^2) creates a natural slew rate.

**How to measure**: Command an instantaneous velocity step (0 -> max speed) and record encoder velocity at high rate. The slope of the velocity ramp gives you the real acceleration limit.

**Tuning**: If the real acceleration is ~80-120 rad/s^2, the sim value is already well-matched. The `Robust` preset uses 80 rad/s^2 for extra conservatism.

### 2c. First-Order Motor Dynamics (Most Important)

**Sim**: Exponential smoothing filter modeling combined motor response.

```python
alpha = dt / (tau + dt)  # tau = 0.05s (50ms) for Realistic
smoothed = alpha x target + (1 - alpha) x smoothed
```

This is the **most critical parameter** for sim-to-real transfer. It determines how quickly commanded velocity changes actually take effect.

**Real-world source**: The combined time constant of:
- Motor electrical time constant (L/R ~= 2-5ms for brushed DC)
- Gearbox compliance and backlash
- RoboClaw PID loop response time
- Mechanical friction and inertia

**How to measure** (the tau extraction procedure):
1. Command a velocity step: 0 -> 1000 ticks/s via `SpeedM1`
2. Log encoder velocity at the highest rate possible (~100+ samples/sec)
3. Find the time when velocity reaches **63.2%** of the target (632 ticks/s)
4. That time is tau by definition of first-order systems
5. Repeat at multiple speeds and under load

**Sim preset values**:

| Preset | tau nominal | tau range (randomized) |
|--------|-----------|----------------------|
| Ideal | N/A (disabled) | N/A |
| Realistic | 50ms | 30-80ms |
| Robust | 60ms | 20-100ms |

**Tuning**:
- If real tau is 40ms -> within Realistic range, no change needed
- If real tau is 15ms -> below Robust range, lower `motor_time_constant_range[0]` in sim
- If real tau is 120ms -> above Robust range, either retrain with wider range or tune RoboClaw PID to reduce response time
- RoboClaw PID tuning directly affects the effective tau -- a well-tuned PID can bring it to 30-50ms

---

## 4. Layer 3: Sensor Noise Models

Each sensor in simulation has a noise model based on real hardware datasheets. The policy was trained to be robust to these noise levels. All sensor noise configs are in [sim_real_cfg.py](source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py).

### IMU (BMI055 in RealSense D555)

The BMI055 is the IMU inside the D555. Noise parameters are derived from its datasheet.

| Parameter | Sim (Realistic) | Sim (Robust) | Datasheet | Unit |
|-----------|-----------------|--------------|-----------|------|
| Accel noise density | 0.0098 | 0.015 | 0.98 mg/sqrtHz = 0.0096 m/s^2/sqrtHz | m/s^2/sqrtHz |
| Accel bias stability | 0.04 | 0.06 | 40 ug | m/s^2 |
| Gyro noise density | 0.00024 | 0.00036 | 0.014 deg/s/sqrtHz = 0.000244 rad/s/sqrtHz | rad/s/sqrtHz |
| Gyro bias stability | 0.0017 | 0.0025 | 0.1 deg/s | rad/s |

**Conversion from noise density to per-sample std**:
```
sigma_sample = noise_density x sqrt(sample_rate_hz)
```
At 30 Hz control rate: accel sigma ~= 0.054 m/s^2, gyro sigma ~= 0.0013 rad/s.

**How to characterize your D555 IMU**:
1. Place the robot on a flat, stable surface
2. Record `/d555/imu/filtered` data for 30 seconds (stationary)
3. Compute:
   - **Accel bias**: mean of each axis minus gravity (Z should be ~9.81 m/s^2)
   - **Accel noise sigma**: std of each axis
   - **Gyro bias**: mean of each axis (should be near zero)
   - **Gyro noise sigma**: std of each axis
4. Compare against sim values above

**If real noise is higher**: Increase sim values. The Robust preset gives 1.5x margin.
**If real noise is lower**: No change needed -- training with more noise than reality is fine.

### Encoders (GoBilda 5203, 537.7 PPR)

| Parameter | Sim (Realistic) | Sim (Robust) | Unit |
|-----------|-----------------|--------------|------|
| Velocity noise sigma | 0.02 | 0.05 | fraction of max velocity |
| Missed tick probability | 0.001 | 0.005 | per tick per step |
| Quantization | 537.7 PPR | 537.7 PPR | ticks/revolution |

At 537.7 PPR, the quantization resolution is 0.0117 rad/tick. Velocity is estimated by differencing encoder counts, so velocity noise depends on the sampling period.

**How to characterize**:
1. Spin each wheel at a constant moderate speed (~1000 ticks/s) using RoboClaw's velocity PID
2. Read `ReadSpeedM1`/`ReadSpeedM2` at ~100 Hz for 5 seconds
3. Compute velocity sigma
4. Normalize: sigma_normalized = sigma_measured / ENCODER_VEL_MAX (3000 ticks/s)
5. Compare against sim's `velocity_noise_std`

**Typical results**: At 1000 ticks/s, expect sigma ~= 10-30 ticks/s, giving sigma_normalized ~= 0.003-0.01. This is often *lower* than the sim's 0.02, which is fine.

### Depth Camera (RealSense D555)

The depth noise model uses the Intel RealSense stereo error propagation formula:

```
sigma_z = (z^2 / (f x B)) x sigma_d
```

Where:
- `z` = depth in meters
- `f` = focal length in pixels (673 px at 1280x720)
- `B` = stereo baseline (0.095 m for D555)
- `sigma_d` = subpixel disparity noise (0.08 px typical)

This means noise **grows quadratically with distance**:

| Distance | sigma_z (Realistic) | sigma_z (Robust) |
|----------|-----------------|--------------|
| 0.5 m | 0.3 mm | 0.6 mm |
| 1.0 m | 1.3 mm | 2.5 mm |
| 2.0 m | 5.0 mm | 10.0 mm |
| 4.0 m | 20.0 mm | 40.0 mm |

**How to characterize**:
1. Point the D555 at a flat wall at known distances (0.5m, 1m, 2m, 4m)
2. Record depth frames for 5 seconds at each distance
3. Crop a central patch (e.g., 20x20 pixels) from the 80x60 downsampled image
4. Compute sigma of depth values across frames
5. Plot sigma vs z^2 -- should be linear. The slope gives `sigma_d / (f x B)`.

**For MVP (NoCam variant)**: Depth tuning doesn't apply. Handle this when you add `PolicyVariant.DEPTH`.

### Per-Sensor Latency

Different sensors have different processing pipelines:

| Sensor | Sim Latency (Realistic) | Real-World Source |
|--------|------------------------|-------------------|
| IMU | 0 steps (< 2ms) | Direct I2C read, minimal processing |
| Encoders | 0 steps (< 5ms) | RoboClaw reads on serial bus |
| Depth camera | 1 step (~33ms) | Stereo matching + USB transfer |
| RGB camera | 1 step (~33ms) | Image processing + USB transfer |

These are modeled as per-sensor observation delays in [sim_real_cfg.py:73-84](source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L73-L84). The values match typical RealSense D555 behavior and generally don't need tuning.

---

## 5. Tuning Procedure

Follow this sequence. Each step builds on the previous.

### Step 1: RoboClaw PID [done] (already tuned)

PID is auto-applied on every startup (P=15000, I=750, D=0, QPPS=2796 from `constants.py`). Re-characterize only if behavior changes or after hardware swap.

Use `source/strafer_ros/tune_pid.py` to run a step-response sweep and verify:
- Settling time < 80ms
- Overshoot < 5%
- Steady-state error < 2%

**Pass criteria**: Step response resembles a first-order system with tau ~= 30-80ms.

### Step 2: Motor Time Constant (tau)

Extract tau from the step response measured in Step 1.

1. Command velocity step: 0 -> 1000 ticks/s
2. Log encoder velocity at >=100 Hz
3. Find time to reach 63.2% of target -> that's tau
4. Repeat for each motor (they may differ slightly due to friction/wiring)

**Pass criteria**: tau within [30ms, 80ms] (Realistic range) or [20ms, 100ms] (Robust range).

**If tau is outside range**: Adjust `motor_time_constant_range` in `sim_real_cfg.py` and retrain, or tune RoboClaw PID to bring tau into range.

### Step 3: Command Latency

1. Timestamp serial write -> first encoder movement
2. Measure at 50 Hz control rate (typical ROS node rate)

**Pass criteria**: Total command latency < 66ms (2 steps at 30 Hz).

### Step 4: IMU Characterization

1. Static recording, 30 seconds
2. Compute bias (mean) and noise (std) for accel and gyro

**Pass criteria**: Noise sigma within Robust envelope (see table in Section 4).

### Step 5: Encoder Noise

1. Constant-speed test, 5 seconds per motor
2. Compute velocity sigma, normalize by 3000 ticks/s

**Pass criteria**: sigma_normalized < 0.05 (Robust `velocity_noise_std`).

### Step 6: End-to-End Validation (the real test)

This is where you verify that everything works together.

1. **Square test**: Command a 1m x 1m square path via `cmd_vel` sequences:
   - Forward 1m (vx=0.3 m/s for ~3.3s)
   - Strafe left 1m (vy=0.3 m/s for ~3.3s)
   - Backward 1m (vx=-0.3 m/s for ~3.3s)
   - Strafe right 1m (vy=-0.3 m/s for ~3.3s)
2. Record odometry throughout
3. Run the same command sequence in Isaac Lab with `Realistic` preset
4. Compare:
   - **Path shape**: Should both be roughly square
   - **Final position error**: Real robot should return within 0.3m of start
   - **Velocity profiles**: Encoder velocity vs. time should have similar shape and time constants

**Pass criteria**: Sim and real trajectories are qualitatively similar. The policy's robustness training compensates for small differences.

---

## 6. Sim-to-Real Validation Checklist

Use this checklist before deploying a trained policy on hardware.

### Actuator Alignment

- [ ] RoboClaw PID tuned: settling < 80ms, overshoot < 5%, SS error < 2%
- [ ] Motor tau measured for all 4 motors: within [20ms, 100ms]
- [ ] Command latency measured: < 66ms total
- [ ] All 4 motors spin in correct direction (verified by wiring test script)
- [ ] Encoder counts increase in expected direction for positive velocity commands

### Sensor Alignment

- [ ] IMU accel noise sigma measured: < 0.08 m/s^2 (Robust threshold)
- [ ] IMU gyro noise sigma measured: < 0.002 rad/s (Robust threshold)
- [ ] IMU frame orientation matches sim (x-forward, y-left, z-up in body frame)
- [ ] Encoder velocity noise sigma_normalized measured: < 0.05
- [ ] Depth camera noise characterized (if using DEPTH variant)

### Integration

- [ ] Square test: sim vs. real trajectories qualitatively match
- [ ] Observation vector assembled correctly: `assemble_observation()` produces correct dims
- [ ] Action denormalization correct: `interpret_action([1,0,0])` -> forward at MAX_LINEAR_VEL
- [ ] Watchdog works: motors stop within 500ms of last `cmd_vel`

### Contract Compliance

- [ ] All constants imported from `strafer_shared.constants` (no hardcoded values)
- [ ] All kinematics from `strafer_shared.mecanum_kinematics` (no reimplementation)
- [ ] Observation assembly via `strafer_shared.policy_interface.assemble_observation()`
- [ ] Action interpretation via `strafer_shared.policy_interface.interpret_action()`

---

## 7. Source File Reference

| File | Relevant Content |
|------|-----------------|
| [sim_real_cfg.py](source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py) | All noise/timing/actuator presets (Ideal, Realistic, Robust) |
| [strafer.py](source/strafer_lab/strafer_lab/assets/strafer.py) | DCMotorCfg: stiffness, damping, torque-speed curve |
| [actions.py](source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py) | Motor dynamics filter, command delay, slew rate (L309-326) |
| [constants.py](source/strafer_shared/strafer_shared/constants.py) | All robot physical constants, PID gains, normalization scales |
| [policy_interface.py](source/strafer_shared/strafer_shared/policy_interface.py) | Observation assembly, action interpretation |
| [tune_pid.py](source/strafer_ros/tune_pid.py) | Interactive PID characterization and step-response sweep |

### Sim-to-Real Contract Presets Summary

| Parameter | Ideal | Realistic | Robust |
|-----------|-------|-----------|--------|
| Motor dynamics | Disabled | tau=50ms, range [30,80]ms | tau=60ms, range [20,100]ms |
| Command delay | 0 steps | 1 step, range [0,2] | 1 step, range [0,4] |
| Slew rate limit | infinity | 100 rad/s^2 | 80 rad/s^2 |
| Control jitter | 0% | +/-5% | +/-10% |
| IMU accel noise density | 0 | 0.0098 m/s^2/sqrtHz | 0.015 m/s^2/sqrtHz |
| IMU gyro noise density | 0 | 0.00024 rad/s/sqrtHz | 0.00036 rad/s/sqrtHz |
| Encoder velocity noise sigma | 0 | 0.02 | 0.05 |
| Depth disparity noise | 0 | 0.08 px | 0.16 px |
| Sensor failures | Disabled | Disabled | Enabled |
| Domain randomization | 0x | 1x | 1.5x |

