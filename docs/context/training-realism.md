# Training realism: what each tier perturbs

Navigation training perturbs the actuation, the sensors and the world on one of three tiers, `ideal` (nothing),
`realistic` and `robust`, set by `RealismCfg(level=...)`
([`composed_env_cfg.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py)) and built from the
tier contracts `IDEAL_SIM_CONTRACT`, `REAL_ROBOT_CONTRACT` and `ROBUST_TRAINING_CONTRACT`
([`sim_real_cfg.py:707-709`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L707)). The RL IDs
named `…-Real-…` train on `realistic`, those named `…-Robust-…` on `robust`; the plane scaffolds `StraferNavCfg_*_Ideal`
use `ideal`. Each row below says what a perturbation stands for on the deployed robot (the Jetson's ROS 2 stack, the
D555 camera and its IMU, the RoboClaw-driven wheels) and the basis of its value. The values are pinned to the code by
[`test_training_realism_doc.py`](../../source/strafer_lab/tests/navigation/test_training_realism_doc.py); a change to
one fails that test until this page follows.

Basis: **measured** (on the real robot or its sensor, or on the sim-in-the-loop rig where said), **datasheet**,
**derived** (computed from a measured or datasheet value), **design** (a chosen value or band), **unknown** (no source
for the value). The control step is 1/30 s and the physics step 1/120 s on every tier.

## Timing and actuation

The action term ([`mdp/actions.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py)) runs once
per control step: delay, then hold, then the acceleration limit, then the motor lag, then the twist clamp.

| perturbation | ideal | realistic | robust | models | basis |
|---|---|---|---|---|---|
| [Command delay](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L192) | off | 1–3 control steps | 1–5 control steps | time from the policy's command to the wheels: topic transport, the driver, the motor controller's serial link | design; not measured. One delay per env, drawn when the action term is built, not at a reset |
| [Command hold](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L215) | off | a fraction U(0, 0.05) of ticks per env, runs of 1–1.2 | same | a tick with no new command, on which the chassis repeats the last one | design; kept small because a held depth frame and a held command are the same event on the deployed node |
| [Acceleration limit](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L237) | off | 175–275 rad/s² per wheel, drawn per env at each reset (field: 700–1100) | 125–300 rad/s² (field: 500–1200) | how fast a wheel's speed can change under torque, load and the battery's state | unknown for the nominal 225 (field: 900); the band is design. The limit is `accel × physics_dt` per control step, so the field's value is 4× the trained one |
| [Motor lag](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L186) | off | first-order, 63 % rise in 0.216 s (field τ 0.05 s) | 0.256 s (field τ 0.06 s) | the motor controller's velocity loop and the motor | unknown; not measured. α = `physics_dt / (τ + physics_dt)` per control step, a continuous-equivalent τ of 0.200 / 0.240 s |
| [Command envelope](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L272) | off | off | s ~ U(0.255, 1.0) per env at each reset, on both caps; the policy does not observe s | the deploy node caps the twist at s = 0.5 of the full envelope | design: a band holding the deploy cap |
| [Twist clamp](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L362) | \|vx\| + \|vy\| ≤ 1.568 m/s, \|ω\| ≤ 4.186 rad/s | same | same, × s | the deploy node's clamp, the same function (`l1_clamp_twist_batched`) | derived from the motor's rated speed and the chassis geometry |
| [Wheel speed and torque](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L456) | 32.67 rad/s; 2.383 N·m | same | same | the drive motor's rated speed (312 rpm) and stall torque | datasheet |
| [Motor strength](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L454) | off | torque limit × U(0.92, 1.08) per wheel, per reset | × U(0.8, 1.2) | motor-to-motor and battery-voltage torque spread | design |
| Control period | 1/30 s | same | same | the deploy node's 30 Hz tick and the D555's 30 Hz stream | design; not randomized on any tier |

Declared, not applied: the contract fields `motor_time_constant_range` and `domain_randomization_scale`
([`sim_real_cfg.py:125`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L125),
[`:492`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L492)) reach no env.

## Sensing

The noise models are in
[`mdp/noise_models.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py). The robust tier's
IMU noise and bias are 1.5× the realistic tier's.

| perturbation | ideal | realistic | robust | models | basis |
|---|---|---|---|---|---|
| [IMU noise](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L182) | off | white 0.0098 m/s²/√Hz and 0.00024 rad/s/√Hz; initial bias ±0.04 m/s², ±0.0017 rad/s; bias random walk | 0.015, 0.00036; ±0.06, ±0.0025 | the D555's IMU | derived from the IMU's stated noise density; the bias values unknown |
| IMU and encoder latency | off | none | 1 control step | the age of a reading at the policy | design |
| [Wheel encoder noise](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L356) | off | σ 0.02 of 3000 ticks/s; a missed tick 0.001, an extra 0.0005; integer ticks/s | σ 0.05; missed 0.005 | the quadrature encoders' speed estimate | design |
| Sensor dropouts | off | none | per step: the IMU reads zero or repeats 1e-4; the encoders read zero 1e-4; the depth frame reads far 1e-3 | a sensor that stops reporting | design |
| [Depth disparity noise](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L479) | off | σ_d 0.08 px, independent per 80×45 cell after the block median | σ_d per env, log-uniform on [0.002, 0.16] px | the D555's stereo depth error | 0.08 px is in the stereo-matching range for this camera and passes the amplitude check against the real unit at mount height; the robust band is design |
| Depth holes and lost frames | off | holes 0.01 of cells; a frame lost 0.001 | 0.03; 0.01 | invalid pixels and dropped frames | design |
| Depth stream hold | off | a fraction U(0, 0.35) of ticks per env, runs of 1–1.6 | same | depth arriving at 22–25 Hz on the deployed node against its 30 Hz tick | design, a band around that measured arrival rate |
| Depth latency | off | per env, 0–2 control steps | 1–3 | the age of the depth frame at the policy | design around a measured median of about 25 ms |
| [Depth range and grid](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py#L45) | under 0.4 m reads 0.2 m; far clamp 6.0 m; 640×360 block-reduced to 80×45 | same | same | the camera's near range and the deploy node's reduction (the same operator) | design |
| [Camera lens](../../source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py#L145) | fx = fy = 321.522 px at 640×360, the principal point at the image centre | same | same | the real unit's intrinsics | measured (the unit's factory calibration); the principal point is not matched |
| [Re-render after reset](../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py#L577) | 1 on every stack that reads a camera, 0 on the camera-free stack | same | same | a fresh frame at an episode's start | design |
| [Mount misalignment](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L505) | off | ±1° per axis per reset, shared by the IMU readings and the rendered camera | ±3° | the D555's mounting tolerance | design |
| [Subgoal-frame drift](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L407) | off | gain U(0, 0.5) per env × 0.166 m RMS and 6.7° heading, correlation time 2 s | same | the SLAM map frame drifting under the subgoal | the RMS and heading measured on the sim-in-the-loop rig's SLAM, not on the robot; the gain band and τ are design |
| [Path and lookahead](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py#L522) | the goal objective has none | waypoint noise 0.025 m; lookahead U(0.9, 1.1) m | U(0.7, 1.3) m | the deployed planner's path and lookahead | design |

Not applied on any tier: RGB noise (no registered ID observes RGB), depth quantization and temporal noise, lens
distortion, body-velocity noise, wheel-odometry drift and TF staleness. Declared, not applied: the contract's IMU
temperature drift and depth temporal-noise fields
([`sim_real_cfg.py:186`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L186),
[`:303`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L303)).

## The world

| perturbation | ideal | realistic | robust | models | basis |
|---|---|---|---|---|---|
| [Friction](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L170) | off | static U(0.6, 1.2) per env at each reset, dynamic 0.9× | U(0.3, 1.5) | floor surfaces | design |
| [Mass](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L334) | off | × U(0.95, 1.05) | × U(0.85, 1.15) | payload and battery | design |
| [Spawn and goal](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L622) | uniform in a box, any heading | uniform over the room's reachable cells, any heading; the goal from the same pool | same | any start the robot could be placed at | design |
| [Room](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L691) | a plane | camera stacks: difficulty U{4…7}, a ceiling on 70 % of episodes at 2.2–2.9 m; camera-free: level 7, open top | same | indoor rooms | design |

Not applied on any tier: a centre-of-mass offset, pushes, gravity or restitution changes. Declared, not applied: the
goal-pose noise term installed on the goal-objective and capture IDs
([`events.py:383`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L383)); the command's reset
resamples the goal after the event writes it.
