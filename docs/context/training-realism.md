# Training realism: what each tier perturbs

Navigation training perturbs the actuation, the sensors and the world on one of three tiers, set by
`RealismCfg(level=...)` ([`composed_env_cfg.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py)):
`ideal` (nothing), `real` (the realistic tier) and `robust`. The action term, the sensor noise and the subgoal-frame
drift come from the tier contracts `IDEAL_SIM_CONTRACT`, `REAL_ROBOT_CONTRACT` and `ROBUST_TRAINING_CONTRACT`
([`sim_real_cfg.py:707-709`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L707)); friction,
mass, motor strength, the mount angle, the room, the lookahead bands and the goal noise are per-tier values in
[`strafer_env_cfg.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/strafer_env_cfg.py), and the re-render is
set per sensor stack in `composed_env_cfg.py`. Every RL ID trains on `real` unless it is named `…-Robust-…`;
the plane scaffolds `StraferNavCfg_*_Ideal` use `ideal`. Each row says what a perturbation stands for on the deployed
robot (the Jetson's ROS stack, the D555 camera and its IMU, the RoboClaw-driven wheels) and the basis of its value.
[`test_training_realism_doc.py`](../../source/strafer_lab/tests/navigation/test_training_realism_doc.py) compares every
number in the tables below with the value the code gives. It fails when an event term, a curriculum term, an action-cfg
field, a command-cfg field or a noise-cfg field of a composed class has no row or not-applied line here; when a policy
group's corruption switch, a noise model's class or sensor, or an observation term's clip or scale is not what the rows
assume; or when the action term with its documented stages off is more than the clamp and the kinematics. What it does
not pin is listed in its docstring.

Basis: **measured** (on the real robot or its sensor, or on the sim-in-the-loop rig where said), **datasheet**,
**derived** (computed from a measured or datasheet value), **design** (a chosen value or band), **unknown** (no source
for the value).

## Timing and actuation

The action term ([`mdp/actions.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py)) runs once
per control step: the twist clamp (× s on `robust`) and the per-wheel speed clamp first, then the delay, the hold, the
acceleration limit and the motor lag.

| perturbation | ideal | real | robust | models | basis |
|---|---|---|---|---|---|
| [Command delay](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L199) | off | 1–3 control steps | 1–5 control steps | time from the policy's command to the wheels: topic transport, the driver, the motor controller's serial link | design; not measured. One delay per env, drawn when the action term is built, not at a reset |
| [Command hold](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L215) | off | a fraction U(0, 0.05) of ticks, in runs of mean U(1, 1.2) ticks, both drawn per env at each reset | same | a residual: a control tick the node does not publish (a skipped or late tick), on which the chassis keeps its last command | design; not measured. The deployed node publishes on every tick, reusing its last depth frame when none arrived, so a held depth frame does not hold the command there |
| [Acceleration limit](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L237) | off | 175–275 rad/s² per wheel, drawn per env at each reset (field: 700–1100) | 125–300 rad/s² (field: 500–1200) | how fast a wheel's speed can change under torque, load and the battery's state | unknown for the nominal 225 (field: 900), which acts only when the band is one value; the band is design. The limit is `accel × physics_dt` per control step, so a field value is 4× the trained one |
| [Motor lag](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L186) | off | first-order; 63.2 % rise in 0.216 s, the lag's time constant (field τ 0.05 s) | 0.256 s (field τ 0.06 s) | the motor controller's velocity loop and the motor | unknown; not measured. α = `physics_dt / (τ + physics_dt)` per control step, so the backward-Euler τ at the control step is 0.200 / 0.240 s, 4× the field |
| [Command envelope](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L272) | off | off | s ~ U(0.255, 1.0) per env at each reset, on both caps; the policy does not observe s | the deploy node caps the twist at s = 0.5 of the full envelope | design: a band holding the deploy cap |
| [Twist clamp](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L362) | \|vx\| + \|vy\| ≤ 1.568 m/s, \|ω\| ≤ 4.186 rad/s | same | same, × s | the deploy node's clamp: `l1_clamp_twist` there, its batched twin `l1_clamp_twist_batched` here, pinned equal by a test | derived from the motor's rated speed and the chassis geometry |
| [Wheel speed and torque](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L456) | 32.67 rad/s; 2.383 N·m at every speed | same | same | the drive motor's rated speed (312 rpm) and stall torque, on a velocity drive (damping 1000) whose torque cap does not fall with speed | datasheet |
| [Motor strength](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L454) | off | torque limit × U(0.92, 1.08), each wheel drawn on its own at each reset | × U(0.8, 1.2) | motor-to-motor torque spread | design |
| Control period | 1/30 s (physics 1/120 s) | same | same | the deploy node's 30 Hz tick and the D555's 30 Hz stream | design; not randomized on any tier |

Declared, not applied: the contract fields `motor_time_constant_range` and `domain_randomization_scale`
([`sim_real_cfg.py:125`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L125),
[`:492`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L492)) reach no env.

## Sensing

The noise models are in
[`mdp/noise_models.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py); each applies once
per control step to the policy's observation.

| perturbation | ideal | real | robust | models | basis |
|---|---|---|---|---|---|
| [IMU noise](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L182) | off | white, σ 0.0537 m/s² and 0.00131 rad/s per sample (density 0.0098 m/s²/√Hz and 0.00024 rad/s/√Hz × √(30 Hz)); initial bias ±0.04 m/s², ±0.0017 rad/s; bias random walk 0.001 m/s² and 0.0001 rad/s per √s, clamped at twice the initial-bias band | σ 0.0822 and 0.00197 (density 0.015 and 0.00036); bias ±0.06, ±0.0025; walk 0.001 and 0.0001 | the D555's IMU, read once per control tick | the density the code gives for the BMI055, converted at the policy rate; the bias values unknown |
| IMU and encoder latency | off | none | 1 control step | the age of a reading at the policy | design |
| [Wheel encoder noise](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L356) | off | σ 0.02 of 3000 ticks/s; a missed tick 0.001, an extra 0.0005, each moving the reading by one tick/s; integer ticks/s | σ 0.05; missed 0.005; extra 0.0005 | the quadrature encoders' speed estimate | design |
| Sensor dropouts | off | none | per control step: each IMU channel (accelerometer, gyroscope) reads zero 1e-4 or repeats its last reading 1e-4; the encoders read zero 1e-4; the depth frame reads far 1e-3 | a robustness margin: training feeds a zero or repeated reading, or an all-far frame. On the robot a silent sensor reads as its last value until the node's timeout, then the node stops the robot | design |
| [Depth disparity noise](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L479) | off | σ_d 0.08 px, independent per 80×45 cell after the block median, scaled to depth as σ_z = z²·σ_d / (f·B) with f 673 px and B 0.095 m: 1.25 mm at 1 m, 5.01 mm at 2 m, 45.0 mm at 6 m | σ_d per env, log-uniform on [0.002, 0.16] px | the D555's stereo depth error | design, from a literature range (not a D555 figure); its amplitude checked against the real unit; the robust band is design |
| Depth holes and lost frames | off | holes on 0.01 of cells, independent, each filled with the median of its valid neighbouring cells (the near fill where none is valid); a frame lost (the previous frame repeated) 0.001 | 0.03; 0.01 | invalid pixels and dropped frames. The deploy node instead writes an invalid pixel as the far clamp before the 8×8 block median, so a mostly invalid block reads far | design |
| Depth stream hold | off | a fraction U(0, 0.35) of ticks, in runs of mean U(1, 1.6) ticks, both drawn per env at each reset | same | depth arriving slower than the 30 Hz tick, the node reusing its last frame | design: a band chosen around the arrival rate of the sim-in-the-loop rig's sessions, not set from a robot-side statistic |
| Depth latency | off | 0–2 control steps, drawn per env at each reset | 1–3 | the age of the depth frame at the policy | design: a per-env spread centred on one control step (two on `robust`); not measured as the age at the policy |
| [Depth range and grid](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py#L45) | under 0.4 m reads 0.2 m; far clamp 6.0 m; 640×360 block-reduced to 80×45 | same | same | the camera's near range and the deploy node's reduction (the same operator) | design |
| [Camera lens](../../source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py#L145) | fx = fy = 321.522 px at 640×360, the principal point at the image centre | same | same | the real unit's intrinsics | measured (the unit's factory calibration); the principal point is not matched |
| [Re-render after reset](../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py#L577) | 1 on every stack that reads a camera, 0 on the camera-free stack | same | same | a fresh frame at an episode's start | design |
| [Mount misalignment](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L505) | off | ±1° per axis per reset, shared by the IMU readings and the rendered camera | ±3° | the D555's mounting tolerance | design |
| [Subgoal-frame drift](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L407) | off | gain U(0, 0.5) per env × 0.166 m RMS and 6.7° heading, correlation time 2 s | same | the SLAM map frame drifting under the subgoal; on the goal-objective IDs it moves the goal | the RMS and heading measured on the sim-in-the-loop rig's SLAM, not on the robot; the gain band and τ are design |
| [Path and lookahead](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py#L656) | no subgoal ID | subgoal IDs only: waypoint noise 0.025 m; lookahead U(0.9, 1.1) m | subgoal IDs only: waypoint noise 0.025 m; lookahead U(0.7, 1.3) m | the deployed planner's path and lookahead | design |

Not applied on any tier: RGB noise (no registered ID observes RGB), depth quantization and temporal noise, lens
distortion, body-velocity noise, wheel-odometry drift and TF staleness. Declared, not applied: the contract's IMU
temperature drift (`enable_temp_effects`, `temp_coefficient`), its depth temporal noise (`enable_temporal_noise`,
`temporal_noise_std`), `hole_cluster_size` and `velocity_quantization_ticks_per_step`
([`sim_real_cfg.py:186`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L186),
[`:303`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L303),
[`:269`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L269),
[`:222`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L222)), and the encoder noise model's
`ticks_per_radian`.

## The world

| perturbation | ideal | real | robust | models | basis |
|---|---|---|---|---|---|
| [Friction](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L170) | off | static U(0.6, 1.2) per env at each reset, dynamic 0.9×, on all 57 collision shapes, the 40 roller covers included; against the floor's static 0.5, averaged, the wheel-floor static grip is U(0.55, 0.85) | U(0.3, 1.5); grip U(0.4, 1.0) | the grip on the robot's floors, and from the same draw the chassis's friction against walls and furniture. The covers' authored rubber material resolves as no physics material, so the term's exception for rubber shapes matches none; without the draw the grip is 0.50 | design |
| [Mass](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L334) | off | × U(0.95, 1.05) per env at each reset, one factor on every body of the articulation (wheels and roller covers included); inertia not rescaled | × U(0.85, 1.15) | a spread on the mechanical chassis, the USD's 5.15 kg; the deployed robot's electronics and battery are not in it | design |
| [Spawn and goal](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L622) | spawn uniform in ±1 m, any heading; goal in a box a success curriculum sets, ±0.5 m wider by 0.25 m after each 5 successes in a row up to ±5 m, at least 1 m away, resampled every 10–15 s | uniform over the room's reachable cells, any heading; the goal from the same pool; on the camera stacks the robot's pool keeps 1 more 0.1 m cell from obstacles | same | any start the robot could be placed at | design |
| [Room](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py#L990) | a plane | camera stacks: difficulty U{4…7}; walls 2.7 m; spans up to 7.5 m per axis, at most 13.4 m together; doors up to 2.0 m; 2 tall columns on 50 % of episodes; clutter biased to the walls on 30 %; a ceiling on 70 % at 2.2–2.9 m. Camera-free: level 7, 1.0 m walls, open top | same | indoor rooms | design |

Not applied on any tier: a centre-of-mass offset, pushes, gravity or restitution changes. Declared, not applied: the goal-pose noise term
([`events.py:383`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L383)), installed on the robust
goal-objective IDs (σ 0.35 m) and the Infinigen capture IDs (σ 0.15 m); the command's reset resamples the goal after the event writes it.
