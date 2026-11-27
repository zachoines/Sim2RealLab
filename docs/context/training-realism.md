# Training realism: what each tier perturbs

Navigation training perturbs the actuation, the sensors and the world on one of three tiers, set by
`RealismCfg(level=...)` ([`composed_env_cfg.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py)):
`ideal` (nothing), `real` (the realistic tier) and `robust`. The action term, the sensor noise and the subgoal-frame
drift come from the tier contracts `IDEAL_SIM_CONTRACT`, `REAL_ROBOT_CONTRACT` and `ROBUST_TRAINING_CONTRACT`
([`sim_real_cfg.py:714-716`](../../source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py#L714)); friction,
mass, motor strength, the mount angle, the room and the lookahead bands are per-tier values in
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
| [Command delay](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L220) | off | 33.3–100 ms, uniform over the whole control steps it spans (1–3) | 33.3–167 ms (1–5) | time from the policy's command to the wheels: topic transport, the driver, the motor controller's serial link | design; not measured. One delay per env, drawn when the action term is built, not at a reset; a redraw at each reset and a part under one control step are built, both off. Set in seconds, so the sim-in-the-loop bridge, which steps the env every physics tick by default, runs the same delay band and mean. |
| [Command hold](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L231) | off | a fraction U(0, 0.05) of control steps, in runs of mean U(33.3, 40) ms, both drawn per env at each reset | same | a residual: a control tick the node does not publish (a skipped or late tick), on which the chassis keeps its last command | design; not measured. The deployed node publishes on every tick, reusing its last depth frame when none arrived, so a held depth frame does not hold the command there. The run length is set in seconds, so the sim-in-the-loop bridge runs the same band, mean run and held share. |
| [Acceleration limit](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L254) | off | 175–275 rad/s² per wheel, drawn per env at each reset | 125–300 rad/s² | how fast a wheel's speed can change under torque and load; a design spread, not a battery state: the motor controllers run from a regulated supply | unknown for the nominal 225, which acts only when the band is one value; the band is design. The limit is `accel × step_dt` per control step. The limit is per second, so the sim-in-the-loop bridge, which steps the env every physics tick by default, runs the trained limit. The limit is flat in the wheel's speed: its fall-off field is off. |
| [Motor lag](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L395) | off | first-order; 63.2 % rise in 0.216 s, the field τ, one value per env (its band at the nominal) | 0.256 s | the motor controller's velocity loop and the motor | unknown; not measured. α is the exponential form of τ at the control step, so the field τ is the rise time at any step, and the sim-in-the-loop bridge, which steps the env every physics tick by default, runs the trained lag |
| [Command envelope](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L304) | off | off | s ~ U(0.255, 1.0) per env at each reset, on both caps; the policy does not observe s | the deploy node caps the twist at s = 0.5 of the full envelope | design: a band holding the deploy cap |
| [Twist clamp](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L451) | \|vx\| + \|vy\| ≤ 1.568 m/s, \|ω\| ≤ 4.186 rad/s | same | same, × s | the deploy node's clamp: `l1_clamp_twist` there, its batched twin `l1_clamp_twist_batched` here, pinned equal by a test | derived from the motor's rated speed and the chassis geometry |
| [Wheel speed and torque](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py#L572) | 32.67 rad/s per wheel, a limit field apart from the action scale and the caps; 2.383 N·m at every speed | same | same | the drive motor's rated speed (312 rpm) and stall torque, on a velocity drive (damping 1000) whose torque cap does not fall with speed | datasheet |
| [Motor strength](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L468) | off | torque limit × U(0.92, 1.08), each wheel drawn on its own at each reset | × U(0.8, 1.2) | motor-to-motor torque spread | design |
| Control period | 1/30 s (physics 1/120 s) | same | same | the deploy node's 30 Hz tick and the D555's 30 Hz stream | design; not randomized on any tier |

## Sensing

The noise models are in
[`mdp/noise_models.py`](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py); each applies once
per control step to the policy's observation.

| perturbation | ideal | real | robust | models | basis |
|---|---|---|---|---|---|
| [IMU noise](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L187) | off | white, σ 0.0537 m/s² and 0.00131 rad/s per sample (density 0.0098 m/s²/√Hz and 0.00024 rad/s/√Hz × √(30 Hz)); initial bias ±0.04 m/s², ±0.0017 rad/s; bias random walk 0.001 m/s² and 0.0001 rad/s per √s, clamped at twice the initial-bias band | σ 0.0822 and 0.00197 (density 0.015 and 0.00036); bias ±0.06, ±0.0025; walk 0.001 and 0.0001 | the D555's IMU, read once per control tick | the density the code gives for the BMI055, converted at the policy rate; the bias values unknown; a per-sample σ in place of the density form is built, unset |
| IMU and encoder latency | off | none | 1 control step | the age of a reading at the policy | design |
| [Wheel encoder noise](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L353) | off | σ 0.02 of 3000 ticks/s; integer ticks/s; one reading per control step, which the body velocity is the kinematics of; the reported speed's lag off | σ 0.05 | the quadrature encoders' speed as the motor controller reports it: `joint_states` and odom carry the same reading | design |
| Sensor dropouts | off | none | per control step: each IMU channel (accelerometer, gyroscope) repeats its last reading 2e-4; the encoders repeat theirs 1e-4; a failed depth frame repeats the previous one, within the frame drop | a sensor that stops reporting: each IMU channel and the encoders repeat their last reading, as a held source does on the robot. The encoders and the body velocity hold together, one reading, as on the robot; the IMU's channels are drawn apart, where the robot's one message holds both; past the node's timeout the robot stops, which training does not model | design |
| [Depth disparity noise](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/noise_models.py#L479) | off | σ_d 0.08 px, independent per 80×45 cell after the block median, scaled to depth as σ_z = z²·σ_d / (f·B) with f 673 px and B 0.095 m: 1.25 mm at 1 m, 5.01 mm at 2 m, 45.0 mm at 6 m | σ_d per env, log-uniform on [0.002, 0.16] px | the D555's stereo depth error | design, from a literature range (not a D555 figure); its amplitude checked against the real unit; the robust band is design |
| Depth holes and lost frames | off | holes on 0.01 of cells, independent, each filled with the median of its valid neighbouring cells (the near fill where none is valid); a frame lost (the previous frame repeated) 0.001 | 0.03; 0.011, a failed frame included | invalid pixels and dropped frames. The deploy node instead writes an invalid pixel as the far clamp before the 8×8 block median, so a mostly invalid block reads far | design |
| Depth stream hold | off | a fraction U(0, 0.35) of ticks, in runs of mean U(1, 1.6) ticks, both drawn per env at each reset | same | depth arriving slower than the 30 Hz tick, the node reusing its last frame | design: a band chosen around the arrival rate of the sim-in-the-loop rig's sessions, not set from a robot-side statistic |
| Depth latency | off | 0–2 control steps, drawn per env at each reset | 1–3 | the age of the depth frame at the policy | design: a per-env spread centred on one control step (two on `robust`); not measured as the age at the policy |
| [Depth range and grid](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/observations.py#L45) | under 0.4 m reads 0.2 m; far clamp 6.0 m; 640×360 block-reduced to 80×45 | same | same | the camera's near range and the deploy node's reduction (the same operator) | design |
| [Camera lens](../../source/strafer_lab/strafer_lab/tasks/navigation/d555_cfg.py#L145) | fx = fy = 321.522 px at 640×360, the principal point at the image centre | same | same | the real unit's intrinsics | measured (the unit's factory calibration); the principal point is not matched |
| [Re-render after reset](../../source/strafer_lab/strafer_lab/tasks/navigation/composed_env_cfg.py#L572) | 1 on every stack that reads a camera, 0 on the camera-free stack | same | same | a fresh frame at an episode's start | design |
| [Mount misalignment](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L519) | off | ±1° per axis per reset, shared by the IMU readings and the rendered camera; a translation of the housing (the IMU's lever arm and the camera's position together) is built, off | ±3° | the D555's mounting tolerance | design |
| [Subgoal-frame drift](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L421) | off | gain U(0, 0.5) per env × 0.166 m RMS and 6.7° heading, correlation time 2 s | same | the SLAM map frame drifting under the subgoal; on the goal-objective IDs it moves the goal | the RMS and heading measured on the sim-in-the-loop rig's SLAM, not on the robot; the gain band and τ are design |
| [Path and lookahead](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/commands.py#L668) | no subgoal ID | subgoal IDs only: waypoint noise 0.025 m; lookahead U(0.9, 1.1) m | subgoal IDs only: waypoint noise 0.025 m; lookahead U(0.7, 1.3) m | the deployed planner's path and lookahead | design |

Not applied on any tier: RGB noise (no registered ID observes RGB), depth quantization and temporal noise, lens
distortion, wheel-odometry drift and TF staleness.

## The world

| perturbation | ideal | real | robust | models | basis |
|---|---|---|---|---|---|
| [Friction](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L175) | off | the wheel-floor static grip U(0.55, 0.85) per env at each reset; against the floor's static 0.5, averaged, the roller covers take static U(0.6, 1.2), dynamic 0.9×, as do all 57 collision shapes, the 40 covers included | grip U(0.4, 1.0); static U(0.3, 1.5) | the grip on the robot's floors, and from the same draw the chassis's friction against walls and furniture (a chassis value of its own is built, off). The covers' authored rubber material resolves as no physics material, so without the draw (the ideal tier) the grip is 0.50 | design |
| [Mass](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L359) | off | × U(0.95, 1.05) per env at each reset, one factor on every body of the articulation (wheels and roller covers included), on its mass and its inertia together | × U(0.85, 1.15) | a spread on the mechanical chassis, the USD's 5.15 kg; the deployed robot's electronics and battery are not in it (a startup payload on the chassis body, through Isaac Lab's mass and centre-of-mass terms, is built, off) | design |
| [Spawn and goal](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/events.py#L657) | spawn uniform in ±1 m, any heading; goal in a box a success curriculum sets, ±0.5 m wider by 0.25 m after each 5 successes in a row up to ±5 m, at least 1 m away, resampled every 10–15 s | uniform over the room's reachable cells, any heading; the goal from the same pool; on the camera stacks the robot's pool keeps 1 more 0.1 m cell from obstacles | same | any start the robot could be placed at | design |
| [Room](../../source/strafer_lab/strafer_lab/tasks/navigation/mdp/proc_room.py#L990) | a plane | camera stacks: difficulty U{4…7}; walls 2.7 m; spans up to 7.5 m per axis, at most 13.4 m together; doors up to 2.0 m; 2 tall columns on 50 % of episodes; clutter biased to the walls on 30 %; a ceiling on 70 % at 2.2–2.9 m. Camera-free: level 7, 1.0 m walls, open top | same | indoor rooms | design |

Not applied on any tier: a centre-of-mass offset, pushes, gravity or restitution changes.
