# Sim-to-Real Policy Transfer Plan

This document is about one narrow problem:

- train a navigation policy in Isaac Lab
- export it in a deployable format
- run the same policy on the real robot through `strafer_inference`
- validate that the policy transfers cleanly to the Jetson ROS runtime

It is not the source of truth for:

- Jetson-side ROS package inventory and current package status
- autonomy, VLM, or natural-language orchestration
- detailed sensor and actuator tuning procedures

Those live in:

- `docs/STRAFER_AUTONOMY_ROS.md`
- `docs/SIM_TO_REAL_TUNING_GUIDE.md`
- the `STRAFER_AUTONOMY_*` docs

## Goal

The current MVP goal is:

- train a policy in Isaac Lab for goal-directed navigation
- export that policy without changing its observation or action contract
- deploy it to the Jetson
- execute it through a planned `strafer_inference` package that plugs into the existing ROS stack

The current MVP does not include:

- Qwen or VLM-assisted grounding
- autonomy planning
- cloud services
- multi-step natural-language execution

## Transfer Path

```text
Isaac Lab training
  -> gym evaluation
  -> exported policy artifact (.pt first, .onnx later)
  -> shared contract in strafer_shared
  -> Jetson-side strafer_inference runtime
  -> ROS topics and robot state
  -> strafer_driver / real hardware
```

The important boundary is:

- Isaac Lab and Jetson must agree on policy I/O through `strafer_shared`

That shared contract is what makes sim-to-real transfer possible without re-implementing observation assembly or action interpretation on the robot.

## Scope

This document covers:

- policy training and evaluation flow
- the shared sim-to-real contract
- export and packaging strategy
- Jetson deployment path through `strafer_inference`
- policy-transfer validation steps

This document does not cover:

- detailed ROS package responsibilities
- current versus planned Jetson-side autonomy work
- VLM integration
- planner execution
- AWS or Databricks deployment

## Shared Contract

The transfer path depends on three shared modules:

### `strafer_shared.constants`

Owns the physical and normalization constants that must not drift between sim and real.

Examples:

- robot geometry
- wheel and drivetrain constants
- RoboClaw PID values
- velocity limits
- sensor normalization scales

### `strafer_shared.mecanum_kinematics`

Owns the mecanum forward and inverse kinematics used by both sides.

Use it for:

- body twist to wheel velocity conversion
- encoder ticks to body-velocity reconstruction
- rad/s to ticks/s conversion

### `strafer_shared.policy_interface`

This is the core policy contract.

It owns:

- `PolicyVariant`
- `assemble_observation()`
- `interpret_action()`
- `load_policy()`

This contract must be the only way the policy is:

- fed observations on the Jetson
- interpreted after inference

If the policy contract changes in Isaac Lab, the Jetson runtime must change through `strafer_shared`, not through local one-off logic.

## Two Evaluation Paths

There are still two valid ways to evaluate a trained policy.

### Path 1: Gym evaluation

Use this during training for:

- fast regression checks
- reward and stability monitoring
- policy comparison
- environment-side debugging

This path stays entirely inside Isaac Lab and Python.

### Path 2: ROS evaluation

Use this for:

- deployability validation
- timing and runtime checks on the Jetson
- sensor and topic integration
- end-to-end robot behavior

This path runs the exported model through the ROS runtime on:

- the real Jetson robot stack
- or Isaac Sim through a ROS bridge if needed

## Training Workflow

### 1. Choose the policy variant

Start with the simplest transfer target:

- `PolicyVariant.NOCAM`

That keeps the first transfer problem focused on:

- IMU
- encoder velocities
- goal-relative state
- action execution

Depth-based variants can come later after the base inference path is proven.

### 2. Train in Isaac Lab

Train the policy in Isaac Lab with the environment and preset intended for transfer.

The key requirement is:

- keep the sim-side observation and action contract aligned with `strafer_shared.policy_interface`

### 3. Evaluate in gym before export

Before exporting, verify:

- the policy is stable in simulation
- the expected variant is being used
- the action outputs are sane
- the model is worth transferring

### 4. Freeze transfer metadata

When a candidate model is selected, record at least:

- environment id
- policy variant
- training preset
- git commit
- export format
- expected observation dimensionality

This prevents silent mismatch when the Jetson runtime loads the artifact later.

## Export And Artifact Handoff

### Initial export target

Use:

- TorchScript `.pt`

This keeps the first deployment simple and aligned with the current shared loader path.

### Later export target

Use:

- ONNX `.onnx`

when:

- model complexity grows
- TensorRT becomes necessary
- Jetson inference optimization matters more

### Artifact expectations

Each exported policy artifact should have:

- model file
- variant identifier
- export metadata
- notes on the training preset and source experiment

The loading boundary on the robot should remain:

- `load_policy(path, variant)`

not a model-specific ad hoc loader inside `strafer_inference`.

## Jetson Deployment Path

The Jetson-side ROS package inventory is documented in:

- `docs/STRAFER_AUTONOMY_ROS.md`

For sim-to-real transfer, the important Jetson-side pieces are:

- `strafer_driver`
- `strafer_perception`
- `strafer_description`
- `strafer_bringup`
- planned `strafer_inference`

### Planned role of `strafer_inference`

`strafer_inference` should be the Jetson package that:

- loads the exported policy artifact
- subscribes to the robot-state inputs needed by the selected `PolicyVariant`
- assembles observations through `strafer_shared.policy_interface`
- runs inference on the Jetson
- converts actions through `interpret_action()`
- outputs robot commands through the ROS control path

### Expected runtime boundary

For the MVP, `strafer_inference` should be concerned with:

- policy execution
- not mission planning
- not VLM grounding
- not autonomy orchestration

That keeps the first transfer target narrow and testable.

## ROS-Side Policy Runtime Contract

The policy-transfer side of the ROS contract is smaller than the full autonomy contract.

The key inputs and outputs for a first `strafer_inference` implementation are:

- input: `/d555/imu/filtered`
- input: `/strafer/joint_states`
- optional input later: `/d555/depth/downsampled`
- input: robot-local goal or subgoal interface
- output: `/strafer/cmd_vel`

The goal-like input should stay interface-agnostic in this document.

Possible first shapes include:
- direct-RL mode:
  - final goal pose or goal-relative target
- hybrid mode:
  - waypoint, path segment, or local subgoal stream from a higher-level planner

An initial `/strafer/goal` topic is still a possible first implementation, but it should not be treated as the long-term required contract here.

The full Jetson-side package and topic inventory should still be maintained in:

- `docs/STRAFER_AUTONOMY_ROS.md`

This document only cares about the subset needed to transfer a trained policy.

## Validation Sequence

### Step 1: Export validation

Before Jetson deployment:

- confirm the artifact loads through `load_policy()`
- confirm the expected `PolicyVariant` is used
- confirm an example observation produces a sane action

### Step 2: Jetson runtime validation

On the Jetson:

- bring up the base ROS stack
- run `strafer_inference` with a hardcoded or simple robot-local goal/subgoal source
- confirm the node consumes the expected topics
- confirm actions are translated into robot commands correctly

### Step 3: Motion validation

Validate on hardware:

- the robot moves in the correct direction
- the observation assembly is correct
- watchdog and stop behavior remain intact
- latency is acceptable

### Step 4: Sim-vs-real comparison

Compare:

- response timing
- trajectory shape
- goal convergence
- failure modes

If transfer is weak, use:

- `docs/SIM_TO_REAL_TUNING_GUIDE.md`

to characterize where the real system falls outside the training envelope.

## Remaining Work

The remaining sim-to-real transfer work is:

1. finish the current Isaac Lab training run
2. export a deployable policy artifact
3. create the planned `strafer_inference` package
4. load the exported policy through `strafer_shared.policy_interface`
5. validate the policy through ROS on the Jetson
6. compare sim and real behavior
7. tune and retrain only where transfer evidence shows it is necessary

## Deliberate Non-Goals

This document should not grow back into:

- a Jetson ROS package catalog
- an autonomy architecture document
- a VLM roadmap
- a hardware wiring manual
- a cloud deployment guide

Those already have better homes elsewhere.

## Key References

- `docs/STRAFER_AUTONOMY_ROS.md`
- `docs/SIM_TO_REAL_TUNING_GUIDE.md`
- `source/strafer_shared/strafer_shared/constants.py`
- `source/strafer_shared/strafer_shared/mecanum_kinematics.py`
- `source/strafer_shared/strafer_shared/policy_interface.py`
- `source/strafer_lab/strafer_lab/tasks/navigation/sim_real_cfg.py`
- `source/strafer_lab/strafer_lab/tasks/navigation/mdp/actions.py`
- planned `source/strafer_ros/strafer_inference/`
