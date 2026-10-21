# Read the RoboClaw encoder count as signed, so a wheel turning backward keeps a continuous position

**Type:** bug (`/strafer/joint_states` positions jump by 2³² ticks when a wheel turns backward past
its power-up count)
**Owner:** Jetson (`strafer_driver`)
**Priority:** P2. The wheel speeds, and the odometry built from them, are read signed and are
unaffected. The positions feed `robot_state_publisher`'s wheel transforms and any recording of
`joint_states`. On the robot a backward spin put FL and RL at about 5×10⁷ rad.
**Estimate:** S
**Branch:** `task/roboclaw-encoder-count-wrap`

## Story

As **whoever reads the wheel joint positions** (TF, a bag, a debug view), I want **a wheel that
turns backward to read a small negative angle**, so that **positions stay continuous through zero
instead of jumping by 50 million radians.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`chassis-bringup-2026-10-19`](../../../measurements/chassis-bringup-2026-10-19/README.md) §7,
  finding 2, and its deposit's `drive/step6b_attempt1_spin.csv`.

## Context

Read at `371e848`, and seen on the robot on 2026-10-19.

- **The read.** `roboclaw_interface.py`'s `read_encoder_m1` and `read_encoder_m2` unpack the count
  as unsigned: `struct.unpack(">I", payload[:4])`. `read_speed_m1` and `read_speed_m2` unpack
  signed (`">i"`).
- **The manual** (Rev 6.0, command 16): "Quadrature encoders have a range of 0 to 4,294,967,295
  as an unsigned value. The same 4-byte value can also be read as signed, providing a range of
  -2,147,483,648 to 2,147,483,647." The status byte flags underflow (bit 0), direction (bit 1)
  and overflow (bit 2).
- **What the node does with it.** `roboclaw_node.py` multiplies each count by
  `ENCODER_TICKS_TO_RADIANS` and publishes the result as the joint position. It never resets the
  encoders, so the counts run from each controller's power-up.
- **Seen on the robot.** In a 0.8 rad/s spin with the wheels lifted, FL and RL turned backward
  from small positive counts. Their positions jumped by +4,294,967,287 and +4,294,967,286 ticks,
  which is 50,187,977 rad, between consecutive samples.

## Acceptance criteria

- [ ] **The count reads signed**, so a backward wheel's position goes negative through zero. The
      reason for signed, rather than unwrapping across samples with the status byte, is
      recorded.
- [ ] **A fake-serial test** steps a count from 3 down through 0 to −5. It checks that the
      published positions are continuous and negative, and it fails on `main`.
- [ ] **The bench tools print the same signed counts.** `diagnose_roboclaw.py` and `tune_pid.py`
      print encoder values.
- [ ] **On the robot,** with the wheels lifted, a backward spin past zero gives a continuous
      negative position.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance).
- [ ] No regression: `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py` (`read_encoder_m1`,
  `read_encoder_m2`).
- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py` (the `joint_states`
  positions).
- `source/strafer_ros/strafer_driver/test/test_port_fence.py` (`FakeRoboClaw`, which answers the
  encoder reads with a count of 0).

## Out of scope

- **A count beyond ±2³¹ ticks**, which is about 4 million wheel turns from power-up.
- **Odometry**, which is built from the signed speeds.
- **The chassis' velocity tuning:**
  [`chassis-velocity-pid-tuning`](chassis-velocity-pid-tuning.md).
