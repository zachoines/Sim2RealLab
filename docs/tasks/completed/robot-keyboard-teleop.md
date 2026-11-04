# Drive the robot from a keyboard through the deploy stack's own /cmd_vel

**Status:** Shipped 2026-11-04 in `edfec99` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/265

**Type:** tooling
**Owner:** Jetson (`strafer_bringup`, `deploy/`)
**Priority:** P2. Nothing is broken, but mapping a room and returning the robot to its start between
reaches on the real robot both need a person to drive it, and the stack has no way to.
**Estimate:** S
**Branch:** `task/robot-keyboard-teleop`

## Story

As **the maintainer at the robot with a keyboard**, I want **to drive it through the running deploy
stack's own `/cmd_vel`**, so that **I can map a room and return the robot to its start between
reaches without a separate tool or a hand-written publisher.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)

## Context

Read at `main` `f16d18d6`, and in `strafer-cpu:humble` and `strafer-gpu:humble` at that commit.

- **No teleop path exists.**
  - Neither image carries a teleop package: no `ros-humble-*teleop*`, `*joy*` or `twist_mux`, and
    `teleop_twist_keyboard` is not importable.
  - No launch file, Makefile target, compose service or documented command drives the robot from
    a keyboard or joystick.
  - The tree's teleop code is the Isaac Lab capture harness (`scripts/teleop_capture.py`), a pygame
    gamepad feeding the env's action tensor, not a ROS publisher.
- **Every drive command shares one topic.**
  - The driver subscribes to `/strafer/cmd_vel` (`roboclaw_node.py`:260-265), and
    `driver.launch.py`:58 remaps it to `/cmd_vel`. The inference node's `/strafer/cmd_vel` is
    remapped to `/cmd_vel` the same way (`inference.launch.py`:85-87).
  - Nav2, the executor's `rotate_in_place` and its cancel-path zero twist, and `donut_warmup` also
    publish `/cmd_vel`. The topic has no mux or priority, so two publishers interleave.
- **The driver's guards.**
  - It zeros the motors when no `/cmd_vel` has arrived for 0.5 s (`WATCHDOG_TIMEOUT_SEC`).
  - It also zeros them when `|v|` exceeds 1.2 × `MAX_LINEAR_VEL` (1.88 m/s) or `|ω|` exceeds 1.2 ×
    `MAX_ANGULAR_VEL` (5.03 rad/s) (`roboclaw_node.py`:74-80, 380-392).
- **`teleop_twist_keyboard`**, Humble release 2.4.1 (`teleop_twist_keyboard.py` at the `2.4.1` tag):
  - It publishes one `Twist` per key read, on the relative `cmd_vel`. It has no repeat timer: a
    held key repeats only as fast as the terminal's autorepeat sends it, and a released key sends
    nothing.
  - Its parameters are `stamped`, `frame_id`, `speed` (default 0.5) and `turn` (default 1.0), all
    read-only. `speed` and `turn` set the starting values only. `q`/`w`/`e` raise them by 10 % per
    press and `z`/`x`/`c` lower them, with no bound in the node.
  - Shift with the move keys strafes (`linear.y`). Any unbound key publishes a stop, and Ctrl-C,
    read as a raw byte, publishes a zero before the node exits.
  - It reads stdin in raw mode, so it needs a TTY. It cannot run under `ros2 launch`.
- **The compose contract.** `TestComposeStopSignal::test_no_service_is_interactive` forbids `tty`
  and `stdin_open` on every compose service, because launch does not pass SIGINT on to a node
  whose stdin is a TTY. `docker compose run` allocates a TTY at run time without either key.

## Acceptance criteria

- [x] **The dependency comes from rosdep.** `teleop_twist_keyboard` is an `exec_depend` of
      `strafer_bringup`, so the image's existing `rosdep install --from-paths src` installs it,
      with no apt line added by hand.
  - *2026-11-04: in `edfec99`; a CPU image built from it carries `ros-humble-teleop-twist-keyboard`
    2.4.1.*
- [x] **One command starts it on the running stack**, with a TTY, publishing `/cmd_vel`: a compose
      service in its own profile, run by a Makefile target with `docker compose run --rm`.
  - It runs on the CPU image's service anchor, so it uses the stack's own DDS environment.
  - It sets no `tty` or `stdin_open`, so the compose contract above holds.
  - *2026-11-04: the `teleop` service (profile `teleop`, the CPU anchor, `restart: "no"`) and
    `make teleop`, which runs `docker compose --profile teleop run --rm teleop`.
    `TestComposeStopSignal` passes with it.*
- [x] **Walking pace.** `speed` and `turn` are set to a walking pace, with the values and the
      reason stated. The PR states that these are starting values that the keys can raise, and
      what bounds them.
  - *2026-11-04: 0.3 m/s, the speed `chassis-bringup-2026-10-19` drove the wheels at through the
    driver, and 0.6 rad/s, the warm-up spin's rate. They are starting values that the keys raise
    10 % per press, bounded only by the driver's 1.88 m/s and 5.03 rad/s (PR #265).*
- [x] **No repeat rate.** Nothing republishes a held command, so the driver's 0.5 s watchdog stops
      the motors once a key is released. `source/strafer_ros/README.md`'s driver paragraph says so
      in one sentence.
  - *2026-11-04: that sentence is in the driver paragraph, in `edfec99`.*
- [x] **A contract test**, in the style of `test_perception_launch.py`:
  - it pins the service's command, its remap to `/cmd_vel` and its parameters, and the
    `exec_depend`;
  - where the package is installed, it checks that the installed node still declares the pinned
    names and no repeat parameter.
  - *2026-11-04: `test_teleop.py`: 7 passed and 3 skipped in `strafer-cpu:humble`, which predates
    the dependency, and 10 passed in the transient image.*
- [x] **The dependency's presence**, asserted in a transient container built from this branch,
      without touching the deployed images (a build under a host hold, with the tags restored), or
      stated as unbuilt.
  - *2026-11-04: built as `strafer-cpu:teleop-edfec99bb910` (`870f5f07cf0c`) under a host hold.
    `:humble` was not touched, and the tag was removed afterwards.*
- [x] **The PR says** that the maintainer validates it, paired, before it is used, and that it is
      merged with `executor-exit-cancels-active-goal` before the next image rebuild.
  - *2026-11-04: PR #265's Scope says both.*
- [x] No regression: `make test-ros` passes.
  - *2026-11-04: 996 passed, 14 skipped, 1 xfailed.*

## Out of scope

- **A hard ceiling at walking pace.** The node has no maximum, and the driver bounds speed only at
  1.2 × its maximum. A ceiling needs a clamp between the node and `/cmd_vel`, or another node.
- **A joystick** (`teleop_twist_joy`).
- **The paired validation itself:** the robot driven at the robot, including how the terminal's
  autorepeat delay meets the 0.5 s watchdog on a held key.
- **Arbitration** between teleop and a running mission on `/cmd_vel`.
