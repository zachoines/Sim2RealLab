# Stop the compose services with SIGINT so a container stop reaches their nodes

**Status:** Shipped 2026-10-17 in `e7e6e2b` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/248

**Type:** bug (a container stop or restart reaches no ROS node; every node dies by SIGKILL)
**Owner:** Jetson (`deploy/`)
**Priority:** P1 for goal-b. On the robot, `docker compose stop base` or `restart base`
SIGKILLs the driver, so the motor stop in its teardown never runs. A stop of `autonomy` never
reaches the executor's exit path, and a stop of `slam` never reaches rtabmap's database save.
**Estimate:** S
**Branch:** `task/compose-stop-signal-sigint`

## Story

As **whoever stops, restarts or recreates a strafer container**, I want **the stop to reach every
node in it**, so that **each node's teardown runs: the driver's motor stop, rtabmap's save, and the
executor's exit path.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [context/deploy-env-config.md](../context/deploy-env-config.md)
- [`executor-exit-cancels-active-goal`](../active/reliability/executor-exit-cancels-active-goal.md), "A container stop
  never reaches the executor's `finally`": the first probe of this path, which left why launch's
  `_on_sigterm` never ran unexplained.
- [`chassis-driver-port-fence-2026-10-14`](../../measurements/chassis-driver-port-fence-2026-10-14/README.md)
  §3 and §6: the launch probes under `docker stop` and under SIGINT.
- [`enriched-lane-rig-stability`](../active/reliability/enriched-lane-rig-stability.md), mode 1: rtabmap's save at close,
  and slam's 180 s grace period.

## Context

Compose files read at `0a975ef`. Image facts read on 2026-10-17 in `strafer-cpu:humble` (revision
label `9f7d20758d5b`): launch 1.0.14, ros2launch 0.19.13, Python 3.10.12. The record is
[`compose-stop-signal-2026-10-17`](../../measurements/compose-stop-signal-2026-10-17/README.md).

- **`ros2 launch` is PID 1 in every service that runs nodes.**
  - The entrypoint `exec`s the command (`deploy/docker/Dockerfile.cpu`:86-95,
    `Dockerfile.gpu`:79-87).
  - The services:
    - on the CPU anchor in `docker-compose.yml`: `base`, `perception`, `slam`, `navigation`,
      `autonomy`, `viewer`, `sim-perception`;
    - `inference`, defined on its own, since there is no GPU anchor;
    - `strafer-sim` in `docker-compose.sim.yml`.
  - None sets `stop_signal`, so `docker compose stop` and `restart`, and `docker stop` and
    `restart` on these containers, send SIGTERM.
- **Launch installs no OS-level SIGTERM handler.**
  - Its signal manager's docstring: "The underlying mechanism is built around
    `signal.set_wakeup_fd` so as to not interfere with regular handlers installed via
    `signal.signal`" (`launch/utilities/signal_management.py`:49-51).
  - `launch_service.py`:215-216 registers `_on_sigint` and `_on_sigterm` with that manager only.
  - CPython writes the wakeup byte in `trip_signal()` (`Modules/signalmodule.c`), which an OS
    signal reaches only through the C-level `signal_handler()`. That handler is installed only by
    `signal.signal()`, and at startup for SIGINT.
  - In a running launch PID 1, `SigCgt` is `0000000100000002`: SIGINT and signal 33, no SIGTERM.
- **The kernel drops it.** From `pid_namespaces(7)`: a process in an ancestor namespace can "send
  signals to the "init" process of a child PID namespace only if the "init" process has established
  a handler for that signal". Only SIGKILL and SIGSTOP are forced through. This is why launch's
  `_on_sigterm` never ran in the earlier probes.
- **Reproduced in a sealed container** (`--network none`, a launch with one Python child that
  cleans up in `finally`):
  - `docker stop`: nothing logged after the signal, exit 137 after 10.2 s;
  - `--stop-signal SIGINT`: launch logged "user interrupted with ctrl-c (SIGINT)" and sent SIGINT,
    and the child's `finally` logged. The container exited 0 in 0.4 s.
- **SIGINT is launch's graceful path, but only when stdin is not a tty.**
  - On SIGINT each process gets SIGINT when `not due_to_sigint or context.noninteractive`
    (`launch/actions/execute_local.py`:426-431).
  - `ros2 launch` sets `noninteractive` to `not sys.stdin.isatty()`
    (`ros2launch/command/launch.py`:77).
  - `tty: true` makes stdin a tty, and `stdin_open: true` alone does not.
  - In the probe with a tty, launch did not pass SIGINT on. Its 5 s SIGTERM escalation ended the
    child instead.
- **`init: true` is not a fix.** With `--init`, a SIGTERM stop ended launch at once (exit 143 in
  0.1 s), because launch is no longer PID 1 and SIGTERM's default action applies. When the init
  exits, the kernel SIGKILLs the rest of the namespace, so no node is signalled.
- **Launch escalates on its own after SIGINT.**
  - A process still running `sigterm_timeout` (5) seconds later gets SIGTERM, and SIGKILL
    `sigkill_timeout` (5) seconds after that (`execute_local.py`:90-93, :443-479). This holds
    whatever the container's grace period, and the container still exits 0.
  - A launch argument raises them: `sigterm_timeout:=60` let a 20 s teardown finish.
- **A daemon stop sends the stop signal too.** In Docker 29.1.3's source (`daemon/daemon.go`), a
  daemon shutdown stops each container with its own stop signal and timeout. It waits for the
  longest StopTimeout plus 5 s when that exceeds `shutdown-timeout`. Not measured here.

## Acceptance criteria

- [x] **The mechanism, stated with its evidence and reproduced sealed:** `docker stop` with
      nothing logged and exit 137 after the grace, then `--stop-signal SIGINT` with the child's
      `finally` logged and exit 0.
      *2026-10-17: Context above; record §1 and §3.*
- [x] **`stop_signal: SIGINT` on every launch-based service:** the CPU anchor, `inference`, and
      `docker-compose.sim.yml`'s `strafer-sim`. `zenoh-bridge` runs no launch and is untouched.
      *2026-10-17, in `e7e6e2b`.*
- [x] **No service sets `tty: true` or `stdin_open: true`**, and the anchor's comment says so in
      one line.
      *2026-10-17: none does, in any tracked compose file. The anchor's third comment line states
      the rule and the fact. Only `tty` stops launch passing SIGINT on; `stdin_open` alone leaves
      stdin a pipe, and both stay out.*
- [x] **`docker compose stop slam` timed before and after**, slam running alone with no camera.
      *2026-10-17 (record §2):*
      - *`main`: 180.4 s, exit 137. rtabmap kept warning about missing input until the SIGKILL,
        and logged no shutdown line.*
      - *with `stop_signal: SIGINT` (slam's rendered config equal to `e7e6e2b`'s): 0.7 s, exit
        0, with rtabmap's "Saving database/long-term memory...done!". The database was last
        written 53 ms after launch passed the SIGINT on, and rtabmap exited 0.27 s after it.*
- [x] **The sealed driver probe:** under SIGINT the driver's `finally` reaches its stop packet on
      an answering fake port.
      *2026-10-17 (record §4): `base.launch.py` against two pseudo-terminals answering with
      `test_port_fence.py`'s `FakeRoboClaw`.*
      - *Under SIGINT, "Shutting down -- stopping motors." was logged, and a zero
        DRIVE_M1M2_SPEED reached and was answered on both ports 46 ms after `docker stop` was
        issued.*
      - *Under SIGTERM the node's 50 Hz loop ran on for 10.06 s until SIGKILL, with no
        teardown.*
- [x] **A test pins it.** `TestComposeStopSignal` in `strafer_bringup/test/test_bringup.py`
      checks four things:
      - SIGINT on every launch-based service in `docker-compose.yml` and
        `docker-compose.sim.yml`. A launch service is one whose command, string or list, runs
        `ros2 launch`, or whose image is a strafer image;
      - the same for any launch service an overlay adds;
      - no `tty` or `stdin_open` in any tracked compose file;
      - no other stop signal anywhere.

      *2026-10-17: 16 cases. On `main`'s compose files 2 fail, one per base file; at `e7e6e2b`
      all pass.*
- [x] **Docs.**
      *2026-10-17:*
      - *slam's grace-period comment states launch's 10 s escalation;*
      - *a stop-semantics line in `deploy/README.md`'s Notes;*
      - *dated notes on [`executor-exit-cancels-active-goal`](../active/reliability/executor-exit-cancels-active-goal.md),
        [`enriched-lane-rig-stability`](../active/reliability/enriched-lane-rig-stability.md) and the port-fence
        record's §3.*
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance).
      *2026-10-17: swept. Only `deploy/README.md`'s Notes changed; the env-recreate advice in the
      cheatsheet and `deploy-env-config.md` is untouched by a stop signal.*
- [x] No regression: the whole ROS suite and `make env-check` pass.
      *2026-10-18: 888 JUnit cases at `e7e6e2b`, of which 876 passed, 11 skipped, 1 xfailed and
      0 failed; `make env-check` passes.*

## Investigation pointers

- `source/strafer_ros/deploy/docker-compose.yml` (the `x-cpu-svc` anchor, `slam`, `inference`) and
  `docker-compose.sim.yml` (`strafer-sim`).
- In the image: `/opt/ros/humble/lib/python3.10/site-packages/launch/` (`launch_service.py`,
  `actions/execute_local.py`, `utilities/signal_management.py`) and `ros2launch/command/launch.py`.

## Out of scope

- **The executor's cancel on exit.** A stop now reaches the executor, but its exit path still
  cancels nothing: [`executor-exit-cancels-active-goal`](../active/reliability/executor-exit-cancels-active-goal.md).
- **rtabmap's save inside launch's escalation.** Launch sends rtabmap SIGTERM 5 s and SIGKILL
  10 s after the SIGINT, so slam's 180 s grace does not bound its save:
  [`slam-stop-save-window`](../active/reliability/slam-stop-save-window.md).
- **`base` on the robot.** The motor stop seen on the controllers after `docker compose stop
  base` needs both RoboClaws. Record §4 is the driver's teardown against fake controllers.
- **A daemon stop or a reboot**, measured. Docker's source says it sends the stop signal too
  (Context). `init: true` stays deferred in
  [`enriched-lane-rig-stability`](../active/reliability/enriched-lane-rig-stability.md).
