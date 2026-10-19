# Cancel the active goal when the executor exits

**Type:** bug (an executor exit leaves its trained-policy goal driving)
**Owner:** Jetson (`strafer_autonomy` executor, `strafer_inference`, `deploy/`)
**Priority:** P1 for goal-b. On the real robot this is an unattended drive: the robot keeps
following the policy after the process that commanded it is gone.
**Estimate:** S–M
**Branch:** `task/executor-exit-cancels-active-goal`

## Story

As **whoever stops, restarts or recreates `autonomy` while a policy goal is running**, I want
**the executor's exit to cancel that goal and stop the robot**, so that **no goal keeps driving
after the process that sent it is gone.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`inference-finished-successor-stop`](inference-finished-successor-stop.md), its reachability
  box: where this was read, as one of the two paths that overlap goals on the node.
- [`executor-cancel-mid-motion-cmd-vel-zero`](../../completed/executor-cancel-mid-motion-cmd-vel-zero.md):
  the cancel path and its zero `/cmd_vel`, which an exit never reaches.

## Context

Code read at `6cd6fd7`. The image facts were read and probed on 2026-10-16 in `strafer-cpu:humble`
and `strafer-gpu:humble` (revision label `9f7d20758d5b`).

- **The exit path cancels nothing.**
  - `executor/main.py`:291-298 spins the command server, catches `KeyboardInterrupt`, and in
    `finally` calls `server.destroy()` and `rclpy.try_shutdown()`. The `runner` that
    `build_command_server` returns (:284) is not used again.
  - `AutonomyCommandServer.destroy` (`command_server.py`:124-138) destroys the action server, the status
    service, the executor and the node. It does not call into the runner.
  - No `atexit`, signal or shutdown hook in `strafer_autonomy` cancels anything.
- **The mission thread dies without its own cancel.** It is a daemon thread
  (`mission_runner.py`:316-326), blocked in `_wait_for_nav_result` (`clients/ros_client.py`:1811-1828).
  That wait never checks `rclpy.ok()`, so the thread dies with the process before its deadline
  cancel (1036-1037; hybrid 1182-1183).
- **The inference node keeps driving.**
  - Its execute loop (`inference_node.py`:983-1032) leaves only on a client cancel (984-989), a
    supersession (991-996), arrival (998-1004), its 60 s bound on the node clock (1018-1027), or
    its own rclpy shutdown (1031-1032). It has no client-liveness check.
  - Until then every tick publishes the policy's action (1209-1217). The bound's stop comes from
    the loop's `finally` (1033-1044).
  - So the robot drives for up to 60 s of node time, more wall time in sim below RTF 1. If the
    restarted executor's first policy goal arrives first, it supersedes the old one with no stop
    in between.
  - The driver's 0.5 s `cmd_vel` watchdog does not help, because `/cmd_vel` keeps arriving.
- **`inference` and `autonomy` are separate containers** (the `inference` and `autonomy`
  services in `deploy/docker-compose.yml`). Restarting `autonomy` leaves `inference` and its goal
  running. The launch runs the executor as a plain process with no respawn
  (`autonomy_nodes.launch.py`:55-63).
- **A container stop never reaches the executor's `finally`.**
  - `ros2 launch` is PID 1: the image's entrypoint `exec`s its command
    (`deploy/docker/Dockerfile.cpu`:86-95). `docker compose stop` and `restart` send SIGTERM.
  - Humble's launch has no SIGTERM path that signals its children. Its `_on_sigterm`
    (`launch/launch_service.py`, launch 1.0.14 in `strafer-cpu:humble`) is written to log "using
    SIGTERM can result in orphaned processes" and cancel its run task. In the probe below it did
    not even do that, and why was not established.
  - A probe launch with one Python child showed both paths. Under `docker stop`, launch logged
    nothing after the signal, the container exited 137 after 10.2 s (Docker's SIGKILL), and the
    child's `finally` never ran. Under
    `docker kill -s SIGINT`, launch sent SIGINT to the child, the child's `finally` ran, and the
    container exited 0 in 0.4 s.
  - `autonomy` sets no `stop_signal`. The same holds for every service on the shared CPU anchor,
    `base` included.
  - *2026-10-17: the stop signal is now in place.
    [`compose-stop-signal-sigint`](../../completed/compose-stop-signal-sigint.md) sets
    `stop_signal: SIGINT` on every launch-based service, `autonomy` included, so a
    `docker compose stop` or `restart` reaches the executor as a SIGINT.*
    - *Why `_on_sigterm` never ran: launch catches no SIGTERM at the OS level, and the kernel
      drops a signal with no handler that is sent to a PID namespace's init
      ([record](../../../measurements/compose-stop-signal-2026-10-17/README.md) §1).*
    - *The cancel remains this brief's own fix. The executor's exit path still cancels nothing,
      and rclpy has shut its context down before `finally` runs, so the boxes below stand.*
- **A cancel sent from `finally` would not go out either.** A SIGINT shuts rclpy's context down
  before `finally` runs, and `rclpy.ok()` is false there, in both images. In `strafer-gpu:humble`
  (rclpy 3.3.22), with the signal raised in-process, `spin` raised `ExternalShutdownException`.
  In `strafer-cpu:humble`, the executor's image, with launch passing on the SIGINT, `spin` raised
  `KeyboardInterrupt`.
- **Scope of the lane.** It needs a policy backend for the executor (`executor/main.py`:269-274).
  Committed canon is `nav2` (`deploy/compose/autonomy.env`:7), and the rig sets a policy backend
  in its local override. The sim-in-the-loop compose runs the executor and, with a policy
  backend, the inference node in one launch (`bringup_sim_in_the_loop.launch.py`:152-163,
  204-221). A container stop or restart there ends both, and only an executor crash leaves the
  node driving. The restart gap is specific to the decomposed deploy under `--profile policy`.

## Acceptance criteria

- [ ] **A test that the executor's exit path cancels the active goal**, failing on `main`.
  - With a mission active, the exit path calls the runner's cancel, which cancels the active
    navigation and publishes a zero `/cmd_vel`, before the server and node are destroyed.
  - With no mission active, the exit path cancels nothing.
  - A raising cancel still lets the destroy run.
  - One seam: a module-level shutdown function in `executor/main.py` that the `finally` calls,
    tested in `tests/test_executor_main.py` with a `MagicMock` ros client, as the
    `_make_runner` helpers in `tests/test_new_skills.py` build one.
- [ ] **The executor cancels and stops before it destroys anything, on a live context, on every
      stop path.**
  - The cancel and the zero twist have to go out before rclpy's signal handling shuts the
    context down. For example, the executor installs its own SIGINT/SIGTERM handling, or cancels
    from a pre-shutdown hook.
  - A container stop has to reach the executor at all. For example, `stop_signal: SIGINT` on
    `autonomy`, so that launch takes its graceful path.
  - The PR says which stop paths it covers: `docker compose stop`, `restart`, `down`, Ctrl-C,
    and a crash (which no handler can cover).
- [ ] **The inference node ends a goal whose client is gone (the second fence).** A
      client-liveness check aborts the goal and publishes a stop, so that a crash or SIGKILL that
      no executor handler sees still ends the drive. The test is a goal whose client stops,
      ending in a stop well inside the 60 s bound.
- [ ] **The bridge-lane check.** On the sim-bridge lane with a policy backend,
      `docker compose restart autonomy` during a policy goal ends that goal (canceled, or aborted
      by the second fence), and a zero `/cmd_vel` is the last command. Record the time from the
      restart to the stop.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-ros` and `make test-autonomy` pass.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/executor/main.py` (`main`, :284-298),
  `executor/command_server.py` (`destroy`, :124-138; the execute loop's shutdown abort at
  :215, :233-234), and `executor/mission_runner.py` (`cancel_active_mission`, :342-367;
  `_cancel_robot_actions`, :2329-2343).
- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py`
  (`cancel_active_navigation`, :1281-1301; the goal-handle slot, :1025-1026 and :1171-1172).
- `source/strafer_ros/strafer_inference/strafer_inference/inference_node.py` (the execute loop,
  :983-1044).
- `deploy/docker-compose.yml`'s CPU-service anchor and the `autonomy` service, for the stop signal.

## Out of scope

- Two submissions racing `start_mission`:
  [`executor-start-mission-atomic`](executor-start-mission-atomic.md).
- The stop ordering when a goal ends inside its predecessor's drain:
  [`inference-finished-successor-stop`](inference-finished-successor-stop.md).
- `base`'s stop path. The driver gets the same SIGKILL on `docker compose stop base`, so its
  teardown, which stops the motors (`destroy_node`, called from `main`'s `finally`), never runs.
  A shared `stop_signal` would cover it, but its check needs the RoboClaws.
