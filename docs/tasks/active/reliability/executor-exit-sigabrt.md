# Stop the executor's ROS client thread before the interpreter exits

**Type:** bug (the executor aborts with SIGABRT on its way out)
**Owner:** Jetson (`strafer_autonomy`)
**Priority:** P2. The robot-side stop runs before the abort. But a handled stop of the `autonomy`
container nearly always ends in a crash exit (15 of 16 idle SIGINTs in the probe), which hides a
real crash among the expected ones.
**Estimate:** S
**Branch:** `task/executor-exit-sigabrt`

## Story

As **the maintainer stopping or restarting the autonomy stack**, I want **the executor to exit
with its own status instead of aborting**, so that **a `-6` in the launch log means a real crash.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)

## Context

- **The symptom.** On the deploy stack the executor's exit ends in
  `terminate called without an active exception`, and launch reports `exit code -6`. It happens in
  two places:
  - at the startup health-check failure, when the VLM or planner service is down;
  - at a handled SIGINT, after "Shutting down autonomy executor.".
- **The client's spin thread is never stopped.** `JetsonRosClient` spins a `SingleThreadedExecutor`
  on a daemon thread, `ros-client-spin` (`strafer_autonomy/clients/ros_client.py:344-349`). Nothing
  stops or joins it, and the class has no shutdown method.
  - Its node's subscriptions are `/d555/color/image_sync`, `/d555/aligned_depth_to_color/image_sync`,
    `/d555/color/camera_info_sync`, `/strafer/odom` and `/global_costmap/costmap`, plus `/tf` and
    `/tf_static` through its `TransformListener` (`ros_client.py:365-377`).
  - On the robot, the camera streams and TF always flow.
- **The mechanism.** This is how rclpy 3.3.21 and CPython 3.10 read in source; no core was
  backtraced.
  - At interpreter exit the thread is still in rclpy's `WaitSet.wait()`, which waits in `rcl_wait`
    inside a `py::gil_scoped_release` scope.
  - A sample on one of its subscriptions, or the context's shutdown, wakes it while the
    interpreter is finalizing, and the guard's destructor retakes the GIL.
  - CPython 3.10 ends a non-main thread that takes the GIL during finalization with `pthread_exit()`.
    The forced unwind meets a `noexcept` frame, and `std::terminate()` aborts the process.
- **The startup path.** The health-check exception leaves `main()` from `build_command_server()`,
  before its `try`/`finally`. Nothing shuts the context down or stops the thread.
- **The SIGINT path.**
  - The executor initializes rclpy without its signal handlers (`executor/main.py:220-226`; see
    [`executor-exit-cancels-active-goal`](../../completed/executor-exit-cancels-active-goal.md)).
    The context therefore shuts down only in `rclpy.try_shutdown()`, as `main()` returns.
  - That shutdown's wake-up reaches the spin thread while the interpreter is finalizing.
  - With rclpy's handlers installed, the context shut down when the signal arrived, so the thread
    was woken out of rclpy's wait while `main()` was still running.
- **The stop still runs first.** `shutdown(server, runner)` runs before `rclpy.try_shutdown()`
  (`main.py:310-312`). When a mission is active, it cancels it and publishes a zero `/cmd_vel`
  (`tests/test_executor_main.py::TestShutdown`). The abort comes after `main()` returns.

## Evidence

The probe record is the Jetson's, on the robot host at `~/rig_next8/sigabrt_probe/`. It holds the
README, one log per run, the scripts, the package drift and `SHA256SUMS`. It deposits as the
fix PR's before-evidence when that PR ships.

- **On the deploy stack.** The executor's launch log in each of three rehearsals of the five-service
  stack shows both paths under launch, each ending in `terminate called without an active
  exception` and `exit code -6`. The logs are on the robot host, under `~/rig_g_real/`.
  - **Startup failure, after the VLM health check's refused connection.** Twice, on 2026-11-08 UTC:
    `gr0-rehearsal/logs/strafer_autonomy.log:84-87` and `gr0-rehearsal2/logs/strafer_autonomy.log:84-87`.
  - **Handled SIGINT, after "Shutting down autonomy executor.".** Once, from launch, on 2026-11-09
    UTC: `gr0-rehearsal3/logs/strafer_autonomy.log:16-20`.
- **The probe's setup.** It ran on 2026-11-09 UTC, 62 runs in all, in transient
  `docker run --rm --network none` containers.
  - The executor was a child of `bash`, with the deploy env and `cyclonedds.autonomy.xml`.
  - Two images: `strafer-cpu:f16d18d6` (`a40d973431e8`, executor code from before
    `executor-exit-cancels-active-goal`) and `strafer-cpu:humble` (`62989627b7ef`, at
    `06f00125c435`).
  - Each image ran its own executor code. Except in the no-traffic startup failure, each also ran
    the other build's code, mounted read-only over the editable install.
- **The probe's modes.**
  - "Startup failure": `VLM_URL` and `PLANNER_URL` point at a closed port, so the check fails at
    once.
  - "Slow startup failure": a stub answers `/health` with 503 after 3 s, so the subscriptions are
    matched and traffic is flowing when the check fails.
  - "Idle SIGINT": the executor gets SIGINT 3 s after it logs "Autonomy executor ready", with no
    mission active.
  - "Traffic": a publisher in the container on the client's camera and odometry topics, with the two
    640x360 images at 30 Hz.
- **Results.** The exit statuses are bash's: 134 is SIGABRT (launch's `-6`), and 139 is SIGSEGV.

  | exit path | traffic | current executor code | code before `executor-exit-cancels-active-goal` |
  |---|---|---|---|
  | startup failure | none | 1 ×3 | 1 ×3 |
  | startup failure | yes | 134 ×6 | 134 ×6 |
  | slow startup failure | yes | 134 ×6 | 134 ×6 |
  | idle SIGINT | none | 134 ×6 | 0 ×6 |
  | idle SIGINT | yes | 134 ×9, 0 ×1 | 0 ×7, 139 ×3 |

- **The abort line.** Every 134 printed `terminate called without an active exception`, and no other
  run did.
- **The SIGINT lines.** Every current-code SIGINT logged "Shutting down autonomy executor." first.
  15 of the 16 earlier-code SIGINTs ended partway through the spin thread's traceback.
- **The images.**
  - Wherever both images ran a code, the outcomes matched to within one run per cell.
  - rclpy 3.3.21, rmw_cyclonedds_cpp 1.3.5, cyclonedds 0.10.5 and Python 3.10.12 are the same in
    both.
- **The cores.** The robot host's `core_pattern` piped cores to apport. Apport kept the probe's newest
  five crash cores, about 160 MB each and root-owned.

## Acceptance criteria

- [ ] **The ROS client's spin thread ends before the interpreter exits**, on every exit path from
      `main()`: a handled SIGINT or SIGTERM, a startup failure, and any other exception.
- [ ] **A handled SIGINT exits 0** and prints no `terminate called`, with the client's camera and
      odometry topics published.
- [ ] **A startup health-check failure exits non-zero with the health check's own error** and prints
      no `terminate called`, with those topics published.
- [ ] **The exit-path order is unchanged.** When a mission is active, it is canceled and a zero
      `/cmd_vel` is published before anything is destroyed, as
      `tests/test_executor_main.py::TestShutdown` pins.
- [ ] **A test** drives the exit with a publisher on at least one of the client's subscribed topics.
      It fails on `main` with the abort and passes on the branch.
- [ ] **The probe holds on the branch's image.** With traffic, three runs each of the startup failure,
      the slow startup failure and the idle SIGINT give no 134 and no 139.
- [ ] If your work invalidates a fact in any referenced context module, package README, top-level
      `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance
      section](../../context/conventions.md#user-facing-documentation-maintenance) for the surface
      list and trigger heuristics.
- [ ] No regression in `make test-autonomy` or `make test-ros`.

## Investigation pointers

- `source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:301-350`: `JetsonRosClient.__init__`,
  its executor and the `ros-client-spin` thread.
- `source/strafer_autonomy/strafer_autonomy/executor/main.py:220-226`: rclpy initialized without
  signal handlers, and `signal.default_int_handler` for SIGINT and SIGTERM.
- `source/strafer_autonomy/strafer_autonomy/executor/main.py:298-312`: `build_command_server()`
  before the `try`, then `shutdown(server, runner)` and `rclpy.try_shutdown()` in the `finally`.
- `source/strafer_autonomy/strafer_autonomy/executor/command_server.py:306-333`: the parallel
  health checks.
- `source/strafer_autonomy/tests/test_executor_main.py`: `TestShutdown`.
- rclpy 3.3.21: `src/rclpy/wait_set.cpp:244-248` and `rclpy/executors.py:320-323`.

## Out of scope

- A goal-active handled stop on the robot, and whether it still cancels its goal first.
- Whether a startup health-check failure should be fatal at all; that is
  [`executor-startup-health-check-contract`](executor-startup-health-check-contract.md).
- rclpy itself, and the inference node's exit.
- The host's core-dump configuration.
