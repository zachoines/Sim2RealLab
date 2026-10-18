# The compose stop signal, 2026-10-17

Every strafer service that runs ROS nodes runs `ros2 launch` as PID 1, and at `0a975ef` none sets a
stop signal, so `docker compose stop` and `restart` send SIGTERM. This record shows that the
SIGTERM reaches no node. Launch installs no OS-level SIGTERM handler, and the kernel drops a signal
with no handler that is sent to a PID namespace's init, so every node runs on until Docker's SIGKILL
at the grace period. The fix in
[`compose-stop-signal-sigint`](../../tasks/completed/compose-stop-signal-sigint.md) sets
`stop_signal: SIGINT` on every launch-based service. On SIGINT, launch takes its Ctrl-C shutdown path
and, since no service has a tty, sends SIGINT to each node.

It makes **no hardware claim**. No camera, no RoboClaw and no robot motion were involved. `slam`
ran alone, and the driver ran against fake controllers on pseudo-terminals in a sealed container.

**Result.**
- **`docker compose stop slam`, slam alone with no camera:**

  | compose file | stop took | exit | rtabmap after the signal | `rtabmap.db` mtime |
  |---|---|---|---|---|
  | `main` `0a975ef` | **180.4 s** | **137** | kept warning "Did not receive data" every 5 s (36 times), then SIGKILL; no shutdown line | unchanged since startup |
  | `stop_signal: SIGINT` (`slam` as at `e7e6e2b`) | **0.7 s** | **0** | "Saving database/long-term memory...done!"; exited 0.27 s after launch passed the SIGINT on | 53 ms after launch passed the SIGINT on |

- **The mechanism.** In a launch PID 1, `/proc/1/status` `SigCgt` is `0000000100000002`: SIGINT and
  signal 33, no SIGTERM (§1).
- **Sealed probes**, launch as PID 1 with one Python child, or under `docker-init` in the two
  `--init` cases (§3):
  - `docker stop`: nothing logged, exit 137 after 10.2 s;
  - `--stop-signal SIGINT`: the child's `finally` logged, exit 0 in 0.4 s;
  - with a tty, launch does not pass SIGINT on. Its 5 s SIGTERM ends the child instead;
  - with `--init`, a SIGTERM ends launch at once (exit 143 in 0.1 s), so no node is signalled.
- **Launch's own escalation.** After the SIGINT, launch sends a node still running SIGTERM at 5 s
  and SIGKILL at 10 s, whatever the container's grace period, and the container still exits 0. So
  slam's 180 s grace does not bound rtabmap's save: launch ends a save still running at 10 s, and
  whether a save survives the 5 s SIGTERM was not measured (§3; filed as
  [`slam-stop-save-window`](../../tasks/active/reliability/slam-stop-save-window.md)).
- **The driver.** `base`'s launch file ran against two answering fake RoboClaws (§4):
  - **SIGTERM:** the node's 50 Hz loop went on for 10.06 s after the signal, until SIGKILL. It had
    no teardown.
  - **SIGINT:** the node logged "Shutting down -- stopping motors.", and each controller received
    and answered a zero DRIVE_M1M2_SPEED 46 ms after `docker stop` was issued. Both ports then
    closed.
- **Tests.** `test_bringup.py`'s `TestComposeStopSignal` fails 2 of its 16 cases on `main`'s compose
  files, and the whole ROS suite passes at `e7e6e2b`: 888 cases, 0 failed (§5).

## 1. The mechanism

Read in `strafer-cpu:humble` (revision label `9f7d20758d5b`): launch 1.0.14, ros2launch 0.19.13,
Python 3.10.12. The excerpts, with their line numbers and file digests, are in
`image_reads/launch_signal_paths.txt`.

- **PID 1.** The entrypoint `exec`s the service's command (`deploy/docker/Dockerfile.cpu`:86-95,
  `Dockerfile.gpu`:79-87), so `ros2 launch` is PID 1. `docker top` on the running `slam` showed it
  with three children: `point_cloud_xyz_node`, `pointcloud_to_laserscan_node` and `rtabmap`
  (`slam/before/top_before_stop.txt`).
- **Launch sets no SIGTERM handler.** Its signal manager's docstring reads: "The underlying
  mechanism is built around `signal.set_wakeup_fd` so as to not interfere with regular handlers
  installed via `signal.signal`" (`launch/utilities/signal_management.py`:49-51).
  `launch_service.py`:215-216 registers `_on_sigint` and `_on_sigterm` with that manager only.
- **CPython writes the wakeup byte only for a signal that has a handler.** In CPython 3.10's
  `Modules/signalmodule.c`, `trip_signal()` writes it, and an OS signal reaches `trip_signal()`
  only through the C-level `signal_handler()`. That handler is installed by `signal.signal()`, and
  at startup for SIGINT.
- **What the kernel sees.** In a running launch PID 1, `SigCgt` is `0000000100000002`: SIGINT and
  signal 33. The rclpy child catches SIGINT, SIGTERM and 33 (`probes/<case>.sigmask.txt` for the
  cases without `--init`).
- **The kernel drops the SIGTERM.** From `pid_namespaces(7)`: "a process in an ancestor namespace
  can—subject to the usual permission checks described in kill(2)—send signals to the "init"
  process of a child PID namespace only if the "init" process has established a handler for that
  signal." Only SIGKILL and SIGSTOP are forced through. So launch's `_on_sigterm`, which would log
  "using SIGTERM can result in orphaned processes", never runs.
- **SIGINT takes launch's Ctrl-C path.** `_on_sigint` starts a shutdown with `due_to_sigint=True`
  (`launch_service.py`:191-200). Each process action then sends its process SIGINT when
  `not due_to_sigint or context.noninteractive` (`launch/actions/execute_local.py`:426-431).
  `ros2 launch` sets `noninteractive` to `not sys.stdin.isatty()`
  (`ros2launch/command/launch.py`:77).
- **A tty stops the forwarding.** Under `tty: true` the container's stdin is a tty. Under
  `stdin_open: true` alone it is not (`probes/isatty.txt`: `-t` True, `-i` False, neither False).
- **Launch's escalation.** A process still running `sigterm_timeout` seconds after the shutdown
  began gets SIGTERM. One still running `sigterm_timeout + sigkill_timeout` seconds after it gets
  SIGKILL. Both are launch configurations defaulting to 5 (`execute_local.py`:90-93, :443-479).

## 2. `slam`'s stop, timed

Both arms ran `docker compose up -d slam`, waited 45 s, timed `docker compose stop slam`, and read
the exit code and logs (`slam/slam_stop_timing.sh`). Each ran slam alone, under the compose project
`stopsig` with a fresh `stopsig_strafer_ros_home` volume, removed after the arm. A one-key
override (`slam/domain87.yml`) set `ROS_DOMAIN_ID` to 87, to keep it off the deploy's domain 42.
Nothing published to slam's inputs.

| | before | after |
|---|---|---|
| compose file | `main` `0a975ef` | the branch's working tree before its work commit; its rendered `slam` config is byte-identical to `e7e6e2b`'s (`slam/after/compose_config_slam_at_e7e6e2b.yml`) |
| `StopSignal` / `StopTimeout` | default (SIGTERM) / 180 | SIGINT / 180 |
| stop sent | 19:36:24.467 UTC | 19:40:11.533 UTC |
| `docker compose stop` returned | 19:39:24.843 (**180.4 s**) | 19:40:12.240 (**0.7 s**) |
| exit code | **137** | **0** |
| `rtabmap.db` mtime after the stop | 19:35:41.437, from startup | 19:40:11.779 |

**Before**, rtabmap's log after the signal shows only its no-input warning, every 5 s until the
SIGKILL (`slam/before/slam.log`, 36 times after 19:36:24). It shows no launch line, no signal line
and no save line.

**After**, abridged from `slam/after/slam.log` (timestamps, rtabmap's parameter line and the two
other nodes' `signal_handler` lines dropped):

```
[WARNING] [launch]: user interrupted with ctrl-c (SIGINT)
[INFO] [rtabmap-3]: sending signal 'SIGINT' to process[rtabmap-3]
[INFO] [pointcloud_to_laserscan_node-2]: sending signal 'SIGINT' to process[pointcloud_to_laserscan_node-2]
[INFO] [point_cloud_xyz_node-1]: sending signal 'SIGINT' to process[point_cloud_xyz_node-1]
[rtabmap-3] [INFO] [...] [rclcpp]: signal_handler(SIGINT/SIGTERM)
[INFO] [pointcloud_to_laserscan_node-2]: process has finished cleanly [pid 24]
[INFO] [point_cloud_xyz_node-1]: process has finished cleanly [pid 22]
[rtabmap-3] rtabmap: Saving database/long-term memory... (located at /root/.ros/rtabmap.db)
[rtabmap-3] rtabmap: Saving database/long-term memory...done! (located at /root/.ros/rtabmap.db, 0 MB)
[INFO] [rtabmap-3]: process has finished cleanly [pid 26]
```

The timing of the save:
- Launch logged the SIGINT at 19:40:11.726.
- The database, new and empty, was last written at 19:40:11.779, and rtabmap exited at 19:40:11.994.
- The two "Saving" lines arrived at 19:40:11.966, after that last write. They are rtabmap's plain
  console output, likely flushed at exit, so their arrival does not time the save.

A real map's save takes longer, and launch's escalation bounds it (§3).

## 3. Sealed probes

`probes/run_probes.sh` runs each case but the two `--init` ones, which `probes/run_init_probes.sh`
runs. Each case is a `strafer-cpu:humble` container with `--network none` and no `/dev` bind. The
script waits for the child to start, reads `/proc/<pid>/status` and `ps`, sends `docker stop`, and
records the time and exit code. The launch file `probe/probe.launch.py` starts one Python child,
`probe/child.py`. That child mirrors `roboclaw_node.main()`: it spins, catches `KeyboardInterrupt`,
and cleans up in `finally`.

| case | container | `docker stop` took | exit | what the log shows |
|---|---|---|---|---|
| `sigterm_child` | default | 10.2 s | 137 | nothing after the signal; the child's `finally` never ran |
| `sigint_child` | `--stop-signal SIGINT` | 0.4 s | 0 | launch: "user interrupted with ctrl-c (SIGINT)", sends SIGINT; child: `KeyboardInterrupt`, "finally ran" |
| `sigint_tty_child` | `-t --stop-signal SIGINT` | 5.4 s | 0 | launch does not send SIGINT; after 5 s "escalating to 'SIGTERM'"; the child's `finally` runs, and it exits 1 on `ExternalShutdownException` |
| `sigint_stdin_child` | `-i --stop-signal SIGINT` | 0.4 s | 0 | as `sigint_child` |
| `sigint_slow_child` | SIGINT, `--stop-timeout 180`, a child whose teardown takes 20 s and ignores SIGTERM | 10.3 s | 0 | SIGTERM at +5.0 s, SIGKILL at +10.0 s; the child dies with -9 mid-teardown |
| `sigint_slow_child_timeout60` | the same, with the launch argument `sigterm_timeout:=60` | 20.4 s | 0 | "teardown complete after 20.0 s" |
| `init_sigterm` | `--init` | 0.1 s | 143 | nothing after the signal; `docker-init` passed SIGTERM to launch, which, no longer PID 1, ended at once |
| `init_sigint` | `--init --stop-signal SIGINT` | 0.4 s | 0 | as `sigint_child` |

**Why `init: true` is not a fix by itself.**
- With an init as PID 1, SIGTERM's default action ends launch before it signals anything. When the
  init then exits, the kernel SIGKILLs the rest of the namespace, the child included.
- The child's own output could not show that, because launch relays it and was already gone. The
  read of launch's signal mask under `--init` also missed: its `pgrep` matched `docker-init`'s
  arguments, so it read `docker-init` twice.
- SIGINT is launch's graceful path either way.

`sigint_slow_child` exits 0 although its child was SIGKILLed mid-teardown. An exit code of 0 is
therefore no proof that a node finished its teardown. Only the node's own log line shows that.

The probe child survives the 5 s SIGTERM because it installs a handler for it. rtabmap's rclcpp
handles SIGTERM as it handles SIGINT; its log line reads `signal_handler(SIGINT/SIGTERM)`. Whether
a save in progress survives that second signal was not measured. Launch therefore ends rtabmap's
save within 5 to 10 s of the SIGINT.

## 4. The driver's teardown, against answering fake controllers

`probe/probe_base.launch.py` starts `probe/fake_roboclaw_pty.py`, then, 3 s later, `base`'s compose
command's launch file, `strafer_bringup/launch/base.launch.py`. The container mounts `main`'s
`strafer_driver` package, launch and config, `strafer_bringup/launch` and `strafer_shared` over the
image's copies. The image itself carries `9f7d207`'s driver, from before the port fence.

The fake puts two pseudo-terminals at `/dev/roboclaw0` and `/dev/roboclaw1`. It answers with
`FakeRoboClaw` from `strafer_driver/test/test_port_fence.py` (0x80 and 0x81), so the node runs
unpatched, through pyserial and a real tty. It logs every packet. On SIGINT it keeps answering,
and exits after 1.5 s with no traffic.

The node auto-detected both, set the PID constants, and ran its 50 Hz loop. Each tick sends each
controller a drive command and four speed and encoder reads. With no `cmd_vel` published, its
watchdog zeroes the drive, so every DRIVE_M1M2_SPEED in both runs carries zero speeds
(`probes/packets_driver_*.jsonl`). Times below are from when `docker stop` was issued; launch
logged the SIGINT 42 ms later.

| | SIGTERM (`driver_sigterm`) | SIGINT (`driver_sigint`) |
|---|---|---|
| `docker stop` took / exit | 10.2 s / 137 | 1.9 s / 0; the node finished at +0.25 s, and the rest is the fake's idle wait |
| node log after the signal | nothing | "Shutting down -- stopping motors." |
| traffic after the signal | 5040 packets over 10.06 s: the loop's 504 ticks, until SIGKILL | 25 packets over 46 ms: the ticks in flight, then the teardown |
| teardown stop | none | zero DRIVE_M1M2_SPEED to 0x80 and 0x81 at +46 ms, both answered |
| ports closed | not before the SIGKILL | the fake saw `/dev/roboclaw0` close at +47 ms and `/dev/roboclaw1` at +67 ms (it polls every 10 ms) |

Abridged from `probes/driver_sigint.log` (timestamps dropped):

```
[WARNING] [launch]: user interrupted with ctrl-c (SIGINT)
[INFO] [roboclaw_node-3]: sending signal 'SIGINT' to process[roboclaw_node-3]
[roboclaw_node-3] [INFO] [...] [roboclaw_node]: Shutting down -- stopping motors.
[fake_roboclaw-1] fake_roboclaw: rx /dev/roboclaw0 0x80 DRIVE_M1M2_SPEED data=0000000000000000 answered=True at 1792266151.564
[fake_roboclaw-1] fake_roboclaw: rx /dev/roboclaw1 0x81 DRIVE_M1M2_SPEED data=0000000000000000 answered=True at 1792266151.564
[fake_roboclaw-1] fake_roboclaw: /dev/roboclaw0 closed at 1792266151.565
[INFO] [roboclaw_node-3]: process has finished cleanly [pid 26]
```

The teardown's `get_logger().info` also printed "Failed to publish log message to rosout:
publisher's context is invalid". rclpy has already shut its context down when `finally` runs, as
[`executor-exit-cancels-active-goal`](../../tasks/active/reliability/executor-exit-cancels-active-goal.md)
records. The serial writes do not depend on that context.

## 5. Tests

`TestComposeStopSignal` in `strafer_bringup/test/test_bringup.py` (16 cases) checks four things:
- every service in `docker-compose.yml` and `docker-compose.sim.yml` that runs a launch has
  `stop_signal: SIGINT`. A launch service is one whose command, as a string or a list, contains
  `ros2 launch`, or whose image is `strafer-cpu` or `strafer-gpu`;
- the same for any launch service that an overlay adds;
- no service in any tracked compose file sets `tty` or `stdin_open`;
- no file sets another stop signal.

| run | tree | result (JUnit) |
|---|---|---|
| `ROS_TEST_PKGS=strafer_bringup … -k TestComposeStopSignal` | `main` `0a975ef` with `e7e6e2b`'s `test_bringup.py` copied in | 16 cases, **2 failed**: `test_launch_services_stop_with_sigint` for both files |
| whole ROS suite (`tools/run_ros_tests.sh ros`) | `e7e6e2b` | 888 cases: 876 passed, 11 skipped, 1 xfailed, 0 failed |

`make env-check` passes at `e7e6e2b` (`tests/env_check.txt`).

## What this does not show

- **`base` on the robot.** §4 is the driver's own teardown against fake controllers. The on-robot
  read, `docker compose stop base` with both RoboClaws attached and the motor stop seen on the
  controllers, needs the hardware.
- **A real map's save time**, and whether rtabmap's save survives launch's 5 s SIGTERM. The timed
  database was empty. Raising launch's timeouts for `slam` is
  [`slam-stop-save-window`](../../tasks/active/reliability/slam-stop-save-window.md).
- **The executor's exit path.** A stop now reaches the executor, but it still cancels nothing on
  exit: [`executor-exit-cancels-active-goal`](../../tasks/active/reliability/executor-exit-cancels-active-goal.md).
- **A daemon stop or a reboot.**
  - In the source of Docker 29.1.3, the robot's version (`daemon/daemon.go`, `shutdownTimeout`
    and `shutdownContainer`), a daemon shutdown stops each container with its own stop signal and
    timeout.
  - It waits for the longest StopTimeout plus 5 s when that exceeds `shutdown-timeout`.
  - So a daemon stop now sends SIGINT too. That path was not measured here.

## Evidence

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `compose-stop-signal-2026-10-17/record-files/` |
| deposit commit | `dfea72bb3566e98d658a98ab067c2a915795ab97` |

The deposit was made in two commits. `23c0abc4` holds the first, with its test runs at an earlier,
unpushed work commit. `dfea72bb` replaces `tests/` with the runs at `e7e6e2b`, adds the `e7e6e2b`
render of `slam`'s config, and dates the correction in `DEPOSIT.md`; no other file changed.

The deposit holds:
- `slam/`: `slam_stop_timing.sh`, `domain87.yml`, and per arm (`before/`, `after/`) the arm
  stamps, the rendered compose config, the `docker inspect` reads, `docker top`, the full `slam`
  logs and the `ls` of the volume after the stop;
- `probes/`: `run_probes.sh` and `run_init_probes.sh`, each case's log, `docker inspect`, `ps` and
  signal-mask read, `isatty.txt` and the two packet logs;
- `probe/`: the probe children, launch files and the fake controller;
- `image_reads/launch_signal_paths.txt`: the launch source lines cited in §1;
- `tests/`: `run_tests_evidence.sh`, the JUnit, console and `tree.txt` of both runs in §5, and
  `env_check.txt`;
- `DEPOSIT.md`, with the host, the trees and the sha256 of every file.

Paths in this record resolve in that directory. Verify the digests with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/compose-stop-signal-2026-10-17/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

Hosts, trees and times are in [`provenance.md`](provenance.md).
