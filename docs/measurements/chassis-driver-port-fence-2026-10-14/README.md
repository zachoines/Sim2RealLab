# The chassis driver's port fence, 2026-10-14

With a RoboClaw missing, the driver at `6cd6fd7` falls back to that controller's configured port:
`/dev/ttyACM0` for the front, `/dev/ttyACM1` for the rear. It writes velocity-PID and drive packets
there. When that port is the D555's CDC serial interface, as `/dev/ttyACM0` is whenever the camera
enumerates first, the packets go to the camera. Detection's probe reaches every `/dev/ttyACM*`
whenever no `/dev/roboclaw*` exists, as on this host, where `99-strafer.rules` is not installed.

This record reads the driver before and after the fence in
[`chassis-driver-port-fence`](../../tasks/completed/chassis-driver-port-fence.md), runs its tests
on both trees, and states how the `base` service maps serial devices. It makes **no hardware
claim**: no RoboClaw was connected, `base` was not started, and no run had the host's
`/dev/ttyACM0` in its container. §7 gives the on-site check as commands.

The fence shipped in two work commits. The first pass, on 2026-10-14, ran at `62bb9ee`. The second
pass, on 2026-10-16, ran at `aa1a1c7`. That commit added:
- a refusal when both addresses resolve to one device;
- the launch's end on a refusal;
- the fence on `diagnose_roboclaw.py`'s explicit ports;
- a reconnect that also closes an interrupted probe's port.

The figures below are the second pass's. The first pass's runs stay in the deposit.

**Result.**
- **On `main` the driver opens the camera's port.** Here a fake serial layer stands in for the
  robot, with the camera alone at `/dev/ttyACM0`. In that layer:
  - `main`'s node starts;
  - it opens `/dev/ttyACM0` three times: two detection probes, then the configured front port;
  - it writes to that port.
- **With the fence, the node refuses.** In the same setup it opens nothing and exits 1. With one
  controller present, the only traffic is detection's address probe of `/dev/roboclaw*`. It also
  refuses when both addresses resolve to one device.
- **A refusal ends `base`.** `driver.launch.py` ends the launch when the driver exits, with exit 1
  after a refusal, so compose's `restart: unless-stopped` runs detection again. Sealed, `base`'s
  compose command exits 1 after the refusal (§6).
- **Tests.** `aa1a1c7`'s test modules were run on three trees:
  - on `main`, 20 of 87 driver tests and all 5 bringup default tests fail;
  - on `3a42c23`, the head before the second pass, 9 of the 11 tests that pass added fail; the
    other two are controls;
  - on `aa1a1c7`, the whole ROS suite passes: 860 passed and 11 skipped of 871.
- **Mutations.** Sixteen mutations each break one piece of the fence, and each fails at least one
  test. A seventeenth is equivalent and fails none, as expected.
- **The real entry point.** `ros2 run strafer_driver roboclaw_node` from `aa1a1c7`, in a container
  with no host `/dev` bind and no network, logs the refusal and exits 1 three ways (§6).
- **Compose.** `base` binds `/dev` wholesale and allows every ttyACM device, so the container sees
  the camera's port, and a missing `/dev/roboclaw*` symlink does not stop it starting. The node's
  refusal is the fence (§3).

## 1. The defect at `6cd6fd7`

All paths are under `source/strafer_ros/`.

| step | where |
|---|---|
| Detection scans `/dev/roboclaw*`, and with none present scans `/dev/ttyACM*` | `strafer_driver/strafer_driver/roboclaw_interface.py`:414-417 |
| Each candidate gets a `read_main_battery` probe for both addresses | `roboclaw_interface.py`:367-387, :422-429 |
| One controller found: the other keeps its configured port | `strafer_driver/strafer_driver/roboclaw_node.py`:120-135 |
| None found: "Falling back to configured ports" | `roboclaw_node.py`:136-140 |
| Each configured port is opened and sent the velocity PID, with no probe | `roboclaw_node.py`:150-165, :225-238 |
| A drive command on every 50 Hz tick | `roboclaw_node.py`:210-211, :309-317 |
| The error state reopens both ports every 2 s, and a bare open counts as success | `roboclaw_node.py`:251-265; `roboclaw_interface.py`:143-150 |
| The configured defaults are `/dev/ttyACM0` and `/dev/ttyACM1` | `strafer_driver/launch/driver.launch.py`:22-31, and `base`, `perception`, `slam`, `navigation` and `autonomy` in `strafer_bringup/launch/` |
| `diagnose_roboclaw.py` takes any `--front`/`--rear` path, and after a failed battery read goes on to the PID read and the duty-cycle drive | `diagnose_roboclaw.py`:114-119, :128-150, :160-161 |

The launch arguments are passed after the params file, so they override `driver_params.yaml`'s
`/dev/roboclaw0` and `/dev/roboclaw1`.

**On this host,** at the time of the runs:
- `/dev/ttyACM0` was the D555's CDC interface: char 166:0, `ID_VENDOR_ID=8086`,
  `ID_MODEL_ID=0b56`, under `usb2/2-2`.
- `99-strafer.rules` was not installed; `/etc/udev/rules.d` held only NVIDIA rules.
- There was no `/dev/roboclaw*`.

## 2. The fence at `aa1a1c7`

| path | behaviour | where |
|---|---|---|
| detection | scans `/dev/roboclaw*` only | `roboclaw_interface.py`:383, :446 |
| `auto_detect` true, fewer than two found | logs the count and the missing addresses, raises `PortFenceError` before any `RoboClawInterface` is opened for driving | `roboclaw_node.py`:127-145, :274-280 |
| `auto_detect` true, both addresses on one device | refused; only the address probe was sent | `roboclaw_node.py`:146-153; `roboclaw_interface.py`:393-395 |
| `auto_detect` false, `front_port` and `rear_port` resolve to one device | refused before anything is opened | `roboclaw_node.py`:163-168 |
| `auto_detect` false, a port that is not a `/dev/roboclaw*` name | refused before anything is opened, unless `allow_unverified_ports` | `roboclaw_node.py`:169-174 |
| `auto_detect` false, a port that does not answer the probe | refused; only the probe was written | `roboclaw_node.py`:180-189 |
| startup PID write | reached only after the refusals above, so only on ports that answered | `roboclaw_node.py`:199-214 |
| error-state reconnect | the reopened port must answer the probe, or it is closed again and the node stays in the error state; an interrupted probe closes it too | `roboclaw_interface.py`:145-164; `roboclaw_node.py`:308-322 |
| a `termios` error during a probe | treated as no answer | `roboclaw_interface.py`:159, :417 |
| `main()` | returns 1 on a refusal, and the console script exits with it | `roboclaw_node.py`:568-586 |
| the driver's exit | ends the launch: with exit 1 after a non-zero exit outside the launch's own shutdown, otherwise with a `Shutdown` | `driver.launch.py`:11-20, :60-61 |
| launch defaults | `/dev/roboclaw0` and `/dev/roboclaw1` in `driver.launch.py` and the five bringup launch files | `driver.launch.py`:36, :41 |
| `diagnose_roboclaw.py` | refuses a `--front`/`--rear` path that is not a `/dev/roboclaw*` name unless `--allow-unverified-ports`, and stops after a failed battery read | `diagnose_roboclaw.py`:169-182, :117-124 |

`allow_unverified_ports` defaults to false (`driver_params.yaml`). It relaxes only the name check
under `auto_detect` false, never the probe, and detection ignores it.

`tune_pid.py` and `diagnose_roboclaw.py` share `detect_roboclaws`, so they now find controllers
only at `/dev/roboclaw*` too. Their messages say so and point at `make udev`.

## 3. How `base` maps serial devices

From `deploy/docker-compose.yml` at `aa1a1c7` (`base` at :65-84, with the `x-cpu-svc` anchor at
:52-59):
- `group_add: [ dialout ]` (:77);
- `device_cgroup_rules: [ "c 166:* rmw" ]` (:80), which admits every ttyACM character device.
  The D555's CDC is major 166 too;
- the volume `/dev:/dev` (:83): the whole host `/dev`, so udev's symlinks appear in the container
  as the host creates them;
- no `devices:` list and no `privileged`, here or in any tracked override under `deploy/`;
- `restart: unless-stopped` from the anchor (:55);
- `command: ros2 launch strafer_bringup base.launch.py` (:84), which includes
  `description.launch.py` and `driver.launch.py` (`base.launch.py`:44, :51).

A missing `/dev/roboclaw*` symlink therefore does not stop the container. The node's refusal is
the fence.

**When the driver exits, the launch ends.**
- `driver.launch.py`'s `on_exit` is a callable. After a non-zero exit outside the launch's own
  shutdown, it raises, so launch stops `robot_state_publisher` and returns 1. The container exits 1,
  and `restart: unless-stopped` starts it again, which runs detection again.
- The callable is needed because Humble's launch, as installed in the image, returns non-zero
  only when it catches an exception. A bare `Shutdown` also ends the launch, but with 0 (§6's control). Compose restarts
  the container either way.
- An exit during the launch's own shutdown, such as Ctrl-C, ends it without the refusal's error.
- The same holds for the bare-metal bringup launches that include `base.launch.py`
  (`perception`, `slam`, `navigation`, `autonomy`): a driver exit now ends the whole launch.
  Before, the rest of the stack ran on without a driver.

**A `docker stop` does not reach the driver.** `docker compose stop` and `restart` send SIGTERM to
`ros2 launch`, which is PID 1. Humble's launch, as installed in the image, did not pass SIGTERM on
to its child: in the probe in §6 it logged nothing after the signal, its child's `finally` never
ran, and the container ended only at Docker's SIGKILL 10 s later (exit 137). Launch's
`_on_sigterm`, which would log an error and cancel the run task, did not run either; why was not
established. The driver's teardown, which stops the motors, does not run on that path. This
was so on `main` too, and the fence does not change it.

*2026-10-17: why, and the fix, are in
[`compose-stop-signal-2026-10-17`](../compose-stop-signal-2026-10-17/README.md). Launch catches no
SIGTERM at the OS level, so the kernel drops it at PID 1. With `stop_signal: SIGINT`, `base`'s
launch passes the stop on, and against answering fake controllers the driver's teardown sends its
zero drive to both.*

**The deployed images on this host carry `main`'s driver.**
- Both images are labelled `9f7d20758d5b`, and `git diff 9f7d207 6cd6fd7` is empty for
  `strafer_driver/`, `strafer_bringup/launch/` and `strafer_shared/`. So the fence reaches `base`
  only after `make images` at a commit that contains it.
- The dev overlay's source mount does not carry it either. The image installs `strafer_driver` as
  plain files under `/ws/install`, with no link into `/ws/src`, though the overlay's header says
  otherwise. `entrypoint/image_layout.log` in the second pass lists
  `site-packages/strafer_driver` and `/ws/build/strafer_driver`, with no symlink in either tree.

## 4. Tests

`source/strafer_ros/strafer_driver/test/test_port_fence.py` fakes the serial layer end to end. An
autouse fixture patches `serial.Serial` and `glob.glob`.
- **The bus.** The fake bus maps device paths to fake devices. A RoboClaw answers packet-serial
  commands sent to its own address, with the CRC over address, command and payload, as
  `RoboClawInterface` reads it. The camera answers nothing. One stand-in device answers both
  addresses, and one interrupts its first read, as Ctrl-C does.
- **Udev names.** A udev symlink is a second path to the same fake device, named
  `/dev/roboclaw<N>` from the kernel number as `99-strafer.rules`' `%n` does.
- **The log.** The bus records every open and every write.
- **The diagnose tool.** Its tests load `diagnose_roboclaw.py` from the tree and run its `main()`
  on the same bus.
- **The launch.** Its tests read `driver.launch.py`'s `Node`, and run the image's `LaunchService`
  over a child process that exits 1 or 0.

`strafer_bringup/test/test_bringup.py` gains a check that each of the five bringup launch files
defaults its ports to the udev names.

| test | on `main` | on `3a42c23` | on `aa1a1c7` |
|---|---|---|---|
| `TestDetectCandidates::test_no_udev_names_probes_nothing` | fails: detection returns `{0x81: /dev/ttyACM1}` | pass | pass |
| `TestDetectCandidates::test_scans_udev_names_only` | pass | pass | pass |
| `TestAutoDetectFence::test_no_controllers_refuses_and_opens_nothing` | fails: started; opened `/dev/ttyACM0` ×3, wrote to it | pass | pass |
| `TestAutoDetectFence::test_one_controller_refuses` | fails: started; opened and wrote to `/dev/ttyACM0` | pass | pass |
| `TestAutoDetectFence::test_two_controllers_start_on_detected_ports` | pass | pass | pass |
| `TestAutoDetectFence::test_main_exits_non_zero_on_refusal` | fails: `main()` returned `None` | pass | pass |
| `TestConfiguredPortFence::test_tty_port_refused` | fails: started; wrote to `/dev/ttyACM0` and `/dev/ttyACM1` | pass | pass |
| `TestConfiguredPortFence::test_tty_port_allowed_with_override` | pass | pass | pass |
| `TestConfiguredPortFence::test_one_tty_port_refused` | fails: started; wrote to `/dev/ttyACM0` | pass | pass |
| `TestConfiguredPortFence::test_udev_ports_accepted` | pass | pass | pass |
| `TestNoPidWithoutProbe::test_configured_ports_probed_before_pid` | fails: PID for 0x80 on `/dev/roboclaw2` before any answered probe | pass | pass |
| `TestNoPidWithoutProbe::test_silent_port_gets_no_pid` | fails: started; wrote to `/dev/ttyACM0` | pass | pass |
| `TestNoPidWithoutProbe::test_wrong_address_gets_no_pid` | fails: started on the swapped controllers | pass | pass |
| `TestReconnectProbe::test_reopened_path_must_answer` | fails: left the error state on a bare reopen | pass | pass |
| `TestReconnectProbe::test_interrupted_probe_leaves_the_port_closed` | fails: a bare reopen sends no probe to interrupt | fails: teardown wrote a drive packet to the unanswered port | pass |
| `TestReconnectProbe::test_answering_path_recovers` | pass | pass | pass |
| `TestDistinctPorts::test_auto_detect_one_device_for_both_refused` | fails: started with both controllers on one port | fails: the same | pass |
| `TestDistinctPorts::test_same_configured_port_refused` | fails: started | fails: started | pass |
| `TestDistinctPorts::test_configured_alias_refused` | fails: started | fails: started | pass |
| `TestDiagnoseTool::test_explicit_tty_port_refused` | fails: ran on the tty port (`main()` returned `None`) | fails: the same | pass |
| `TestDiagnoseTool::test_silent_port_stops_after_battery_read` | fails: no `--allow-unverified-ports` flag (argparse exits 2) | fails: the same | pass |
| `TestDiagnoseTool::test_answering_port_is_driven` | pass | pass | pass |
| `test_driver_launch_defaults_are_udev_names` | fails: `/dev/ttyACM0`, `/dev/ttyACM1` | pass | pass |
| `test_launch_exit_code` (×3) | 2 fail: `driver.launch.py` has no `end_launch_on_exit`; the `Shutdown()` control passes | the same | 3 pass |
| `test_driver_exit_ends_the_launch` | fails: no `end_launch_on_exit` | fails: the same | pass |
| `test_bringup.py::TestArguments::test_port_defaults_are_udev_names` (×5) | 5 fail | not run (5 pass at `62bb9ee`, first pass; no launch file or test changed since) | 5 pass |

The passes on `main` are the controls: detection with udev names present, two controllers, the
override with answering ports, udev ports under `auto_detect` false, a reconnect on a path that
still answers, the diagnose tool on an answering RoboClaw, and the bare-`Shutdown` launch. Before
the second pass the diagnose tool had no override flag, so its silent-port test fails on argparse
there; its mutation in §5 shows the test catches the drive after a failed battery read.

| run | tree | result (JUnit) |
|---|---|---|
| driver suite | `main` `6cd6fd7` with `aa1a1c7`'s two test modules | 87 tests, 20 failed |
| bringup, `-k test_port_defaults` | the same | 5 tests, 5 failed |
| driver suite | `3a42c23` with `aa1a1c7`'s two test modules | 87 tests, 9 failed |
| whole ROS suite | `aa1a1c7` | 871 tests: 860 passed, 11 skipped |

The copied test modules are byte-identical to `aa1a1c7`'s (sha256 `0a5b8119…` and `d9975aca…`,
`tests/main/tree.txt` and `tests/prev/tree.txt`).

## 5. Mutation runs

`tests/mutate2.py` applies one mutation at a time to a worktree at `aa1a1c7`, runs the driver
suite, and restores the file. The baseline run passes 87 of 87.

| mutation | failing tests |
|---|---|
| detection scans `/dev/ttyACM*` when no `/dev/roboclaw*` exists | 3 |
| `auto_detect` refuses only when no controller is found | 1 |
| `auto_detect` false accepts any port name | 2 |
| `auto_detect` false opens configured ports without the address probe | 3 |
| `main()` returns `None` on a refusal | 1 |
| `reconnect()` counts a bare reopen as success | 2 |
| an interrupted reconnect probe leaves its port open | 1 |
| the override skips the probe | 1 |
| `auto_detect` starts with both addresses on one device | 1 |
| `auto_detect` false accepts two names for one device | 2 |
| `same_device` compares path strings, not resolved paths | 1 |
| `diagnose_roboclaw.py` accepts any `--front`/`--rear` path | 1 |
| `diagnose_roboclaw.py` goes on after a failed battery read | 1 |
| `driver.launch.py`'s `Node` has no `on_exit` | 1 |
| the launch ends with 0 after a refusal | 2 |
| a non-zero exit during the launch's own shutdown is reported as a refusal | 1 |
| a missing controller keeps its configured port (equivalent: the refusal returns first) | 0 |

## 6. Sealed runs

The node runs mount `aa1a1c7`'s `strafer_driver` package over the image's, in a
`strafer-cpu:humble` container with `--network none` and no `/dev` bind. The container's `/dev`
held only `fd`, `full`, `mqueue`, `null`, `ptmx`, `pts`, `random`, `shm`, `stderr`, `stdin`, `stdout`,
`tty`, `urandom` and `zero`.

The node, abridged from `entrypoint/sealed_run.log` (timestamps and the `ros2run` lines dropped):

```
$ ros2 run strafer_driver roboclaw_node
[INFO] [roboclaw_node]: Auto-detecting RoboClaw ports on /dev/roboclaw*...
[ERROR] [roboclaw_node]: Auto-detect: found 0/2 controllers on /dev/roboclaw* (none). Missing addresses: 0x80, 0x81. Refusing to start; only the address probe was sent.
exit=1
$ ros2 run strafer_driver roboclaw_node --ros-args -p auto_detect:=false -p front_port:=/dev/ttyACM0 -p rear_port:=/dev/ttyACM1
[ERROR] [roboclaw_node]: Not /dev/roboclaw* names (99-strafer.rules): /dev/ttyACM0, /dev/ttyACM1; allow_unverified_ports is false. Refusing to start; no port was opened.
exit=1
$ ros2 run strafer_driver roboclaw_node --ros-args -p auto_detect:=false -p front_port:=/dev/roboclaw1 -p rear_port:=/dev/roboclaw1
[ERROR] [roboclaw_node]: front_port and rear_port name one device (/dev/roboclaw1, /dev/roboclaw1); each controller needs its own port. Refusing to start; no port was opened.
exit=1
```

The image's console script is `sys.exit(load_entry_point(...)())`, so `main()`'s 1 is the exit
code. The mounted `roboclaw_node.py` hashes to `aa1a1c7`'s.

`base`'s compose command, `ros2 launch strafer_bringup base.launch.py`, also ran through the
image's entrypoint. `aa1a1c7`'s driver package, `driver.launch.py` and `base.launch.py` were
mounted. Abridged from `entrypoint/sealed_launch.log` (timestamps and the full command dropped):

```
[roboclaw_node-2] [ERROR] [roboclaw_node]: Auto-detect: found 0/2 controllers on /dev/roboclaw* (none). Missing addresses: 0x80, 0x81. Refusing to start; only the address probe was sent.
[ERROR] [roboclaw_node-2]: process has died [pid 24, exit code 1, cmd '...'].
[ERROR] [launch]: Caught exception in launch (see debug for traceback): roboclaw_node exited 1; ending the launch
[INFO] [robot_state_publisher-1]: sending signal 'SIGINT' to process[robot_state_publisher-1]
container_exit=1
```

The control, `entrypoint/sealed_launch_shutdown_only.log`, is the same launch with
`on_exit=Shutdown()`. It ends the launch too, and the container exits 0.

**The launch probes** (`entrypoint/launch_probe/`). They put `aa1a1c7`'s `end_launch_on_exit`
under the image's launch, with stand-in children:

| case | the container exits |
|---|---|
| a child that exits 1 on its own, as a refusal does | 1 |
| SIGINT to the launch (Ctrl-C, or `docker kill -s SIGINT`); the child exits 1 when signalled | 0, with no refusal error |
| SIGINT to the launch, with a clean child | 0, after the child's `finally` ran |
| `docker stop` (SIGTERM) | 137, after Docker's 10 s SIGKILL; the child's `finally` did not run |

**The diagnose tool** (`entrypoint/sealed_diagnose.log`): `diagnose_roboclaw.py --front
/dev/ttyACM0 --rear /dev/roboclaw1` prints its refusal and exits 2.

## 7. The on-site check

This needs the robot, and it is not done here. Run it on an image built from a commit that carries
the fence.

```bash
cd ~/Sim2RealLab && git rev-parse --short=12 HEAD && git status --short --untracked-files=no
make images                            # stamps that hash (+ -dirty if tracked files changed); the entrypoint prints it
make udev                              # 99-strafer.rules, then plug both RoboClaws in
ls -la /dev/roboclaw* /dev/ttyACM*
for d in /dev/ttyACM*; do echo "$d $(udevadm info -q property -n $d | grep -E '^ID_(VENDOR|MODEL)_ID=' | tr '\n' ' ')"; done
CAM=$(for d in /dev/ttyACM*; do udevadm info -q property -n $d | grep -q '^ID_VENDOR_ID=8086' && echo $d; done)
sudo udevadm test "$(udevadm info -q path -n "$(ls /dev/roboclaw* | head -1)")" 2>&1 | grep -i roboclaw
cd source/strafer_ros/deploy && docker compose up -d base     # camera attached
docker logs strafer_base 2>&1 | grep -E 'revision=|Auto-detect|RoboClawNode starting'
sudo fuser -v "$CAM"                 # the camera's CDC (vendor 8086): no strafer process holds it
```

The camera takes `/dev/ttyACM0` when it enumerates first, as on this host. Plug order decides the
numbers, so `$CAM` finds it by vendor.

- **What to expect.**
  - The log's `revision=` is the 12-character hash that `git rev-parse --short=12 HEAD` printed,
    with no `-dirty` suffix. `make images` adds the suffix when tracked files are modified, so
    build from a clean checkout of the merged commit.
  - The fenced line `Auto-detecting RoboClaw ports on /dev/roboclaw*...` is there.
  - The `Auto-detect:` line names a `/dev/roboclaw<N>` for each of 0x80 and 0x81.
  - The two names resolve to ttyACM devices with vendor `03eb`.
  - `fuser` lists no driver process on the camera's port.
- **The gate.** Do not start the negative half unless the positive half's log shows both the
  merged revision and the fenced `Auto-detecting` line. On an image without the fence, the
  negative half is the hazard itself: the driver falls back to the camera's port.
- **The negative half.** This is [`jetson-untether`](../../tasks/active/reliability/jetson-untether.md)'s
  fence box. A running node does not re-detect: unplug one RoboClaw, then recreate `base` and
  read only the new container's log.

```bash
docker compose up -d --force-recreate base       # from source/strafer_ros/deploy, camera attached
sleep 20; docker inspect -f '{{.RestartCount}} {{.State.Status}}' strafer_base   # want 2 or more; a read a minute later should be higher
docker logs strafer_base 2>&1 | grep -E 'revision=|Auto-detect|Refusing|ending the launch'
sudo fuser -v "$CAM"
```

  Expect `base`'s restart count to climb, more slowly over time: Docker doubles the delay between
  restarts, up to a minute. Each refused attempt is up for a second or two, so one `docker ps`
  sample can read `Up`. Expect the `Refusing to start` line and
  `roboclaw_node exited 1; ending the launch` in `docker logs`, once per attempt, and nothing on
  the camera's port. Plug the RoboClaw back in. A later restart finds both, and `docker logs`
  shows `RoboClawNode starting`.

## Evidence

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `chassis-driver-port-fence-2026-10-14/record-files/` |
| deposit commit | `c12d4f30c0573002539820c6bc34d293270e8fd1` |

The deposit was made in four commits:
- `e4a45b71c40dd1c2061908b461afa99232d57e04` holds the first pass;
- `ac50a32f` corrects three statements in its `DEPOSIT.md` and changes no other file;
- `46317a15` adds the second pass under `pass2/` and rewrites `DEPOSIT.md`, which dates
  the correction and lists the digests of every file. It changes no first-pass file;
- `c12d4f30` corrects two second-pass statements in `DEPOSIT.md` (a module digest, and how many
  added tests fail at `3a42c23`) and changes no other file.

The deposit holds:
- **The first pass, at `62bb9ee`:**
  - `run_evidence.sh`, the script behind every first-pass run but the mutation re-run, and
    `run_stamps.txt`;
  - `tests/main/`: the driver and bringup runs on `main`, with JUnit, console and `tree.txt`;
  - `tests/fix/`: the whole ROS suite, seven JUnit files and the console;
  - `tests/mutate.py`, `tests/mutation_runs.txt` and `tests/mutations/*.xml`;
  - `entrypoint/sealed_run.log`.
- **The second pass, at `aa1a1c7`, under `pass2/`:**
  - `run_evidence2.sh` and `run_stamps.txt`;
  - `tests/main/`, `tests/prev/` and `tests/fix/`, as in §4;
  - `tests/mutate2.py`, `tests/mutation_runs.txt` and `tests/mutations/*.xml`;
  - `entrypoint/`: the image layout and label, the node, `base`'s launch and its control, the
    launch probes, and the diagnose tool.
- `DEPOSIT.md`, with the host, the trees and the sha256 of every file.

Paths in this record resolve in that directory: `tests/...` and `entrypoint/...` are the second
pass's under `pass2/`, unless the text says first pass. Verify the digests with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/chassis-driver-port-fence-2026-10-14/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

Hosts, trees and times are in [`provenance.md`](provenance.md).
