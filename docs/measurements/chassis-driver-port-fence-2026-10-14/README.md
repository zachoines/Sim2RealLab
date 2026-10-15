# The chassis driver's port fence, 2026-10-14

With either RoboClaw missing, the driver at `6cd6fd7` writes velocity-PID and drive packets to
`/dev/ttyACM0`. On the robot that is the D555's CDC serial interface. This record reads the
driver before and after the fence in
[`chassis-driver-port-fence`](../../tasks/completed/chassis-driver-port-fence.md), runs its tests
on both trees, and states how the `base` service maps serial devices. It makes **no hardware
claim**: no RoboClaw was connected, `base` was not started, and no run had the host's
`/dev/ttyACM0` in its container. §7 gives the on-site check as commands.

**Result.**
- **On `main` the driver opens the camera's port.** With a fake serial layer standing in for the
  robot, the camera alone at `/dev/ttyACM0`:
  - `main`'s node starts;
  - it opens `/dev/ttyACM0` three times: two detection probes, then the configured front port;
  - it writes to that port.
- **On the work commit `62bb9ee` the node refuses.** In the same setup it opens nothing and exits
  1. With one controller present, the only traffic is detection's address probe of
  `/dev/roboclaw*`.
- **Tests.** The work commit's test modules, run on `main`, fail 11 of 76 driver tests and all 5
  new bringup tests. On `62bb9ee` the whole ROS suite passes: 849 passed and 11 skipped of 860.
- **Mutations.** Eight mutations each put back one piece of the defect, and each fails at least
  one test. A ninth is equivalent and fails none, as expected.
- **The real entry point.** `ros2 run strafer_driver roboclaw_node` from `62bb9ee`, in a container
  with no `/dev` and no network, logs the refusal and exits 1 (§6).
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

The launch arguments are passed after the params file, so they override `driver_params.yaml`'s
`/dev/roboclaw0` and `/dev/roboclaw1`.

**On this host,** at the time of the runs:
- `/dev/ttyACM0` was the D555's CDC interface: char 166:0, `ID_VENDOR_ID=8086`,
  `ID_MODEL_ID=0b56`, under `usb2/2-2`.
- `99-strafer.rules` was not installed; `/etc/udev/rules.d` held only NVIDIA rules.
- There was no `/dev/roboclaw*`.

## 2. The fence at `62bb9ee`

| path | behaviour | where |
|---|---|---|
| detection | scans `/dev/roboclaw*` only | `roboclaw_interface.py`:378, :436 |
| `auto_detect` true, fewer than two found | logs the count and the missing addresses, raises `PortFenceError` before any `RoboClawInterface` is opened for driving | `roboclaw_node.py`:125-142, :260-266 |
| `auto_detect` false, a port that is not a `/dev/roboclaw*` name | refused before anything is opened, unless `allow_unverified_ports` | `roboclaw_node.py`:155-160 |
| `auto_detect` false, a port that does not answer the probe | refused; only the probe was written | `roboclaw_node.py`:166-174 |
| startup PID write | reached only after the refusals above, so only on ports that answered | `roboclaw_node.py`:185-200 |
| error-state reconnect | the reopened port must answer the probe, or it is closed again and the node stays in the error state | `roboclaw_interface.py`:145-159; `roboclaw_node.py`:294-308 |
| a `termios` error during a probe | treated as no answer | `roboclaw_interface.py`:157, :407 |
| `main()` | returns 1 on a refusal, and the console script exits with it | `roboclaw_node.py`:554-561 |
| launch defaults | `/dev/roboclaw0` and `/dev/roboclaw1` in `driver.launch.py` and the five bringup launch files | `driver.launch.py`:24, :29 |

`allow_unverified_ports` defaults to false (`driver_params.yaml`). It relaxes only the name check
under `auto_detect` false, never the probe, and detection ignores it.

`tune_pid.py` and `diagnose_roboclaw.py` share `detect_roboclaws`, so they now find controllers
only at `/dev/roboclaw*` too. Their messages say so and point at `make udev`.
`diagnose_roboclaw.py` still takes `--front` and `--rear` paths.

## 3. How `base` maps serial devices

From `deploy/docker-compose.yml` at `62bb9ee` (:64-83, with the `x-cpu-svc` anchor at :51-58):
- `group_add: [ dialout ]` (:76);
- `device_cgroup_rules: [ "c 166:* rmw" ]` (:79), which admits every ttyACM character device.
  The D555's CDC is major 166 too;
- the volume `/dev:/dev` (:82): the whole host `/dev`, so udev's symlinks appear in the container
  as the host creates them;
- no `devices:` list and no `privileged`, here or in any tracked override under `deploy/`;
- `restart: unless-stopped` from the anchor (:54);
- `command: ros2 launch strafer_bringup base.launch.py` (:83), which includes
  `description.launch.py` and `driver.launch.py` (`base.launch.py`:42, :49).

A missing `/dev/roboclaw*` symlink therefore does not stop the container. The node's refusal is
the fence. When the driver exits, `robot_state_publisher` keeps the launch, and the container, up.
The driver's `Node` has no `on_exit` (`driver.launch.py`:32-48), and `ros2 launch` keeps running
while any process does. So `docker ps` shows `base` up with no driver, and the log is where the
refusal shows. Once both controllers are connected, `base` needs a restart.

The deployed images on this host carry `main`'s driver. Both are labelled `9f7d20758d5b`, and
`git diff 9f7d207 6cd6fd7` is empty for `strafer_driver/`, `strafer_bringup/launch/` and
`strafer_shared/`. The fence reaches `base` only after `make images` at a commit that contains it.

## 4. Tests

`source/strafer_ros/strafer_driver/test/test_port_fence.py` fakes the serial layer end to end. An
autouse fixture patches `serial.Serial` and `glob.glob`.
- **The bus.** The fake bus maps device paths to fake devices. A RoboClaw answers packet-serial
  commands sent to its own address, with the CRC over address, command and payload, as
  `RoboClawInterface` reads it. The camera answers nothing.
- **Udev names.** A udev symlink is a second path to the same fake device, named
  `/dev/roboclaw<N>` from the kernel number as `99-strafer.rules`' `%n` does.
- **The log.** The bus records every open and every write.

`strafer_bringup/test/test_bringup.py` gains a check that each of the five bringup launch files
defaults its ports to the udev names.

| test | on `main` | on `62bb9ee` |
|---|---|---|
| `TestDetectCandidates::test_no_udev_names_probes_nothing` | fails: detection returns `{0x81: /dev/ttyACM1}` | pass |
| `TestDetectCandidates::test_scans_udev_names_only` | pass | pass |
| `TestAutoDetectFence::test_no_controllers_refuses_and_opens_nothing` | fails: started; opened `/dev/ttyACM0` ×3, wrote to it | pass |
| `TestAutoDetectFence::test_one_controller_refuses` | fails: started; opened and wrote to `/dev/ttyACM0` | pass |
| `TestAutoDetectFence::test_two_controllers_start_on_detected_ports` | pass | pass |
| `TestAutoDetectFence::test_main_exits_non_zero_on_refusal` | fails: `main()` returned `None` | pass |
| `TestConfiguredPortFence::test_tty_port_refused` | fails: started; wrote to `/dev/ttyACM0` and `/dev/ttyACM1` | pass |
| `TestConfiguredPortFence::test_tty_port_allowed_with_override` | pass | pass |
| `TestConfiguredPortFence::test_one_tty_port_refused` | fails: started; wrote to `/dev/ttyACM0` | pass |
| `TestConfiguredPortFence::test_udev_ports_accepted` | pass | pass |
| `TestNoPidWithoutProbe::test_configured_ports_probed_before_pid` | fails: PID for 0x80 on `/dev/roboclaw2` before any answered probe | pass |
| `TestNoPidWithoutProbe::test_silent_port_gets_no_pid` | fails: started; wrote to `/dev/ttyACM0` | pass |
| `TestNoPidWithoutProbe::test_wrong_address_gets_no_pid` | fails: started on the swapped controllers | pass |
| `TestReconnectProbe::test_reopened_path_must_answer` | fails: left the error state on a bare reopen | pass |
| `TestReconnectProbe::test_answering_path_recovers` | pass | pass |
| `test_driver_launch_defaults_are_udev_names` | fails: `/dev/ttyACM0`, `/dev/ttyACM1` | pass |
| `test_bringup.py::TestArguments::test_port_defaults_are_udev_names` (×5) | 5 fail | 5 pass |

The passes on `main` are the controls: two controllers, the override with answering ports, and
udev ports under `auto_detect` false. `main` starts in each of those cases too.

| run | tree | result |
|---|---|---|
| driver suite | `main` `6cd6fd7` with `62bb9ee`'s two test modules | 76 tests, 11 failed |
| bringup, `-k test_port_defaults` | the same | 5 tests, 5 failed |
| whole ROS suite | `62bb9ee` | 860 tests: 849 passed, 11 skipped |

The test modules on `main` are byte-identical to `62bb9ee`'s (sha256 `6c6741bc…` and
`d9975aca…`, `tests/main/tree.txt`).

## 5. Mutation runs

`tests/mutate.py` applies one mutation at a time to a worktree at `62bb9ee`, runs the driver
suite, and restores the file.

| mutation | failing tests |
|---|---|
| detection scans `/dev/ttyACM*` when no `/dev/roboclaw*` exists | 3 |
| `auto_detect` refuses only when no controller is found | 1 |
| `auto_detect` false accepts any port name | 2 |
| `auto_detect` false opens configured ports without the probe | 3 |
| `main()` returns `None` on a refusal | 1 |
| `reconnect()` counts a bare reopen as success | 1 |
| the override skips the probe | 1 |
| a missing controller keeps its configured port (equivalent: the refusal returns first) | 0 |

The baseline run passes 76 of 76.

## 6. The entry point, sealed

`62bb9ee`'s `strafer_driver` package was mounted over the image's in a `strafer-cpu:humble`
container with `--network none` and no `/dev` bind. The container's `/dev` held only `fd`, `full`,
`mqueue`, `null`, `ptmx`, `pts`, `random`, `shm`, `stderr`, `stdin`, `stdout`, `tty`, `urandom` and
`zero`.

```
$ ros2 run strafer_driver roboclaw_node
[INFO] [roboclaw_node]: Auto-detecting RoboClaw ports on /dev/roboclaw*...
[ERROR] [roboclaw_node]: Auto-detect: found 0/2 controllers on /dev/roboclaw* (none). Missing addresses: 0x80, 0x81. Refusing to start; only the address probe was sent.
exit=1
$ ros2 run strafer_driver roboclaw_node --ros-args -p auto_detect:=false -p front_port:=/dev/ttyACM0 -p rear_port:=/dev/ttyACM1
[ERROR] [roboclaw_node]: Not /dev/roboclaw* names (99-strafer.rules): /dev/ttyACM0, /dev/ttyACM1; allow_unverified_ports is false. Refusing to start; no port was opened.
exit=1
```

The image's console script is `sys.exit(load_entry_point(...)())`, so `main()`'s 1 is the exit
code. The mounted `roboclaw_node.py` hashes to `62bb9ee`'s.

## 7. The on-site check

This needs the robot, and it is not done here. Run it on an image built from a commit that carries
the fence.

```bash
cd ~/Sim2RealLab && make images        # the entrypoint prints the revision on every start
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
  - The `Auto-detect:` line names a `/dev/roboclaw<N>` for each of 0x80 and 0x81.
  - The two names resolve to ttyACM devices with vendor `03eb`.
  - `fuser` lists no driver process on the camera's port.
- **The negative half.** This is [`jetson-untether`](../../tasks/active/reliability/jetson-untether.md)'s
  fence box. Repeat the last three commands with one RoboClaw unplugged and the camera attached.
  Expect the `Refusing to start` line and nothing on the camera's port.

## Evidence

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `chassis-driver-port-fence-2026-10-14/record-files/` |
| deposit commit | `e4a45b71c40dd1c2061908b461afa99232d57e04` |

The deposit holds:
- `run_evidence.sh`, the script behind every run but the mutation re-run, and `run_stamps.txt`;
- `tests/main/`: the driver and bringup runs on `main`, with JUnit, console and `tree.txt`;
- `tests/fix/`: the whole ROS suite at `62bb9ee`, seven JUnit files and the console;
- `tests/mutate.py`, `tests/mutation_runs.txt` and `tests/mutations/*.xml`;
- `entrypoint/sealed_run.log`;
- `DEPOSIT.md`, with the host, the trees and the sha256 of every file.

Paths in this record (`tests/...`, `entrypoint/...`) resolve in that directory. Verify the digests
with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cd Sim2RealLab-Artifacts/chassis-driver-port-fence-2026-10-14/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

Hosts, trees and times are in [`provenance.md`](provenance.md).
