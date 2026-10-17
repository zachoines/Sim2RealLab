# Provenance: the chassis driver's port fence, 2026-10-14

captured_date: first pass 2026-10-14 CDT (2026-10-15 UTC); second pass 2026-10-16 CDT
(2026-10-17 UTC)

## Host

| role | host | system | kernel |
|---|---|---|---|
| every run | strafer-nx | Jetson Orin NX 16 GB, L4T R36.4.3 | 5.15.148-tegra |

All JUnit files carry `hostname="strafer-nx"`.

- **The serial device.** `/dev/ttyACM0` was the D555's CDC interface: char 166:0, vendor `8086`,
  model `0b56`, devpath `.../3610000.usb/usb2/2-2/2-2:1.7/tty/ttyACM0`.
- **No udev rule.** `99-strafer.rules` was not installed, and there was no `/dev/roboclaw*`.
- **No RoboClaw** was connected.

## Image and runner

| | |
|---|---|
| image | `strafer-cpu:humble`, revision label `9f7d20758d5b` (image `sha256:52d829c3…`, `pass2/entrypoint/image_label.txt`) |
| runner | `tools/run_ros_tests.sh`: `docker run --rm --network host`, mounting the tree's `source/strafer_ros` and `source/strafer_shared` only. No `/dev` bind and no device cgroup rule |
| sealed runs | `docker run --rm --network none`, mounting the work commit's `strafer_driver` package over the image's installed copy, and for the launch runs its `driver.launch.py` and `base.launch.py`. The launch probes run named containers (`probe_*`), removed after their logs are read |
| import check | the image holds `9f7d207`'s driver, which equals `main`'s. The fence tests fail on `main`'s tree and pass on the work commit's, in the same image, so the runs import the mounted tree's driver |

## Trees

| pass | runs | tree |
|---|---|---|
| first | `tests/main/` | `main` at `6cd6fd7`, with `62bb9ee`'s `test_port_fence.py` and `test_bringup.py` copied in (identical by sha256, `tests/main/tree.txt`) |
| first | `tests/fix/` | `task/chassis-driver-port-fence` at the work commit `62bb9ee`. The ship commit's brief edit was uncommitted; no test reads it |
| first | `entrypoint/`, `tests/mutations/` | a detached worktree at `62bb9ee`; `tests/mutate.py` restores each file after its run |
| second | `pass2/tests/main/` | `main` at `6cd6fd7`, with `aa1a1c7`'s two test modules copied in (identical by sha256, `pass2/tests/main/tree.txt`) |
| second | `pass2/tests/prev/` | `3a42c23`, the branch head before the second pass, with `aa1a1c7`'s two test modules copied in (`pass2/tests/prev/tree.txt`) |
| second | `pass2/tests/fix/` | the branch at the work commit `aa1a1c7`, clean (`pass2/tests/fix/tree.txt`) |
| second | `pass2/entrypoint/`, `pass2/tests/mutations/` | a detached worktree at `aa1a1c7`; `pass2/tests/mutate2.py` restores each file after its run |

## Timeline (UTC)

| event | time |
|---|---|
| first pass, working runs on the uncommitted tree | 2026-10-15 00:21–00:25 |
| first pass, a working run of the driver suite on the uncommitted tree, before the work commit | 00:42 |
| first pass, driver and bringup suites on `main` | 00:44:03–00:44:26 |
| first pass, whole ROS suite at `62bb9ee` | 00:44:26–00:45:05 |
| first pass, sealed entry-point run | 00:45:05–00:45:09 |
| first pass, mutation runs (the deposited set) | 00:47:20–00:48:51 (JUnit timestamps) |
| deposit commits `e4a45b71` (00:49:36), then `ac50a32f` (01:07:41, `DEPOSIT.md` corrections only) | after the runs |
| second pass, working runs and probes on the uncommitted tree | 2026-10-17 00:15–00:33 |
| second pass, runs at an earlier work commit, `8dde347` (stopped; not deposited) | 00:35–00:39 |
| second pass, `main` and `3a42c23` suites | 00:40:35–00:41:06 |
| second pass, whole ROS suite at `aa1a1c7` | 00:41:06–00:41:41 |
| second pass, sealed runs and launch probes | 00:41:41–00:42:30 |
| second pass, mutation runs | 00:42:30–00:45:36 |
| deposit commits `46317a15`, then `c12d4f30` (`DEPOSIT.md` corrections only) | after the runs |

The first pass's working runs are not deposited. Between the 00:21–00:25 runs and the 00:42 run,
the reconnect probe, the `termios` catch, three tests and wording fixes were added.

The second pass's runs at an earlier work commit, `8dde347`, found one test defect. In
the run on `3a42c23` with `8dde347`'s test modules copied in, the interrupted-probe test's stand-in
device interrupted every read. The teardown's own read then raised `KeyboardInterrupt` out of the
test and ended the pytest session, with no failure recorded. `8dde347` was amended to `aa1a1c7`,
with the device interrupting only its first read. The runs made at `8dde347` were stopped and are
not deposited. No pushed commit holds `8dde347`.
