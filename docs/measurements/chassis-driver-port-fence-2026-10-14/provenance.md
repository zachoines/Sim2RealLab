# Provenance: the chassis driver's port fence, 2026-10-14

captured_date: 2026-10-14 CDT (2026-10-15 UTC)

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
| image | `strafer-cpu:humble`, revision label `9f7d20758d5b` |
| runner | `tools/run_ros_tests.sh`: `docker run --rm --network host`, mounting the tree's `source/strafer_ros` and `source/strafer_shared` only. No `/dev` bind and no device cgroup rule |
| sealed entry-point run | `docker run --rm --network none`, mounting `62bb9ee`'s `strafer_driver` package over the image's installed copy |
| import check | the image holds `9f7d207`'s driver, which equals `main`'s. The fence tests fail on `main`'s tree and pass on `62bb9ee`'s, in the same image, so the runs import the mounted tree's driver |

## Trees

| runs | tree |
|---|---|
| `tests/main/` | `main` at `6cd6fd7`, with `62bb9ee`'s `test_port_fence.py` and `test_bringup.py` copied in (identical by sha256, `tests/main/tree.txt`) |
| `tests/fix/` | `task/chassis-driver-port-fence` at the work commit `62bb9ee`. The ship commit's brief edit was uncommitted; no test reads it |
| `entrypoint/`, `tests/mutations/` | a detached worktree at `62bb9ee`; `tests/mutate.py` restores each file after its run |

## Timeline (UTC)

| event | time |
|---|---|
| driver and bringup suites on `main` | 2026-10-15 00:44:03–00:44:26 |
| whole ROS suite at `62bb9ee` | 00:44:26–00:45:05 |
| sealed entry-point run | 00:45:05–00:45:09 |
| mutation runs (the deposited set) | 00:47:20–00:48:51 (JUnit timestamps) |
| deposit commits `e4a45b71`, then `ac50a32f` (`DEPOSIT.md` corrections only) | after the runs |

Earlier passes ran the same way on the uncommitted working tree: at 00:21–00:25, before review,
and the driver suite at 00:42, after review's changes but before the work commit. Review added the
reconnect probe, the `termios` catch, three tests and wording fixes. Those passes are superseded
and not deposited.
