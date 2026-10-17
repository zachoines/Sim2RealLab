# Fence the chassis driver to ports that answer as a RoboClaw

**Status:** Shipped 2026-10-14 in `62bb9ee` and 2026-10-16 in `aa1a1c7` (Jetson).
**PR:** https://github.com/zachoines/Sim2RealLab/pull/243

**Type:** bug (the RoboClaw driver writes motor packets to the D555's serial interface)
**Owner:** Jetson (`strafer_driver`, `strafer_bringup`, `deploy/`)
**Priority:** P1 for goal-b. A chassis bringup with the camera attached is the real lane's first
step. With a RoboClaw missing, the driver writes PID and drive packets to that controller's
configured port, and the front's is the camera's whenever the camera enumerates first.
**Estimate:** S
**Branch:** `task/chassis-driver-port-fence`

## Story

As **whoever starts `base` on the robot**, I want **the RoboClaw driver to write only to a port
that has answered as a RoboClaw**, so that **a missing controller stops the driver instead of
sending motor packets to the camera's serial interface.**

## Context bundle

- [context/repo-topology.md](../context/repo-topology.md)
- [context/ownership-boundaries.md](../context/ownership-boundaries.md)
- [context/conventions.md](../context/conventions.md)
- [context/branching-and-prs.md](../context/branching-and-prs.md)
- [`jetson-untether`](../active/reliability/jetson-untether.md), "Camera, RoboClaw and boot hazards" and the fence box:
  the hazard as first read, and the half of the fence that needs the robot.

## Context

**The defect, at `6cd6fd7`:**
- **Detection falls back to every tty.** `detect_roboclaws` scans `/dev/roboclaw*` and, finding
  none, scans `/dev/ttyACM*` (`strafer_driver/roboclaw_interface.py`:414-417). It sends each
  candidate a `read_main_battery` probe for both addresses (:367-387, :422-429).
- **The node starts in every case** (`strafer_driver/roboclaw_node.py`:106-140): with both
  controllers found; with one found and the configured default for the other (:120-135); and with
  none found, "Falling back to configured ports" (:136-140).
- **It writes before it knows what is there.** It opens each configured port and writes the
  velocity PID to it (:150-165, :225-238) with no probe, then a drive command on every 50 Hz tick
  (:210-211, :309-317). After ten failures it enters an error state and reopens both ports every
  2 s, and a bare open counts as success (:251-265; `roboclaw_interface.py`:143-150).
- **A configured default can be the camera's port.** `driver.launch.py`:22-31 defaults to
  `/dev/ttyACM0` and `/dev/ttyACM1`, and so does every bringup launch file that includes the
  driver (`base`, `perception`, `slam`, `navigation`, `autonomy`). They are passed after the
  params file, so they override `driver_params.yaml`'s `/dev/roboclaw0` and `/dev/roboclaw1`.
- **On the robot the D555 enumerates a CDC serial interface at `/dev/ttyACM0`**
  ([`real-d555-hardware-readback-2026-09-25`](../../measurements/real-d555-hardware-readback-2026-09-25/README.md):57).

`99-strafer.rules`:20 already names every Basicmicro RoboClaw (`03eb:2404`) `/dev/roboclaw<N>`, so
the fence is to trust only those names, and to write nothing to a port before it answers the
address probe.

**Compose.** The `base` service binds `/dev` wholesale and allows every ttyACM character device
(`device_cgroup_rules: [ "c 166:* rmw" ]`, `deploy/docker-compose.yml`:74-78). It has no
`devices:` list and is not `privileged`. The container sees every `/dev/ttyACM*` node, the
camera's among them, and a missing `/dev/roboclaw*` symlink does not stop it starting. The node's
own refusal has to be the fence.

## Acceptance criteria

- [x] `detect_roboclaws` scans `/dev/roboclaw*` only. The `/dev/ttyACM*` fallback is gone.
      *2026-10-14: `roboclaw_interface.py`:378, :436 at `62bb9ee`.*
- [x] With `auto_detect` true, fewer than two controllers found: the node logs the count and the
      missing addresses and exits non-zero before opening any port to drive it. Detection's
      address probe of `/dev/roboclaw*` is the only traffic, and nothing falls back to a
      configured default.
      *2026-10-14, at `62bb9ee`: `roboclaw_node.py`:125-143 and `_refuse` (:260-266) raise
      `PortFenceError`; `main()` returns 1 (:554-561), and the console script exits with it. On
      `main`, in the fake serial layer, the node started and wrote to the camera's port.*
- [x] With `auto_detect` false, a configured port that is not a `/dev/roboclaw*` name is refused
      the same way, unless a new parameter `allow_unverified_ports` (default false, documented as a
      bench-only override) is set.
      *2026-10-14, at `62bb9ee`: :155-160; documented in `driver_params.yaml`. The override
      relaxes only the name check, never the probe.*
- [x] PID constants are written only to a port that answered the address probe; the startup
      order (open, then set PID) cannot reach a port that was never probed.
      *2026-10-14, at `62bb9ee`: under `auto_detect` false each configured port is probed first
      (:166-175); the opens and PID writes (:185-200) follow the refusals.*
- [x] A reconnect from the error state counts only when the reopened port answers the probe, so
      a path that now leads to another device is not driven.
      *2026-10-14: added before the work commit. `reconnect()` re-probes and closes the port
      again on no answer (`roboclaw_interface.py`:145-159 at `62bb9ee`); `probe_address` also
      treats a `termios` error as no answer.*
      *2026-10-16: an interrupted probe (Ctrl-C) now closes the port too, so the node's teardown
      writes nothing to it (`roboclaw_interface.py`:145-164 at `aa1a1c7`).*
- [x] The launch defaults are `/dev/roboclaw0` and `/dev/roboclaw1`, in `driver.launch.py` and in
      every bringup launch file that passes them on.
      *2026-10-14: `driver.launch.py` and `base`, `perception`, `slam`, `navigation`, `autonomy`;
      pinned by `test_bringup.py`'s `test_port_defaults_are_udev_names`.*
- [x] Tests in `strafer_driver/test/` with a fake serial layer:
  - (a) no controllers: the node refuses and opens nothing;
  - (b) one controller: it refuses;
  - (c) two: it starts on the detected ports;
  - (d) `auto_detect` false with a `/dev/ttyACM*` port: it refuses unless the override is set;
  - (e) no PID write without an answered probe.

  (a), (b), (d)'s refusal and (e) fail on `main`, and that output is deposited.

  *2026-10-14: 16 tests in `test_port_fence.py`, among them a mixed udev-and-tty pair and a
  reconnect whose path no longer answers. With them on `main`, 11 of 76 driver tests and the 5
  bringup default tests fail. Mutation runs: seven of eight mutations fail at least one test, and
  the eighth is equivalent (the refusal returns first). Record and deposit below.*

  *2026-10-14: two criteria were reworded while the work ran. The `auto_detect` one says "before
  opening any port to drive it", since detection's probe opens `/dev/roboclaw*`. The reconnect
  one was added when a reading of the error state found that a reconnect counted a bare reopen.
  The out-of-scope items for the reconnect and the shared constants' comments went, because the
  work did both.*

  *2026-10-16: 27 tests in `test_port_fence.py` at `aa1a1c7`. With them on `main`, 20 of 87
  driver tests fail, and the 5 bringup default tests fail. Mutation runs: sixteen of seventeen
  mutations fail at least one test; the seventeenth is the equivalent one.*
- [x] **One device is never both controllers** *(added 2026-10-16)*. The node refuses when
      auto-detect finds both addresses on one device, or when `front_port` and `rear_port`
      resolve to one device, with fake-serial tests.
      *2026-10-16, at `aa1a1c7`: `roboclaw_node.py`:146-153 and :163-168, `same_device`
      (`roboclaw_interface.py`:393-395); `TestDistinctPorts`, three tests.*
- [x] **A refusal ends `base`** *(added 2026-10-16)*. The driver's launch ends when the node exits,
      non-zero after a refusal, so compose's `restart: unless-stopped` re-runs detection.
      *2026-10-16: `driver.launch.py`:11-20, :60-61. Humble's launch returns non-zero only for an
      exception, so `on_exit` is a callable that raises after a non-zero exit outside the
      launch's own shutdown. Sealed, `base`'s compose command exits 1 after the refusal, and the
      bare-`Shutdown` control exits 0; `test_launch_exit_code` pins both under the image's
      launch.*
- [x] **`diagnose_roboclaw.py` is fenced too** *(added 2026-10-16)*. Its `--front`/`--rear` paths
      must be `/dev/roboclaw*` names unless `--allow-unverified-ports` is given, and it stops
      after a failed battery read instead of going on to the PID read and the duty-cycle drive.
      *2026-10-16: `diagnose_roboclaw.py`:169-182, :117-124; `TestDiagnoseTool`, three tests.*
- [x] A record states the code reads, the test runs, and how `base` maps serial devices, with no
      hardware claims. It gives the maintainer's on-site check as commands.
      *2026-10-14: [`chassis-driver-port-fence-2026-10-14`](../../measurements/chassis-driver-port-fence-2026-10-14/README.md), deposit Sim2RealLab-Artifacts `ac50a32f`.*
      *2026-10-16: the record's second pass, deposit `c12d4f30`. The on-site check now
      gates its negative half on the positive half's log, and reads a refusal as a climbing
      restart count.*
- [x] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
      *2026-10-14: the compose header and cgroup comment, the udev rule's comment, both READMEs,
      `driver_params.yaml`, the shared port constants' comments, the bench tools' messages,
      `jetson-untether` and the parked low-battery brief's line citations.*
      *2026-10-16: the compose header and cgroup comment again, the README's hazard and
      diagnose lines, the `base` and `perception` launch docstrings, `jetson-untether`'s compose
      citations, fence range and fence note, and the parked low-battery brief's line citations.*
- [x] No regression: `make test-ros` passes.
      *2026-10-14: 849 passed, 11 skipped at `62bb9ee`. 2026-10-16: 860 passed, 11 skipped at
      `aa1a1c7`.*

## Investigation pointers

- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py` (`probe_address`,
  `detect_roboclaws`) and `roboclaw_node.py` (`__init__`, `main`).
- `source/strafer_ros/strafer_driver/test/test_roboclaw_interface.py`'s `TestAutoDetect`, which
  patches `probe_address` and `glob.glob` the same way.
- `source/strafer_ros/tune_pid.py` and `diagnose_roboclaw.py` call `detect_roboclaws` too.

## Out of scope

- Installing `99-strafer.rules` on the robot, and the camera-attached check with `fuser` and the
  node's log: [`jetson-untether`](../active/reliability/jetson-untether.md)'s fence box.
- Re-writing the PID after a reconnect.
- `base`'s stop path. `docker compose stop` sends SIGTERM, which Humble's launch does not pass on
  to its children, so the driver is killed after 10 s without its teardown (record §3).
