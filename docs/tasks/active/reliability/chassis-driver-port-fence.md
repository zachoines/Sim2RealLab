# Fence the chassis driver to ports that answer as a RoboClaw

**Type:** bug (the RoboClaw driver writes motor packets to the D555's serial interface)
**Owner:** Jetson (`strafer_driver`, `strafer_bringup`, `deploy/`)
**Priority:** P1 for goal-b. A chassis bringup with the camera attached is the real lane's first
step, and with either RoboClaw missing the driver writes PID and drive packets to the camera.
**Estimate:** S
**Branch:** `task/chassis-driver-port-fence`

## Story

As **whoever starts `base` on the robot**, I want **the RoboClaw driver to write only to a port
that has answered as a RoboClaw**, so that **a missing controller stops the driver instead of
sending motor packets to the camera's serial interface.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`jetson-untether`](jetson-untether.md), "Camera, RoboClaw and boot hazards" and the fence box:
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
- **The configured defaults are the camera's port.** `driver.launch.py`:22-31 defaults to
  `/dev/ttyACM0` and `/dev/ttyACM1`, and so does every bringup launch file that includes the
  driver (`base`, `perception`, `slam`, `navigation`, `autonomy`). They are passed after the
  params file, so they override `driver_params.yaml`'s `/dev/roboclaw0` and `/dev/roboclaw1`.
- **On the robot the D555 enumerates a CDC serial interface at `/dev/ttyACM0`**
  ([`real-d555-hardware-readback-2026-09-25`](../../../measurements/real-d555-hardware-readback-2026-09-25/README.md):57).

`99-strafer.rules`:20 already names every Basicmicro RoboClaw (`03eb:2404`) `/dev/roboclaw<N>`, so
the fence is to trust only those names, and to write nothing to a port before it answers the
address probe.

**Compose.** The `base` service binds `/dev` wholesale and allows every ttyACM character device
(`device_cgroup_rules: [ "c 166:* rmw" ]`, `deploy/docker-compose.yml`:74-78). It has no
`devices:` list and is not `privileged`. The container sees every `/dev/ttyACM*` node, the
camera's among them, and a missing `/dev/roboclaw*` symlink does not stop it starting. The node's
own refusal has to be the fence.

## Acceptance criteria

- [ ] `detect_roboclaws` scans `/dev/roboclaw*` only. The `/dev/ttyACM*` fallback is gone.
- [ ] With `auto_detect` true, fewer than two controllers found: the node logs the count and the
      missing addresses and exits non-zero before opening any port. Nothing falls back to a
      configured default.
- [ ] With `auto_detect` false, a configured port that is not a `/dev/roboclaw*` name is refused
      the same way, unless a new parameter `allow_unverified_ports` (default false, documented as a
      bench-only override) is set.
- [ ] PID constants are written only to a port that answered the address probe; the startup
      order (open, then set PID) cannot reach a port that was never probed.
- [ ] The launch defaults are `/dev/roboclaw0` and `/dev/roboclaw1`, in `driver.launch.py` and in
      every bringup launch file that passes them on.
- [ ] Tests in `strafer_driver/test/` with a fake serial layer:
  - (a) no controllers: the node refuses and opens nothing;
  - (b) one controller: it refuses;
  - (c) two: it starts on the detected ports;
  - (d) `auto_detect` false with a `/dev/ttyACM*` port: it refuses unless the override is set;
  - (e) no PID write without an answered probe.

  (a), (b), (d)'s refusal and (e) fail on `main`, and that output is deposited.
- [ ] A record states the code reads, the test runs, and how `base` maps serial devices, with no
      hardware claims. It gives the maintainer's on-site check as commands.
- [ ] If your work invalidates a fact in any referenced context module, package README,
      top-level `Readme.md`, or guide under `docs/`, update those in the same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
- [ ] No regression: `make test-ros` passes.

## Investigation pointers

- `source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py` (`probe_address`,
  `detect_roboclaws`) and `roboclaw_node.py` (`__init__`, `main`).
- `source/strafer_ros/strafer_driver/test/test_roboclaw_interface.py`'s `TestAutoDetect`, which
  patches `probe_address` and `glob.glob` the same way.
- `source/strafer_ros/tune_pid.py` and `diagnose_roboclaw.py` call `detect_roboclaws` too.

## Out of scope

- Installing `99-strafer.rules` on the robot, and the camera-attached check with `fuser` and the
  node's log: [`jetson-untether`](jetson-untether.md)'s fence box.
- `reconnect()` counting a bare open as success. With the fence the reopened path is one that
  answered the probe at startup.
- Re-writing the PID after a reconnect.
- Shutting `base`'s launch down when the driver exits.
- `strafer_shared.constants`' port comments, which the shared boundary makes append-only.
