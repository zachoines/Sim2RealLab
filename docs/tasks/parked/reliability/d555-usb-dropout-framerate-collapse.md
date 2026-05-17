# Detect and recover from D555 USB dropout / framerate collapse

**Type:** task / bug (filed-on-trigger)
**Owner:** Jetson agent (`source/strafer_ros/strafer_perception/`)
**Priority:** P2 — pickup only when the real D555 is connected to the
Jetson and either (a) a multi-hour mission has been completed where
USB dropouts could surface, or (b) the operator observes the first
tegra-xusb dropout in a real-robot bringup.
**Estimate:** M (~1–2 days; framerate watchdog + downstream pause +
optional UVC reset + tests)
**Branch:** task/d555-usb-dropout-framerate-collapse

## Story

As a **mission operator running the real Strafer chassis on Jetson
Orin Nano**, I want **the perception stack to detect D555 dropouts
(USB stall, framerate collapse, tegra-xusb error) and either pause
mission dispatch or signal `perception_unavailable` to the executor**,
so that **the autonomy stack doesn't drive forward on a frozen
camera frame for the seconds-to-minutes a tegra-xusb stall can take
to recover**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)
- [completed/d555-distortion-model-explicit.md](../../completed/d555-distortion-model-explicit.md)
  — last work on the perception camera contract; informs the
  framerate watchdog target.

## Context

The perception stack today has **no framerate watchdog**, no
frame-drop counter, and `initial_reset: false` in the realsense2
launch (operator note in the launch: tegra-xusb enumeration hangs on
re-enumeration). Concretely:

- [`source/strafer_ros/strafer_perception/strafer_perception/timestamp_fixer.py`](../../../../source/strafer_ros/strafer_perception/strafer_perception/timestamp_fixer.py)
  re-stamps every incoming frame and forwards it. Stale frames are
  forwarded as-is.
- [`source/strafer_ros/strafer_perception/strafer_perception/depth_downsampler.py`](../../../../source/strafer_ros/strafer_perception/strafer_perception/depth_downsampler.py)
  increments a `_frame_count` but never publishes a rate, never logs a
  collapse.
- [`source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py`](../../../../source/strafer_ros/strafer_perception/strafer_perception/goal_projection_node.py)
  waits for `_latest_depth` to be populated; if D555 has stopped
  publishing, the service handler blocks indefinitely.
- The executor's `capture_scene_observation`
  ([`ros_client.py:502-582`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py))
  has a freshness check (line 524-527: `age = time.monotonic() -
  color_rx_t`) but it's a snapshot at capture-time, not a running
  watchdog. A staging mission's nav leg can complete while frames
  are frozen, then `capture_scene_observation` at the next
  re-ground succeeds against the stale 30-second-old frame.

The Jetson-Orin-on-D5xx dropout class is well-documented on the NVIDIA
forums and Intel issue tracker:
- [Jetson Orin NX + D435 long-term streaming USB disconnections (NVIDIA Developer Forums, 2024)](https://forums.developer.nvidia.com/t/realsense-d435-and-additional-uvc-device-usb-disconnections-on-jetson-orin-nx-three-scenarios-with-tegra-xusb-errors-emi-and-potential-hub-resets/342718) —
  tegra-xusb errors, EMI scenarios, hub-reset patterns.
- [IntelRealSense/librealsense#13456 — D455 on Jetson Orin NX + Humble + JetPack 6.0](https://github.com/IntelRealSense/librealsense/issues/13456) —
  RSUSB backend workaround.
- Intel recommends compiling librealsense with the `RSUSB` backend
  on Jetson to bypass V4L2 stalls.

The real-robot trigger ("once a stall is observed") is the right
disposition — the sim lane uses the DGX bridge to publish the same
topics, so this is not exercised in `make sim-bridge`. File now,
park, pick up when D555 hardware is actually being driven.

## Approach

### A. Framerate watchdog + downstream pause (recommended)

1. **Framerate monitor node** in `strafer_perception` (or a class
   inside `timestamp_fixer.py`):
   - Subscribes to `/d555/color/image_raw` and
     `/d555/depth/image_rect_raw`.
   - Tracks inter-frame interval; computes rolling p50 / p95 over
     last 30 frames.
   - Publishes `/perception/health` (custom
     `strafer_msgs/PerceptionHealth.msg`) with
     `{color_fps_p50, depth_fps_p50, color_age_s, depth_age_s,
     status: "OK" | "STALE" | "DROPPED"}` at 1 Hz.
   - `STALE` when last-frame age exceeds `stale_age_s` (default 1.0 s
     wall-clock — real-robot lane, sim disabled).
   - `DROPPED` when last-frame age exceeds `dropped_age_s` (default
     5.0 s) — escalates from STALE.

2. **Executor-side gate.** `JetsonRosClient.capture_scene_observation`
   ([`ros_client.py:502-582`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py))
   subscribes to `/perception/health` and:
   - On `STALE`, returns a `SkillResult(..., status="failed",
     error_code="perception_stale", message=<details>)` instead of a
     stale observation.
   - On `DROPPED`, signals `perception_unavailable` to the executor
     which pauses the mission (mirrors the cancel path).

3. **Mid-mission gate.** Reuses the same precheck pattern from
   [`executor-slam-tracking-precheck-mid-mission.md`](../../active/reliability/executor-slam-tracking-precheck-mid-mission.md)
   — bounded wait for perception to recover before failing.

### B. Add an automatic UVC reset on DROPPED (defer)

The realsense2 driver supports `initial_reset:=true` but it's disabled
on Strafer due to tegra-xusb hangs. The
[NVIDIA forum thread](https://forums.developer.nvidia.com/t/realsense-d435-and-additional-uvc-device-usb-disconnections-on-jetson-orin-nx-three-scenarios-with-tegra-xusb-errors-emi-and-potential-hub-resets/342718)
suggests an external `usbreset`/`ioctl(USBDEVFS_RESET)` approach
rather than the driver's own reset. **Park as a follow-up** — first
ship A so the operator at least sees the stall; the right reset
strategy needs hardware-time tuning.

### C. Switch librealsense to RSUSB backend

Intel's official workaround for Jetson + V4L2 stalls. Requires a
rebuild of `librealsense` with `cmake -DFORCE_RSUSB_BACKEND=ON`.
**Park** — the right time to do this is when the framerate watchdog
proves USB stalls are happening *and* the operator wants to upgrade
the backend rather than just reset on stall.

**Recommended:** A first; B and C as follow-up briefs once A's
telemetry confirms the failure mode in the field.

## Acceptance criteria

- [ ] `/perception/health` topic exists, publishes at 1 Hz, schema as
      described above (custom `strafer_msgs/PerceptionHealth.msg`).
- [ ] `STALE` / `DROPPED` thresholds default to wall-clock 1.0 / 5.0 s
      and are env-overridable
      (`STRAFER_PERCEPTION_STALE_AGE_S`,
      `STRAFER_PERCEPTION_DROPPED_AGE_S`).
- [ ] `JetsonRosClient.capture_scene_observation` consults
      `/perception/health` and returns
      `error_code="perception_stale"` or `"perception_unavailable"`
      instead of forwarding a stale observation.
- [ ] On the sim lane (where the DGX bridge publishes frames at
      whatever cadence Isaac Sim's RTF supports), thresholds must be
      auto-relaxed via the existing `HARDWARE_PRESENT=false` env
      surface (set in
      [`bringup_sim_in_the_loop.launch.py:151`](../../../../source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py)),
      or skip the watchdog entirely on sim. Document the decision.
- [ ] Unit tests cover (a) healthy stream → no impact, (b) stale stream
      → `perception_stale` error, (c) dropped stream → mission pause.
- [ ] Integration smoke (real-robot): physically unplug the D555 USB
      mid-mission; executor pauses with `perception_unavailable`;
      replug the camera; executor resumes (or fails cleanly after the
      bounded wait timeout).
- [ ] Documentation update: `docs/INTEGRATION_REAL_ROBOT.md` (or
      equivalent) gets a "what to do when the framerate watchdog
      fires" runbook entry.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit.

## Investigation pointers

- [`source/strafer_ros/strafer_perception/launch/perception.launch.py`](../../../../source/strafer_ros/strafer_perception/launch/perception.launch.py) —
  `initial_reset: false` and the operator-note explaining the
  tegra-xusb enumeration hangs.
- [`source/strafer_ros/strafer_perception/strafer_perception/timestamp_fixer.py`](../../../../source/strafer_ros/strafer_perception/strafer_perception/timestamp_fixer.py) —
  natural home for the framerate monitor; already subscribes to
  every incoming camera topic.
- [`source/strafer_autonomy/strafer_autonomy/clients/ros_client.py:502-582`](../../../../source/strafer_autonomy/strafer_autonomy/clients/ros_client.py) —
  `capture_scene_observation` freshness logic. Already has the
  pattern; needs the health-topic consumer.
- Peer reference: [Intel + NVIDIA Jetson Orin RealSense issues](https://github.com/IntelRealSense/librealsense/issues/13456),
  [NVIDIA forum tegra-xusb scenarios](https://forums.developer.nvidia.com/t/realsense-d435-and-additional-uvc-device-usb-disconnections-on-jetson-orin-nx-three-scenarios-with-tegra-xusb-errors-emi-and-potential-hub-resets/342718).
- librealsense RSUSB backend rebuild reference:
  [librealsense Release Notes wiki](https://github.com/IntelRealSense/librealsense/wiki/Release-Notes).

## Out of scope

- **Auto-reset on DROPPED.** Filed separately if/when A's telemetry
  shows it's needed (option B above).
- **RSUSB backend rebuild.** Filed separately (option C above).
- **Sim-side D555 dropout simulation.** The bridge doesn't model USB
  drops; sim-side validation for this brief is via unit tests against
  mocked subscriptions. Real-robot smoke is the integration test.
- **D555 firmware upgrade.** Lives in the bringup runbook, not in a
  reliability brief.
- **Recovery from a *different* RealSense (D435, D455).** D555 is the
  shipped target; if a future swap happens, file a follow-up.
