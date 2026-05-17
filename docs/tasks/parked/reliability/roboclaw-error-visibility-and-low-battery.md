# Expose RoboClaw CRC error rate + battery voltage + low-battery mode

**Type:** task / enhancement (filed-on-trigger)
**Owner:** Jetson agent (`source/strafer_ros/strafer_driver/`)
**Priority:** P2 — pickup gated on real-robot bringup. On sim the
RoboClaw is not in the stack at all; this brief is exclusively a
hardware-lane reliability investment.
**Estimate:** M (~1–2 days; per-call CRC counter + diagnostics publisher
+ battery topic + low-battery degraded mode + tests + smoke on a
running chassis)
**Branch:** task/roboclaw-error-visibility-and-low-battery

## Story

As a **real-robot operator running the Strafer chassis on a 12 V LiPo
through two RoboClaws on USB**, I want **CRC-error rate, consecutive-
failure count, and main battery voltage published as diagnostics, and
a low-battery degraded-mode behavior (limit speed, then refuse new
missions) wired up**, so that **a flaky USB cable / shielded motor
EMI / dying battery shows up as a leading indicator before it surfaces
as silent communication failures or a chassis that suddenly halts
mid-mission**.

## Context bundle

Read these before starting:
- [context/repo-topology.md](../../context/repo-topology.md)
- [context/ownership-boundaries.md](../../context/ownership-boundaries.md)

## Context

`roboclaw_interface.py` and `roboclaw_node.py` today:

- **`max_retries = 1` default** ([`roboclaw_interface.py:208-219`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py)
  `_retry`), with a 5 ms inter-retry pause. The peer reference driver
  [wimblerobotics/ros2_roboclaw_driver](https://github.com/wimblerobotics/ros2_roboclaw_driver)
  defaults to `max_retries=3` and recommends close-and-reopen on
  repeated I/O failure.
- **`_consecutive_failures` counter** ([`roboclaw_node.py:424-446`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py)
  `_handle_serial_failure`). Threshold 10 → error state; reconnect
  every 2 s. Not published in diagnostics.
- **CRC errors counted nowhere.** `RoboClawChecksumError` exceptions
  in `_retry()` are caught but not tallied. A high CRC rate (bad
  cable, motor noise on the line) is invisible until it crosses the
  consecutive-failure threshold.
- **Battery voltage readable but never read.** `read_main_battery()`
  exists at [`roboclaw_interface.py:349-353`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py)
  (uint16 / 10.0 = volts). The node never calls it. Diagnostics
  publish ([`roboclaw_node.py:448-478`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py))
  do not include battery voltage.
- **No low-battery degraded mode.** The chassis will execute commands
  at full speed until the LiPo's BMS cuts out or the RoboClaw browns
  out, neither of which produces useful telemetry.

Symptoms an operator will see *without* this brief:
- "The robot keeps stopping randomly mid-mission." → consecutive
  failures crossing threshold from low-rate CRC errors.
- "The chassis just dies." → BMS cutoff under load with no voltage
  warning.
- "The encoders are jittery." → marginal communication noise that
  passes single-CRC-retry but corrupts subsequent reads.

Peer references on production RoboClaw integration:
- [wimblerobotics/ros2_roboclaw_driver](https://github.com/wimblerobotics/ros2_roboclaw_driver) —
  ROS2 driver with `max_retries=3` default, status publisher,
  close-and-reopen on I/O failure.
- [Pololu forum thread on CRC failure modes](https://forum.pololu.com/t/roboclaw-crc-failure/3011)
  documents the typical causes (cable quality, motor brush noise, USB
  hub conflicts).
- [`docs/tasks/parked/reliability/d555-usb-dropout-framerate-collapse.md`](d555-usb-dropout-framerate-collapse.md)
  is the sibling brief for the camera; both belong to the
  USB-bus-reliability theme on Jetson.

## Approach

### Three-part fix; all parts ship together (no individual landing)

**Part 1 — CRC + consecutive-failure counters in diagnostics.**

- Add `self._crc_error_count` to `RoboClawInterface`, incremented in
  the `RoboClawChecksumError` branch of `_retry`.
- Add `self._read_call_count`, `self._write_call_count` for rate
  computation.
- `roboclaw_node._publish_diagnostics`
  ([`roboclaw_node.py:448-478`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py))
  reads these from each interface (front + rear) and publishes
  `crc_error_rate_per_s_p60`, `consecutive_failure_count`, and
  `reconnects_since_boot` in the diagnostic_msgs/DiagnosticArray.
- Bump `max_retries` default from 1 to 3 to match the peer-reference
  driver.

**Part 2 — Battery voltage in diagnostics + topic.**

- New topic `/strafer/battery_voltage` (`sensor_msgs/BatteryState`)
  publishing at 1 Hz from `roboclaw_node._timer_callback`.
- Reads `front.read_main_battery()` (the front controller is on the
  bus that powers the 12 V rail; the rear shares the same battery
  through the harness).
- Diagnostic level computed from voltage: `> 11.5 V = OK`,
  `10.5–11.5 V = WARN`, `< 10.5 V = ERROR`. Thresholds env-tunable
  (`STRAFER_BATTERY_WARN_V`, `STRAFER_BATTERY_ERROR_V`).

**Part 3 — Low-battery degraded mode.**

- When battery dips into `WARN`: clamp `vx_max` / `vy_max` / `wz_max`
  by an envelope factor (default 0.6) applied in the velocity
  watchdog ([`roboclaw_node.py:284-293`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py)).
  Log the clamp once per transition.
- When battery dips into `ERROR`: publish a `/strafer/halt` signal
  (custom or `std_msgs/Bool`) that `MissionRunner` watches; new
  missions get rejected with `error_code="battery_critical"`,
  in-flight missions are canceled (reusing the existing cancel path
  patched by
  [`executor-cancel-mid-motion-cmd-vel-zero.md`](../../active/reliability/executor-cancel-mid-motion-cmd-vel-zero.md)).
- Hysteresis: once `ERROR`, stay there until battery recovers above
  `STRAFER_BATTERY_OK_V` (default 11.7 V) for 30 wall-seconds.

## Acceptance criteria

- [ ] Part 1: `diagnostic_msgs/DiagnosticArray` published from
      `roboclaw_node` includes `crc_error_count`,
      `consecutive_failure_count`, `reconnects_since_boot` fields per
      RoboClaw (front + rear).
- [ ] Part 1: `max_retries` default raised to 3; configurable via the
      existing param.
- [ ] Part 2: `/strafer/battery_voltage` (`sensor_msgs/BatteryState`)
      publishes at 1 Hz with `voltage`, `percentage`
      (linear interpolation 10.5–12.6 V), `power_supply_status`.
- [ ] Part 2: Diagnostic level reflects voltage threshold.
- [ ] Part 3: At `WARN` threshold, velocity clamp by `envelope_factor`
      (default 0.6) is applied; logged once per transition.
- [ ] Part 3: At `ERROR` threshold, `/strafer/halt` is asserted;
      `MissionRunner` rejects new missions with `battery_critical`
      and cancels in-flight missions via the existing cancel path.
- [ ] Part 3: Hysteresis prevents thrash — `ERROR` → `OK` requires 30 s
      sustained above `STRAFER_BATTERY_OK_V`.
- [ ] Foxglove layout updated to surface battery voltage gauge + CRC
      rate plot (`strafer_layout.json`).
- [ ] Unit tests cover (a) CRC counter increments, (b) battery
      threshold transitions, (c) velocity clamp activation/deactivation,
      (d) halt assertion + clear hysteresis.
- [ ] Integration smoke on a real chassis at 60% battery (~11.0 V):
      mission still completes, velocity-clamp WARN visible in
      diagnostics. At ~10.4 V (drained): mission gets rejected with
      `battery_critical`.
- [ ] No regression on sim — `HARDWARE_PRESENT=false`
      ([`bringup_sim_in_the_loop.launch.py:151`](../../../../source/strafer_ros/strafer_bringup/launch/bringup_sim_in_the_loop.launch.py))
      bypasses `roboclaw_node` entirely; this brief doesn't change
      that.
- [ ] If your work invalidates a fact in any referenced context
      module, package README, top-level `Readme.md`, or guide under
      `docs/`, update those in the same commit.

## Investigation pointers

- [`source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py:208-219`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py) —
  `_retry`. Add `_crc_error_count` increment in the
  `RoboClawChecksumError` branch.
- [`source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py:349-353`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_interface.py) —
  `read_main_battery`. Already exists; call it from
  `_timer_callback`.
- [`source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py:251-265`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py) —
  `_timer_callback`. Add battery read + diagnostics publish.
- [`source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py:284-293`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py) —
  `VELOCITY_WATCHDOG_MULTIPLIER`. Add the battery-WARN envelope factor.
- [`source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py:424-446`](../../../../source/strafer_ros/strafer_driver/strafer_driver/roboclaw_node.py) —
  `_handle_serial_failure`. Existing reconnect logic; bump max_retries
  default at the constructor.
- Peer driver reference: [wimblerobotics/ros2_roboclaw_driver](https://github.com/wimblerobotics/ros2_roboclaw_driver)
  for the close-and-reopen pattern.
- Pololu CRC failure-modes thread: https://forum.pololu.com/t/roboclaw-crc-failure/3011

## Out of scope

- **Replacing the RoboClaw driver entirely** with the wimblerobotics
  driver. Their driver has different command semantics (their
  `RoboClawStatus` message shape is incompatible with our diagnostics
  consumers). The peer reference is for the pattern, not the
  drop-in replacement.
- **Reading battery from a smart-BMS over a separate bus.** Some
  battery packs publish their own voltage; if the operator wires
  one up, the topic could be sourced there instead of from the
  RoboClaw. File a follow-up if/when that happens.
- **Wheel slip detection.** Filed separately as a follow-up if/when
  it surfaces (encoders + commanded velocity comparison is a
  separate signal-processing problem).
- **IMU bias drift / re-calibration triggers.** Sibling brief
  [`imu-yaw-drift-no-magnetometer.md`](imu-yaw-drift-no-magnetometer.md).
- **`/cmd_vel` zero on `battery_critical`.** Handled by the existing
  cancel path patched by
  [`executor-cancel-mid-motion-cmd-vel-zero.md`](../../active/reliability/executor-cancel-mid-motion-cmd-vel-zero.md).
  This brief only asserts the halt signal; the cancel path takes it
  from there.
