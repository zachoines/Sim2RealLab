# Make the D555 stream whole on the Jetson's kernel: stamps, one message per frame, aligned depth, the IMU

**Type:** bug (deploy runtime — sensor stream)
**Owner:** Jetson
**Priority:** P1 — two of the four defects block the real lane outright. The
inference node's watchdog requires the IMU, so the policy holds `/cmd_vel` at
zero. RTAB-Map takes its depth from the aligned topic, which publishes nothing.
**Estimate:** M (a host-kernel or driver-backend change, then a read-back of
all four)
**Branch:** `task/d555-l4t-stream-integrity`

## Story

As the **real-robot deploy lane**, I need **the D555 to deliver advancing
timestamps, one message per exposure, aligned depth and IMU samples on the
Jetson's own kernel**, so that **the policy's watchdog, SLAM and every
recording of the camera see the stream the stack was designed around.**

## Context bundle

- [context/repo-topology.md](../../context/repo-topology.md)
- [context/conventions.md](../../context/conventions.md)
- [context/branching-and-prs.md](../../context/branching-and-prs.md)
- [`real-d555-hardware-readback-2026-09-25`](../../../measurements/real-d555-hardware-readback-2026-09-25/README.md),
  "The stream on this host": every number below, and the probes that took
  them.

## Context

These defects were measured on the Jetson Orin NX (L4T R36.4.3, kernel
`5.15.148-tegra`) with `realsense2_camera` 4.58.4 and librealsense 2.58.4, on
the deployed launch, with a D555 on firmware 7.56.19918.835 over USB 3.2:

1. **Header stamps do not advance.**
   - Raw depth and colour carry one stamp per probe window, 20 s for raw depth.
   - The metadata's `global_time` `frame_timestamp` holds one value for minutes
     while its `time_of_arrival` advances.
   - With `depth_module.global_time_enabled` set false at runtime, the clock
     domain becomes `hardware_clock` and `frame_timestamp` reads **0.0** on
     every frame. The device's per-frame timestamp never reaches the driver.
2. **Every depth frame is published twice.**
   - Raw depth delivers 60 messages a second of 30 fps content: each payload,
     and each `frame_number`, arrives as a consecutive pair.
   - Colour is not duplicated.
   - Turning global time off does not change this.
3. **Aligned depth publishes nothing.**
   - `/d555/aligned_depth_to_color/image_raw` has a publisher and delivered 0
     messages in 5 s windows at two node starts.
   - `timestamp_fixer` relays it as `/d555/aligned_depth_to_color/image_sync`,
     which is RTAB-Map's depth input.
4. **The IMU is disabled.**
   - librealsense logs "No HID info provided, IMU is disabled", and `/d555/imu`
     has no publisher.
   - The kernel is built with `CONFIG_HID_SENSOR_HUB` unset, and no
     `hid-sensor` module exists for the running kernel.
   - [`docs/D555_IMU_KERNEL_FIX.md`](../../../D555_IMU_KERNEL_FIX.md) builds
     those modules out of tree, but it was written for R36.5.0 and
     `5.15.185-tegra` and has not been applied to this host's kernel.
   - `inference` lists `/d555/imu/filtered` as a watchdog source, and SLAM takes
     it too.

Items 1–3 may share one cause: per-frame metadata not reaching librealsense's
kernel backend on this L4T kernel, which is also the backend that needs kernel
HID sensor support for item 4. **This is a hypothesis.** The record narrows the
clock but tests no cause. Whether the duplicates or the silent aligned topic
follow from the zero timestamps, or from `align_depth.enable` itself, was not
tested.

The inference node survives items 1 and 2 today. Its watchdog runs on receive
time, and its timer reuses the newest frame, so duplicates add no inferences.
Only its `depth_age` diagnostic and its `depth rx` / `z16 frames` counters
read wrong. Recorders and parity tooling do not survive them.

## Acceptance criteria

- [ ] Raw depth and colour header stamps advance once per exposure over a
      ≥ 60 s window on the deployed launch, and the clock they are on is stated
      and read back. Consumers of raw stamps are checked against it:
      `timestamp_fixer`'s first-frame log, the inference node's `depth_age`,
      and bag tooling.
- [ ] `/d555/depth/image_rect_raw` carries one message per exposure: distinct
      payloads equal messages, and each `frame_number` appears once, over
      ≥ 60 s.
- [ ] `/d555/aligned_depth_to_color/image_raw` publishes at the colour rate,
      and `/d555/aligned_depth_to_color/image_sync` reaches RTAB-Map.
- [ ] `/d555/imu` publishes gyro and accelerometer samples, and
      `/d555/imu/filtered` follows; with `inference` up, `imu` drops out of its
      stale sources.
- [ ] Whatever host or image change this takes survives a reboot and is
      written down for the host's actual L4T and kernel.
      `docs/D555_IMU_KERNEL_FIX.md` is updated or generalised if the IMU route
      goes through it.
- [ ] Each item is read back with the record's probes (its deposit's
      `hw/probe/`) and recorded.
- [ ] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.

## Investigation pointers

- `source/strafer_ros/strafer_perception/launch/perception.launch.py`: the
  `rs_launch.py` include. It sets `align_depth.enable`, and its three
  `global_time_enabled` arguments are dropped by 4.58.4 (see
  [`d555-params-file-inert`](../../completed/d555-params-file-inert.md),
  "Adjacent finding").
- `source/strafer_ros/strafer_perception/strafer_perception/timestamp_fixer.py`:
  what it relays and restamps.
- `source/strafer_ros/strafer_inference/strafer_inference/inference_node.py`
  (`imu_topic`) and `watchdog.py` (`stale_sources`): why a missing IMU holds
  the policy.
- `source/strafer_ros/strafer_slam/launch/slam.launch.py`: SLAM's depth and IMU
  inputs.
- On the host: `zcat /proc/config.gz | grep -E 'HID_SENSOR|USB_VIDEO'`,
  `cat /etc/nv_tegra_release`, and `ls /sys/bus/iio/devices`.
- librealsense has two routes on Jetson: patched L4T kernel modules, or its
  libusb-based backend, which does not go through `uvcvideo` metadata or IIO
  HID. Which one fits a deb-installed 2.58.4 inside `strafer-cpu:humble` is
  part of the task; neither was tried here.

## Out of scope

- **The texture capture.**
  [`real-d555-depth-texture-capture`](../trained-policy/real-d555-depth-texture-capture.md)
  can proceed without this: it dedupes and times on receive time. It needs the
  IMU only for its second stillness check.
- **The decode brief's remaining hardware items** (`inferences` advancing, the
  close-wall check) and its `depth_qos` pin, owned by
  [`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md).
- **Any training-side change.**
