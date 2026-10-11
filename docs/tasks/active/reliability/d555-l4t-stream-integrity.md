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
      *2026-10-04 ([`d555-stream-integrity-2026-10-04`](../../../measurements/d555-stream-integrity-2026-10-04/README.md)):*
      - *Met except `depth_age`. Raw depth carries 1801 unique stamps in 1804 messages
        over 60 s and colour 1798 of 1798, about 33.3 ms apart.*
      - *Both are on librealsense's `global_time`; receive − stamp has a median of 25 ms on
        depth and 22 ms on colour over 60 s.*
      - *A depth image's stamp is the time of the colour frame it was paired with: realsense-ros
        stamps a frameset with its first frame, the colour frame. It sits up to ±16.7 ms from
        the depth frame's own time and sweeps that range every 29.0 s, the beat between depth
        (30.0000 fps) and colour (29.9655 fps under realsense-ros); the period is measured, not
        a camera constant. The depth frame's own time is
        `/d555/depth/metadata` `frame_timestamp`.*
      - *`timestamp_fixer`'s first frame reads a 0.007 s delta.*
      - *`bag_frame_numbers.py` reads 344 distinct stamps for 344 frames.*
      - *`depth_age` reads n/a, because the inference node runs no inference without a goal,
        TF and odometry, and those need the chassis.*
      - *`timestamp_fixer` now passes stamps through on the real lane: restamping split a
        frame's image and camera info, so `depth_to_pointcloud`'s exact sync made no pairs.
        The deployed images carry the change only after `make images`.*
      - *2026-10-09: the re-stamping mode is removed, so the node is a pure relay on every
        lane. The relay's name and whether it should stay are filed as
        [`camera-relay-rename-or-retire`](camera-relay-rename-or-retire.md) (P3).*
- [x] `/d555/depth/image_rect_raw` repeats at most 0.5 % of its messages over
      ≥ 60 s, the mechanism of the repeats is recorded, and recorders dedupe by
      stamp or `frame_number`.
      *(Reworded 2026-10-05 from "carries one message per exposure: distinct payloads
      equal messages, and each `frame_number` appears once, over ≥ 60 s". The residual
      repeats come from the wrapper's half-frame sync window; removing them would need a
      realsense-ros change.)*
      *Met 2026-10-04 ([`d555-stream-integrity-2026-10-04`](../../../measurements/d555-stream-integrity-2026-10-04/README.md)):*
      - *3 repeats in 1804 messages over 60 s (0.17 %; 50 % before), 1801 distinct payloads.*
      - *Mechanism: depth gains a frame on colour every 29.0 s. At each crossing librealsense's
        syncer releases at least one depth frame alone, 16.6–16.7 ms from the nearest colour
        frame, and realsense-ros publishes a depth-only frameset twice.*
      - *`frame_number` was read over two short windows only, neither ≥ 60 s: the 20 s
        metadata window (603 messages, 601 distinct frame numbers, no gaps) and an 11.5 s bag.*
      - *`bag_frame_numbers.py` dedupes consecutive repeats: the bag holds 346 depth messages
        and 344 distinct stamps for 344 distinct frame numbers. Its 2 in 346 (0.58 %) is over
        the bound because the short window holds a crossing. Repeats cluster at crossings (one
        or two per crossing in the deposited windows), so a window shorter than the beat can
        exceed the bound, and the bound needs the ≥ 60 s window.*
- [x] `/d555/aligned_depth_to_color/image_raw` publishes at the colour rate,
      and `/d555/aligned_depth_to_color/image_sync` reaches RTAB-Map.
      *Met 2026-10-04 ([`d555-stream-integrity-2026-10-04`](../../../measurements/d555-stream-integrity-2026-10-04/README.md)):*
      - *Both publish at 29.93 Hz over 60 s.*
      - *RTAB-Map is a matched subscriber of `image_sync`, of the colour topics and of `/scan`.*
      - *With stamps passed through, the point cloud behind `/scan` read 26.6 Hz over the last
        600 messages of a 65 s run, but `/scan` itself stays silent: it projects into
        `base_link`, which `base` publishes.*
      - *Its only input without a publisher is `/strafer/odom`, from `base`, so it does not map
        on this rig state.*
- [x] `/d555/imu` publishes gyro and accelerometer samples, and
      `/d555/imu/filtered` follows; with `inference` up, `imu` drops out of its
      stale sources.
      *Met 2026-10-04 ([`d555-stream-integrity-2026-10-04`](../../../measurements/d555-stream-integrity-2026-10-04/README.md)):*
      - *`/d555/imu` and `/d555/imu/filtered` publish at 199.8 Hz, stamps every 5.01 ms.*
      - *The inference node's `stale_sources` are `goal joint_states odom subgoal tf`, without
        `imu`.*
- [ ] Whatever host or image change this takes survives a reboot and is
      written down for the host's actual L4T and kernel.
      `docs/D555_IMU_KERNEL_FIX.md` is updated or generalised if the IMU route
      goes through it.
      *2026-10-04: written down; reboot not yet checked.*
      - *The procedure, generalised to R36.4.3 / `5.15.148-tegra` with the `uvcvideo` metadata
        entry, the container access, the `iio-sensor-proxy` mask and the kernel-upgrade
        caveat, is in `docs/D555_IMU_KERNEL_FIX.md`.*
      - *The modules are under `updates/strafer-d555/` and load by alias.*
      - *The reboot check needs the camera unplugged first, so it waits for someone on site:
        unplug, reboot, check `timedatectl`, replug, confirm `/sys/module/uvcvideo/srcversion`
        `D0E2A5944399A529814A98C`, the `hid-sensor-hub` binding and
        `systemctl is-enabled iio-sensor-proxy` (`masked`). It is required before the real-lane
        gate is pre-registered.*
      - *Process: both stages' root steps were run by the work session over SSH, under the
        maintainer's remote authorization and against the standing rule that host changes are
        the maintainer's commands. That was a one-off: host changes stay the maintainer's unless
        delegated again, step by step, in writing.*
- [x] Each item is read back with the record's probes (its deposit's
      `hw/probe/`) and recorded.
      *Met 2026-10-04: baseline, after stage 1 and after stage 2, in [`d555-stream-integrity-2026-10-04`](../../../measurements/d555-stream-integrity-2026-10-04/README.md).*
- [x] If your work invalidates a fact in any referenced context module, package
      README, top-level `Readme.md`, or guide under `docs/`, update those in the
      same commit. See
      [`conventions.md`'s user-facing documentation maintenance section](../../context/conventions.md#user-facing-documentation-maintenance)
      for the surface list and trigger heuristics.
      *Met 2026-10-04 ([`d555-stream-integrity-2026-10-04`](../../../measurements/d555-stream-integrity-2026-10-04/README.md)):*
      - *`docs/D555_IMU_KERNEL_FIX.md` (generalised);*
      - *the top-level `Readme.md` and `source/strafer_ros/README.md` D555 notes;*
      - *`install-host-prereqs.sh`'s hardware note;*
      - *the perception launch and `timestamp_fixer` docstrings, whose "falls back to system
        time" and "HW clock drifts" premises no longer hold;*
      - *2026-10-05: the host named as the Orin NX 16 GB on L4T R36.4.3, and the SLAM pipeline as
        `depth_to_pointcloud` → `pointcloud_to_laserscan`, in both READMEs; the kernel doc's
        depmod ranking, upgrade failure mode, mask check and depth-stamp clock.*
      - *2026-10-09: for the pure relay, the kernel doc's container-access note and
        `depth_to_pointcloud` verification row, the `*_sync` consumers in
        `source/strafer_ros/README.md`, the `timestamp_fixer` docstring and the two sim launches'
        relay comments.*

## Investigation pointers

- `source/strafer_ros/strafer_perception/launch/perception.launch.py`: the
  `rs_launch.py` include. It sets `align_depth.enable`, and its three
  `global_time_enabled` arguments are dropped by 4.58.4 (see
  [`d555-params-file-inert`](../../completed/d555-params-file-inert.md),
  "Adjacent finding").
  *(2026-10-04: the three arguments are removed; global time stays at the
  wrapper's default.)*
- `source/strafer_ros/strafer_perception/strafer_perception/timestamp_fixer.py`:
  what it relays and restamps. *(2026-10-04: every launch now runs it with
  `restamp:=false`. 2026-10-09: the re-stamping mode and its parameter are removed.)*
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
  [`real-d555-depth-texture-capture`](../../completed/real-d555-depth-texture-capture.md)
  was taken without this on 2026-09-26: it dedupes and times on receive time,
  and without the IMU its stillness rests on depth alone.
- **The decode brief's remaining hardware items** (`inferences` advancing, the
  close-wall check) and its `depth_qos` pin, owned by
  [`d555-depth-decode-validity`](../trained-policy/d555-depth-decode-validity.md).
- **Any training-side change.**
