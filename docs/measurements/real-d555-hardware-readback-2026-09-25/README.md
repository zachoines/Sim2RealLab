# Real D555 on the Jetson: stream state, the filter and decode read-backs, and a one-pose texture pilot, 2026-09-25

The D555 streams on USB 3.2 and is not in recovery. On the running node, the four post-processing filters
read **off** and depth auto-exposure reads **on**, exactly as `perception.launch.py` requests them.
That closes the one hardware item of
[`d555-params-file-inert`](../../tasks/completed/d555-params-file-inert.md).

The node-side Z16 decode ran for 170 s on the real stream with `depth_bad_encoding` and
`depth_bad_shape` at **0** on every counter line, and `z16 frames` advanced throughout. That is part of
[`d555-depth-decode-validity`](../../tasks/active/trained-policy/d555-depth-decode-validity.md)'s
hardware item, not all of it. `inferences` cannot advance without a goal, TF and odometry, and the rig
has no motor controller. The close-wall check was not run.

The stream itself has four defects on this host:
- the header stamp does not advance;
- every depth frame is published twice;
- the aligned depth topic publishes nothing;
- the IMU is disabled.

The last two block the real lane outright. The policy's watchdog needs the IMU, and SLAM takes its
depth from the aligned topic.

The texture capture for
[`real-d555-depth-texture-capture`](../../tasks/active/trained-policy/real-d555-depth-texture-capture.md)
is **on hold**. The camera is tethered by USB to a Jetson that sits beside the DGX, so it cannot
reach robot mount height across a room. One pose was recorded at benchtop height before that became
clear, and is reported below as a pilot for the 0.4–1.0 m band only.

## Setup

- **Host:** the Jetson Orin NX 16 GB, L4T R36.4.3, kernel `5.15.148-tegra`.
- **Images:** `strafer-cpu:humble` and `strafer-gpu:humble`, rebuilt from a clean worktree at
  `main` `de3a865`. Both are labelled `de3a865e5810`, so they include #231 and #232. The CPU image
  carries `realsense2_camera` 4.58.4 and `librealsense2` 2.58.4.
- **Services:**
  - `perception` ran through the deploy compose file with the host's direct-link DDS pin.
  - `inference` (policy profile) ran with `strafer_depth_subgoal_v3_999.onnx`, sha256
    `c866bfd5…32eb91`.
- **Topic probes:** small rclpy subscribers, RELIABLE, run inside the `perception` container. They
  record each message's receive time, header stamp and payload digest.
- **Node starts:** the node was started four times. The read-backs were taken on the second start,
  repeated on the third before the pose-1 bag, and repeated again on the fourth, which was left
  streaming. They agree on every value read at more than one start.

## Camera state

| | |
|---|---|
| USB | `8086:0b56` "Intel(R) RealSense(TM) Depth Camera 555"; `/dev/video0`–`5` |
| link | 5000 Mbit/s, `bcdUSB` 3.20; the node logs "Device USB type: 3.2" |
| serial | `409122301816` (node); `471300003-02` (USB descriptor) |
| firmware | `7.56.19918.835`, product ID `0x0B56` |
| driver | "RealSense ROS v4.58.4"; no "No RealSense devices were found" |

The camera's CDC interface appears as `/dev/ttyACM0`. That device is the camera, not a motor
controller.

## The launch's pinned values, read back off the node

`ros2 param get /d555 <name>`:

| parameter | requested by the launch | node reports |
|---|---|---|
| `decimation_filter.enable` | `false` | `False` |
| `spatial_filter.enable` | `false` | `False` |
| `temporal_filter.enable` | `false` | `False` |
| `hole_filling_filter.enable` | `false` | `False` |
| `depth_module.enable_auto_exposure` | `true` | `True` |
| `depth_module.global_time_enabled` | `false` (dropped: "not supported") | `True` |
| `rgb_camera.global_time_enabled` | `false` (dropped: "not supported") | `True` |
| `motion_module.global_time_enabled` | `false` (dropped: "not supported") | "Parameter not set" |

The five pins #231 made reach the node as requested. The three `global_time_enabled` arguments do
not. The wrapper warns and drops each one at launch, and the node then declares the two sensor options
it has at their librealsense default, `True`. The motion module's option is never declared, because
the IMU is disabled (below). This is the outcome the brief's adjacent finding predicted from the
wrapper's source.

Context values from the same read-back:
- `disparity_filter.enable` and `hdr_merge.enable` read `False`, `align_depth.enable` reads `True`
  and `enable_sync` reads `False`.
- `depth_module.depth_profile` is `640x360x30` and `depth_format` is `Z16`.
- The emitter is on, with `laser_power` 150.0, `exposure` 8500 and `gain` 16.

## The Z16 decode on the real stream

`strafer_inference` (`DEPTH_SUBGOAL`, v3, TensorRT/CUDA/CPU providers) ran on the live topic for
170 s of counter lines, one every 10 s:

| | first line | last line | over the window |
|---|---:|---:|---|
| `depth rx` | 401 | 10588 | 59.9 per second |
| `z16 frames` | 401 | 10588 | equal to `depth rx` on every line |
| `bad_encoding` | 0 | 0 | 0 on all 18 lines |
| `bad_shape` | 0 | 0 | 0 on all 18 lines |
| `majority_invalid_cells` | 72.7 % | 72.8 % | 72.7–73.5 % |
| `inferences` | 0 | 0 | the `ready` parameter stays `False` |

- **Exceptions:** the log has no traceback and no error line.
- **Frame rate:** 59.9 frames a second is twice the camera's 30 fps, because every frame arrives
  twice (below).
- **`majority_invalid_cells`:** 72.8 % is the scene, not the sensor. The robot sat on a desk facing
  clutter inside the stereo minimum.
- **Stale sources:** goal, IMU, joint states, odometry, subgoal and TF were listed as stale
  throughout, and 5308 of 5309 timer ticks were watchdog skips.

Two parts of the hardware item remain unverified:
- **Whether `inferences` advances.** That needs an active goal, TF and odometry, and this rig has no
  motor controller. Even with those present, the IMU source would hold it (below).
- **The close-wall check** at 0.15, 0.25 and 0.35 m. It was not run.

## The stream on this host

**1. The header stamp does not advance.**
- Every probe window of 2–20 s found **one** distinct header stamp across all its messages, on raw
  depth and on colour, from 8 s after a start onward. Over 20 s, for example, raw depth carried
  1201 messages with one stamp.
- The per-frame metadata reports clock domain `global_time` and a `frame_timestamp` that stays
  constant while its `time_of_arrival` advances. On the second start one value, 23:07:44.6 UTC,
  held from at least 23:07:07 to 23:10:48. It began 37 s ahead of the host clock and ended 3
  minutes behind it.
- On the third start the stamp read 1525 s ahead of the host clock.

`timestamp_fixer` restamps the colour and aligned-depth topics it relays. The raw depth topic that
`inference` subscribes to keeps the stuck stamp. The node's watchdog runs on receive time, so
nothing gates on it; only its `depth_age` diagnostic reads the stamp.

**2. Every depth frame is published twice.**
- Across 20 s: 1201 messages, **601** distinct payloads, and 600 consecutive pairs bit-identical.
- The metadata's `frame_number` also arrives in pairs.
- In the 40 s pose-1 bag: 2380 depth messages, 1190 consecutive duplicates, and exactly 1190
  distinct frame numbers with no gaps.
- Colour is not duplicated: 301 of 301 distinct in 10 s at 29.96 Hz.

Anything that counts depth messages sees 60 Hz of 30 fps content. The inference node schedules on
its own timer and reuses the newest frame, so the duplicates add no inferences. They do double its
`depth rx` and `z16 frames`.

**3. The aligned depth topic publishes nothing.**
- `/d555/aligned_depth_to_color/image_raw` has a publisher.
- It delivered **0** messages in each 5 s window, at two separate node starts.

RTAB-Map takes depth from the relayed `/d555/aligned_depth_to_color/image_sync`, so on this host
SLAM receives no depth.

**4. The IMU is disabled.**
- librealsense logs "No HID info provided, IMU is disabled", and `/d555/imu` has **0** publishers.
- The kernel is built with `# CONFIG_HID_SENSOR_HUB is not set`. The camera's HID interface binds
  only to `usbhid`, and no `hid-sensor` module exists for the running kernel.

[`docs/D555_IMU_KERNEL_FIX.md`](../../D555_IMU_KERNEL_FIX.md) documents building those modules out of
tree. It was written for L4T R36.5.0 and `5.15.185-tegra`; this host runs R36.4.3 and
`5.15.148-tegra`, and the procedure has not been applied to this kernel.

`inference` lists `/d555/imu/filtered` among its watchdog sources, so on this host the policy would
hold `/cmd_vel` at zero even with a goal, TF and odometry. SLAM takes the same filtered IMU topic.

**5. A diagnostic on the clock.** At runtime, `depth_module.global_time_enabled` was set to `false`,
then `rgb_camera.global_time_enabled` as well:
- The metadata's clock domain becomes `hardware_clock`, and **`frame_timestamp` reads 0.0 on every
  frame.**
- The duplicate publication and the silent aligned topic did not change.
- The node was then recreated, which restored `True`.

So the device's per-frame timestamp never reaches the driver on this host, and the stuck global-time
stamp is that zero passed through the driver's clock mapping. A possible single cause for items 1–3 is that per-frame
metadata does not reach librealsense's kernel backend on this L4T kernel. The same backend needs
kernel support for the HID IMU. **This was not tested.**

These are filed as
[`d555-l4t-stream-integrity`](../../tasks/active/reliability/d555-l4t-stream-integrity.md).

## The texture capture: on hold, with a one-pose pilot

The capture needs at least 3 static poses at robot mount height across a room, together covering
0.4 to 5.5 m. The Jetson is not on the robot, and the camera's USB tether does not reach floor
positions across the room, so the capture stopped after one pose.

**Pose 1:**
- **Height:** benchtop, not mount height. The pose was first reported as on the floor and corrected
  after recording.
- **Scene:** a painted wall on the optical axis, tape-measured at 0.74 m. The temporal median of the
  centre 40×40 pixels read 0.735 m, which checks the Z16 scale.
- **Frame edges:** a cardboard sheet at the left edge hid a PC case with fans, and read mostly
  invalid. Window blinds sat at the right edge; it was evening, with no sun.
- **Capture:** filters and auto-exposure were read back immediately before the bag, and matched the
  table above.
- **Frames:** 1130 frames were analysed after dropping duplicates and the first 60. They cover
  37.6 s on bag receive time at 30.00 fps, with no gaps.
- **Stillness:** by depth, **STILL**. 0.14 % of the 137 840 always-valid pixels moved their median
  by more than 4 standard errors between the first and last thirds, against a 1 % limit. There was
  no IMU to check against.

**Only the 0.4–1.0 m band has data.** The other four bands hold 0–13 raw pixels and no cells.

| 0.4–1.0 m | value |
|---|---|
| raw σ p50 / p90 | 0.767 / 1.325 mm over n = 137 809 pixels |
| post-reduction σ p50 / p90 | 0.679 / 1.136 mm over n = 1922 cells, median depth 0.732 m |
| r = σ_post / σ_raw | 0.8848 |
| ρ = (r² − π/128) / (1 − π/128) | (0.78291 − 0.02454) / 0.97546 = **0.777** |
| ρ, other estimators | 0.802 (raw over the cells' own pixels), 0.825 (median of per-cell ratios), 0.815 (flat cells only, n = 1541) |
| ρ with the Z16 rounding simulated out | 0.738 / 0.760 (two grid-phase models) |
| side of the 2026-09-17 crossover (0.293) | **above**, on every estimator and both corrections |
| side of this capture's own crossover (0.629) | above |

**Survivorship.** A raw pixel counts if it is nonzero in every analysed frame. A cell counts if
all 64 of its pixels are nonzero in every frame and its output is never the near fill.

**Z16 caveat.** The band is quantisation-limited. Z16's 1 mm step has a variance (Δ²/12) that is
14 % of the median raw variance, and the median of 64 rounded values lands on a half-step grid. That
moves the measured ρ up, which is why the simulated corrections are listed.

**Gate (B)** is from `noise-texture-parity-2026-09-17` §8: the training term's injected σ_z against
the post-reduction σ, with a window of [0.5×, 2.0×] below 3.5 m. f·B = 63.935 px·m.

| z | σ_d | σ_z mm | ratio σ_z / σ_post | verdict |
|---|---|---:|---:|---|
| 0.732 m (median cell) | 0.08 (default) | 0.670 | 0.987 | pass |
| 0.700 m (band midpoint) | 0.08 (default) | 0.613 | 0.903 | pass |
| 0.732 m | 0.002 (robust low) | 0.017 | 0.025 | fail |
| 0.732 m | 0.16 (robust high) | 1.341 | 1.975 | pass |
| 0.732 m | 0.0179 (robust geometric mean) | 0.150 | 0.221 | fail |

The σ_d that makes training equal the pilot's σ_post is 0.081 at the median depth and 0.089 at the
midpoint. Both are inside the robust band [0.002, 0.16]. Per cell at the default, the ratio is 0.986
at p50 (p10 0.584, p90 1.502), and 94 % of cells fall inside the window.

**Texture statistic.** On the deployed path the 80×45 high-pass p95 is 3.0 mm, with an exact-zero
share of 0.588. Against the per-cell temporal median it is 2.0 mm, with an exact-zero share of
0.534. The Z16 half-step grid inflates exact ties.

**Invalid blocks, pose 1.** On the deployed path, 35.0 % of cells far-clamp (6.0 m) and 0.24 %
take the near fill; this is identical cell for cell to the training convention. The pre-#232 path
would have near-filled 35.4 % of cells. 29.4 % of blocks have no valid pixel. The cardboard and the
stereo edge band account for most of that.

**Revisit trigger** of the parked native-resolution noise option: **not evaluated.** It asks whether
the real reduction residual is "materially larger" than sim's across the depth range. One band from
one benchtop wall cannot answer that.

**What the pilot is.** It is one scene, one band, at the wrong height, so none of the texture
brief's criteria is met by it.
- It shows that the capture and analysis path works on the real stream: the dedupe, receive-time
  timing, depth-only stillness, and the analysis calling the production decode and masked
  reduction.
- It gives a first reading for the near band. The deployed near-band σ is not far below training's
  at σ_d = 0.08; the ratio is about 1.

## What this record does not claim

- **ρ for any band but 0.4–1.0 m, or any mount-height reading.** The pilot bears on no training
  parameter, and none was changed.
- **That `d555-depth-decode-validity`'s hardware item is met.** Two of its three parts are
  unverified.
- **A cause for the stuck stamp, the duplicates or the silent aligned topic.** Item 5 narrows it
  and states a hypothesis; it does not test one.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `real-d555-hardware-readback-2026-09-25/record-files/` |
| deposit commit | `7ff3f707ca3568c608239de7ca4e663cee17f3a3` |

The deposit holds:
- **`hw/`:** the camera state, the node logs, the read-backs and full parameter dumps, the topic,
  metadata and stamp probes and the clock diagnostic. Also the `inference` log with its counter
  lines, the image build log and stamps, the analysis self-test (213 checks, pass) and the probe
  scripts.
- **`capture/pose1/`:** the pose-1 read-back, notes, recorder logs and frame-number check. It also
  holds the depth bag, gzip-compressed and split in two, with both the stored and the uncompressed
  digests in its `DEPOSIT.md`.
- **`capture/check_pose1/`:** the analysis report and JSON the pilot numbers come from.
- **`tools/`:** the capture procedure and the tools that produced all of it.

The 2 s colour clip and the scene frame extracted from it show a private home. They stay on the
host and are listed in `DEPOSIT.md` by digest only; no number here depends on them.

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/real-d555-hardware-readback-2026-09-25/record-files/. \
      docs/measurements/real-d555-hardware-readback-2026-09-25/
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/real-d555-hardware-readback-2026-09-25/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
7086c4ad4851c4c5730a3f7792a410942d0dae2c3b8409db417e76b7afcef93c  capture/check_pose1/d555_texture_report.md
059e860e641f263840138eb0e52dda7afc3da6cecb7b674d0b15b3056b8f0fb9  capture/check_pose1/d555_texture_summary.json
eef97bbc86fd22d4fecde3f1a456f040537a1dc54873d4f0dee582dff6f966a6  capture/check_pose1.log
7d2e7db601b2d4bafbebcefa8c769c531295a30e47f570418b981cfceb90f7fb  capture/pose1/bag_info.txt
5170c4686a334cdd8fcc5244360623b4e0cec5fb05561212a5f2c54ea149c370  capture/pose1/bag/metadata.yaml
020762c5e17615fa7b0cb71076003af1573da19bc7c4c47a4951ea6180b4afdb  capture/pose1/bag/pose1_0.db3.gz.part-00
1f591f0313c76ba78c522e88dd26e9ea816237ba0153bcf4b17cdcc87b6f2356  capture/pose1/bag/pose1_0.db3.gz.part-01
2725fd5ff948c45ac34667c53c9c91e7e5d4b5cfd1176897b32071e78c3619cb  capture/pose1/frame_numbers.json
1339ae7e155d576c262da90940a79f345c5d7c55cdf37fffe40ea23b3f04a177  capture/pose1/notes.txt
9b62c71d87bd86d784d0d50ef2c172561decc31134e34c3d00485547f72ed067  capture/pose1/params_dump.yaml
3ba667efb74aa05ecb5a6ced481dbc22a0f15b6405e0b424ea67b62b5ccf9220  capture/pose1/params_readback.tsv
cc7a440c176be4b1968d2d36db6751c9168c01d5ee881288419cc61971148789  capture/pose1/record_color.log
9435e345d2ac7daea77a4e8d2430986183059cc352f229ac8806769a28407be5  capture/pose1/record.log
09d964d23ab2a4c1379d22bc0499b1ff97ea9e76893f63271d283da9750fcc12  capture/pose1/t_end_utc.txt
fdff120ea14ef59efa0a50b25bbdd5eb018ae18fc8a7129ce48acd201a0e5fe0  capture/pose1/t_start_utc.txt
b9a3ec373399a16eb932f7ed81787053168ee203e4ceea85c123008d720c3ee8  hw/camera_usb_state.txt
1df5be2cf6b596e1b86ace8415f35954153ae5b21ef9c8485f482a0264bc5a2d  hw/depth_metadata_samples.txt
a7f141c0e13c1bee76a5a942c6c5b8bbe38e9bafaaec63916413430028b8130a  hw/depth_topic_readback.txt
3e34214a1776f25eb9e7d3a804ff2b03e4d29d9f1a21846ee6a3da3848edcd5d  hw/global_time_diagnostic.txt
9f5ad4c9c76e7186f89f1b283dc78595a9dcad13d338a2be60edeab5bcb011e2  hw/image_realsense_pkgs.txt
dea45276b44623641cd42ecd621f0c95cbe459a16812194e8137103be6e71e9c  hw/image_stamps.txt
d65bde3af0a22846a55b0d74591a28c76727c6c42d9d3d5183c39824822f7f7c  hw/inference_cadence_lines.txt
ccb9b2e2455a53849210a7378dde16809cbc870f94495f0b1912e3924dc57273  hw/inference_node.log
d65744b2146bd8f6e35eb747915162ebefb8525fbd155c52786663832a2a1aad  hw/inference_up_utc.txt
44d85bed642f92497b4f929b9e73586d35346243563a4800ed3cbe4b99b313ad  hw/make_images_main.log
9b62c71d87bd86d784d0d50ef2c172561decc31134e34c3d00485547f72ed067  hw/params_dump_231.yaml
7ce6e299fe30641aac206d8351b0bc83a72f500236c79b5c7b7892ab459626ce  hw/params_readback_231.tsv
e3917c0d88fd62988910c280c3950e3939eb76c00838f59220cf2c4fe8690410  hw/perception_node.log
cb77b852d91439bcb9beb256f4017b86bbac9f0dcb2eec86dcf0ca512a64dba1  hw/perception_up2_utc.txt
0758dd3766594a6709cd0f3cfc89f0d1e7e465d9db46ac5c09cf7334ffda346b  hw/perception_up3_utc.txt
d6628b4e7bfab65d77bdd0e3cac2c1685f094980232c58acee8b9cc1ffcbbc35  hw/perception_up4_utc.txt
a8ee179d0bab89720f238a60882cfed2bf84c9539f891db578140d7664b756f6  hw/perception_up_utc.txt
b328bb8afb0e555fcf0cd2445c2264ba1bd318876985856ab8a72a71440dea74  hw/probe/depth_stamp_probe.py
f3d72f63a8cc63f094ec6c9121bbc2bde71b12fd95bd1a16649cbcc58cd00f6f  hw/probe/metadata_probe.py
cebe3240370339e89484e74c2d854f9167366f5ae477dabf32499a1001b9dbf5  hw/probe/snap.py
b9cf9014f70b9ea304d4fa1f0878e038b86403500b25486f808aadd2e9e8ac01  hw/selftest_v3.log
3746e7a751983a2d910e1fc67ae00948f5318161e30b91e57548ec8201264831  hw/stamp_freeze_timeline.txt
0476639daa021c0c67f3d66d28d2d12f2d9911dcf5ea469302a883d2f3953665  tools/analyse_bags.sh
c1bd7b5d21ffbb5c725d8ade99d42d721f4e0ab4c7933a8c00a6a33ca0a36a58  tools/bag_frame_numbers.py
4001fd2dbfa4560806b4719c0bb9be45a8e99df71e40a861d97e8db413776176  tools/bag_to_npz.py
1765daa9a25a7675d0d9f97df92790f74f45a23480eb655160d9f4a30b730aff  tools/CAPTURE.md
39c75cdc6a4e4ab972ea243bb8106cbbbcadfb8d501ae8482eed31a46bc723a3  tools/d555_texture_analysis.py
4663491584a3b9a742f950c968a8bef2ebbe67221ebe78904b5f656dcfa30be0  tools/pose_capture.sh
```
