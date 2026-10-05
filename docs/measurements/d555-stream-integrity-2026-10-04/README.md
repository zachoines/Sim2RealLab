# The D555's IMU, stamps and aligned depth fixed on the Jetson's kernel, 2026-10-04

On the robot host's kernel (L4T R36.4.3, `5.15.148-tegra`), the D555's IMU now publishes. Its
depth and colour carry stamps that advance once per frame, on librealsense's global time, and
aligned depth publishes at the colour rate. Raw depth repeats fell from 50 % of messages to
0.17 % (3 in 1804); the rest recur at a beat between the depth and colour frame rates, a
wrapper-side mechanism (below). A depth image's stamp is the time of the colour frame it was
paired with; the depth frame's own time is in `/d555/depth/metadata` `frame_timestamp`. Before,
the stamps were frozen, depth arrived twice, aligned depth was silent and there was no IMU.

The fix is six out-of-tree kernel modules for the running kernel, plus IIO device access for
the perception container:
- five HID-sensor modules for the IMU;
- `uvcvideo` rebuilt with librealsense's own D555 metadata entry.

It was applied in two stages chosen to test one hypothesis, written down and deposited before
any module change: the stamps, the duplicates and the silent aligned topic share one cause, the
camera's metadata not reaching librealsense. At stage 1 the IMU came up and the other three
defects stayed; the depth stamps went from 1 distinct value to 5 within 8 ms, where the
prediction was no change, and that change cannot be attributed (Deviations). At stage 2 the
stamps and aligned depth resolved and the duplicates fell to 0.17 %.

**Not yet verified:** that the change survives a reboot. A reboot needs the camera unplugged
first, and nobody was on site.

## Setup

| | |
|---|---|
| host | Jetson Orin NX 16 GB (Seeed image), L4T R36.4.3, kernel `5.15.148-tegra` built by Seeed; NVIDIA's `nvidia-l4t-kernel-headers 5.15.148-tegra-36.4.3` |
| camera | D555, `8086:0b56`, bcdDevice `50c5`, FW 7.56.19918.835, USB 3.2 at 5 Gb/s; plugged in after boot (uptime 24 days); `base` never started |
| stack | `strafer-cpu:humble` / `strafer-gpu:humble` revision `9b80e9976961`; `realsense2_camera` 4.58.4 on librealsense 2.58.4 (deb, kernel V4L2/IIO backend); the deployed `perception` launch, with the #231 filter pins |
| probes | the 2026-09-25 record's `depth_stamp_probe.py` and `metadata_probe.py`; `uvc_header_probe.py` (raw V4L2 metadata, read-only); `imu_stamp_probe.py`; the 2026-09-26 record's `bag_frame_numbers.py` (sha256 `c1bd7b5d21ffbb5c725d8ade99d42d721f4e0ab4c7933a8c00a6a33ca0a36a58`); `record_figures.py` recomputes the figures below from the deposited JSON |
| windows | 60 s for raw depth and colour; 20 s for metadata topics; the other rows name their own |
| clock | NTP-synchronized; no reboot during the measurements |

## Route decision, before any module change

The decision and its pre-checks were committed to the deposit (`f462601c`, `route_decision.md`
sha256 `f4367f58…`) before any module, sysfs, udev, service or image change. The compose change
came about 2.5 minutes earlier, at 02:29:57Z against the commit's 02:32:26Z (see Deviations). A
review then checked the mechanism against the
kernel and librealsense sources; its corrections to the root steps are in the deposited scripts.

Pre-checks in `f462601c`, none changing the host, all run 01:57–02:08Z:
- **Interfaces.** Both VideoControl interfaces report UVC 1.5 (protocol 01). The HID interface was
  bound to `usbhid`.
- **Metadata formats.** Every metadata node offered only `UVCH`, the standard header.
- **Streamed raw** (depth Z16 and colour YUYV, 640×360 at 30, 10 s each): **295 of 295 metadata
  buffers empty** on both.
- **Baseline on the deployed launch.** All four defects reproduced (table below).
- **`align_depth.enable` set false at runtime.** Raw depth dropped to **30.00 Hz, 1801 of 1801
  payloads unique and 601 of 601 frame numbers distinct**, with the stamps still frozen. Set back
  to true, the 60 Hz pairs returned. The duplicates therefore come from the sync/align path, and
  the frozen stamps do not.

Read before stage 1 (02:52:07–02:52:17Z, after the compose change) and deposited with the
post-change read-backs (`1743ee5f`, `readback/stage0_stats/`): `uvcvideo`'s debugfs stream statistics, a root read that changes
nothing. A PTS was present in all 296 depth payloads, and an SCR in all 296 with SOF 0. Under
`UVCH` an entry is written only when the SCR changes. The direct `D4XX` read after stage 2 shows
the SCR is all zero on every frame (below), so under `UVCH` no entry is ever written.

From the code, `jetson_36.4.3` `uvc_video.c` and librealsense 2.58.4 `backend-v4l2.cpp`:
- librealsense stores `bytesused − 10` in a `uint8_t`, so an empty metadata buffer reads as 246
  bytes of zeros. The frame timestamp is 0, so the `global_time` fit returns a constant.
- With every frame stamped alike, librealsense's syncer, which realsense-ros runs whenever align
  is enabled, never pairs depth with colour. realsense-ros publishes a lone depth frameset twice,
  and aligned depth is made only from a frameset holding both.

**Decision.** Route (i), host kernel modules, in two stages. It keeps the deb-installed
librealsense 2.58.4 unchanged. Route (ii), RSUSB librealsense built into the image, was kept as
the fallback.

| stage | change | prediction if the hypothesis holds |
|---|---|---|
| 1 | `hid-sensor-hub`, `-iio-common`, `-trigger`, `-accel-3d`, `-gyro-3d`; `perception` gets `c 248:* rmw` and a writable `/sys/devices` | the IMU publishes; stamps, duplicates and aligned depth unchanged |
| 2 | `uvcvideo` from the same `jetson_36.4.3` sources with librealsense v2.58.4's unmodified `realsense-metadata-jammy-master.patch` | `D4XX` offered and non-empty with a header longer than 12 bytes; stamps advance per exposure; one message per frame; aligned depth at the colour rate |

The modules were built as a normal user against the installed headers (`readback/build_checks.txt`):
- vermagic `5.15.148-tegra SMP preempt mod_unload modversions aarch64` on all six;
- of the 285 versioned symbols they import, the headers' `Module.symvers` lists 243 (185 from
  `vmlinux`, 58 from stock modules such as `videodev`), with 0 CRC mismatches. The other 42 are
  exported among the six themselves, and match the build's own `Module.symvers`. The patched
  `uvcvideo`'s 155 imported CRCs are identical to the stock module's;
- the patched `uvcvideo` has `srcversion D0E2A5944399A529814A98C` and 95 aliases;
- an unpatched rebuild reproduces the stock module's 61-entry alias table, parameters and
  dependencies exactly.

`readback/build_checks.txt` is the output of `tools/build_checks.sh`, run with the stack stopped;
re-run with the same modules loaded, it reproduces the file byte for byte. `tools/build_raw.sh`
writes the inputs it summarises to `readback/build_raw/`: each module's `modinfo` and
`--dump-modversions`, the per-symbol CRC comparison, and the digests of the stock `uvcvideo`, the
unpatched control module and the headers' `Module.symvers`. The control module itself is not
deposited.

## Read-backs

| | baseline | after stage 1 | **after stage 2** |
|---|---|---|---|
| raw depth, 60 s | 3599 msgs, 60.00 Hz; **1 unique stamp**; 1800 unique payloads | 3596 msgs, 59.93 Hz; 5 unique stamps spanning 0.008 s; 1801 unique payloads | **1804 msgs, 30.05 Hz; 1801 unique stamps spanning 60.002 s; 1801 unique payloads**; receive − stamp 25 ms (median; per-second medians 22.0–38.4 ms, below) |
| colour, 60 s | 1798 msgs, 29.97 Hz; **1 unique stamp** | 1801, 30.00 Hz; 5 unique stamps | **1798, 29.97 Hz; 1798 unique stamps**, dt 32.9–34.2 ms; receive − stamp 22 ms (median) |
| depth metadata, 20 s | 1201 msgs, 601 distinct `frame_number`; `frame_timestamp` 1 distinct (`global_time`) | 1201 / 601; 1 distinct | **603 / 601; `frame_timestamp` 601 distinct**; new keys `hw_timestamp`, `sensor_timestamp`, `actual_exposure`, `gain_level`, `frame_laser_power` |
| aligned depth (raw / `image_sync`); 20 s, 20 s, then 60 s | publisher 1, **0 msgs** / 0 | 0 / 0 | **29.93 / 29.93 Hz**, all stamps unique |
| `/d555/imu`; topic info, 15 s `ros2 topic hz`, then 60 s | **0 publishers**; "No HID info provided, IMU is disabled" | **199.8 Hz** | **199.79 Hz**; stamps every 5.01 ms, none going backwards, 1.2 ms from receive time; `/d555/imu/filtered` the same |
| `timestamp_fixer` first frame | delta −909.8 s | delta −774.7 s | **delta 0.007 s** |

**What the camera sends, read directly after stage 2.** The metadata nodes offered `['UVCH',
'D4XX']`. With `D4XX` selected, perception stopped, and 10 s each of depth and colour:
- every buffer non-empty (297 of 297 depth, 298 of 298 colour), one entry per frame, 258 bytes
  each;
- **`bHeaderLength` 248**, which is librealsense's D4xx metadata size;
- a PTS on every frame, stepping 33,332 µs (depth) and 33,331 µs (colour) at the median;
- an SCR present on every frame, and zero on every frame.

The camera was sending per-exposure timing all along; under `UVCH` it was discarded.

From stage 2 on, each perception start logged `HID set_power 1 failed` once for each IIO
buffer's `buffer/enable` (`readback/perception_stage2.log` and both checks' `perception.log`);
the stage-1 start did not. The IMU streamed at 199.8 Hz regardless. It matches the known issue
in `docs/D555_IMU_KERNEL_FIX.md`, where a stop while streaming leaves the buffers enabled and the
next start's enable logs the warning; the cause was not tested here.

**Residual repeats, and which clock the depth stamp is on.** After stage 2, 3 of 1804 raw-depth
messages repeat the one before (0.17 %, against 50 % before). They come from how the wrapper
pairs and stamps frames, which the kernel fix does not touch:
- **One stamp per frameset.** realsense-ros stamps every message of a synced frameset with the
  frameset's time, which librealsense takes from its first frame, the colour frame. A paired
  depth image therefore carries the colour frame's global time, not its own. In the depth
  metadata the header stamp sits 10.7, 10.7 and 0.2 ms from the depth frame's own
  `frame_timestamp` on the three printed rows; on colour it is 0.0 ms. So depth and colour
  sharing a stamp on 1797 of 1800 depth frames (99.8 %) is by construction, not because the two
  frames are simultaneous.
- **The beat.** On the camera's clock (`hw_timestamp`), depth ran at 30.0000 fps and colour at
  29.9655 fps under realsense-ros, with colour auto-exposure on (`gain_level` 248, `actual_fps`
  29954–29961), so depth gains a frame on colour every 29.0 s. In the direct V4L2 read, with
  perception stopped, colour ran at 30.0002 fps, so the period is a value measured in these
  windows, not a camera constant. The
  offset between a depth frame and the colour frame it is paired with sweeps through about
  ±16.7 ms each cycle, and depth receive − stamp follows it: per-second medians fall from 38.4 ms
  to a plateau near 23 ms, then step up by about 15 ms at each crossing (02:59:27.8Z and 02:59:56.8Z, 29.0 s
  apart). The 300 s colour trace has 11 such steps, 29.0 s apart on average.
- **The repeats.** At a crossing, librealsense's syncer releases at least one depth frame alone,
  16.6–16.7 ms from the nearest colour frame, which is about half a frame interval, its pairing
  window. realsense-ros publishes a depth-only frameset twice. The three repeats are at indices
  762/763 (the 02:59:27.8Z crossing) and 1631/1632 and 1634/1635 (02:59:56.8Z, where the pairing
  flipped twice). The 11.5 s bag shows the same: 2 of 346.
- **Aligned depth.** Where the offset sits on the window edge, a colour frame is also released
  alone and gets no aligned depth: one at the 02:59:56.8Z crossing, none at 02:59:27.8Z. A later
  60 s capture has 1796 aligned messages against the 1798 colour frames expected over its span,
  with two 66.7 ms gaps 28.9 s apart.
- **Consequences.** The inference node runs on receive time and reuses the newest frame, so it is
  unaffected. RTAB-Map's approximate sync tolerates the ±16.7 ms. A tool that needs a depth
  frame's own time reads `/d555/depth/metadata` `frame_timestamp` (601 distinct values in 20 s);
  a recorder dedupes by stamp or `frame_number`.

`tools/record_figures.py` recomputes these figures from the deposited JSON.

## Consumers

- **Inference node** (`strafer_inference`, v3, the real lane on wall time).
  - `stale_sources[goal joint_states odom subgoal tf]`: **`imu` is no longer among them**. The
    2026-09-25 record listed it.
  - The rest need `base`, TF and a goal, which this rig state does not have.
  - `depth rx` advances 30 per second (59.9 before) with `bad_encoding` 0. `depth_age` reads n/a
    because no inference runs without a goal.
- **`timestamp_fixer`.**
  - Its first-frame log now reads a 0.007 s delta.
  - Restamping each message on arrival gave a frame's image and camera info different stamps, so
    `depth_to_pointcloud`'s exact sync made **0 pairs** (30 of each per second received) and
    `/scan` stayed empty.
  - With the relay swapped to `restamp:=false` in the deployed container
    (`readback/passthrough_check/`, `relay_passthrough.log`), the `slam` log shows 0 exact-sync
    warnings from its start at 03:32:52Z to the check's end at 03:39:38Z (6 min 46 s). The
    aligned point cloud read 26.6 Hz over the last 600 messages of a 65 s `ros2 topic hz` run
    ending 03:34:27Z (`points_hz.txt`). That run did not time the cloud's input; the 60 s
    `image_sync` rate after stage 2, with the restamping relay, was 29.93 Hz
    (`stage2_supp/aligned_sync60.txt`). This record does not diagnose the difference.
  - `/scan` still did not publish, for a reason outside the stream: `pointcloud_to_laserscan`
    projects into `base_link`, which comes from `base`'s `robot_state_publisher`. A re-check
    (`readback/scan_check/`) found 0 `/scan` messages in 30 s, the cloud at 27.0 Hz over the same
    30 s, and `tf2_echo` reporting that `base_link` does not exist.
  - The laserscan node's message filter gave one reason only, "discarding message because the
    queue is full", as a throttled line about every 2.5 s. Up to each check's end it was logged
    161 times (`passthrough_check/tf_filter_drop_lines.txt`) and 24 times
    (`scan_check/laserscan_drop_reasons.txt`). Both `slam.log` files run past their check's end,
    to 03:43:29Z and 03:52:05Z, and hold 186 and 28 such lines.
  - `perception.launch.py` now passes stamps through, as the sim lanes already did. The images
    still carry the restamping launch until they are rebuilt.
  - Over 300 s of colour with stamps passed through, all 8990 stamps were distinct, and
    receive − stamp held at a median of 24.3–25.1 ms in each minute.
- **RTAB-Map** (`slam`, real lane, separate scene key; `readback/passthrough_check/rtabmap_input_graph.txt`).
  It is a matched subscriber of `/d555/color/image_sync`, `/d555/color/camera_info_sync`,
  `/d555/aligned_depth_to_color/image_sync` and `/scan`, each with one publisher. Its only input
  with no publisher is `/strafer/odom`, which comes from `base`, and `/scan` is silent without
  `base` (above). So it cannot map on this rig state.
- **Bag tooling.** `bag_frame_numbers.py` on an 11.5 s bag: 346 depth messages, 2 consecutive
  duplicates, **344 distinct header stamps = 344 distinct frame numbers**, 0 gaps, `global_time`.
  The 2026-09-25 pose-1 bag gave 2380 / 1190 / 1 stamp.

## What persists, and what does not

- **Modules.** They are under `/lib/modules/5.15.148-tegra/updates/strafer-d555/`, which `depmod`
  ranks above the in-tree drivers. They load by alias when the camera appears.
- **`iio-sensor-proxy`** is masked. The desktop service would otherwise claim the accelerometer.
- **Container access.** The `perception` service's IIO access is in the tracked compose file.
- **Not yet checked: a reboot.** The procedure is: unplug the camera, reboot, check the clock,
  replug, then confirm `/sys/module/uvcvideo/srcversion`, the HID binding and
  `systemctl is-enabled iio-sensor-proxy` (`masked`).
- **Kernel package upgrades.** An `nvidia-l4t-kernel` upgrade within R36.4.x would **not**
  remove these modules. If the new kernel keeps the CRCs of the symbols they import, the rebuilt
  `uvcvideo` keeps shadowing the new kernel's; if any changes, it fails to load and `modprobe`
  does not fall back to the in-tree module, so the camera gets no `/dev/video*`. The procedure
  (`docs/D555_IMU_KERNEL_FIX.md`) says to remove them before upgrading.

## Deviations

- **The compose change preceded the decision commit.** `perception` was given `c 248:* rmw` and
  a writable `/sys/devices` and restarted at 02:29:57Z (`prechecks/perception_composechange.log`).
  The decision was committed at 02:32:26Z, about 2.5 minutes later, and `route_decision.md` says
  no compose change had been made, which is wrong.
  - Every pre-check ran 01:57–02:08Z, before the change, so none of them is affected.
  - The stage-1 read-back, taken with the change in place, still shows items 1–3 broken.
  - No raw-depth stamp capture falls between 02:08:15Z and the stage-1 read-back at 02:53:36Z.
    In that gap come the compose change, the stage-0 read, the HID modules, the
    `iio-sensor-proxy` mask and the USB re-enumeration, so stage 1's change from 1 to 5 stamps
    cannot be attributed to any one of them. The 5 values are four steps of a still-constant
    stamp, not a per-frame advance.
- **Root steps run remotely, without a replug.** The root scripts were run over SSH in the
  pre-registered order (stage-0 read, stage 1, read-back, stage 2, read-back), each stopping at
  its first failure. The camera stayed attached throughout.
  - At stage 1 a USB re-enumeration through the device's `authorized` attribute replaced the
    physical replug.
  - At stage 2 the module swap re-probed the device.
- **No reboot.** A remote reboot with the camera attached is not done on this rig, and nobody was
  on site to unplug it.
- **Images not rebuilt** with the launch change: root has 4.4 GB free, and a CPU image rebuild
  needs about 4 GB. The passthrough was verified by swapping the relay in the deployed container.
  Until `make images` runs on the robot, the deployed relay still restamps, and the point cloud
  and `/scan` stay empty on the real lane.
- **Skipped pre-check.** A pre-check restarting perception at `LRS_LOG_LEVEL=INFO` was not run;
  the direct `D4XX` read replaced it.
- **Rollback script corrected after the run.** `maintainer/rollback_remote.sh` in `1743ee5f`
  removed the modules before unloading them, which fails once `depmod` no longer indexes them. It
  also unloaded the HID-sensor modules while the attached camera's IIO devices still held them,
  and ignored that failure. The deposit's `ff5d74cf` replaces it with the order in
  `docs/D555_IMU_KERNEL_FIX.md`. Neither form has been run. The first form is recoverable only
  from commit `1743ee5f` in the evidence repository.

## What this record does not claim

- **That the change survives a reboot**, or a cold boot with the camera attached.
- **That the inference node infers on the real stream.** It has no goal, TF or odometry here; that
  needs the chassis.
- **Exactly one message per frame.** Three in 1804 repeat, by the mechanism above.
- **`/scan` or RTAB-Map mapping.** Both need `base`.
- **That the deployed images pass stamps through.** They need rebuilding first.
- **The 2.5–3.5 m texture pose.** It needs the camera at robot mount height, on site.

## Evidence — deposit

| | |
|---|---|
| repository | https://github.com/zachoines/Sim2RealLab-Artifacts |
| deposit directory | `d555-stream-integrity-2026-10-04/record-files/` |
| deposit commit | `2ea7baab462c438cb8116ba47cafff388aad20b7` |
| earlier commits | `f462601c870ef23b41e8d39458c722ce5a3b0cdb`: the route decision, pre-checks, tools, on-site scripts and build digests, before any module change; `1743ee5f14ba9832645bcae8e686e445b908fd6f`: the post-change read-backs and the remote scripts; `ff5d74cf7a4bee45a318dd96dd634db34182b48a`: the passthrough and `/scan` re-checks, the build checks and the corrected rollback script |

Restore into this record's directory with:

```
git clone git@github.com:zachoines/Sim2RealLab-Artifacts.git
cp -a Sim2RealLab-Artifacts/d555-stream-integrity-2026-10-04/record-files/. \
      docs/measurements/d555-stream-integrity-2026-10-04/
```

Verify digests with:

```
cd Sim2RealLab-Artifacts/d555-stream-integrity-2026-10-04/record-files
grep -E '^[0-9a-f]{64}  ' DEPOSIT.md | sha256sum -c -
```

sha256 of every file in the deposit:

```
e16bc0e361c198b61abf6e277744c5ec8ab674810433e0ee4c25ee1438622467  build/SOURCES.txt
39a1d73d3d5af45e2cca56a47a1573fd98c590c04e770115158b4d62cf9c8af4  build/build_modules.log
50ca11127bbf9d498fad5d2707145e8940fbf5e74ded896298d7582409d85215  build/modules_sha256.txt
b3d0a0c3b38a5a109855de71a13e320fa571a20c57cf6201d191fb4d94f63d60  build/realsense-metadata-jammy-master.patch
f604fc5a2329e4cf32dc0b064a83d0f6f5621f156be64d0fa03e008f4ad090b5  build/sources_sha256.txt
8890655d1896c1a5af3e31453a955a8b155c377bfb78de89a953c5a7920a39f0  checks/check_env_sync.txt
4c8b97d0fe950ccbcb01683797a840d8f55d2b1748d7147a804f8f72ca12ee2a  checks/make_test_ros.txt
1e0c827f05c7b62fe9b74aaccf7f4691c6e49f582d374c904098cd182a8f10d3  checks/make_test_ros_3983c41.txt
050152e2b043f1668bf6a2be63a48b3a7ba1ff1a248e0e1da2760a97e8028c99  maintainer/remote_common.sh
658e2793be0cf09bb31883b17e9405019804778f7a0e9b021a0e5ba4cfb333cd  maintainer/rollback.sh
8480be8ff8cd6c8cf259f72d48cade11fbad05010718bf38ffa90bcdb632bd8d  maintainer/rollback_remote.sh
456395f27221869d10be2a55279b8306e045890fe2fba1a99a6ea61e0b3a431b  maintainer/stage0_stats.sh
42cffde849fec16b9d894c10ad54ff367a9eb1e0e615ff20bc9640e4047c5be5  maintainer/stage1.sh
d76eb0d6fa5aede223f36467b1b8d7742cccf41c0e711bb4bbd3a25f1f6db647  maintainer/stage1_remote.sh
753c0af84745d458b72c778329fc6270106a82daa84a1e06bc01f4b34ab108b0  maintainer/stage2.sh
387e04058b023716e5ad11526c1bb6202bcfc39ba65d5b0a8a187ad642c01dee  maintainer/stage2_remote.sh
4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945  prechecks/baseline/aligned_raw.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  prechecks/baseline/aligned_raw.txt
4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945  prechecks/baseline/aligned_sync.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  prechecks/baseline/aligned_sync.txt
0999ba946d71e21c2c68254a06d1a3eb5f477ecda35b18e6f6c8406834ed557c  prechecks/baseline/aligned_topic_info.txt
f224b1eb97b3f938d364ce1a82c4436c5a1f4c8dde175dbf9cbb9dfc50da2705  prechecks/baseline/color_metadata.txt
0a6cd34247fdbbea894f46cc9aac3a7009c8dd74de6cc036c372d6e07778e822  prechecks/baseline/color_raw.json
6b84d726b6010ba6c021a7cdb4ecbbf8d4803b2e89cb2364ac75d39a40bac5b5  prechecks/baseline/color_raw.txt
583a9aebb70794a3163cf64feb7cd647f1e3c7ee9f715d9618b29cd6038e42f5  prechecks/baseline/depth_metadata.txt
f9c4299efc2d436187b867e3f5c7e20018b611f3420fdca3dbdde74435176610  prechecks/baseline/depth_raw.json
941be425527e3ecf3cb3760a8fef33cefdd9ea28dcd61586393720f5c9092fbd  prechecks/baseline/depth_raw.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  prechecks/baseline/iio_devices.txt
95d66dff1372722210103cc7d6b6e477d5510f151de34cd2692a80968235fafd  prechecks/baseline/imu.txt
f198df9495dfda1d1c7a3c916545031f289a967357bc3e25cbfe667b4b001742  prechecks/baseline/params.tsv
a7854cd6f91ccf76b45939e08f9e35c2e6f20098bc32d34ebcdc29260d34b4dc  prechecks/baseline/probe_sha256.txt
2110547e099247fafa70b99451ba538f674df3a8c6a4eecceff3c149080215c3  prechecks/baseline/t0.txt
c63e27f5e61900a33f5905820d79b95c6d5660bca6a8347dc0bde8716ae5cf38  prechecks/baseline/t1.txt
60d929bc68ab8f365bea639eb849f2edc27edcc745bc97e2369a6b87c97bfda8  prechecks/meta_formats.txt
8277e932726041838dbeb11b1c6e868065bb7063592d121b0af8695e36b1e643  prechecks/perception_baseline.log
b24de5a19afa2133f817d50f6922a88c2023e970f68a8a14bb4fb26b304f55ed  prechecks/perception_composechange.log
746a909bd42520d2584b4c9ef5b147b61d5084af53d09a2b2691c67ed59c66cf  prechecks/perception_up_t0.txt
e18725fb97aa0e3ee033c1f6e0c50a5649c554e010aec349819b4813c9d914d6  prechecks/r05_align_toggle/depth_align_off.json
a045332388a3da7b0a0ef8841526952d3ef29f8322507dd553ba1426f843856a  prechecks/r05_align_toggle/depth_align_on_again.json
7d57f4c1ffb426bb53668429e850cf08321e4b5a27706909bf84fff9023f0268  prechecks/r05_align_toggle/r05.txt
005cb200720133030515e5a61fda7bfa1041e15b9cdeee74b6f9ef16c495a21b  prechecks/usb_interfaces.txt
4c75346a6c28e8e43264c5e60094551791dad7824e4b1e187bc2930e2d7b6704  prechecks/uvch_color.json
9dd5c5d7f90dcf0bc0c3cd4072b7c111e8c0cb86d74198e0d9ffa4a5f54877d7  prechecks/uvch_color.summary.txt
a16db252a712e2e2b93ec0731b519524092e24dd313f37aa1ca1911654fcbe97  prechecks/uvch_depth.json
4d527770c52323179b23995a0e6620c88585b8ecbec78b26d587c3d4d1a97c4f  prechecks/uvch_depth.summary.txt
e3768630558b90e1003be14713855dc1cb1d08ac5cc9fef459b256e8a0923e72  readback/build_checks.rerun-20261005.txt
e3768630558b90e1003be14713855dc1cb1d08ac5cc9fef459b256e8a0923e72  readback/build_checks.txt
bd41877e94f16bab74067699d8bd03ba340a2703383fadc54160b3a34b95ff04  readback/build_raw/crc_compare.tsv
bc0b993d9d322320f5915fd23e8812d0ec62d0e475c6880c556018f4d9cd0487  readback/build_raw/crc_summary.txt
614551f2f3c29d6c7f868bbfacc1ab9cacd6b8a8974252557e86cf2db96db50a  readback/build_raw/digests.txt
90b45c915aaea7c4d12196db544fb58617c301e70e908b1f82983adbe114ddb9  readback/build_raw/modinfo/built_hid-sensor-accel-3d.txt
823fcecbd09e57f9a49996d1dd64d2021f70a2aa315f4995149bdbfdca567700  readback/build_raw/modinfo/built_hid-sensor-gyro-3d.txt
9100034fde7738be6df8b87cf8d881ca6552f6d3217fa23165d060c88da2ca7f  readback/build_raw/modinfo/built_hid-sensor-hub.txt
2154084ff47b919987174354895010f30f281ea27c22bd91e4a692c8ab46661a  readback/build_raw/modinfo/built_hid-sensor-iio-common.txt
7273f0bda9e46093540165f66dc31bdf7997354c2a7e08228c313729a4f82240  readback/build_raw/modinfo/built_hid-sensor-trigger.txt
d4477ee915c32cb8996c1d4979e85c9c21307fbfca58f293b3dbac5c0e3d1282  readback/build_raw/modinfo/built_uvcvideo.txt
9fe4232704133f9c44f7d2802dbf5baf1ad9d8a4eba02993681585695f7f836b  readback/build_raw/modinfo/stock_uvcvideo.txt
ad9b96985aed0add7ee0ef6cdba29efdd8e877aa165113abe43c5865bd0a0a81  readback/build_raw/modinfo/unpatched_uvcvideo.txt
ca8151c2f7f59b7ef583147f31b5811b0b0fdcee9100906a481c88466d6d64d9  readback/build_raw/modversions/built_hid-sensor-accel-3d.txt
ca8151c2f7f59b7ef583147f31b5811b0b0fdcee9100906a481c88466d6d64d9  readback/build_raw/modversions/built_hid-sensor-gyro-3d.txt
af8de114d6752003e600be8ee0041ffbd45f63844529e7802d4cf12ea477308d  readback/build_raw/modversions/built_hid-sensor-hub.txt
43dc8b4f9e20cdc820108c3a6c66479e4d6e6130c6ab7da576d4aef34a2da64f  readback/build_raw/modversions/built_hid-sensor-iio-common.txt
1feec0cc78975ebceee1f51c4046009cca422ed2885a27ab928147254c5594e8  readback/build_raw/modversions/built_hid-sensor-trigger.txt
ea1c45c91713c771633efbd51a200e2d525c41df3a30612378c6c0ab9e357c59  readback/build_raw/modversions/built_uvcvideo.txt
ea1c45c91713c771633efbd51a200e2d525c41df3a30612378c6c0ab9e357c59  readback/build_raw/modversions/stock_uvcvideo.txt
ea1c45c91713c771633efbd51a200e2d525c41df3a30612378c6c0ab9e357c59  readback/build_raw/modversions/unpatched_uvcvideo.txt
a4ac132ab0eee7239521db176771793d4282f50010c150dd9c3b31876d9e9513  readback/passthrough_check/color_300s.json
5d3d188ee2b6f5f382b9be7bd67e388558eb7cdeac34af30335bc4e73c560987  readback/passthrough_check/color_300s.txt
2a759a6e3e2e0fb388cbdd2535a77255daf9b52082a854d384dc1f81b88b824b  readback/passthrough_check/compose_perception.txt
ef7d4d6653a0dfff5c427c706131a542eee67490c31f53171c83d8af19d5770d  readback/passthrough_check/compose_slam.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  readback/passthrough_check/global_time_rebase_lines.txt
d99bf611a85b5211cb38ff2bd860d0be58ea973f7389377b1d2861188c34bd43  readback/passthrough_check/perception.log
10eb63d54992995c1a44b5d94331089ead333320f8c8476eefce5430d8c8082f  readback/passthrough_check/points_hz.txt
78e4796f176d1a9dc2df394376a4eddd0fc45a8e9d71cdfbc17ded4f285d35aa  readback/passthrough_check/relay_passthrough.log
3e8955a8dc5dcb944c7ab9b8bda2edfd27a7f23de9d2e85aa540d8ecce823a28  readback/passthrough_check/rtabmap_input_graph.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  readback/passthrough_check/scan_hz.txt
624cb4f37500d90c93ae9ca274c8aea983cc751bec5d24bb53656b8a970dc394  readback/passthrough_check/slam.log
9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa  readback/passthrough_check/sync_warnings_count.txt
945a4fa398f3ca320d95bd7274e07d21c89ff3aa210e51f36f988849e98171d0  readback/passthrough_check/t0.txt
e46788d6b7ca9bb84f73798f0269322e7c441cf8811759d47fcc8974a191d564  readback/passthrough_check/t1.txt
28ecb872ffcae37c9a911de7f74c412b9e23b8a6ca5b0267decab4d34410d377  readback/passthrough_check/t_relay_swap.txt
ed0f61e6f6796d3d9f1ec1eb3851c8743e8b78c793741b0b4ba541e9e8a0313c  readback/passthrough_check/tf_filter_drop_lines.txt
767e57a853a83323fcb98fec32c634e563b080648c95e9c1dcd8a4b824ee10b1  readback/passthrough_check/tool_sha256.txt
8277e932726041838dbeb11b1c6e868065bb7063592d121b0af8695e36b1e643  readback/perception_baseline.log
b24de5a19afa2133f817d50f6922a88c2023e970f68a8a14bb4fb26b304f55ed  readback/perception_composechange.log
460dc97484c64b3c36ac9af32791cec683b9b7cf55dff945cd4f0099630ead21  readback/perception_stage1.log
e253eaec07251d93c81df512dab1005de49880129319f61f82a2f6db686c9a47  readback/perception_stage2.log
746a909bd42520d2584b4c9ef5b147b61d5084af53d09a2b2691c67ed59c66cf  readback/perception_up_t0.txt
2a759a6e3e2e0fb388cbdd2535a77255daf9b52082a854d384dc1f81b88b824b  readback/scan_check/compose_perception.txt
ef7d4d6653a0dfff5c427c706131a542eee67490c31f53171c83d8af19d5770d  readback/scan_check/compose_slam.txt
41fd2a3c17cb2b0b379736868e319e610ddae6f5c348a1feebea2b58faec8e00  readback/scan_check/compose_stop.txt
8e7cf3e86223cb7e01c0e3e58b0fe982ca0acc4be4f28a491b68c7f05ea411e2  readback/scan_check/laserscan_drop_reasons.txt
00604c3a8f26f707dac8a1c7c96889a3e2e012df90171953872d28c8bd4dc7f9  readback/scan_check/perception.log
7f6b519c67a98279820d8209a3261d06d6be5ced184a769815f0da2e293bfdfa  readback/scan_check/points_hz.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  readback/scan_check/scan_hz.txt
5f8fea14bfa446f2e1671fa619f73040902b7feb6d8b2a43860d33b09161b140  readback/scan_check/scan_info.txt
cd920d92d8c5c86e6584f108e6a600f467b811edd4f11e1db1552847e8e3c3b2  readback/scan_check/slam.log
2a366bcb6c46097b882d702e9d68de56c0db742bbfd08b0a0279b0e3b0f0aa98  readback/scan_check/t0.txt
8750bb8d8467034ac8fc7a79844371ec36e70a8a27580806af3738a1c7f588a6  readback/scan_check/t1.txt
f1b56eff300d6b8db7352fcb404aaa937867c1e5993c0a534976205bd695768d  readback/scan_check/tf_base_link.txt
b3208cd31a3a54bf8016237bce6c7502774cf615c84bbc190b97ab1a088092a4  readback/scan_check/tool_sha256.txt
9d70d6d0259f8195497fdacfb030ce93ae40d0aab40cd13efd1a936cf7672ab1  readback/stage0_stats/uvch_depth_root.json
2ebbef13ef4b10c3bd0e858e21aecafe5588e4aa4360689025f0dfa85ac2c130  readback/stage0_stats/uvcvideo_debugfs_stats.txt
4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945  readback/stage1/aligned_raw.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  readback/stage1/aligned_raw.txt
4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945  readback/stage1/aligned_sync.json
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  readback/stage1/aligned_sync.txt
0999ba946d71e21c2c68254a06d1a3eb5f477ecda35b18e6f6c8406834ed557c  readback/stage1/aligned_topic_info.txt
5e47f913a666ab381c6315668cd9758bff331387eaa9c4cf3ce271ddc9403be0  readback/stage1/color_metadata.txt
584a74e9a18fd4727bd18ac052c299a8784d7f8ca3229068d4d18af0a307325f  readback/stage1/color_raw.json
92d95c2d79003bcdccc174bbb0910b0d95b1d84afb4cb1f93ab593a8d8d78342  readback/stage1/color_raw.txt
89f7263c1f65641d1dcc098ff88cb85b84ba047d41935d10a5190974d89964cb  readback/stage1/depth_metadata.txt
f2083cd3a689c3fe92a405ac64e17d3a8ecf2ed9c3458fe6b170ed4123c1f4f4  readback/stage1/depth_raw.json
215097d0e30ab4f413cf8629b500cf3209a48bf1c970a5ebc7d7e2d749859b89  readback/stage1/depth_raw.txt
cc3dd15ef50b7387e67699c066d9f299db81c39462ba8271042d27f80057ae6d  readback/stage1/iio_devices.txt
136cb686efd0fe2c9840ba3d02e8f6c7ba0c528d6767d974287b31c44aef0331  readback/stage1/imu.txt
18fa69b4d4ed6ac34ed0d6f9c0b7900bc840828d4798376af0d95fb085cfd63f  readback/stage1/params.tsv
a7854cd6f91ccf76b45939e08f9e35c2e6f20098bc32d34ebcdc29260d34b4dc  readback/stage1/probe_sha256.txt
d48472d2893bb8cf531df909f0cf7ad2dc819c3136798d919448ca9fc2f1eff9  readback/stage1/t0.txt
0cef030b0314b47ea736f58c1566531bfd35a94224f0c5437ea3260724e7002d  readback/stage1/t1.txt
801b015b952950776f3a89cfb9f799209206b1fbdea1ae38f8a0962ccd07bd32  readback/stage1_remote.out
e1119842e280ba823059dea6c36c811747cb2f71c6f0444699d2901be9e80eaf  readback/stage2/aligned_raw.json
45a931e591afdd3c89ef8693569488257cea26f851b6040433874e628d8eac48  readback/stage2/aligned_raw.txt
4d75191fed29db69d62f5ed7be05f2632d347b71f13334e7a574839a6195801b  readback/stage2/aligned_sync.json
3ada0fd2fc3bfbb58b2eb697050ad0b4a11ff1319ccf364d1c523ed9717ee857  readback/stage2/aligned_sync.txt
0999ba946d71e21c2c68254a06d1a3eb5f477ecda35b18e6f6c8406834ed557c  readback/stage2/aligned_topic_info.txt
e5c59fa112f307e0b4de04c5ada12b4da1b7318c2a1604d094dbe5847792e659  readback/stage2/color_metadata.txt
58482fc32538a3a3ff286c058f5f1674f9761b4d4bbace6c99eefd6f2aa70642  readback/stage2/color_raw.json
a66e465f9f6df437feeb2246f239c1d02d9a472bc521536cde2877c8d8d06858  readback/stage2/color_raw.txt
e94b2b7465ff0492e958956a499adffcdb52241ef7b82863b70a4f3298cf7bf9  readback/stage2/d4xx_color.json
6d31e1916539514bb664f330f04ab06a7cb9d8f829d6ba680214843cec9ae10a  readback/stage2/d4xx_color.summary.json
11d9c82ea22f222fe1e00232f39d406f3e7260b6b0ff6fa5147ddce425386851  readback/stage2/d4xx_depth.json
966d7223aed3f720295cfe583b711cbbf6ab3a6f5ae0410b4bf9fa7a4be242cf  readback/stage2/d4xx_depth.summary.json
ad2e9149a6df56a85c8a437a435d9d308f87e95962dd47b84d926be17155202b  readback/stage2/depth_metadata.txt
d126adb47aa307375474493e68c485fe40ebbe52e842703493d2ca3bc4f1a325  readback/stage2/depth_raw.json
bdb866ce832222fad8cbcae4d70d973e5d322528c51678993f8cfe82034f1909  readback/stage2/depth_raw.txt
cc3dd15ef50b7387e67699c066d9f299db81c39462ba8271042d27f80057ae6d  readback/stage2/iio_devices.txt
983a6d1e19f3fc2c7af4fcb9e05e0791c1a2c181bcb96a78bcb5bc64a63c8afa  readback/stage2/imu.txt
2d06c6e6976382c53167fd5353df4100603ee0a7be5e6cf77328cc83ea4b4f46  readback/stage2/meta_formats.txt
18fa69b4d4ed6ac34ed0d6f9c0b7900bc840828d4798376af0d95fb085cfd63f  readback/stage2/params.tsv
4993676b03584e91796595bd46cb0056342071fd54a7cbc3ef5e40f2a7eef95a  readback/stage2/perception_up_t0.txt
a7854cd6f91ccf76b45939e08f9e35c2e6f20098bc32d34ebcdc29260d34b4dc  readback/stage2/probe_sha256.txt
d9a82c8af33263c11be83ef6d792a0f0680df93d9923234dcdfa60acb3e46250  readback/stage2/t0.txt
1c27e8e97981889db549467c6557c8fab1572e10bd3d533357571b416fb0a01b  readback/stage2/t1.txt
52319520937eb4614bfc5feb7358f67e90c952e32ac7dadd0de897c7d51c314d  readback/stage2_remote.out
7a36fc669b1a3005cbba7673f387992d99e1372f706691876cfa5210d93b8ad7  readback/stage2_supp/aligned_raw60.json
e5f888aa0677171b41a8515053dfc2d1e15f9bff9564928b3fd25eda45d1ba46  readback/stage2_supp/aligned_raw60.txt
1d1ba9c6b0c78caba55cc9216f93c3d2789c1b9d5e3a4c17a2b84287c9c4a5df  readback/stage2_supp/aligned_sync60.json
b85572414d8584ef2825ce84358644f4a43b86b9de1a7a9c835fabbb191556b7  readback/stage2_supp/aligned_sync60.txt
1575781bb38d6a1dcf7ae07b54e2fd94ae4809bac232e5f847a8d057976acdac  readback/stage2_supp/bag_frame_numbers.txt
5cc25663d024d3798f9d1d84ed66b8826ab51b9abbba71f7f55c32ba8eb36f15  readback/stage2_supp/imu60.json
71cea0f943aa862be97f718c317c89a71d9f228f501d85f7ed11e6564cd41432  readback/stage2_supp/imu60.txt
99e708b7c13539fbab43be2d2a4f2b7722cf52e2cda2fc400807ca5303fe1294  readback/stage2_supp/imuf60.json
a82507f047bce841d3914a463d99bc5cfc3320695711ebcab27a9b4a1c7e6fd0  readback/stage2_supp/imuf60.txt
8d0576dc5b89590a2033e0fa3224caaf1eb0c6229da3961f843b6555fd28e187  readback/stage2_supp/rtabmap_inputs.txt
e5bc9d6178d93bd7142f179e672dade6f3929cc8d29e8703b8f1f51c65a87e51  readback/stage2_supp/rtabmap_missing_inputs.txt
f03c16058755b41a00e2d65b0386d03fdb5898f9cf027c08b7b550db85b9fc6c  readback/stage2_supp/strafer_inference.log
d07b240c6ff3533e1815994579852a6d6b8ec5f6a8795e46f9ea7e72da5082f4  readback/stage2_supp/strafer_slam.log
f4367f5858c56991846388217c5cf4e65afb50e4d2bc889a77e1e9b23cd0a711  route_decision.md
7a744e6a812afbe14c025610e93bd725436d954aad25921b031cf59567aa52b1  tools/build_checks.sh
89008ad141807fd96414806405bc28395d38fbae8618aacc9c4873f7f496c730  tools/build_raw.sh
b328bb8afb0e555fcf0cd2445c2264ba1bd318876985856ab8a72a71440dea74  tools/depth_stamp_probe.py
554d530a008c6da82c2586f78c04e2d190da97620f3cdb9edf2818629164d310  tools/imu_stamp_probe.py
a2e34e877a72bdeea13d756024e58ea47c9300c91cc3642c1afbc4e094a58926  tools/meta_fmt_probe.py
f3d72f63a8cc63f094ec6c9121bbc2bde71b12fd95bd1a16649cbcc58cd00f6f  tools/metadata_probe.py
7a4c8db0d1ed96cb25cc5d60be3748246da3c956a89aea37445157a7925e6c1c  tools/passthrough_check.sh
c1ab2e819e23ae585a4220698de991e1eaede949e37e3b9af3eef821025ab687  tools/r05_align_toggle.sh
5e9daa0f10d5c35718746470aa58a21d7f029661a6adced7e290601df6e5d4df  tools/readback.sh
10d92cf29ee5edcc07bf111aef92fdb3c34e85510e4bb1bcc22ffa15f3c56e40  tools/record_figures.py
71f0c70bb26e8f5df00fd7310d87fac46fd52b61655559851a31dccb6fa1c5e6  tools/scan_check.sh
cebe3240370339e89484e74c2d854f9167366f5ae477dabf32499a1001b9dbf5  tools/snap.py
39260133c6d877af004e87c65eb96b2f56cd7c8a9c5456e36ece6efb465f4a85  tools/uvc_header_probe.py
48aecb9d5855a0709bf556570bfdc2bdb784c270bd8e264c443a95d88c8cd1bf  tools/uvc_header_probe.v1.py
```
