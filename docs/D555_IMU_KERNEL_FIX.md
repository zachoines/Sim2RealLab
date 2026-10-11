# Enabling the RealSense D555's IMU and per-frame metadata on Jetson (Tegra kernel)

Two pieces of the D555 never reach librealsense on a stock Jetson kernel:

1. **The IMU.** The Tegra kernel ships without the HID-sensor drivers, so the
   accelerometer and gyroscope never become IIO devices.
2. **Per-frame metadata.** The kernel's `uvcvideo` has no device entry for the
   D555, so the camera's metadata block never reaches the driver. Without it the
   frame stamps freeze, every depth frame is published twice, and aligned depth is
   never produced.

Both are fixed with out-of-tree kernel modules for the running kernel: five HID/IIO
modules for the IMU and a rebuilt `uvcvideo` for the metadata. No kernel rebuild is
needed. The perception container also needs IIO device access, which the deploy
compose file already grants.

## Platforms

| | first applied (IMU only) | strafer-nx, 2026-10-04 (IMU + metadata) |
|---|---|---|
| Board | Jetson Orin NX | Jetson Orin NX 16 GB (Seeed image) |
| L4T | R36.5.0 | R36.4.3 |
| Kernel | `5.15.185-tegra` | `5.15.148-tegra` (Seeed build; NVIDIA headers are ABI-compatible) |
| Module sources | Ubuntu `linux-source-5.15.0` tarball | the kernel's own tree: GitLab `nvidia/nv-tegra/3rdparty/canonical/linux-jammy`, tag `jetson_36.4.3` |
| librealsense | 2.56.4 (deb) | 2.58.4 (deb, kernel V4L2/IIO backend), realsense-ros 4.58.4, in `strafer-cpu:humble` |
| Camera | D555, USB PID `0B56`, FW 7.56.19918.835 | same, on USB 3.2 |

The record for the second column is
[`d555-stream-integrity-2026-10-04`](measurements/d555-stream-integrity-2026-10-04/README.md).

## The problems

### IMU: no HID-sensor stack

The D555's Bosch BMI055 IMU is exposed as an HID Sensor Hub device (USB interface 5,
`bInterfaceClass=3`). On a kernel with `CONFIG_HID_SENSOR_HUB=m`, `hid-sensor-hub`
claims that interface and its accel and gyro drivers create IIO devices, which
librealsense's `iio_hid_sensor` backend reads.

The Tegra kernel has `CONFIG_HID_SENSOR_HUB` unset, so the interface binds
`hid-generic` and no IIO device appears. The symptoms:
- `/sys/bus/iio/devices/` is empty;
- librealsense logs `No HID info provided, IMU is disabled`;
- `/d555/imu` has no publisher.

`CONFIG_IIO`, `IIO_BUFFER`, `IIO_TRIGGER`, `IIO_KFIFO_BUF=m` and
`IIO_TRIGGERED_BUFFER=m` are present. Only the HID-sensor drivers are missing.
Reading `hidraw` does not help: librealsense reads the IMU only through IIO.

### Metadata: `uvcvideo` has no D555 entry

The D555 sends a **248-byte UVC payload header** on every frame. It carries a PTS
that advances once per exposure (about 33.3 ms at 30 fps), an **all-zero SCR**, and
Intel's metadata block (hardware timestamp, exposure, gain, laser power). The stock
`uvcvideo` alias table has no `8086:0b56` entry, so its metadata nodes offer only
`UVCH`, the standard 12-byte header. Under `UVCH`:
- `uvcvideo` writes an entry only for a payload whose SCR changed. The D555's SCR is
  always zero, so **every metadata buffer comes back empty**.
- librealsense 2.58.4 sizes metadata as `bytesused − 10` in a `uint8_t`, so an empty
  buffer reads as 246 bytes of zeros. The frame timestamp is 0, so the stamps freeze:
  one stamp for minutes on `global_time`, 0.0 on `hardware_clock`.
- With every frame stamped alike, realsense-ros's syncer, which is on because
  `align_depth.enable` is a filter, never pairs depth with colour. Each lone depth
  frameset is published twice, so raw depth runs at 60 Hz, and aligned depth, made
  only from a paired frameset, never appears.

Turning `align_depth.enable` off at runtime confirmed the last step: raw depth then
runs at 30 Hz, one message per frame, with the stamps still frozen.

librealsense's own L4T patch,
`scripts/realsense-metadata-jammy-master.patch` (v2.58.4), adds the D555 entry
(`0x0b56`, VideoControl protocol `UVC_PC_PROTOCOL_15`,
`UVC_INFO_META(V4L2_META_FMT_D4XX)`) and raises `UVC_MAX_STATUS_SIZE` from 16 to 32.
With it the metadata nodes offer `D4XX`, which copies the whole header whenever it
is longer than its standard part, and librealsense receives the metadata block.

## The fix: six out-of-tree modules

Everything below builds as a normal user. Installing and loading need root. `K` is
the running kernel and `B` a build directory.

```bash
K=$(uname -r)                  # e.g. 5.15.148-tegra
B=$HOME/d555-modules/src       # any build directory
```

### 1. Headers and sources

The kernel headers come with JetPack (`nvidia-l4t-kernel-headers`), at
`/lib/modules/$K/build`. Two checks before building:
- The headers' `include/linux/hid-sensor-hub.h` and `hid-sensor-ids.h` must exist. On
  R36.4.3 they ship; on R36.5.0 they had to be copied in from the Ubuntu source.
- If the running kernel is not NVIDIA's own build (strafer-nx runs a Seeed build),
  confirm the headers match it. On strafer-nx, every symbol CRC the new modules import
  matches the running kernel's `__kcrctab`, and the stock `uvcvideo` imports the same
  155 symbols as the rebuilt one.

Fetch the sources from the kernel's own tree. For R36.4.3 the tag is
`jetson_36.4.3`; raw files download without credentials.

```bash
BASE=https://gitlab.com/nvidia/nv-tegra/3rdparty/canonical/linux-jammy/-/raw/jetson_36.4.3
for f in drivers/hid/hid-sensor-hub.c drivers/hid/hid-ids.h \
         drivers/iio/common/hid-sensors/hid-sensor-attributes.c \
         drivers/iio/common/hid-sensors/hid-sensor-trigger.c \
         drivers/iio/common/hid-sensors/hid-sensor-trigger.h \
         drivers/iio/accel/hid-sensor-accel-3d.c drivers/iio/gyro/hid-sensor-gyro-3d.c \
         drivers/media/usb/uvc/{Makefile,uvc_ctrl.c,uvc_debugfs.c,uvc_driver.c,uvc_entity.c,uvc_isight.c,uvc_metadata.c,uvc_queue.c,uvc_status.c,uvc_v4l2.c,uvc_video.c,uvcvideo.h}; do
  mkdir -p "$B/$(dirname $f)"; curl -fsSL "$BASE/$f" -o "$B/$f"
done
```

On R36.5.0, the same files came from `/usr/src/linux-source-5.15.0/linux-source-5.15.0.tar.bz2`
(`apt-get install linux-source-5.15.0`).

### 2. Patch `uvcvideo` with librealsense's metadata patch

Use the librealsense release that matches the image's deb, unmodified. On
`jetson_36.4.3` it applies with offsets only. Only the metadata patch is needed:
Z16 and YUYV already enumerate, so the formats and power-line patches are not.

```bash
curl -fsSL https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.58.4/scripts/realsense-metadata-jammy-master.patch \
  -o "$B/realsense-metadata-jammy-master.patch"
sha256sum "$B/realsense-metadata-jammy-master.patch"   # b3d0a0c3b38a5a109855de71a13e320fa571a20c57cf6201d191fb4d94f63d60 (v2.58.4)
cd "$B" && patch -p1 < realsense-metadata-jammy-master.patch
grep -n 0x0b56 drivers/media/usb/uvc/uvc_driver.c       # the D555 entry
```

### 3. Build

```bash
KDIR=/lib/modules/$K/build
echo 'obj-m := hid-sensor-hub.o' > $B/drivers/hid/Kbuild
printf 'obj-m += hid-sensor-iio-common.o\nobj-m += hid-sensor-trigger.o\nhid-sensor-iio-common-y := hid-sensor-attributes.o\n' \
  > $B/drivers/iio/common/hid-sensors/Kbuild
echo 'obj-m := hid-sensor-accel-3d.o' > $B/drivers/iio/accel/Kbuild
echo 'obj-m := hid-sensor-gyro-3d.o'  > $B/drivers/iio/gyro/Kbuild

make -C $KDIR M=$B/drivers/hid modules
make -C $KDIR M=$B/drivers/iio/common/hid-sensors KBUILD_EXTRA_SYMBOLS=$B/drivers/hid/Module.symvers modules
make -C $KDIR M=$B/drivers/iio/accel KBUILD_EXTRA_SYMBOLS="$B/drivers/hid/Module.symvers $B/drivers/iio/common/hid-sensors/Module.symvers" modules
make -C $KDIR M=$B/drivers/iio/gyro  KBUILD_EXTRA_SYMBOLS="$B/drivers/hid/Module.symvers $B/drivers/iio/common/hid-sensors/Module.symvers" modules
make -C $KDIR M=$B/drivers/media/usb/uvc CONFIG_USB_VIDEO_CLASS=m modules

modinfo -F vermagic $B/drivers/media/usb/uvc/uvcvideo.ko    # must match the running kernel
modinfo -F alias $B/drivers/media/usb/uvc/uvcvideo.ko | grep p0B56   # usb:v8086p0B56...ic0Eisc01ip01
```

The "compiler differs" warning is cosmetic when the version matches.

### 4. Install under `updates/`

`/etc/depmod.d/ubuntu.conf` (`search updates ubuntu built-in`) ranks `updates/` above
every other directory, so a module under `updates/` replaces the in-tree `uvcvideo`.
A directory the search line does not name, including the `extra/` the R36.5.0
procedure used, ranks equal to `kernel/` as `built-in`. When two copies rank equal,
`depmod` keeps the first one it finds, and that order depends on the filesystem, so
`extra/` works for modules with no in-tree twin but cannot be relied on to replace
`uvcvideo`.

```bash
D=/lib/modules/$K/updates/strafer-d555
sudo install -d $D
sudo install -m 0644 $B/drivers/hid/hid-sensor-hub.ko \
  $B/drivers/iio/common/hid-sensors/hid-sensor-iio-common.ko \
  $B/drivers/iio/common/hid-sensors/hid-sensor-trigger.ko \
  $B/drivers/iio/accel/hid-sensor-accel-3d.ko $B/drivers/iio/gyro/hid-sensor-gyro-3d.ko \
  $B/drivers/media/usb/uvc/uvcvideo.ko $D/
sudo depmod -a
modinfo -n uvcvideo hid-sensor-hub      # both under updates/strafer-d555/
```

### 5. Keep `iio-sensor-proxy` off the IMU

Once the accelerometer is an IIO device, the host's `iio-sensor-proxy` (a desktop
screen-rotation service) is started for it. It could take the IIO buffer that
librealsense needs, and a robot has no screen to rotate, so mask it:

```bash
sudo systemctl mask --now iio-sensor-proxy.service
```

### 6. Load: no reboot needed

With nothing streaming the camera (perception stopped), load the HID drivers and
swap `uvcvideo`. The camera may stay attached.

```bash
sudo modprobe -a hid-sensor-accel-3d hid-sensor-gyro-3d  # pulls in the hub, iio-common, trigger
[ "$(cat /sys/module/uvcvideo/refcnt)" = 0 ] && sudo modprobe -r uvcvideo && sudo modprobe uvcvideo
cat /sys/module/uvcvideo/srcversion                       # the rebuilt module's, not the stock one
```

If the camera was attached while the HID drivers loaded, re-enumerate it so its HID
interface binds `hid-sensor-hub`. Writing the device's `authorized` attribute
unbinds and re-probes it without cutting power:

```bash
P=$(for d in /sys/bus/usb/devices/*; do
      [ "$(cat $d/idVendor 2>/dev/null)" = 8086 ] && [ "$(cat $d/idProduct 2>/dev/null)" = 0b56 ] && echo $d; done)
[ -n "$P" ] || { echo "no D555 (8086:0b56) on the bus"; exit 1; }
echo 0 | sudo tee $P/authorized; sleep 2; echo 1 | sudo tee $P/authorized
```

The modules load by alias whenever the camera appears, so no `modules-load.d` entry
is needed.

### 7. Container access

librealsense writes the IIO devices' `scan_elements`, `buffer` and
`sampling_frequency` attributes, and reads `/dev/iio:deviceN` (character major 248 on
these kernels; check with `grep iio /proc/devices`). The `perception` service in
`source/strafer_ros/deploy/docker-compose.yml` therefore has:
- `c 248:* rmw` in its device cgroup rules;
- `/sys/devices` mounted writable over the container's read-only sysfs.

Without both, librealsense finds the Motion Module but cannot start it.
- **Scope.** The writable `/sys/devices` lets the root-run container write any host
  device attribute, not only the IIO ones. It is narrower than a privileged container
  or a writable `/sys`, which would also expose debugfs, configfs and module
  parameters.
- **AppArmor.** It works on these hosts because Docker runs there without AppArmor.
  Docker's default AppArmor profile denies these writes.

Recreate perception the same way the stack was brought up (`docker compose up` in
`deploy/`), so any host-local overrides still apply. The images must carry a
`timestamp_fixer` that passes stamps through (see Verification).

A non-root process on the host would instead need the IIO line of
`source/strafer_ros/99-strafer.rules`. The container runs as root and does not.

## Verification

Read back on the deployed launch, over windows of at least 60 s:

| check | expected |
|---|---|
| `meta_fmt_probe.py` from the record's deposit (`tools/`; QUERYCAP and ENUM_FMT on each `/dev/video*`, no root) | metadata nodes offer `['UVCH', 'D4XX']` |
| `/sys/bus/hid/devices/*:8086:0B56.*/driver` | `hid-sensor-hub` |
| `cat /sys/bus/iio/devices/iio:device*/name` | `accel_3d`, `gyro_3d` |
| perception log | `Starting Sensor: Motion Module`; no `No HID info provided`; `timestamp_fixer`: `TimestampFixer ready (passthrough)`, then `First frame relayed unchanged: stamp=…` |
| `/d555/depth/image_rect_raw` | ~30 Hz, header stamps advancing ~33.3 ms, one message per frame apart from one or two repeats at each depth/colour crossing, about every 29 s (see below) |
| `/d555/depth/metadata` | `clock_domain global_time`; `hw_timestamp`, `actual_exposure`, `gain_level` present; `frame_timestamp` distinct per frame |
| `/d555/aligned_depth_to_color/image_raw`, `.../image_sync` | ~30 Hz |
| `/d555/imu`, `/d555/imu/filtered` | ~200 Hz, stamps every 5 ms |
| `strafer_inference` cadence line | `imu` absent from `stale_sources` |
| `depth_to_pointcloud` (in `slam`) | no "do not appear to be synchronized" warnings; `/d555/aligned_depth_to_color/points` publishing. This needs `timestamp_fixer` to pass stamps through; an image whose perception log reads `TimestampFixer ready (restamp)` re-stamps and must be rebuilt. `/scan` also needs `base_link`, which `base` publishes |
| `systemctl is-enabled iio-sensor-proxy` | `masked` |

The record's probes and figures are in
[`d555-stream-integrity-2026-10-04`](measurements/d555-stream-integrity-2026-10-04/README.md).

## IMU stream profiles

| Stream | Rates available | Format |
|---|---|---|
| Accelerometer | 100 Hz, 200 Hz | MOTION_XYZ32F |
| Gyroscope | 200 Hz, 400 Hz | MOTION_XYZ32F |

The perception launch uses the unified IMU (`unite_imu_method: 2`): gyro at 200 Hz,
accel at 100 Hz, published together on `/d555/imu` at 200 Hz.

## Known issues

### Remaining duplicate depth frames, and the depth stamp's clock

realsense-ros publishes raw depth twice for any frameset without a colour frame. With
the metadata path this happens only at a beat between the two streams: on strafer-nx,
under realsense-ros with colour auto-exposure on, depth ran at 30.0000 fps and colour at
29.9655 fps (camera clock), so depth gains a frame every 29.0 s. The period depends on
colour's delivered rate, so it is not a constant. At each crossing librealsense's syncer, which pairs frames within
about half a frame interval, releases at least one depth frame alone, and sometimes a
colour frame too, which then gets no aligned depth. That gave 3 repeats in 1804
messages over 60 s, against half of all messages before.

realsense-ros stamps every message of a frameset with the frameset's time, which is the
colour frame's. A paired depth image therefore carries the colour frame's time, up to
±16.7 ms from the depth frame's own timestamp. A tool that needs the depth frame's own time reads
`frame_timestamp` on `/d555/depth/metadata`.

### The camera attached at power-on comes up without a driver

On strafer-nx the D555 came up driverless (PID `0bdc`, no `/dev/video*`) in 5 of 5
boots with it attached at power-on. Plug it in after the host boots, and unplug it
before a reboot.

### `initial_reset` causes USB errors

`initial_reset: true` triggers a reset that causes `tegra-xusb` transfer errors
(`xioctl(VIDIOC_QBUF) failed: No such device`) and repeated re-enumeration. Leave it
`false`.

### `HID set_power 1 failed` after an unclean stop

If perception was stopped while streaming, the IIO buffers stay enabled, and the next
start's attempt to enable them logs this warning. The IMU streams anyway.

### Kernel package upgrades do not remove these modules

R36.4.x kernels share one module directory (`/lib/modules/5.15.148-tegra`). The kernel
package does not own `updates/strafer-d555`, and its post-install `depmod -a` indexes
these modules again. After an `nvidia-l4t-kernel` upgrade:
- if the new kernel exports the same symbol CRCs, the rebuilt `uvcvideo` loads and keeps
  shadowing the new kernel's own;
- if any symbol it imports changed CRC, it fails to load (`disagrees about version of
  symbol` in `dmesg`). `modprobe` does not fall back to the in-tree `uvcvideo`, which is
  no longer indexed, so the camera gets no `/dev/video*`. The HID modules fail the same
  way, leaving no IMU.

To avoid both:
- **Before** upgrading, remove `updates/strafer-d555` and run `depmod -a`.
- After upgrading, rebuild against the new headers and reinstall.
- Alternatively, hold the `nvidia-l4t-kernel*` packages.

When the kernel version string changes, modules under the old directory simply stop
being used, and the procedure is repeated for the new one.

## Rollback

With perception stopped, in this order:

```bash
echo 0 | sudo tee $P/authorized       # 1. detach the camera ($P as in step 6): its IIO buffers
                                      #    hold the accel and gyro modules in use otherwise
sudo modprobe -r hid_sensor_gyro_3d hid_sensor_accel_3d hid_sensor_trigger hid_sensor_iio_common hid_sensor_hub
sudo modprobe -r uvcvideo             # 2. unload while the modules are still indexed
sudo rm -r /lib/modules/$(uname -r)/updates/strafer-d555 && sudo depmod -a   # 3. remove, re-index
sudo modprobe uvcvideo                # 4. the stock module
sudo systemctl unmask iio-sensor-proxy.service
echo 1 | sudo tee $P/authorized       # 5. re-attach; the HID interface binds hid-generic again
```

Removing the modules before unloading them fails: once re-indexed, `modprobe -r` no
longer finds them. A reboot with the camera unplugged also unloads everything.

## File inventory

| File | Purpose |
|---|---|
| `/lib/modules/<kernel>/updates/strafer-d555/*.ko` | the five HID/IIO modules and the rebuilt `uvcvideo` (R36.4.3 layout) |
| `/lib/modules/5.15.185-tegra/extra/*.ko`, `/etc/modules-load.d/hid-sensor-imu.conf` | the R36.5.0 layout (IMU only) |
| `/etc/systemd/system/iio-sensor-proxy.service` → `/dev/null` | the mask |
| `source/strafer_ros/deploy/docker-compose.yml` (`perception`) | IIO device access for the container |
| `source/strafer_ros/99-strafer.rules` | host udev rules, including the IIO permissions for non-root use |
