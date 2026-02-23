# Enabling the RealSense D555 IMU on Jetson Orin (Tegra Kernel)

## Platform

| Component | Version |
|---|---|
| Board | NVIDIA Jetson Orin NX |
| L4T | R36.5.0 |
| Kernel | 5.15.185-tegra |
| Ubuntu | 22.04 (Jammy) |
| Camera | Intel RealSense D555 (USB PID `0B56`, FW 7.56.19918.835) |
| librealsense | 2.56.4 (deb), pyrealsense2 2.56.5.9235 (pip) |
| ROS 2 | Humble |

## The Problem

The RealSense D555 contains a Bosch BMI055 IMU (3-axis accelerometer + 3-axis
gyroscope) exposed over USB as an HID Sensor Hub device (USB interface 5,
`bInterfaceClass=3`).

On standard x86 Ubuntu systems this works out of the box because the kernel is
compiled with `CONFIG_HID_SENSOR_HUB=m`. When the camera is plugged in the
`hid-sensor-hub` driver claims the HID interface, creates IIO devices in
`/sys/bus/iio/devices/`, and librealsense reads the IMU data through the Linux
IIO subsystem (`iio_hid_sensor` backend).

On the **Jetson Orin's Tegra kernel** (5.15.185-tegra), `CONFIG_HID_SENSOR_HUB`
is **not set**. The HID interface is instead bound to `hid-generic`, which does
not create any IIO devices. As a result:

- `/sys/bus/iio/devices/` is empty.
- librealsense logs `"No HID info provided, IMU is disabled"`.
- `rs-enumerate-devices` shows Stereo Module and RGB Camera but no Motion Module.
- The policy observation vector (indices 0–5: `imu_accel`, `imu_gyro`) gets zeros.

### How We Confirmed the Root Cause

1. **USB descriptors** — `lsusb -v -d 8086:0b56` showed interface 5 with
   `bInterfaceClass 3` (HID).

2. **HID report descriptor** — Reading the 1002-byte report descriptor via
   `/sys/bus/usb/devices/.../report_descriptor` and parsing with
   `hidrd-convert` confirmed `Accelerometer 3D (0x73)` and
   `Gyroscope 3D (0x76)` usage pages are present in the firmware.

3. **Driver binding** — `udevadm info` on the hidraw device showed
   `DRIVER=hid-generic` instead of the expected `hid-sensor-hub`.

4. **Kernel config** — `zcat /proc/config.gz | grep HID_SENSOR` confirmed
   `CONFIG_HID_SENSOR_HUB is not set`.

5. **IIO dependencies satisfied** — The kernel *does* have the IIO subsystem
   (`CONFIG_IIO=y`, `CONFIG_IIO_BUFFER=y`, `CONFIG_IIO_KFIFO_BUF=m`,
   `CONFIG_IIO_TRIGGERED_BUFFER=m`), so only the HID sensor driver stack
   is missing.

### Why Hidraw Alone Doesn't Work

librealsense on Linux uses the `iio_hid_sensor` backend for IMU access. It does
**not** read from `/dev/hidrawN` directly. Even with `MODE="0666"` on the hidraw
device, librealsense still cannot see the IMU because no IIO devices exist.
The fix must provide IIO devices, which requires the kernel's HID sensor driver
stack.

## The Fix — Building Out-of-Tree Kernel Modules

Since recompiling the entire Tegra kernel is impractical (and risks breaking
NVIDIA-specific patches), we build just the 5 missing modules out-of-tree from
the Ubuntu `linux-source-5.15.0` package.

### Prerequisites

```bash
# Kernel headers (already installed with JetPack)
dpkg -l nvidia-l4t-kernel-headers
# → 5.15.185-tegra-36.5.0-...

# Ubuntu kernel source (provides compatible HID sensor code)
sudo apt-get install -y linux-source-5.15.0
# → /usr/src/linux-source-5.15.0/linux-source-5.15.0.tar.bz2

# Build tools
sudo apt-get install -y build-essential
```

### Step 1: Extract Source Files

The tarball is large and bz2-decompression is slow on ARM. Use Python:

```python
import tarfile, bz2, io

TARBALL = "/usr/src/linux-source-5.15.0/linux-source-5.15.0.tar.bz2"
OUTDIR  = "/tmp/hid-sensor-build"

with open(TARBALL, "rb") as f:
    data = bz2.decompress(f.read())

tf = tarfile.open(fileobj=io.BytesIO(data))
members = [m for m in tf.getmembers() if any(x in m.name for x in [
    "drivers/hid/hid-sensor-hub.c",
    "drivers/hid/hid-sensor-custom.c",
    "drivers/hid/hid-ids.h",
    "drivers/iio/accel/hid-sensor-accel-3d.c",
    "drivers/iio/gyro/hid-sensor-gyro-3d.c",
    "drivers/iio/common/hid-sensors/",
    "include/linux/hid-sensor-hub.h",
    "include/linux/hid-sensor-ids.h",
])]
tf.extractall(OUTDIR, members)
tf.close()
```

### Step 2: Install Missing Headers

The Tegra kernel headers don't include the HID sensor headers. Copy them in:

```bash
SRC=/tmp/hid-sensor-build/linux-source-5.15.0
sudo cp $SRC/include/linux/hid-sensor-hub.h /lib/modules/$(uname -r)/build/include/linux/
sudo cp $SRC/include/linux/hid-sensor-ids.h /lib/modules/$(uname -r)/build/include/linux/
```

### Step 3: Create Kbuild Files

Each module directory needs a `Kbuild` file:

```bash
cd /tmp/hid-sensor-build/linux-source-5.15.0

# HID sensor hub
echo 'obj-m := hid-sensor-hub.o' > drivers/hid/Kbuild

# IIO common (attributes + trigger)
printf 'obj-m += hid-sensor-iio-common.o\nobj-m += hid-sensor-trigger.o\nhid-sensor-iio-common-y := hid-sensor-attributes.o\n' \
    > drivers/iio/common/hid-sensors/Kbuild

# Accelerometer
echo 'obj-m := hid-sensor-accel-3d.o' > drivers/iio/accel/Kbuild

# Gyroscope
echo 'obj-m := hid-sensor-gyro-3d.o' > drivers/iio/gyro/Kbuild
```

### Step 4: Compile Modules

Build in dependency order, passing each previous module's `Module.symvers` so
cross-module symbols resolve:

```bash
KDIR=/lib/modules/$(uname -r)/build
SRC=/tmp/hid-sensor-build/linux-source-5.15.0

# 1. hid-sensor-hub.ko
make -C $KDIR M=$SRC/drivers/hid modules

# 2. hid-sensor-iio-common.ko + hid-sensor-trigger.ko
make -C $KDIR M=$SRC/drivers/iio/common/hid-sensors \
    KBUILD_EXTRA_SYMBOLS=$SRC/drivers/hid/Module.symvers modules

# 3. hid-sensor-accel-3d.ko
make -C $KDIR M=$SRC/drivers/iio/accel \
    KBUILD_EXTRA_SYMBOLS="$SRC/drivers/hid/Module.symvers $SRC/drivers/iio/common/hid-sensors/Module.symvers" modules

# 4. hid-sensor-gyro-3d.ko
make -C $KDIR M=$SRC/drivers/iio/gyro \
    KBUILD_EXTRA_SYMBOLS="$SRC/drivers/hid/Module.symvers $SRC/drivers/iio/common/hid-sensors/Module.symvers" modules
```

> **Note:** The compiler version warning (`11.4.0 vs 11.4.0-2`) is cosmetic and
> can be ignored — the ABI is identical.

### Step 5: Install Modules

```bash
sudo mkdir -p /lib/modules/$(uname -r)/extra
sudo cp $SRC/drivers/hid/hid-sensor-hub.ko                         /lib/modules/$(uname -r)/extra/
sudo cp $SRC/drivers/iio/common/hid-sensors/hid-sensor-iio-common.ko /lib/modules/$(uname -r)/extra/
sudo cp $SRC/drivers/iio/common/hid-sensors/hid-sensor-trigger.ko    /lib/modules/$(uname -r)/extra/
sudo cp $SRC/drivers/iio/accel/hid-sensor-accel-3d.ko               /lib/modules/$(uname -r)/extra/
sudo cp $SRC/drivers/iio/gyro/hid-sensor-gyro-3d.ko                 /lib/modules/$(uname -r)/extra/

sudo depmod -a
```

### Step 6: Load Modules (First Time)

Load prerequisite IIO modules, then the HID sensor stack:

```bash
sudo modprobe industrialio-triggered-buffer

sudo insmod /lib/modules/$(uname -r)/extra/hid-sensor-hub.ko
sudo insmod /lib/modules/$(uname -r)/extra/hid-sensor-iio-common.ko
sudo insmod /lib/modules/$(uname -r)/extra/hid-sensor-trigger.ko
sudo insmod /lib/modules/$(uname -r)/extra/hid-sensor-accel-3d.ko
sudo insmod /lib/modules/$(uname -r)/extra/hid-sensor-gyro-3d.ko
```

After loading, verify IIO devices appeared:

```bash
ls /sys/bus/iio/devices/
# Expected:  iio:device0  iio:device1  trigger0  trigger1

cat /sys/bus/iio/devices/iio:device0/name   # → accel_3d
cat /sys/bus/iio/devices/iio:device1/name   # → gyro_3d
```

### Step 7: Configure Auto-Load on Boot

```bash
cat <<'EOF' | sudo tee /etc/modules-load.d/hid-sensor-imu.conf
# HID Sensor modules for RealSense D555 IMU
hid-sensor-hub
hid-sensor-iio-common
hid-sensor-trigger
hid-sensor-accel-3d
hid-sensor-gyro-3d
EOF
```

### Step 8: Udev Rule for Non-Root Access

librealsense needs write access to IIO sysfs attributes (`scan_elements/*_en`,
`buffer/enable`, etc.) and read/write on `/dev/iio:device*`. The IIO devices
are recreated on each USB plug/rebind, so a udev rule is required.

Add to `99-strafer.rules`:

```
SUBSYSTEM=="iio", KERNEL=="iio:device*", MODE="0666", \
    RUN+="/bin/sh -c 'find /sys%p -type f -exec chmod a+rw {} + 2>/dev/null'"
```

> **Gotcha:** On this udev version, `DEVTYPE=="iio_device"` is reported as an
> invalid key. Use `KERNEL=="iio:device*"` instead.

Install the rule:

```bash
sudo cp 99-strafer.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=iio
```

## Verification

### pyrealsense2 — Motion Module Visible

```python
import pyrealsense2 as rs
ctx = rs.context()
dev = ctx.query_devices()[0]
for s in dev.query_sensors():
    print(s.get_info(rs.camera_info.name))
```

Expected output includes **Motion Module** alongside Stereo Module and RGB Camera.

### pyrealsense2 — Live IMU Data

```python
import pyrealsense2 as rs

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
pipe.start(cfg)

for _ in range(5):
    frames = pipe.wait_for_frames(1000)
    for f in frames:
        m = f.as_motion_frame()
        d = m.get_motion_data()
        stream = "Accel" if f.get_profile().stream_type() == rs.stream.accel else "Gyro"
        print(f"{stream}: x={d.x:.4f}  y={d.y:.4f}  z={d.z:.4f}")

pipe.stop()
```

Expected: Accel ≈ (0, -9.8, 0) for gravity, Gyro ≈ (0, 0, 0) at rest.

### ROS 2 — /d555/imu Topic

```bash
ros2 launch strafer_perception perception.launch.py
# In another terminal:
ros2 topic hz /d555/imu        # → ~200 Hz
ros2 topic echo /d555/imu --once
```

The message should be `sensor_msgs/Imu` with populated `linear_acceleration`
and `angular_velocity` fields.

## IMU Stream Profiles

| Stream | Rates Available | Format |
|---|---|---|
| Accelerometer | 100 Hz, 200 Hz | MOTION_XYZ32F |
| Gyroscope | 200 Hz, 400 Hz | MOTION_XYZ32F |

The perception launch uses 200 Hz unified IMU (`unite_imu_method: 2`) which
combines accel and gyro into a single `/d555/imu` topic.

## Known Issues

### `initial_reset` Causes USB Errors

Setting `initial_reset: true` in the RealSense ROS 2 launch triggers a USB
device reset that causes `tegra-xusb` transfer errors
(`xioctl(VIDIOC_QBUF) failed: No such device`). The camera re-enumerates 2–3
times during the reset cycle, sometimes degrading to SuperSpeed (5 Gbps) from
SuperSpeed+ (10 Gbps). **Leave `initial_reset: false`.**

### Permissions Lost on USB Re-Plug

The IIO devices are destroyed and recreated when the camera is unplugged or the
USB bus resets. The udev rule automatically reapplies permissions, but if
running a pipeline at the time, it will crash and need restarting.

### Kernel Updates

If the Tegra kernel is updated (e.g., JetPack upgrade), the modules in
`/lib/modules/<old-version>/extra/` will no longer load. You'll need to
rebuild them against the new kernel headers. The same procedure applies — just
point `KDIR` at the new `/lib/modules/$(uname -r)/build`.

## File Inventory

| File | Purpose |
|---|---|
| `/lib/modules/5.15.185-tegra/extra/*.ko` | 5 compiled kernel modules |
| `/etc/modules-load.d/hid-sensor-imu.conf` | Auto-load modules on boot |
| `/etc/udev/rules.d/99-strafer.rules` | Device permissions (IIO + hidraw) |
| `/lib/modules/5.15.185-tegra/build/include/linux/hid-sensor-hub.h` | Installed header |
| `/lib/modules/5.15.185-tegra/build/include/linux/hid-sensor-ids.h` | Installed header |
| `/tmp/hid-sensor-build/linux-source-5.15.0/` | Build tree (ephemeral) |
| `source/strafer_ros/99-strafer.rules` | Source-controlled copy of udev rules |
