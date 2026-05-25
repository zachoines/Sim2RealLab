# Windows workstation bringup for the sim-side bridge

End-to-end recipe for running `make sim-bridge` / `make sim-bridge-gui`
/ `make sim-harness` on a Windows workstation against the same Jetson
autonomy stack the Linux DGX serves. Lets the bridge run on a
gaming-class GPU (RTX 4080 reference) so the DGX is freed up for
training.

The architecture is **fully WSL2 Ubuntu-22.04**, not native Windows.
The forcing reason is at the bottom of this section; the short answer
is that NVIDIA's Isaac Sim 6 documentation states *"Isaac Sim supports
Cyclone DDS middleware for Linux only. Windows is not supported at
this time."* and this project's cross-host wire is pinned to
`rmw_cyclonedds_cpp` for documented reasons (see `.env.example`). The
WSL2-Ubuntu host is on Linux as far as DDS is concerned, so all the
existing Linux scripts (`env_setup.sh`, `Makefile`, `Scripts/`) work
unchanged.

## System snapshot (reference workstation)

| Item | Value | Status |
|---|---|---|
| OS | Windows 11 Pro, build 26200 | 22H2+ required (for WSL2 mirrored networking) |
| GPU | NVIDIA GeForce RTX 4080, 16 GB | Supported (RT cores present) |
| Driver | NVIDIA 595.97 | Supports CUDA-on-WSL2 (driver 530+ required) |
| WSL | 2.6.3.0, WSLg 1.0.71 | `wsl --version` confirms |
| WSL kernel | 6.6.87.2-microsoft-standard-WSL2+ | Carries `dxgkrnl` for CUDA passthrough |
| Python (in WSL) | 3.12.13 (conda) | Required for Isaac Sim 6 |
| Free disk on C: | ≥ 100 GB | Isaac Sim install ≈ 25 GB; Isaac Lab + ROS 2 + caches ≈ 20 GB |

## Companion docs

- [`docs/tasks/context/repo-topology.md`](tasks/context/repo-topology.md)
  — hosts, repo paths, conda env contract. The Windows host slots into
  the same DGX role for the bridge.
- [`docs/tasks/context/bridge-runtime-invariants.md`](tasks/context/bridge-runtime-invariants.md)
  — bridge mainloop assumptions, `--profile` harness, per-phase
  reference numbers.
- [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](INTEGRATION_SIM_IN_THE_LOOP.md)
  — Stage 2-6 stage-by-stage mission validation. Once the Windows
  workstation reaches Stage 2 (bridge alone, manually driven) it
  serves as the DGX in that runbook with no further changes.
- [`docs/DGX_SPARK_SETUP.md`](DGX_SPARK_SETUP.md) — the Linux DGX
  recipe this Windows recipe mirrors. Step numbering is intentionally
  parallel.

---

## Step 0: Enable WSL2 and install Ubuntu-22.04

WSL2 itself is shipped with Windows 11; if the `wsl` command isn't
already on PATH, install once:

```powershell
# Elevated PowerShell — one-time only
wsl --install --no-distribution
# Reboot if prompted, then:
wsl --update
```

Verify:

```powershell
wsl --version
# Expect WSL 2.x; WSLg present; D3D/DXCore present.
```

Install the Ubuntu-22.04 distro **without launching it** (we skip the
OOBE wizard and provision the UNIX user non-interactively):

```powershell
wsl --install -d Ubuntu-22.04 --no-launch
wsl --list --verbose
# Ubuntu-22.04 should appear, State=Stopped, Version=2.
```

If you already use the `NvidiaSDKM_Ubuntu_*` distro for Jetson
flashing, leave it alone — Sim2RealLab uses a separate Ubuntu-22.04
distro so the SDK Manager workflow is unaffected.

---

## Step 1: Configure WSL2 mirrored networking

Default WSL2 puts the distro on its own NAT'd subnet (172.x.x.x),
which breaks CycloneDDS multicast discovery to the Jetson on
192.168.x.x. **Mirrored networking** puts the WSL2 distro on the same
LAN interface as the Windows host so the Jetson can discover its
topics natively.

Mirrored mode requires Windows 11 22H2+ (any build ≥ 22621). The
reference workstation is build 26200.

Edit `~/.wslconfig` on the Windows host. If you already have one (e.g.
from NVIDIA SDK Manager), back it up first:

```powershell
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.pre-bringup.bak -Force
```

Write/append the following keys under `[wsl2]`. **Preserve any
existing keys** in that section (e.g. `kernel=...` set by other
NVIDIA tooling):

```ini
[wsl2]
# ... any existing keys (kernel=, memory=, processors=, ...) stay here
networkingMode=mirrored
firewall=true
dnsTunneling=true
autoProxy=true
vmIdleTimeout=-1   # never auto-shutdown the VM; lets long-running bridge sessions survive idle terminals

[experimental]
hostAddressLoopback=true
```

`vmIdleTimeout=-1` is the deviation from the DGX setup: a long-lived
bridge or harness run on the DGX is naturally backed by a long-lived
terminal, but on Windows the bridge is one PID inside the WSL2 VM,
and the default 60-second idle timeout will reap it if the launching
terminal exits. Setting `-1` removes that footgun.

Apply:

```powershell
wsl --shutdown
# Then re-launch any distro; it will pick up the new networking mode.
```

Verify after re-launch (from inside the new distro):

```bash
hostname -I
# Expect a 192.168.x.x address on the SAME subnet as the Jetson and the
# Windows host's LAN IP, NOT a 172.x.x.x.
ip -4 route | head -3
# default via 192.168.x.1 dev eth2 ...
```

If the IP still starts with `172.`, mirrored mode didn't take. Check
the Windows build number (22H2+), confirm the `.wslconfig` lives
exactly at `$env:USERPROFILE\.wslconfig`, and that `wsl --shutdown`
was actually run.

---

## Step 2: Bootstrap the UNIX user (skip the OOBE wizard)

The reference workstation uses UNIX username `zacho` (matches the
Windows username) with passwordless sudo so non-interactive setup
scripts work cleanly. Substitute your own username.

```powershell
$bootstrap = @'
set -euo pipefail
useradd -m -s /bin/bash -G sudo zacho
printf '%s\n' "zacho ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/99-zacho
chmod 440 /etc/sudoers.d/99-zacho
cat > /etc/wsl.conf <<EOF
[boot]
systemd=true
[user]
default=zacho
[network]
generateResolvConf=true
hostname=sim2real-win
[interop]
enabled=true
appendWindowsPath=true
EOF
'@
$bytes = [System.Text.Encoding]::UTF8.GetBytes($bootstrap)
$b64 = [Convert]::ToBase64String($bytes)
wsl -d Ubuntu-22.04 --user root -- bash -lc "echo $b64 | base64 -d | bash -l"
wsl --terminate Ubuntu-22.04
```

Verify:

```powershell
wsl -d Ubuntu-22.04 -- whoami    # → zacho
wsl -d Ubuntu-22.04 -- hostname  # → sim2real-win
wsl -d Ubuntu-22.04 -- nvidia-smi -L   # → "GPU 0: NVIDIA GeForce RTX 4080 ..."
```

`nvidia-smi -L` returning the host GPU confirms CUDA-on-WSL2 works.
If it errors with "command not found", install nvidia-cuda-toolkit's
helper utils via apt; if the GPU isn't visible, the host NVIDIA driver
is too old (need ≥ 530 for CUDA-on-WSL2).

---

## Step 3: Install apt prerequisites

All commands below run **inside the WSL2 Ubuntu shell** unless noted.
Open one with the launcher:

```powershell
.\Scripts\Open-Sim2RealLab-Wsl.ps1
```

Or directly:

```powershell
wsl -d Ubuntu-22.04
```

Inside Ubuntu:

```bash
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  curl ca-certificates gnupg lsb-release wget git build-essential \
  software-properties-common locales pkg-config cmake \
  libglu1-mesa libxi6 libxrandr2 libxcursor1 libxinerama1 \
  libgl1 libegl1 libxkbcommon0 libxrender1 libsm6 libice6 \
  libvulkan1 vulkan-tools mesa-vulkan-drivers \
  iputils-ping net-tools
sudo locale-gen en_US.UTF-8

# WSL2 ships nvidia-smi at /usr/lib/wsl/lib/nvidia-smi; some Kit plugins
# look in /usr/bin. Symlink to silence the noise.
if [ ! -e /usr/bin/nvidia-smi ] && [ -e /usr/lib/wsl/lib/nvidia-smi ]; then
  sudo ln -sf /usr/lib/wsl/lib/nvidia-smi /usr/bin/nvidia-smi
fi
```

The X11/GL libs are required by Isaac Sim's Kit runtime even on
headless launches (Kit dynamically loads them). The Vulkan loader
package (`libvulkan1`) is required by Kit's renderer; see the
"Vulkan-on-WSL2" gotcha below for the NVIDIA ICD setup that
`libvulkan1` alone doesn't cover.

---

## Step 4: Install miniconda + create `env_isaaclab3`

The DGX uses `env_isaaclab3` (Python 3.12) as the canonical Isaac
Lab env. Mirror it exactly so the same `env_setup.sh` works.

```bash
curl -fsSL -o /tmp/miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-1-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
rm /tmp/miniconda.sh
~/miniconda3/bin/conda init bash
exec bash  # reload to pick up the conda hook

conda config --set auto_activate_base false
# Conda 25.x requires accepting Terms of Service for default channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n env_isaaclab3 python=3.12 -y
conda activate env_isaaclab3
pip install --upgrade pip
python --version  # → Python 3.12.x
```

---

## Step 5: Install Isaac Sim 6 via pip

Matches the DGX recipe (Isaac Sim ≥ 6.0.0, the line that requires
Python 3.12). This is the largest single download — expect 20-30 GB
across the dependency closure and 30-60 minutes on a typical
connection.

```bash
conda activate env_isaaclab3
pip install "isaacsim[all]>=6.0.0" --extra-index-url https://pypi.nvidia.com
```

Sanity check:

```bash
python -c "import isaacsim; print(isaacsim.__version__)"
# → 6.0.0.0 (or newer)
```

---

## Step 6: Install ROS 2 Humble + CycloneDDS

```bash
sudo add-apt-repository -y universe
sudo curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
sudo apt-get update -qq
sudo apt-get install -y \
  ros-humble-ros-base \
  ros-humble-rmw-cyclonedds-cpp \
  ros-humble-demo-nodes-cpp \
  ros-humble-foxglove-bridge \
  ros-humble-cv-bridge \
  python3-colcon-common-extensions \
  python3-argcomplete python3-rosdep
```

Verify the Humble CLI sees CycloneDDS:

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 doctor --report 2>&1 | head -20
# Look for: middleware: rmw_cyclonedds_cpp
```

---

## Step 7: Clone + install Isaac Lab pinned to the DGX commit

```bash
mkdir -p ~/Documents/repos
git clone https://github.com/isaac-sim/IsaacLab.git ~/Documents/repos/IsaacLab
cd ~/Documents/repos/IsaacLab
git checkout ae41e2aca68    # VERSION 3.0.0 pre-release; matches what env_isaaclab3 on the DGX runs as of 2026-05
conda activate env_isaaclab3
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y PRIVACY_CONSENT=Y
./isaaclab.sh --install rl
```

`./isaaclab.sh --install` is the supported installer; it discovers the
conda env's pip-installed `isaacsim` via `$CONDA_PREFIX` and runs
`pip install -e` on every Isaac Lab submodule (`isaaclab`,
`isaaclab_assets`, `isaaclab_rl`, `isaaclab_tasks`, etc.) plus the
`rl` extras (rsl_rl + skrl + sb3). Expect ~5 minutes of pip work. The
EULA exports keep the Kit bootstrap that runs at the end of the
install non-interactive.

**Why the SHA pin.** As of 2026-05 the strafer_lab module imports
break against IsaacLab `develop` tip — PR #4741 lazy-loaded
`isaaclab.utils`, so `from isaaclab.utils import configclass`
resolves to a submodule instead of the function. The DGX is on
commit `ae41e2aca68` (`VERSION = 3.0.0`); pinning the Windows
checkout to the same commit keeps the two hosts behaviorally
identical. The drift catch-up is tracked in
[`docs/tasks/active/tooling/isaaclab-develop-upgrade.md`](tasks/active/tooling/isaaclab-develop-upgrade.md);
when that brief ships, drop this pin.

**Then upgrade rsl_rl to 5.x.** `./isaaclab.sh --install rl` at this
commit pins `rsl-rl-lib==3.1.2`, but strafer_lab uses the rsl_rl 5.x
distribution API (`from rsl_rl.modules.distribution import
Distribution`). Force-upgrade after the IsaacLab install:

```bash
pip install --upgrade "rsl-rl-lib"   # ⇒ 5.3.0 or newer
python -c "from rsl_rl.modules.distribution import Distribution; print('ok')"
```

(The IsaacLab develop-upgrade brief above tracks pulling rsl_rl into
IsaacLab's own pin; until then, the manual upgrade is required.)

---

## Step 8: Clone Sim2RealLab + install strafer packages

The Windows workstation gets **its own** WSL-side clone of the repo,
separate from any Windows-side checkout at `C:\Workspace\Sim2RealLab`.
Cross-OS filesystem sharing (working out of `/mnt/c/`) is explicitly
out of scope per the brief — file IO across the WSL2/9P boundary is
~10× slower than ext4 native, and the bridge's TiledCamera readback
path is IO-sensitive.

```bash
mkdir -p ~/Workspace
git clone https://github.com/zachoines/Sim2RealLab.git ~/Workspace/Sim2RealLab
cd ~/Workspace/Sim2RealLab
# (If you authored work in a Windows-side checkout, sync it in:
#   git remote add windows /mnt/c/Workspace/Sim2RealLab
#   git fetch windows
#   git checkout -b <branch> windows/<branch>
# )

conda activate env_isaaclab3
pip install -e source/strafer_shared
pip install -e source/strafer_lab
pip install -e source/strafer_autonomy
pip install -e source/strafer_vlm
```

---

## Step 9: Configure `.env`

Copy the template and fill in WSL-side paths:

```bash
cp .env.example .env
$EDITOR .env
```

WSL-specific values for the reference workstation:

```bash
STRAFER_ROOT=/home/zacho/Workspace/Sim2RealLab
CONDA_ROOT=/home/zacho/miniconda3
CONDA_ENV=env_isaaclab3
ISAACLAB=/home/zacho/Documents/repos/IsaacLab/isaaclab.sh
STRAFER_ISAACLAB_PYTHON=/home/zacho/miniconda3/envs/env_isaaclab3/bin/python
# Infinigen + Blender are optional on the Windows host — scene generation
# stays on the Linux DGX. Leave STRAFER_BLENDER_BIN / INFINIGEN_ROOT unset
# unless you also intend to generate scenes here.

# ROS 2 cross-host — same values as the DGX side
ROS_DOMAIN_ID=42
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Jetson reachable on the same LAN via mirrored networking
STRAFER_JETSON_HOST=192.168.50.24
```

`env_setup.sh` is reused verbatim. On x86_64 WSL2 the aarch64-only
`LD_PRELOAD` branch is skipped automatically; the script's Isaac
Sim 6 bundled-humble detection (looking for
`isaacsim.ros2.core/humble/lib/librmw_cyclonedds_cpp.so` under the
conda env) finds the pip-installed payload and exports `ROS_DISTRO`,
`LD_LIBRARY_PATH`, and the cleaned `PYTHONPATH` for Kit's 3.12
interpreter.

Smoke source:

```bash
cd ~/Workspace/Sim2RealLab
source env_setup.sh
# Expect "[env_setup] loaded .env", then a status block listing
# STRAFER_ROOT, ISAACSIM_PATH, CONDA_ENV, ROS_DISTRO, etc.
```

---

## Step 10: Smoke tests

### 10a — Isaac Sim launches and Kit boots

```bash
cd ~/Documents/repos/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
# A black viewport window should pop up via WSLg. Ctrl+C to exit.
```

If the window never appears, WSLg isn't forwarding GPU rendering;
confirm `wsl --version` lists a `WSLg version` line.

### 10b — bridge launches headless

```bash
cd ~/Workspace/Sim2RealLab
source env_setup.sh
make sim-bridge
```

Expected console (same as the DGX recipe):

```
[sim_in_the_loop] bridge graph built at /World/ROS2Bridge
[sim_in_the_loop] chassis_prim=/World/envs/env_0/Robot/strafer/body_link
[sim_in_the_loop] color camera prim=/World/envs/env_0/Robot/strafer/body_link/d555_camera_perception
[sim_in_the_loop] action tensor shape = (1, 3)
[sim_in_the_loop] async camera publisher up: ...
```

From a second WSL terminal:

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=42
ros2 topic list | sort
# Expect /cmd_vel, /d555/color/image_raw, /d555/depth/image_rect_raw,
# /strafer/odom, /tf, /clock.
ros2 topic hz /d555/color/image_raw
# Expect ~30 Hz after a few seconds.
```

### 10c — bridge throughput baseline

```bash
$ISAACLAB -p source/strafer_lab/scripts/run_sim_in_the_loop.py \
  --mode bridge --headless --enable_cameras \
  --profile --profile-interval 10 --profile-window 200
```

Capture the rolling p50/p99 numbers per phase and compare against the
DGX reference table in
[`docs/tasks/context/bridge-runtime-invariants.md`](tasks/context/bridge-runtime-invariants.md#phase-level-profiler---profile).
RTX 4080's `sim.render` is expected to be substantially faster than
the DGX's iGPU; PhysX (`sim.step`) is CPU-bound and should match the
DGX within a few ms.

### 10d — cross-host DDS reach (requires the Jetson powered up)

```bash
ping 192.168.50.24                      # must respond
ros2 topic pub /windows_test std_msgs/String "data: hi" --once
# From the Jetson:  ros2 topic list  → /windows_test should appear.
```

If the Jetson sees `/cmd_vel` and `/d555/*` published by the WSL
bridge, you have full end-to-end DDS through mirrored networking.

---

## Operator one-liners

All daily-driver targets work identically to the DGX side. From
inside the WSL shell (use `Open-Sim2RealLab-Wsl.ps1` to open one):

| What | Command |
|---|---|
| Headless bridge | `make sim-bridge` |
| Bridge with Kit viewport | `make sim-bridge-gui` |
| Sim-in-the-loop harness | `SCENE_META=... SCENE_USD=... OUTPUT_DIR=... make sim-harness` |
| VLM service | `make serve-vlm` |
| Planner service | `make serve-planner` |
| DGX-side tests | `make test-dgx` |

From a Windows PowerShell prompt without first opening a WSL shell:

```powershell
.\Scripts\Open-Sim2RealLab-Wsl.ps1 -Command "make sim-bridge"
```

---

## Known gotchas on Windows-via-WSL2

- **Idle VM reaping.** Without `vmIdleTimeout=-1` in `.wslconfig`,
  the VM auto-shuts down 60 s after the last attached `wsl.exe`
  process exits. Background bridge / harness runs launched via
  `nohup ... &` from a then-closed terminal will be SIGKILLed mid-flight.
- **`/tmp` is wiped on distro restart** by systemd-tmpfiles. Don't
  stash anything you want to keep between sessions there; use
  `~/logs` instead.
- **JetPack kernel pin in `.wslconfig` is silently ignored** by WSL
  2.6.x because the path contains spaces. New distros boot on the
  default MS kernel anyway, which is what we want (it carries
  `dxgkrnl`). If you ever switch to a kernel pin that *does* apply,
  verify `nvidia-smi -L` still works inside the distro.
- **Two checkouts of the repo.** Window-side at
  `C:\Workspace\Sim2RealLab` (for IDE editing) and WSL-side at
  `~/Workspace/Sim2RealLab` (for execution). Per the brief, NOT a
  shared mount. Sync edits across with `git remote add windows
  /mnt/c/Workspace/Sim2RealLab; git fetch windows; git merge
  windows/<branch>`.
- **Python interpreters are not interchangeable across the
  WSL/system-Humble boundary.** The bridge runs under Kit's 3.12
  Python (via the conda env's isaacsim bundle); ad-hoc `ros2 topic
  pub` and `rclpy` clients run under system Python 3.10 from
  `/opt/ros/humble`. Don't `pip install rclpy` into `env_isaaclab3` —
  use Kit's bundled rclpy when inside Kit, system rclpy when outside.

- **Vulkan-on-WSL2 with NVIDIA: ICD JSON not auto-installed.**
  CUDA-on-WSL2 works out of the box (Isaac Sim's compute kernels run
  on the RTX 4080 fine), but Isaac Sim's *Kit renderer* uses Vulkan.
  Out of the box, `vulkaninfo` inside the WSL distro only sees
  `llvmpipe` (software Vulkan), not the host NVIDIA GPU — so Kit
  errors `vkCreateInstance failed. Vulkan 1.1 is not supported, or
  your driver requires an update.` and `Failed to create any GPU
  devices`.

  The reason is that the Linux NVIDIA Vulkan ICD JSON (e.g.
  `/etc/vulkan/icd.d/nvidia_icd.json`) is not part of the
  CUDA-on-WSL2 shim that NVIDIA ships at `/usr/lib/wsl/lib/`.
  Driver-version-dependent: some Windows NVIDIA driver builds carry
  the Linux Vulkan support in their WSL package; some don't. The
  bringup reference host (driver 595.97) is in the latter category.

  Workarounds (in order of preference):

  1. **Upgrade the Windows-side NVIDIA driver** to a 600.x+ build
     that ships the WSL Linux Vulkan ICD. Confirmed by checking
     `/etc/vulkan/icd.d/nvidia_icd.json` after `wsl --shutdown` +
     re-launch.
  2. **Install `libnvidia-gl-<branch>` from NVIDIA's CUDA-on-WSL apt
     repo** per [NVIDIA's CUDA-on-WSL user
     guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html);
     that package historically delivered the ICD JSON to the right
     path.
  3. **Software Vulkan (LLVMpipe)** for contract / import smoke
     only: `export
     VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`
     before launching Kit. Kit boots and the bridge OmniGraph
     constructs, but rendering is CPU-bound and unrepresentative of
     RTX 4080 perf. Not suitable for the brief's perf-baseline
     bullet.

  Until one of the workarounds is in place, the bridge cold-starts
  to the renderer-init stage then aborts; the install path itself
  (imports, conda env, Isaac Sim, IsaacLab, strafer_lab) is
  otherwise validated. The Vulkan-on-WSL2 setup investigation is
  tracked in
  [`docs/tasks/active/tooling/windows-workstation-bringup.md`](tasks/active/tooling/windows-workstation-bringup.md)'s
  decision log.

---

## Why not native Windows or hybrid?

The brief considered three architectures up front; this section
records the Phase 1 finding that ruled the other two out.

- **Native Windows (PowerShell port of `env_setup.sh` + `Makefile`).**
  NVIDIA's Isaac Sim 6 docs at
  <https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_ros_other_platforms.html>
  state plainly: *"Isaac Sim supports Cyclone DDS middleware for
  Linux only. Windows is not supported at this time."* The project's
  `.env.example` pins `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` for the
  documented reason that FastDDS's shared-memory transport doesn't
  traverse machine boundaries. So a Windows-native bridge can't speak
  CycloneDDS to the Jetson at all; the only way through is a
  FastDDS↔CycloneDDS translator in WSL2, which adds latency that
  would invalidate the per-phase reference numbers the brief requires.
- **Hybrid (sim native + ROS 2 in WSL2).** Same blocker: the
  CycloneDDS-Windows wall is on the Windows half, not the ROS 2 half.
  Splitting hosts does not move the wall.
- **WSL2-only.** WSL2 Ubuntu is Linux as far as DDS is concerned.
  CycloneDDS Just Works. CUDA-on-WSL2 has been GA since driver 460+.
  Isaac Sim 6 + Isaac Lab develop are both officially Linux-supported.
  Mirrored networking puts the WSL2 distro on the same LAN as the
  Jetson with no port forwarding. The only WSL2-specific moving part
  is the `vmIdleTimeout=-1` line in `.wslconfig`; everything else is
  vanilla.
