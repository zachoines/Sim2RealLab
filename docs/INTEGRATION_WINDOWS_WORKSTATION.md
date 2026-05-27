# Windows workstation bringup

End-to-end recipes for running Sim2RealLab work on a Windows
workstation (RTX 4080 reference). Two distinct paths because two
distinct use cases hit two distinct constraints:

| Use case | Path | Status |
|---|---|---|
| Data collection / teleop / `collect_demos.py` / headed env inspection / standalone harness | **Path A — Native Windows** | Works today |
| `make sim-bridge` cross-host DDS to Jetson (mission-driven sim-in-the-loop) | **Path B — WSL2 Ubuntu-22.04** | Scaffolding ready; blocked on an open NVIDIA-Vulkan-on-WSL2 bug — see [`windows-workstation-bringup-sim-bridge.md`](tasks/active/tooling/windows-workstation-bringup-sim-bridge.md) |

The split is forced by NVIDIA's docs:

- *"Isaac Sim supports Cyclone DDS middleware for Linux only. Windows is not supported at this time."* This rules out a native-Windows sim-bridge (the project's cross-host wire is `rmw_cyclonedds_cpp` per `.env.example`).
- And as of NVIDIA driver 610.47 + Ubuntu 22.04 + WSL 2.6.3, the Linux NVIDIA Vulkan ICD is not in the WSL2 shim — `vulkaninfo` inside WSL2 only sees `llvmpipe` (software Vulkan), and Isaac Sim's Kit renderer requires Vulkan 1.1+. Per NVIDIA's own developer forum (March 2026) this is an unresolved upstream bug.

So: Path A for the things you can do today (data collection on the GPU directly); Path B preserved end-to-end for the day the Vulkan-on-WSL2 hole closes (the install runs cleanly, the only failing step is Kit's renderer init).

## System snapshot (reference workstation)

| Item | Value | Status |
|---|---|---|
| OS | Windows 11 Pro, build 26200 | Both paths supported (Path B needs 22H2+ for WSL2 mirrored networking) |
| GPU | NVIDIA GeForce RTX 4080, 16 GB | RT cores present |
| Driver | NVIDIA 610.47 | Path A: CUDA + Vulkan work natively. Path B: CUDA works in WSL2; Vulkan does not (upstream NVIDIA bug). |
| Python (native) | 3.12.10 via py launcher | Required for Isaac Sim 6 |
| WSL | 2.6.3.0, WSLg 1.0.71 | Path B only |
| Free disk on C: | ≥ 100 GB | Native install ≈ 25 GB; WSL2 stack adds another ~25 GB |

## Companion docs

- [`docs/tasks/context/repo-topology.md`](tasks/context/repo-topology.md) — hosts, repo paths, conda env contract. The Windows host parallels the DGX role for data collection (Path A); for sim-bridge (Path B) it parallels the DGX once Path B unblocks.
- [`docs/tasks/context/bridge-runtime-invariants.md`](tasks/context/bridge-runtime-invariants.md) — bridge mainloop assumptions, `--profile` harness, per-phase reference numbers (used once Path B works).
- [`docs/DGX_SPARK_SETUP.md`](DGX_SPARK_SETUP.md) — the Linux DGX recipe. Both paths mirror its install structure step-for-step.
- [`docs/INTEGRATION_SIM_IN_THE_LOOP.md`](INTEGRATION_SIM_IN_THE_LOOP.md) — stage-by-stage mission validation against the sim-bridge. The Path B install plugs into Stage 2 of that runbook when Vulkan-on-WSL2 unblocks.

---

# Path A — Native Windows (data collection / harness / teleop)

## Step A0: Install Python 3.12

The default Python on a fresh Windows 11 box is often 3.10/3.11. Isaac Sim 6 requires Python 3.12.

```powershell
winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
py --list   # → Python 3.12 (64-bit) appears as default
```

## Step A1: Enable Windows long-path support

Isaac Sim's dependency tree contains paths longer than 260 chars; without long-path support, pip install errors deep into the install.

```powershell
# Run elevated PowerShell once
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -Type DWord
git config --system core.longpaths true
# Reboot only required if you previously had long paths disabled and have
# files past the limit; for a fresh checkout, no reboot needed.
```

## Step A2: Create `venv_isaac` at the repo root

```powershell
Set-Location C:\Workspace\Sim2RealLab
py -3.12 -m venv venv_isaac
& venv_isaac\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

`venv_isaac/` is in `.gitignore`. The Scripts/launch_isaac_*.ps1 helpers expect this exact location.

## Step A3: Install Isaac Sim 6 via pip

Matches the DGX recipe (Isaac Sim ≥ 6.0.0, Python 3.12 line). Largest single download — expect 25-35 GB total (Isaac Sim core ≈ 20 GB; the `extscache` Kit-extension cache adds another ≈ 5 GB) and 30-90 minutes.

```powershell
pip install "isaacsim[all,extscache]>=6.0.0" --extra-index-url https://pypi.nvidia.com
```

**The `extscache` extra is mandatory** even though Isaac Sim's docs sometimes show it as optional. Without it, Kit fails at startup with `No versions of isaacsim.anim.robot.schema that satisfies: isaacsim.exp.base-6.0.0` because the bundled Kit experience references extensions Kit can only resolve through that cache (the live `kit/default` + `kit/sdk` registries are typically unreachable from a normal network). The DGX recipe uses `isaacsim[all,extscache]==5.1.0` for the same reason.

Sanity check:

```powershell
python -c "import isaacsim; print(isaacsim.__version__)"
# → 6.0.0.0 (or newer)
```

## Step A4: Clone + install Isaac Lab pinned to the DGX commit

```powershell
Set-Location C:\Workspace\Sim2RealLab
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
Set-Location IsaacLab
git checkout ae41e2aca68    # VERSION 3.0.0 pre-release; matches what env_isaaclab3 on the DGX runs as of 2026-05
$env:ACCEPT_EULA = "Y"; $env:OMNI_KIT_ACCEPT_EULA = "Y"; $env:PRIVACY_CONSENT = "Y"
.\isaaclab.bat --install rl
# .bat installs isaaclab + isaaclab_rl + isaaclab_mimic; the rest of the
# submodules strafer_lab depends on are not auto-installed on the Windows
# .bat path (Linux .sh installs more by default — drift between the two).
# Install them explicitly with --no-deps so pip doesn't try to fetch them
# from PyPI (they're source-only):
Set-Location ..
pip install -e IsaacLab\source\isaaclab_assets --no-deps
pip install -e IsaacLab\source\isaaclab_tasks --no-deps
```

`IsaacLab/` is in `.gitignore`. The clone lives inside the repo for path-locality but is not tracked.

**Why the SHA pin.** As of 2026-05 strafer_lab's bridge imports break against IsaacLab `develop` tip — PR #4741 lazy-loaded `isaaclab.utils`, so `from isaaclab.utils import configclass` resolves to a submodule instead of the function. The DGX is on commit `ae41e2aca68` (`VERSION = 3.0.0`); pinning the Windows checkout to the same commit keeps the two hosts behaviorally identical. The drift catch-up is tracked in [`docs/tasks/active/tooling/isaaclab-develop-upgrade.md`](tasks/active/tooling/isaaclab-develop-upgrade.md); when that brief ships, drop this pin.

**Then upgrade rsl_rl to 5.x.** `isaaclab.bat --install rl` at this commit pins `rsl-rl-lib==3.1.2`, but strafer_lab uses the rsl_rl 5.x distribution API (`from rsl_rl.modules.distribution import Distribution`). Force-upgrade after the IsaacLab install:

```powershell
pip install --upgrade "rsl-rl-lib"   # ⇒ 5.3.0 or newer
python -c "from rsl_rl.modules.distribution import Distribution; print('ok')"
```

**Then patch IsaacLab's USD spawner for Windows.** At commit
`ae41e2aca68`, `IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py`
unconditionally `import fcntl` at line 8 — Unix-only — so any env that
spawns a USD asset (i.e. every strafer_lab env) crashes on Windows.
The `fcntl` is only actually called when `LOCAL_WORLD_SIZE > 1` (multi-rank
distributed training), so the fix is a conditional import:

```powershell
# In IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py,
# replace `import fcntl` (line 8) with:
#
#   import sys
#   if sys.platform != "win32":
#       import fcntl
#   else:
#       fcntl = None  # type: ignore[assignment]
#
# Single-process runs never enter the _world_size > 1 branch that uses
# fcntl, so this is safe.
```

This patch lives in the (gitignored) IsaacLab clone — re-apply after any
`git pull` of IsaacLab. Tracked in
[`docs/tasks/active/tooling/isaaclab-develop-upgrade.md`](tasks/active/tooling/isaaclab-develop-upgrade.md)
for upstream resolution.

## Step A5: Install strafer packages in editable mode

```powershell
Set-Location C:\Workspace\Sim2RealLab
pip install -e source\strafer_shared
pip install -e source\strafer_lab
# strafer_autonomy + strafer_vlm only needed if you also serve planner/VLM on
# this Windows host. The data-collection path does not require them.
```

## Step A6: Smoke tests

### A6a — Isaac Sim Kit boots headed (WSLg-free, runs on RTX 4080 directly)

```powershell
.\Scripts\launch_isaac_sim.ps1
# → Kit editor viewport opens. Ctrl+C in PowerShell to exit.
```

### A6b — Env smoke through the launcher

```powershell
.\Scripts\launch_isaac_lab.ps1 Scripts\test_strafer_env.py `
    --env Isaac-Strafer-Nav-Real-NoCam-v0 --num_envs 8 --duration 10
```

NoCam variant — fastest, no scene-asset dependencies. Expect a clean exit after 10 seconds. PhysX runs on the GPU natively (no WSL2 indirection), so no `No CUDA context manager` warnings.

### A6c — The actual use case: gamepad demo collection

```powershell
.\Scripts\launch_isaac_lab.ps1 source\strafer_lab\scripts\collect_demos.py `
    --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 `
    --output demos\ --max_episodes 100 --viz kit
```

Headed Kit viewport + gamepad teleop. The whole point of the native Windows path: iterate fast on data collection runs without the DGX being tied up, then `rsync` / `git lfs` / scp the demos directory back to the DGX for fine-tuning.

## Path A operator one-liners

| Want to | Run |
|---|---|
| Open Isaac Sim editor | `.\Scripts\launch_isaac_sim.ps1` |
| Run any Isaac Lab script (drop-in for `$ISAACLAB -p`) | `.\Scripts\launch_isaac_lab.ps1 <script> [args]` |
| Headed env smoke | `.\Scripts\launch_isaac_lab.ps1 Scripts\test_strafer_env.py --env <env_id> --num_envs N --duration S` |
| Gamepad demo collection | `.\Scripts\launch_isaac_lab.ps1 source\strafer_lab\scripts\collect_demos.py --task <env_id> --output demos\ --max_episodes N --viz kit` |
| Headed inference rollout | `.\Scripts\launch_isaac_lab.ps1 Scripts\play_strafer_navigation.py --env <env_id> --checkpoint <path> --viz kit --real_time` |
| PPO training (if you want to use the RTX 4080 for it) | `.\Scripts\launch_isaac_lab.ps1 Scripts\train_strafer_navigation.py --env <env_id> --num_envs 64 --headless` |

## Path A known gotchas

- **No cross-host DDS.** Native Windows has no CycloneDDS — the bridge path is Path B. If you try `make sim-bridge` equivalents here, the bridge process will start but the Jetson won't see its topics. This is by design; use Path B for that.
- **Two clones of the repo, none shared with the WSL distro.** The Windows-native install operates entirely out of `C:\Workspace\Sim2RealLab\`. Path B operates out of `~/Workspace/Sim2RealLab/` inside the WSL distro. Sync between them via `git pull` from each side; not via filesystem mounts.
- **PowerShell execution policy.** First time running `Activate.ps1` may error with "execution of scripts is disabled." Fix once per user with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

---

# Path B — WSL2 Ubuntu-22.04 (sim-bridge to Jetson)

This path is preserved end-to-end because it's the only architecturally viable home for `make sim-bridge` on Windows (CycloneDDS-Linux-only forces WSL2). The install runs cleanly; the **renderer-init step fails today** because the NVIDIA Linux Vulkan ICD isn't in WSL2's GPU shim. Track the unblock in [`docs/tasks/active/tooling/windows-workstation-bringup-sim-bridge.md`](tasks/active/tooling/windows-workstation-bringup-sim-bridge.md).

Reference WSL2 environment captured during the spike: user `zacho`, hostname `sim2real-win`, mirrored LAN IP `192.168.50.83`, default kernel `6.6.87.2-microsoft-standard-WSL2+`.

## Step B0: Enable WSL2 and install Ubuntu-22.04

```powershell
# Elevated PowerShell — one-time only
wsl --install --no-distribution
wsl --update
wsl --install -d Ubuntu-22.04 --no-launch
wsl --list --verbose
```

If you already use the `NvidiaSDKM_Ubuntu_*` distro for Jetson flashing, leave it alone — the bringup uses a separate Ubuntu-22.04 distro so the SDK Manager workflow is unaffected.

## Step B1: Configure WSL2 mirrored networking

Default WSL2 NAT puts the distro on its own subnet (172.x.x.x), which breaks CycloneDDS multicast discovery to the Jetson on 192.168.x.x. **Mirrored networking** puts the WSL2 distro on the same LAN interface as the Windows host.

Mirrored mode requires Windows 11 22H2+ (any build ≥ 22621). The reference workstation is build 26200.

```powershell
# Back up any existing .wslconfig before editing
Copy-Item $env:USERPROFILE\.wslconfig $env:USERPROFILE\.wslconfig.pre-bringup.bak -Force -ErrorAction SilentlyContinue
```

Append the following keys under `[wsl2]`. Preserve any existing keys (e.g. `kernel=` set by other NVIDIA tooling):

```ini
[wsl2]
# ... any existing keys stay here ...
networkingMode=mirrored
firewall=true
dnsTunneling=true
autoProxy=true
vmIdleTimeout=-1   # never auto-shutdown the VM; lets long-running bridge sessions survive idle terminals

[experimental]
hostAddressLoopback=true
```

`vmIdleTimeout=-1` matters: long bridge / harness runs are one PID inside the WSL2 VM, and the default 60-second idle timeout will reap them when the launching terminal exits.

Apply:

```powershell
wsl --shutdown
```

Verify after re-launch (from inside the new distro):

```bash
hostname -I
# Expect a 192.168.x.x address on the SAME subnet as the Jetson and the
# Windows host's LAN IP, NOT a 172.x.x.x.
```

## Step B2: Bootstrap the UNIX user

Skip the interactive Ubuntu OOBE wizard by booting as root and provisioning the user non-interactively. Substitute your own username.

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
wsl -d Ubuntu-22.04 -- nvidia-smi -L   # → "GPU 0: NVIDIA GeForce RTX 4080 ..."
```

`nvidia-smi -L` returning the host GPU confirms CUDA-on-WSL2 works. (Vulkan is the separate broken piece — see the gotchas at the end of this section.)

## Step B3: Install apt prerequisites

All commands below run inside the WSL2 Ubuntu shell. Open one with the launcher:

```powershell
.\Scripts\Open-Sim2RealLab-Wsl.ps1
```

Or directly: `wsl -d Ubuntu-22.04`. Inside Ubuntu:

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

## Step B4: Install miniconda + create `env_isaaclab3`

Mirror the DGX exactly so the same `env_setup.sh` works.

```bash
curl -fsSL -o /tmp/miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-1-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
rm /tmp/miniconda.sh
~/miniconda3/bin/conda init bash
exec bash

conda config --set auto_activate_base false
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n env_isaaclab3 python=3.12 -y
conda activate env_isaaclab3
pip install --upgrade pip
```

## Step B5: Install Isaac Sim 6 via pip

```bash
conda activate env_isaaclab3
pip install "isaacsim[all]>=6.0.0" --extra-index-url https://pypi.nvidia.com
python -c "import isaacsim; print(isaacsim.__version__)"
```

## Step B6: Install ROS 2 Humble + CycloneDDS

```bash
sudo add-apt-repository -y universe
sudo curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
sudo apt-get update -qq
sudo apt-get install -y \
  ros-humble-ros-base ros-humble-rmw-cyclonedds-cpp \
  ros-humble-demo-nodes-cpp ros-humble-foxglove-bridge ros-humble-cv-bridge \
  python3-colcon-common-extensions python3-argcomplete python3-rosdep
```

## Step B7: Clone + install Isaac Lab pinned to the DGX commit

```bash
mkdir -p ~/Documents/repos
git clone https://github.com/isaac-sim/IsaacLab.git ~/Documents/repos/IsaacLab
cd ~/Documents/repos/IsaacLab
git checkout ae41e2aca68
conda activate env_isaaclab3
export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=Y PRIVACY_CONSENT=Y
./isaaclab.sh --install rl
pip install --upgrade "rsl-rl-lib"   # 5.x; same drift fix as Path A
```

## Step B8: Clone Sim2RealLab inside the WSL home

The Windows-native checkout at `C:\Workspace\Sim2RealLab` is separate. Cross-OS shared mounts are not supported (~10× slowdown on the `/mnt/c` 9P boundary).

```bash
mkdir -p ~/Workspace
git clone https://github.com/zachoines/Sim2RealLab.git ~/Workspace/Sim2RealLab
cd ~/Workspace/Sim2RealLab
conda activate env_isaaclab3
pip install -e source/strafer_shared
pip install -e source/strafer_lab
```

## Step B9: `.env` + smoke

```bash
cp .env.example .env
# Edit .env: STRAFER_ROOT, CONDA_ROOT, CONDA_ENV, ISAACLAB, STRAFER_ISAACLAB_PYTHON,
# ROS_DOMAIN_ID=42, RMW_IMPLEMENTATION=rmw_cyclonedds_cpp,
# STRAFER_JETSON_HOST=192.168.50.24
source env_setup.sh
# Expect: ROS_DISTRO=humble (isaacsim.ros2.core bundle), all STRAFER_* exports

# Cross-host DDS proof (with Jetson up on LAN):
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=42
ros2 topic pub /windows_test std_msgs/String "data: 'hi from wsl2'" --once
# Jetson side: ros2 topic list  → /windows_test should appear
```

## Path B known gotchas

- **`make sim-bridge` aborts at Kit renderer init.** The current showstopper. CUDA-on-WSL2 works (Isaac Sim's compute kernels import and run), but Kit's renderer needs Vulkan 1.1+, and the NVIDIA Linux Vulkan ICD is not present in the WSL2 GPU shim with driver 610.47 (`/etc/vulkan/icd.d/` is empty for NVIDIA; `vulkaninfo` only sees `llvmpipe`). Error: `vkCreateInstance failed. Vulkan 1.1 is not supported`. Per [NVIDIA's developer forum](https://forums.developer.nvidia.com/t/critical-bug-rtx-5070-ti-mobile-missing-nvidia-icd-json-in-wsl2-causes-vulkan-initialization-failure-on-driver-591-86/359933) (March 2026) this is an unresolved upstream bug — no working manual ICD JSON, no working `dzn`/Mesa-Dozen route on Ubuntu 22.04's mesa 23.2. Workarounds to try when investigating: (a) upgrade the host Windows NVIDIA driver to a build that ships the WSL Linux Vulkan shim — verify `/etc/vulkan/icd.d/nvidia_icd.json` appears after `wsl --shutdown`; (b) install `libnvidia-gl-<branch>` from NVIDIA's CUDA-on-WSL apt repo (historically delivered the ICD JSON); (c) `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json` to fall back to LLVMpipe software Vulkan — Kit boots but rendering is CPU-bound, suitable for contract / import smoke only.
- **Idle VM reaping** without `vmIdleTimeout=-1` in `.wslconfig`: the VM auto-shuts down 60 s after the last attached `wsl.exe` exits. Background bridge / harness runs launched via `nohup ... &` from a then-closed terminal will be SIGKILLed.
- **`/tmp` is wiped on distro restart** by systemd-tmpfiles. Stash work-in-progress under `~/logs`.
- **JetPack kernel pin in `.wslconfig` is silently ignored** by WSL 2.6.x (the path contains spaces). New distros boot on the default MS kernel anyway, which is what we want (it carries `dxgkrnl` for CUDA passthrough).
- **Two clones of the repo, none shared with the Windows-native install.** Path B operates out of `~/Workspace/Sim2RealLab/` inside the WSL distro; Path A out of `C:\Workspace\Sim2RealLab\`. Not a shared mount.
- **Python interpreters are not interchangeable across the WSL/system-Humble boundary.** Bridge runs under Kit's 3.12 (via conda env's isaacsim bundle); ad-hoc `ros2 topic pub` runs under system Python 3.10 from `/opt/ros/humble`. Don't `pip install rclpy` into `env_isaaclab3`.

---

# Cross-cutting: which architecture for which use case

Both paths can coexist on the same workstation. Pick by what you're trying to do.

| Activity | Path |
|---|---|
| Gamepad teleop / `collect_demos.py` runs / behavior-cloning data capture | **A** — Native Windows; uses RTX 4080 directly through DirectX-/Vulkan-native Kit |
| `Scripts/test_strafer_env.py` headed env inspection | **A** |
| `play_strafer_navigation.py` headed inference rollout | **A** |
| Local PPO training (RTX 4080 has less memory than DGX Spark but can still iterate fast on small `num_envs`) | **A** |
| `make sim-bridge` / `make sim-bridge-gui` / `make sim-harness` (cross-host DDS to Jetson) | **B**, once the NVIDIA-Vulkan-on-WSL2 bug is resolved |
| Cross-host DDS without Isaac Sim involvement (e.g. an `rclpy` test that just publishes to the Jetson) | **B** works today — Cyclone + mirrored networking are fine; only Isaac Sim's Kit renderer is blocked |

Bridge perf reference numbers (`bridge-runtime-invariants.md`) will be captured against Path B when it unblocks.

---

# Decision-log appendix: why two paths, not one

The original brief considered three architectures: full WSL2 Ubuntu (the brief's preferred path), hybrid (sim native + ROS 2 in WSL2), and full native Windows (PowerShell port of the Linux toolchain).

The forcing functions:

1. **CycloneDDS is Linux-only**, per NVIDIA's Isaac Sim 6 docs. The project's cross-host wire is pinned to `rmw_cyclonedds_cpp` for documented reasons (FastDDS's shared-memory transport doesn't traverse machines). A native-Windows bridge cannot speak CycloneDDS to the Jetson at all. → Rules out native Windows for the bridge.
2. **NVIDIA Vulkan on WSL2 is currently broken**, per NVIDIA's developer forum (March 2026; confirmed inside Phase 2 of this bringup against driver 610.47). `vulkaninfo` inside WSL2 only sees `llvmpipe`. → Rules out WSL2 for any Isaac Sim use case until the bug is fixed.

For the immediate use case (data collection / harness / teleop), the user does not need cross-host DDS — those workflows are standalone on the sim host, with the demo dataset rsync'd to the DGX after the fact. Native Windows handles that fine.

For the future bridge use case, WSL2 is preserved as the only architecturally viable home (CycloneDDS-Linux-only). The Phase 2 install proved the WSL2 stack is otherwise functional end-to-end: CUDA passthrough works, conda + isaacsim + isaaclab + rsl_rl install clean, mirrored networking puts the distro on the LAN, env_setup.sh sources fine, all strafer_lab imports succeed. The single gap is Kit's renderer init. When NVIDIA ships a driver that includes the Linux Vulkan ICD, Path B unblocks.

The Phase 1 alternative (FastDDS↔CycloneDDS translator in WSL2) was considered and rejected — it adds latency that would invalidate the perf reference numbers the brief requires. If WSL2-Vulkan-fix never lands, the bridge stays on the Linux DGX (status quo) and the translator approach gets revisited.
