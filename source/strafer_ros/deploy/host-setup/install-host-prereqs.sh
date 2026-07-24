#!/usr/bin/env bash
# Host prerequisites for running the strafer containers on a Jetson.
# Idempotent -- safe to re-run: it does NOT restart docker unless the runtime
# config actually changes, so re-running won't kill a live robot stack.
# Run with sudo.
#
#   sudo bash host-setup/install-host-prereqs.sh
#
# References CANONICAL host files by path (no byte-copies in deploy/):
#   strafer_bringup/config/99-cyclonedds-rmem.conf   (rmem sysctl)
#   99-strafer.rules                                 (RoboClaw + D555 IMU udev)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRAFER_ROS="$(cd "$HERE/../.." && pwd)"                 # .../strafer_ros

echo "[host-setup] CycloneDDS receive-buffer sysctl (16 MB) ..."
install -m 0644 "$STRAFER_ROS/strafer_bringup/config/99-cyclonedds-rmem.conf" \
  /etc/sysctl.d/99-cyclonedds-rmem.conf
sysctl --system >/dev/null
echo "  net.core.rmem_max = $(cat /proc/sys/net/core/rmem_max)  (expect 16777216)"

echo "[host-setup] RoboClaw / RealSense-D555-IMU udev rules ..."
install -m 0644 "$STRAFER_ROS/99-strafer.rules" /etc/udev/rules.d/99-strafer.rules
udevadm control --reload-rules && udevadm trigger || true
echo "  installed 99-strafer.rules (/dev/roboclaw{0,1}; D555 IMU iio/hidraw perms)"

# --- Docker + NVIDIA runtime + compose ---------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  echo "[host-setup] Docker not found. Install first:  sudo apt-get install -y docker.io"
else
  if command -v nvidia-ctk >/dev/null 2>&1; then
    cur="$(docker info 2>/dev/null | awk -F': ' '/Default Runtime/{print $2}')"
    if [ "$cur" != "nvidia" ]; then
      echo "[host-setup] Setting NVIDIA as the default docker runtime ..."
      nvidia-ctk runtime configure --runtime=docker --set-as-default >/dev/null 2>&1 || true
      systemctl restart docker || true            # only when it changed -- re-runs won't nuke a live stack
      echo "  default-runtime: $(docker info 2>/dev/null | awk -F': ' '/Default Runtime/{print $2}')"
    else
      echo "  default-runtime already nvidia (no docker restart)"
    fi
  fi
  # Docker bridge netfilter modules (L4T lacks iptable_raw; strafer runs host-net
  # and images build with `network: host`, so this is for general docker health).
  printf 'br_netfilter\niptable_nat\niptable_filter\n' > /etc/modules-load.d/docker-netfilter.conf
  modprobe br_netfilter iptable_nat iptable_filter 2>/dev/null || true
  # Docker Compose v2 plugin (docker.io does not bundle it; curl may be absent).
  if ! docker compose version >/dev/null 2>&1; then
    echo "[host-setup] Installing docker compose v2 plugin ..."
    apt-get install -y docker-compose-v2 >/dev/null 2>&1 || {
      apt-get install -y wget >/dev/null 2>&1
      mkdir -p /usr/libexec/docker/cli-plugins
      wget -qO /usr/libexec/docker/cli-plugins/docker-compose \
        "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
        && chmod +x /usr/libexec/docker/cli-plugins/docker-compose
    }
    echo "  docker compose: $(docker compose version 2>/dev/null | head -1)"
  fi
fi

# --- Deferred hardware note --------------------------------------------------
# The D555 IMU needs custom-built hid-sensor-hub + IIO kernel modules on the NX
# kernel (host-side; see docs/D555_IMU_KERNEL_FIX.md) -- kernel modules cannot
# live in a container. 99-strafer.rules (installed above) grants the iio/hidraw
# perms those modules expose.
echo "[host-setup] done."
