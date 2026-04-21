#!/usr/bin/env bash
# Sim2RealLab per-shell environment setup.
#
# Usage:
#
#     source env_setup.sh
#
# This script loads variables from .env (if present at the repo root) into
# the current shell and applies platform-specific shell state that Python
# code cannot manage itself — notably LD_PRELOAD on aarch64 hosts, which
# is read by the dynamic loader at process start and therefore must be set
# before Python or any subprocess launches.
#
# Running it multiple times is idempotent.
#
# Why this file is shell, not Python: LD_PRELOAD is consumed by ld.so when
# a process starts. Setting os.environ["LD_PRELOAD"] inside a Python script
# is too late — the preload decision is frozen before Python imports its
# first module. The strafer_autonomy / strafer_lab codebase reads plain
# env vars via os.environ.get(...), so anything STRAFER_BLENDER_BIN-shaped
# can come from either Python dotenv parsing or this shell wrapper, but
# LD_PRELOAD has to come from here.

# ---------------------------------------------------------------------------
# Locate this script's own directory regardless of where `source` was called
# from. Required because `source env_setup.sh` may be run from any CWD.
# ---------------------------------------------------------------------------

if [ -n "${BASH_SOURCE[0]}" ]; then
    _ENV_SETUP_SELF="${BASH_SOURCE[0]}"
else
    # zsh fallback
    _ENV_SETUP_SELF="${(%):-%x}"
fi
_REPO_ROOT="$(cd "$(dirname "${_ENV_SETUP_SELF}")" && pwd -P)"
unset _ENV_SETUP_SELF

# ---------------------------------------------------------------------------
# Load .env (if it exists). Naive KEY=VALUE parser — no quoting interpreter,
# no command substitution. Keep it boring so subtle parse bugs don't hide
# misconfiguration. The Python side has its own parser in prep_room_usds.py
# for the single STRAFER_BLENDER_BIN var, so this loop stays simple.
# ---------------------------------------------------------------------------

if [ -f "${_REPO_ROOT}/.env" ]; then
    set -a  # export everything assigned below
    # shellcheck disable=SC1091
    source "${_REPO_ROOT}/.env"
    set +a
    echo "[env_setup] loaded ${_REPO_ROOT}/.env"
else
    echo "[env_setup] no .env at ${_REPO_ROOT} — copy .env.example to .env" >&2
fi

# ---------------------------------------------------------------------------
# Platform-specific LD_PRELOAD for aarch64 / Grace.
# Isaac Sim's bundled torch loads libgomp dynamically and collides with the
# system's libgomp unless /lib/aarch64-linux-gnu/libgomp.so.1 is preloaded.
# This is documented in the NVIDIA/Arm Isaac Sim on DGX Spark install guide.
# ---------------------------------------------------------------------------

case "$(uname -m)" in
    aarch64|arm64)
        _LIBGOMP="/lib/aarch64-linux-gnu/libgomp.so.1"
        if [ -f "${_LIBGOMP}" ]; then
            # Prepend so our preload takes priority over anything already set
            case "${LD_PRELOAD:-}" in
                *"${_LIBGOMP}"*) ;;  # already in LD_PRELOAD, skip
                "") export LD_PRELOAD="${_LIBGOMP}" ;;
                *) export LD_PRELOAD="${_LIBGOMP}:${LD_PRELOAD}" ;;
            esac
            echo "[env_setup] LD_PRELOAD=${LD_PRELOAD}"
        else
            echo "[env_setup] WARNING: ${_LIBGOMP} not found — Isaac Sim may fail to launch" >&2
        fi
        unset _LIBGOMP
        ;;
    *)
        # Non-aarch64 hosts (Windows workstation, x86_64 Linux CI) don't need
        # the libgomp preload. Leave LD_PRELOAD alone.
        ;;
esac

# ---------------------------------------------------------------------------
# Infinigen's launch_blender.py hardcodes `<infinigen_root>/blender/blender`
# as the binary path. If the user has both STRAFER_BLENDER_BIN and
# INFINIGEN_ROOT set, create that symlink for them so `python -m
# infinigen.launch_blender` works out of the box.
# ---------------------------------------------------------------------------

if [ -n "${INFINIGEN_ROOT:-}" ] && [ -n "${STRAFER_BLENDER_BIN:-}" ]; then
    _INF_LINK="${INFINIGEN_ROOT}/blender"
    _BLENDER_BIN_DIR="$(dirname "${STRAFER_BLENDER_BIN}")"
    if [ -d "${_BLENDER_BIN_DIR}" ] && [ ! -e "${_INF_LINK}" ]; then
        if ln -s "${_BLENDER_BIN_DIR}" "${_INF_LINK}" 2>/dev/null; then
            echo "[env_setup] created ${_INF_LINK} -> ${_BLENDER_BIN_DIR}"
        fi
    fi
    unset _INF_LINK _BLENDER_BIN_DIR
fi

# ---------------------------------------------------------------------------
# ROS2 cross-host defaults
# ---------------------------------------------------------------------------
# Only applied when the values come from `.env`. If the user already has
# ROS_DOMAIN_ID / RMW_IMPLEMENTATION set from elsewhere (system defaults,
# another project's env script), we leave them alone. Python scripts do not
# read these — only ROS2 processes on this shell or its children do.

if [ -n "${ROS_DOMAIN_ID:-}" ]; then
    export ROS_DOMAIN_ID
fi
if [ -n "${RMW_IMPLEMENTATION:-}" ]; then
    export RMW_IMPLEMENTATION
fi

# ---------------------------------------------------------------------------
# Source Isaac Sim's vendored ROS 2 Humble
# ---------------------------------------------------------------------------
# Isaac Sim's `isaacsim.ros2.bridge` extension defaults to `system_default`,
# meaning "whatever ROS 2 is sourced in the shell; otherwise fall back to the
# bundled default." The bundled default is Jazzy, which (a) is ABI-incompatible
# with the Jetson's Humble stack and (b) fails to load unless its own lib dir
# is on LD_LIBRARY_PATH. Sourcing the vendored Humble setup.bash fixes both:
# it sets AMENT_PREFIX_PATH / LD_LIBRARY_PATH / PYTHONPATH and advertises
# ROS_DISTRO=humble to the bridge.
#
# Idempotency: guard on whether the specific packman humble lib path is
# already on LD_LIBRARY_PATH. Guarding on ROS_DISTRO alone turned out to be
# too loose — a parent shell (or a leaked export from a previous Kit run)
# can leave ROS_DISTRO set without the lib path actually being present,
# causing the bridge to fail with "librmw_cyclonedds_cpp.so: cannot open
# shared object file" even though env_setup.sh appeared to succeed.

_HUMBLE_PREFIX=$(shopt -s nullglob; set -- "$HOME"/.cache/packman/chk/nv_ros2/humble_py_*; [ $# -gt 0 ] && printf '%s' "$1")
if [ -n "${_HUMBLE_PREFIX}" ] && [ -f "${_HUMBLE_PREFIX}/setup.bash" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${_HUMBLE_PREFIX}/lib:"*)
            # Already on LD_LIBRARY_PATH — earlier env_setup.sh call in this
            # shell chain took care of it.
            ;;
        *)
            # shellcheck disable=SC1091
            source "${_HUMBLE_PREFIX}/setup.bash"
            echo "[env_setup] ROS_DISTRO=${ROS_DISTRO:-humble} (sourced ${_HUMBLE_PREFIX})"
            ;;
    esac
else
    echo "[env_setup] WARNING: no Humble packman bundle found under ~/.cache/packman/chk/nv_ros2/; Isaac Sim bridge may fall back to Jazzy" >&2
fi
unset _HUMBLE_PREFIX

# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

echo "[env_setup] STRAFER_ROOT=${STRAFER_ROOT:-<unset>}"
echo "[env_setup] STRAFER_BLENDER_BIN=${STRAFER_BLENDER_BIN:-<unset>}"
echo "[env_setup] INFINIGEN_ROOT=${INFINIGEN_ROOT:-<unset>}"
echo "[env_setup] ISAACSIM_PATH=${ISAACSIM_PATH:-<unset>}"
echo "[env_setup] STRAFER_INFINIGEN_PYTHON=${STRAFER_INFINIGEN_PYTHON:-<unset>}"
echo "[env_setup] STRAFER_ISAACLAB_PYTHON=${STRAFER_ISAACLAB_PYTHON:-<unset>}"
echo "[env_setup] HF_HOME=${HF_HOME:-<unset>}"
echo "[env_setup] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}"
echo "[env_setup] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>}"
echo "[env_setup] STRAFER_JETSON_HOST=${STRAFER_JETSON_HOST:-<unset>}"

unset _REPO_ROOT
