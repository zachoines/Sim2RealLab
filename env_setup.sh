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
# Advertise ROS 2 Humble runtime to Isaac Sim's ros2 extension
# ---------------------------------------------------------------------------
# Isaac Sim 6 ships a self-contained Humble payload (rclpy compiled for
# Kit's Python 3.12 plus every runtime .so — librmw_cyclonedds_cpp,
# libddsc, librcl, etc.) inside the isaacsim.ros2.core extension. The
# extension activates that payload automatically on startup, but only
# when (a) ROS_DISTRO=humble, and (b) the payload's lib/ directory is
# already on LD_LIBRARY_PATH so the C++ OmniGraph publishers can dlopen
# their dependencies.
#
# Preference order:
#   1) Isaac Sim 6 bundled Humble (STRAFER_ISAACLAB_PYTHON points at a
#      conda env with isaacsim[all]>=6.0.0 installed). This is the
#      canonical setup on the Isaac Lab 3.0 stack and keeps the rclpy
#      ABI aligned with Kit's Python, which is a prerequisite for any
#      Python-side ROS 2 publisher/subscriber running inside Kit.
#   2) Legacy packman-cached Humble bundle (~/.cache/packman/chk/
#      nv_ros2/humble_py_*). Used on hosts where isaacsim is not
#      pip-installed (e.g. an Isaac Sim 5.x source build). The packman
#      payload is Python-3.11-flavoured, so its rclpy will NOT import
#      inside Kit's 3.12 Python — but the C++ DDS libs still work for
#      the OmniGraph publishers, which is enough for bridge mode.
#
# Guard on whether the chosen lib directory is already on
# LD_LIBRARY_PATH to keep a nested `source env_setup.sh` idempotent.

_ISAACSIM_HUMBLE_LIB=
if [ -n "${STRAFER_ISAACLAB_PYTHON:-}" ]; then
    _CONDA_ENV_PREFIX="$(dirname "$(dirname "${STRAFER_ISAACLAB_PYTHON}")")"
    _CANDIDATE="${_CONDA_ENV_PREFIX}/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/humble/lib"
    if [ -f "${_CANDIDATE}/librmw_cyclonedds_cpp.so" ]; then
        _ISAACSIM_HUMBLE_LIB="${_CANDIDATE}"
    fi
    unset _CANDIDATE _CONDA_ENV_PREFIX
fi

if [ -n "${_ISAACSIM_HUMBLE_LIB}" ]; then
    # Defensive: scrub any leftover ROS 2 shell state from prior
    # sessions (e.g. a previous source of packman's 3.11 humble
    # setup.bash). Kit's isaacsim.ros2.core extension calls
    # restore_ros2_python_paths() at startup, which reads OLD_PYTHONPATH
    # / AMENT_PREFIX_PATH and re-injects those entries onto sys.path
    # BEFORE falling back to its own bundled 3.12 rclpy. If those
    # envvars still point at the packman 3.11 bundle, Python finds the
    # wrong rclpy first and import blows up on ABI mismatch. Unsetting
    # them here keeps restore_ros2_python_paths a no-op so the extension
    # goes straight to its bundled 3.12 rclpy.
    unset OLD_PYTHONPATH AMENT_PREFIX_PATH AMENT_CURRENT_PREFIX
    # Scrub packman humble entries from PYTHONPATH while preserving
    # anything else the user put there. The packman bundle is 3.11
    # flavoured; keeping it on Python 3.12's sys.path will shadow the
    # isaacsim bundled 3.12 rclpy.
    if [ -n "${PYTHONPATH:-}" ]; then
        _CLEANED=
        IFS=':'; for _entry in ${PYTHONPATH}; do
            case "${_entry}" in
                *"/packman/chk/nv_ros2/humble_py_3.11_"*) ;;
                *) _CLEANED="${_CLEANED:+${_CLEANED}:}${_entry}" ;;
            esac
        done
        unset IFS
        if [ -n "${_CLEANED}" ]; then
            export PYTHONPATH="${_CLEANED}"
        else
            unset PYTHONPATH
        fi
        unset _CLEANED _entry
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${_ISAACSIM_HUMBLE_LIB}:"*) ;;  # idempotent
        *)
            export LD_LIBRARY_PATH="${_ISAACSIM_HUMBLE_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            ;;
    esac
    export ROS_DISTRO="${ROS_DISTRO:-humble}"
    echo "[env_setup] ROS_DISTRO=${ROS_DISTRO} (isaacsim.ros2.core bundle)"
else
    _HUMBLE_PREFIX=$(shopt -s nullglob; set -- "$HOME"/.cache/packman/chk/nv_ros2/humble_py_*; [ $# -gt 0 ] && printf '%s' "$1")
    if [ -n "${_HUMBLE_PREFIX}" ] && [ -f "${_HUMBLE_PREFIX}/setup.bash" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${_HUMBLE_PREFIX}/lib:"*) ;;
            *)
                # shellcheck disable=SC1091
                source "${_HUMBLE_PREFIX}/setup.bash"
                echo "[env_setup] ROS_DISTRO=${ROS_DISTRO:-humble} (sourced packman ${_HUMBLE_PREFIX})"
                ;;
        esac
    else
        echo "[env_setup] WARNING: no Humble payload found (neither isaacsim.ros2.core nor packman); Isaac Sim bridge may fall back to Jazzy" >&2
    fi
    unset _HUMBLE_PREFIX
fi
unset _ISAACSIM_HUMBLE_LIB

# ---------------------------------------------------------------------------
# Python 3.11 ROS 2 Humble — opt-in rclpy location for non-Kit clients
# ---------------------------------------------------------------------------
# Off-Kit tooling that runs in a Python 3.11 conda env (e.g. the
# sim-in-the-loop bench subscriber invoked through STRAFER_INFINIGEN_PYTHON)
# cannot reuse the Isaac Sim 6 bundle above — its rclpy C extension is
# 3.12-flavoured. Export the packman-cached 3.11 Humble bundle's
# site-packages as STRAFER_ROS2_HUMBLE_PY311_PYTHONPATH so those callers
# can prepend it to PYTHONPATH on a per-invocation basis. Kept as an
# env var (not auto-added to PYTHONPATH) so Kit's 3.12 interpreter never
# sees the ABI-wrong rclpy on its sys.path.

_PACKMAN_PY311=$(shopt -s nullglob; set -- "$HOME"/.cache/packman/chk/nv_ros2/humble_py_3.11_*; [ $# -gt 0 ] && printf '%s' "$1")
if [ -n "${_PACKMAN_PY311}" ]; then
    if [ -f "${_PACKMAN_PY311}/local/lib/python3.11/dist-packages/rclpy/__init__.py" ]; then
        export STRAFER_ROS2_HUMBLE_PY311_PYTHONPATH="${_PACKMAN_PY311}/local/lib/python3.11/dist-packages:${_PACKMAN_PY311}/lib/python3.11/site-packages"
        export STRAFER_ROS2_HUMBLE_PY311_LIB="${_PACKMAN_PY311}/lib"
    fi
fi
unset _PACKMAN_PY311

# ---------------------------------------------------------------------------
# Status report
# ---------------------------------------------------------------------------

echo "[env_setup] STRAFER_ROOT=${STRAFER_ROOT:-<unset>}"
echo "[env_setup] STRAFER_BLENDER_BIN=${STRAFER_BLENDER_BIN:-<unset>}"
echo "[env_setup] INFINIGEN_ROOT=${INFINIGEN_ROOT:-<unset>}"
echo "[env_setup] ISAACSIM_PATH=${ISAACSIM_PATH:-<unset>}"
echo "[env_setup] STRAFER_INFINIGEN_PYTHON=${STRAFER_INFINIGEN_PYTHON:-<unset>}"
echo "[env_setup] STRAFER_ISAACLAB_PYTHON=${STRAFER_ISAACLAB_PYTHON:-<unset>}"
echo "[env_setup] CONDA_ROOT=${CONDA_ROOT:-<unset>}"
echo "[env_setup] CONDA_ENV=${CONDA_ENV:-<unset>}"
echo "[env_setup] ISAACLAB=${ISAACLAB:-<unset>}"
echo "[env_setup] COLCON_WS=${COLCON_WS:-<unset>}"
echo "[env_setup] STRAFER_ROS2_HUMBLE_PY311_PYTHONPATH=${STRAFER_ROS2_HUMBLE_PY311_PYTHONPATH:-<unset>}"
echo "[env_setup] HF_HOME=${HF_HOME:-<unset>}"
echo "[env_setup] ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>}"
echo "[env_setup] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>}"
echo "[env_setup] STRAFER_JETSON_HOST=${STRAFER_JETSON_HOST:-<unset>}"

unset _REPO_ROOT
