#!/usr/bin/env bash
# New-pair shell environment for the Stage 3 validation session.
#
# Usage:
#
#     source docs/measurements/isaac-lab-upgrade-stage3-2026-08-23/newpin_env.sh
#
# This is a measurement artifact, not repo configuration. It exists because
# `env_setup.sh` sources `.env` with `set -a`, and `.env` names the OLD pair
# (`env_isaaclab3` / `~/Documents/repos/IsaacLab`). Sourcing `env_setup.sh`
# therefore binds the old pair, and every Make target that sources it — or
# activates `$(CONDA_ENV)` — is unusable for a new-pin run without editing
# `.env`. The Stage 3 session validates the new pair *without* flipping the
# pointer, so the derivations `env_setup.sh` performs are mirrored here
# against the new prefix instead.
#
# Everything below is a line-for-line mirror of `env_setup.sh`, with the
# three pair-valued variables (STRAFER_ISAACLAB_PYTHON, CONDA_ENV, ISAACLAB)
# redirected to the new pair and every non-pair value taken verbatim from
# `.env`. Blender/Infinigen symlink handling is omitted: no leg of Stage 3
# invokes Infinigen, and creating a symlink is a side effect a measurement
# script should not have.

set -u

NEW_CONDA_ENV=env_isaaclab3beta2
NEW_ENV_PREFIX=/home/zachoines/miniconda3/envs/${NEW_CONDA_ENV}
NEW_LAB_ROOT=/home/zachoines/Documents/repos/IsaacLab-3beta2

# ---------------------------------------------------------------------------
# Activate the new conda env. `isaaclab.sh` resolves its interpreter from
# CONDA_PREFIX (isaaclab.sh:17-18), so activation is what binds the launcher
# to the new pair — not PATH order.
# ---------------------------------------------------------------------------

if [ "${CONDA_PREFIX:-}" != "${NEW_ENV_PREFIX}" ]; then
    # shellcheck disable=SC1091
    source /home/zachoines/miniconda3/etc/profile.d/conda.sh
    conda activate "${NEW_CONDA_ENV}"
fi

# ---------------------------------------------------------------------------
# `.env` values. Non-pair variables are verbatim; the three pair-valued ones
# point at the new pair. `.env` itself is NOT sourced — that is the whole
# point of this file.
# ---------------------------------------------------------------------------

export STRAFER_ROOT=/home/zachoines/Workspace/Sim2RealLab
export STRAFER_BLENDER_BIN=/home/zachoines/Workspace/blender-build/build_blender/bin/blender
export INFINIGEN_ROOT=/home/zachoines/Workspace/infinigen
export ISAACSIM_PATH=/home/zachoines/Workspace/IsaacSim/_build/linux-aarch64/release
export STRAFER_INFINIGEN_PYTHON=/home/zachoines/miniconda3/envs/env_infinigen/bin/python
export CONDA_ROOT=/home/zachoines/miniconda3
export COLCON_WS=/home/zachoines/strafer_ws
export HF_HOME=/home/zachoines/.cache/huggingface
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export STRAFER_JETSON_HOST=192.168.50.24
export STRAFER_JETSON_USER=jetson
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file:///home/zachoines/.config/cyclonedds/direct-link.xml}"

# --- the three pair-valued variables, redirected to the new pair ---
export STRAFER_ISAACLAB_PYTHON="${NEW_ENV_PREFIX}/bin/python"
export CONDA_ENV="${NEW_CONDA_ENV}"
export ISAACLAB="${NEW_LAB_ROOT}/isaaclab.sh"

# ---------------------------------------------------------------------------
# aarch64 libgomp preload (env_setup.sh's platform block). Consumed by ld.so
# at process start, so it cannot be set from inside Python.
# ---------------------------------------------------------------------------

_LIBGOMP=/lib/aarch64-linux-gnu/libgomp.so.1
if [ -f "${_LIBGOMP}" ]; then
    case "${LD_PRELOAD:-}" in
        *"${_LIBGOMP}"*) ;;
        "") export LD_PRELOAD="${_LIBGOMP}" ;;
        *) export LD_PRELOAD="${_LIBGOMP}:${LD_PRELOAD}" ;;
    esac
else
    echo "[newpin_env] WARNING: ${_LIBGOMP} not found" >&2
fi
unset _LIBGOMP

# ---------------------------------------------------------------------------
# Bundled-Humble payload from the NEW prefix (env_setup.sh's ROS block).
# Isaac Sim 6 ships Humble inside isaacsim.ros2.core; the extension activates
# it only when ROS_DISTRO=humble and its lib/ is already on LD_LIBRARY_PATH so
# the C++ OmniGraph publishers can dlopen their dependencies.
# ---------------------------------------------------------------------------

_HUMBLE_LIB="${NEW_ENV_PREFIX}/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/humble/lib"
if [ -f "${_HUMBLE_LIB}/librmw_cyclonedds_cpp.so" ]; then
    # Keep restore_ros2_python_paths() a no-op so the extension goes straight
    # to its bundled 3.12 rclpy instead of a packman 3.11 one.
    unset OLD_PYTHONPATH AMENT_PREFIX_PATH AMENT_CURRENT_PREFIX
    if [ -n "${PYTHONPATH:-}" ]; then
        _CLEANED=
        IFS=':'; for _entry in ${PYTHONPATH}; do
            case "${_entry}" in
                *"/packman/chk/nv_ros2/humble_py_3.11_"*) ;;
                *) _CLEANED="${_CLEANED:+${_CLEANED}:}${_entry}" ;;
            esac
        done
        unset IFS
        if [ -n "${_CLEANED}" ]; then export PYTHONPATH="${_CLEANED}"; else unset PYTHONPATH; fi
        unset _CLEANED _entry
    fi
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${_HUMBLE_LIB}:"*) ;;
        *) export LD_LIBRARY_PATH="${_HUMBLE_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    export ROS_DISTRO="${ROS_DISTRO:-humble}"
else
    echo "[newpin_env] WARNING: no bundled Humble payload under ${NEW_ENV_PREFIX}" >&2
fi
unset _HUMBLE_LIB

# ---------------------------------------------------------------------------
# Python 3.11 Humble bundle for off-Kit clients (env_setup.sh's opt-in block).
# Exported, never auto-added to PYTHONPATH: its rclpy is ABI-wrong for Kit.
# ---------------------------------------------------------------------------

_PACKMAN_PY311=$(shopt -s nullglob; set -- "$HOME"/.cache/packman/chk/nv_ros2/humble_py_3.11_*; [ $# -gt 0 ] && printf '%s' "$1")
if [ -n "${_PACKMAN_PY311}" ] && [ -f "${_PACKMAN_PY311}/local/lib/python3.11/dist-packages/rclpy/__init__.py" ]; then
    export STRAFER_ROS2_HUMBLE_PY311_PYTHONPATH="${_PACKMAN_PY311}/local/lib/python3.11/dist-packages:${_PACKMAN_PY311}/lib/python3.11/site-packages"
    export STRAFER_ROS2_HUMBLE_PY311_LIB="${_PACKMAN_PY311}/lib"
fi
unset _PACKMAN_PY311

# Isaac tears processes down with os._exit, which discards buffered stdout.
export PYTHONUNBUFFERED=1

# Convenience handles used by the session's commands.
export NEWPY="${NEW_ENV_PREFIX}/bin/python"
export NEWLAB="${NEW_LAB_ROOT}/isaaclab.sh"

set +u

echo "[newpin_env] CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "[newpin_env] NEWPY=${NEWPY}"
echo "[newpin_env] NEWLAB=${NEWLAB}"
echo "[newpin_env] STRAFER_ISAACLAB_PYTHON=${STRAFER_ISAACLAB_PYTHON}"
echo "[newpin_env] ISAACLAB=${ISAACLAB}"
echo "[newpin_env] CONDA_ENV=${CONDA_ENV}"
echo "[newpin_env] LD_PRELOAD=${LD_PRELOAD:-<unset>}"
echo "[newpin_env] ROS_DISTRO=${ROS_DISTRO:-<unset>}"
echo "[newpin_env] RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<unset>}"
echo "[newpin_env] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo "[newpin_env] PYTHONPATH=${PYTHONPATH:-<unset>}"
echo "[newpin_env] CYCLONEDDS_URI=${CYCLONEDDS_URI:-<unset>}"
