"""Isaac Lab version compatibility layer (2.x ↔ 3.0).

Isaac Lab 3.0 introduces several breaking changes:
  - Quaternion convention: WXYZ → XYZW
  - Data accessors return ``wp.array`` instead of ``torch.Tensor``
  - ``write_*_to_sim()`` → ``write_*_to_sim_index()``
  - ``root_physx_view`` → ``root_view``
  - ``object_link_*`` → ``body_link_*``

This module detects the installed version at import time and exposes
helpers that work on both versions, so the rest of the codebase never
touches version-specific APIs directly.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------

_ISAACLAB_V3: bool = False
try:
    import isaaclab as _il

    _v = getattr(_il, "__version__", "0.0.0")
    _ISAACLAB_V3 = int(_v.split(".")[0]) >= 3
except Exception:
    pass

# ---------------------------------------------------------------------------
# Quaternion convention
# ---------------------------------------------------------------------------
# QW/QX/QY/QZ give the *index* within a 4-element quaternion for each
# component, accounting for the ordering used by the installed version.
#   2.x stores (w, x, y, z) → QW=0, QX=1, QY=2, QZ=3
#   3.0 stores (x, y, z, w) → QW=3, QX=0, QY=1, QZ=2

if _ISAACLAB_V3:
    QW, QX, QY, QZ = 3, 0, 1, 2
else:
    QW, QX, QY, QZ = 0, 1, 2, 3

# Pre-built tuples for config / ``OffsetCfg.rot`` usage.
IDENTITY_QUAT: tuple[float, float, float, float] = (
    (0.0, 0.0, 0.0, 1.0) if _ISAACLAB_V3 else (1.0, 0.0, 0.0, 0.0)
)


def make_quat_tuple(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
    """Build a quaternion tuple in the native convention from (x, y, z, w)."""
    if _ISAACLAB_V3:
        return (x, y, z, w)
    return (w, x, y, z)


def extract_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw from a quaternion tensor (any convention)."""
    w = quat[..., QW]
    x = quat[..., QX]
    y = quat[..., QY]
    z = quat[..., QZ]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat_tensor(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles → quaternion tensor in the native convention."""
    q = torch.zeros(*yaw.shape, 4, device=yaw.device)
    q[..., QW] = torch.cos(yaw / 2)
    q[..., QZ] = torch.sin(yaw / 2)
    return q


def set_identity_quat(state: torch.Tensor, offset: int = 3) -> None:
    """Write an identity quaternion into ``state[..., offset:offset+4]``."""
    state[..., offset : offset + 4] = 0.0
    state[..., offset + QW] = 1.0


def set_yaw_quat(state: torch.Tensor, yaw: torch.Tensor, offset: int = 3) -> None:
    """Write a yaw-only quaternion into ``state[..., offset:offset+4]``."""
    state[..., offset : offset + 4] = 0.0
    state[..., offset + QW] = torch.cos(yaw / 2)
    state[..., offset + QZ] = torch.sin(yaw / 2)


# ---------------------------------------------------------------------------
# Data tensor access
# ---------------------------------------------------------------------------


def ensure_torch(x: object) -> torch.Tensor:
    """Guarantee *x* is a ``torch.Tensor``.

    On Isaac Lab 2.x ``.data.*`` already returns ``torch.Tensor`` — pass through.
    On 3.0 it returns ``wp.array`` — convert via ``wp.to_torch()``.
    """
    if isinstance(x, torch.Tensor):
        return x
    import warp as wp  # deferred so 2.x never pays the import cost

    return wp.to_torch(x)


# ---------------------------------------------------------------------------
# Asset write helpers
# ---------------------------------------------------------------------------


def write_root_state(asset: object, state: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    """Write root state (pos+quat+vel) to sim, splitting into pose/velocity calls on 3.0."""
    if hasattr(asset, "write_root_pose_to_sim_index"):
        # Isaac Lab 3.0 develop: separate pose (7) and velocity (6) writes
        asset.write_root_pose_to_sim_index(root_pose=state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=state[:, 7:], env_ids=env_ids)
    elif env_ids is not None:
        asset.write_root_state_to_sim(state, env_ids)
    else:
        asset.write_root_state_to_sim(state)


def write_joint_state(
    asset: object,
    pos: torch.Tensor,
    vel: torch.Tensor,
    env_ids: torch.Tensor | None = None,
) -> None:
    """Write joint position and velocity to sim."""
    if hasattr(asset, "write_joint_position_to_sim_index"):
        # Isaac Lab 3.0 develop: separate position and velocity writes
        asset.write_joint_position_to_sim_index(position=pos, env_ids=env_ids)
        asset.write_joint_velocity_to_sim_index(velocity=vel, env_ids=env_ids)
    else:
        asset.write_joint_state_to_sim(pos, vel, env_ids=env_ids)


# ---------------------------------------------------------------------------
# PhysX / root view
# ---------------------------------------------------------------------------


def get_root_view(asset: object):
    """Return ``asset.root_view`` (3.0) or ``asset.root_physx_view`` (2.x)."""
    if hasattr(asset, "root_view"):
        return asset.root_view
    return asset.root_physx_view


# ---------------------------------------------------------------------------
# RigidObjectCollection helpers
# ---------------------------------------------------------------------------


def get_body_link_pos_w(collection: object):
    d = collection.data
    return getattr(d, "body_link_pos_w", None) or d.object_link_pos_w


def get_body_link_quat_w(collection: object):
    d = collection.data
    return getattr(d, "body_link_quat_w", None) or d.object_link_quat_w


def write_body_link_pose(collection: object, poses, env_ids, object_ids) -> None:
    if hasattr(collection, "write_body_link_pose_to_sim_index"):
        collection.write_body_link_pose_to_sim_index(poses, env_ids, object_ids)
    else:
        collection.write_object_link_pose_to_sim(poses, env_ids, object_ids)
