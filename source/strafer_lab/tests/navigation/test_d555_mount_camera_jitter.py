# Copyright (c) 2025, Strafer Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-free checks for the rendered-camera arm of the D555 mount DR.

The mount offset is sampled once per episode and consumed twice: the IMU
observation rotates its readings through it, and the enriched depth variants
also point the rendered camera prim through it. These pin the composition sense
that makes those two the same misalignment, and pin which variants carry the
term — the vanilla and NOCAM arms must not, or their frozen contracts move.

Config construction and pure quaternion math only — no Kit / GPU.
"""
from __future__ import annotations

import math

import pytest
import torch

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

import strafer_lab.tasks.navigation.composed_env_cfg as composed
from strafer_lab.tasks.navigation.mdp.events import (
    d555_camera_mount_local_quat,
    jitter_d555_camera_prim_pose,
)

_TERM = "jitter_d555_camera_prim"

_ENRICHED = (
    "RLDepthEnriched_Real",
    "RLDepthEnriched_Robust",
    "RLDepthEnriched_Real_PLAY",
    "RLDepthEnriched_Robust_PLAY",
    "RLDepthSubgoalEnriched_Real",
    "RLDepthSubgoalEnriched_Robust",
    "RLDepthSubgoalEnriched_Real_PLAY",
    "RLDepthSubgoalEnriched_Robust_PLAY",
)
_UNENRICHED = (
    "RLDepth_Real",
    "RLDepth_Robust",
    "RLDepthSubgoal_Real",
    "RLDepthSubgoal_Robust",
    "RLNoCam",
    "RLNoCam_PLAY",
    "RLNoCamSubgoal_Real",
    "RLNoCamSubgoal_Robust",
)


def _events(name):
    return getattr(composed, f"StraferNavCfg_{name}")().events


def _quat_close(a, b, atol=1e-6):
    """Quaternions up to sign — ``q`` and ``-q`` are one rotation."""
    return torch.allclose(a, b, atol=atol) or torch.allclose(a, -b, atol=atol)


# ---------------------------------------------------------------------------
# Which variants carry the term
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ENRICHED)
def test_enriched_depth_variants_point_the_rendered_camera(name):
    assert hasattr(_events(name), _TERM)


@pytest.mark.parametrize("name", _UNENRICHED)
def test_unenriched_variants_leave_the_rendered_camera_alone(name):
    assert not hasattr(_events(name), _TERM)


@pytest.mark.parametrize("name", _ENRICHED)
def test_the_term_runs_after_the_offset_is_sampled(name):
    """It consumes the sampled offset, so it must not run before the draw."""
    fields = [k for k in _events(name).__dict__ if not k.startswith("_")]
    assert fields.index(_TERM) > fields.index("randomize_d555_mount")


def test_both_arms_read_one_band_per_realism_tier():
    """The IMU and the render must not be able to drift onto different bands."""
    for realism, expected in (("Real", 1.0), ("Robust", 3.0)):
        for prefix in ("RLDepth", "RLDepthEnriched"):
            events = _events(f"{prefix}_{realism}")
            assert events.randomize_d555_mount.params["max_angle_deg"] == expected
        # The rendered arm takes no band of its own — it consumes the draw.
        enriched = _events(f"RLDepthEnriched_{realism}")
        assert "max_angle_deg" not in getattr(enriched, _TERM).params


# ---------------------------------------------------------------------------
# Composition sense
# ---------------------------------------------------------------------------


def test_camera_prim_takes_the_inverse_of_the_observation_rotation():
    """``_d555_mount_quat`` maps a body vector to its components in the
    misaligned sensor frame, so the housing's own rotation — what the prim's
    frame orientation carries — is the inverse. Getting this backwards doubles
    the misalignment between the two sensors instead of matching them."""
    mount = quat_from_euler_xyz(
        torch.tensor([math.radians(1.0)]),
        torch.tensor([math.radians(-2.0)]),
        torch.tensor([math.radians(3.0)]),
    )
    nominal = torch.tensor([[-0.5, 0.5, -0.5, 0.5]])
    local = d555_camera_mount_local_quat(mount, nominal)

    housing = quat_mul(local, quat_inv(nominal))
    assert _quat_close(housing, quat_inv(mount))
    assert not _quat_close(housing, mount), "a nonzero offset must not be symmetric"

    # The reading the IMU reports is the body vector resolved on the rotated
    # housing's axes — the same statement, read off the prim.
    v = torch.tensor([[0.3, -0.7, 0.2]])
    assert torch.allclose(
        quat_apply(mount, v), quat_apply(quat_inv(housing), v), atol=1e-6
    )


def test_zero_offset_leaves_the_camera_at_its_nominal_mount():
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    nominal = torch.tensor([[-0.5, 0.5, -0.5, 0.5]])
    assert _quat_close(d555_camera_mount_local_quat(identity, nominal), nominal)


def test_offsets_are_per_env():
    mount = quat_from_euler_xyz(
        torch.tensor([0.0, 0.02]), torch.tensor([0.0, -0.01]), torch.tensor([0.0, 0.03])
    )
    nominal = torch.tensor([[-0.5, 0.5, -0.5, 0.5]])
    local = d555_camera_mount_local_quat(mount, nominal)
    assert _quat_close(local[0:1], nominal)
    assert not _quat_close(local[1:2], nominal)


# ---------------------------------------------------------------------------
# The no-op guards — the term is shared by variants that lack the camera
# ---------------------------------------------------------------------------


class _Scene:
    def __init__(self, sensors):
        self.sensors = sensors


class _Env:
    def __init__(self, sensors=None, mount_quat=None):
        self.scene = _Scene(sensors or {})
        if mount_quat is not None:
            self._d555_mount_quat = mount_quat


class _ExplodingCamera:
    cfg = None

    @property
    def _view(self):
        raise AssertionError("the camera must not be touched")


def test_no_op_without_env_ids():
    env = _Env({"d555_camera": _ExplodingCamera()}, torch.zeros(2, 4))
    jitter_d555_camera_prim_pose(env, torch.empty(0, dtype=torch.long))


def test_no_op_when_the_offset_was_never_sampled():
    env = _Env({"d555_camera": _ExplodingCamera()})
    jitter_d555_camera_prim_pose(env, torch.arange(2))


def test_no_op_when_the_scene_has_no_such_camera():
    env = _Env({}, torch.zeros(2, 4))
    jitter_d555_camera_prim_pose(env, torch.arange(2))
