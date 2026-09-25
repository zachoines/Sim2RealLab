"""The obs_parity CLI's bag self-check on a synthetic Z16 bag.

Loads ``scripts/obs_parity.py`` by path and feeds ``_self_check_stream`` real
message objects in place of a rosbag, so the decode and the validity mask it
hands to ``reassemble_obs_from_extracted`` are exercised end to end. Needs
rclpy and tf2_ros (``bag_io`` builds a tf2 buffer), so it lives apart from
the rclpy-free ``test_parity``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tf2_ros")

from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import Image, Imu, JointState  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from strafer_inference import parity as P  # noqa: E402
from strafer_shared.constants import (  # noqa: E402
    DEPTH_MAX,
    DEPTH_NEARFIELD_FILL,
    DEPTH_SCALE,
    PERCEPTION_HEIGHT,
    PERCEPTION_WIDTH,
    WHEEL_JOINT_NAMES,
)
from strafer_shared.policy_interface import PolicyVariant  # noqa: E402

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "obs_parity.py"
_T = 1.0


def _load_obs_parity():
    spec = importlib.util.spec_from_file_location("obs_parity_cli", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stamp(msg, t: float = _T):
    msg.header.stamp.sec = int(t)
    msg.header.stamp.nanosec = int(round((t % 1.0) * 1e9))
    return msg


def _z16_frame(raw_mm: np.ndarray) -> Image:
    msg = Image()
    msg.encoding = "16UC1"
    msg.height, msg.width = raw_mm.shape
    msg.step = raw_mm.shape[1] * 2
    msg.data = raw_mm.astype("<u2").tobytes()
    return _stamp(msg)


def _bag(depth: Image) -> dict:
    mod = _load_obs_parity()
    imu = _stamp(Imu())
    imu.linear_acceleration.z = 9.81
    js = _stamp(JointState())
    js.name = list(WHEEL_JOINT_NAMES)
    js.velocity = [0.0, 0.0, 0.0, 0.0]
    tf = _stamp(TransformStamped())
    tf.header.frame_id = "map"
    tf.child_frame_id = "base_link"
    tf.transform.rotation.w = 1.0
    return {
        mod._TOPIC_IMU: [imu],
        mod._TOPIC_JOINTS: [js],
        mod._TOPIC_ODOM: [_stamp(Odometry())],
        mod._TOPIC_DEPTH: [depth],
        "/tf": [TFMessage(transforms=[tf])],
        "/tf_static": [],
    }


def _node_stream() -> P.ObsStream:
    v = PolicyVariant.DEPTH_SUBGOAL
    return P.parse_obs_records(
        [{
            "t_sim": _T,
            "variant": v.name,
            "obs": [0.0] * v.obs_dim,
            "referent": {"x": 1.0, "y": 0.0},
        }],
        source="node",
    )


def _self_check_depth(depth: Image) -> tuple[np.ndarray, int]:
    mod = _load_obs_parity()
    ref, dropped = mod._self_check_stream(
        _node_stream(), _bag(depth), map_frame="map", base_frame="base_link"
    )
    _, (dstart, dstop) = P.split_indices(ref.variant)
    return ref.obs[:, dstart:dstop], dropped


def test_self_check_masks_z16_zeros_as_depth_max():
    raw = np.full((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), 2500, np.uint16)
    raw[0:8, 0:8] = 0
    depth_obs, dropped = _self_check_depth(_z16_frame(raw))
    assert dropped == 0
    assert depth_obs.shape[0] == 1
    assert depth_obs[0, 0] == pytest.approx(DEPTH_MAX * DEPTH_SCALE, abs=1e-6)
    assert depth_obs[0, 0] != pytest.approx(
        DEPTH_NEARFIELD_FILL * DEPTH_SCALE, abs=1e-6
    )
    np.testing.assert_allclose(depth_obs[0, 1:], 2.5 * DEPTH_SCALE, atol=1e-6)


def test_self_check_drops_a_frame_the_node_would_drop():
    raw = np.full((PERCEPTION_HEIGHT, PERCEPTION_WIDTH), 2500, np.uint16)
    bad = _z16_frame(raw)
    bad.encoding = "rgb8"
    with pytest.raises(SystemExit):
        _self_check_depth(bad)
