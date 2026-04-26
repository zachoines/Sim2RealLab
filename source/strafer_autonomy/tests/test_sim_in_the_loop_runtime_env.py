"""Tests for strafer_lab.sim_in_the_loop.runtime_env.IsaacLabEnvAdapter.

Pure Python — runs in .venv_vlm via the strafer_lab namespace stub.
The adapter is exercised against a fake env handle that mimics the
Isaac Lab attribute surface (``unwrapped.scene[...]``, ``data.output``,
``data.root_pos_w``, etc.) plus a fake ``cmd_vel_reader`` and a fake
``torch`` module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from strafer_lab.sim_in_the_loop.runtime_env import IsaacLabEnvAdapter
from strafer_shared.constants import MAX_ANGULAR_VEL, MAX_LINEAR_VEL


# ---------------------------------------------------------------------------
# Tensor / scene fakes
# ---------------------------------------------------------------------------


class FakeTensor:
    """Mimics the subset of torch.Tensor the adapter uses.

    Supports ``[i]`` indexing, ``.shape``, ``.dim()``, ``.detach()``,
    ``.cpu()``, ``.numpy()``, ``.tolist()``, scalar ``__setitem__``,
    and ``.clone()``. Exposed via the FakeTorch.zeros / FakeTorch.tensor
    factories below.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data)
        self.shape = self.data.shape

    def dim(self) -> int:
        return self.data.ndim

    def __getitem__(self, key) -> "FakeTensor":
        return FakeTensor(self.data[key])

    def __setitem__(self, key, value) -> None:
        self.data[key] = value

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.data

    def tolist(self) -> list:
        return self.data.tolist()

    def clone(self) -> "FakeTensor":
        return FakeTensor(self.data.copy())


class FakeTorch:
    """Stand-in for the ``torch`` module the adapter constructor takes."""

    @staticmethod
    def zeros(shape, *, device=None) -> FakeTensor:  # noqa: ARG004 — device unused
        return FakeTensor(np.zeros(shape, dtype=np.float32))


# ---------------------------------------------------------------------------
# Env / scene / sensor fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeCameraData:
    output: dict = field(default_factory=dict)
    pos_w: Any = None
    quat_w_world: Any = None


@dataclass
class FakeCameraCfg:
    width: int = 640
    height: int = 360


@dataclass
class FakeCamera:
    data: FakeCameraData
    cfg: FakeCameraCfg = field(default_factory=FakeCameraCfg)


@dataclass
class FakeRobotData:
    root_pos_w: Any = None
    root_quat_w: Any = None


@dataclass
class FakeRobot:
    data: FakeRobotData


@dataclass
class FakeScene:
    sensors: dict = field(default_factory=dict)

    def __getitem__(self, key: str):
        return self.sensors[key]


@dataclass
class FakeActionManager:
    action: FakeTensor


@dataclass
class FakeUnwrapped:
    scene: FakeScene
    action_manager: FakeActionManager
    device: str = "cpu"


@dataclass
class FakeEnv:
    unwrapped: FakeUnwrapped
    resets: int = 0
    last_action: Any = None

    def reset(self) -> None:
        self.resets += 1

    def step(self, action) -> None:
        self.last_action = action


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_camera(width: int = 640, height: int = 360) -> FakeCamera:
    return FakeCamera(
        data=FakeCameraData(
            output={
                "rgb": FakeTensor(
                    np.full((1, height, width, 4), 200, dtype=np.uint8)
                ),
                "distance_to_image_plane": FakeTensor(
                    np.full((1, height, width, 1), 2.5, dtype=np.float32)
                ),
            },
            pos_w=FakeTensor(np.array([[0.2, 0.0, 0.25]], dtype=np.float32)),
            quat_w_world=FakeTensor(
                np.array([[0.5, -0.5, 0.5, -0.5]], dtype=np.float32)
            ),
        ),
        cfg=FakeCameraCfg(width=width, height=height),
    )


def _build_robot() -> FakeRobot:
    return FakeRobot(
        data=FakeRobotData(
            root_pos_w=FakeTensor(np.array([[1.0, 2.0, 0.05]], dtype=np.float32)),
            root_quat_w=FakeTensor(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            ),
        )
    )


def _build_env(*, action_dim: int = 3) -> FakeEnv:
    scene = FakeScene(
        sensors={
            "d555_camera_perception": _build_camera(),
            "robot": _build_robot(),
        }
    )
    action_tensor = FakeTensor(np.zeros((1, action_dim), dtype=np.float32))
    return FakeEnv(
        unwrapped=FakeUnwrapped(
            scene=scene,
            action_manager=FakeActionManager(action=action_tensor),
        )
    )


def _build_adapter(env: FakeEnv, **overrides) -> IsaacLabEnvAdapter:
    cmd_vel = overrides.pop(
        "cmd_vel_reader",
        lambda _gp: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )
    return IsaacLabEnvAdapter(
        env=env,
        graph_path="/World/Bridge",
        scene_name="kitchen_01",
        cmd_vel_reader=cmd_vel,
        torch_module=FakeTorch,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReset:
    def test_matching_scene_calls_env_reset(self):
        env = _build_env()
        adapter = _build_adapter(env)
        adapter.reset(scene_name="kitchen_01")
        assert env.resets == 1

    def test_mismatched_scene_raises(self):
        env = _build_env()
        adapter = _build_adapter(env)
        with pytest.raises(RuntimeError, match="Scene swap"):
            adapter.reset(scene_name="bedroom_02")
        assert env.resets == 0


class TestStep:
    def test_step_calls_env_with_built_action(self):
        env = _build_env()
        adapter = _build_adapter(
            env,
            cmd_vel_reader=lambda _gp: ((0.5, 0.1, 0.0), (0.0, 0.0, -0.3)),
        )
        adapter.step()
        assert env.last_action is not None
        action_data = env.last_action.data
        # Strafer mecanum layout: [linear_x, linear_y, angular_z]
        # /cmd_vel arrives in physical units (m/s, rad/s); the bridge
        # divides by MAX_LINEAR_VEL / MAX_ANGULAR_VEL so the action term
        # (which expects [-1, 1]) gets the contract it documents.
        assert action_data[0, 0] == pytest.approx(0.5 / MAX_LINEAR_VEL)
        assert action_data[0, 1] == pytest.approx(0.1 / MAX_LINEAR_VEL)
        assert action_data[0, 2] == pytest.approx(-0.3 / MAX_ANGULAR_VEL)

    def test_zero_cmd_vel_produces_zero_action(self):
        env = _build_env()
        adapter = _build_adapter(env)
        adapter.step()
        assert env.last_action is not None
        assert np.allclose(env.last_action.data, 0.0)

    def test_max_cmd_vel_normalizes_to_unit_action(self):
        """A /cmd_vel at the chassis cap maps to ±1.0 in the action tensor."""
        env = _build_env()
        adapter = _build_adapter(
            env,
            cmd_vel_reader=lambda _gp: (
                (MAX_LINEAR_VEL, -MAX_LINEAR_VEL, 0.0),
                (0.0, 0.0, MAX_ANGULAR_VEL),
            ),
        )
        adapter.step()
        action_data = env.last_action.data
        assert action_data[0, 0] == pytest.approx(1.0)
        assert action_data[0, 1] == pytest.approx(-1.0)
        assert action_data[0, 2] == pytest.approx(1.0)

    def test_above_cap_cmd_vel_clamps_to_unit(self):
        """A misbehaving publisher sending above-cap velocities is clamped."""
        env = _build_env()
        adapter = _build_adapter(
            env,
            cmd_vel_reader=lambda _gp: (
                (10.0 * MAX_LINEAR_VEL, 0.0, 0.0),
                (0.0, 0.0, -10.0 * MAX_ANGULAR_VEL),
            ),
        )
        adapter.step()
        action_data = env.last_action.data
        assert action_data[0, 0] == pytest.approx(1.0)
        assert action_data[0, 2] == pytest.approx(-1.0)

    def test_step_passes_graph_path_to_reader(self):
        env = _build_env()
        seen = []

        def reader(graph_path):
            seen.append(graph_path)
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        adapter = _build_adapter(env, cmd_vel_reader=reader)
        adapter.step()
        assert seen == ["/World/Bridge"]

    def test_action_dim_smaller_than_3_is_safely_skipped(self):
        env = _build_env(action_dim=2)
        adapter = _build_adapter(
            env,
            cmd_vel_reader=lambda _gp: ((0.5, 0.1, 0.0), (0.0, 0.0, -0.3)),
        )
        # Should not raise; action stays zeros because the layout assumption
        # ``shape[-1] >= 3`` is not met.
        adapter.step()
        assert env.last_action is not None
        assert np.allclose(env.last_action.data, 0.0)


class TestCapture:
    def test_returns_frame_bundle_with_image_dims(self):
        env = _build_env()
        adapter = _build_adapter(env)
        bundle = adapter.capture()
        assert bundle.image_width == 640
        assert bundle.image_height == 360

    def test_rgb_is_2d_or_3d_numpy(self):
        env = _build_env()
        adapter = _build_adapter(env)
        bundle = adapter.capture()
        assert bundle.rgb.shape == (360, 640, 4)
        assert bundle.rgb.dtype == np.uint8

    def test_depth_is_2d_float32(self):
        env = _build_env()
        adapter = _build_adapter(env)
        bundle = adapter.capture()
        assert bundle.depth is not None
        assert bundle.depth.shape == (360, 640)
        assert bundle.depth.dtype == np.float32

    def test_depth_disabled_when_save_depth_false(self):
        env = _build_env()
        adapter = _build_adapter(env, save_depth=False)
        bundle = adapter.capture()
        assert bundle.depth is None

    def test_robot_pose_extracted(self):
        env = _build_env()
        adapter = _build_adapter(env)
        bundle = adapter.capture()
        assert bundle.robot_pos == pytest.approx((1.0, 2.0, 0.05))
        assert bundle.robot_quat == pytest.approx((1.0, 0.0, 0.0, 0.0))

    def test_camera_pose_extracted(self):
        env = _build_env()
        adapter = _build_adapter(env)
        bundle = adapter.capture()
        assert bundle.cam_pos == pytest.approx((0.2, 0.0, 0.25))
        assert bundle.cam_quat == pytest.approx((0.5, -0.5, 0.5, -0.5))

    def test_missing_depth_output_returns_none(self):
        env = _build_env()
        # Drop the depth channel from the camera output.
        env.unwrapped.scene["d555_camera_perception"].data.output.pop(
            "distance_to_image_plane",
        )
        adapter = _build_adapter(env)
        bundle = adapter.capture()
        assert bundle.depth is None

    def test_bboxes_default_to_empty_list(self):
        env = _build_env()
        adapter = _build_adapter(env)
        bundle = adapter.capture()
        assert bundle.bboxes == []
