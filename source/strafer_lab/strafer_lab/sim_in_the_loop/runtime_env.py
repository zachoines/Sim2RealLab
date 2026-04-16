"""Runtime ``EnvAdapter`` that drives a live Isaac Lab env.

This wraps a single instantiated ``Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0``
env (or compatible variant carrying a ``d555_camera_perception`` sensor)
and exposes the :class:`strafer_lab.sim_in_the_loop.harness.EnvAdapter`
protocol the orchestrator expects.

What ``step()`` does:
  - Reads ``/cmd_vel`` from the bridge OmniGraph (via
    :func:`strafer_lab.bridge.graph.read_cmd_vel`).
  - Maps the Twist into the Strafer mecanum action layout
    ``[linear_x, linear_y, angular_z]``.
  - Calls ``env.step(action)`` once.

What ``capture()`` does:
  - Pulls RGB + depth from ``scene["d555_camera_perception"]``.
  - Reads robot + camera world-frame poses from the env scene handles.
  - Wraps everything in a :class:`FrameBundle` for the writer.

Scene swap policy: this adapter is single-scene. ``reset(scene_name=...)``
verifies the requested scene matches the one the env was built with —
swapping Infinigen scene USDs at runtime requires re-instantiating the
env, which the launch script handles by re-launching itself per scene.

Imports of ``torch`` and ``strafer_lab.bridge.graph`` are deferred to
the constructor (and the ``cmd_vel_reader`` is injectable) so this
module is importable from a plain Python environment for unit testing.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from strafer_lab.sim_in_the_loop.harness import EnvAdapter, FrameBundle


# Type alias for the cmd_vel reader. Returns
# ``((lx, ly, lz), (ax, ay, az))`` in m/s and rad/s.
CmdVelReader = Callable[[str], tuple[tuple[float, float, float], tuple[float, float, float]]]


class IsaacLabEnvAdapter(EnvAdapter):
    """``EnvAdapter`` backed by a live Isaac Lab env + the ROS2 bridge graph.

    Parameters
    ----------
    env :
        The gym env returned by ``gym.make(task, cfg=env_cfg)``. Must
        expose ``unwrapped.scene["d555_camera_perception"]``,
        ``unwrapped.scene["robot"]``, and ``unwrapped.action_manager``.
    graph_path :
        USD path of the bridge OmniGraph (matches what
        ``strafer_lab.bridge.config.build_default_bridge_config`` was
        constructed with). Used to read ``/cmd_vel`` each step.
    scene_name :
        Name the harness will pass to ``reset(scene_name=...)``. Used
        only as a sanity check — the env was built around one scene
        USD and this adapter does not swap it.
    perception_sensor_key :
        Key under ``scene[...]`` for the 640x360 perception camera.
        Defaults to the canonical ``"d555_camera_perception"``.
    robot_sensor_key :
        Key under ``scene[...]`` for the robot articulation.
    cmd_vel_reader :
        Callable that reads the latest Twist from the bridge graph.
        Defaults to :func:`strafer_lab.bridge.graph.read_cmd_vel`.
        Tests inject a fake.
    torch_module :
        Used to construct the action tensor. Defaults to ``import torch``.
    save_depth :
        If True, ``capture()`` includes the ``distance_to_image_plane``
        depth array. If False, depth is left as ``None``.
    """

    _ACTION_LINEAR_X = 0
    _ACTION_LINEAR_Y = 1
    _ACTION_ANGULAR_Z = 2

    def __init__(
        self,
        *,
        env: Any,
        graph_path: str,
        scene_name: str,
        perception_sensor_key: str = "d555_camera_perception",
        robot_sensor_key: str = "robot",
        cmd_vel_reader: CmdVelReader | None = None,
        torch_module: Any = None,
        save_depth: bool = True,
    ) -> None:
        self._env = env
        self._graph_path = graph_path
        self._scene_name = scene_name
        self._perception_key = perception_sensor_key
        self._robot_key = robot_sensor_key
        self._save_depth = bool(save_depth)

        if cmd_vel_reader is None:
            from strafer_lab.bridge.graph import read_cmd_vel as _read_cmd_vel
            cmd_vel_reader = _read_cmd_vel
        self._cmd_vel_reader = cmd_vel_reader

        if torch_module is None:
            import torch as _torch  # noqa: PLC0415 — deferred so .venv_vlm tests work
            torch_module = _torch
        self._torch = torch_module

        self._device = self._env.unwrapped.device
        self._action_shape = tuple(
            int(d) for d in self._env.unwrapped.action_manager.action.shape
        )
        self._zero_action = self._torch.zeros(
            self._action_shape, device=self._device,
        )

    # ------------------------------------------------------------------
    # EnvAdapter protocol
    # ------------------------------------------------------------------

    def reset(self, *, scene_name: str) -> None:
        if scene_name != self._scene_name:
            raise RuntimeError(
                f"IsaacLabEnvAdapter is bound to scene {self._scene_name!r}; "
                f"got reset request for {scene_name!r}. "
                "Scene swap requires re-instantiating the env — re-launch the "
                "script with --scene/--scene-usd pointing at the new scene."
            )
        self._env.reset()

    def step(self) -> None:
        linear, angular = self._cmd_vel_reader(self._graph_path)
        action = self._build_action(linear, angular)
        self._env.step(action)

    def capture(self) -> FrameBundle:
        scene = self._env.unwrapped.scene
        camera = scene[self._perception_key]
        robot = scene[self._robot_key]

        rgb_np = self._tensor_to_numpy_image(camera.data.output["rgb"])
        depth_np = None
        if self._save_depth:
            depth_out = camera.data.output.get("distance_to_image_plane")
            if depth_out is not None:
                depth_np = self._tensor_to_numpy_depth(depth_out)

        robot_pos = self._tensor_to_xyz(robot.data.root_pos_w)
        robot_quat = self._tensor_to_xyzw(robot.data.root_quat_w)
        cam_pos = self._tensor_to_xyz(camera.data.pos_w)
        cam_quat = self._tensor_to_xyzw(camera.data.quat_w_world)

        return FrameBundle(
            rgb=rgb_np,
            depth=depth_np,
            robot_pos=robot_pos,
            robot_quat=robot_quat,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            bboxes=[],
            image_width=int(camera.cfg.width),
            image_height=int(camera.cfg.height),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_action(
        self,
        linear: Sequence[float],
        angular: Sequence[float],
    ) -> Any:
        """Convert one Twist sample into the env's action tensor.

        The action layout assumed here is the Strafer mecanum convention
        ``[linear_x, linear_y, angular_z]`` at indices 0, 1, 2 of the
        first env's action row. Variants whose action layout differs
        (e.g. wheel-velocity actions) need a different adapter.
        """

        action = self._zero_action.clone()
        if action.shape[-1] >= 3:
            action[0, self._ACTION_LINEAR_X] = float(linear[0])
            action[0, self._ACTION_LINEAR_Y] = float(linear[1])
            action[0, self._ACTION_ANGULAR_Z] = float(angular[2])
        return action

    @staticmethod
    def _tensor_to_numpy_image(tensor: Any) -> Any:
        """``(num_envs, H, W, C)`` torch tensor → ``(H, W, C)`` numpy."""

        if tensor.dim() == 4:
            arr = tensor[0].detach().cpu().numpy()
        else:
            arr = tensor.detach().cpu().numpy()
        return arr

    @staticmethod
    def _tensor_to_numpy_depth(tensor: Any) -> Any:
        """``(num_envs, H, W, 1)`` or ``(num_envs, H, W)`` → ``(H, W)`` float32."""

        import numpy as np  # noqa: PLC0415 — keep top-of-file imports light

        if tensor.dim() == 4:
            arr = tensor[0].detach().cpu().numpy()
        else:
            arr = tensor.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr.astype(np.float32)

    @staticmethod
    def _tensor_to_xyz(tensor: Any) -> tuple[float, float, float]:
        vals = tensor[0].detach().cpu().tolist()
        return (float(vals[0]), float(vals[1]), float(vals[2]))

    @staticmethod
    def _tensor_to_xyzw(tensor: Any) -> tuple[float, float, float, float]:
        vals = tensor[0].detach().cpu().tolist()
        return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
