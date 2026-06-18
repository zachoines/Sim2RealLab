"""Runtime ``EnvAdapter`` that drives a live Isaac Lab env.

Wraps a single instantiated ``Isaac-Strafer-Nav-Capture-Bridge-v0`` env
(or compatible variant carrying a ``d555_camera_perception`` sensor)
and exposes the :class:`strafer_lab.sim_in_the_loop.harness.EnvAdapter`
protocol the orchestrator expects.

What ``step()`` does:
  - Samples the latest ``/cmd_vel`` Twist from ``cmd_vel_source`` (the
    bridge's rclpy subscription).
  - Normalizes it into the Strafer mecanum action contract
    ``[linear_x, linear_y, angular_z]`` in ``[-1, 1]`` and calls
    ``env.step(action)`` once.
  - Advances the adapter's sim-time accumulator and invokes
    ``on_stepped(sim_time_s)`` so the launch script can pump the
    telemetry publisher, the camera publisher, and the Kit app in
    lock-step with the env — the same per-tick protocol as the bridge
    mainloop. Sim time accumulates across episodes (``reset`` does not
    rewind it) so the published ``/clock`` stays monotonic.

What ``capture()`` does:
  - Pulls the camera channels selected by ``cameras_required`` (the
    same token vocabulary as the env composition and the LeRobot
    writer), the robot pose + body-frame velocity, the most recent
    normalized action, and — when a ``detections_source`` is wired —
    the perception camera's parsed 2D bboxes.
  - Wraps everything in a :class:`FrameBundle` for the recorder.

Scene swap policy: this adapter is single-scene. ``reset(scene_name=...)``
verifies the requested scene matches the one the env was built with —
swapping Infinigen scene USDs at runtime requires re-instantiating the
env, which the launch script handles by re-launching itself per scene.

Heavy imports (``torch``, ``warp``) are deferred / injectable so this
module is importable from a plain Python environment for unit testing.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from strafer_lab.sim_in_the_loop.harness import EnvAdapter, FrameBundle
from strafer_shared.constants import MAX_ANGULAR_VEL, MAX_LINEAR_VEL


# Returns ``((lx, ly, lz), (ax, ay, az))`` in m/s and rad/s — the shape
# of ``StraferAsyncPublisher.get_cmd_vel``.
CmdVelSource = Callable[[], tuple[tuple[float, float, float], tuple[float, float, float]]]


def _clamp_unit(value: float) -> float:
    """Clamp ``value`` into the action term's ``[-1, 1]`` contract."""
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return value


class IsaacLabEnvAdapter(EnvAdapter):
    """``EnvAdapter`` backed by a live Isaac Lab env + the ROS2 bridge.

    Parameters
    ----------
    env :
        The gym env returned by ``gym.make(task, cfg=env_cfg)``. Must
        expose ``unwrapped.scene["d555_camera_perception"]``,
        ``unwrapped.scene["robot"]``, and ``unwrapped.action_manager``.
    scene_name :
        Name the harness will pass to ``reset(scene_name=...)``. Used
        only as a sanity check — the env was built around one scene
        USD and this adapter does not swap it.
    cmd_vel_source :
        Callable returning the latest Twist sample. The launch script
        wires this to ``StraferAsyncPublisher.get_cmd_vel``; tests
        inject a fake.
    cameras_required :
        Sensor-stack tokens over ``rgb_full`` / ``depth_full`` /
        ``rgb_policy`` / ``depth_policy`` deciding which camera channels
        ``capture()`` reads. Must match the writer's declared stack.
    detections_source :
        Optional callable returning the current frame's parsed bboxes
        (``ReplicatorBboxExtractor.extract``). ``None`` leaves
        ``FrameBundle.detections`` as ``None``.
    on_stepped :
        Optional per-step callback receiving the accumulated sim time;
        the launch script composes telemetry publish + camera notify +
        the Kit app pump here.
    step_dt :
        Sim seconds advanced per ``env.step``. Derived from the env's
        physics dt × decimation when omitted; tests pass it explicitly.
    torch_module :
        Used to construct the action tensor. Defaults to ``import torch``.
    """

    _ACTION_LINEAR_X = 0
    _ACTION_LINEAR_Y = 1
    _ACTION_ANGULAR_Z = 2

    def __init__(
        self,
        *,
        env: Any,
        scene_name: str,
        cmd_vel_source: CmdVelSource,
        cameras_required: Sequence[str] = ("rgb_full", "depth_full", "depth_policy"),
        perception_sensor_key: str = "d555_camera_perception",
        policy_sensor_key: str = "d555_camera",
        robot_sensor_key: str = "robot",
        detections_source: Callable[[], Sequence[Any]] | None = None,
        on_stepped: Callable[[float], None] | None = None,
        step_dt: float | None = None,
        torch_module: Any = None,
    ) -> None:
        self._env = env
        self._scene_name = scene_name
        self._cmd_vel_source = cmd_vel_source
        self._cameras_required = tuple(cameras_required)
        self._perception_key = perception_sensor_key
        self._policy_key = policy_sensor_key
        self._robot_key = robot_sensor_key
        self._detections_source = detections_source
        self._on_stepped = on_stepped

        if torch_module is None:
            import torch as _torch  # noqa: PLC0415 — deferred so plain-Python tests work
            torch_module = _torch
        self._torch = torch_module

        unwrapped = self._env.unwrapped
        if step_dt is None:
            step_dt = float(unwrapped.sim.get_physics_dt()) * int(unwrapped.cfg.decimation)
        self._step_dt = float(step_dt)
        self._sim_time_s = 0.0
        self._last_action = (0.0, 0.0, 0.0)

        self._device = unwrapped.device
        self._action_shape = tuple(
            int(d) for d in unwrapped.action_manager.action.shape
        )
        self._zero_action = self._torch.zeros(
            self._action_shape, device=self._device,
        )

    @property
    def sim_time_s(self) -> float:
        return self._sim_time_s

    # ------------------------------------------------------------------
    # EnvAdapter protocol
    # ------------------------------------------------------------------

    def reset(self, *, scene_name: str) -> None:
        if scene_name != self._scene_name:
            raise RuntimeError(
                f"IsaacLabEnvAdapter is bound to scene {self._scene_name!r}; "
                f"got reset request for {scene_name!r}. "
                "Scene swap requires re-instantiating the env — re-launch the "
                "script with --scene-name/--scene-usd pointing at the new scene."
            )
        self._env.reset()
        self._last_action = (0.0, 0.0, 0.0)
        # One zero-action warm-up step so the tiled cameras hold a freshly
        # rendered frame before the harness captures the starting view.
        self._env.step(self._zero_action.clone())
        self._sim_time_s += self._step_dt
        if self._on_stepped is not None:
            self._on_stepped(self._sim_time_s)

    def step(self) -> None:
        linear, angular = self._cmd_vel_source()
        action = self._build_action(linear, angular)
        self._env.step(action)
        self._sim_time_s += self._step_dt
        if self._on_stepped is not None:
            self._on_stepped(self._sim_time_s)

    def capture(self) -> FrameBundle:
        scene = self._env.unwrapped.scene
        robot = scene[self._robot_key]

        rgb_np = None
        depth_np = None
        rgb_policy_np = None
        depth_policy_np = None
        if "rgb_full" in self._cameras_required or "depth_full" in self._cameras_required:
            camera = scene[self._perception_key]
            if "rgb_full" in self._cameras_required:
                rgb_np = self._to_uint8_hwc(camera.data.output["rgb"])
            if "depth_full" in self._cameras_required:
                depth_np = self._to_float32_hw(
                    camera.data.output["distance_to_image_plane"],
                )
        if "rgb_policy" in self._cameras_required or "depth_policy" in self._cameras_required:
            policy_camera = scene[self._policy_key]
            if "rgb_policy" in self._cameras_required:
                rgb_policy_np = self._to_uint8_hwc(policy_camera.data.output["rgb"])
            if "depth_policy" in self._cameras_required:
                depth_policy_np = self._to_float32_hw(
                    policy_camera.data.output["distance_to_image_plane"],
                )

        lin_b = self._to_torch(robot.data.root_lin_vel_b)[0].detach().cpu().tolist()
        ang_w = self._to_torch(robot.data.root_ang_vel_w)[0].detach().cpu().tolist()
        detections = (
            list(self._detections_source())
            if self._detections_source is not None else None
        )

        return FrameBundle(
            rgb=rgb_np,
            depth=depth_np,
            robot_pos=self._tensor_to_xyz(robot.data.root_pos_w),
            robot_quat=self._tensor_to_xyzw(robot.data.root_quat_w),
            achieved_vel=(float(lin_b[0]), float(lin_b[1]), float(ang_w[2])),
            action=self._last_action,
            sim_time_s=self._sim_time_s,
            depth_policy=depth_policy_np,
            rgb_policy=rgb_policy_np,
            detections=detections,
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

        ``MecanumWheelAction.process_actions`` consumes actions in the
        normalized ``[-1, 1]`` contract and scales them up to physical
        velocities via ``_velocity_scale = [MAX_LINEAR_VEL, MAX_LINEAR_VEL,
        MAX_ANGULAR_VEL]``. ``/cmd_vel`` from Nav2 / teleop arrives in
        physical units (m/s, rad/s), so this method divides by the same
        ``strafer_shared.constants`` values that the action term scales
        with — keeping the bridge-side and env-side conversions on a
        single source of truth. Out-of-band velocities are clamped to
        ``[-1, 1]`` here for explicitness; ``process_actions`` clamps too,
        but expressing intent in the bridge stops a bad ``/cmd_vel``
        publisher from saturating the per-wheel motor cap silently.

        The action layout assumed here is the Strafer mecanum convention
        ``[linear_x, linear_y, angular_z]`` at indices 0, 1, 2 of the
        first env's action row. Variants whose action layout differs
        (e.g. wheel-velocity actions) need a different adapter.
        """

        vx = _clamp_unit(float(linear[0]) / MAX_LINEAR_VEL)
        vy = _clamp_unit(float(linear[1]) / MAX_LINEAR_VEL)
        wz = _clamp_unit(float(angular[2]) / MAX_ANGULAR_VEL)
        self._last_action = (vx, vy, wz)

        action = self._zero_action.clone()
        if action.shape[-1] >= 3:
            action[0, self._ACTION_LINEAR_X] = vx
            action[0, self._ACTION_LINEAR_Y] = vy
            action[0, self._ACTION_ANGULAR_Z] = wz
        return action

    def _to_torch(self, arr: Any) -> Any:
        """Coerce a ``wp.array`` or ``torch.Tensor`` to a torch tensor.

        Isaac Lab sensor / articulation outputs may be warp arrays,
        which no longer support item indexing directly.
        """
        if hasattr(arr, "detach"):
            return arr
        import warp as wp  # noqa: PLC0415 — only reached on live Isaac Sim data

        return wp.to_torch(arr)

    def _to_uint8_hwc(self, tensor: Any) -> Any:
        """``(N, H, W, C)`` RGB(A) → contiguous ``(H, W, 3)`` uint8."""
        import numpy as np  # noqa: PLC0415 — keep top-of-file imports light

        t = self._to_torch(tensor)
        arr = (t[0] if t.dim() == 4 else t).detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def _to_float32_hw(self, tensor: Any) -> Any:
        """``(N, H, W, 1)`` or ``(N, H, W)`` distance → ``(H, W)`` float32."""
        import numpy as np  # noqa: PLC0415 — keep top-of-file imports light

        t = self._to_torch(tensor)
        arr = (t[0] if t.dim() == 4 else t).detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr.astype(np.float32)

    def _tensor_to_xyz(self, tensor: Any) -> tuple[float, float, float]:
        vals = self._to_torch(tensor)[0].detach().cpu().tolist()
        return (float(vals[0]), float(vals[1]), float(vals[2]))

    def _tensor_to_xyzw(self, tensor: Any) -> tuple[float, float, float, float]:
        vals = self._to_torch(tensor)[0].detach().cpu().tolist()
        return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
