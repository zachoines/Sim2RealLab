"""Live start-frame provider for the mission generator's grounding pass.

The mission generator
(:func:`strafer_lab.tools.build_mission_queue.build_mission_queue`) takes an
optional ``grounding_frame_provider``: a callable ``(obj, start_pose) -> frame``
invoked inside ``_ground_start_frame`` to source the egocentric image a VL model
judges target visibility against. This module builds that provider so it renders
the frame in-process, from the robot's ``d555_camera_perception`` camera, at the
exact ``start_pose`` the generator just computed.

Rendering at generation time is deliberate: ``start_pose`` is not an enumerable
parameter. It is RNG-sampled mid-traversal (the room representative or a seeded
free interior point) and the yaw is derived from it, off a single stateful RNG
advanced by every prior target. So the ``(scene, target) -> start_pose`` set is
reproducible only if the whole traversal is byte-identical between runs; a
decoupled "pre-render the poses to disk" pass cannot know which poses to render
without re-running the generator, and any drift turns a key miss into a silent
skip. Rendering in the same traversal that computes the pose removes the key
entirely, and a re-rolled seed (a same-room "no") renders natively when the loop
reaches the next seeded pose.

Two pieces:

- :func:`make_render_grounding_frame_provider` — the closure over a live Isaac
  Lab env. It teleports the robot to ``start_pose``, steps the env so the tiled
  camera holds a freshly rendered frame, reads the rgb, and returns it as a
  ``PIL.Image``. Authoring this needs no Kit; **running** it does (an operator
  step under ``AppLauncher`` — see ``scripts/render_grounded_mission_corpus.py``).
- :func:`coerce_frame_return` — the frame-contract sanitizer. The default
  grounding runner consumes the frame as ``frame if isinstance(frame,
  Image.Image) else Image.open(frame).convert("RGB")`` with no guard, so a
  provider that returns a path must validate it is a readable image first; a bad
  or missing render then surfaces as a clean ``None`` (a counted skip) instead
  of raising inside the runner. Pure Python; unit-tested without Kit.

Scene-source-agnostic: this reads only the ``(x, y, yaw)`` start pose and never
imports any scene-source package.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

# Robot root z (meters above the env origin) at which the teleported robot is
# placed for the render. 0.1 m assumes the floor surface sits at world z=0.
# Infinigen floors sit several cm above world origin, so pass ``floor_top_z +
# wheel clearance`` there — the same operator-tunable knob the floor-spawn reset
# event exposes. Kept a parameter rather than silently hard-coded.
DEFAULT_SPAWN_Z = 0.1

# Zero-action env steps after the teleport before the rgb is read. One step is
# the documented minimum for the tiled camera to hold a freshly rendered frame;
# settle quality cannot be verified headless, so the operator may need to raise
# this if the teleported robot has not settled before the frame is read.
DEFAULT_GROUNDING_WARMUP_STEPS = 2


def coerce_frame_return(frame: Any) -> Any:
    """Sanitize a provider's return for the un-guarded runner frame contract.

    The default grounding runner does ``frame if isinstance(frame, Image.Image)
    else Image.open(frame).convert("RGB")`` with no try/except, so:

    - a ``PIL.Image`` passes straight through;
    - a path (``str`` / ``os.PathLike``) is stat-checked and probed as an image
      here, so a missing or unreadable render returns ``None`` (a counted skip)
      rather than raising ``Image.open`` deep inside the runner. The validated
      *path* is returned, not the opened image, so the runner's own
      ``Image.open(...).convert("RGB")`` stays the single decode site;
    - ``None`` stays ``None``;
    - any other type (e.g. a raw numpy array, which the runner does not handle)
      returns ``None`` — surfaced as a skip, never a runner crash.
    """
    if frame is None:
        return None
    from PIL import Image  # noqa: PLC0415 — deferred so module import stays light

    if isinstance(frame, Image.Image):
        return frame
    if isinstance(frame, (str, os.PathLike)):
        path = Path(frame)
        if not path.is_file():
            return None
        try:
            with Image.open(path) as probe:
                probe.verify()
        except Exception:
            return None
        return str(path)
    return None


def _wp_to_torch(arr: Any) -> Any:
    """Coerce a ``wp.array`` to a torch tensor (live Isaac Sim data only)."""
    import warp as wp  # noqa: PLC0415 — only reached on live Isaac Sim data

    return wp.to_torch(arr)


def _to_uint8_hwc(tensor: Any, torch: Any) -> Any:
    """``(N, H, W, C)`` rgb(a) tensor -> contiguous ``(H, W, 3)`` uint8 ndarray.

    Mirrors the runtime adapter's ``IsaacLabEnvAdapter._to_uint8_hwc`` readback
    so the grounded frame matches the captured perception frame.
    """
    import numpy as np  # noqa: PLC0415 — keep top-of-file imports light

    del torch  # signature symmetry; the conversion only needs the tensor
    t = tensor if hasattr(tensor, "detach") else _wp_to_torch(tensor)
    arr = (t[0] if t.dim() == 4 else t).detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _teleport_robot(
    *,
    robot: Any,
    scene: Any,
    env_ids: Any,
    device: Any,
    x: float,
    y: float,
    yaw: float,
    spawn_z: float,
    torch: Any,
) -> None:
    """Teleport the env-0 robot to ``(x, y, spawn_z)`` facing ``yaw``.

    Mirrors the reset event's root write: a yaw-only XYZW quaternion
    (``qz = sin(yaw/2)``, ``qw = cos(yaw/2)``), positions offset by the env
    origin, and both root and joint velocities zeroed so the teleported robot
    does not lurch from stale wheel spin before the frame is read.
    """
    import warp as wp  # noqa: PLC0415 — only reached on live Isaac Sim data

    quat = torch.zeros(1, 4, device=device)
    quat[:, 2] = math.sin(yaw / 2.0)  # qz (XYZW)
    quat[:, 3] = math.cos(yaw / 2.0)  # qw
    root_pose = wp.to_torch(robot.data.default_root_pose)[env_ids].clone()
    env_origins = scene.env_origins[env_ids]
    root_pose[:, 0] = env_origins[:, 0] + x
    root_pose[:, 1] = env_origins[:, 1] + y
    root_pose[:, 2] = env_origins[:, 2] + spawn_z
    root_pose[:, 3:7] = quat
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(1, 6, device=device), env_ids=env_ids
    )
    joint_pos = wp.to_torch(robot.data.default_joint_pos)[env_ids].clone()
    joint_vel = torch.zeros_like(wp.to_torch(robot.data.default_joint_vel)[env_ids])
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


def make_render_grounding_frame_provider(
    *,
    env: Any,
    app_update: Callable[[], None] | None = None,
    robot_key: str = "robot",
    perception_camera_key: str = "d555_camera_perception",
    spawn_z: float = DEFAULT_SPAWN_Z,
    warmup_steps: int = DEFAULT_GROUNDING_WARMUP_STEPS,
    torch_module: Any = None,
) -> Callable[[dict[str, Any], tuple[float, float, float]], Any]:
    """Build a live ``grounding_frame_provider`` over a running Isaac Lab env.

    The returned callable matches the generator seam ``(obj, start_pose) ->
    PIL.Image | None``: it teleports the robot to ``start_pose`` (env 0), steps
    the env ``warmup_steps`` times so the tiled perception camera holds a fresh
    frame, reads the rgb, and returns it as a ``PIL.Image`` (in memory — no PNG
    written). ``obj`` is unused: the frame is the robot's egocentric view from
    the start pose, and the VL runner judges whether the target named in the
    mission text is visible in it.

    ``app_update`` is the host's Kit pump (``simulation_app.update``); it is
    invoked once per warm-up step. With ``unwrapped.render_enabled = False`` set
    by the launcher, that is the sole Kit pump per step (mirrors the bridge
    driver), so the editor viewport / OmniGraph advance in lock-step with the
    render. ``None`` skips it (a headless run whose ``env.step`` already pumps).

    Authoring this needs no Kit; running it does — it is invoked only from the
    Kit-launched corpus sibling (``scripts/render_grounded_mission_corpus.py``).
    """
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415 — deferred so import stays Kit-free
    torch = torch_module

    unwrapped = env.unwrapped
    device = unwrapped.device
    scene = unwrapped.scene
    robot = scene[robot_key]
    action_shape = tuple(int(d) for d in unwrapped.action_manager.action.shape)
    zero_action = torch.zeros(action_shape, device=device)
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    steps = max(1, int(warmup_steps))

    def _provider(obj: dict[str, Any], start_pose: tuple[float, float, float]) -> Any:
        del obj  # the frame is pose-egocentric; the runner judges the target
        x, y, yaw = float(start_pose[0]), float(start_pose[1]), float(start_pose[2])
        _teleport_robot(
            robot=robot,
            scene=scene,
            env_ids=env_ids,
            device=device,
            x=x,
            y=y,
            yaw=yaw,
            spawn_z=spawn_z,
            torch=torch,
        )
        for _ in range(steps):
            env.step(zero_action.clone())
            if app_update is not None:
                app_update()
        rgb = scene[perception_camera_key].data.output["rgb"]
        arr = _to_uint8_hwc(rgb, torch)
        from PIL import Image  # noqa: PLC0415 — deferred so import stays Kit-free

        return coerce_frame_return(Image.fromarray(arr))

    return _provider
