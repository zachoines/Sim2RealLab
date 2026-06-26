"""Live start-frame provider for the mission generator's geometric grounding pass.

``build_mission_queue`` takes an optional ``grounding_frame_provider`` — a
callable ``(obj, start_pose) -> struct`` invoked inside ``_ground_start_frame``.
This module builds that provider over a running Isaac Lab env: it renders the
start frame from the robot's ``d555_camera_perception`` camera at the computed
``start_pose`` and reads Replicator annotators into the target-visibility struct
the model-free :func:`build_mission_queue.geometric_visibility_verdict` judges.

The render happens in-process at generation time because ``start_pose`` is
RNG-sampled mid-traversal off a single stateful RNG shared across targets, so the
poses are not enumerable and cannot be pre-rendered to disk without re-running
the generator.

- :func:`make_render_grounding_frame_provider` — the closure over a live env:
  teleports the robot to ``start_pose`` (env 0), steps so the perception camera
  holds a fresh frame, reads the ``bounding_box_2d_tight`` +
  ``instance_id_segmentation`` annotators, and returns the struct (or ``None``
  on an unusable read — a counted skip). The target is pinned by its
  ``obj["prim_path"]`` through the instance map, NOT by label, so a same-label
  sibling cannot satisfy the gate. Authoring needs no Kit; running does (an
  ``AppLauncher`` step — see ``scripts/render_grounded_mission_corpus.py``).
- :func:`build_visibility_struct` — pure struct-shaping over parsed annotator
  data.

Scene-source-agnostic: reads only the ``(x, y, yaw)`` pose + the target's
``prim_path``, never importing a scene-source package.
"""

from __future__ import annotations

import math
from typing import Any, Callable

# Robot root z (m above env origin) when teleported for the render. 0.1 m
# assumes the floor sits at world z=0; Infinigen floors sit a few cm higher, so
# pass ``floor_top_z + wheel clearance`` (the floor-spawn reset event's knob).
DEFAULT_SPAWN_Z = 0.1

# Zero-action env steps after the teleport before the rgb is read. One step is
# the minimum for the tiled camera to hold a fresh frame; raise it if the
# teleported robot has not settled by the read.
DEFAULT_GROUNDING_WARMUP_STEPS = 2


def build_visibility_struct(
    bboxes: Any, seg: Any, prim_path: str | None, label: str | None = None
) -> dict[str, Any] | None:
    """Shape parsed annotator reads into the target-visibility struct, or ``None``.

    ``bboxes`` is the parsed bounding_box_2d_tight list; ``seg`` is the parsed
    ``InstanceSegmentation`` (or ``None`` if no frame); ``prim_path`` is the
    target's USD prim path; ``label`` is the target's class (used to prefer the
    target's own class row for occlusion over a foreground occluder's). Returns
    ``None`` (a counted skip) when the read is unusable (no seg frame, or no
    prim_path). A successful read where the target has no segment yields an
    ``in_frame=False`` struct (a "no" that re-rolls a same-room target) —
    distinct from a skip.
    """
    from strafer_lab.tools.bbox_extractor import (  # noqa: PLC0415 — keep import light
        bbox_row_for_segment,
        segment_ids_for_prim_path,
        segment_pixel_extent,
    )

    if seg is None:
        return None  # unusable instance-seg read -> skip (cannot locate target)
    frame_w, frame_h = int(seg.frame_w), int(seg.frame_h)
    if not prim_path:
        return None  # target carries no prim_path -> cannot key -> skip

    seg_ids = segment_ids_for_prim_path(seg.info, prim_path)
    extent = segment_pixel_extent(seg.mask, seg_ids) if seg_ids else None
    if not seg_ids or extent is None:
        # The read succeeded but the target was not rendered into any segment:
        # it is not visible -> a real "no" verdict (drives the same-room
        # re-roll), distinct from a skipped/unusable read.
        return {
            "in_frame": False,
            "bbox": None,
            "occlusion_ratio": None,
            "frame_w": frame_w,
            "frame_h": frame_h,
        }

    _count, mask_bbox = extent
    # Occlusion is per-class only (from the bbox row overlapping the segment); the
    # bbox is the per-instance mask box. Pass the target label so a foreground
    # occluder of a different class cannot supply the occlusion.
    row = bbox_row_for_segment(bboxes or [], mask_bbox, label=label)
    occlusion_ratio = row.occlusion_ratio if row is not None else None
    return {
        "in_frame": True,
        "bbox": tuple(int(v) for v in mask_bbox),
        "occlusion_ratio": occlusion_ratio,
        "frame_w": frame_w,
        "frame_h": frame_h,
    }


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
    import warp as wp  # noqa: PLC0415 — warp is Kit-only

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
    """Build a live ``grounding_frame_provider`` (the ``(obj, start_pose) -> struct | None`` seam).

    Attaches the two annotators once here (the only ``omni.replicator.core``
    touch-point) to the perception camera's render product via
    :func:`bbox_extractor.resolve_render_product_path`. ``app_update`` is the
    host's Kit pump, invoked once per warm-up step; with the launcher's
    ``render_enabled = False`` it is the sole pump per step, so viewport and
    OmniGraph advance in lock-step with the render. ``None`` skips it (a headless
    run whose ``env.step`` already pumps). Running this needs Kit; it is invoked
    only from ``scripts/render_grounded_mission_corpus.py``.
    """
    if torch_module is None:
        import torch as torch_module  # noqa: PLC0415 — deferred so import stays Kit-free
    torch = torch_module

    from strafer_lab.tools.bbox_extractor import (  # noqa: PLC0415 — Kit-only attach path
        ReplicatorBboxExtractor,
        ReplicatorInstanceSegExtractor,
        resolve_render_product_path,
    )

    unwrapped = env.unwrapped
    device = unwrapped.device
    scene = unwrapped.scene
    robot = scene[robot_key]
    action_shape = tuple(int(d) for d in unwrapped.action_manager.action.shape)
    zero_action = torch.zeros(action_shape, device=device)
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    steps = max(1, int(warmup_steps))

    # Attach both annotators once to the perception camera's render product.
    # ``semantic_types=("class",)`` matches how the scene authors UsdSemantics
    # labels; the instance-seg annotator pins the specific target instance.
    render_product_path = resolve_render_product_path(scene[perception_camera_key])
    bbox_extractor = ReplicatorBboxExtractor(
        render_product_path, semantic_types=("class",)
    )
    inst_seg_extractor = ReplicatorInstanceSegExtractor(render_product_path)

    def _provider(obj: dict[str, Any], start_pose: tuple[float, float, float]) -> Any:
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
        bboxes = bbox_extractor.extract()
        seg = inst_seg_extractor.extract()
        prim_path = obj.get("prim_path") if isinstance(obj, dict) else None
        label = obj.get("label") if isinstance(obj, dict) else None
        return build_visibility_struct(bboxes, seg, prim_path, label=label)

    return _provider
