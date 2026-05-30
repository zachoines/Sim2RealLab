"""In-process Isaac Lab teleop driver for the harness data-capture system.

Boots Isaac Sim headed, loads the InfinigenPerception env, reads a
gamepad, drives ``env.step()``, captures frames via
:class:`strafer_lab.tools.lerobot_writer.StraferLeRobotWriter`, and
writes a LeRobot v3 dataset under the supplied output root.

Operator UX surfaces:

- Live third-person view via the Isaac Sim editor viewport.
- First-person PIP via an OpenCV window showing the
  ``d555_camera_perception`` stream. The PIP renders to a separate
  top-level Qt/X surface; the camera's render product writes to disk
  before any viewport-side overlay, so the PIP never leaks into
  captured frames.
- Console HUD echoing the active ``mission_text`` and a
  ``[REC]`` / ``[PAUSED]`` banner each tick. Hold ``A`` to pause the
  writer (env keeps stepping; ``add_frame`` is skipped).

Invocation (preferred — via the unified entry point)::

    isaaclab -p Scripts/capture.py \\
        --driver teleop --mission-source scene-metadata \\
        --scene <scene_id> --output <dataset_root>

Direct invocation (rarely needed)::

    isaaclab -p source/strafer_lab/scripts/teleop_capture.py \\
        --scene <scene_id> --output <dataset_root>
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI — parsed BEFORE AppLauncher boot so Kit sees the headless / cameras
# flags before it brings up the simulation app.
# ---------------------------------------------------------------------------

_DEFAULT_TASK = "Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", type=str, default=_DEFAULT_TASK,
        help="Registered Isaac Lab task name. Default is the perception "
        "env that carries both the policy + perception cameras.",
    )
    parser.add_argument(
        "--scene", type=str, required=True,
        help="Scene name (used as scene_id in per-episode metadata + as "
        "the LeRobot dataset's repo_id suffix). Also used to resolve "
        "--scene-metadata when that flag is not supplied.",
    )
    parser.add_argument(
        "--scene-metadata", type=str, default=None,
        help="Path to the scene's scene_metadata.json. Defaults to "
        "Assets/generated/scenes/<scene>/scene_metadata.json.",
    )
    parser.add_argument(
        "--scene-usd", type=str, default=None,
        help="Override the env's default scene USD path. Optional; when "
        "omitted, the env keeps whatever USD the scene cfg picks.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="LeRobot dataset root. Must not exist (LeRobotDataset.create "
        "refuses to overwrite).",
    )
    parser.add_argument(
        "--repo-id", type=str, default=None,
        help="LeRobot repo_id. Defaults to 'strafer/<scene>'.",
    )
    parser.add_argument(
        "--fps", type=int, default=8,
        help="Capture rate in Hz.",
    )
    parser.add_argument(
        "--vcodec", type=str, default="h264",
        help="LeRobot v3 video codec. h264 is the broadest-availability "
        "choice on ARM64.",
    )
    parser.add_argument(
        "--capture-policy-cam",
        action=argparse.BooleanOptionalAction, default=True,
        help="Capture the 80×60 policy camera alongside the 640×360 "
        "perception camera.",
    )
    parser.add_argument(
        "--operator-handle", type=str, default=None,
        help="Operator identifier saved on every episode (per-session).",
    )
    parser.add_argument(
        "--session-id", type=str, default=None,
        help="Session identifier; defaults to a wall-clock timestamp.",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=200,
        help="Stop after this many KEPT episodes (discards do not count).",
    )
    parser.add_argument(
        "--max-steps-per-episode", type=int, default=1500,
        help="Hard cap on env steps within one episode (operator can end "
        "earlier via Y/B/X/SELECT).",
    )
    parser.add_argument(
        "--deadzone", type=float, default=0.15,
        help="Gamepad stick deadzone.",
    )
    parser.add_argument(
        "--no-pip-window", action="store_true",
        help="Suppress the first-person PIP cv2 window (still captures "
        "frames normally). Useful on headless runs that nonetheless need "
        "a viewport for Isaac Sim's editor.",
    )
    parser.add_argument(
        "--target-label-filter", nargs="*", default=None,
        help="If set, only objects whose label is in this list appear in "
        "the mission-picker.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("world_arcade", "egocentric"),
        default="world_arcade",
        help="world_arcade (default): overhead viewport, stick = world-frame "
        "velocity (today's behavior). egocentric: viewport follows the robot, "
        "stick = body-frame velocity. Useful for CLIP-style coverage where "
        "the operator needs to see what the robot sees.",
    )
    parser.add_argument(
        "--hide-overhead",
        action="store_true",
        help="Set overhead structure prims (ceilings / roof / exterior hull) "
        "to invisible at startup so the top-down view is unobstructed. "
        "Matches the Infinigen *_ceiling_*, *_roof_*, *_attic_*, "
        "*_exterior_* naming conventions. Does NOT modify the scene USDC "
        "on disk; visibility is reset on next launch.",
    )
    parser.add_argument(
        "--overhead-regex",
        type=str,
        default=None,
        help="Custom regex (re.search semantics, case-insensitive) used by "
        "--hide-overhead. Override when the default pattern misses your "
        "scene's specific structure prim naming.",
    )
    parser.add_argument(
        "--no-target-marker",
        action="store_true",
        help="Suppress the debug-draw target marker (operator-only sphere "
        "at the active target's position). Marker never enters captured "
        "frames either way; this flag is for operators who find the marker "
        "visually noisy.",
    )
    parser.add_argument(
        "--capture-rate-hz",
        type=float,
        default=None,
        help="Writer sample rate in Hz, decoupled from the env step rate. "
        "Defaults to --fps. Env still steps at full sim tick rate; "
        "writer.add_frame is called every "
        "round(env_step_hz / capture_rate_hz) ticks.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


_PARSER = _build_parser()
args_cli, hydra_args = _PARSER.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Force headed + cameras — non-negotiable for the teleop UX. Pass
# --visualizer kit so the Kit editor viewport opens.
args_cli.enable_cameras = True
if not getattr(args_cli, "visualizer", None):
    args_cli.visualizer = "kit"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Post-launch imports — these pull in omni / isaaclab modules that need
# the Kit runtime to be active.
# ---------------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401 — registers Isaac Lab task metadata
import strafer_lab.tasks  # noqa: F401 — registers Strafer envs

from isaaclab.envs.common import ViewerCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from strafer_lab.tools.gamepad_reader import GamepadReader
from strafer_lab.tools.lerobot_writer import (
    StraferLeRobotWriter,
    hash_scene_metadata,
)
from strafer_lab.tools.scene_paths import resolve_scene_metadata_path
from strafer_lab.tools.teleop_buttons import (
    button_state_to_episode_outcome,
    describe_button_layout,
)
from strafer_lab.tools.teleop_mission_picker import (
    MissionCandidate,
    load_candidates,
    prompt_for_target,
)


# Optional cv2 — required for the PIP window but the script degrades to
# console-only if cv2 isn't installed.
_CV2_AVAILABLE = False
try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_rev_parse_head(repo_root: Path) -> str:
    """Return ``git rev-parse HEAD`` for the repo, or an empty string."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _resolve_active_spawn_points(scene: str) -> list[list[float]]:
    """Return spawn_points_xy ONLY for the active scene, not pooled across all scenes."""
    combined = Path("Assets/generated/scenes/scenes_metadata.json")
    if not combined.is_file():
        return []
    try:
        import json as _json
        data = _json.loads(combined.read_text(encoding="utf-8"))
    except Exception:
        return []
    scene_block = data.get("scenes", {}).get(scene, {})
    pool = scene_block.get("spawn_points_xy", [])
    return [list(map(float, pt)) for pt in pool if len(pt) >= 2]


# Hide ANY of these overhead structure tokens. Infinigen's room-structure
# regex (generate_scenes_metadata.py:62) uses ceiling | exterior | floor |
# wall | staircase as the trailing label. For top-down operator UX, the
# operator needs ceiling + exterior (roof hull) hidden, NOT walls/floors.
# Some scenes also expose attic / roof labels.
_OVERHEAD_PRIM_RE = re.compile(
    r"(?:^|_)(ceiling|roof|attic|exterior)(?:_\d+)?$",
    re.IGNORECASE,
)


def _hide_overhead_prims(unwrapped, custom_regex: str | None = None) -> tuple[int, list[str]]:
    """Set every overhead structure prim (ceiling / roof / exterior) to invisible.

    Returns ``(count_hidden, sample_paths)`` where ``sample_paths`` is up
    to 20 paths matched — used for diagnostics so the operator can see
    exactly what got hidden.

    Operator-only effect: invisible prims still collide (UsdGeom
    visibility is a render attribute, not a physics one), so the robot
    still respects the ceiling for collision. Perception camera output
    similarly skips invisible prims — which is fine since the
    horizontally-looking camera rarely sees them anyway.
    """
    from pxr import UsdGeom  # type: ignore

    pattern = re.compile(custom_regex, re.IGNORECASE) if custom_regex else _OVERHEAD_PRIM_RE

    stage = unwrapped.scene.stage
    hidden = 0
    sample_paths: list[str] = []
    for prim in stage.Traverse():
        name = prim.GetName()
        if pattern.search(name):
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
                hidden += 1
                if len(sample_paths) < 20:
                    sample_paths.append(str(prim.GetPath()))
    return hidden, sample_paths


def _possess_d555_viewport(prim_path: str) -> bool:
    """Point Kit's active viewport at the d555 camera prim.

    Egocentric control mode does not need a separate ViewerCfg
    follow-cam (a second render pass). Reusing the perception camera's
    render product keeps the viewport in lock-step with what the
    writer sees and avoids the duplicate render. Returns True if the
    repoint succeeded.
    """
    try:
        import omni.kit.viewport.utility as vp_util  # type: ignore
    except ImportError:
        return False
    try:
        viewport = vp_util.get_active_viewport()
        if viewport is None:
            return False
        viewport.camera_path = prim_path
        return True
    except Exception as exc:
        print(
            f"[teleop_capture] viewport possession failed "
            f"({exc.__class__.__name__}: {exc}); using default viewer cam.",
            file=sys.stderr, flush=True,
        )
        return False


_KIT_PERSP_CAM_PRIM = "/OmniverseKit_Persp"

# Only re-pose the follow-cam after the robot has translated this far.
# Calling set_camera_view every env step clobbers operator scroll-wheel
# input before Kit can flush its scroll-handler's pending writes;
# the gate gives Kit quiet intervals to apply zoom.
_ARCADE_FOLLOW_XY_DELTA_M = 0.30


def _compute_arcade_eye(
    robot_xy: tuple[float, float], altitude_z: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return ``(eye, target)`` for the top-down follow-cam.

    Eye and target intentionally share XY so set_camera_view hits its
    explicit looking-down quat branch; any XY offset routes to a
    lookat resolver whose screen-up depends on Kit's world-up
    reference and rotates 90° away from world +Y.
    """
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    z = float(altitude_z)
    return ((rx, ry, z), (rx, ry, 0.0))


def _arcade_camera_init(
    scene_centroid_xy: tuple[float, float], altitude_z: float = 12.0,
) -> bool:
    """Pin the Kit viewport to /OmniverseKit_Persp at a top-down arcade pose.

    Resets ``camera_path`` first to defend against a prior egocentric
    session that left the viewport possessed by the d555 prim, then
    drives the persp camera through ``isaacsim.core.utils.viewports.
    set_camera_view`` (Kit's FSD-safe ``TransformPrimCommand`` path) —
    necessary because :class:`ViewportCameraController` doesn't
    reliably apply :class:`ViewerCfg` to the active viewport after
    env.reset.
    """
    try:
        import omni.kit.viewport.utility as vp_util  # type: ignore
        from isaacsim.core.utils.viewports import set_camera_view  # type: ignore
    except ImportError:
        return False
    try:
        viewport = vp_util.get_active_viewport()
        if viewport is not None:
            viewport.camera_path = _KIT_PERSP_CAM_PRIM
        eye, target = _compute_arcade_eye(scene_centroid_xy, altitude_z)
        set_camera_view(
            eye=list(eye), target=list(target),
            camera_prim_path=_KIT_PERSP_CAM_PRIM,
        )
        return True
    except Exception as exc:
        print(
            f"[teleop_capture] arcade camera init failed "
            f"({exc.__class__.__name__}: {exc}); using default viewer cam.",
            file=sys.stderr, flush=True,
        )
        return False


def _arcade_follow_tick(unwrapped, robot_xy: tuple[float, float]) -> None:
    """Per-tick follow: re-pose /OmniverseKit_Persp above ``robot_xy``.

    Reads the viewport camera's current world-Z before re-setting eye
    so the operator's scroll-wheel zoom is preserved between ticks (we
    only overwrite XY). Silent on failure so a one-shot Kit hiccup
    doesn't break the loop.
    """
    try:
        import omni.kit.viewport.utility as vp_util  # type: ignore
        from isaacsim.core.utils.viewports import set_camera_view  # type: ignore
        from pxr import Usd, UsdGeom  # type: ignore
    except ImportError:
        return
    try:
        viewport = vp_util.get_active_viewport()
        if viewport is None or viewport.camera_path != _KIT_PERSP_CAM_PRIM:
            return
        # Read current viewport altitude so operator scroll-zoom persists.
        stage = unwrapped.scene.stage
        cam_prim = stage.GetPrimAtPath(viewport.camera_path)
        cur_z = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default(),
        ).ExtractTranslation()[2]
        altitude = max(1.5, float(cur_z))  # clamp floor so we don't sink into geometry
        eye, target = _compute_arcade_eye(robot_xy, altitude)
        set_camera_view(
            eye=list(eye), target=list(target),
            camera_prim_path=_KIT_PERSP_CAM_PRIM,
        )
    except Exception:
        # Per-tick path — swallow to avoid breaking the env step loop on
        # transient Kit state. A persistent issue surfaces on init.
        pass


def _as_torch(arr):
    """Coerce a ``wp.array`` or ``torch.Tensor`` to a ``torch.Tensor``.

    Isaac Lab sometimes returns warp arrays (sensor outputs, articulation
    data) and sometimes torch tensors depending on the field. Warp's
    recent API drops ``__getitem__`` on wp.array, so indexing without
    a conversion raises ``RuntimeError: Item indexing is not supported
    on wp.array objects``. This helper centralizes the conversion so the
    caller can always do ``_as_torch(x)[0].cpu().numpy()``.
    """
    if isinstance(arr, torch.Tensor):
        return arr
    return wp.to_torch(arr)


def _robot_pose(unwrapped) -> tuple[tuple[float, float, float, float, float, float, float], float]:
    """Return ``(pose_xyzqxqyqzqw, yaw_rad)`` for env 0.

    ``Articulation.data.root_quat_w`` returns the quaternion in XYZW
    order — a yaw formula that assumes WXYZ happens to coincide at the
    spawn yaw=π pose but collapses to a step function elsewhere.
    """
    scene = unwrapped.scene
    pos = _as_torch(scene["robot"].data.root_pos_w)[0].cpu().numpy()
    quat_xyzw = _as_torch(scene["robot"].data.root_quat_w)[0].cpu().numpy()
    x, y, z, w = (float(quat_xyzw[i]) for i in range(4))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pose = (
        float(pos[0]), float(pos[1]), float(pos[2]),
        x, y, z, w,
    )
    return pose, yaw


def _achieved_vel(unwrapped) -> tuple[float, float, float]:
    """Return ``(vx, vy, omega_z)`` from the robot's body-frame velocities."""
    robot = unwrapped.scene["robot"]
    lin_b = _as_torch(robot.data.root_lin_vel_b)[0].cpu().numpy()
    ang_w = _as_torch(robot.data.root_ang_vel_w)[0].cpu().numpy()
    return float(lin_b[0]), float(lin_b[1]), float(ang_w[2])


def _stick_to_body_action(
    lx: float, ly: float, rx: float, heading: float,
    *,
    control_mode: str = "world_arcade",
) -> tuple[float, float, float]:
    """Stick → body-frame ``(vx, vy, omega_z)``.

    ``control_mode``:

    - ``world_arcade``: today's behavior. Stick is interpreted in the
      world frame (overhead-viewport convention); we rotate into the
      robot's body frame using its current heading. Right-stick X =
      yaw rate, negated so stick-left = CCW positive omega.
    - ``egocentric``: stick is interpreted directly in the body frame
      (no heading rotation). Push the left stick forward and the robot
      moves forward regardless of which way it's facing — classic
      first-person controls. Right-stick X still maps to yaw rate.

    The 0.01 noise floor below is the post-normalization
    "anything actually pressed" threshold (well below the operator-
    tunable ``--deadzone`` applied at the gamepad reader); it exists
    to keep floating-point jitter from emitting non-zero commands when
    the stick is at rest.
    """
    if control_mode == "egocentric":
        body_vx = -ly  # stick-up = forward
        body_vy = -lx  # stick-left = strafe left
        stick_mag = min(1.0, math.sqrt(body_vx ** 2 + body_vy ** 2))
        norm_mag = math.sqrt(body_vx ** 2 + body_vy ** 2)
        if norm_mag > 0.0 and stick_mag > 0.01:
            body_vx *= stick_mag / norm_mag
            body_vy *= stick_mag / norm_mag
        omega = -rx
        if stick_mag < 0.01 and abs(omega) < 0.01:
            return 0.0, 0.0, 0.0
        return body_vx, body_vy, omega

    # world_arcade (default)
    world_vx = lx
    world_vy = -ly
    stick_mag = min(1.0, math.sqrt(world_vx ** 2 + world_vy ** 2))

    norm_mag = math.sqrt(world_vx ** 2 + world_vy ** 2)
    if norm_mag > 0.0 and stick_mag > 0.01:
        world_vx *= stick_mag / norm_mag
        world_vy *= stick_mag / norm_mag

    cos_h, sin_h = math.cos(heading), math.sin(heading)
    body_vx = cos_h * world_vx + sin_h * world_vy
    body_vy = -sin_h * world_vx + cos_h * world_vy
    omega = -rx

    if stick_mag < 0.01 and abs(omega) < 0.01:
        return 0.0, 0.0, 0.0
    return body_vx, body_vy, omega


class _TargetMarker:
    """Operator-only debug-draw marker at the active target position.

    Uses Isaac Sim's ``isaacsim.util.debug_draw`` interface which draws
    to the editor viewport but NOT into Replicator render products — so
    the marker never enters captured RGB frames. Lifetime is one
    episode: ``set_target`` on begin_episode, ``clear`` on end_episode.
    """

    def __init__(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self._iface = None
        if not self._enabled:
            return
        try:
            from isaacsim.util.debug_draw import _debug_draw  # type: ignore
            self._iface = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:
            print(
                f"[teleop_capture] target marker disabled (debug-draw "
                f"unavailable: {exc.__class__.__name__}).",
                flush=True,
            )
            self._enabled = False

    def set_target(self, xyz: tuple[float, float, float]) -> None:
        if not self._enabled or self._iface is None:
            return
        # Clear any previous marker before drawing the next one.
        self._iface.clear_points()
        self._iface.draw_points(
            [(float(xyz[0]), float(xyz[1]), float(xyz[2]) + 0.30)],
            [(0.1, 1.0, 0.1, 1.0)],  # bright green RGBA
            [40.0],  # point size
        )

    def clear(self) -> None:
        if self._enabled and self._iface is not None:
            try:
                self._iface.clear_points()
            except Exception:
                pass


def _rgb_to_uint8_hwc(tensor) -> np.ndarray:
    """Isaac Sim ``(N, H, W, C)`` RGB → ``(H, W, 3)`` uint8.

    Drops the alpha channel if present. Replicator camera outputs are
    typically uint8 already; clamp + cast just in case. Accepts
    ``wp.array`` or ``torch.Tensor``.
    """
    t = _as_torch(tensor)
    if t.dim() == 4:
        arr = t[0].cpu().numpy()
    else:
        arr = t.cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _depth_to_float32_hw(tensor) -> np.ndarray:
    """Isaac Sim ``(N, H, W, 1)`` distance → ``(H, W)`` float32 meters.

    Accepts ``wp.array`` or ``torch.Tensor``.
    """
    t = _as_torch(tensor)
    if t.dim() == 4:
        arr = t[0].cpu().numpy()
    else:
        arr = t.cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Driver — pulled into its own function so main() stays short.
# ---------------------------------------------------------------------------


def _build_writer(
    *,
    output_root: Path,
    repo_id: str,
    fps: int,
    vcodec: str,
    capture_policy_cam: bool,
    capture_git_sha: str,
    scene_metadata_hash: str,
    operator_handle: str | None,
    session_id: str | None,
) -> StraferLeRobotWriter:
    """Construct the writer with the supplied invariants."""
    return StraferLeRobotWriter(
        root=output_root,
        repo_id=repo_id,
        fps=fps,
        capture_git_sha=capture_git_sha,
        scene_metadata_hash=scene_metadata_hash,
        capture_policy_cam=capture_policy_cam,
        vcodec=vcodec,
        operator_handle=operator_handle,
        session_id=session_id,
    )


class _PipWindow:
    """Wraps the cv2 PIP window with no-op fallbacks.

    Three states:
    - disabled by flag (``enabled=False``) → never opens, no-op everywhere.
    - cv2 not importable → no-op + one-shot warning.
    - cv2 importable but the build has no GUI backend
      (``opencv-python-headless`` ships without GTK / Qt) → no-op + one-shot
      warning so the operator knows the editor viewport is their only
      live view.
    """

    def __init__(self, enabled: bool) -> None:
        self._enabled = False
        if not enabled:
            return
        if not _CV2_AVAILABLE:
            print(
                "[teleop_capture] cv2 not installed; PIP disabled. "
                "Use the Isaac Sim editor viewport for the live view.",
            )
            return
        try:
            cv2.namedWindow("perception-PIP", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("perception-PIP", 640, 360)
            self._enabled = True
        except cv2.error as exc:
            # The 'not implemented' / 'Rebuild the library with Windows,
            # GTK+ 2.x or Cocoa support' family — opencv-python-headless
            # has no GUI bindings by design.
            print(
                f"[teleop_capture] cv2.namedWindow failed ({exc.__class__.__name__}): "
                "the installed opencv has no GUI backend (likely "
                "opencv-python-headless). PIP disabled — the Isaac Sim "
                "editor viewport is your only live view this session. "
                "Pass --no-pip-window to silence this warning.",
            )

    def show(self, rgb_hwc_uint8: np.ndarray, status_text: str) -> None:
        if not self._enabled:
            return
        # cv2 wants BGR; perception camera comes in RGB.
        bgr = cv2.cvtColor(rgb_hwc_uint8, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr, status_text, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )
        cv2.imshow("perception-PIP", bgr)
        cv2.waitKey(1)

    def close(self) -> None:
        if self._enabled:
            try:
                cv2.destroyWindow("perception-PIP")
            except Exception:
                pass


def main() -> int:
    args = args_cli
    repo_root = Path(__file__).resolve().parents[3]
    capture_git_sha = _git_rev_parse_head(repo_root)

    scene_metadata_path = resolve_scene_metadata_path(
        scene=args.scene,
        metadata_override=args.scene_metadata,
        usd_override=args.scene_usd,
    )
    scene_metadata_sha = hash_scene_metadata(scene_metadata_path)
    print(f"[teleop_capture] scene_metadata: {scene_metadata_path}")
    print(f"[teleop_capture] scene_metadata sha256: {scene_metadata_sha[:16]}...")
    print(f"[teleop_capture] capture_git_sha: {capture_git_sha[:12] or '(none)'}")

    candidates = load_candidates(
        scene_metadata_path,
        allowed_labels=args.target_label_filter,
    )
    if not candidates:
        print(
            f"[teleop_capture] ERROR: scene_metadata at {scene_metadata_path} "
            "has no targets after filtering. Nothing to do.",
            file=sys.stderr,
        )
        simulation_app.close()
        return 2
    print(f"[teleop_capture] loaded {len(candidates)} pickable targets")

    # Resolve env cfg + scene-USD override
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    if args.scene_usd:
        env_cfg.scene.scene_geometry.spawn.usd_path = str(Path(args.scene_usd).resolve())
        print(f"[teleop_capture] scene USD override → {env_cfg.scene.scene_geometry.spawn.usd_path}")

    # Override the spawn-point pool to use only the ACTIVE scene's
    # points. The env_cfg default pools across every scene listed in
    # scenes_metadata.json, which can place the robot in a non-active
    # scene's coordinates when more than one scene is registered.
    active_spawn_points = _resolve_active_spawn_points(args.scene)
    scene_centroid_xy: tuple[float, float] = (0.0, 0.0)
    spawn_bbox: tuple[float, float, float, float] | None = None  # (xmin, ymin, xmax, ymax)
    if active_spawn_points:
        env_cfg.events.reset_robot.params["spawn_points_xy"] = active_spawn_points
        if hasattr(env_cfg.commands, "goal_command"):
            env_cfg.commands.goal_command.spawn_points_xy = active_spawn_points
        xs = [p[0] for p in active_spawn_points]
        ys = [p[1] for p in active_spawn_points]
        scene_centroid_xy = (sum(xs) / len(xs), sum(ys) / len(ys))
        spawn_bbox = (min(xs), min(ys), max(xs), max(ys))
        print(
            f"[teleop_capture] using {len(active_spawn_points)} spawn points "
            f"for active scene {args.scene!r} (overriding pooled default)",
            flush=True,
        )
        print(
            f"[teleop_capture]   spawn-pool centroid=({scene_centroid_xy[0]:+.2f}, "
            f"{scene_centroid_xy[1]:+.2f})  bbox=x[{spawn_bbox[0]:+.2f}, "
            f"{spawn_bbox[2]:+.2f}] y[{spawn_bbox[1]:+.2f}, {spawn_bbox[3]:+.2f}]",
            flush=True,
        )
    else:
        print(
            f"[teleop_capture] WARNING: no active-scene spawn_points_xy found "
            f"for {args.scene!r} in Assets/generated/scenes/scenes_metadata.json. "
            "Falling back to the env_cfg default (pooled across all scenes), "
            "which may spawn the robot outside this scene's room.",
            file=sys.stderr, flush=True,
        )

    # Suppress the env's RL goal marker (sphere + cone). Teleop isn't
    # using the RL goal signal; the operator decides episode end via
    # buttons. The marker is operator-confusing residue from the
    # underlying training env. Also lock goal resampling so the env
    # doesn't keep teleporting a goal we're not tracking (avoids any
    # cost from goal_command's per-tick work).
    if hasattr(env_cfg.commands, "goal_command"):
        env_cfg.commands.goal_command.debug_vis = False
        env_cfg.commands.goal_command.resampling_time_range = (1.0e6, 1.0e6)
        print(
            "[teleop_capture] suppressed env goal_command.debug_vis + locked "
            "resampling — the sphere/cone marker is a training-side residue.",
            flush=True,
        )

    if args.control_mode == "egocentric":
        # Don't author a follow-cam ViewerCfg here — a separate
        # ViewerCfg costs an extra render pass. After env.reset()
        # we instead re-point Kit's active viewport at the
        # d555_camera_perception prim, sharing the Replicator camera
        # render that the writer already consumes.
        pass
    else:  # world_arcade
        # ViewerCfg here is only the initial seed — after env.reset()
        # we drive /OmniverseKit_Persp directly via
        # ``isaacsim.core.utils.viewports.set_camera_view`` because
        # Isaac Lab's ViewportCameraController doesn't reliably apply
        # ViewerCfg to the active viewport in this launch mode (see
        # _arcade_camera_init + collect_demos.py:282-311). The per-tick
        # follower then re-poses the camera over the robot's XY each
        # step so the operator never has to chase the robot or the goal.
        cx, cy = scene_centroid_xy
        env_cfg.viewer = ViewerCfg(
            eye=(cx, cy, 12.0),
            lookat=(cx, cy, 0.0),  # eye and lookat share XY — see _compute_arcade_eye
            origin_type="world",
            env_index=0,
            resolution=(1280, 720),
        )

    # Match camera update_period to the writer cadence. Gates Isaac
    # Lab's lazy buffer-render trigger; does NOT gate Kit's viewport
    # pump (which fires every env.step regardless). Small but real
    # savings on the warp kernel + CUDA sync at lazy buffer access.
    _render_rate_hz = float(args.capture_rate_hz or args.fps)
    _render_period_s = 1.0 / max(_render_rate_hz, 1e-3)
    for _cam_attr in ("d555_camera_perception", "d555_camera"):
        _cam_cfg = getattr(env_cfg.scene, _cam_attr, None)
        if _cam_cfg is not None and hasattr(_cam_cfg, "update_period"):
            _cam_cfg.update_period = _render_period_s
    print(
        f"[teleop_capture] cameras lowered to update_period={_render_period_s:.4f}s "
        f"(={_render_rate_hz:.1f} Hz, matches writer cadence) — saves buffer "
        f"trigger only; Kit viewport pump still fires every env.step.",
        flush=True,
    )

    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device
    scene = unwrapped.scene

    if "d555_camera_perception" not in scene.sensors:
        print(
            "[teleop_capture] ERROR: d555_camera_perception not in scene. "
            "This driver requires the Infinigen perception env.",
            file=sys.stderr,
        )
        env.close()
        simulation_app.close()
        return 2

    perception_camera = scene.sensors["d555_camera_perception"]
    policy_camera = scene.sensors.get("d555_camera") if args.capture_policy_cam else None

    if args.hide_overhead:
        try:
            n_hidden, sample_paths = _hide_overhead_prims(
                unwrapped, custom_regex=args.overhead_regex,
            )
            print(
                f"[teleop_capture] hid {n_hidden} overhead prim(s) "
                f"(--hide-overhead, regex={args.overhead_regex or '(default)'}).",
                flush=True,
            )
            if sample_paths:
                print(
                    f"[teleop_capture]   first up to 20 paths:",
                    flush=True,
                )
                for path in sample_paths:
                    print(f"[teleop_capture]     {path}", flush=True)
            if n_hidden == 0:
                print(
                    "[teleop_capture]   WARNING: nothing matched the regex. "
                    "Roof / exterior prims may use a different naming "
                    "convention in this scene. Pass --overhead-regex to "
                    "customize.",
                    file=sys.stderr, flush=True,
                )
        except Exception as exc:
            print(f"[teleop_capture] --hide-overhead failed: "
                  f"{exc.__class__.__name__}: {exc}", file=sys.stderr, flush=True)

    target_marker = _TargetMarker(enabled=not args.no_target_marker)

    output_root = Path(args.output).resolve()
    if output_root.exists():
        # LeRobotDataset.create refuses to overwrite. Auto-suffix with a
        # session timestamp so a half-baked previous run doesn't block
        # the next attempt — operator pattern would otherwise be to
        # always re-export $RUN_ID, which they don't always remember.
        suffix = time.strftime("_%Y%m%dT%H%M%S")
        output_root = output_root.with_name(output_root.name + suffix)
        print(
            f"[teleop_capture] requested --output already exists; using "
            f"auto-suffixed path → {output_root}",
            flush=True,
        )

    session_id = args.session_id or time.strftime("%Y%m%dT%H%M%S")
    repo_id = args.repo_id or f"strafer/{args.scene}"

    # Resolve writer fps + capture cadence. --capture-rate-hz, when set,
    # decouples the writer's add_frame cadence from the env step rate:
    # env still steps every tick (smoothness preserved); writer samples
    # every `round(env_step_hz / capture_rate_hz)` ticks. When unset,
    # we fall back to --fps and add_frame on every tick (today's
    # behavior).
    env_step_dt = float(getattr(env_cfg.sim, "dt", 1.0 / 60.0)) * int(
        getattr(env_cfg, "decimation", 1),
    )
    env_step_hz = 1.0 / max(env_step_dt, 1e-6)
    capture_rate_hz = float(args.capture_rate_hz or args.fps)
    ticks_per_capture = max(1, round(env_step_hz / max(capture_rate_hz, 1e-6)))
    writer_fps = int(round(capture_rate_hz))
    print(
        f"[teleop_capture] env_step_hz={env_step_hz:.2f}  "
        f"capture_rate_hz={capture_rate_hz:.2f}  "
        f"ticks_per_capture={ticks_per_capture}",
        flush=True,
    )

    writer = _build_writer(
        output_root=output_root,
        repo_id=repo_id,
        fps=writer_fps,
        vcodec=args.vcodec,
        capture_policy_cam=bool(args.capture_policy_cam),
        capture_git_sha=capture_git_sha,
        scene_metadata_hash=scene_metadata_sha,
        operator_handle=args.operator_handle,
        session_id=session_id,
    )
    print(f"[teleop_capture] writer ready → {output_root}")
    print(f"[teleop_capture] repo_id={repo_id} session_id={session_id}")
    print(f"[teleop_capture] capture_policy_cam={bool(args.capture_policy_cam)}")

    gamepad = GamepadReader(deadzone=args.deadzone)
    pip = _PipWindow(enabled=not args.no_pip_window)

    print("\n" + "=" * 64)
    print("HARNESS TELEOP — scene-metadata mission source")
    print(f"  scene_id          : {args.scene}")
    print(f"  output            : {output_root}")
    print(f"  fps (writer)      : {writer_fps}  "
          f"(capture cadence; env_step_hz={env_step_hz:.1f})")
    print(f"  control_mode      : {args.control_mode}")
    print(f"  hide_overhead     : {bool(args.hide_overhead)}")
    print(f"  target_marker     : {not bool(args.no_target_marker)}")
    print(f"  max episodes      : {args.max_episodes}")
    print(f"  max steps/episode : {args.max_steps_per_episode}")
    print(f"  visualizer        : {getattr(args_cli, 'visualizer', None)!r}  "
          f"(kit = editor viewport; required for teleop)")
    print(f"  enable_cameras    : {bool(args_cli.enable_cameras)}")
    print(f"  simulation_app.is_running(): {simulation_app.is_running()}")
    print("-" * 64)
    print(describe_button_layout())
    print("=" * 64 + "\n", flush=True)

    obs, info = env.reset()

    if args.control_mode == "egocentric":
        # Possess the perception-camera prim now that env.reset has
        # instantiated it. Saves a render pass vs a follow-cam
        # ViewerCfg by sharing the Replicator camera the writer
        # already consumes.
        d555_prim = "/World/envs/env_0/Robot/strafer/body_link/d555_camera_perception"
        if _possess_d555_viewport(d555_prim):
            print(
                f"[teleop_capture] egocentric viewport now possessing "
                f"{d555_prim} (no follow-cam render pass).",
                flush=True,
            )
    else:  # world_arcade
        # Drive the Kit persp camera through set_camera_view (FSD-safe)
        # because ViewerCfg may not have stuck post-env.reset. The
        # per-tick _arcade_follow_tick below re-poses it over the robot
        # each step while preserving the operator's scroll-zoom altitude.
        if _arcade_camera_init(scene_centroid_xy, altitude_z=12.0):
            print(
                f"[teleop_capture] world_arcade viewport pinned to "
                f"/OmniverseKit_Persp; centered at "
                f"({scene_centroid_xy[0]:+.2f}, {scene_centroid_xy[1]:+.2f}, 12.00); "
                f"per-tick follower will track robot XY (zoom preserved).",
                flush=True,
            )

    quit_requested = False
    start_hold_frames = 0
    _START_HOLD_THRESHOLD = int(max(1.0, args.fps)) * 1  # ~1 s of held Start
    rec_paused = False
    a_was_down = False  # rising-edge detect for the A button (pause toggle)
    kept_episodes = 0
    current_candidate: MissionCandidate | None = None
    # Last robot XY at which the follow-cam was repositioned. Seeded
    # far away so the first qualifying step always re-poses, then
    # updated on each successful follow tick. See the gate inside the
    # env-step loop + _ARCADE_FOLLOW_XY_DELTA_M.
    _last_arcade_follow_xy: tuple[float, float] = (float("-inf"), float("-inf"))

    def _begin_next_episode() -> MissionCandidate | None:
        """Prompt operator, reset env, open writer's episode buffer."""
        nonlocal current_candidate
        cand = prompt_for_target(candidates)
        if cand is None:
            return None
        current_candidate = cand
        # Each step prints + flushes so an exception is attributable.
        print(f"[teleop_capture] resetting env for target={cand.label!r} "
              f"id={cand.instance_id}", flush=True)
        env.reset()
        print("[teleop_capture] env.reset OK; sampling robot pose...", flush=True)
        pose, _yaw = _robot_pose(unwrapped)
        # Classify the pose against the active spawn-pool bbox. If the
        # actual pose lands OUTSIDE the bbox, either the env's reset
        # transformed the spawn point (env_origin offset?) or the pool
        # itself includes points outside the room. Either way the
        # operator sees a quantitative signal in the log.
        in_bbox = False
        if spawn_bbox is not None:
            in_bbox = (
                spawn_bbox[0] - 0.5 <= pose[0] <= spawn_bbox[2] + 0.5
                and spawn_bbox[1] - 0.5 <= pose[1] <= spawn_bbox[3] + 0.5
            )
        bbox_marker = "IN spawn-pool bbox" if in_bbox else "OUTSIDE spawn-pool bbox"
        print(
            f"[teleop_capture] start_pose=({pose[0]:+.2f}, {pose[1]:+.2f}, "
            f"yaw={_yaw:+.2f}) — {bbox_marker}",
            flush=True,
        )
        if spawn_bbox is not None and not in_bbox:
            print(
                "[teleop_capture]   WARNING: robot landed outside the active "
                "spawn pool. Either scenes_metadata.json is stale (regenerate "
                "via `generate_scenes_metadata.py`) or env_cfg applies an "
                "env-origin offset the override doesn't account for.",
                file=sys.stderr, flush=True,
            )
        start_xy_yaw = (pose[0], pose[1], _yaw)
        leg_dist = math.sqrt(
            (cand.target_position_3d[0] - pose[0]) ** 2
            + (cand.target_position_3d[1] - pose[1]) ** 2
        )
        writer.begin_episode(
            mission_text=cand.mission_text,
            scene_id=args.scene,
            target_label=cand.label,
            target_object_id=str(cand.instance_id),
            target_position_3d=list(cand.target_position_3d),
            start_pose=list(start_xy_yaw),
            source_driver="teleop",
            source_mission_source="scene-metadata",
            leg_initial_distance_m=leg_dist,
        )
        # Drop the operator-only target marker into the viewport so the
        # operator can see where the chosen object is. Debug-draw is
        # outside Replicator's render product capture, so the marker
        # never enters saved frames.
        target_marker.set_target(cand.target_position_3d)
        print(
            f"\n[teleop_capture] episode {writer.num_episodes} opened: "
            f"target={cand.label!r} id={cand.instance_id} "
            f"pos=({cand.target_position_3d[0]:+.2f}, "
            f"{cand.target_position_3d[1]:+.2f}) "
            f"distance={leg_dist:.2f} m",
            flush=True,
        )
        return cand

    try:
        # Open the first episode by prompting the operator.
        if _begin_next_episode() is None:
            print("[teleop_capture] operator quit at first prompt; exiting cleanly.",
                  flush=True)
            return 0

        episode_step = 0
        last_hud_t = 0.0

        # Hard check Kit's state before entering the loop. A False here
        # means the Sim window never came up (or Kit shut down during
        # env.reset), and entering the while loop would exit immediately
        # with no observable reason. Be loud instead.
        if not simulation_app.is_running():
            print(
                "[teleop_capture] ERROR: simulation_app.is_running() is False "
                "before the capture loop. Kit did not start a viewport or "
                "shut down during env setup. Check that --headless is not "
                "set, that DISPLAY is exported, and that the Isaac Sim "
                "editor window is reachable on this X session.",
                file=sys.stderr, flush=True,
            )
            return 3

        while simulation_app.is_running() and not quit_requested:
            frame = gamepad.read()

            # Save+quit via held Start.
            if frame.buttons.get("start"):
                start_hold_frames += 1
                if start_hold_frames == 1:
                    print("[teleop_capture] Start held — keep holding ~1s to save + quit.")
                if start_hold_frames >= _START_HOLD_THRESHOLD:
                    print(
                        f"[teleop_capture] Start sustained — saving "
                        f"{writer.num_episodes} kept episodes and exiting.",
                    )
                    # Discard whatever in-flight episode is still open.
                    writer.end_episode(discard=True)
                    break
                continue
            else:
                start_hold_frames = 0

            # A toggles REC ↔ PAUSED (rising-edge).
            a_now = bool(frame.buttons.get("a"))
            if a_now and not a_was_down:
                rec_paused = not rec_paused
                print(f"[teleop_capture] {'[PAUSED]' if rec_paused else '[REC]'} capture toggled.")
            a_was_down = a_now

            # Episode-end / discard chord.
            decision = button_state_to_episode_outcome(
                frame.buttons, dpad_x=frame.dpad_x, dpad_y=frame.dpad_y,
            )
            if decision is not None:
                if decision.discard:
                    print(
                        f"  [episode {writer.num_episodes}] DISCARDED "
                        f"({episode_step} steps)",
                    )
                    writer.end_episode(discard=True)
                else:
                    writer.end_episode(
                        outcome=decision.outcome,
                        outcome_category=decision.outcome_category,
                        hard_negative_category=decision.hard_negative_category,
                    )
                    kept_episodes = writer.num_episodes
                    label_hint = (
                        current_candidate.label if current_candidate else "?"
                    )
                    print(
                        f"  [episode {kept_episodes}] outcome={decision.outcome} "
                        f"(target={label_hint!r}, {episode_step} steps)",
                    )
                target_marker.clear()
                episode_step = 0
                time.sleep(0.3)  # debounce

                if kept_episodes >= args.max_episodes:
                    print(
                        f"\n[teleop_capture] reached --max-episodes "
                        f"({args.max_episodes}); saving + quitting.",
                    )
                    break

                if _begin_next_episode() is None:
                    print("[teleop_capture] operator quit at picker; saving + exiting cleanly.")
                    break
                continue

            # Stick → action. Control mode picks world-frame vs body-frame
            # interpretation of the left stick (yaw rate is body-frame in
            # both modes).
            pose, yaw = _robot_pose(unwrapped)
            body_vx, body_vy, omega = _stick_to_body_action(
                frame.lx, frame.ly, frame.rx, yaw,
                control_mode=args.control_mode,
            )

            action = torch.tensor(
                [[body_vx, body_vy, omega]], dtype=torch.float32, device=device,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            episode_step += 1

            # World-arcade follow: re-pose /OmniverseKit_Persp only
            # after the robot has moved past the gate (see the
            # _ARCADE_FOLLOW_XY_DELTA_M comment). Reuses the pose
            # fetched above so we don't pay an extra .cpu() sync.
            if args.control_mode == "world_arcade":
                _dx = pose[0] - _last_arcade_follow_xy[0]
                _dy = pose[1] - _last_arcade_follow_xy[1]
                if (_dx * _dx + _dy * _dy) >= (_ARCADE_FOLLOW_XY_DELTA_M ** 2):
                    _arcade_follow_tick(unwrapped, (pose[0], pose[1]))
                    _last_arcade_follow_xy = (pose[0], pose[1])

            # Only sample for the writer at the chosen capture cadence.
            # env.step ran every loop iteration (full sim tick rate),
            # but writer.add_frame is gated to capture_rate_hz so the
            # operator sees smooth motion in the viewport without
            # inflating the dataset's effective sample rate.
            should_capture = (
                not rec_paused and (episode_step % ticks_per_capture == 0)
            )

            # Pull frames + write to LeRobot only when we're capturing.
            if should_capture:
                rgb_perception = _rgb_to_uint8_hwc(
                    perception_camera.data.output["rgb"],
                )
                depth_m = _depth_to_float32_hw(
                    perception_camera.data.output["distance_to_image_plane"],
                )
                rgb_policy = None
                depth_policy_m = None
                if policy_camera is not None:
                    rgb_policy = _rgb_to_uint8_hwc(
                        policy_camera.data.output["rgb"],
                    )
                    depth_policy_m = _depth_to_float32_hw(
                        policy_camera.data.output["distance_to_image_plane"],
                    )
                achieved = _achieved_vel(unwrapped)
                pose_after, _ = _robot_pose(unwrapped)
                writer.add_frame(
                    sim_time=float(episode_step) / env_step_hz,
                    pose=list(pose_after),
                    achieved_vel=list(achieved),
                    action=[body_vx, body_vy, omega],
                    rgb_perception=rgb_perception,
                    rgb_policy=rgb_policy,
                    depth_m=depth_m,
                    depth_policy_m=depth_policy_m,
                )
            else:
                # No PIP fetch on non-capture steps. PIP redraws at the
                # writer cadence (8 Hz default) — plenty for an operator
                # preview, and saves a cv2.cvtColor + cv2.imshow round
                # trip per env.step that doesn't capture (~3 of every 4
                # ticks at 30 Hz env / 8 Hz writer).
                rgb_perception = None

            # PIP overlay (cosmetic; never reaches LeRobot frames). Only
            # shown on capture steps so we never read perception RGB just
            # to feed PIP.
            target_xy = (
                current_candidate.target_position_3d[:2]
                if current_candidate is not None else (0.0, 0.0)
            )
            dist = math.sqrt(
                (target_xy[0] - pose[0]) ** 2 + (target_xy[1] - pose[1]) ** 2,
            )
            rec_label = "PAUSED" if rec_paused else "REC"
            hud = f"[{rec_label}]  ep={writer.num_episodes}  step={episode_step}  d={dist:.2f}m"
            if should_capture and rgb_perception is not None:
                pip.show(rgb_perception, hud)

            # Console HUD once per second.
            now = time.monotonic()
            if now - last_hud_t >= 1.0:
                mission = current_candidate.mission_text if current_candidate else "?"
                print(
                    f"  {hud}  mission={mission!r}",
                )
                last_hud_t = now

            done = bool(terminated.any().item() or truncated.any().item())
            hit_cap = episode_step >= args.max_steps_per_episode
            if done or hit_cap:
                reason = "termination/timeout" if done else "step cap"
                print(
                    f"  [episode {writer.num_episodes}] auto-closed at "
                    f"{episode_step} steps ({reason}); marking failed.",
                )
                writer.end_episode(outcome="failed", outcome_category="on_course")
                target_marker.clear()
                kept_episodes = writer.num_episodes
                episode_step = 0
                if kept_episodes >= args.max_episodes:
                    print(
                        f"\n[teleop_capture] reached --max-episodes "
                        f"({args.max_episodes}); saving + quitting.",
                    )
                    break
                if _begin_next_episode() is None:
                    print("[teleop_capture] operator quit at picker; saving + exiting cleanly.")
                    break

    except KeyboardInterrupt:
        print("\n[teleop_capture] interrupted — saving collected episodes.",
              flush=True)

    except BaseException as exc:
        # Print and flush BEFORE the finally block runs env.close() /
        # simulation_app.close() — otherwise the Kit shutdown can mask
        # the traceback by collapsing the terminal. The user has been
        # bitten by silent exits ("session vanished after picker") and
        # the symptom is always a missing print here.
        import traceback as _tb
        sys.stderr.write(
            f"\n[teleop_capture] UNCAUGHT EXCEPTION in main loop: "
            f"{type(exc).__name__}: {exc}\n",
        )
        _tb.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        raise

    finally:
        try:
            writer.finalize()
        except Exception as exc:
            print(f"[teleop_capture] writer finalize raised: {exc}", file=sys.stderr)
        try:
            gamepad.close()
        except Exception:
            pass
        try:
            pip.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception as exc:
            print(f"[teleop_capture] env.close raised: {exc}", file=sys.stderr)
        try:
            simulation_app.close()
        except Exception as exc:
            print(f"[teleop_capture] simulation_app.close raised: {exc}",
                  file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()

    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException as exc:
        # Final-line backstop: print + flush before Python exits so the
        # operator always sees the failure reason.
        import traceback as _tb
        sys.stderr.write(
            f"\n[teleop_capture] PROCESS EXIT via {type(exc).__name__}\n",
        )
        _tb.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.stdout.flush()
        sys.exit(1)
    sys.exit(rc)
