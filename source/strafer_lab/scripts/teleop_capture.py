"""In-process Isaac Lab teleop driver for the harness data-capture system.

The implementation of ``Scripts/capture.py --driver teleop
--mission-source scene-metadata`` per
``docs/tasks/active/harness/harness-architecture.md`` (Tier 1). Boots
Isaac Sim headed, loads the InfinigenPerception env, reads a gamepad,
drives ``env.step()``, captures frames via
:class:`strafer_lab.tools.lerobot_writer.StraferLeRobotWriter`, and
writes a LeRobot v3 dataset under the supplied output root.

Operator UX baseline (per the brief's "Operator UX" section):

- **Live third-person view**: Isaac Sim editor viewport with an
  overhead camera (eye=(0,0,12), origin at env 0). Same setup as
  ``collect_demos.py`` so operators don't have to relearn the camera.
- **First-person PIP**: an OpenCV ``cv2.imshow`` window showing the
  ``d555_camera_perception`` 640×360 stream. Catches "target visible to
  operator but blocked from D555 FoV" failures before they pollute the
  corpus. **The window is a separate top-level Qt/X surface; the
  perception camera's render product writes to disk before any
  viewport-side overlay drawing**, so the PIP does NOT leak into
  captured frames. The brief explicitly calls this out as an acceptance
  criterion.
- **HUD mission text**: console echo of the active ``mission_text``
  once per second + when it changes.
- **REC indicator**: ``[REC]`` / ``[PAUSED]`` console banner each tick;
  the operator can hold ``A`` to pause capture (keeps the env stepping
  but the writer's ``add_frame`` is skipped) when they want to
  reposition without recording.

Invocation (preferred — via the unified entry point)::

    isaaclab -p Scripts/capture.py \\
        --driver teleop --mission-source scene-metadata \\
        --scene scene_high_quality_dgx_000_seed0 \\
        --output data/sim_in_the_loop/scene_high_quality_dgx_000_seed0

Direct invocation (rarely needed; ``capture.py`` is the canonical entry
point)::

    isaaclab -p source/strafer_lab/scripts/teleop_capture.py \\
        --scene scene_high_quality_dgx_000_seed0 \\
        --scene-metadata Assets/generated/scenes/.../scene_metadata.json \\
        --output data/sim_in_the_loop/scene_high_quality_dgx_000_seed0

This script must NOT live under ``source/strafer_lab/strafer_lab/`` —
that namespace is for non-Isaac-Sim-importable code (see
``strafer_lab/__init__.py`` docstring). It belongs alongside
``collect_demos.py`` in ``source/strafer_lab/scripts/``.
"""

from __future__ import annotations

import argparse
import math
import os
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
        help="Capture rate in Hz. Default 8 matches the bridge mainloop's "
        "post-decimation rate (see harness brief).",
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
    AppLauncher.add_app_launcher_args(parser)
    return parser


_PARSER = _build_parser()
args_cli, hydra_args = _PARSER.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Force headed + cameras — these are non-negotiable for the teleop UX.
args_cli.enable_cameras = True
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Post-launch imports — these pull in omni / isaaclab modules that need
# the Kit runtime to be active.
# ---------------------------------------------------------------------------

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401 — registers Isaac Lab task metadata
import strafer_lab.tasks  # noqa: F401 — registers Strafer envs

from isaaclab.envs.common import ViewerCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from strafer_lab.tools.gamepad_reader import GamepadReader
from strafer_lab.tools.lerobot_writer import (
    StraferLeRobotWriter,
    hash_scene_metadata,
)
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


def _resolve_scene_metadata_path(scene: str, override: str | None) -> Path:
    """Resolve the scene_metadata.json path from --scene + optional override."""
    if override:
        path = Path(override)
    else:
        path = Path("Assets/generated/scenes") / scene / "scene_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"scene_metadata.json not found: {path}. Pass --scene-metadata "
            "explicitly if the scene lives outside Assets/generated/scenes/.",
        )
    return path


def _robot_pose(unwrapped) -> tuple[tuple[float, float, float, float, float, float, float], float]:
    """Return ``(pose_xyzqxqyqzqw, yaw_rad)`` for env 0.

    Isaac Lab quaternions are ``(w, x, y, z)``; we re-order to the
    ``(qx, qy, qz, qw)`` layout the LeRobot writer expects.
    """
    scene = unwrapped.scene
    pos = scene["robot"].data.root_pos_w[0].cpu().numpy()
    quat_wxyz = scene["robot"].data.root_quat_w[0].cpu().numpy()
    w, x, y, z = (float(quat_wxyz[i]) for i in range(4))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    pose = (
        float(pos[0]), float(pos[1]), float(pos[2]),
        x, y, z, w,
    )
    return pose, yaw


def _achieved_vel(unwrapped) -> tuple[float, float, float]:
    """Return ``(vx, vy, omega_z)`` from the robot's body-frame velocities."""
    robot = unwrapped.scene["robot"]
    lin_b = robot.data.root_lin_vel_b[0].cpu().numpy()
    ang_w = robot.data.root_ang_vel_w[0].cpu().numpy()
    return float(lin_b[0]), float(lin_b[1]), float(ang_w[2])


def _stick_to_body_action(
    lx: float, ly: float, rx: float, heading: float,
) -> tuple[float, float, float]:
    """World-frame stick → body-frame ``(vx, vy, omega_z)``.

    Mirrors ``collect_demos.py``: overhead viewport convention,
    right-stick X = angular vel (negated so stick-left = CCW positive
    omega).
    """
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


def _rgb_to_uint8_hwc(tensor) -> np.ndarray:
    """Isaac Sim ``(N, H, W, C)`` RGB → ``(H, W, 3)`` uint8.

    Drops the alpha channel if present. Replicator camera outputs are
    typically uint8 already; clamp + cast just in case.
    """
    if tensor.dim() == 4:
        arr = tensor[0].cpu().numpy()
    else:
        arr = tensor.cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _depth_to_float32_hw(tensor) -> np.ndarray:
    """Isaac Sim ``(N, H, W, 1)`` distance → ``(H, W)`` float32 meters."""
    if tensor.dim() == 4:
        arr = tensor[0].cpu().numpy()
    else:
        arr = tensor.cpu().numpy()
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
      (``opencv-python-headless`` ships without GTK / Qt — env_isaaclab3
      installs this variant) → no-op + one-shot warning naming the
      culprit so the operator knows the editor viewport is their only
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

    scene_metadata_path = _resolve_scene_metadata_path(args.scene, args.scene_metadata)
    scene_metadata_sha = hash_scene_metadata(scene_metadata_path)
    print(f"[teleop_capture] scene_metadata: {scene_metadata_path}")
    print(f"[teleop_capture] scene_metadata sha256: {scene_metadata_sha[:16]}...")
    print(f"[teleop_capture] capture_git_sha: {capture_git_sha[:12] or '(none)'}")

    candidates = load_candidates(
        scene_metadata_path,
        allowed_labels=args.target_label_filter,
    )
    # Drop degenerate (0, 0, 0) targets. Infinigen spawns creature
    # prims (carnivore / herbivore / etc.) at origin until placement
    # finalizes; --from-usd extraction keeps their entries but the
    # position is meaningless. Picker offering them sends the operator
    # toward 0,0 and surfaces nothing usable.
    pre_drop = len(candidates)
    candidates = [
        c for c in candidates
        if (
            abs(c.target_position_3d[0]) > 1e-3
            or abs(c.target_position_3d[1]) > 1e-3
            or abs(c.target_position_3d[2]) > 1e-3
        )
    ]
    # Reindex so indices are dense again after the filter.
    candidates = [
        type(c)(
            index=i,
            instance_id=c.instance_id,
            label=c.label,
            target_position_3d=c.target_position_3d,
            target_room_idx=c.target_room_idx,
            target_room_type=c.target_room_type,
            prim_path=c.prim_path,
            mission_text=c.mission_text,
        )
        for i, c in enumerate(candidates)
    ]
    dropped = pre_drop - len(candidates)
    if dropped:
        print(
            f"[teleop_capture] dropped {dropped} target(s) at origin "
            "(Infinigen pre-placement prims); "
            f"{len(candidates)} remain after filter.",
            flush=True,
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
    env_cfg.viewer = ViewerCfg(
        eye=(0.0, 0.0, 12.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
        env_index=0,
        resolution=(1280, 720),
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

    writer = _build_writer(
        output_root=output_root,
        repo_id=repo_id,
        fps=int(args.fps),
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
    print(f"  fps               : {args.fps}")
    print(f"  max episodes      : {args.max_episodes}")
    print(f"  max steps/episode : {args.max_steps_per_episode}")
    print(f"  headless          : {bool(args_cli.headless)}  "
          f"(False means Isaac Sim editor viewport should be visible)")
    print(f"  enable_cameras    : {bool(args_cli.enable_cameras)}")
    print(f"  simulation_app.is_running(): {simulation_app.is_running()}")
    print("-" * 64)
    print(describe_button_layout())
    print("=" * 64 + "\n", flush=True)

    obs, info = env.reset()

    quit_requested = False
    start_hold_frames = 0
    _START_HOLD_THRESHOLD = int(max(1.0, args.fps)) * 1  # ~1 s of held Start
    rec_paused = False
    a_was_down = False  # rising-edge detect for the A button (pause toggle)
    kept_episodes = 0
    current_candidate: MissionCandidate | None = None

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
        print(f"[teleop_capture] start_pose=({pose[0]:+.2f}, {pose[1]:+.2f}, "
              f"yaw={_yaw:+.2f})", flush=True)
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
        # with no observable reason — a silent-exit failure mode that
        # has bitten the operator before. Be loud instead.
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

            # Stick → action.
            pose, yaw = _robot_pose(unwrapped)
            body_vx, body_vy, omega = _stick_to_body_action(
                frame.lx, frame.ly, frame.rx, yaw,
            )
            action = torch.tensor(
                [[body_vx, body_vy, omega]], dtype=torch.float32, device=device,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            episode_step += 1

            # Pull frames + write to LeRobot.
            rgb_perception = _rgb_to_uint8_hwc(perception_camera.data.output["rgb"])
            depth_m = _depth_to_float32_hw(
                perception_camera.data.output["distance_to_image_plane"],
            )
            rgb_policy = None
            if policy_camera is not None:
                rgb_policy = _rgb_to_uint8_hwc(policy_camera.data.output["rgb"])

            if not rec_paused:
                achieved = _achieved_vel(unwrapped)
                pose_after, _ = _robot_pose(unwrapped)
                writer.add_frame(
                    sim_time=float(episode_step) / float(args.fps),
                    pose=list(pose_after),
                    achieved_vel=list(achieved),
                    action=[body_vx, body_vy, omega],
                    rgb_perception=rgb_perception,
                    rgb_policy=rgb_policy,
                    depth_m=depth_m,
                )

            # PIP overlay (cosmetic; never reaches LeRobot frames).
            target_xy = (
                current_candidate.target_position_3d[:2]
                if current_candidate is not None else (0.0, 0.0)
            )
            dist = math.sqrt(
                (target_xy[0] - pose[0]) ** 2 + (target_xy[1] - pose[1]) ** 2,
            )
            rec_label = "PAUSED" if rec_paused else "REC"
            hud = f"[{rec_label}]  ep={writer.num_episodes}  step={episode_step}  d={dist:.2f}m"
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
