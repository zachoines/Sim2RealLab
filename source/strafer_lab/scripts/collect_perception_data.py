"""Gamepad-driven perception data collection for Infinigen scenes.

Launches the ``Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0`` env
(640x360 D555 perception camera + policy camera + Infinigen scene
geometry, see ``d555_cfg.py`` / ``strafer_env_cfg.py``) and records a
human-driven teleop session as one episode directory per run:

    <output>/
        episode_0000/
            frames.jsonl
            frame_0000.jpg
            frame_0000.depth.npy   # if --save-depth (default on)
            frame_0001.jpg
            ...
        episode_0001/
            ...
        writer_stats.json

The layout matches what :mod:`strafer_lab.scripts.generate_descriptions`
and :mod:`strafer_lab.scripts.prepare_vlm_finetune_data` already consume,
so the downstream description / SFT pipelines can run on this tree the
moment a session finishes with no translation step.

Controls (same as ``collect_demos.py``):

    Left stick  → world-frame velocity (direction = overhead view direction)
    Right stick → angular velocity
    A button    → mark current episode "keep" (also the default on reset)
    B button    → discard current episode
    Start       → save & quit

The env always advances. After the user keeps or discards an episode
the script calls ``env.reset()`` and starts the next episode. If
``--max-episodes`` is reached the loop exits and writer stats are
flushed.

Usage:

    # Single scene
    isaaclab -p scripts/collect_perception_data.py \\
        --scene scene_001 \\
        --output data/perception/ \\
        --max-episodes 20

    # Different scene USD (override the auto-picked first Infinigen scene)
    isaaclab -p scripts/collect_perception_data.py \\
        --scene-usd Assets/generated/scenes/kitchen_42/kitchen_42.usd \\
        --output data/perception/

Not intended for RL training — this env runs at 1 env only because
640x360 caps Isaac Sim throughput.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI parsing (must happen before AppLauncher boot)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0",
    help="Registered Isaac Lab task name. Default is the perception "
    "env that carries both the policy and perception cameras.",
)
parser.add_argument(
    "--scene",
    type=str,
    default=None,
    help="Friendly scene name (for the frames.jsonl scene_name field). "
    "Also used to resolve --scene-usd when that flag is not set.",
)
parser.add_argument(
    "--scene-usd",
    type=str,
    default=None,
    help="Absolute path to the Infinigen scene USD file. If omitted the env "
    "keeps whatever usd_path _apply_infinigen_scene_setup chose as default.",
)
parser.add_argument(
    "--output",
    type=str,
    default="data/perception/",
    help="Output directory root. One episode_NNNN/ subdir is created per "
    "teleop run. Incremental across sessions — existing episode dirs are "
    "preserved and the next index continues after the highest existing one.",
)
parser.add_argument(
    "--max-episodes",
    type=int,
    default=20,
    help="Stop after this many KEPT episodes (discards do not count).",
)
parser.add_argument(
    "--max-steps-per-episode",
    type=int,
    default=500,
    help="Hard cap on frames per episode.",
)
parser.add_argument(
    "--deadzone",
    type=float,
    default=0.15,
    help="Gamepad stick deadzone.",
)
parser.add_argument(
    "--save-depth",
    action="store_true",
    default=True,
    help="Save distance_to_image_plane as <frame>.depth.npy (default on).",
)
parser.add_argument(
    "--no-save-depth",
    dest="save_depth",
    action="store_false",
    help="Disable depth save to shrink disk usage.",
)
parser.add_argument(
    "--jpeg-quality",
    type=int,
    default=90,
    help="JPEG quality for RGB frames.",
)
parser.add_argument(
    "--min-visible",
    type=float,
    default=0.1,
    help="Drop bboxes whose visible fraction (1 - occlusion_ratio) is below "
    "this threshold. Default 0.1 keeps almost everything; raise to 0.5 for "
    "only clearly-visible objects.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv so Hydra does not choke on our args
sys.argv = [sys.argv[0]] + hydra_args

# Perception env always needs cameras enabled
args_cli.enable_cameras = True
# Teleop needs a viewport
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Post-launch imports — must come AFTER AppLauncher because they pull in
# isaaclab / omni modules that require the Kit runtime to be active.
# ---------------------------------------------------------------------------

import math

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401 — registers Isaac Lab task metadata
import strafer_lab.tasks  # noqa: F401 — registers Strafer envs

from isaaclab.envs.common import ViewerCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

from strafer_lab.tools.bbox_extractor import ReplicatorBboxExtractor
from strafer_lab.tools.perception_writer import PerceptionFrameWriter


# Pygame gamepad — lazy so the script still imports on headless CI for
# --help etc.
_PYGAME_AVAILABLE = False
try:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Gamepad wrapper — copy of collect_demos.py's GamepadReader. Kept inline
# here (rather than extracted into a shared module) so this script stands
# alone and does not force a refactor of collect_demos.py.
# ---------------------------------------------------------------------------


class GamepadReader:
    """Thin wrapper around pygame joystick for Xbox / DualSense gamepads."""

    _AXIS_MAPS = {
        "xbox": {"lx": 0, "ly": 1, "rx": 2},
        "ps5": {"lx": 0, "ly": 1, "rx": 2},
    }
    _BUTTON_MAPS = {
        "xbox": {"a": 0, "b": 1, "start": 7},
        "ps5": {"a": 0, "b": 1, "start": 9},
    }

    def __init__(self, deadzone: float = 0.12):
        if not _PYGAME_AVAILABLE:
            raise RuntimeError(
                "pygame is required for gamepad input. Install with: pip install pygame"
            )
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected. Connect a controller and retry.")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.deadzone = deadzone

        name = self.joystick.get_name().lower()
        if any(tag in name for tag in ("dualsense", "ps5", "sony", "wireless controller")):
            self._family = "ps5"
        else:
            self._family = "xbox"
        bmap = self._BUTTON_MAPS[self._family]
        self.BTN_A = bmap["a"]
        self.BTN_B = bmap["b"]
        self.BTN_START = bmap["start"]
        amap = self._AXIS_MAPS[self._family]
        self.AXIS_LX = amap["lx"]
        self.AXIS_LY = amap["ly"]
        self.AXIS_RX = amap["rx"]

        print(f"[Gamepad] Connected: {self.joystick.get_name()} (family={self._family})")

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def read(self) -> tuple[float, float, float, dict[str, bool]]:
        pygame.event.pump()
        lx = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LX))
        ly = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LY))
        rx = self._apply_deadzone(self.joystick.get_axis(self.AXIS_RX))
        buttons = {
            "a": bool(self.joystick.get_button(self.BTN_A)),
            "b": bool(self.joystick.get_button(self.BTN_B)),
            "start": bool(self.joystick.get_button(self.BTN_START)),
        }
        return lx, ly, rx, buttons

    def close(self):
        pygame.joystick.quit()
        pygame.quit()


# ---------------------------------------------------------------------------
# Collection helpers — pulled out of main() so the loop body stays short.
# ---------------------------------------------------------------------------


def _robot_heading(unwrapped) -> float:
    """Extract robot yaw from the ``(w, x, y, z)`` base quaternion."""
    quat = unwrapped.scene["robot"].data.root_quat_w[0]
    w, x, y, z = (quat[i].item() for i in range(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _stick_to_body_action(
    lx: float, ly: float, rx: float, heading: float,
) -> tuple[float, float, float]:
    """Convert left-stick world-frame intent to body-frame ``(vx, vy, omega)``.

    Mirrors collect_demos.py: left stick drives world-frame velocity
    (overhead view convention), then a yaw rotation into body frame.
    Right-stick-X is negated so stick-left = CCW positive omega, also
    matching collect_demos.py's pattern.
    """
    world_vx = lx
    world_vy = -ly
    stick_mag = min(1.0, math.sqrt(world_vx ** 2 + world_vy ** 2))

    if stick_mag > 0.01:
        # Normalize + scale so the magnitude of the command never exceeds 1.
        norm_mag = math.sqrt(world_vx ** 2 + world_vy ** 2)
        if norm_mag > 0.0:
            world_vx *= stick_mag / norm_mag
            world_vy *= stick_mag / norm_mag

    cos_h, sin_h = math.cos(heading), math.sin(heading)
    body_vx = cos_h * world_vx + sin_h * world_vy
    body_vy = -sin_h * world_vx + cos_h * world_vy
    omega = -rx

    if stick_mag < 0.01 and abs(omega) < 0.01:
        return 0.0, 0.0, 0.0
    return body_vx, body_vy, omega


def _rgb_to_numpy(rgb_tensor) -> np.ndarray:
    """Isaac Sim returns ``(num_envs, H, W, 3|4)`` on GPU. Take env 0, move to CPU."""
    if rgb_tensor.dim() == 4:
        arr = rgb_tensor[0].cpu().numpy()
    else:
        arr = rgb_tensor.cpu().numpy()
    return arr


def _depth_to_numpy(depth_tensor) -> np.ndarray:
    """``(num_envs, H, W, 1)`` or ``(num_envs, H, W)`` → ``(H, W)`` float32."""
    if depth_tensor.dim() == 4:
        arr = depth_tensor[0].cpu().numpy()
    else:
        arr = depth_tensor.cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def _vec3_to_list(tensor) -> list[float]:
    return [float(x) for x in tensor[0].cpu().numpy().tolist()]


def _vec4_to_list(tensor) -> list[float]:
    return [float(x) for x in tensor[0].cpu().numpy().tolist()]


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    # Resolve env cfg and override the scene USD path if the user asked.
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")

    if args_cli.scene_usd is not None:
        env_cfg.scene.scene_geometry.spawn.usd_path = str(Path(args_cli.scene_usd).resolve())
        print(f"[Perception] scene USD override → {env_cfg.scene.scene_geometry.spawn.usd_path}")

    # Overhead camera for gamepad teleop, matching collect_demos.py.
    env_cfg.viewer = ViewerCfg(
        eye=(0.0, 0.0, 12.0),
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
        env_index=0,
        resolution=(1280, 720),
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device

    scene = unwrapped.scene
    if "d555_camera_perception" not in scene.sensors:
        print(
            "[Perception] ERROR: d555_camera_perception not in scene. "
            "This script only works with the Infinigen perception env.",
            file=sys.stderr,
        )
        env.close()
        simulation_app.close()
        return 2

    perception_camera = scene.sensors["d555_camera_perception"]
    render_product_path = perception_camera.render_product_paths[0]
    print(f"[Perception] perception camera render product: {render_product_path}")
    print(
        f"[Perception] perception resolution: "
        f"{perception_camera.cfg.width}x{perception_camera.cfg.height}"
    )

    bbox_extractor = ReplicatorBboxExtractor(
        camera_render_product_path=render_product_path,
        min_occlusion_visible=args_cli.min_visible,
    )

    output_root = Path(args_cli.output)
    writer = PerceptionFrameWriter(
        output_root=output_root,
        jpeg_quality=args_cli.jpeg_quality,
        depth_enabled=args_cli.save_depth,
    )
    print(
        f"[Perception] writer output root: {writer.output_root} "
        f"(next episode index: {writer.next_episode_index})"
    )

    gamepad = GamepadReader(deadzone=args_cli.deadzone)

    scene_name_field = (
        args_cli.scene
        or (Path(args_cli.scene_usd).stem if args_cli.scene_usd else "unknown_infinigen_scene")
    )

    print("\n" + "=" * 60)
    print("PERCEPTION DATA COLLECTION — World-frame gamepad (overhead view)")
    print(f"  scene_name (in frames.jsonl): {scene_name_field!r}")
    print(f"  max kept episodes: {args_cli.max_episodes}")
    print(f"  max steps / episode: {args_cli.max_steps_per_episode}")
    print(f"  save_depth: {args_cli.save_depth}")
    print(f"  min_visible (occlusion filter): {args_cli.min_visible}")
    print("  Left stick  → world-frame velocity")
    print("  Right stick → angular velocity")
    print("  A button    → keep episode (default)")
    print("  B button    → discard episode")
    print("  Start       → save & quit")
    print("=" * 60 + "\n")

    obs, info = env.reset()
    writer.begin_episode()
    episode_step = 0
    quit_requested = False

    try:
        while simulation_app.is_running() and not quit_requested:
            lx, ly, rx, buttons = gamepad.read()

            if buttons["start"]:
                quit_requested = True
                break

            if buttons["b"]:
                print(
                    f"  [Episode {writer.stats.episodes_kept + writer.stats.episodes_discarded}]"
                    f" DISCARDED ({episode_step} steps)"
                )
                writer.end_episode(keep=False)
                obs, info = env.reset()
                writer.begin_episode()
                episode_step = 0
                time.sleep(0.3)  # debounce
                continue

            heading = _robot_heading(unwrapped)
            body_vx, body_vy, omega = _stick_to_body_action(lx, ly, rx, heading)
            action = torch.tensor(
                [[body_vx, body_vy, omega]], dtype=torch.float32, device=device,
            )

            # Step the sim BEFORE reading camera / bbox data so the
            # annotators correspond to the frame we are about to record.
            obs, reward, terminated, truncated, info = env.step(action)

            rgb = _rgb_to_numpy(perception_camera.data.output["rgb"])
            depth = None
            if args_cli.save_depth:
                depth = _depth_to_numpy(
                    perception_camera.data.output["distance_to_image_plane"]
                )
            bboxes = bbox_extractor.extract_as_dicts()

            robot = scene["robot"]
            writer.save_frame(
                frame_id=episode_step,
                rgb=rgb,
                depth=depth,
                scene_name=scene_name_field,
                scene_type="infinigen",
                robot_pos=_vec3_to_list(robot.data.root_pos_w),
                robot_quat=_vec4_to_list(robot.data.root_quat_w),
                cam_pos=_vec3_to_list(perception_camera.data.pos_w),
                cam_quat=_vec4_to_list(perception_camera.data.quat_w_world),
                bboxes=bboxes,
                image_width=int(perception_camera.cfg.width),
                image_height=int(perception_camera.cfg.height),
                extras={
                    "gamepad": {"lx": float(lx), "ly": float(ly), "rx": float(rx)},
                },
            )
            episode_step += 1

            done = bool(terminated.any().item() or truncated.any().item())
            hit_step_cap = episode_step >= args_cli.max_steps_per_episode
            if done or hit_step_cap:
                writer.end_episode(keep=True)
                n = writer.stats.episodes_kept
                reason = "timeout/termination" if done else "step cap"
                print(f"  [Episode {n}] kept ({episode_step} steps, {reason})")
                if n >= args_cli.max_episodes:
                    print(f"\nReached {args_cli.max_episodes} kept episodes — stopping.")
                    quit_requested = True
                    break
                obs, info = env.reset()
                writer.begin_episode()
                episode_step = 0

    except KeyboardInterrupt:
        print("\nInterrupted — saving collected episodes.")

    finally:
        writer.close()
        stats_path = writer.write_stats()
        print(f"[Perception] writer stats → {stats_path}")
        print(f"[Perception] {writer.stats.to_dict()}")
        try:
            gamepad.close()
        except Exception:
            pass
        env.close()
        simulation_app.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
