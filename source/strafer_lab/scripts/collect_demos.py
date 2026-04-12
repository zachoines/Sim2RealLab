"""Gamepad-driven demonstration collection for behavior cloning.

Launches a single-environment Isaac Lab simulation and records expert
(human) demonstrations using an Xbox / generic gamepad.  Transitions
are saved to HDF5 for offline imitation learning.

World-frame control mode (overhead camera, stick = viewport motion):
    Left stick     → world-frame velocity (stick direction = viewport direction)
    Right stick X  → angular velocity (proportional, ±max)
    A button       → mark current episode as "good" (default)
    B button       → discard current episode
    Start button   → save & quit

Output:
    --output can be a ``.h5`` file or a directory.  When a directory is given
    (or a path without ``.h5`` extension), a timestamped filename is
    auto-generated inside it (e.g., ``demos/demos_20260321_143000.h5``).
    This allows incremental collection across sessions — point the training
    script at the folder and all ``.h5`` files will be loaded and concatenated.

Usage:
    # Single file (classic):
    isaaclab -p scripts/collect_demos.py \
        --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \
        --output demos.h5

    # Incremental folder (recommended):
    isaaclab -p scripts/collect_demos.py \
        --task Isaac-Strafer-Nav-Real-ProcRoom-Depth-Play-v0 \
        --output demos/ --max_episodes 40

    # Train on the folder:
    isaaclab -p Scripts/train_strafer_navigation.py \
        --aux dapg --dapg_demos demos/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

# --- CLI parsing (must happen before AppLauncher) ---
parser = argparse.ArgumentParser(description="Collect gamepad demos for BC.")
parser.add_argument("--task", type=str, required=True,
                    help="Registered Isaac Lab task name (Play variant)")
parser.add_argument("--output", type=str, default="demos.h5",
                    help="Output path: .h5 file or directory (auto-generates timestamped filename)")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments (default 1 for teleop)")
parser.add_argument("--max_episodes", type=int, default=100,
                    help="Stop after N episodes")
parser.add_argument("--deadzone", type=float, default=0.15,
                    help="Gamepad stick deadzone")
parser.add_argument("--show_depth", action="store_true",
                    help="Show real-time depth camera feed (matplotlib window)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Auto-enable cameras for variants that use depth/RGB
if "NoCam" not in args_cli.task:
    args_cli.enable_cameras = True

# Launch simulator (not headless — need viewport for teleop)
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Post-launch imports ---
import math

import h5py
import numpy as np
import torch

import gymnasium as gym
import warp as wp

import isaaclab_tasks  # noqa: F401 — registers Isaac Lab tasks
import strafer_lab.tasks  # noqa: F401 — registers Strafer tasks

from isaaclab.envs.common import ViewerCfg
from isaaclab_tasks.utils import parse_env_cfg

# ---------------------------------------------------------------------------
# Pygame gamepad (lazy import — not available in headless servers)
# ---------------------------------------------------------------------------

_PYGAME_AVAILABLE = False
try:
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    pass

_MPL_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("TkAgg")  # Interactive backend for live updates
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except (ImportError, Exception):
    pass


# ---------------------------------------------------------------------------
# Gamepad wrapper
# ---------------------------------------------------------------------------

class GamepadReader:
    """Thin wrapper around pygame joystick for gamepad input.

    Auto-detects Xbox vs PS5 DualSense controllers and maps buttons
    accordingly.  Stick axes are identical across both (0=LX, 1=LY, 3=RX).
    """

    # Stick axes per controller family
    _AXIS_MAPS = {
        "xbox": {"lx": 0, "ly": 1, "rx": 2},
        "ps5":  {"lx": 0, "ly": 1, "rx": 2},
        "switch": {"lx": 0, "ly": 1, "rx": 2},
    }

    # Button mappings per controller family
    _BUTTON_MAPS = {
        "xbox": {"a": 0, "b": 1, "start": 7},
        "ps5":  {"a": 0, "b": 1, "start": 9},  # Cross, Circle, Options
        "switch": {"a": 0, "b": 1, "start": 9},  # B(east), A(south), Plus
    }

    def __init__(self, deadzone: float = 0.12):
        if not _PYGAME_AVAILABLE:
            raise RuntimeError(
                "pygame is required for gamepad input.  Install with: pip install pygame"
            )
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected.  Connect a controller and retry.")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.deadzone = deadzone

        # Auto-detect controller type from name
        name = self.joystick.get_name().lower()
        if "dualsense" in name or "ps5" in name or "sony" in name or "wireless controller" in name:
            self._family = "ps5"
        elif "pro controller" in name or "switch" in name or "nintendo" in name:
            self._family = "switch"
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

        # Prevent pygame from generating QUIT events (which can interfere
        # with the Kit/Omniverse application lifecycle)
        pygame.event.set_blocked(pygame.QUIT)

        print(f"[Gamepad] Connected: {self.joystick.get_name()} (detected: {self._family})")
        print(f"[Gamepad] Axes: {self.joystick.get_numaxes()}, "
              f"Buttons: {self.joystick.get_numbuttons()}")
        print(f"[Gamepad] Mapping: A/Cross={self.BTN_A}, B/Circle={self.BTN_B}, "
              f"Start/Options={self.BTN_START}")

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        # Remap [deadzone, 1] → [0, 1] preserving sign
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def read(self) -> tuple[float, float, float, float, dict[str, bool]]:
        """Read raw gamepad stick values.

        Returns:
            (lx, ly, rx, ry) raw stick axes in [-1, 1] and dict of button states.
            lx/ly: left stick (world-frame velocity direction).
            rx: right stick X (heading target).
            ry: right stick Y (unused, reserved).
        """
        pygame.event.pump()
        # Block QUIT events so pygame doesn't signal the process to exit
        pygame.event.set_blocked(pygame.QUIT)
        lx = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LX))
        ly = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LY))
        rx = self._apply_deadzone(self.joystick.get_axis(self.AXIS_RX))

        buttons = {
            "a": self.joystick.get_button(self.BTN_A),
            "b": self.joystick.get_button(self.BTN_B),
            "start": self.joystick.get_button(self.BTN_START),
        }
        return lx, ly, rx, buttons

    def close(self):
        pygame.joystick.quit()
        pygame.quit()


# ---------------------------------------------------------------------------
# Depth visualizer (OpenCV)
# ---------------------------------------------------------------------------

class DepthVisualizer:
    """Real-time depth image display using matplotlib.

    Shows the robot's depth camera feed in a separate window using the
    turbo colormap (blue=near, red=far) with a colorbar legend.
    Uses matplotlib instead of OpenCV highgui to avoid headless-build issues.
    """

    def __init__(self, height: int = 60, width: int = 80, max_depth: float = 6.0):
        if not _MPL_AVAILABLE:
            raise RuntimeError("matplotlib with TkAgg backend required for depth visualization.")
        self.height = height
        self.width = width
        self.max_depth = max_depth

        plt.ion()
        self._fig, self._ax = plt.subplots(1, 1, figsize=(8, 5))
        self._fig.canvas.manager.set_window_title("Depth Camera (D555)")
        # Initialize with blank image
        blank = np.full((height, width), max_depth, dtype=np.float32)
        self._im = self._ax.imshow(blank, cmap="turbo", vmin=0.0, vmax=max_depth,
                                   interpolation="nearest", aspect="auto")
        self._fig.colorbar(self._im, ax=self._ax, label="Depth (m)")
        self._ax.set_title("D555 Depth")
        self._ax.set_xlabel("px")
        self._ax.set_ylabel("px")
        self._fig.tight_layout()
        self._fig.show()

    def update(self, depth_tensor: torch.Tensor):
        """Display depth image from sensor output.

        Args:
            depth_tensor: Raw depth from camera, shape (num_envs, H, W, 1) or (H, W, 1).
                          Values in meters.
        """
        # Take env 0, squeeze to (H, W)
        if depth_tensor.dim() == 4:
            depth = depth_tensor[0, :, :, 0].cpu().numpy()
        elif depth_tensor.dim() == 3:
            depth = depth_tensor[:, :, 0].cpu().numpy()
        else:
            depth = depth_tensor.cpu().numpy().reshape(self.height, self.width)

        # Replace inf/nan with max_depth
        depth = np.where(np.isfinite(depth), depth, self.max_depth)
        depth = np.clip(depth, 0.0, self.max_depth)

        self._im.set_data(depth)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self):
        plt.close(self._fig)


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

class DemoWriter:
    """Accumulates episodes and writes to HDF5."""

    def __init__(self, path: str, env_name: str = ""):
        self.path = path
        self.env_name = env_name
        self.episodes: list[dict[str, list]] = []
        self._current: dict[str, list] | None = None

    def begin_episode(self):
        self._current = {"obs": [], "actions": [], "rewards": []}

    def add_step(self, obs: np.ndarray, action: np.ndarray, reward: float = 0.0):
        assert self._current is not None, "Call begin_episode() first"
        self._current["obs"].append(obs)
        self._current["actions"].append(action)
        self._current["rewards"].append(reward)

    def end_episode(self, keep: bool = True):
        if self._current is None:
            return
        if keep and len(self._current["obs"]) > 0:
            self.episodes.append(self._current)
        self._current = None

    def save(self):
        if not self.episodes:
            print("[DemoWriter] No episodes to save.")
            return
        total_steps = sum(len(ep["obs"]) for ep in self.episodes)
        print(f"[DemoWriter] Saving {len(self.episodes)} episodes "
              f"({total_steps} total steps) → {self.path}")
        obs_dim = self.episodes[0]["obs"][0].shape[-1] if self.episodes else 0
        with h5py.File(self.path, "w") as f:
            f.attrs["num_episodes"] = len(self.episodes)
            f.attrs["total_steps"] = total_steps
            f.attrs["obs_dim"] = obs_dim
            f.attrs["has_depth"] = bool(obs_dim > 100)
            f.attrs["env_name"] = self.env_name
            for i, ep in enumerate(self.episodes):
                grp = f.create_group(f"episode_{i:04d}")
                grp.create_dataset("obs", data=np.stack(ep["obs"]))
                grp.create_dataset("actions", data=np.stack(ep["actions"]))
                grp.create_dataset("rewards", data=np.array(ep["rewards"], dtype=np.float32))
                grp.attrs["length"] = len(ep["obs"])
                grp.attrs["return"] = float(sum(ep["rewards"]))

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def main():
    # Resolve the env config from the gym registry (parse_env_cfg resolves
    # preset defaults and sets sim.device — matches test_strafer_env.py)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device if hasattr(args_cli, "device") else "cuda:0",
        num_envs=args_cli.num_envs,
    )

    # Sync UI with physics: render every step for responsive teleop
    # env_cfg.sim.render_interval = 1

    # Overhead camera aligned with world axes so stick directions match viewport:
    #   screen right = world +X,  screen up = world +Y
    env_cfg.viewer = ViewerCfg(
        eye=(0.0, 0.0, 12),
        lookat=(0.0, 0.0, 0.0),
        origin_type="env",
        env_index=0,
        resolution=(1280, 720),
    )

    import random

    # Create the environment with the resolved config
    env = gym.make(args_cli.task, cfg=env_cfg)

    unwrapped = env.unwrapped

    # Force the viewport camera to the overhead position. ViewerCfg sets the
    # desired pose but ViewportCameraController may not apply it immediately
    # (or at all in some launch modes). isaacsim's set_camera_view goes
    # through Kit's TransformPrimCommand which is FSD-safe.
    import numpy as np
    from isaacsim.core.utils.viewports import set_camera_view as isaacsim_set_camera_view

    origin = unwrapped.scene.env_origins[0].cpu().numpy()
    cam_eye = origin + np.array(env_cfg.viewer.eye, dtype=float)
    cam_target = origin + np.array(env_cfg.viewer.lookat, dtype=float)
    isaacsim_set_camera_view(
        eye=cam_eye.tolist(),
        target=cam_target.tolist(),
        camera_prim_path=env_cfg.viewer.cam_prim_path,
    )
    print(f"[Demo] Camera set: eye={cam_eye.tolist()}, target={cam_target.tolist()}")

    # Enable random ProcRoom difficulty per episode (levels 0-7)
    max_difficulty = 7
    if hasattr(unwrapped, "_proc_room_difficulty"):
        has_proc_room = True
        print(f"[Demo] ProcRoom detected — randomizing difficulty 0-{max_difficulty} per episode")
    else:
        has_proc_room = False
        # Create the attribute so generate_proc_room uses it
        try:
            unwrapped._proc_room_difficulty = torch.randint(
                0, max_difficulty + 1, (env_cfg.scene.num_envs,), device=unwrapped.device
            )
            has_proc_room = True
            print(f"[Demo] ProcRoom difficulty injected — randomizing 0-{max_difficulty} per episode")
        except Exception:
            print("[Demo] No ProcRoom curriculum — using default env config")

    def _randomize_difficulty():
        if has_proc_room:
            unwrapped._proc_room_difficulty[:] = random.randint(0, max_difficulty)

    _randomize_difficulty()
    obs, info = env.reset()

    # Extract the policy observation
    if isinstance(obs, dict):
        obs_key = "policy" if "policy" in obs else list(obs.keys())[0]
    else:
        obs_key = None

    def _get_obs(o):
        if obs_key is not None:
            return o[obs_key].cpu().numpy().squeeze(0)
        return o.cpu().numpy().squeeze(0)

    # Report obs dimensionality so user can verify depth is included
    sample_obs = _get_obs(obs)
    obs_dim = sample_obs.shape[-1]
    has_depth = obs_dim > 100  # 4819 for depth, 19 for proprio
    print(f"[Demo] obs_dim={obs_dim} ({'includes depth' if has_depth else 'proprio only'})")

    # --- Depth visualization ---
    depth_viz = None
    if args_cli.show_depth and has_depth:
        if _MPL_AVAILABLE:
            try:
                depth_viz = DepthVisualizer(height=60, width=80, max_depth=6.0)
                print("[Demo] Depth visualization enabled (matplotlib window)")
            except Exception as e:
                print(f"[Demo] Depth visualization failed: {e}")
        else:
            print("[Demo] --show_depth requires matplotlib — install with: pip install matplotlib")

    # --- Resolve output path ---
    # If --output is a directory (or doesn't end in .h5), treat it as a folder
    # and auto-generate a timestamped filename inside it.
    output_path = Path(args_cli.output)
    if output_path.is_dir() or (not output_path.suffix and not output_path.exists()):
        from datetime import datetime
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path / f"demos_{timestamp}.h5"
        print(f"[Demo] Output directory mode → {output_path}")

    # --- Gamepad + writer ---
    gamepad = GamepadReader(deadzone=args_cli.deadzone)
    writer = DemoWriter(str(output_path), env_name=args_cli.task)
    writer.begin_episode()

    episode_step = 0
    _heading_print_interval = 60  # Print heading every ~1s (at 60 Hz physics)
    _diag_print_interval = 120  # Print detailed diagnostics every ~2s
    _start_hold_frames = 0  # Counter for sustained Start-button hold
    _START_HOLD_THRESHOLD = 60  # ~1 second at 60 Hz before save & quit

    # Cache wheel joint indices for diagnostics
    robot_asset = unwrapped.scene["robot"]
    _wheel_joint_names = ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]
    _wheel_joint_ids = [robot_asset.joint_names.index(n) for n in _wheel_joint_names]
    print(f"[Demo] Wheel joint indices: {_wheel_joint_ids}")

    def _get_robot_heading() -> float:
        """Extract robot yaw from quaternion (x, y, z, w) — XYZW."""
        quat = wp.to_torch(unwrapped.scene["robot"].data.root_quat_w)[0]  # (4,) — single env
        x, y, z, w = quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    print("\n" + "=" * 60)
    print("DEMO COLLECTION — World-frame gamepad (overhead view)")
    print("  Left stick  → world-frame velocity (stick direction = viewport direction)")
    print("  Right stick → angular velocity (proportional, ±max)")
    print("  A button    → keep episode (auto on reset)")
    print("  B button    → discard episode")
    print("  Start       → save & quit")
    print("=" * 60 + "\n")

    try:
        while simulation_app.is_running():
            lx, ly, rx, buttons = gamepad.read()

            # Check for pygame QUIT events (window close, etc.) which can
            # cause is_running() to return False on the next iteration
            for ev in pygame.event.get(pygame.QUIT):
                print(f"[Demo] WARNING: pygame QUIT event received — ignoring: {ev}")

            if buttons["start"]:
                _start_hold_frames += 1
                if _start_hold_frames == 1:
                    print("[Demo] Start button detected — hold for 1 second to save & quit...")
                if _start_hold_frames >= _START_HOLD_THRESHOLD:
                    print(f"[Demo] Start held — saving {writer.num_episodes} episodes and exiting.")
                    writer.end_episode(keep=True)
                    break
                continue
            else:
                if _start_hold_frames > 0 and _start_hold_frames < _START_HOLD_THRESHOLD:
                    print(f"[Demo] Start released early ({_start_hold_frames} frames) — cancelled.")
                _start_hold_frames = 0

            if buttons["b"]:
                print(f"  [Episode {writer.num_episodes}] DISCARDED ({episode_step} steps)")
                writer.end_episode(keep=False)
                _randomize_difficulty()
                obs, info = env.reset()
                writer.begin_episode()
                episode_step = 0
                time.sleep(0.3)  # Debounce
                continue

            # --- World-frame control ---
            robot_heading = _get_robot_heading()

            # Periodic heading debug print
            if episode_step % _heading_print_interval == 0:
                deg = math.degrees(robot_heading)
                print(f"    [dbg] step={episode_step} heading={deg:+.1f}° "
                      f"stick=({lx:+.2f},{ly:+.2f}) rx={rx:+.2f}")

            # Left stick → world-frame velocity (aligned with overhead viewport)
            # stick-right (+lx) = world +X = screen right
            # stick-up    (-ly) = world +Y = screen up
            world_vx = lx
            world_vy = -ly
            stick_mag = min(1.0, math.sqrt(world_vx ** 2 + world_vy ** 2))

            if stick_mag > 0.01:
                # Normalize direction then scale by magnitude
                inv_norm = stick_mag / math.sqrt(world_vx ** 2 + world_vy ** 2)
                world_vx *= inv_norm
                world_vy *= inv_norm

            # Transform world-frame velocity to robot body frame
            cos_h = math.cos(robot_heading)
            sin_h = math.sin(robot_heading)
            body_vx = cos_h * world_vx + sin_h * world_vy
            body_vy = -sin_h * world_vx + cos_h * world_vy

            # Right stick X → angular velocity (negate: stick-left = CCW = positive omega)
            omega = -rx

            # Zero-threshold: if no gamepad input, send exact zero to prevent
            # micro-movements from floating-point residuals
            if stick_mag < 0.01 and abs(omega) < 0.01:
                body_vx = 0.0
                body_vy = 0.0
                omega = 0.0

            # Build action tensor [body_vx, body_vy, omega]
            action = torch.tensor([[body_vx, body_vy, omega]], dtype=torch.float32,
                                  device=unwrapped.device)

            # Record obs+action before step, reward after step
            current_obs = _get_obs(obs)
            current_action = np.array([body_vx, body_vy, omega], dtype=np.float32)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            step_reward = float(reward.squeeze().item())
            writer.add_step(current_obs, current_action, reward=step_reward)
            episode_step += 1

            # Update depth visualization (every 2nd frame to reduce overhead)
            if depth_viz is not None and episode_step % 2 == 0:
                try:
                    camera = unwrapped.scene.sensors["d555_camera"]
                    depth_data = wp.to_torch(camera.data.output["distance_to_image_plane"])
                    depth_viz.update(depth_data)
                except Exception:
                    pass  # Don't crash demo collection on viz errors

            if episode_step % _diag_print_interval == 0:
                rd = robot_asset.data
                wv = wp.to_torch(rd.joint_vel)[0, _wheel_joint_ids].cpu().numpy()
                bv = wp.to_torch(rd.root_lin_vel_b)[0, :2].cpu().numpy()
                print(f"    [diag] wheel_vel(rad/s)=[{wv[0]:+.2f},{wv[1]:+.2f},{wv[2]:+.2f},{wv[3]:+.2f}]"
                      f"  body_vel(m/s)=[vx={bv[0]:+.3f}, vy={bv[1]:+.3f}]"
                      f"  cmd=[{body_vx:+.2f},{body_vy:+.2f},{omega:+.2f}]")

            # Episode ended (timeout or termination)
            done = terminated.any().item() or truncated.any().item()
            if done:
                writer.end_episode(keep=True)
                n = writer.num_episodes
                lvl = unwrapped._proc_room_difficulty[0].item() if has_proc_room else "N/A"
                print(f"  [Episode {n}] saved ({episode_step} steps, difficulty={lvl})")
                if n >= args_cli.max_episodes:
                    print(f"\nReached {args_cli.max_episodes} episodes — stopping.")
                    break
                _randomize_difficulty()
                obs, info = env.reset()
                writer.begin_episode()
                episode_step = 0

    except KeyboardInterrupt:
        print("\nInterrupted — saving collected episodes...")
        writer.end_episode(keep=True)

    except Exception as e:
        import traceback
        print(f"\n[Demo] ERROR: Unexpected exception — {type(e).__name__}: {e}")
        traceback.print_exc()
        writer.end_episode(keep=True)

    else:
        # Loop exited normally (simulation_app.is_running() returned False)
        print(f"\n[Demo] WARNING: simulation_app.is_running() returned False — Kit shut down.")
        print(f"[Demo] Collected {writer.num_episodes} episodes before shutdown.")
        writer.end_episode(keep=True)

    finally:
        writer.save()
        if depth_viz is not None:
            depth_viz.close()
        gamepad.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
