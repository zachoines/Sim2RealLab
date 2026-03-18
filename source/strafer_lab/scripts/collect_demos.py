"""Gamepad-driven demonstration collection for behavior cloning.

Launches a single-environment Isaac Lab simulation and records expert
(human) demonstrations using an Xbox / generic gamepad.  Transitions
are saved to HDF5 for offline imitation learning.

Controls:
    Left stick Y   → vx  (forward / backward)
    Left stick X   → vy  (strafe left / right)
    Right stick X  → ω   (yaw rotation)
    A button       → mark current episode as "good" (default)
    B button       → discard current episode
    Start button   → save & quit

Usage:
    isaaclab -p scripts/collect_demos.py \
        --task Isaac-Strafer-Nav-Real-ProcRoom-NoCam-Play-v0 \
        --output demos.h5 \
        [--deadzone 0.12] [--max_episodes 100]
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
                    help="Output HDF5 file path")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments (default 1 for teleop)")
parser.add_argument("--max_episodes", type=int, default=100,
                    help="Stop after N episodes")
parser.add_argument("--deadzone", type=float, default=0.12,
                    help="Gamepad stick deadzone")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Clear sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch simulator (not headless — need viewport for teleop)
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Post-launch imports ---
import h5py
import numpy as np
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401 — registers Isaac Lab tasks
import strafer_lab.tasks  # noqa: F401 — registers Strafer tasks

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

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
    }

    # Button mappings per controller family
    _BUTTON_MAPS = {
        "xbox": {"a": 0, "b": 1, "start": 7},
        "ps5":  {"a": 0, "b": 1, "start": 9},  # Cross, Circle, Options
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

    def read(self) -> tuple[float, float, float, dict[str, bool]]:
        """Read gamepad state.

        Returns:
            (vx, vy, omega) in [-1, 1] and dict of button states.
        """
        pygame.event.pump()
        lx = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LX))
        ly = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LY))
        rx = self._apply_deadzone(self.joystick.get_axis(self.AXIS_RX))

        # stick-up (negative ly) → positive vx (forward)
        # stick-right (positive lx) → negative vy (strafe right, +vy = left)
        # stick-right on rx (positive rx) → negative omega (rotate clockwise/right)
        vx = -ly
        vy = -lx
        omega = -rx

        buttons = {
            "a": self.joystick.get_button(self.BTN_A),
            "b": self.joystick.get_button(self.BTN_B),
            "start": self.joystick.get_button(self.BTN_START),
        }
        return vx, vy, omega, buttons

    def close(self):
        pygame.joystick.quit()
        pygame.quit()


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

class DemoWriter:
    """Accumulates episodes and writes to HDF5."""

    def __init__(self, path: str):
        self.path = path
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
        with h5py.File(self.path, "w") as f:
            f.attrs["num_episodes"] = len(self.episodes)
            f.attrs["total_steps"] = total_steps
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
    # Resolve the env config from the gym registry
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs

    import random

    # Create the environment with the resolved config
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Enable random ProcRoom difficulty per episode (levels 0-7)
    max_difficulty = 7
    unwrapped = env.unwrapped
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

    # --- Gamepad + writer ---
    gamepad = GamepadReader(deadzone=args_cli.deadzone)
    writer = DemoWriter(args_cli.output)
    writer.begin_episode()

    episode_step = 0
    print("\n" + "=" * 60)
    print("DEMO COLLECTION — Use gamepad to drive the robot toward goals")
    print("  Left stick  → forward/strafe")
    print("  Right stick → rotate")
    print("  A button    → keep episode (auto on reset)")
    print("  B button    → discard episode")
    print("  Start       → save & quit")
    print("=" * 60 + "\n")

    try:
        while simulation_app.is_running():
            vx, vy, omega, buttons = gamepad.read()

            if buttons["start"]:
                writer.end_episode(keep=True)
                break

            if buttons["b"]:
                print(f"  [Episode {writer.num_episodes}] DISCARDED ({episode_step} steps)")
                writer.end_episode(keep=False)
                _randomize_difficulty()
                obs, info = env.reset()
                writer.begin_episode()
                episode_step = 0
                time.sleep(0.3)  # Debounce
                continue

            # Build action tensor
            action = torch.tensor([[vx, vy, omega]], dtype=torch.float32,
                                  device=env.unwrapped.device)

            # Record obs+action before step, reward after step
            current_obs = _get_obs(obs)
            current_action = np.array([vx, vy, omega], dtype=np.float32)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            step_reward = float(reward.squeeze().item())
            writer.add_step(current_obs, current_action, reward=step_reward)
            episode_step += 1

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

    finally:
        writer.save()
        gamepad.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
