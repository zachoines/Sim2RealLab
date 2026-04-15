"""Boot the Strafer perception env with the ROS2 bridge wired up.

This is the DGX-side entry point for sim-in-the-loop runs. It launches
Isaac Sim via ``AppLauncher``, instantiates the
``Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0`` env at num_envs=1,
enables the bundled ``isaacsim.ros2.bridge`` extension, builds the
Strafer OmniGraph (cameras, odom, TF, /cmd_vel subscribe), and then
drives the env step loop while reading /cmd_vel from the subscribed
Twist node and injecting it into the Isaac Lab action tensor.

The Jetson at ``STRAFER_JETSON_HOST`` then runs the real-robot nav stack
unmodified against the topics published here:

    /d555/color/image_raw          sensor_msgs/Image
    /d555/color/camera_info        sensor_msgs/CameraInfo
    /d555/depth/image_rect_raw     sensor_msgs/Image
    /d555/depth/camera_info        sensor_msgs/CameraInfo
    /strafer/odom                  nav_msgs/Odometry
    tf                             odom→base_link, base_link→d555_link
    /cmd_vel                       geometry_msgs/Twist  (SUBSCRIBE)

Usage:

    source env_setup.sh   # loads ROS_DOMAIN_ID, RMW_IMPLEMENTATION, LD_PRELOAD
    isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py

Verify from another shell (Jetson or DGX):

    ros2 topic list
    ros2 topic hz /d555/color/image_raw
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.3}}'
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0",
        help="Isaac Lab task carrying the 640x360 perception camera. "
        "num_envs is forced to 1 for sim-in-the-loop.",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="/World/ROS2Bridge",
        help="USD path for the OmniGraph prim the bridge is built into.",
    )
    parser.add_argument(
        "--cmd-vel-timeout",
        type=float,
        default=0.5,
        help="Seconds with no /cmd_vel message before the robot is held still.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # AppLauncher must boot Isaac Sim before any omni.* / strafer_lab.tasks imports.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import strafer_lab.tasks  # noqa: F401

    from isaacsim.core.utils.extensions import enable_extension

    from strafer_lab.bridge.config import build_default_bridge_config
    from strafer_lab.bridge.graph import build_bridge_graph, read_cmd_vel

    # --- 1. Enable the bridge extension before any OmniGraph nodes are created.
    # ``isaacsim.ros2.bridge`` registers its node types with OmniGraph at
    # extension-enable time; attempting ``og.Controller.edit(...)`` with
    # ``isaacsim.ros2.bridge.*`` before the extension is hot throws a
    # "node type not registered" error.
    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    # --- 2. Build the env at num_envs=1.
    env = gym.make(args.task, num_envs=1)
    unwrapped = env.unwrapped

    # --- 3. Wire the ROS2 bridge graph onto the live stage.
    config = build_default_bridge_config(graph_path=args.graph_path)
    build_bridge_graph(config)
    print(f"[sim_in_the_loop] bridge graph built at {args.graph_path}")
    print(f"[sim_in_the_loop] chassis_prim={config.chassis_prim_path}")
    print(f"[sim_in_the_loop] color camera prim={config.color_camera.camera_prim_path}")

    # --- 4. Reset and enter the drive loop.
    env.reset()

    action_shape = unwrapped.action_manager.action.shape
    print(f"[sim_in_the_loop] action tensor shape = {tuple(action_shape)}")

    device = unwrapped.device
    zero_action = torch.zeros(action_shape, device=device)

    last_cmd_time = time.monotonic()

    while simulation_app.is_running():
        linear, angular = read_cmd_vel(args.graph_path)
        vx, vy, _vz = linear
        _wx, _wy, wz = angular

        now = time.monotonic()
        if any(abs(v) > 1e-6 for v in (vx, vy, wz)):
            last_cmd_time = now
            action = zero_action.clone()
            # Strafer mecanum action layout: [linear_x, linear_y, angular_z]
            # assumed at index 0, 1, 2 of the first env's action row. Any
            # different task wiring will need a per-task adapter.
            if action.shape[-1] >= 3:
                action[0, 0] = float(vx)
                action[0, 1] = float(vy)
                action[0, 2] = float(wz)
        elif now - last_cmd_time > args.cmd_vel_timeout:
            action = zero_action
        else:
            action = zero_action

        env.step(action)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
