"""Custom event functions for Strafer navigation task.

Events are used for:
- Resetting robot state at episode start
- Domain randomization during training
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_robot_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict,
) -> None:
    """Reset robot to random initial pose.
    
    Args:
        env: The environment instance.
        env_ids: Indices of environments to reset.
        pose_range: Dictionary with keys 'x', 'y', 'yaw' specifying
                   (min, max) ranges for each component.
    """
    robot = env.scene["robot"]
    num_resets = len(env_ids)
    device = env.device
    
    # Sample random positions
    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    yaw_range = pose_range.get("yaw", (0.0, 0.0))
    
    x = torch.rand(num_resets, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y = torch.rand(num_resets, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    z = torch.full((num_resets,), 0.1, device=device)  # Fixed height above ground
    
    yaw = torch.rand(num_resets, device=device) * (yaw_range[1] - yaw_range[0]) + yaw_range[0]
    
    # Convert yaw to quaternion (w, x, y, z)
    quat = torch.zeros(num_resets, 4, device=device)
    quat[:, 0] = torch.cos(yaw / 2)  # w
    quat[:, 3] = torch.sin(yaw / 2)  # z
    
    # Set robot state
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0] = x  # pos_x
    root_state[:, 1] = y  # pos_y
    root_state[:, 2] = z  # pos_z
    root_state[:, 3:7] = quat  # orientation
    root_state[:, 7:] = 0.0  # zero velocity
    
    robot.write_root_state_to_sim(root_state, env_ids)


def randomize_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    friction_range: tuple[float, float],
) -> None:
    """Randomize wheel-ground friction coefficient for domain randomization.

    Modifies the physics material properties of the robot's contact surfaces
    (wheels) to simulate varying floor conditions. This helps with sim-to-real
    transfer by training policies robust to different friction levels.

    The friction value is randomized per-environment at reset time, simulating
    conditions like carpet vs tile vs polished concrete.

    Args:
        env: The environment instance.
        env_ids: Indices of environments to randomize.
        friction_range: (min, max) friction coefficient range.
            Typical values: (0.3, 1.2) for indoor surfaces.
    """
    if len(env_ids) == 0:
        return

    robot = env.scene["robot"]
    device = env.device

    # Get current material properties from PhysX view
    # Shape: (num_envs, max_num_shapes, 3) where 3 = [static, dynamic, restitution]
    materials = robot.root_physx_view.get_material_properties()

    # Sample random friction values for each environment being reset
    num_resets = len(env_ids)
    friction_values = (
        torch.rand(num_resets, device=device)
        * (friction_range[1] - friction_range[0])
        + friction_range[0]
    )

    # Apply friction to all shapes in the reset environments
    # Expand friction to match shape dimension: (num_resets,) -> (num_resets, num_shapes)
    num_shapes = materials.shape[1]
    friction_expanded = friction_values.unsqueeze(1).expand(-1, num_shapes)

    # Update static friction (index 0) and dynamic friction (index 1)
    # Dynamic friction should be <= static friction for physical consistency
    materials[env_ids, :, 0] = friction_expanded  # Static friction
    materials[env_ids, :, 1] = friction_expanded * 0.9  # Dynamic friction (slightly lower)
    # Keep restitution (index 2) unchanged

    # Apply the modified materials back to simulation
    robot.root_physx_view.set_material_properties(materials, env_ids)
