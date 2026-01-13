"""Custom action terms for Strafer mecanum wheel robot.

Includes sim-to-real transfer features:
- Motor dynamics (first-order response)
- Command latency (delay buffer)
- Velocity smoothing
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MecanumWheelAction(ActionTerm):
    """Action term for controlling mecanum wheel velocities.
    
    This action term converts velocity commands to individual wheel velocities
    for a 4-wheel mecanum drive robot (GoBilda Strafer chassis).
    
    The action space is [vx, vy, omega] representing:
    - vx: Forward/backward velocity (normalized -1 to 1)
    - vy: Left/right (strafe) velocity (normalized -1 to 1)  
    - omega: Rotational velocity (normalized -1 to 1)
    
    These are converted to individual wheel velocities using mecanum kinematics.
    
    Hardware specs (GoBilda 5203 Series Yellow Jacket 19.2:1):
    - Motor output RPM: 312 RPM (after 19.2:1 planetary gearbox)
    - Miter gears: 1:1 ratio (90° direction change)
    - Wheel diameter: 96mm (mecanum wheel)
    - Max wheel angular velocity: 32.67 rad/s
    - Max linear velocity: ~1.57 m/s
    
    Wheel layout (looking from above, robot facing +X in world frame):
    
        +X (Forward)
           ^
           |
       FL  |  FR     wheel_1    wheel_4
           [==]
       RL  |  RR     wheel_2    wheel_3
           |
        -X (Back)
       
       <-- -Y     +Y -->
    
    USD wheel numbering (counter-clockwise from front-left):
    - wheel_1 = Front-Left  (X=+20.7, Y=-16.8)
    - wheel_2 = Rear-Left   (X=-20.7, Y=-16.8)  
    - wheel_3 = Rear-Right  (X=-20.7, Y=+16.8)
    - wheel_4 = Front-Right (X=+20.7, Y=+16.8)
    
    Note: Left wheels (1,2) have opposite joint rotation direction from right wheels (3,4)
    due to joint frame orientation in the USD.
    """

    cfg: MecanumWheelActionCfg
    _asset: object  # Articulation

    def __init__(self, cfg: MecanumWheelActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # Get the robot articulation
        self._asset = env.scene[cfg.asset_name]
        
        # Find wheel joint indices - note: find_joints returns in USD hierarchy order
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        
        # Create reorder mapping: for each USD joint position, which kinematic index to use?
        # Kinematic matrix rows: [wheel_1, wheel_2, wheel_3, wheel_4]
        # USD order example: ['wheel_1_drive', 'wheel_4_drive', 'wheel_2_drive', 'wheel_3_drive']
        # We need: USD[0]=kinematic[0], USD[1]=kinematic[3], USD[2]=kinematic[1], USD[3]=kinematic[2]
        
        kinematic_names = ["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"]
        
        # For each joint in USD order, find which kinematic index provides its velocity
        self._joint_reorder = []
        for usd_name in self._joint_names:
            for kin_idx, kin_name in enumerate(kinematic_names):
                if kin_name in usd_name or usd_name == kin_name:
                    self._joint_reorder.append(kin_idx)
                    break
        self._joint_reorder = torch.tensor(self._joint_reorder, device=env.device, dtype=torch.long)
        
        # ============================================================
        # Robot physical parameters (GoBilda Strafer with 5203 motors)
        # ============================================================
        
        # Wheel geometry
        self._wheel_radius = cfg.wheel_radius  # meters (96mm diameter wheel)
        
        # Robot frame dimensions (center-to-center of wheels)
        self._wheel_base = cfg.wheel_base      # front-to-rear axle distance
        self._track_width = cfg.track_width    # left-to-right axle distance
        
        # Motor limits (5203 Yellow Jacket 19.2:1 + 1:1 miter gears)
        # 312 RPM = 312 * 2π / 60 = 32.67 rad/s max wheel angular velocity
        self._max_wheel_angular_vel = cfg.max_wheel_angular_vel  # rad/s
        
        # Derived velocity limits
        # Max linear velocity = wheel_radius * max_angular_vel
        self._max_linear_vel = self._wheel_radius * self._max_wheel_angular_vel  # m/s
        
        # Max rotational velocity of robot body
        # For pure rotation: omega_robot = v_wheel / (L/2 + W/2)
        L_half = self._wheel_base / 2
        W_half = self._track_width / 2
        self._max_angular_vel = self._max_linear_vel / (L_half + W_half)  # rad/s
        
        # ============================================================
        # Mecanum kinematic matrix
        # ============================================================
        # Maps body velocities [vx, vy, omega] to wheel angular velocities
        # 
        # Standard mecanum equations (X-configuration rollers):
        #   ω_fl = (1/r) * (vx - vy - (L/2 + W/2)*ω_body)
        #   ω_fr = (1/r) * (vx + vy + (L/2 + W/2)*ω_body)
        #   ω_rl = (1/r) * (vx + vy - (L/2 + W/2)*ω_body)
        #   ω_rr = (1/r) * (vx - vy + (L/2 + W/2)*ω_body)
        #
        # Matrix form: ω_wheels = K * [vx, vy, ω_body]^T
        #
        # USD wheel layout (based on actual positions):
        #   wheel_1 = FRONT-LEFT  (X=+20.7, Y=-16.8)
        #   wheel_2 = REAR-LEFT   (X=-20.7, Y=-16.8)
        #   wheel_3 = REAR-RIGHT  (X=-20.7, Y=+16.8)
        #   wheel_4 = FRONT-RIGHT (X=+20.7, Y=+16.8)
        #
        # Counter-clockwise numbering: FL(1) → RL(2) → RR(3) → FR(4)
        #
        # Joint rotation sign convention:
        #   Left wheels (1,2): positive velocity = wheel spins one way
        #   Right wheels (3,4): positive velocity = wheel spins opposite way
        #   For forward motion, left wheels need NEGATIVE velocity,
        #   right wheels need POSITIVE velocity (based on joint frame orientation)
        
        k = L_half + W_half  # Combined lever arm for rotation
        r = self._wheel_radius
        
        # Kinematic matrix (maps body velocity to wheel angular velocity)
        # Rows: [wheel_1=FL, wheel_2=RL, wheel_3=RR, wheel_4=FR]
        # 
        # Robot body frame convention (from USD):
        #   - Robot "forward" is aligned with -Y axis
        #   - Robot "left" is aligned with +X axis  
        #   - Robot "up" is aligned with +Z axis
        #
        # Action convention:
        #   - action[0] = vx (forward velocity in robot frame)
        #   - action[1] = vy (strafe left velocity in robot frame)
        #   - action[2] = omega (CCW rotation)
        #
        # Since forward is -Y in the USD, positive vx should produce -Y motion.
        # We negate the first column to achieve this.
        self._kinematic_matrix = torch.tensor([
            [-1/r, -1/r,  k/r],  # wheel_1 = Front-Left  
            [ 1/r, -1/r,  k/r],  # wheel_2 = Rear-Left   
            [ 1/r,  1/r,  k/r],  # wheel_3 = Rear-Right  
            [-1/r,  1/r,  k/r],  # wheel_4 = Front-Right 
        ], dtype=torch.float32, device=env.device)
        
        # Velocity scaling for normalized actions [-1, 1]
        # This converts normalized actions to physical velocities
        self._velocity_scale = torch.tensor([
            self._max_linear_vel,   # vx scale (m/s)
            self._max_linear_vel,   # vy scale (m/s)
            self._max_angular_vel,  # omega scale (rad/s)
        ], dtype=torch.float32, device=env.device)
        
        # Initialize action buffers
        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 4, device=env.device)
        
        # ============================================================
        # Sim-to-Real: Motor dynamics and command latency
        # ============================================================
        
        # Motor dynamics (first-order low-pass filter)
        self._enable_motor_dynamics = cfg.enable_motor_dynamics
        if self._enable_motor_dynamics:
            # Low-pass filter coefficient: alpha = dt / (tau + dt)
            # where tau = motor time constant, dt = physics step time
            physics_dt = env.physics_dt
            self._motor_alpha = physics_dt / (cfg.motor_time_constant + physics_dt)
            self._smoothed_wheel_vels = torch.zeros(env.num_envs, 4, device=env.device)
        
        # Command delay buffer
        self._enable_command_delay = cfg.max_delay_steps > 0
        if self._enable_command_delay:
            self._action_delay_buffer = DelayBuffer(
                history_length=cfg.max_delay_steps,
                batch_size=env.num_envs,
                device=env.device,
            )
            # Set random delay per environment
            if cfg.min_delay_steps < cfg.max_delay_steps:
                time_lags = torch.randint(
                    low=cfg.min_delay_steps,
                    high=cfg.max_delay_steps + 1,
                    size=(env.num_envs,),
                    dtype=torch.int,
                    device=env.device,
                )
            else:
                time_lags = torch.full(
                    (env.num_envs,), cfg.max_delay_steps, dtype=torch.int, device=env.device
                )
            self._action_delay_buffer.set_time_lag(time_lags)
        
        # Slew rate limiting (max acceleration)
        self._enable_slew_rate = cfg.max_acceleration_rad_s2 < float('inf')
        if self._enable_slew_rate:
            self._max_delta_per_step = cfg.max_acceleration_rad_s2 * env.physics_dt
            self._prev_wheel_vels = torch.zeros(env.num_envs, 4, device=env.device)
        
        # Log configuration
        print(f"[MecanumWheelAction] Initialized with:")
        print(f"  Wheel radius: {self._wheel_radius*1000:.1f} mm")
        print(f"  Wheel base: {self._wheel_base*1000:.1f} mm")
        print(f"  Track width: {self._track_width*1000:.1f} mm")
        print(f"  Max wheel angular vel: {self._max_wheel_angular_vel:.2f} rad/s ({cfg.motor_rpm:.0f} RPM)")
        print(f"  Max linear vel: {self._max_linear_vel:.2f} m/s")
        print(f"  Max angular vel: {self._max_angular_vel:.2f} rad/s")
        print(f"  Joint names (USD order): {self._joint_names}")
        print(f"  Joint reorder (kinematic->USD): {self._joint_reorder.tolist()}")
        print(f"  Motor dynamics: {self._enable_motor_dynamics} (tau={cfg.motor_time_constant}s)")
        print(f"  Command delay: {self._enable_command_delay} (steps={cfg.min_delay_steps}-{cfg.max_delay_steps})")
        print(f"  Slew rate limit: {self._enable_slew_rate} (max_accel={cfg.max_acceleration_rad_s2} rad/s²)")

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action state for specified environments."""
        if env_ids is None:
            env_ids = slice(None)
        
        # Reset motor dynamics filter
        if self._enable_motor_dynamics:
            self._smoothed_wheel_vels[env_ids] = 0.0
        
        # Reset delay buffer
        if self._enable_command_delay:
            self._action_delay_buffer.reset(env_ids)
        
        # Reset slew rate state
        if self._enable_slew_rate:
            self._prev_wheel_vels[env_ids] = 0.0

    @property
    def action_dim(self) -> int:
        """Dimension of the action space: [vx, vy, omega]."""
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions before processing (normalized -1 to 1)."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed wheel angular velocity commands (rad/s)."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Convert normalized [vx, vy, omega] actions to wheel angular velocities.
        
        Args:
            actions: Tensor of shape (num_envs, 3) with values in [-1, 1]
                     representing [vx, vy, omega] normalized commands.
        """
        # Store raw actions
        self._raw_actions = actions.clone()
        
        # Clamp to valid range
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        
        # Scale normalized actions to physical velocities
        # [vx, vy, omega] in [m/s, m/s, rad/s]
        body_velocities = clamped_actions * self._velocity_scale
        
        # Convert to wheel angular velocities using mecanum kinematics
        # wheel_vels_kinematic is in kinematic order: [wheel_1, wheel_2, wheel_3, wheel_4]
        wheel_vels_kinematic = torch.matmul(body_velocities, self._kinematic_matrix.T)
        
        # Clamp wheel velocities to motor limits
        wheel_vels = torch.clamp(wheel_vels_kinematic, 
                                  -self._max_wheel_angular_vel, 
                                  self._max_wheel_angular_vel)
        
        # Reorder from kinematic order to USD joint order
        wheel_vels = wheel_vels[:, self._joint_reorder]
        
        # ============================================================
        # Apply sim-to-real processing
        # ============================================================
        
        # 1. Command delay (simulate network/driver latency)
        if self._enable_command_delay:
            wheel_vels = self._action_delay_buffer.compute(wheel_vels)
        
        # 2. Slew rate limiting (max acceleration)
        if self._enable_slew_rate:
            delta = wheel_vels - self._prev_wheel_vels
            delta = torch.clamp(delta, -self._max_delta_per_step, self._max_delta_per_step)
            wheel_vels = self._prev_wheel_vels + delta
            self._prev_wheel_vels = wheel_vels.clone()
        
        # 3. Motor dynamics (first-order low-pass filter)
        if self._enable_motor_dynamics:
            self._smoothed_wheel_vels = (
                self._motor_alpha * wheel_vels + 
                (1 - self._motor_alpha) * self._smoothed_wheel_vels
            )
            wheel_vels = self._smoothed_wheel_vels
        
        self._processed_actions = wheel_vels

    def apply_actions(self):
        """Apply wheel velocity targets to the robot."""
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class MecanumWheelActionCfg(ActionTermCfg):
    """Configuration for mecanum wheel action term.
    
    Default values are for GoBilda Strafer chassis with 5203 Yellow Jacket motors.
    Includes sim-to-real transfer parameters for motor dynamics and latency.
    """

    class_type: type = MecanumWheelAction

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    joint_names: list[str] = MISSING
    """Regex pattern(s) for wheel joint names. Order: [FL, FR, RL, RR]."""

    # ============================================================
    # Physical robot parameters
    # ============================================================
    
    # Wheel geometry
    wheel_radius: float = 0.048
    """Wheel radius in meters. Default: 48mm (96mm diameter mecanum wheel)."""
    
    # Frame dimensions (center-to-center of wheel axles)
    wheel_base: float = 0.304
    """Distance between front and rear axles in meters. Default: 304mm."""
    
    track_width: float = 0.304
    """Distance between left and right wheels in meters. Default: 304mm."""
    
    # Motor specifications (5203 Yellow Jacket 19.2:1 ratio)
    motor_rpm: float = 312.0
    """Motor output RPM after gearbox. Default: 312 RPM (5203 @ 19.2:1)."""
    
    max_wheel_angular_vel: float = 32.67
    """Maximum wheel angular velocity in rad/s. Default: 312 RPM = 32.67 rad/s."""
    
    # ============================================================
    # Sim-to-Real: Motor dynamics
    # ============================================================
    
    enable_motor_dynamics: bool = False
    """Enable first-order motor response model. Default: False (ideal response)."""
    
    motor_time_constant: float = 0.05
    """Motor time constant in seconds. Controls response speed.
    Real GoBilda 5203 under load: ~50ms. Default: 0.05s."""
    
    # ============================================================
    # Sim-to-Real: Command latency
    # ============================================================
    
    min_delay_steps: int = 0
    """Minimum command delay in physics steps. Default: 0."""
    
    max_delay_steps: int = 0
    """Maximum command delay in physics steps. Randomized per reset.
    Default: 0 (no delay)."""
    
    # ============================================================
    # Sim-to-Real: Slew rate limiting  
    # ============================================================
    
    max_acceleration_rad_s2: float = float('inf')
    """Maximum wheel angular acceleration in rad/s². Prevents instant velocity changes.
    Default: inf (no limit)."""
