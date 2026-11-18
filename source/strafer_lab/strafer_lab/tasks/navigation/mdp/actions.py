"""Custom action terms for Strafer mecanum wheel robot.

Includes sim-to-real transfer features:
- Motor dynamics (first-order response)
- Command latency (delay buffer, and a sub-step part in apply_actions)
- Velocity smoothing
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.buffers import DelayBuffer
from strafer_shared.mecanum_kinematics import (
    KINEMATIC_MATRIX,
    l1_clamp_twist_batched,
)

from .hold_process import HoldProcess

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
       FL  |  FR     wheel_1    wheel_2
           [==]
       RL  |  RR     wheel_3    wheel_4
           |
        -X (Back)
       
       <-- -Y (Right)     +Y (Left) -->
    
    Wheel numbering (front axle left->right, rear axle left->right):
    - wheel_1 = Front-Left  (FL)
    - wheel_2 = Front-Right (FR)
    - wheel_3 = Rear-Left   (RL)
    - wheel_4 = Rear-Right  (RR)
    
    Note: This assumes wheel joint axes are consistent across all wheels.
    If a wheel joint axis is flipped in USD, apply a per-wheel sign correction.
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
        # Kinematic matrix rows: [wheel_1=FL, wheel_2=FR, wheel_3=RL, wheel_4=RR]
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

        # Per-wheel sign correction in kinematic order (FL, FR, RL, RR)
        self._wheel_axis_signs = torch.tensor(
            cfg.wheel_axis_signs, device=env.device, dtype=torch.float32
        )
        if self._wheel_axis_signs.numel() != 4:
            raise ValueError(
                f"wheel_axis_signs must have 4 entries (got {self._wheel_axis_signs.numel()})"
            )
        
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
        # The per-wheel limit on the targets; it moves apart from the action scale and the caps
        self._wheel_speed_limit = (
            cfg.max_wheel_angular_vel if cfg.wheel_speed_limit is None else cfg.wheel_speed_limit
        )
        
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
        # Wheel numbering (front axle left->right, rear axle left->right):
        #   wheel_1 = FRONT-LEFT (FL)
        #   wheel_2 = FRONT-RIGHT (FR)
        #   wheel_3 = REAR-LEFT (RL)
        #   wheel_4 = REAR-RIGHT (RR)
        #
        # If any wheel joint axis is flipped in USD, apply a per-wheel sign correction
        # rather than changing the kinematic matrix form.
        
        # Kinematic matrix from strafer_shared (single source of truth)
        # Converts body velocities [vx, vy, omega] -> wheel angular velocities [FL, FR, RL, RR]
        # wheel_axis_signs are applied separately (sim-only USD joint axis correction)
        self._kinematic_matrix = torch.from_numpy(KINEMATIC_MATRIX).to(
            dtype=torch.float32, device=env.device
        )
        
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
            # alpha = 1 - exp(-dt / tau) per env, dt the control step: the filter advances once per
            # process_actions call and a step reaches 63.2 % of its target after tau at any step
            self._motor_dt = env.step_dt
            band = cfg.motor_time_constant_range
            self._tau_range = tuple(band) if band is not None and band[0] < band[1] else None
            self._motor_alpha = torch.empty(env.num_envs, 1, device=env.device)
            self._motor_keep = torch.empty(env.num_envs, 1, device=env.device)
            self._set_motor_tau(torch.full((env.num_envs,), cfg.motor_time_constant, dtype=torch.float64))
            self._smoothed_wheel_vels = torch.zeros(env.num_envs, 4, device=env.device)
        
        # Command delay, in seconds: whole env steps in the delay buffer, the part under one env step held at the
        # start of each step's physics sub-steps. Converted here, so it means the same at any decimation.
        self._decimation = max(1, int(round(env.step_dt / env.physics_dt)))
        self._delay_steps_range = _whole_units(cfg.delay_range_s, env.step_dt, "delay_range_s", "env step",
                                               "substep_delay_range_s")
        self._substep_ticks_range = _whole_units(cfg.substep_delay_range_s, env.physics_dt, "substep_delay_range_s",
                                                 "physics tick", None)
        self._redraw_delay = cfg.delay_redraw_on_reset
        delay_lo, delay_hi = self._delay_steps_range
        tick_hi = self._substep_ticks_range[1]
        history = delay_hi + tick_hi // self._decimation
        self._enable_command_delay = history > 0
        self._enable_substep_delay = tick_hi > 0
        self._delay_env_steps = torch.full((env.num_envs,), delay_hi, dtype=torch.int, device=env.device)
        self._substep_ticks = torch.zeros(env.num_envs, dtype=torch.int, device=env.device)
        if self._enable_command_delay:
            self._action_delay_buffer = DelayBuffer(
                history_length=history,
                batch_size=env.num_envs,
                device=env.device,
            )
            # Set random delay per environment
            if delay_lo < delay_hi:
                self._delay_env_steps = torch.randint(
                    low=delay_lo,
                    high=delay_hi + 1,
                    size=(env.num_envs,),
                    dtype=torch.int,
                    device=env.device,
                )

        # Command holds: no new command this control step, so the chassis keeps
        # executing the last one it received. Distinct from the delay buffer
        # above — a delay shifts a command in time, a hold repeats it.
        self._hold = HoldProcess(
            env.num_envs,
            env.device,
            hold_fraction_range=cfg.hold_fraction_range,
            hold_run_range=tuple(round(s / env.step_dt, 9) for s in cfg.hold_run_range_s),
        )
        if self._hold.enabled:
            self._held_command = torch.zeros(env.num_envs, 4, device=env.device)

        # Slew rate limiting (max acceleration) — per-env randomization. The limit
        # applies once per process_actions call, so its step is accel x control step.
        self._enable_slew_rate = cfg.max_acceleration_rad_s2 < float('inf')
        if self._enable_slew_rate:
            self._accel_range = cfg.max_acceleration_range
            self._accel_nominal = cfg.max_acceleration_rad_s2
            self._control_dt = env.step_dt
            self._num_envs = env.num_envs
            lo, hi = self._accel_range
            if lo < hi:
                accel_per_env = lo + (hi - lo) * torch.rand(env.num_envs, device=env.device)
            else:
                accel_per_env = torch.full((env.num_envs,), self._accel_nominal, device=env.device)
            # Shape (num_envs, 1) for broadcasting with (num_envs, 4) wheel vels
            self._max_delta_per_step = (accel_per_env * env.step_dt).unsqueeze(-1)
            self._prev_wheel_vels = torch.zeros(env.num_envs, 4, device=env.device)
            # Speed fall-off of the limit: a(w) = a0 (1 - |w| / w_nl); inf leaves it flat
            self._accel_falloff_speed = cfg.accel_falloff_speed

        # Per-env time-constant draw, after the draws above so a band leaves their order alone
        if self._enable_motor_dynamics and self._tau_range is not None:
            self._set_motor_tau(self._draw_tau(env.num_envs, env.device))
        
        # ============================================================
        # Command envelope (per-env scale on both caps, clamp only)
        # ============================================================
        # One scale s per env, drawn at every reset, multiplies the L1 linear
        # cap and the angular cap the clamp applies. _velocity_scale stays at
        # the full envelope, as the deploy node scales the action by the full
        # envelope before it caps.
        self._speed_scale_range = cfg.speed_scale_range
        if self._speed_scale_range is not None:
            lo, hi = (float(x) for x in self._speed_scale_range)
            if not 0.0 < lo <= hi <= 1.0:
                raise ValueError(f"speed_scale_range must satisfy 0 < low <= high <= 1, got {self._speed_scale_range}")
            self._speed_scale = torch.ones(env.num_envs, device=env.device)
            self._draw_speed_scale(None)

        # The delay's sub-step part, drawn after every draw above so a band leaves their order alone
        if self._enable_substep_delay:
            self._draw_substep(None)
            self._applied = torch.zeros(env.num_envs, 4, device=env.device)
            self._prev_applied = torch.zeros(env.num_envs, 4, device=env.device)
            self._substep = 0
        self._set_delay_lag(None)

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
        print(f"  Wheel axis signs (FL,FR,RL,RR): {self._wheel_axis_signs.tolist()}")
        print(f"  Motor dynamics: {self._enable_motor_dynamics} (tau={cfg.motor_time_constant}s)")
        print(f"  Command delay: {self._enable_command_delay} (band={cfg.delay_range_s} s, sub-step band="
              f"{cfg.substep_delay_range_s} s, redrawn at reset: {self._redraw_delay})")
        print(f"  Command hold: {self._hold.enabled} (fraction={cfg.hold_fraction_range}, run={cfg.hold_run_range_s} s)")
        print(f"  Slew rate limit: {self._enable_slew_rate} (max_accel={cfg.max_acceleration_rad_s2} rad/s², range={cfg.max_acceleration_range})")
        print(f"  Command envelope: {self._speed_scale_range is not None} (scale range={self._speed_scale_range})")

    def _draw_speed_scale(self, env_ids: torch.Tensor | None) -> None:
        """Draw each env's envelope scale uniformly in speed_scale_range."""
        lo, hi = (float(x) for x in self._speed_scale_range)
        n = self._speed_scale.shape[0] if env_ids is None else env_ids.shape[0]
        draw = lo + (hi - lo) * torch.rand(n, device=self._speed_scale.device)
        if env_ids is None:
            self._speed_scale[:] = draw
        else:
            self._speed_scale[env_ids] = draw

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action state for specified environments."""
        all_envs = env_ids is None
        if all_envs:
            env_ids = slice(None)

        # Redraw the command envelope
        if self._speed_scale_range is not None:
            self._draw_speed_scale(None if all_envs else env_ids)

        # Reset motor dynamics filter
        if self._enable_motor_dynamics:
            self._smoothed_wheel_vels[env_ids] = 0.0

        # Reset delay buffer (expects Tensor or None, not slice)
        if self._enable_command_delay:
            self._action_delay_buffer.reset(None if all_envs else env_ids)

        # Reset command-hold state and redraw its per-env parameters
        if self._hold.enabled:
            self._held_command[env_ids] = 0.0
            self._hold.reset(None if all_envs else env_ids)

        # Reset slew rate state and re-randomize acceleration per env
        if self._enable_slew_rate:
            self._prev_wheel_vels[env_ids] = 0.0
            lo, hi = self._accel_range
            if lo < hi:
                if all_envs:
                    n = self._num_envs
                else:
                    n = env_ids.shape[0]
                new_accel = lo + (hi - lo) * torch.rand(n, device=self._prev_wheel_vels.device)
                self._max_delta_per_step[env_ids] = (new_accel * self._control_dt).unsqueeze(-1)

        # Redraw the per-env time constant
        if self._enable_motor_dynamics and self._tau_range is not None:
            n = self._motor_alpha.shape[0] if all_envs else env_ids.shape[0]
            self._set_motor_tau(self._draw_tau(n, self._motor_alpha.device), None if all_envs else env_ids)

        # The delay's redraw and its sub-step part, the reset's last draws
        ids = None if all_envs else env_ids
        redraw = self._redraw_delay and self._enable_command_delay and self._delay_steps_range[0] < self._delay_steps_range[1]
        if redraw:
            self._draw_delay(ids)
        if self._enable_substep_delay:
            self._draw_substep(ids)
            self._applied[env_ids] = 0.0
        if redraw or self._enable_substep_delay:
            self._set_delay_lag(ids)

    def _draw_delay(self, env_ids: torch.Tensor | None) -> None:
        lo, hi = self._delay_steps_range
        n = self._delay_env_steps.shape[0] if env_ids is None else env_ids.shape[0]
        draw = torch.randint(lo, hi + 1, (n,), dtype=torch.int, device=self._delay_env_steps.device)
        self._delay_env_steps[slice(None) if env_ids is None else env_ids] = draw

    def _draw_substep(self, env_ids: torch.Tensor | None) -> None:
        lo, hi = self._substep_ticks_range
        n = self._substep_ticks.shape[0] if env_ids is None else env_ids.shape[0]
        draw = torch.randint(lo, hi + 1, (n,), dtype=torch.int, device=self._substep_ticks.device)
        self._substep_ticks[slice(None) if env_ids is None else env_ids] = draw

    def _set_delay_lag(self, env_ids: torch.Tensor | None) -> None:
        """The buffer's lag in whole env steps: the drawn delay plus the sub-step part's whole steps (at decimation 1
        every tick is a step, so the sub-step part carries into the buffer and the delay keeps its seconds)."""
        if not self._enable_command_delay:
            return
        lag = self._delay_env_steps + self._substep_ticks // self._decimation
        if env_ids is None:
            self._action_delay_buffer.set_time_lag(lag)
        else:
            self._action_delay_buffer.set_time_lag(lag[env_ids], env_ids)

    def _draw_tau(self, n: int, device) -> torch.Tensor:
        lo, hi = self._tau_range
        return lo + (hi - lo) * torch.rand(n, device=device).double()

    def _set_motor_tau(self, tau: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        """Set alpha = 1 - exp(-dt / tau) and its complement for the given envs, computed in double."""
        tau = tau.double().cpu()
        alpha = -torch.expm1(-self._motor_dt / tau)
        rows = slice(None) if env_ids is None else env_ids
        self._motor_alpha[rows] = alpha.to(self._motor_alpha.dtype).to(self._motor_alpha.device).unsqueeze(-1)
        self._motor_keep[rows] = (1.0 - alpha).to(self._motor_keep.dtype).to(self._motor_keep.device).unsqueeze(-1)

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

    @property
    def speed_scale(self) -> torch.Tensor:
        """Each env's command-envelope scale s, shape (num_envs,). Exists only when the cfg sets
        ``speed_scale_range``; reading it otherwise raises AttributeError."""
        return self._speed_scale

    def process_actions(self, actions: torch.Tensor):
        """Convert normalized [vx, vy, omega] actions to wheel angular velocities.
        
        Args:
            actions: Tensor of shape (num_envs, 3) with values in [-1, 1]
                     representing [vx, vy, omega] normalized commands.
        """
        # Store raw actions
        self._raw_actions = actions.clone()

        # The node's stale-source stop: while a sensor is stale past its timeout the commanded twist is zero, and the
        # wheels come to rest through the plant below as the robot's do
        outage = getattr(self._env, "_sensor_outages", None)
        if outage is not None:
            outage.advance()
            actions = torch.where(outage.stopped.unsqueeze(-1), torch.zeros_like(actions), actions)

        # Clamp to valid range
        clamped_actions = torch.clamp(actions, -1.0, 1.0)

        # Scale normalized actions to physical velocities
        # [vx, vy, omega] in [m/s, m/s, rad/s]
        body_velocities = clamped_actions * self._velocity_scale

        # L1 clamp on (vx, vy) to match the deployment-time command
        # shaping: the chassis cannot deliver max-forward + max-strafe
        # simultaneously (per-wheel motor cap), so the Jetson scales
        # body-frame (vx, vy) jointly before publishing /cmd_vel. Without
        # this clamp, sim's per-wheel torch.clamp below saturates each
        # wheel independently and distorts the commanded heading on
        # diagonal motions — a contract gap the Jetson side closes via
        # strafer_inference.obs_pipeline.l1_clamp_velocity.
        if self._speed_scale_range is None:
            cap_linear, cap_angular = self._max_linear_vel, self._max_angular_vel
        else:
            cap_linear = self._speed_scale * self._max_linear_vel
            cap_angular = self._speed_scale * self._max_angular_vel
        body_velocities = l1_clamp_twist_batched(
            body_velocities,
            vel_cap_linear_m_s=cap_linear,
            vel_cap_angular_rad_s=cap_angular,
        )

        # Convert to wheel angular velocities using mecanum kinematics
        # wheel_vels_kinematic is in kinematic order: [wheel_1, wheel_2, wheel_3, wheel_4]
        wheel_vels_kinematic = torch.matmul(body_velocities, self._kinematic_matrix.T)
        # Apply per-wheel sign correction (joint axis direction)
        wheel_vels_kinematic = wheel_vels_kinematic * self._wheel_axis_signs
        
        # Clamp wheel velocities to the per-wheel limit
        wheel_vels = torch.clamp(wheel_vels_kinematic, -self._wheel_speed_limit, self._wheel_speed_limit)
        
        # Reorder from kinematic order to USD joint order
        wheel_vels = wheel_vels[:, self._joint_reorder]
        
        # ============================================================
        # Apply sim-to-real processing
        # ============================================================
        
        # 1. Command delay (simulate network/driver latency)
        if self._enable_command_delay:
            wheel_vels = self._action_delay_buffer.compute(wheel_vels)

        # 1b. Command hold (no new command arrived this step)
        if self._hold.enabled:
            held = self._hold.step()
            wheel_vels = torch.where(held.unsqueeze(-1), self._held_command, wheel_vels)
            self._held_command = wheel_vels.clone()

        # 2. Slew rate limiting (max acceleration)
        if self._enable_slew_rate:
            delta = wheel_vels - self._prev_wheel_vels
            max_delta = self._max_delta_per_step
            if self._accel_falloff_speed < float('inf'):
                speed_share = self._prev_wheel_vels.abs() / self._accel_falloff_speed
                max_delta = max_delta * torch.clamp(1.0 - speed_share, min=0.0)
            delta = torch.clamp(delta, -max_delta, max_delta)
            wheel_vels = self._prev_wheel_vels + delta
            self._prev_wheel_vels = wheel_vels.clone()
        
        # 3. Motor dynamics (first-order low-pass filter)
        if self._enable_motor_dynamics:
            self._smoothed_wheel_vels = (
                self._motor_alpha * wheel_vels +
                self._motor_keep * self._smoothed_wheel_vels
            )
            wheel_vels = self._smoothed_wheel_vels
        
        self._processed_actions = wheel_vels
        if self._enable_substep_delay:
            self._prev_applied = self._applied.clone()
            self._substep = 0

    def apply_actions(self):
        """Apply wheel velocity targets to the robot. With a sub-step delay, an env keeps the previous step's target for
        the first k of this step's physics sub-steps, k its drawn ticks less the whole steps the buffer carries."""
        target = self._processed_actions
        if self._enable_substep_delay:
            held = (self._substep < self._substep_ticks % self._decimation).unsqueeze(-1)
            target = torch.where(held, self._prev_applied, target)
            self._applied = target
            self._substep += 1
        self._asset.set_joint_velocity_target(target, joint_ids=self._joint_ids)


def _whole_units(band_s, unit_s: float, name: str, unit: str, remainder_field: str | None) -> tuple[int, int]:
    """A band in seconds as whole units (env steps or physics ticks), ends within 1e-6 of a unit."""
    lo, hi = sorted(float(x) for x in band_s)
    if lo < 0.0:
        raise ValueError(f"{name} {band_s} is negative")
    units = []
    for end in (lo, hi):
        n = end / unit_s
        if abs(n - round(n)) > 1e-6:
            where = f"; the part under one {unit} goes in {remainder_field}" if remainder_field else ""
            raise ValueError(f"{name} end {end} s is not a whole {unit} ({unit_s} s){where}")
        units.append(int(round(n)))
    return units[0], units[1]


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

    wheel_axis_signs: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    """Per-wheel sign correction for joint axis direction in kinematic order [FL, FR, RL, RR]."""

    # ============================================================
    # Physical robot parameters
    # ============================================================
    
    # Wheel geometry
    wheel_radius: float = 0.048
    """Wheel radius in meters. Default: 48mm (96mm diameter mecanum wheel)."""
    
    # Frame dimensions (center-to-center of wheel axles)
    wheel_base: float = 0.336
    """Distance between front and rear axles in meters. Default: 336mm."""
    
    track_width: float = 0.4132
    """Distance between left and right wheels in meters. Default: 413.2mm."""
    
    # Motor specifications (5203 Yellow Jacket 19.2:1 ratio)
    motor_rpm: float = 312.0
    """Motor output RPM after gearbox. Default: 312 RPM (5203 @ 19.2:1)."""
    
    max_wheel_angular_vel: float = 32.67
    """Maximum wheel angular velocity in rad/s. Default: 312 RPM = 32.67 rad/s."""

    wheel_speed_limit: float | None = None
    """Per-wheel limit in rad/s on the wheel targets after the kinematics. None uses
    ``max_wheel_angular_vel``, which also sets the action scale and the twist caps;
    this field moves the limit alone."""
    
    # ============================================================
    # Sim-to-Real: Motor dynamics
    # ============================================================
    
    enable_motor_dynamics: bool = False
    """Enable first-order motor response model. Default: False (ideal response)."""
    
    motor_time_constant: float = 0.2
    """Time constant in seconds of the first-order wheel lag: a step reaches 63.2 % of
    its target after it, at any control step (alpha = 1 - exp(-dt / tau) per control
    step, dt the control step). Default: 0.2 s."""

    motor_time_constant_range: tuple[float, float] | None = None
    """Per-env band [min, max] for the time constant, drawn at each reset. None, or
    min == max, fixes it at ``motor_time_constant``."""
    
    # ============================================================
    # Sim-to-Real: Command latency
    # ============================================================
    
    delay_range_s: tuple[float, float] = (0.0, 0.0)
    """Command delay band [min, max] in seconds. Converted to whole env steps when the
    term is built (each end must be one, within 1e-6 of a step) and drawn per env
    uniformly over them; the part under one env step is ``substep_delay_range_s``.
    In seconds, so the delay means the same at any decimation. Default: (0, 0), none."""

    delay_redraw_on_reset: bool = False
    """Redraw each env's delay at its resets, after the reset's other draws. Default:
    False (one delay per env, drawn when the term is built)."""

    substep_delay_range_s: tuple[float, float] = (0.0, 0.0)
    """The delay's part under one env step, a band [min, max] in seconds converted to
    physics ticks (each end within 1e-6 of a tick), drawn per env when the term is built
    and at each reset, after the reset's other draws. An env keeps the previous step's
    wheel target for that many of the step's physics sub-steps; ticks of a whole env step
    or more go into the delay buffer. Default: (0, 0), none."""

    # ============================================================
    # Sim-to-Real: Command holds
    # ============================================================

    hold_fraction_range: tuple[float, float] = (0.0, 0.0)
    """Per-env stationary share of control steps on which no new command
    arrives and the previous one is re-executed, sampled at reset. A max of 0
    leaves the hold process inert."""

    hold_run_range_s: tuple[float, float] = (0.0, 0.0)
    """Per-env mean hold-run length [min, max] in seconds, sampled at reset, converted
    to env steps when the term is built; a run shorter than one env step is one."""

    # ============================================================
    # Command envelope
    # ============================================================

    speed_scale_range: tuple[float, float] | None = None
    """Per-env scale s ~ U[low, high], drawn at every reset, on both command
    caps: the L1 linear cap becomes s times the full linear speed and the
    angular cap s times the full angular speed. The action is still scaled by
    the full envelope, and only the clamp tightens. None (the default) leaves
    the caps at the full envelope. A range with low == high pins s, which is
    how an evaluation fixes the cap."""

    # ============================================================
    # Sim-to-Real: Slew rate limiting
    # ============================================================
    
    max_acceleration_rad_s2: float = float('inf')
    """Maximum wheel angular acceleration in rad/s², applied once per control step
    (a change of at most accel x control step). Default: inf (no limit)."""

    max_acceleration_range: tuple[float, float] = (float('inf'), float('inf'))
    """Range for per-env randomization of max acceleration [min, max] rad/s².
    A design spread: the motor controllers run from a regulated 12 V supply."""

    accel_falloff_speed: float = float('inf')
    """Wheel speed in rad/s at which the acceleration limit falls to zero, linearly in
    the wheel's speed (a(w) = a0 (1 - |w| / accel_falloff_speed)); it acts only where
    the limit is set. Default: inf (no fall-off)."""
