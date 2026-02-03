"""Environment configuration for Strafer navigation task.

This defines the full RL environment including:
- Scene configuration (robot, ground plane, lights)
- Observation space
- Action space  
- Reward functions
- Termination conditions
- Domain randomization events

All sim-to-real parameters (noise, latency, motor dynamics) are driven by
SimRealContractCfg presets for consistency:
  - IDEAL: No noise/delays (debugging)
  - REALISTIC: Matches real hardware (sim-to-real target)
  - ROBUST: Aggressive noise (stress-testing)

Environment Matrix (12 registered = 6 configs × Train/Play):
    | Realism  | Sensors    | Train ID                          |
    |----------|------------|-----------------------------------|
    | Ideal    | Full       | Isaac-Strafer-Nav-v0              |
    | Ideal    | Depth-only | Isaac-Strafer-Nav-Depth-v0        |
    | Ideal    | NoCam      | Isaac-Strafer-Nav-NoCam-v0        |
    | Realistic| Full       | Isaac-Strafer-Nav-Real-v0         |
    | Realistic| Depth-only | Isaac-Strafer-Nav-Real-Depth-v0   |
    | Robust   | Full       | Isaac-Strafer-Nav-Robust-v0       |
"""

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import TiledCameraCfg, ImuCfg

# Import custom MDP functions
from . import mdp

# Import robot configuration
from strafer_lab.assets import STRAFER_CFG

# Import sim-real contracts and helper functions
from .sim_real_cfg import (
    SimRealContractCfg,
    IDEAL_SIM_CONTRACT,
    REAL_ROBOT_CONTRACT,
    ROBUST_TRAINING_CONTRACT,
    get_imu_accel_noise,
    get_imu_gyro_noise,
    get_encoder_noise,
    get_depth_noise,
    get_rgb_noise,
)


# =============================================================================
# Scene Configurations
# =============================================================================


@configclass
class StraferSceneCfg(InteractiveSceneCfg):
    """Full scene with RGB+Depth camera + IMU."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = STRAFER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
    )

    # Intel RealSense D555 depth camera (87° FOV, 0.4-6m range)
    # Camera mount: 20cm forward (-Y), 25cm up (+Z) from body_link
    # Rotation: 90° around Z axis to point camera in robot's forward direction (-Y in USD)
    d555_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link/d555_camera",
        update_period=1.0 / 30.0,
        height=60,
        width=80,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            horizontal_aperture=3.68,
            clipping_range=(0.4, 6.0),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.20, 0.25),
            # ROS camera convention: Z-forward, X-right, Y-down
            # (0, 0, 0.707, -0.707) is 90° rotation around Z to point -Y
            rot=(0.0, 0.0, 0.707, -0.707),
            convention="ros",
        ),
    )

    # Intel RealSense D555 IMU (BMI055)
    # Co-located with camera for sensor fusion
    d555_imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=1.0 / 200.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, -0.20, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )


@configclass
class StraferSceneCfg_NoCam(InteractiveSceneCfg):
    """Scene without camera (IMU only) for faster training."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = STRAFER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8)),
    )

    # IMU only (no camera overhead)
    # Same placement as camera scene for consistency
    d555_imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=1.0 / 200.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.0, -0.20, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )


# =============================================================================
# Action Configurations - Per Realism Level
# =============================================================================


@configclass
class ActionsCfg_Ideal:
    """Ideal actions - no motor dynamics or delays (debugging/baselines)."""
    wheel_velocities = mdp.MecanumWheelActionCfg(
        asset_name="robot",
        joint_names=["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"],
        wheel_radius=0.048,
        wheel_base=0.304,
        track_width=0.304,
        motor_rpm=312.0,
        max_wheel_angular_vel=32.67,
        enable_motor_dynamics=False,
        min_delay_steps=0,
        max_delay_steps=0,
    )


@configclass
class ActionsCfg_Realistic:
    """Realistic actions - motor dynamics + delays (sim-to-real target)."""
    wheel_velocities = mdp.MecanumWheelActionCfg(
        asset_name="robot",
        joint_names=["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"],
        wheel_radius=0.048,
        wheel_base=0.304,
        track_width=0.304,
        motor_rpm=312.0,
        max_wheel_angular_vel=32.67,
        enable_motor_dynamics=True,
        motor_time_constant=0.05,
        min_delay_steps=1,
        max_delay_steps=3,
        max_acceleration_rad_s2=100.0,
    )


@configclass
class ActionsCfg_Robust:
    """Robust actions - aggressive dynamics + delays (stress-testing)."""
    wheel_velocities = mdp.MecanumWheelActionCfg(
        asset_name="robot",
        joint_names=["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"],
        wheel_radius=0.048,
        wheel_base=0.304,
        track_width=0.304,
        motor_rpm=312.0,
        max_wheel_angular_vel=32.67,
        enable_motor_dynamics=True,
        motor_time_constant=0.06,
        min_delay_steps=1,
        max_delay_steps=5,
        max_acceleration_rad_s2=80.0,
    )


# =============================================================================
# Observation Configurations - Per Realism Level
# =============================================================================

# Sensor normalization constants (for scale parameter)
# Noise is applied to RAW values, then scaled to normalize for neural network
IMU_ACCEL_MAX = 156.96  # ±16g in m/s²
IMU_GYRO_MAX = 34.9     # ±2000 °/s in rad/s
ENCODER_VEL_MAX = 5000.0  # Max ticks/sec
DEPTH_MAX = 6.0  # Max depth in meters

# Scale factors: scale = 1/max_value to normalize to [0, 1] or [-1, 1]
_IMU_ACCEL_SCALE = 1.0 / IMU_ACCEL_MAX
_IMU_GYRO_SCALE = 1.0 / IMU_GYRO_MAX
_ENCODER_SCALE = 1.0 / ENCODER_VEL_MAX
_DEPTH_SCALE = 1.0 / DEPTH_MAX  # Normalizes depth to [0, 1]

# Helper: Standard sensor params (no normalization in obs functions - handled by scale)
_IMU_ACCEL_PARAMS = {"sensor_cfg": SceneEntityCfg("d555_imu")}
_IMU_GYRO_PARAMS = {"sensor_cfg": SceneEntityCfg("d555_imu")}
_ENCODER_PARAMS = {}  # No params needed - raw output
_DEPTH_PARAMS = {"sensor_cfg": SceneEntityCfg("d555_camera"), "max_depth": DEPTH_MAX}
_RGB_PARAMS = {"sensor_cfg": SceneEntityCfg("d555_camera")}  # Returns [0,1] directly


# -----------------------------------------------------------------------------
# IDEAL: No noise
# -----------------------------------------------------------------------------

@configclass
class ObsCfg_Full_Ideal:
    """Full sensors (RGB+Depth), no noise."""
    @configclass
    class PolicyCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObsCfg_Depth_Ideal:
    """Depth-only, no noise."""
    @configclass
    class PolicyCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObsCfg_NoCam_Ideal:
    """Proprioceptive-only, no noise."""
    @configclass
    class PolicyCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# REALISTIC: Realistic noise from REAL_ROBOT_CONTRACT
# -----------------------------------------------------------------------------

# Pre-compute noise configs from contract
_REAL_ACCEL_NOISE = get_imu_accel_noise(REAL_ROBOT_CONTRACT)
_REAL_GYRO_NOISE = get_imu_gyro_noise(REAL_ROBOT_CONTRACT)
_REAL_ENCODER_NOISE = get_encoder_noise(REAL_ROBOT_CONTRACT)
_REAL_DEPTH_NOISE = get_depth_noise(REAL_ROBOT_CONTRACT)
_REAL_RGB_NOISE = get_rgb_noise(REAL_ROBOT_CONTRACT)


@configclass
class ObsCfg_Full_Realistic:
    """Full sensors with realistic noise."""
    @configclass
    class PolicyCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, noise=_REAL_ACCEL_NOISE, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, noise=_REAL_GYRO_NOISE, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, noise=_REAL_ENCODER_NOISE, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, noise=_REAL_DEPTH_NOISE, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS, noise=_REAL_RGB_NOISE)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObsCfg_Depth_Realistic:
    """Depth-only with realistic noise."""
    @configclass
    class PolicyCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, noise=_REAL_ACCEL_NOISE, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, noise=_REAL_GYRO_NOISE, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, noise=_REAL_ENCODER_NOISE, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, noise=_REAL_DEPTH_NOISE, scale=_DEPTH_SCALE)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObsCfg_NoCam_Realistic:
    """Proprioceptive-only with realistic noise."""
    @configclass
    class PolicyCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, noise=_REAL_ACCEL_NOISE, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, noise=_REAL_GYRO_NOISE, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, noise=_REAL_ENCODER_NOISE, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# -----------------------------------------------------------------------------
# ROBUST: Aggressive noise from ROBUST_TRAINING_CONTRACT
# -----------------------------------------------------------------------------

# Pre-compute noise configs from contract
_ROBUST_ACCEL_NOISE = get_imu_accel_noise(ROBUST_TRAINING_CONTRACT)
_ROBUST_GYRO_NOISE = get_imu_gyro_noise(ROBUST_TRAINING_CONTRACT)
_ROBUST_ENCODER_NOISE = get_encoder_noise(ROBUST_TRAINING_CONTRACT)
_ROBUST_DEPTH_NOISE = get_depth_noise(ROBUST_TRAINING_CONTRACT)
_ROBUST_RGB_NOISE = get_rgb_noise(ROBUST_TRAINING_CONTRACT)


@configclass
class ObsCfg_Full_Robust:
    """Full sensors with aggressive noise for robust training."""
    @configclass
    class PolicyCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, noise=_ROBUST_ACCEL_NOISE, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, noise=_ROBUST_GYRO_NOISE, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, noise=_ROBUST_ENCODER_NOISE, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, noise=_ROBUST_DEPTH_NOISE, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS, noise=_ROBUST_RGB_NOISE)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# =============================================================================
# Shared MDP Components
# =============================================================================


@configclass
class CommandsCfg:
    """Command terms for the MDP."""
    goal_command = mdp.GoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 15.0),
        debug_vis=True,
        goal_range=mdp.GoalCommandCfg.Ranges(pos_x=(-5.0, 5.0), pos_y=(-5.0, 5.0)),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    goal_reached = RewTerm(func=mdp.goal_reached_reward, weight=10.0, params={"threshold": 0.5, "command_name": "goal_command"})
    goal_progress = RewTerm(func=mdp.goal_progress_reward, weight=1.0, params={"command_name": "goal_command"})
    heading_alignment = RewTerm(func=mdp.heading_to_goal_reward, weight=0.5, params={"command_name": "goal_command"})
    energy_penalty = RewTerm(func=mdp.energy_penalty, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-0.1)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_flipped = DoneTerm(func=mdp.robot_flipped, params={"threshold": 0.5})


@configclass
class EventsCfg:
    """Event/randomization terms for the MDP."""
    reset_robot = EventTerm(
        func=mdp.reset_robot_state,
        mode="reset",
        params={"pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-math.pi, math.pi)}},
    )
    randomize_friction = EventTerm(
        func=mdp.randomize_friction,
        mode="reset",
        params={"friction_range": (0.5, 1.5)},
    )


# =============================================================================
# Environment Configurations - 12 Variants (6 configs × Train/Play)
# =============================================================================

# -----------------------------------------------------------------------------
# IDEAL: No noise, no motor dynamics (debugging/baselines)
# -----------------------------------------------------------------------------

@configclass
class StraferNavEnvCfg(ManagerBasedRLEnvCfg):
    """Ideal Full (RGB+Depth) - baseline for debugging."""
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg_Ideal = ActionsCfg_Ideal()
    observations: ObsCfg_Full_Ideal = ObsCfg_Full_Ideal()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class StraferNavEnvCfg_PLAY(StraferNavEnvCfg):
    """Play/eval config for Ideal Full."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


@configclass
class StraferNavEnvCfg_Depth(ManagerBasedRLEnvCfg):
    """Ideal Depth-only."""
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg_Ideal = ActionsCfg_Ideal()
    observations: ObsCfg_Depth_Ideal = ObsCfg_Depth_Ideal()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class StraferNavEnvCfg_Depth_PLAY(StraferNavEnvCfg_Depth):
    """Play/eval config for Ideal Depth-only."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


@configclass
class StraferNavEnvCfg_NoCam(ManagerBasedRLEnvCfg):
    """Ideal Proprioceptive-only (fastest training)."""
    scene: StraferSceneCfg_NoCam = StraferSceneCfg_NoCam(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg_Ideal = ActionsCfg_Ideal()
    observations: ObsCfg_NoCam_Ideal = ObsCfg_NoCam_Ideal()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class StraferNavEnvCfg_NoCam_PLAY(StraferNavEnvCfg_NoCam):
    """Play/eval config for Ideal Proprioceptive-only."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


# -----------------------------------------------------------------------------
# REALISTIC: Realistic noise + motor dynamics (sim-to-real target)
# -----------------------------------------------------------------------------

@configclass
class StraferNavEnvCfg_Real(ManagerBasedRLEnvCfg):
    """Realistic Full (RGB+Depth) - main sim-to-real target."""
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_Full_Realistic = ObsCfg_Full_Realistic()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class StraferNavEnvCfg_Real_PLAY(StraferNavEnvCfg_Real):
    """Play/eval config for Realistic Full."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        # Keep noise enabled to match training distribution


@configclass
class StraferNavEnvCfg_Real_Depth(ManagerBasedRLEnvCfg):
    """Realistic Depth-only - balanced sim-to-real."""
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_Depth_Realistic = ObsCfg_Depth_Realistic()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class StraferNavEnvCfg_Real_Depth_PLAY(StraferNavEnvCfg_Real_Depth):
    """Play/eval config for Realistic Depth-only."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


# -----------------------------------------------------------------------------
# ROBUST: Aggressive noise + dynamics for stress-testing
# -----------------------------------------------------------------------------

@configclass
class StraferNavEnvCfg_Robust(ManagerBasedRLEnvCfg):
    """Robust Full (RGB+Depth) - aggressive noise for worst-case robustness."""
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_Full_Robust = ObsCfg_Full_Robust()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class StraferNavEnvCfg_Robust_PLAY(StraferNavEnvCfg_Robust):
    """Play/eval config for Robust Full."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
