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

Environment Matrix (30 registered = 15 configs x Train/Play):

    | Realism   | Sensors          | Train ID                                    |
    |-----------|------------------|---------------------------------------------|
    | Ideal     | Full             | Isaac-Strafer-Nav-v0                        |
    | Ideal     | Depth-only       | Isaac-Strafer-Nav-Depth-v0                  |
    | Ideal     | NoCam            | Isaac-Strafer-Nav-NoCam-v0                  |
    | Realistic | Full             | Isaac-Strafer-Nav-Real-v0                   |
    | Realistic | Depth-only       | Isaac-Strafer-Nav-Real-Depth-v0             |
    | Realistic | NoCam            | Isaac-Strafer-Nav-Real-NoCam-v0             |
    | Realistic | InfinigenDepth   | Isaac-Strafer-Nav-Real-InfinigenDepth-v0    |
    | Realistic | ProcRoom NoCam   | Isaac-Strafer-Nav-Real-ProcRoom-NoCam-v0    |
    | Realistic | ProcRoom Depth   | Isaac-Strafer-Nav-Real-ProcRoom-Depth-v0    |
    | Robust    | Full             | Isaac-Strafer-Nav-Robust-v0                 |
    | Robust    | Depth-only       | Isaac-Strafer-Nav-Robust-Depth-v0           |
    | Robust    | NoCam            | Isaac-Strafer-Nav-Robust-NoCam-v0           |
    | Robust    | InfinigenDepth   | Isaac-Strafer-Nav-Robust-InfinigenDepth-v0  |
    | Robust    | ProcRoom NoCam   | Isaac-Strafer-Nav-Robust-ProcRoom-NoCam-v0  |
    | Robust    | ProcRoom Depth   | Isaac-Strafer-Nav-Robust-ProcRoom-Depth-v0  |

Each has a -Play-v0 variant for evaluation (fewer envs).
"""

import json
import math
from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import TiledCameraCfg, ImuCfg, ContactSensorCfg
from isaaclab_physx.physics import PhysxCfg

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
    get_action_config_params,
)
from .d555_cfg import make_d555_camera_cfg, make_d555_imu_cfg


# =============================================================================
# Scene USD Discovery (Phase 6 - Procedural Scenes)
# =============================================================================

_ASSET_ROOT = Path(__file__).resolve().parents[5] / "Assets"

# Composed scene USDs (offline pipeline output)
SCENE_USD_DIR = _ASSET_ROOT / "generated" / "scenes"



def _get_scene_usd_paths() -> list[str]:
    """Discover composed scene USDC files for Infinigen training (offline pipeline).

    Returns absolute paths to valid scene .usdc files in SCENE_USD_DIR.
    If scenes_metadata.json exists, only returns scenes listed there
    (degenerate scenes are excluded by prep_room_usds.py).
    """
    if not SCENE_USD_DIR.is_dir():
        raise FileNotFoundError(
            f"Scene USD directory not found: {SCENE_USD_DIR}\n"
            "Run prep_room_usds.py first to generate scenes."
        )
    # Use metadata to filter to valid scenes only
    meta_path = SCENE_USD_DIR / "scenes_metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        valid_names = set(meta["scenes"].keys())
        paths = sorted(
            str(p) for p in SCENE_USD_DIR.iterdir()
            if p.suffix in (".usdc", ".usd") and p.stem in valid_names
        )
    else:
        paths = sorted(
            str(p) for p in SCENE_USD_DIR.iterdir()
            if p.suffix in (".usdc", ".usd") and p.stem.startswith("scene_")
        )
    if not paths:
        raise FileNotFoundError(
            f"No valid scene_*.usdc files found in: {SCENE_USD_DIR}\n"
            "Run prep_room_usds.py first to generate scenes."
        )
    return paths


def _get_scenes_metadata() -> dict | None:
    """Load full scenes metadata including per-scene spawn points.

    Returns the full metadata dict, or None if not found.
    """
    meta_path = SCENE_USD_DIR / "scenes_metadata.json"
    if not meta_path.is_file():
        return None
    return json.loads(meta_path.read_text())





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

    # Intel RealSense D555 depth camera (87° FOV, 0.4-6m usable range)
    # Camera mount: 20cm forward (+X), 25cm up (+Z) from body_link (ROS frame)
    # Rotation: ROS camera frame (Z-forward, X-right, Y-down) aligned to robot frame
    #
    # Near clip set to 0.01m (not 0.4m) so sim renders close objects.
    # The depth_image() observation handles the D555's real 0.4m blind zone
    # by filling nearfield pixels with a saturated value (see observations.py).
    d555_camera: TiledCameraCfg = make_d555_camera_cfg(
        data_types=("rgb", "distance_to_image_plane"),
    )

    # Intel RealSense D555 IMU (BMI055)
    # Co-located with camera for sensor fusion
    d555_imu: ImuCfg = make_d555_imu_cfg()

    # Contact sensor on body_link for collision detection.
    # Uses net_forces_w — body_link is wheel-suspended (~10cm above ground),
    # so net forces detect collisions with scene geometry, not ground contact.
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=0.0,
        history_length=1,
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
    # Same placement as camera scene for consistency (ROS frame)
    d555_imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=1.0 / 200.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.20, 0.0, 0.25),
            rot=(0.0, 0.0, 0.0, 1.0),  # identity quaternion (XYZW)
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )

    # Contact sensor on body_link for collision detection.
    # Uses net_forces_w — body_link is wheel-suspended (~10cm above ground),
    # so net forces detect collisions with scene geometry, not ground contact.
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=0.0,
        history_length=1,
    )


@configclass
class StraferSceneCfg_Infinigen(InteractiveSceneCfg):
    """Scene with procedural Infinigen room geometry as a global shared prim.

    Scene geometry is loaded once at /World/Room (collision_group=-1) so all
    robot environments share a single copy. Environments are co-located at the
    origin (env_spacing=0) — per-env collision filtering keeps robots isolated
    from each other while the global room geometry collides with all of them.

    Contact sensor uses net_forces_w (no filter) since scene mesh prims are
    variable and unknown at config time.
    """

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

    directional_light = AssetBaseCfg(
        prim_path="/World/DirectionalLight",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.866, 0.0, 0.5, 0.0)),  # 60° from vertical
    )

    # Intel RealSense D555 depth camera (near clip 0.01m, see StraferSceneCfg comment)
    d555_camera: TiledCameraCfg = make_d555_camera_cfg(
        data_types=("rgb", "distance_to_image_plane"),
    )

    # Intel RealSense D555 IMU (same as StraferSceneCfg)
    d555_imu: ImuCfg = make_d555_imu_cfg()

    # Procedural scene geometry loaded as a global prim (single copy).
    # All robot environments share this scene via collision_group=-1.
    # UsdFileCfg path is set in __post_init__ to avoid import-time file
    # discovery that would break other env configs when scenes aren't generated.
    scene_geometry: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Room",
        spawn=sim_utils.UsdFileCfg(usd_path=""),  # populated in env config __post_init__
        collision_group=-1,
    )

    # Contact sensor on body_link 
    # Uses net_forces_w for collision detection against all scene geometry.
    # body_link is wheel-suspended (~10cm above ground), so net_forces_w on
    # body_link effectively detects only collisions with scene geometry
    # (walls, furniture), not ground plane contact.
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=0.0,
        history_length=1,
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
        wheel_axis_signs=(-1.0, 1.0, -1.0, 1.0),
        wheel_radius=0.048,
        wheel_base=0.336,
        track_width=0.4132,
        **get_action_config_params(IDEAL_SIM_CONTRACT),
    )


@configclass
class ActionsCfg_Realistic:
    """Realistic actions - motor dynamics + delays (sim-to-real target)."""
    wheel_velocities = mdp.MecanumWheelActionCfg(
        asset_name="robot",
        joint_names=["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"],
        wheel_axis_signs=(-1.0, 1.0, -1.0, 1.0),
        wheel_radius=0.048,
        wheel_base=0.336,
        track_width=0.4132,
        **get_action_config_params(REAL_ROBOT_CONTRACT),
    )


@configclass
class ActionsCfg_Robust:
    """Robust actions - aggressive dynamics + delays (stress-testing)."""
    wheel_velocities = mdp.MecanumWheelActionCfg(
        asset_name="robot",
        joint_names=["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"],
        wheel_axis_signs=(-1.0, 1.0, -1.0, 1.0),
        wheel_radius=0.048,
        wheel_base=0.336,
        track_width=0.4132,
        **get_action_config_params(ROBUST_TRAINING_CONTRACT),
    )


# =============================================================================
# Observation Configurations - Per Realism Level
# =============================================================================

# Sensor normalization constants (for scale parameter)
# Noise is applied to RAW values, then scaled to normalize for neural network
IMU_ACCEL_MAX = 156.96  # ±16g in m/s²
IMU_GYRO_MAX = 34.9     # ±2000 °/s in rad/s
ENCODER_VEL_MAX = 3000.0  # Max ticks/sec (312 RPM ≈ 2796 ticks/s + margin)
DEPTH_MAX = 6.0  # Max depth in meters
BODY_VEL_MAX = 2.0  # Max body velocity in m/s (robot tops ~1.57 m/s)
GOAL_DIST_MAX = 10.0  # Max goal distance in meters

# Scale factors: scale = 1/max_value to normalize to [0, 1] or [-1, 1]
_IMU_ACCEL_SCALE = 1.0 / IMU_ACCEL_MAX
_IMU_GYRO_SCALE = 1.0 / IMU_GYRO_MAX
_ENCODER_SCALE = 1.0 / ENCODER_VEL_MAX
_DEPTH_SCALE = 1.0 / DEPTH_MAX  # Normalizes depth to [0, 1]
_BODY_VEL_SCALE = 1.0 / BODY_VEL_MAX
_GOAL_DIST_SCALE = 1.0 / GOAL_DIST_MAX
_HEADING_SCALE = 1.0 / math.pi  # Heading error [-pi, pi] -> [-1, 1]

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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObsCfg_Depth_Robust:
    """Depth-only with aggressive noise for robust training."""
    @configclass
    class PolicyCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, noise=_ROBUST_ACCEL_NOISE, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, noise=_ROBUST_GYRO_NOISE, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, noise=_ROBUST_ENCODER_NOISE, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, noise=_ROBUST_DEPTH_NOISE, scale=_DEPTH_SCALE)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ObsCfg_NoCam_Robust:
    """Proprioceptive-only with aggressive noise for robust training."""
    @configclass
    class PolicyCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, noise=_ROBUST_ACCEL_NOISE, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, noise=_ROBUST_GYRO_NOISE, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, noise=_ROBUST_ENCODER_NOISE, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        imu_linear_acceleration = ObsTerm(func=mdp.imu_linear_acceleration, params=_IMU_ACCEL_PARAMS, scale=_IMU_ACCEL_SCALE)
        imu_angular_velocity = ObsTerm(func=mdp.imu_angular_velocity, params=_IMU_GYRO_PARAMS, scale=_IMU_GYRO_SCALE)
        wheel_encoder_velocities = ObsTerm(func=mdp.wheel_encoder_velocities, params=_ENCODER_PARAMS, scale=_ENCODER_SCALE)
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_to_goal = ObsTerm(func=mdp.goal_heading_to_goal, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
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
        goal_range=mdp.GoalCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0)),
    )


@configclass
class RewardsCfg:
    """Shared reward terms for navigation environments.

    The current baseline keeps only the core goal-reaching and collision terms
    active by default. Secondary shaping can be re-enabled environment-by-
    environment once the core loop is stable.
    """
    # --- Primary task signal (dense potential shaping) ---
    goal_progress = RewTerm(func=mdp.goal_progress_reward, weight=10.0, params={"command_name": "goal_command"})
    goal_proximity = RewTerm(func=mdp.goal_proximity_potential, weight=0.0, params={"command_name": "goal_command", "sigma": 0.3})
    # --- Sparse completion bonus (LARGE — must dominate any shaping residual) ---
    goal_reached = RewTerm(func=mdp.goal_reached_reward, weight=200.0, params={"threshold": 0.3, "command_name": "goal_command"})
    # --- Collision avoidance (must outweigh single-step progress to prevent b-lining) ---
    collision = RewTerm(func=mdp.collision_penalty_net, weight=-10.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
    collision_sustained = RewTerm(func=mdp.collision_sustained_penalty_net, weight=-5.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
    # --- Secondary signals (disabled by default) ---
    # When enabled, reward facing the goal direction rather than a random arrival yaw.
    heading_alignment = RewTerm(func=mdp.heading_to_goal_reward, weight=0.0, params={"command_name": "goal_command"})
    speed_near_goal = RewTerm(func=mdp.speed_near_goal_penalty, weight=0.0, params={"command_name": "goal_command", "distance_threshold": 0.8, "min_speed": 0.15})
    energy_penalty = RewTerm(func=mdp.energy_penalty, weight=0.0)
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Note: goal_reached is NOT a termination. Multi-goal resampling in
    GoalCommand._update_command() handles goal reach by issuing a new goal
    mid-episode. Episodes end only on time-out, robot flip, or sustained collision.
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_flipped = DoneTerm(func=mdp.robot_flipped, params={"threshold": 0.5})
    sustained_collision = DoneTerm(
        func=mdp.sustained_collision,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0, "max_steps": 5},
    )



# ---------------------------------------------------------------------------
# Events: tiered domain randomization (Ideal / Realistic / Robust)
#
# Structural events (robot reset) are identical across tiers.
# Domain randomization (friction, mass, motor strength, mount offset,
# goal noise) scales with realism level.
# ---------------------------------------------------------------------------

# Structural event shared by all tiers
_RESET_ROBOT = EventTerm(
    func=mdp.reset_robot_state,
    mode="reset",
    params={"pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-math.pi, math.pi)}},
)


@configclass
class EventsCfg_Ideal:
    """Ideal: no domain randomization. For debugging and ablation."""
    reset_robot = _RESET_ROBOT


@configclass
class EventsCfg_Realistic:
    """Realistic: moderate domain randomization for sim-to-real transfer."""
    reset_robot = _RESET_ROBOT
    randomize_friction = EventTerm(
        func=mdp.randomize_friction, mode="reset",
        params={"friction_range": (0.6, 1.2)},
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_mass, mode="reset",
        params={"mass_range": (0.95, 1.05)},
    )
    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength, mode="reset",
        params={"strength_range": (0.92, 1.08)},
    )
    randomize_d555_mount = EventTerm(
        func=mdp.randomize_d555_mount_offset, mode="reset",
        params={"max_angle_deg": 1.0},
    )
    randomize_goal_noise = EventTerm(
        func=mdp.randomize_goal_noise, mode="reset",
        params={"command_name": "goal_command", "noise_std": 0.15},
    )


@configclass
class EventsCfg_Robust:
    """Robust: aggressive domain randomization for worst-case robustness."""
    reset_robot = _RESET_ROBOT
    randomize_friction = EventTerm(
        func=mdp.randomize_friction, mode="reset",
        params={"friction_range": (0.3, 1.5)},
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_mass, mode="reset",
        params={"mass_range": (0.85, 1.15)},
    )
    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength, mode="reset",
        params={"strength_range": (0.80, 1.20)},
    )
    randomize_d555_mount = EventTerm(
        func=mdp.randomize_d555_mount_offset, mode="reset",
        params={"max_angle_deg": 3.0},
    )
    randomize_goal_noise = EventTerm(
        func=mdp.randomize_goal_noise, mode="reset",
        params={"command_name": "goal_command", "noise_std": 0.35},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms — gradually increase task difficulty during training."""

    goal_distance = CurrTerm(
        func=mdp.GoalDistanceCurriculum,
        params={
            "command_name": "goal_command",
            "initial_range": 0.5,
            "max_range": 5.0,
            "step_size": 0.25,
            "success_threshold": 5,
            "goal_threshold": 0.3,
        },
    )

@configclass
class CommandsCfg_Infinigen:
    """Commands for Infinigen — tighter goal range to fit within rooms (4-6m)."""
    goal_command = mdp.GoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 15.0),
        debug_vis=True,
        goal_range=mdp.GoalCommandCfg.Ranges(pos_x=(-2.5, 2.5), pos_y=(-2.5, 2.5)),
    )


@configclass
class RewardsCfg_ProcRoom(RewardsCfg):
    """ProcRoom rewards with dense obstacle-clearance shaping."""

    # Backward motion penalty: the robot has a forward-facing depth camera,
    # so driving backward is blind. This prevents the policy from exploiting
    # the depth-gradient asymmetry (backing into goals it can't see).
    backward_motion = RewTerm(func=mdp.backward_motion_penalty, weight=-2.0)

    obstacle_proximity = RewTerm(
        func=mdp.procroom_obstacle_proximity_penalty,
        weight=-1.0,
        params={
            "collection_name": "room_primitives",
            "sigma": 0.12,
            "distance_threshold": 0.35,
        },
    )


# Structural event for Infinigen: spawn on interior floor points
_RESET_ROBOT_INFINIGEN = EventTerm(
    func=mdp.reset_robot_state_on_floor,
    mode="reset",
    params={"spawn_points_xy": [], "yaw_range": (-math.pi, math.pi)},  # populated in __post_init__
)


@configclass
class EventsCfg_Infinigen_Realistic:
    """Realistic DR for Infinigen — no obstacle randomization, tighter spawn."""
    reset_robot = _RESET_ROBOT_INFINIGEN
    randomize_friction = EventTerm(
        func=mdp.randomize_friction, mode="reset",
        params={"friction_range": (0.6, 1.2)},
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_mass, mode="reset",
        params={"mass_range": (0.95, 1.05)},
    )
    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength, mode="reset",
        params={"strength_range": (0.92, 1.08)},
    )
    randomize_d555_mount = EventTerm(
        func=mdp.randomize_d555_mount_offset, mode="reset",
        params={"max_angle_deg": 1.0},
    )
    randomize_goal_noise = EventTerm(
        func=mdp.randomize_goal_noise, mode="reset",
        params={"command_name": "goal_command", "noise_std": 0.15},
    )


@configclass
class EventsCfg_Infinigen_Robust:
    """Robust DR for Infinigen — no obstacle randomization, tighter spawn."""
    reset_robot = _RESET_ROBOT_INFINIGEN
    randomize_friction = EventTerm(
        func=mdp.randomize_friction, mode="reset",
        params={"friction_range": (0.3, 1.5)},
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_mass, mode="reset",
        params={"mass_range": (0.85, 1.15)},
    )
    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength, mode="reset",
        params={"strength_range": (0.80, 1.20)},
    )
    randomize_d555_mount = EventTerm(
        func=mdp.randomize_d555_mount_offset, mode="reset",
        params={"max_angle_deg": 3.0},
    )
    randomize_goal_noise = EventTerm(
        func=mdp.randomize_goal_noise, mode="reset",
        params={"command_name": "goal_command", "noise_std": 0.35},
    )


@configclass
class CurriculumCfg_Infinigen:
    """Curriculum for Infinigen — goal distance only, no obstacle count."""
    goal_distance = CurrTerm(
        func=mdp.GoalDistanceCurriculum,
        params={
            "command_name": "goal_command",
            "initial_range": 0.5,
            "max_range": 3.0,
            "step_size": 0.25,
            "success_threshold": 5,
            "goal_threshold": 0.3,
        },
    )



# =============================================================================
# Environment Configurations - 30 Variants (15 configs × Train/Play)
# =============================================================================

_DEFAULT_NAV_SIM_DT = 1.0 / 120.0
_DEFAULT_NAV_RENDER_INTERVAL = 4
_DEFAULT_NAV_DECIMATION = 4
_DEFAULT_NAV_EPISODE_LENGTH_S = 20.0

_STANDARD_TRAIN_NUM_ENVS = 4096
_STANDARD_PLAY_NUM_ENVS = 50
_STANDARD_ENV_SPACING = 8.0

_INFINIGEN_TRAIN_NUM_ENVS = 3
_INFINIGEN_PLAY_NUM_ENVS = 8

_PROCROOM_NOCAM_TRAIN_NUM_ENVS = 256
_PROCROOM_DEPTH_TRAIN_NUM_ENVS = 64
_PROCROOM_NOCAM_PLAY_NUM_ENVS = 50
_PROCROOM_DEPTH_PLAY_NUM_ENVS = 8
_PROCROOM_ENV_SPACING = 10.0


def _apply_default_nav_runtime(cfg: ManagerBasedRLEnvCfg) -> None:
    """Apply the shared runtime defaults used by all navigation environments."""
    cfg.sim.dt = _DEFAULT_NAV_SIM_DT
    cfg.sim.render_interval = _DEFAULT_NAV_RENDER_INTERVAL
    cfg.decimation = _DEFAULT_NAV_DECIMATION
    cfg.episode_length_s = _DEFAULT_NAV_EPISODE_LENGTH_S


def _apply_play_num_envs(cfg: ManagerBasedRLEnvCfg, *, num_envs: int) -> None:
    """Shrink train configs for play/eval without changing other behavior."""
    cfg.scene.num_envs = num_envs


def _get_infinigen_spawn_points_xy() -> list[list[float]]:
    """Pool interior spawn points from the offline Infinigen scene metadata."""
    meta = _get_scenes_metadata()
    if not meta:
        return []

    spawn_points_xy = []
    for scene_data in meta["scenes"].values():
        spawn_points_xy.extend(scene_data.get("spawn_points_xy", []))
    return spawn_points_xy


def _apply_infinigen_scene_setup(cfg: ManagerBasedRLEnvCfg) -> None:
    """Attach the first scene USD and pooled floor spawn points to an env cfg."""
    cfg.scene.scene_geometry.spawn.usd_path = _get_scene_usd_paths()[0]

    spawn_points_xy = _get_infinigen_spawn_points_xy()
    if spawn_points_xy:
        cfg.events.reset_robot.params["spawn_points_xy"] = spawn_points_xy
        cfg.commands.goal_command.spawn_points_xy = spawn_points_xy


class _PlayEnvCfgMixin:
    """Shared play-mode override that only reduces the scene env count."""

    play_num_envs = _STANDARD_PLAY_NUM_ENVS

    def __post_init__(self):
        super().__post_init__()
        _apply_play_num_envs(self, num_envs=self.play_num_envs)


@configclass
class _BaseStraferNavEnvCfg(ManagerBasedRLEnvCfg):
    """Shared runtime defaults for all navigation environment configs."""

    seed: int = 42

    def __post_init__(self):
        _apply_default_nav_runtime(self)


@configclass
class _BaseStandardCameraNavEnvCfg(_BaseStraferNavEnvCfg):
    """Common train-time config for camera-equipped plane navigation envs."""

    scene: StraferSceneCfg = StraferSceneCfg(num_envs=_STANDARD_TRAIN_NUM_ENVS, env_spacing=_STANDARD_ENV_SPACING)
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class _BaseStandardNoCamNavEnvCfg(_BaseStraferNavEnvCfg):
    """Common train-time config for NoCam plane navigation envs."""

    scene: StraferSceneCfg_NoCam = StraferSceneCfg_NoCam(num_envs=_STANDARD_TRAIN_NUM_ENVS, env_spacing=_STANDARD_ENV_SPACING)
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


@configclass
class _BaseInfinigenDepthNavEnvCfg(_BaseStraferNavEnvCfg):
    """Common train-time config for depth-only Infinigen scene variants."""

    scene: StraferSceneCfg_Infinigen = StraferSceneCfg_Infinigen(num_envs=_INFINIGEN_TRAIN_NUM_ENVS, env_spacing=0.0)
    commands: CommandsCfg_Infinigen = CommandsCfg_Infinigen()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg_Infinigen = CurriculumCfg_Infinigen()

    def __post_init__(self):
        super().__post_init__()
        _apply_infinigen_scene_setup(self)


# -----------------------------------------------------------------------------
# IDEAL: No noise, no motor dynamics (debugging/baselines)
# -----------------------------------------------------------------------------

@configclass
class StraferNavEnvCfg(_BaseStandardCameraNavEnvCfg):
    """Ideal Full (RGB+Depth) - baseline for debugging."""
    actions: ActionsCfg_Ideal = ActionsCfg_Ideal()
    observations: ObsCfg_Full_Ideal = ObsCfg_Full_Ideal()
    events: EventsCfg_Ideal = EventsCfg_Ideal()


@configclass
class StraferNavEnvCfg_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg):
    """Play/eval config for Ideal Full."""


@configclass
class StraferNavEnvCfg_Depth(_BaseStandardCameraNavEnvCfg):
    """Ideal Depth-only."""
    actions: ActionsCfg_Ideal = ActionsCfg_Ideal()
    observations: ObsCfg_Depth_Ideal = ObsCfg_Depth_Ideal()
    events: EventsCfg_Ideal = EventsCfg_Ideal()


@configclass
class StraferNavEnvCfg_Depth_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Depth):
    """Play/eval config for Ideal Depth-only."""


@configclass
class StraferNavEnvCfg_NoCam(_BaseStandardNoCamNavEnvCfg):
    """Ideal Proprioceptive-only (fastest training)."""
    actions: ActionsCfg_Ideal = ActionsCfg_Ideal()
    observations: ObsCfg_NoCam_Ideal = ObsCfg_NoCam_Ideal()
    events: EventsCfg_Ideal = EventsCfg_Ideal()


@configclass
class StraferNavEnvCfg_NoCam_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_NoCam):
    """Play/eval config for Ideal Proprioceptive-only."""


# -----------------------------------------------------------------------------
# REALISTIC: Realistic noise + motor dynamics (sim-to-real target)
# -----------------------------------------------------------------------------

@configclass
class StraferNavEnvCfg_Real(_BaseStandardCameraNavEnvCfg):
    """Realistic Full (RGB+Depth) - main sim-to-real target."""
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_Full_Realistic = ObsCfg_Full_Realistic()
    events: EventsCfg_Realistic = EventsCfg_Realistic()


@configclass
class StraferNavEnvCfg_Real_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Real):
    """Play/eval config for Realistic Full; keeps noise enabled."""


@configclass
class StraferNavEnvCfg_Real_Depth(_BaseStandardCameraNavEnvCfg):
    """Realistic Depth-only - balanced sim-to-real."""
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_Depth_Realistic = ObsCfg_Depth_Realistic()
    events: EventsCfg_Realistic = EventsCfg_Realistic()


@configclass
class StraferNavEnvCfg_Real_Depth_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Real_Depth):
    """Play/eval config for Realistic Depth-only."""


@configclass
class StraferNavEnvCfg_Real_NoCam(_BaseStandardNoCamNavEnvCfg):
    """Realistic Proprioceptive-only - fast training with realistic dynamics."""
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_NoCam_Realistic = ObsCfg_NoCam_Realistic()
    events: EventsCfg_Realistic = EventsCfg_Realistic()


@configclass
class StraferNavEnvCfg_Real_NoCam_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Real_NoCam):
    """Play/eval config for Realistic Proprioceptive-only."""


# -----------------------------------------------------------------------------
# ROBUST: Aggressive noise + dynamics for stress-testing
# -----------------------------------------------------------------------------

@configclass
class StraferNavEnvCfg_Robust(_BaseStandardCameraNavEnvCfg):
    """Robust Full (RGB+Depth) - aggressive noise for worst-case robustness."""
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_Full_Robust = ObsCfg_Full_Robust()
    events: EventsCfg_Robust = EventsCfg_Robust()


@configclass
class StraferNavEnvCfg_Robust_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Robust):
    """Play/eval config for Robust Full."""


@configclass
class StraferNavEnvCfg_Robust_Depth(_BaseStandardCameraNavEnvCfg):
    """Robust Depth-only - aggressive noise for worst-case robustness."""
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_Depth_Robust = ObsCfg_Depth_Robust()
    events: EventsCfg_Robust = EventsCfg_Robust()


@configclass
class StraferNavEnvCfg_Robust_Depth_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Robust_Depth):
    """Play/eval config for Robust Depth-only."""


@configclass
class StraferNavEnvCfg_Robust_NoCam(_BaseStandardNoCamNavEnvCfg):
    """Robust Proprioceptive-only - aggressive noise for worst-case robustness."""
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_NoCam_Robust = ObsCfg_NoCam_Robust()
    events: EventsCfg_Robust = EventsCfg_Robust()


@configclass
class StraferNavEnvCfg_Robust_NoCam_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Robust_NoCam):
    """Play/eval config for Robust Proprioceptive-only."""


# =============================================================================
# INFINIGEN: Procedural Infinigen scene geometry
#
# These variants use StraferSceneCfg_Infinigen (Infinigen room USD scenes)
# instead of box obstacles. Uses net_forces_w for collision detection,
# tighter spawn/goal ranges to fit within rooms, and no obstacle curriculum.
# =============================================================================

@configclass
class StraferNavEnvCfg_Real_InfinigenDepth(_BaseInfinigenDepthNavEnvCfg):
    """Realistic Depth with procedural Infinigen scene geometry."""
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_Depth_Realistic = ObsCfg_Depth_Realistic()
    events: EventsCfg_Infinigen_Realistic = EventsCfg_Infinigen_Realistic()


@configclass
class StraferNavEnvCfg_Real_InfinigenDepth_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Real_InfinigenDepth):
    """Play/eval config for Realistic InfinigenDepth."""
    play_num_envs = _INFINIGEN_PLAY_NUM_ENVS


@configclass
class StraferNavEnvCfg_Robust_InfinigenDepth(_BaseInfinigenDepthNavEnvCfg):
    """Robust Depth with procedural Infinigen scene geometry."""
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_Depth_Robust = ObsCfg_Depth_Robust()
    events: EventsCfg_Infinigen_Robust = EventsCfg_Infinigen_Robust()


@configclass
class StraferNavEnvCfg_Robust_InfinigenDepth_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Robust_InfinigenDepth):
    """Play/eval config for Robust InfinigenDepth."""
    play_num_envs = _INFINIGEN_PLAY_NUM_ENVS


# =============================================================================
# PROCROOM: Procedural primitive rooms with GPU BFS solvability
#
# Lightweight primitive shapes (walls, furniture, clutter) generated at each
# episode reset. Supports 256+ envs with replicated physics. GPU BFS
# guarantees solvable layouts.
# =============================================================================

# Import procedural room palette builder
from .mdp.proc_room import build_proc_room_collection_cfg


@configclass
class StraferSceneCfg_ProcRoom(InteractiveSceneCfg):
    """Scene with procedural primitive rooms and depth camera."""

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

    # Intel RealSense D555 depth camera (near clip 0.01m, see StraferSceneCfg comment)
    d555_camera: TiledCameraCfg = make_d555_camera_cfg(
        data_types=("distance_to_image_plane",),
    )

    # Intel RealSense D555 IMU (same as StraferSceneCfg)
    d555_imu: ImuCfg = make_d555_imu_cfg()

    # 44-object primitive palette (walls, furniture, clutter)
    room_primitives: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects=build_proc_room_collection_cfg(),
    )

    # Contact sensor on body_link for collision detection
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=0.0,
        history_length=1,
    )


@configclass
class StraferSceneCfg_ProcRoom_NoCam(InteractiveSceneCfg):
    """Scene with procedural primitive rooms, no camera (fastest training)."""

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
    d555_imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=1.0 / 200.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.20, 0.0, 0.25),
            rot=(0.0, 0.0, 0.0, 1.0),  # identity quaternion (XYZW)
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )

    # 44-object primitive palette (walls, furniture, clutter)
    room_primitives: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects=build_proc_room_collection_cfg(),
    )

    # Contact sensor on body_link for collision detection
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=0.0,
        history_length=1,
    )


@configclass
class CommandsCfg_ProcRoom:
    """Commands for ProcRoom — one goal per episode from BFS reachable points."""
    goal_command = mdp.GoalCommandProcRoomCfg(
        asset_name="robot",
        # Keep a single goal active for the whole episode to narrow return variance.
        resampling_time_range=(1.0e6, 1.0e6),
        multi_goal=False,
        debug_vis=True,
        goal_range=mdp.GoalCommandProcRoomCfg.Ranges(pos_x=(-3.5, 3.5), pos_y=(-3.5, 3.5)),
    )


# Structural event for ProcRoom: generate room THEN reset robot
_GENERATE_PROC_ROOM = EventTerm(
    func=mdp.generate_proc_room,
    mode="reset",
    params={"collection_name": "room_primitives"},
)

_RESET_ROBOT_PROC_ROOM = EventTerm(
    func=mdp.reset_robot_proc_room,
    mode="reset",
    params={"yaw_range": (-math.pi, math.pi)},
)


_RANDOMIZE_PROC_ROOM_DIFFICULTY = EventTerm(
    func=mdp.randomize_proc_room_difficulty,
    mode="reset",
    params={"min_level": 7, "max_level": 7},
)


@configclass
class EventsCfg_ProcRoom_Realistic:
    """Realistic DR for ProcRoom — room generation + robot reset + DR."""
    randomize_difficulty = _RANDOMIZE_PROC_ROOM_DIFFICULTY
    generate_room = _GENERATE_PROC_ROOM
    reset_robot = _RESET_ROBOT_PROC_ROOM
    randomize_friction = EventTerm(
        func=mdp.randomize_friction, mode="reset",
        params={"friction_range": (0.6, 1.2)},
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_mass, mode="reset",
        params={"mass_range": (0.95, 1.05)},
    )
    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength, mode="reset",
        params={"strength_range": (0.92, 1.08)},
    )
    randomize_d555_mount = EventTerm(
        func=mdp.randomize_d555_mount_offset, mode="reset",
        params={"max_angle_deg": 1.0},
    )
    # Goal noise disabled for initial training (Phase 1).
    # Re-enable with noise_std=0.15 for Phase 3 robustness hardening.


@configclass
class EventsCfg_ProcRoom_Robust:
    """Robust DR for ProcRoom — room generation + robot reset + aggressive DR."""
    randomize_difficulty = _RANDOMIZE_PROC_ROOM_DIFFICULTY
    generate_room = _GENERATE_PROC_ROOM
    reset_robot = _RESET_ROBOT_PROC_ROOM
    randomize_friction = EventTerm(
        func=mdp.randomize_friction, mode="reset",
        params={"friction_range": (0.3, 1.5)},
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_mass, mode="reset",
        params={"mass_range": (0.85, 1.15)},
    )
    randomize_motor_strength = EventTerm(
        func=mdp.randomize_motor_strength, mode="reset",
        params={"strength_range": (0.80, 1.20)},
    )
    randomize_d555_mount = EventTerm(
        func=mdp.randomize_d555_mount_offset, mode="reset",
        params={"max_angle_deg": 3.0},
    )
    randomize_goal_noise = EventTerm(
        func=mdp.randomize_goal_noise, mode="reset",
        params={"command_name": "goal_command", "noise_std": 0.35},
    )


@configclass
class CurriculumCfg_ProcRoom:
    """No curriculum for ProcRoom — difficulty is randomized per-env via event.

    GoalDistanceCurriculum and RoomComplexityCurriculum were removed to match
    the demo distribution (random difficulty per episode). The curriculum classes
    remain in mdp/curriculums.py for future use.
    """
    pass


@configclass
class TerminationsCfg_ProcRoom(TerminationsCfg):
    """ProcRoom termination tweaks for more stable navigation training."""

    goal_reached = DoneTerm(
        func=mdp.goal_reached,
        params={"command_name": "goal_command", "threshold": 0.3},
    )
    sustained_collision = DoneTerm(
        func=mdp.sustained_collision,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0, "max_steps": 3},
    )


# --- ProcRoom PhysX buffer sizing ---
#
# ProcRoom envs have high rigid body counts (44 room primitives + robot with
# 40 roller joints per env).  The mecanum rollers continuously make/break
# ground contact as they spin, creating heavy churn in PhysX's broadphase
# found/lost pair tracking.  At high env counts (>24), the default GPU
# buffers overflow silently, causing dropped contacts → asymmetric wheel
# support → robot flipping.
#
# These overrides increase the relevant buffers by 4× to handle up to 256
# parallel ProcRoom environments without contact loss.


def _apply_procroom_physx_buffers(cfg) -> None:
    """Increase PhysX GPU buffers and stabilize solver for ProcRoom."""
    cfg.sim.physics = PhysxCfg(
        gpu_found_lost_pairs_capacity=2**23,             # 4× default (roller contact churn)
        gpu_max_rigid_contact_count=2**24,               # 2× default
        gpu_max_rigid_patch_count=2**18,                  # ~1.6× default
        gpu_found_lost_aggregate_pairs_capacity=2**26,   # ~2× default
        gpu_max_num_partitions=1,                         # single solver partition
        enable_stabilization=True,
        bounce_threshold_velocity=2.0,
    )


# --- ProcRoom environment configs ---

@configclass
class _BaseProcRoomNoCamNavEnvCfg(_BaseStraferNavEnvCfg):
    """Common train-time config for NoCam ProcRoom scene variants."""

    scene: StraferSceneCfg_ProcRoom_NoCam = StraferSceneCfg_ProcRoom_NoCam(
        num_envs=_PROCROOM_NOCAM_TRAIN_NUM_ENVS,
        env_spacing=_PROCROOM_ENV_SPACING,
    )
    commands: CommandsCfg_ProcRoom = CommandsCfg_ProcRoom()
    rewards: RewardsCfg_ProcRoom = RewardsCfg_ProcRoom()
    terminations: TerminationsCfg_ProcRoom = TerminationsCfg_ProcRoom()
    curriculum: CurriculumCfg_ProcRoom = CurriculumCfg_ProcRoom()

    def __post_init__(self):
        super().__post_init__()
        _apply_procroom_physx_buffers(self)


@configclass
class _BaseProcRoomDepthNavEnvCfg(_BaseStraferNavEnvCfg):
    """Common train-time config for depth ProcRoom scene variants."""

    scene: StraferSceneCfg_ProcRoom = StraferSceneCfg_ProcRoom(
        num_envs=_PROCROOM_DEPTH_TRAIN_NUM_ENVS,
        env_spacing=_PROCROOM_ENV_SPACING,
    )
    commands: CommandsCfg_ProcRoom = CommandsCfg_ProcRoom()
    rewards: RewardsCfg_ProcRoom = RewardsCfg_ProcRoom()
    terminations: TerminationsCfg_ProcRoom = TerminationsCfg_ProcRoom()
    curriculum: CurriculumCfg_ProcRoom = CurriculumCfg_ProcRoom()

    def __post_init__(self):
        super().__post_init__()
        _apply_procroom_physx_buffers(self)

@configclass
class StraferNavEnvCfg_Real_ProcRoom_NoCam(_BaseProcRoomNoCamNavEnvCfg):
    """Realistic NoCam ProcRoom — high env count for fast training."""
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_NoCam_Realistic = ObsCfg_NoCam_Realistic()
    events: EventsCfg_ProcRoom_Realistic = EventsCfg_ProcRoom_Realistic()


@configclass
class StraferNavEnvCfg_Real_ProcRoom_NoCam_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Real_ProcRoom_NoCam):
    """Play/eval config for Realistic ProcRoom NoCam."""
    play_num_envs = _PROCROOM_NOCAM_PLAY_NUM_ENVS


@configclass
class StraferNavEnvCfg_Real_ProcRoom_Depth(_BaseProcRoomDepthNavEnvCfg):
    """Realistic Depth ProcRoom — depth training with room geometry."""
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_Depth_Realistic = ObsCfg_Depth_Realistic()
    events: EventsCfg_ProcRoom_Realistic = EventsCfg_ProcRoom_Realistic()


@configclass
class StraferNavEnvCfg_Real_ProcRoom_Depth_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Real_ProcRoom_Depth):
    """Play/eval config for Realistic ProcRoom Depth."""
    play_num_envs = _PROCROOM_DEPTH_PLAY_NUM_ENVS


@configclass
class StraferNavEnvCfg_Robust_ProcRoom_NoCam(_BaseProcRoomNoCamNavEnvCfg):
    """Robust NoCam ProcRoom — aggressive noise with high env count."""
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_NoCam_Robust = ObsCfg_NoCam_Robust()
    events: EventsCfg_ProcRoom_Robust = EventsCfg_ProcRoom_Robust()


@configclass
class StraferNavEnvCfg_Robust_ProcRoom_NoCam_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Robust_ProcRoom_NoCam):
    """Play/eval config for Robust ProcRoom NoCam."""
    play_num_envs = _PROCROOM_NOCAM_PLAY_NUM_ENVS


@configclass
class StraferNavEnvCfg_Robust_ProcRoom_Depth(_BaseProcRoomDepthNavEnvCfg):
    """Robust Depth ProcRoom — aggressive noise with depth camera."""
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_Depth_Robust = ObsCfg_Depth_Robust()
    events: EventsCfg_ProcRoom_Robust = EventsCfg_ProcRoom_Robust()


@configclass
class StraferNavEnvCfg_Robust_ProcRoom_Depth_PLAY(_PlayEnvCfgMixin, StraferNavEnvCfg_Robust_ProcRoom_Depth):
    """Play/eval config for Robust ProcRoom Depth."""
    play_num_envs = _PROCROOM_DEPTH_PLAY_NUM_ENVS


