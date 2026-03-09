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

Environment Matrix (22 registered = 11 configs x Train/Play):

    | Realism   | Sensors    | Train ID                            |
    |-----------|------------|-------------------------------------|
    | Ideal     | Full       | Isaac-Strafer-Nav-v0                |
    | Ideal     | Depth-only | Isaac-Strafer-Nav-Depth-v0          |
    | Ideal     | NoCam      | Isaac-Strafer-Nav-NoCam-v0          |
    | Realistic | Full       | Isaac-Strafer-Nav-Real-v0           |
    | Realistic | Depth-only | Isaac-Strafer-Nav-Real-Depth-v0     |
    | Realistic | NoCam      | Isaac-Strafer-Nav-Real-NoCam-v0     |
    | Realistic | ProcDepth  | Isaac-Strafer-Nav-Real-ProcDepth-v0 |
    | Robust    | Full       | Isaac-Strafer-Nav-Robust-v0         |
    | Robust    | Depth-only | Isaac-Strafer-Nav-Robust-Depth-v0   |
    | Robust    | NoCam      | Isaac-Strafer-Nav-Robust-NoCam-v0   |
    | Robust    | ProcDepth  | Isaac-Strafer-Nav-Robust-ProcDepth-v0 |

Each has a -Play-v0 variant for evaluation (fewer envs).
"""

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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.wrappers import MultiUsdFileCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import TiledCameraCfg, ImuCfg, ContactSensorCfg

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


# =============================================================================
# Obstacle Template
# =============================================================================

NUM_OBSTACLES = 8

OBSTACLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Obstacle",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(50.0, 50.0, 0.15)),  # far away; repositioned at reset
    spawn=sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,  # static obstacles
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.2)),
    ),
)


# =============================================================================
# Scene USD Discovery (Phase 6 - Procedural Scenes)
# =============================================================================

_ASSET_ROOT = Path(__file__).resolve().parents[5] / "Assets"

# Composed scene USDs (offline pipeline output)
SCENE_USD_DIR = _ASSET_ROOT / "generated" / "scenes"



def _get_scene_usd_paths() -> list[str]:
    """Discover composed scene USDC files for ProcScene training (offline pipeline).

    Returns absolute paths to all .usdc/.usd files in SCENE_USD_DIR,
    sorted by name for deterministic ordering.
    """
    if not SCENE_USD_DIR.is_dir():
        raise FileNotFoundError(
            f"Scene USD directory not found: {SCENE_USD_DIR}\n"
            "Run compose_scenes_replicator.py first to generate scenes."
        )
    paths = sorted(
        str(p) for p in SCENE_USD_DIR.iterdir()
        if p.suffix in (".usdc", ".usd") and p.stem.startswith("scene_")
    )
    if not paths:
        raise FileNotFoundError(
            f"No scene_*.usdc files found in: {SCENE_USD_DIR}\n"
            "Run compose_scenes_replicator.py first to generate scenes."
        )
    return paths




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
    # Camera mount: 20cm forward (+X), 25cm up (+Z) from body_link (ROS frame)
    # Rotation: ROS camera frame (Z-forward, X-right, Y-down) aligned to robot frame
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
            pos=(0.20, 0.0, 0.25),
            # ROS camera convention: Z-forward, X-right, Y-down
            # (0.5, -0.5, 0.5, -0.5) aligns camera forward (+Z) to robot +X
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # Intel RealSense D555 IMU (BMI055)
    # Co-located with camera for sensor fusion
    d555_imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=1.0 / 200.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.20, 0.0, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )

    # Obstacles: 8 rigid body boxes for obstacle avoidance training
    # Repositioned randomly at episode reset via events.randomize_obstacles()
    obstacle_0: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_0")
    obstacle_1: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    obstacle_2: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2")
    obstacle_3: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_3")
    obstacle_4: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_4")
    obstacle_5: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_5")
    obstacle_6: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_6")
    obstacle_7: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_7")

    # Contact sensor on body_link for obstacle collision detection.
    # body_link has a collision box (added by setup_physics.py) covering
    # the full robot footprint (0.48 x 0.44 x 0.216 m, frame bottom to
    # support rail top). Detects collisions from any direction.
    # filter_prim_paths_expr isolates obstacle contacts from ground contact.
    # Each obstacle is listed separately because PhysX tensors API requires
    # each filter pattern to match exactly 1 prim per env (wildcard Obstacle_.*
    # matches 8 per env → error). This gives force_matrix_w shape (N, 1, 8, 3).
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Obstacle_0",
            "{ENV_REGEX_NS}/Obstacle_1",
            "{ENV_REGEX_NS}/Obstacle_2",
            "{ENV_REGEX_NS}/Obstacle_3",
            "{ENV_REGEX_NS}/Obstacle_4",
            "{ENV_REGEX_NS}/Obstacle_5",
            "{ENV_REGEX_NS}/Obstacle_6",
            "{ENV_REGEX_NS}/Obstacle_7",
        ],
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
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )

    # Obstacles (same as Full scene)
    obstacle_0: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_0")
    obstacle_1: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_1")
    obstacle_2: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_2")
    obstacle_3: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_3")
    obstacle_4: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_4")
    obstacle_5: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_5")
    obstacle_6: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_6")
    obstacle_7: RigidObjectCfg = OBSTACLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Obstacle_7")

    # Contact sensor on body_link for obstacle collision detection.
    # body_link has a collision box (added by setup_physics.py) covering
    # the full robot footprint (0.48 x 0.44 x 0.216 m, frame bottom to
    # support rail top). Detects collisions from any direction.
    # filter_prim_paths_expr isolates obstacle contacts from ground contact.
    # Each obstacle is listed separately because PhysX tensors API requires
    # each filter pattern to match exactly 1 prim per env (wildcard Obstacle_.*
    # matches 8 per env → error). This gives force_matrix_w shape (N, 1, 8, 3).
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Obstacle_0",
            "{ENV_REGEX_NS}/Obstacle_1",
            "{ENV_REGEX_NS}/Obstacle_2",
            "{ENV_REGEX_NS}/Obstacle_3",
            "{ENV_REGEX_NS}/Obstacle_4",
            "{ENV_REGEX_NS}/Obstacle_5",
            "{ENV_REGEX_NS}/Obstacle_6",
            "{ENV_REGEX_NS}/Obstacle_7",
        ],
    )


@configclass
class StraferSceneCfg_ProcScene(InteractiveSceneCfg):
    """Scene with procedural Infinigen room geometry instead of box obstacles.

    Uses MultiUsdFileCfg to load random composed scene USDs per environment.
    Requires replicate_physics=False since each env gets a different scene.
    Contact sensor uses net_forces_w (no filter) since scene mesh prims are
    variable and unknown at config time.
    """

    replicate_physics = False

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

    # Intel RealSense D555 depth camera (same as StraferSceneCfg)
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
            pos=(0.20, 0.0, 0.25),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # Intel RealSense D555 IMU (same as StraferSceneCfg)
    d555_imu: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link",
        update_period=1.0 / 200.0,
        offset=ImuCfg.OffsetCfg(
            pos=(0.20, 0.0, 0.25),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        gravity_bias=(0.0, 0.0, 9.81),
    )

    # Procedural scene geometry loaded from composed scene USDs.
    # MultiUsdFileCfg with random_choice=True assigns a random scene per env.
    # collision_props ensures collision meshes are active on all geometry.
    # Note: usd_path is set in __post_init__ to avoid import-time file discovery
    # that would break other env configs when scenes aren't generated yet.
    scene_geometry: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Scene",
        spawn=MultiUsdFileCfg(
            usd_path=[],  # populated in env config __post_init__
            random_choice=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
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
        track_width=0.4284,
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
        track_width=0.4284,
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
        track_width=0.4284,
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS)
        privileged = ObsTerm(func=mdp.privileged_ground_truth, params={"command_name": "goal_command"})
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
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        privileged = ObsTerm(func=mdp.privileged_ground_truth, params={"command_name": "goal_command"})
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
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        privileged = ObsTerm(func=mdp.privileged_ground_truth, params={"command_name": "goal_command"})
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS)
        privileged = ObsTerm(func=mdp.privileged_ground_truth, params={"command_name": "goal_command"})
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
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        privileged = ObsTerm(func=mdp.privileged_ground_truth, params={"command_name": "goal_command"})
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
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        privileged = ObsTerm(func=mdp.privileged_ground_truth, params={"command_name": "goal_command"})
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        rgb_image = ObsTerm(func=mdp.rgb_image, params=_RGB_PARAMS)
        privileged = ObsTerm(func=mdp.privileged_ground_truth, params={"command_name": "goal_command"})
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(func=mdp.depth_image, params=_DEPTH_PARAMS, scale=_DEPTH_SCALE)
        privileged = ObsTerm(func=mdp.privileged_ground_truth, params={"command_name": "goal_command"})
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
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
        goal_position = ObsTerm(func=mdp.goal_position_relative, params={"command_name": "goal_command"})
        goal_distance = ObsTerm(func=mdp.goal_distance, params={"command_name": "goal_command"}, scale=_GOAL_DIST_SCALE)
        goal_heading_relative = ObsTerm(func=mdp.goal_heading_relative, params={"command_name": "goal_command"}, scale=_HEADING_SCALE)
        body_velocity_xy = ObsTerm(func=mdp.body_velocity_xy, scale=_BODY_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        privileged = ObsTerm(func=mdp.privileged_ground_truth, params={"command_name": "goal_command"})
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
    """Reward terms for the MDP.

    Design principles (sim-to-real navigation):
    - Dense shaping (progress/proximity) for the main task signal
    - Sparse bonus for goal completion
    - Small penalties as guardrails (collision, jerk), not primary signal
    - Positive-dominant: robot should want to reach goals, not just avoid punishment
    """
    # --- Primary task signal (dense) ---
    goal_progress = RewTerm(func=mdp.goal_progress_reward, weight=2.0, params={"command_name": "goal_command"})
    goal_proximity = RewTerm(func=mdp.goal_proximity_reward, weight=1.5, params={"command_name": "goal_command", "sigma": 0.3})
    # --- Sparse completion bonus ---
    goal_reached = RewTerm(func=mdp.goal_reached_reward, weight=10.0, params={"threshold": 0.3, "command_name": "goal_command"})
    # --- Heading (low weight — mecanum can strafe, heading is secondary) ---
    heading_alignment = RewTerm(func=mdp.heading_to_goal_reward, weight=0.05, params={"command_name": "goal_command"})
    # --- Collision avoidance ---
    collision = RewTerm(func=mdp.collision_penalty, weight=-5.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
    collision_sustained = RewTerm(func=mdp.collision_sustained_penalty, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
    # --- Slow down near goal ---
    speed_near_goal = RewTerm(func=mdp.speed_near_goal_penalty, weight=-0.3, params={"command_name": "goal_command", "distance_threshold": 0.8})
    # --- Regularization (smooth, energy-efficient motion transfers better to real hardware) ---
    energy_penalty = RewTerm(func=mdp.energy_penalty, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-0.05)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Note: goal_reached is NOT a termination. Multi-goal resampling in
    GoalCommand._update_command() handles goal reach by issuing a new goal
    mid-episode. Episodes end only on time-out or robot flip.
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_flipped = DoneTerm(func=mdp.robot_flipped, params={"threshold": 0.5})


_OBSTACLE_NAMES = [f"obstacle_{i}" for i in range(NUM_OBSTACLES)]


# ---------------------------------------------------------------------------
# Events: tiered domain randomization (Ideal / Realistic / Robust)
#
# Structural events (robot reset, obstacle placement) are identical across
# tiers. Domain randomization (friction, mass, motor strength, mount offset,
# goal noise) scales with realism level.
# ---------------------------------------------------------------------------

# Structural events shared by all tiers
_RESET_ROBOT = EventTerm(
    func=mdp.reset_robot_state,
    mode="reset",
    params={"pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-math.pi, math.pi)}},
)
_RANDOMIZE_OBSTACLES = EventTerm(
    func=mdp.randomize_obstacles,
    mode="reset",
    params={
        "obstacle_names": _OBSTACLE_NAMES,
        "position_range": {"x": (-4.0, 4.0), "y": (-4.0, 4.0)},
        "min_robot_dist": 0.6,
    },
)


@configclass
class EventsCfg_Ideal:
    """Ideal: no domain randomization. For debugging and ablation."""
    reset_robot = _RESET_ROBOT
    randomize_obstacles = _RANDOMIZE_OBSTACLES


@configclass
class EventsCfg_Realistic:
    """Realistic: moderate domain randomization for sim-to-real transfer."""
    reset_robot = _RESET_ROBOT
    randomize_obstacles = _RANDOMIZE_OBSTACLES
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
    randomize_obstacles = _RANDOMIZE_OBSTACLES
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
            "initial_range": 2.0,
            "max_range": 5.0,
            "step_size": 0.5,
            "success_threshold": 5,
            "goal_threshold": 0.3,
        },
    )

    obstacle_difficulty = CurrTerm(
        func=mdp.ObstacleCurriculum,
        params={
            "command_name": "goal_command",
            "initial_count": 2,
            "max_count": 8,
            "step_size": 2,
            "success_threshold": 10,
            "goal_threshold": 0.3,
        },
    )

@configclass
class CommandsCfg_ProcScene:
    """Commands for ProcScene — tighter goal range to fit within rooms (4-6m)."""
    goal_command = mdp.GoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 15.0),
        debug_vis=True,
        goal_range=mdp.GoalCommandCfg.Ranges(pos_x=(-2.5, 2.5), pos_y=(-2.5, 2.5)),
    )


@configclass
class RewardsCfg_ProcScene:
    """Rewards for ProcScene — uses net_forces_w collision instead of force_matrix_w."""
    # --- Primary task signal (dense) ---
    goal_progress = RewTerm(func=mdp.goal_progress_reward, weight=2.0, params={"command_name": "goal_command"})
    goal_proximity = RewTerm(func=mdp.goal_proximity_reward, weight=1.5, params={"command_name": "goal_command", "sigma": 0.3})
    # --- Sparse completion bonus ---
    goal_reached = RewTerm(func=mdp.goal_reached_reward, weight=10.0, params={"threshold": 0.3, "command_name": "goal_command"})
    # --- Heading (low weight — mecanum can strafe, heading is secondary) ---
    heading_alignment = RewTerm(func=mdp.heading_to_goal_reward, weight=0.05, params={"command_name": "goal_command"})
    # --- Collision avoidance (net_forces_w — works with any scene geometry) ---
    collision = RewTerm(func=mdp.collision_penalty_net, weight=-5.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
    collision_sustained = RewTerm(func=mdp.collision_sustained_penalty_net, weight=-2.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
    # --- Slow down near goal ---
    speed_near_goal = RewTerm(func=mdp.speed_near_goal_penalty, weight=-0.3, params={"command_name": "goal_command", "distance_threshold": 0.8})
    # --- Regularization ---
    energy_penalty = RewTerm(func=mdp.energy_penalty, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-0.05)


# Structural event for ProcScene: tighter robot spawn to stay in navigable core
_RESET_ROBOT_PROCSCENE = EventTerm(
    func=mdp.reset_robot_state,
    mode="reset",
    params={"pose_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "yaw": (-math.pi, math.pi)}},
)


@configclass
class EventsCfg_ProcScene_Realistic:
    """Realistic DR for ProcScene — no obstacle randomization, tighter spawn."""
    reset_robot = _RESET_ROBOT_PROCSCENE
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
class EventsCfg_ProcScene_Robust:
    """Robust DR for ProcScene — no obstacle randomization, tighter spawn."""
    reset_robot = _RESET_ROBOT_PROCSCENE
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
class CurriculumCfg_ProcScene:
    """Curriculum for ProcScene — goal distance only, no obstacle count."""
    goal_distance = CurrTerm(
        func=mdp.GoalDistanceCurriculum,
        params={
            "command_name": "goal_command",
            "initial_range": 1.5,
            "max_range": 3.0,
            "step_size": 0.5,
            "success_threshold": 5,
            "goal_threshold": 0.3,
        },
    )



# =============================================================================
# Environment Configurations - 18 Variants (9 configs × Train/Play)
# =============================================================================

# -----------------------------------------------------------------------------
# IDEAL: No noise, no motor dynamics (debugging/baselines)
# -----------------------------------------------------------------------------

@configclass
class StraferNavEnvCfg(ManagerBasedRLEnvCfg):
    """Ideal Full (RGB+Depth) - baseline for debugging."""
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg_Ideal = ActionsCfg_Ideal()
    observations: ObsCfg_Full_Ideal = ObsCfg_Full_Ideal()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_Ideal = EventsCfg_Ideal()
    curriculum: CurriculumCfg = CurriculumCfg()
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
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg_Ideal = ActionsCfg_Ideal()
    observations: ObsCfg_Depth_Ideal = ObsCfg_Depth_Ideal()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_Ideal = EventsCfg_Ideal()
    curriculum: CurriculumCfg = CurriculumCfg()
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
    scene: StraferSceneCfg_NoCam = StraferSceneCfg_NoCam(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg_Ideal = ActionsCfg_Ideal()
    observations: ObsCfg_NoCam_Ideal = ObsCfg_NoCam_Ideal()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_Ideal = EventsCfg_Ideal()
    curriculum: CurriculumCfg = CurriculumCfg()
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
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_Full_Realistic = ObsCfg_Full_Realistic()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_Realistic = EventsCfg_Realistic()
    curriculum: CurriculumCfg = CurriculumCfg()
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
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_Depth_Realistic = ObsCfg_Depth_Realistic()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_Realistic = EventsCfg_Realistic()
    curriculum: CurriculumCfg = CurriculumCfg()
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


@configclass
class StraferNavEnvCfg_Real_NoCam(ManagerBasedRLEnvCfg):
    """Realistic Proprioceptive-only - fast training with realistic dynamics."""
    scene: StraferSceneCfg_NoCam = StraferSceneCfg_NoCam(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_NoCam_Realistic = ObsCfg_NoCam_Realistic()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_Realistic = EventsCfg_Realistic()
    curriculum: CurriculumCfg = CurriculumCfg()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class StraferNavEnvCfg_Real_NoCam_PLAY(StraferNavEnvCfg_Real_NoCam):
    """Play/eval config for Realistic Proprioceptive-only."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


# -----------------------------------------------------------------------------
# ROBUST: Aggressive noise + dynamics for stress-testing
# -----------------------------------------------------------------------------

@configclass
class StraferNavEnvCfg_Robust(ManagerBasedRLEnvCfg):
    """Robust Full (RGB+Depth) - aggressive noise for worst-case robustness."""
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_Full_Robust = ObsCfg_Full_Robust()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_Robust = EventsCfg_Robust()
    curriculum: CurriculumCfg = CurriculumCfg()
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


@configclass
class StraferNavEnvCfg_Robust_Depth(ManagerBasedRLEnvCfg):
    """Robust Depth-only - aggressive noise for worst-case robustness."""
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_Depth_Robust = ObsCfg_Depth_Robust()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_Robust = EventsCfg_Robust()
    curriculum: CurriculumCfg = CurriculumCfg()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class StraferNavEnvCfg_Robust_Depth_PLAY(StraferNavEnvCfg_Robust_Depth):
    """Play/eval config for Robust Depth-only."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


@configclass
class StraferNavEnvCfg_Robust_NoCam(ManagerBasedRLEnvCfg):
    """Robust Proprioceptive-only - aggressive noise for worst-case robustness."""
    scene: StraferSceneCfg_NoCam = StraferSceneCfg_NoCam(num_envs=4096, env_spacing=8.0)
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_NoCam_Robust = ObsCfg_NoCam_Robust()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_Robust = EventsCfg_Robust()
    curriculum: CurriculumCfg = CurriculumCfg()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0


@configclass
class StraferNavEnvCfg_Robust_NoCam_PLAY(StraferNavEnvCfg_Robust_NoCam):
    """Play/eval config for Robust Proprioceptive-only."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50


# =============================================================================
# PROC-SCENE: Procedural scene geometry
#
# These variants use StraferSceneCfg_ProcScene (Infinigen room USD scenes)
# instead of box obstacles. Uses net_forces_w for collision detection,
# tighter spawn/goal ranges to fit within rooms, and no obstacle curriculum.
# =============================================================================

@configclass
class StraferNavEnvCfg_Real_ProcDepth(ManagerBasedRLEnvCfg):
    """Realistic Depth with procedural Infinigen scene geometry."""
    scene: StraferSceneCfg_ProcScene = StraferSceneCfg_ProcScene(num_envs=24, env_spacing=8.0)
    actions: ActionsCfg_Realistic = ActionsCfg_Realistic()
    observations: ObsCfg_Depth_Realistic = ObsCfg_Depth_Realistic()
    commands: CommandsCfg_ProcScene = CommandsCfg_ProcScene()
    rewards: RewardsCfg_ProcScene = RewardsCfg_ProcScene()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_ProcScene_Realistic = EventsCfg_ProcScene_Realistic()
    curriculum: CurriculumCfg_ProcScene = CurriculumCfg_ProcScene()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0
        self.scene.scene_geometry.spawn.usd_path = _get_scene_usd_paths()


@configclass
class StraferNavEnvCfg_Real_ProcDepth_PLAY(StraferNavEnvCfg_Real_ProcDepth):
    """Play/eval config for Realistic ProcDepth."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8


@configclass
class StraferNavEnvCfg_Robust_ProcDepth(ManagerBasedRLEnvCfg):
    """Robust Depth with procedural Infinigen scene geometry."""
    scene: StraferSceneCfg_ProcScene = StraferSceneCfg_ProcScene(num_envs=24, env_spacing=8.0)
    actions: ActionsCfg_Robust = ActionsCfg_Robust()
    observations: ObsCfg_Depth_Robust = ObsCfg_Depth_Robust()
    commands: CommandsCfg_ProcScene = CommandsCfg_ProcScene()
    rewards: RewardsCfg_ProcScene = RewardsCfg_ProcScene()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg_ProcScene_Robust = EventsCfg_ProcScene_Robust()
    curriculum: CurriculumCfg_ProcScene = CurriculumCfg_ProcScene()
    seed: int = 42

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = 4
        self.decimation = 4
        self.episode_length_s = 20.0
        self.scene.scene_geometry.spawn.usd_path = _get_scene_usd_paths()


@configclass
class StraferNavEnvCfg_Robust_ProcDepth_PLAY(StraferNavEnvCfg_Robust_ProcDepth):
    """Play/eval config for Robust ProcDepth."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8


