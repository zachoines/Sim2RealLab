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

This module holds the shared building blocks — scene cfgs, per-realism
observation / action / event cfgs, commands / rewards / terminations /
curriculum, and the sizing constants and scene-setup helpers. The concrete
navigation environments are not hand-written here per matrix cell; they are
composed over the sensor / scene-source / realism axes in
``composed_env_cfg.py`` and registered in ``__init__.py``.
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
from .d555_cfg import (
    make_d555_camera_cfg,
    make_d555_imu_cfg,
    make_d555_perception_camera_cfg,
)


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


def _lightest_scene_usd_path() -> str:
    """Pick the lightest registered scene — the safe single-scene default.

    This is the guard against loading a massive scene by default. Unless the
    caller pins one with ``SCENE_USD`` / ``--scene-usd``, the heaviest scene can
    exhaust available memory and get the process killed at load time, so the
    default binds the lightest scene instead of the alphabetical first.

    Iterates the same valid set :func:`_get_scene_usd_paths` returns, resolves
    each top-level ``<scene>.usdc`` symlink to its real file, and returns the one
    whose resolved file is smallest on disk. Ties break by name (the input is
    already name-sorted, so ``min`` keeps the first entry on a tie) for
    determinism. The resolved ``.usdc`` file size alone is a sufficient proxy for
    scene weight — a full-quality room and a low-quality single room differ by an
    order of magnitude at the file level (~10 GB vs ~0.5 GB), so there is no need
    to sum each scene's texture tree.
    """
    paths = _get_scene_usd_paths()

    def _resolved_size(p: str) -> float:
        try:
            return float(Path(p).resolve().stat().st_size)
        except OSError:
            # Broken/unresolvable target: never prefer it over a real scene, but
            # still allow it as a last-resort fallback if it is the only entry.
            return float("inf")

    sized = [(_resolved_size(p), p) for p in paths]
    lightest_size, lightest = min(sized, key=lambda item: item[0])
    print(
        f"[scene-select] {Path(lightest).stem} ({lightest_size / 1024**3:.1f} GB)"
        f" — lightest of {len(paths)} registered scenes;"
        f" override with SCENE_USD / --scene-usd"
    )
    return lightest


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

    ``replicate_physics`` is disabled because the scene already explicitly
    shares ``/World/Room`` across all envs (Isaac Lab's per-env replication
    would be a no-op here) AND the prestartup ``lift_ground_plane_to_floor``
    event has to author USD-level changes before ``sim.reset()`` builds the
    PhysX collider tensors — that path is gated on ``replicate_physics=False``.
    """

    replicate_physics: bool = False

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


@configclass
class StraferSceneCfg_InfinigenPerception(InteractiveSceneCfg):
    """Infinigen scene with the high-resolution perception camera attached.

    Same Infinigen room geometry and global-prim layout as
    :class:`StraferSceneCfg_Infinigen` but with both cameras present:

    - ``d555_camera`` (80x45) — the RL policy camera, kept for parity with
      the training scenes so the deployed model sees the same input shape.
    - ``d555_camera_perception`` (640x360) — the perception data-collection
      camera used by Replicator bbox extraction, gamepad teleop capture, and
      the Isaac Sim ROS2 bridge. Consumers access it via
      ``env.scene["d555_camera_perception"].data.output["rgb"]`` /
      ``["distance_to_image_plane"]``.

    This scene is intentionally NOT used in RL training. At 640x360 Isaac
    Sim caps parallel envs at ~1-8 (vs. 256+ at 80x45), so this scene is
    reserved for data collection and ROS bridge work.

    See :class:`StraferSceneCfg_Infinigen` for why ``replicate_physics`` is
    disabled.
    """

    replicate_physics: bool = False

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

    # Policy camera (80x45) — kept so the perception env matches the
    # deployed observation shape; cheap to render alongside the perception
    # camera when only 1-8 envs are active.
    d555_camera: TiledCameraCfg = make_d555_camera_cfg(
        data_types=("rgb", "distance_to_image_plane"),
    )

    # Perception camera (640x360 RGB + depth) — the reason this scene exists.
    d555_camera_perception: TiledCameraCfg = make_d555_perception_camera_cfg()

    d555_imu: ImuCfg = make_d555_imu_cfg()

    # Procedural scene geometry as a global prim (shared across envs via
    # collision_group=-1). The USD path is populated in __post_init__ by
    # the env config so this class stays portable across Infinigen batches.
    scene_geometry: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Room",
        spawn=sim_utils.UsdFileCfg(usd_path=""),
        collision_group=-1,
    )

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

# Structural event for Infinigen: lift /World/ground to the loaded scene's
# floor height. Floor meshes have their colliders stripped at bake time
# (see scripts/postprocess_scene_usd.py) so robot collision flows entirely
# through this lifted plane. ``target_z`` is populated in __post_init__
# with ``floor_top_z - 0.002`` to keep the plane just below the visible
# Infinigen floor and avoid z-fighting.
#
# mode="prestartup" is required: PhysX builds collider tensors during
# ``sim.reset()`` from the USD prim transforms as they exist at that
# moment, so the translate must land before sim.reset(). Running at
# "startup" left the visual plane lifted but the collider stuck at z=0.
# prestartup is gated on ``replicate_physics=False`` (set on the
# Infinigen scene cfgs); that's a no-op for these scenes since they
# already share a single ``/World/Room`` with ``env_spacing=0`` and
# ``collision_group=-1``.
_LIFT_GROUND_INFINIGEN = EventTerm(
    func=mdp.lift_ground_plane_to_floor,
    mode="prestartup",
    params={"target_z": 0.0},  # populated in __post_init__
)

# Stamp opencvPinhole lens-distortion schema on the perception camera prim
# so the ROS 2 bridge's CameraInfo publisher takes the supported code path
# instead of the deprecated physicalDistortionModel fallback. Self-gates on
# the perception camera being present, so it's a no-op for the depth-only
# Infinigen variants that share these EventsCfg classes.
_STAMP_D555_OPENCV_PINHOLE = EventTerm(
    func=mdp.stamp_d555_perception_opencv_pinhole,
    mode="startup",
    params={},
)


@configclass
class EventsCfg_Infinigen_Realistic:
    """Realistic DR for Infinigen — no obstacle randomization, tighter spawn."""
    reset_robot = _RESET_ROBOT_INFINIGEN
    lift_ground = _LIFT_GROUND_INFINIGEN
    stamp_d555_opencv_pinhole = _STAMP_D555_OPENCV_PINHOLE
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
    lift_ground = _LIFT_GROUND_INFINIGEN
    stamp_d555_opencv_pinhole = _STAMP_D555_OPENCV_PINHOLE
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
# Shared runtime defaults, sizing constants, and scene-setup helpers
#
# Consumed by the composition root (composed_env_cfg.py) when it materializes
# a variant from the sensor / scene-source / realism axes.
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
    # PGS solver (solver_type=0) instead of the TGS default. TGS injects
    # spurious velocity into the near-massless, free-spinning mecanum rollers
    # (over-spinning them several-fold), which pumps a growing chassis bounce
    # at sustained high yaw rate. PGS keeps roller velocities physical and
    # removes the bounce at the native rate, with no substep-count penalty.
    # solver_type is not part of the hashed policy contract. Do not revert to
    # TGS without re-checking the high-yaw roller bounce.
    # Contact stabilization for the passive rollers stays on (ProcRoom
    # overrides cfg.sim.physics with its own PhysxCfg afterward).
    cfg.sim.physics = PhysxCfg(enable_stabilization=True, solver_type=0)


def _apply_play_num_envs(cfg: ManagerBasedRLEnvCfg, *, num_envs: int) -> None:
    """Shrink train configs for play/eval without changing other behavior."""
    cfg.scene.num_envs = num_envs


# Bridge/training robot spawn pool size: a bounded, spatially-strided subsample
# of the loaded scene's free in-room occupancy cells (the reset event and the
# goal command each sample one uniformly per reset). Bounded so the baked cfg
# stays small and deterministic, and spread across the room floor rather than
# clustered in one strip; roughly the prior floor-sample pool scale. A richer
# per-env spawn distribution is a follow-up — v1 ships one per-loaded-scene pool.
_INFINIGEN_SPAWN_POOL_SIZE = 256


def derive_infinigen_scene_spawn(
    scene_usd: Path | str, *, n: int | None = _INFINIGEN_SPAWN_POOL_SIZE
) -> list[list[float]]:
    """Occupancy-derived free-space spawn pool for ONE scene's USD (env-local).

    The shared bridge/training spawn derivation. Loads ONLY the named scene's
    occupancy sidecar and embedded room footprints, robot-radius-inflates the
    grid, seals it to the rooms, and returns free + in-room cells through the
    shared :func:`strafer_lab.tools.scene_connectivity.spawn_pool_from_occupancy`
    core — the *same* core the coverage driver's spawn selection uses. There is
    no cross-scene union: every point lies inside THIS scene's rooms, in the
    scene-authored (== env-local) frame the reset event expects before it adds
    the env origin. Used both at config time (:func:`_apply_infinigen_scene_setup`)
    and at the ``run_sim_in_the_loop --scene-usd`` runtime override so the bound
    config-default and an overridden scene derive spawn identically.

    Returns ``[]`` when the scene has no occupancy sidecar / room metadata yet
    (fresh/pre-metadata scenes, smoke tests) — the reset event then spawns at the
    env origin. Raises ``RuntimeError`` on a present-but-degenerate grid (no free
    in-room cell): the occupancy must be regenerated; it is never silently masked.
    """
    from strafer_lab.tools import scene_connectivity, scene_metadata_reader

    scene_usd = Path(scene_usd)
    try:
        occupancy = scene_connectivity.load_occupancy(
            scene_connectivity.scene_dir_for(scene_usd)
        )
    except (FileNotFoundError, OSError):
        return []
    try:
        rooms = scene_metadata_reader.load(scene_usd).get("rooms", [])
    except scene_metadata_reader.SceneMetadataError:
        return []
    if not rooms:
        return []
    free_space = scene_connectivity.occupancy_to_free_space(
        occupancy.grid, grid_res=occupancy.resolution_m
    )
    free_space = scene_connectivity.seal_free_space_to_rooms(
        free_space, rooms,
        origin_xy=occupancy.origin_xy, grid_res=occupancy.resolution_m,
    )
    return scene_connectivity.spawn_pool_from_occupancy(
        free_space, rooms, occupancy, n=n
    )


def _get_infinigen_active_scene_floor_top_z(scene_stem: str) -> float | None:
    """Return ``floor_top_z`` for a single scene by stem name, or None.

    Used by the ground-lift event which needs the *active* scene's floor
    height, not the pooled max across all scenes.
    """
    meta = _get_scenes_metadata()
    if not meta:
        return None
    scene_data = meta.get("scenes", {}).get(scene_stem)
    if not scene_data:
        return None
    return scene_data.get("floor_top_z")


# The scene's bright baked light emitters render with RTX auto-exposure off, so
# the recorded perception RGB clips to white. Enable RTX histogram auto-exposure
# (what the real D555 does); RGB-only, depth is unaffected. whiteScale is the
# exposure target (lower => darker), tuned on a production ceiling-on capture.
_INFINIGEN_RENDER_EXPOSURE_CARB: dict[str, object] = {
    "rtx.post.histogram.enabled": True,
    "rtx.post.histogram.whiteScale": 7.0,
}


def _apply_infinigen_render_exposure(cfg: ManagerBasedRLEnvCfg) -> None:
    """Merge the auto-exposure carb settings into the sim's RenderCfg."""
    existing = cfg.sim.render.carb_settings or {}
    cfg.sim.render.carb_settings = {**existing, **_INFINIGEN_RENDER_EXPOSURE_CARB}


def _apply_infinigen_scene_setup(cfg: ManagerBasedRLEnvCfg) -> None:
    """Bind ONE loaded scene's USD + its occupancy-derived spawn to an env cfg.

    The scene path is resolved from its symlink — USD's asset resolver
    anchors relative texture refs against the symlink location, not the
    target, and that breaks every ``./textures/*`` lookup.

    Spawn points, the robot spawn-z, and the ground-lift height are all pinned
    to the SINGLE scene this cfg loads (``_lightest_scene_usd_path()`` — the
    lightest registered scene, so the bare default does not exhaust memory on the
    heaviest room) — no cross-scene union / pooled-max, which would otherwise
    place the robot and goals at a non-loaded scene's coordinates. The
    ``run_sim_in_the_loop --scene-usd`` runtime override and the coverage driver
    re-derive the same way per overridden / loaded scene (see
    :func:`derive_infinigen_scene_spawn`).
    """
    scene_link = Path(_lightest_scene_usd_path())
    scene_path = scene_link.resolve()
    cfg.scene.scene_geometry.spawn.usd_path = str(scene_path)

    _apply_infinigen_render_exposure(cfg)

    spawn_points_xy = derive_infinigen_scene_spawn(scene_link)
    if spawn_points_xy:
        cfg.events.reset_robot.params["spawn_points_xy"] = spawn_points_xy
        cfg.commands.goal_command.spawn_points_xy = list(spawn_points_xy)

    active_floor_top_z = _get_infinigen_active_scene_floor_top_z(scene_link.stem)
    if active_floor_top_z is not None:
        floor_z = float(active_floor_top_z)
        # Robot spawn-z: wheel clearance above THIS scene's floor (was a pooled
        # MAX across all scenes, wrong for whichever single scene is loaded).
        cfg.events.reset_robot.params["spawn_z"] = floor_z + 0.1
        # Sit the ground plane 2 mm below the visible Infinigen floor: the gap
        # is imperceptible and prevents z-fighting between the two.
        cfg.events.lift_ground.params["target_z"] = floor_z - 0.002


@configclass
class _BaseStraferNavEnvCfg(ManagerBasedRLEnvCfg):
    """Shared runtime defaults for all navigation environment configs."""

    seed: int = 42

    def __post_init__(self):
        _apply_default_nav_runtime(self)


# Perception data-collection envs run at 640x360 and therefore cap parallel
# env count at 1-8. Start at 1 — the Isaac Sim ROS2 bridge and gamepad
# teleop are both single-env workflows. The play override below can bump
# this for batch captures.
_INFINIGEN_PERCEPTION_TRAIN_NUM_ENVS = 1


# =============================================================================
# PROCROOM: Procedural primitive rooms with GPU BFS solvability
#
# Lightweight primitive shapes (walls, furniture, clutter) generated at each
# episode reset. Supports 256+ envs with replicated physics. GPU BFS
# guarantees solvable layouts.
# =============================================================================

# Import procedural room palette builder + the planner inflation geometry the
# subgoal off-path corridor is sized from.
from .mdp.proc_room import GRID_RES, INFLATION_CELLS, build_proc_room_collection_cfg


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
    # RGB included alongside depth so the RTX renderer initializes the full
    # colour pipeline — required for viewport / video recording to produce
    # non-black frames even though observations only use depth.
    d555_camera: TiledCameraCfg = make_d555_camera_cfg(
        data_types=("rgb", "distance_to_image_plane"),
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


# --- ProcRoom subgoal-tracking building blocks ---
#
# The subgoal objective swaps the episode's task from "converge on a fixed
# goal pose" to "track a rolling subgoal along a planned path": same scenes,
# same observation/action/DR tiers, different command + reward + termination
# blocks. The command term is registered under the same ``goal_command`` name
# so the goal-shaped observation terms re-bind to the subgoal automatically.

# Maximum cross-track distance before the episode is cut. Shared by the
# off-path termination and its paired one-shot reward penalty.
#
# Sized to the planner's obstacle-inflation margin (INFLATION_CELLS * GRID_RES
# = the robot radius A* clears around obstacles), NOT a looser hand-picked
# value: at this cross-track the robot has reached the edge of the corridor
# the planner cleared, so "on-path" stays equivalent to "inside cleared free
# space." A wider tolerance would let a closely-tracking robot leave the
# cleared corridor while still counting as on-path. This is a corridor bound,
# not a collision guarantee — actual contacts are caught independently by the
# sustained-collision termination — and tight passages (doorways) can clear a
# narrower corridor than this maximum.
_SUBGOAL_MAX_OFF_PATH_M = INFLATION_CELLS * GRID_RES  # 0.3 m

# Lookahead-distance randomization bands (meters), centered on the nominal
# SUBGOAL_LOOKAHEAD_M. Both tiers randomize, following the same realistic-vs-
# robust convention as the other DR knobs (friction, mass, motor strength):
# realistic = a tight band, robust = a wider one — not fixed-vs-randomized.
# The deployed pure-pursuit lookahead is a not-yet-finalized tunable and Nav2's
# closest-point projection adds small realized jitter, so even the
# realistic/eval tier should train over a band rather than assume one exact
# distance. The training workflow is train-on-robust, evaluate-on-realistic.
_SUBGOAL_REAL_LOOKAHEAD_BAND = (0.9, 1.1)
_SUBGOAL_ROBUST_LOOKAHEAD_BAND = (0.7, 1.3)


@configclass
class CommandsCfg_ProcRoom_Subgoal:
    """Commands for ProcRoom subgoal tracking — one planned path per episode."""
    goal_command = mdp.SubgoalCommandCfg(
        asset_name="robot",
        # One path per episode; episodes end via path_complete / off-path /
        # timeout terminations, mirroring the single-goal ProcRoom setup.
        resampling_time_range=(1.0e6, 1.0e6),
        # Push goals farther than the 1 m default so episodes carry a longer
        # path to track (the disc-inflated free space makes >=2 m endpoints
        # readily reachable). Small rooms fall back to the farthest reachable
        # point when no candidate clears this distance.
        min_goal_distance=2.0,
        # Dwell-based success: the path completes only after the robot parks —
        # holds inside dwell_radius_m at/under dwell_speed_max_m_s for
        # dwell_steps consecutive control steps — so the sparse completion bonus
        # pays for stopping at the goal, not touching it at speed. Surfaced here
        # as the training sweep surface (values match the SubgoalCommandCfg
        # defaults); ~0.33 s of dwell at 30 Hz control.
        dwell_radius_m=0.3,
        dwell_speed_max_m_s=0.1,
        dwell_steps=10,
        debug_vis=True,
    )


@configclass
class RewardsCfg_ProcRoom_Subgoal:
    """Path-tracking rewards for the ProcRoom subgoal objective.

    Weights are starting points for the training run, not tuned constants:
    along-track progress mirrors the goal-progress scale, the sparse
    completion bonus must dominate any shaping residual, and the collision /
    backward-motion / clearance terms carry over from the goal-directed
    ProcRoom setup unchanged.
    """
    # --- Primary task signal: track the planned path ---
    along_track_progress = RewTerm(func=mdp.path_along_track_progress, weight=10.0, params={"command_name": "goal_command"})
    cross_track_error = RewTerm(func=mdp.path_cross_track_error, weight=-2.0, params={"command_name": "goal_command"})
    # --- Sparse completion bonus (LARGE — must dominate any shaping residual) ---
    path_complete = RewTerm(func=mdp.path_complete_reward, weight=200.0, params={"command_name": "goal_command"})
    # --- One-shot penalty paired with the off-path termination ---
    off_path = RewTerm(
        func=mdp.off_path_divergence_penalty, weight=-50.0,
        params={"command_name": "goal_command", "max_off_path_m": _SUBGOAL_MAX_OFF_PATH_M},
    )
    # --- Collision avoidance (carried over from the goal-directed setup) ---
    collision = RewTerm(func=mdp.collision_penalty_net, weight=-10.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
    collision_sustained = RewTerm(func=mdp.collision_sustained_penalty_net, weight=-5.0, params={"sensor_cfg": SceneEntityCfg("contact_sensor"), "threshold": 1.0})
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
    # --- Secondary signals (disabled by default; enable when tuning) ---
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=0.0)
    energy_penalty = RewTerm(func=mdp.energy_penalty, weight=0.0)


@configclass
class RewardsCfg_ProcRoom_Subgoal_Depth(RewardsCfg_ProcRoom_Subgoal):
    """Path-tracking rewards for the DEPTH_SUBGOAL objective.

    Inherits the NOCAM-subgoal reward set verbatim — path tracking, the
    dwell-gated completion bonus, the off-path penalty, the contact-force
    collision terms, and the geometric ``obstacle_proximity`` shaping — and
    *adds* the depth-sensed ``depth_obstacle_proximity``.

    The two proximity terms coexist by design and are not double-counting: they
    read different things. ``obstacle_proximity`` reads ground-truth primitive
    geometry — ideal shaping toward clearance the sensor can't provide.
    ``depth_obstacle_proximity`` reads the depth camera the policy actually
    sees — teaching it to react to *sensed* obstacles, including late-arriving
    ones the planned path drives through. Depth makes obstacles observable, so
    (unlike NOCAM_SUBGOAL, where the policy is blind and the analogous gradient
    would be noise) this proximity gradient is real signal.

    **Shipped inert (weight 0.0).** The depth term is kept wired but zero-
    weighted, so it is a native no-op (Isaac Lab's RewardManager skips zero-
    weight terms before calling the func) and re-enabling it is a one-float
    flip. It ships off because the current ProcRoom env structurally starves
    it: the A* planner line-of-sight-shortcuts paths straight and inflates by
    the robot radius, so a planned path is a pre-cleared near-straight shot and
    the forward camera rarely faces an on-path obstacle to steer around — the
    validated DEPTH_SUBGOAL win is that the depth *observation* tracks the path
    better than proprioception, not reactive sensed-obstacle avoidance. The
    term stays as a distinct subclass (not folded into the base) so the
    NOCAM_SUBGOAL reward contract stays byte-identical.

    Weight discipline (for the re-enable): the depth term must stay a *shaping*
    signal strictly inside the task economics — planned paths legitimately pass
    within the penalty's threshold of primitives and walls, and the off-path /
    completion one-shots realize at ``weight * step_dt``, so an over-weighted
    dense penalty makes terminating early return-optimal and training collapses
    onto an off-path bail-out instead of path completion. Re-enable is a
    warm-start onto the converged depth tracker at a modest weight, in an env
    that guarantees on-path obstacles and widens the off-path corridor so
    deviating to avoid is admissible — retune against the episode-return
    arithmetic, not in isolation.
    """

    depth_obstacle_proximity = RewTerm(
        func=mdp.depth_obstacle_proximity_penalty,
        weight=0.0,  # shipped inert; re-enable in the hardened env (see docstring)
        params={
            "sensor_cfg": SceneEntityCfg("d555_camera"),
            "distance_threshold": 1.0,
            "saturation_depth": 0.3,
            "epsilon": 0.1,
            "max_depth": DEPTH_MAX,
            "floor_margin": 0.07,
        },
    )


@configclass
class TerminationsCfg_ProcRoom_Subgoal(TerminationsCfg):
    """Terminations for the ProcRoom subgoal objective.

    Episodes end on path completion, off-path divergence, sustained
    collision (ProcRoom-tight threshold), robot flip, or timeout.
    """

    path_complete = DoneTerm(
        func=mdp.path_complete,
        params={"command_name": "goal_command"},
    )
    off_path_divergence = DoneTerm(
        func=mdp.off_path_divergence,
        params={"command_name": "goal_command", "max_off_path_m": _SUBGOAL_MAX_OFF_PATH_M},
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
        # PGS solver — see _apply_default_nav_runtime. Keeps the passive
        # rollers from over-spinning (TGS velocity noise) and bouncing the
        # chassis at high yaw rate. This override replaces the default-nav
        # PhysxCfg, so it must carry solver_type=0 too.
        solver_type=0,
    )


# --- ProcRoom environment configs ---


