"""Environment configuration for Strafer navigation task.

This defines the full RL environment including:
- Scene configuration (robot, ground plane, lights)
- Observation space
- Action space  
- Reward functions
- Termination conditions
- Domain randomization events
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

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import TiledCameraCfg

# Import custom MDP functions
from . import mdp

# Import robot configuration
from strafer_lab.assets import STRAFER_CFG


@configclass
class StraferSceneCfg(InteractiveSceneCfg):
    """Configuration for the Strafer navigation scene."""

    # Ground plane - using terrain importer for flat ground
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

    # Strafer robot
    robot: ArticulationCfg = STRAFER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.8, 0.8, 0.8),
        ),
    )

    # Intel RealSense D555 depth camera
    # Specs: 87° x 58° FOV, 0.4m-6m depth range, RGB + Depth
    # Mounted on front of robot, facing forward (-Y direction in robot frame)
    d555_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/strafer/body_link/d555_camera",
        update_period=1.0 / 30.0,  # 30 Hz update rate
        height=60,   # Reduced resolution for RL (original: 720)
        width=80,    # Maintains ~4:3 aspect ratio
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,              # Calculated for ~87° HFOV
            horizontal_aperture=3.68,       # Sensor size for D555 FOV
            clipping_range=(0.4, 6.0),      # D555 depth range: 0.4m - 6m indoor
        ),
        offset=TiledCameraCfg.OffsetCfg(
            # Position: front of robot, raised slightly
            # Robot frame: -Y is forward, +Z is up
            pos=(0.0, -0.15, 0.10),  # 15cm forward, 10cm up from body_link
            # Rotation: camera facing forward (-Y direction)
            # ROS convention: camera +Z forward, so rotate 90° around X
            rot=(0.5, -0.5, 0.5, -0.5),  # Points camera in -Y direction (forward)
            convention="ros",
        ),
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP.
    
    Actions are normalized velocity commands [-1, 1] for [vx, vy, omega].
    The mecanum kinematics controller converts these to wheel angular velocities.
    
    Physical parameters are configured based on GoBilda Strafer chassis:
    - 5203 Yellow Jacket motors (19.2:1 ratio, 312 RPM output)
    - 96mm diameter mecanum wheels
    - 432mm x 360mm chassis
    """

    # Velocity commands: [vx, vy, omega] normalized to [-1, 1]
    # Joint order: wheel_1=FL, wheel_2=RL, wheel_3=RR, wheel_4=FR (counter-clockwise)
    # We specify joints explicitly to ensure consistent ordering
    wheel_velocities = mdp.MecanumWheelActionCfg(
        asset_name="robot",
        joint_names=["wheel_1_drive", "wheel_2_drive", "wheel_3_drive", "wheel_4_drive"],
        # Physical parameters (GoBilda Strafer defaults)
        wheel_radius=0.048,          # 96mm diameter / 2
        wheel_base=0.304,            # ~304mm axle-to-axle
        track_width=0.304,           # ~304mm left-right spacing
        motor_rpm=312.0,             # 5203 @ 19.2:1 ratio
        max_wheel_angular_vel=32.67, # 312 RPM in rad/s
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP - Full camera (RGB + Depth).
    
    Includes proprioceptive state and D555 RGB+depth camera observations.
    Total: 14 + 4800 (depth) + 14400 (RGB) = 19214 dimensions.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy network.
        
        Proprioceptive observations (14 dims):
        - base_lin_vel: (3,) - linear velocity in local frame
        - base_ang_vel: (3,) - angular velocity in local frame  
        - projected_gravity: (3,) - gravity vector in local frame
        - goal_position: (2,) - relative goal position
        - last_action: (3,) - previous action [vx, vy, omega]
        
        Camera observations:
        - depth_image: (4800,) - Flattened depth image (80 x 60)
        - rgb_image: (14400,) - Flattened RGB image (80 x 60 x 3)
        
        Total: 19214 dimensions
        """

        # Robot base velocity (local frame)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        # Robot orientation (projected gravity)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Goal position relative to robot
        goal_position = ObsTerm(
            func=mdp.goal_position_relative,
            params={"command_name": "goal_command"},
        )

        # Previous actions (for temporal consistency)
        last_action = ObsTerm(func=mdp.last_action)

        # D555 depth image (flattened) - 4800 dims (80x60)
        depth_image = ObsTerm(
            func=mdp.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("d555_camera"),
                "max_depth": 6.0,  # D555 max range
            },
        )

        # D555 RGB image (flattened) - 14400 dims (80x60x3)
        rgb_image = ObsTerm(
            func=mdp.rgb_image,
            params={
                "sensor_cfg": SceneEntityCfg("d555_camera"),
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for the critic network (if using asymmetric actor-critic).
        
        Contains same observations as policy but potentially with privileged info.
        """

        # Same as policy observations
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        goal_position = ObsTerm(
            func=mdp.goal_position_relative,
            params={"command_name": "goal_command"},
        )
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(
            func=mdp.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("d555_camera"),
                "max_depth": 6.0,
            },
        )
        rgb_image = ObsTerm(
            func=mdp.rgb_image,
            params={
                "sensor_cfg": SceneEntityCfg("d555_camera"),
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False  # No noise for critic
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    goal_command = mdp.GoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 15.0),
        debug_vis=True,
        goal_range=mdp.GoalCommandCfg.Ranges(
            pos_x=(-5.0, 5.0),
            pos_y=(-5.0, 5.0),
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Goal reaching reward
    goal_reached = RewTerm(
        func=mdp.goal_reached_reward,
        weight=10.0,
        params={"threshold": 0.5, "command_name": "goal_command"},
    )

    # Progress toward goal
    goal_progress = RewTerm(
        func=mdp.goal_progress_reward,
        weight=1.0,
        params={"command_name": "goal_command"},
    )

    # Heading alignment to goal
    heading_alignment = RewTerm(
        func=mdp.heading_to_goal_reward,
        weight=0.5,
        params={"command_name": "goal_command"},
    )

    # Energy penalty (discourage excessive motor use)
    energy_penalty = RewTerm(
        func=mdp.energy_penalty,
        weight=-0.01,
    )

    # Smoothness penalty (discourage jerky motion)
    action_smoothness = RewTerm(
        func=mdp.action_smoothness_penalty,
        weight=-0.1,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Robot flipped over
    robot_flipped = DoneTerm(
        func=mdp.robot_flipped,
        params={"threshold": 0.5},  # cos(60 degrees)
    )


@configclass
class EventsCfg:
    """Event/randomization terms for the MDP."""

    # Reset robot to random pose
    reset_robot = EventTerm(
        func=mdp.reset_robot_state,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "yaw": (-math.pi, math.pi),
            },
        },
    )

    # Randomize friction (domain randomization)
    randomize_friction = EventTerm(
        func=mdp.randomize_friction,
        mode="reset",
        params={"friction_range": (0.5, 1.5)},
    )


@configclass
class StraferNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Strafer navigation environment."""

    # Scene
    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=4.0)

    # MDP components
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    # Set seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        """Post initialization."""
        # Simulation settings
        self.sim.dt = 1.0 / 120.0  # 120 Hz physics
        self.sim.render_interval = 4  # Render every 4 physics steps (30 Hz)
        self.decimation = 4  # Policy runs at 30 Hz

        # Episode length
        self.episode_length_s = 20.0


@configclass
class StraferNavigationEnvCfg_PLAY(StraferNavigationEnvCfg):
    """Play/evaluation configuration with fewer environments."""

    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0
        # Disable observation noise for evaluation
        self.observations.policy.enable_corruption = False


# =============================================================================
# Alternative Environment Configurations
# =============================================================================


@configclass
class ProprioceptiveObservationsCfg:
    """Observation specifications without camera - proprioceptive only.
    
    Useful for faster training without vision or as a baseline.
    Total: 14 dimensions (no depth image)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioceptive-only observations."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        goal_position = ObsTerm(
            func=mdp.goal_position_relative,
            params={"command_name": "goal_command"},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class StraferSceneCfg_NoCamera(InteractiveSceneCfg):
    """Scene configuration without camera sensor for faster training."""

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
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.8, 0.8, 0.8),
        ),
    )


@configclass
class StraferNavigationEnvCfg_NoCam(ManagerBasedRLEnvCfg):
    """Navigation environment without camera - proprioceptive only.
    
    Faster training for baseline experiments or when vision is not needed.
    Observation space: 14 dimensions
    """

    scene: StraferSceneCfg_NoCamera = StraferSceneCfg_NoCamera(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ProprioceptiveObservationsCfg = ProprioceptiveObservationsCfg()
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
class StraferNavigationEnvCfg_NoCam_PLAY(StraferNavigationEnvCfg_NoCam):
    """Play configuration for no-camera environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


# =============================================================================
# Depth-Only Observations (4814 dims)
# =============================================================================


@configclass
class DepthOnlyObservationsCfg:
    """Observation specifications with depth camera only (no RGB).
    
    Total: 14 + 4800 = 4814 dimensions.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Depth-only camera observations."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        goal_position = ObsTerm(
            func=mdp.goal_position_relative,
            params={"command_name": "goal_command"},
        )
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(
            func=mdp.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("d555_camera"),
                "max_depth": 6.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for depth-only."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        goal_position = ObsTerm(
            func=mdp.goal_position_relative,
            params={"command_name": "goal_command"},
        )
        last_action = ObsTerm(func=mdp.last_action)
        depth_image = ObsTerm(
            func=mdp.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("d555_camera"),
                "max_depth": 6.0,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class StraferNavigationEnvCfg_DepthOnly(ManagerBasedRLEnvCfg):
    """Navigation environment with depth camera only.
    
    Observation space: 4814 dimensions (14 proprio + 4800 depth).
    Good balance between training speed and vision capability.
    """

    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: DepthOnlyObservationsCfg = DepthOnlyObservationsCfg()
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
class StraferNavigationEnvCfg_DepthOnly_PLAY(StraferNavigationEnvCfg_DepthOnly):
    """Play configuration for depth-only environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False


# =============================================================================
# RGB-Only Observations (14414 dims)
# =============================================================================


@configclass
class RGBOnlyObservationsCfg:
    """Observation specifications with RGB camera only (no depth).
    
    Total: 14 + 14400 = 14414 dimensions.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """RGB-only camera observations."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        goal_position = ObsTerm(
            func=mdp.goal_position_relative,
            params={"command_name": "goal_command"},
        )
        last_action = ObsTerm(func=mdp.last_action)
        rgb_image = ObsTerm(
            func=mdp.rgb_image,
            params={
                "sensor_cfg": SceneEntityCfg("d555_camera"),
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations for RGB-only."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        goal_position = ObsTerm(
            func=mdp.goal_position_relative,
            params={"command_name": "goal_command"},
        )
        last_action = ObsTerm(func=mdp.last_action)
        rgb_image = ObsTerm(
            func=mdp.rgb_image,
            params={
                "sensor_cfg": SceneEntityCfg("d555_camera"),
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class StraferNavigationEnvCfg_RGBOnly(ManagerBasedRLEnvCfg):
    """Navigation environment with RGB camera only.
    
    Observation space: 14414 dimensions (14 proprio + 14400 RGB).
    Useful for color/texture-based navigation tasks.
    """

    scene: StraferSceneCfg = StraferSceneCfg(num_envs=4096, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: RGBOnlyObservationsCfg = RGBOnlyObservationsCfg()
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
class StraferNavigationEnvCfg_RGBOnly_PLAY(StraferNavigationEnvCfg_RGBOnly):
    """Play configuration for RGB-only environment."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
