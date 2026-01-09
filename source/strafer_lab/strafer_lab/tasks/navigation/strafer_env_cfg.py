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


@configclass
class ActionsCfg:
    """Action terms for the MDP.
    
    Actions are wheel velocities for the 4 mecanum wheels.
    The mecanum kinematics controller converts these to robot motion.
    """

    # Wheel velocity commands: [front_left, front_right, rear_left, rear_right]
    # Joint names from setup_physics.py: wheel_1_drive, wheel_2_drive, etc.
    wheel_velocities = mdp.MecanumWheelActionCfg(
        asset_name="robot",
        joint_names=["wheel_[1-4]_drive"],
        scale=10.0,  # Scale factor for velocity commands
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy network."""

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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()


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
