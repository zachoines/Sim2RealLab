"""Configuration for the Gobilda Strafer mecanum wheel robot.

The Strafer chassis is a 4-wheel mecanum drive platform with:
- 4 mecanum wheels (96mm diameter), each with 10 rollers
- 5203 Series Yellow Jacket motors (19.2:1 ratio, 312 RPM output)
- 1:1 miter gears (90° power transfer)
- Intel RealSense D555 camera (optional sensor)

Physical specifications:
- Chassis: 432mm (L) x 360mm (W)
- Width with wheels: 451.2mm
- Wheel diameter: 96mm
- Motor RPM: 312 (after 19.2:1 gearbox)
- Max wheel angular velocity: 32.67 rad/s

Reference: https://www.gobilda.com/strafer-chassis-kit/
"""

from pathlib import Path
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the physics-rigged USD asset
_STRAFER_USD_PATH = str(Path(__file__).parents[4] / "Assets" / "3209-0001-0006-v6" / "3209-0001-0006-physics.usd")

# ============================================================
# Motor specifications (5203 Yellow Jacket 19.2:1 @ 12V)
# ============================================================
# GoBilda specs list no-load speed and stall torque at 12V.
# Use a DC motor torque-speed curve (explicit actuator model).
_RPM_TO_RAD_S = 2.0 * math.pi / 60.0
_KGCM_TO_NM = 0.0980665

_MOTOR_NO_LOAD_RPM = 312.0
_MOTOR_NO_LOAD_RAD_S = _MOTOR_NO_LOAD_RPM * _RPM_TO_RAD_S

_MOTOR_STALL_TORQUE_KGCM = 24.3
_MOTOR_STALL_TORQUE_NM = _MOTOR_STALL_TORQUE_KGCM * _KGCM_TO_NM

# GoBilda does not publish continuous torque; use a conservative fraction.
# Set to 1.0 if you want continuous torque equal to stall torque.
_MOTOR_CONTINUOUS_TORQUE_FRACTION = 0.5
_MOTOR_CONTINUOUS_TORQUE_NM = _MOTOR_STALL_TORQUE_NM * _MOTOR_CONTINUOUS_TORQUE_FRACTION

_MOTOR_DAMPING = 10.0


STRAFER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_STRAFER_USD_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),  # Spawn slightly above ground
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        joint_pos={
            # All joints start at rest
            ".*": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        # Mecanum wheel motors - velocity controlled via DC motor model
        # 5203 Yellow Jacket motors @ 19.2:1 ratio through 1:1 miter gears
        # Joint names: wheel_1_drive, wheel_2_drive, wheel_3_drive, wheel_4_drive
        "wheel_drives": DCMotorCfg(
            joint_names_expr=["wheel_[1-4]_drive"],
            # DC motor torque-speed curve parameters (joint/output shaft)
            saturation_effort=_MOTOR_STALL_TORQUE_NM,
            effort_limit=_MOTOR_CONTINUOUS_TORQUE_NM,
            velocity_limit=_MOTOR_NO_LOAD_RAD_S,
            stiffness=0.0,          # velocity control (D-only)
            damping=_MOTOR_DAMPING, # motor damping coefficient
        ),
        # Roller joints - passive (no drive, free spinning)
        # 10 rollers per wheel (40 total), named: wheel_N_roller_0 through wheel_N_roller_9
        "roller_bearings": ImplicitActuatorCfg(
            joint_names_expr=["wheel_[1-4]_roller_[0-9]"],
            effort_limit_sim=0.0,       # No active drive
            velocity_limit_sim=100.0,   # Allow free spinning (high limit)
            stiffness=0.0,
            damping=0.01,               # Very low friction for free rolling
        ),
    },
)
"""Configuration for the Gobilda Strafer mecanum wheel robot.

The robot has two actuator groups:
- **wheel_drives**: The 4 main wheel motors (velocity controlled)
- **roller_bearings**: The passive rollers on each wheel (40 total, free spinning)

Control mode: Velocity control on wheel drives
- Set target velocities for each wheel to achieve desired motion
- Mecanum kinematics allow omnidirectional movement

Example usage::

    from strafer_lab.assets import STRAFER_CFG
    
    robot = Articulation(cfg=STRAFER_CFG)
    robot.write_joint_velocity_target_to_sim(velocities, joint_ids=wheel_joint_ids)
"""


# Play/evaluation configuration with reduced environment count
STRAFER_CFG_PLAY = STRAFER_CFG.copy()
STRAFER_CFG_PLAY.spawn.articulation_props.enabled_self_collisions = False
