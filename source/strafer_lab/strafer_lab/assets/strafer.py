"""Configuration for the Gobilda Strafer mecanum wheel robot.

The Strafer chassis is a 4-wheel mecanum drive platform with:
- 4 mecanum wheels (96mm diameter), each with 10 rollers
- 5203 Series Yellow Jacket motors (19.2:1 ratio, 312 RPM output)
- 1:1 miter gears (90° power transfer)
- Intel RealSense D555 camera (optional sensor)

Physical specifications:
- Chassis: 432mm (L) x 360mm (W)
- Width with wheels: 413.2mm (center-to-center of wheel contact patches)
- Track width (wheel center-to-center): 413.2mm
- Wheel base (front-to-rear center-to-center): 336mm
- Wheel diameter: 96mm, width 38mm
- Motor RPM: 312 (after 19.2:1 gearbox)
- Max wheel angular velocity: 32.67 rad/s

Reference: https://www.gobilda.com/strafer-chassis-kit/
"""

from pathlib import Path
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the physics-rigged USD asset
_STRAFER_USD_PATH = str(Path(__file__).parents[4] / "Assets" / "strafer" / "3209-0001-0006-physics.usd")

# ============================================================
# Motor specifications (5203 Yellow Jacket 19.2:1 @ 12V)
# ============================================================
# GoBilda specs list no-load speed and stall torque at 12V.
# Reference: https://www.gobilda.com/5203-series-yellow-jacket-planetary-gear-motor-19-2-1-ratio-24mm-length-8mm-rex-shaft-312-rpm-3-3-5v-encoder/
_RPM_TO_RAD_S = 2.0 * math.pi / 60.0
_KGCM_TO_NM = 0.0980665

_MOTOR_NO_LOAD_RPM = 312.0
_MOTOR_NO_LOAD_RAD_S = _MOTOR_NO_LOAD_RPM * _RPM_TO_RAD_S

_MOTOR_STALL_TORQUE_KGCM = 24.3
_MOTOR_STALL_TORQUE_NM = _MOTOR_STALL_TORQUE_KGCM * _KGCM_TO_NM

# Damping gain for velocity control (kd).  PhysX implicit solver is
# unconditionally stable, so a high gain gives tight velocity tracking
# without the discrete-time overshoot that plagued the DCMotorCfg.
_MOTOR_DAMPING = 1000.0


STRAFER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_STRAFER_USD_PATH,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=16,
        ),
        activate_contact_sensors=True,
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
        # Mecanum wheel motors — velocity controlled via implicit PD
        # 5203 Yellow Jacket motors @ 19.2:1 ratio through 1:1 miter gears
        # Joint names: wheel_1_drive, wheel_2_drive, wheel_3_drive, wheel_4_drive
        #
        # Uses ImplicitActuatorCfg instead of DCMotorCfg because the mecanum
        # wheel inertia is very small (~0.001-0.01 kg⋅m²).  DCMotorCfg's
        # explicit discrete-time PD overshoots and oscillates at any damping
        # that gives useful velocity tracking (damping*dt/I >> 1).  The
        # implicit solver integrates the PD continuously and is unconditionally
        # stable.
        "wheel_drives": ImplicitActuatorCfg(
            joint_names_expr=["wheel_[1-4]_drive"],
            effort_limit_sim=_MOTOR_STALL_TORQUE_NM,
            velocity_limit_sim=_MOTOR_NO_LOAD_RAD_S,
            stiffness=0.0,          # pure velocity control (D-only)
            damping=_MOTOR_DAMPING,  # kd for velocity tracking
        ),
        # Roller joints - passive (no drive, free spinning)
        # 10 rollers per wheel (40 total), named: wheel_N_roller_0 through wheel_N_roller_9
        "roller_bearings": ImplicitActuatorCfg(
            joint_names_expr=["wheel_[1-4]_roller_[0-9]"],
            effort_limit_sim=0.0,       # No active drive
            velocity_limit_sim=1000.0,   # Allow free spinning (high limit)
            stiffness=0.0,
            damping=0.5,                # Low friction for free rolling; 0.01 caused GPU
                                        # TGS solver divergence at high env counts (>24),
                                        # flipping the robot via phantom constraint forces.
        ),
    },
)
