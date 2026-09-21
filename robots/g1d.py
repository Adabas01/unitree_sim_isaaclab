"""Parked G1-D articulation using the locally imported Unitree + Dex1 USD."""

from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import isaaclab.sim as sim_utils


USD_PATH = Path(__file__).resolve().parents[1] / "assets/robots/g1d-dex1/usd/g1d_dex1.usd"
from .g1d_joints import ARM_JOINTS, FINGER_JOINTS, GRIPPER_CLOSED, GRIPPER_OPEN


def robot_config(settings):
    if not USD_PATH.is_file():
        raise FileNotFoundError(
            f"Import the G1-D URDF first; missing {USD_PATH}. "
            "See tasks/g1d_tasks/sort_shapes/README.md"
        )
    q = dict(zip(ARM_JOINTS, settings.get("arm_positions", [0.0] * len(ARM_JOINTS)), strict=True))
    q.update({name: GRIPPER_OPEN for name in FINGER_JOINTS})
    q.update({
        "Left_Wheel_Joint": 0.0, "Right_Wheel_Joint": 0.0,
        "LZ_mt_Joint": settings["lift_positions"][0], "LZ_it_Joint": settings["lift_positions"][1],
        # Despite its name, Yaw_Joint is the URDF's joint about the Y (pitch) axis.
        "Yaw_Joint": settings["torso_pitch"], "torso_Joint": settings["torso_yaw"],
    })
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=0.5,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True, enabled_self_collisions=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, settings["base_height"]), joint_pos=q, joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint"],
                stiffness=160.0, damping=8.0, effort_limit_sim=25.0, velocity_limit_sim=3.0,
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_.*_joint"],
                stiffness=40.0, damping=2.0, effort_limit_sim=5.0, velocity_limit_sim=3.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=FINGER_JOINTS,
                stiffness=1000.0, damping=20.0, effort_limit_sim=20.0, velocity_limit_sim=0.2,
            ),
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["LZ_mt_Joint", "LZ_it_Joint"],
                stiffness=100000.0, damping=1000.0, effort_limit_sim=500.0, velocity_limit_sim=0.05,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["Yaw_Joint", "torso_Joint"], stiffness=2000.0, damping=80.0,
                effort_limit_sim={"Yaw_Joint": 80.0, "torso_Joint": 30.0}, velocity_limit_sim=1.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["Left_Wheel_Joint", "Right_Wheel_Joint"],
                stiffness=100.0, damping=10.0, effort_limit_sim=8.0, velocity_limit_sim=1.0,
            ),
        },
    )
