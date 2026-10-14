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
    aligned = settings.get("controller_alignment", False)
    arm_gains = (80.0, 3.0) if aligned else (160.0, 8.0)
    arm_velocity = 37.0 if aligned else 3.0
    wrist_damping = 1.5 if aligned else 2.0
    wrist_effort = ({".*_wrist_roll_joint": 25.0, ".*_wrist_(pitch|yaw)_joint": 5.0}
                     if aligned else 5.0)
    wrist_velocity = ({".*_wrist_roll_joint": 37.0, ".*_wrist_(pitch|yaw)_joint": 22.0}
                       if aligned else 3.0)
    torso_stiffness = {"Yaw_Joint": 2000.0, "torso_Joint": 300.0} if aligned else 2000.0
    torso_damping = {"Yaw_Joint": 80.0, "torso_Joint": 3.0} if aligned else 80.0
    torso_velocity = {"Yaw_Joint": 1.0, "torso_Joint": 5.0} if aligned else 1.0
    q = dict(zip(ARM_JOINTS, settings.get("arm_positions", [0.0] * len(ARM_JOINTS)), strict=True))
    q.update({name: GRIPPER_OPEN for name in FINGER_JOINTS})
    q.update({
        "Left_Wheel_Joint": 0.0, "Right_Wheel_Joint": 0.0,
        "LZ_mt_Joint": settings["lift_positions"][0], "LZ_it_Joint": settings["lift_positions"][1],
        # Despite its name, Yaw_Joint is the URDF's joint about the Y (pitch) axis.
        "Yaw_Joint": settings["torso_pitch"], "torso_Joint": settings["torso_yaw"],
    })
    gripper = settings["gripper_actuator"]
    collisions = settings["collisions"]
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=collisions["contact_offset_m"],
                rest_offset=collisions["rest_offset_m"],
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=collisions["self_collisions"],
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
                stiffness=arm_gains[0], damping=arm_gains[1],
                effort_limit_sim=25.0, velocity_limit_sim=arm_velocity,
            ),
            "wrists": ImplicitActuatorCfg(
                joint_names_expr=[".*_wrist_.*_joint"],
                stiffness=40.0, damping=wrist_damping,
                effort_limit_sim=wrist_effort, velocity_limit_sim=wrist_velocity,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=FINGER_JOINTS,
                stiffness=gripper["stiffness_n_m"],
                damping=gripper["damping_n_s_m"],
                effort_limit_sim=gripper["effort_limit_n"],
                velocity_limit_sim=gripper["velocity_limit_m_s"],
            ),
            "lift": ImplicitActuatorCfg(
                joint_names_expr=["LZ_mt_Joint", "LZ_it_Joint"],
                stiffness=100000.0, damping=1000.0, effort_limit_sim=500.0, velocity_limit_sim=0.05,
            ),
            "torso": ImplicitActuatorCfg(
                joint_names_expr=["Yaw_Joint", "torso_Joint"],
                stiffness=torso_stiffness, damping=torso_damping,
                effort_limit_sim={"Yaw_Joint": 80.0, "torso_Joint": 30.0},
                velocity_limit_sim=torso_velocity,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["Left_Wheel_Joint", "Right_Wheel_Joint"],
                stiffness=100.0, damping=10.0, effort_limit_sim=8.0, velocity_limit_sim=1.0,
            ),
        },
    )
