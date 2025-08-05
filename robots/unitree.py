# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Configuration for Unitree robots."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import os
project_root = os.environ.get("PROJECT_ROOT")
G129_CFG_WITH_DEX3_BASE_FIX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{project_root}/assets/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fix.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,

        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            # waist joints
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            
            # arms joints
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            
            # fingers joints
            "left_hand_index_0_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            
            "right_hand_index_0_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", 
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint", 
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=0.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            damping={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=None,
            velocity_limit=None,
             stiffness={  # increase the stiffness (kp)
                 ".*_shoulder_.*_joint": 300.0,
                 ".*_elbow_joint": 400.0,
                 ".*_wrist_.*_joint": 400.0,
            },
             damping={    # increase the damping (kd)
                 ".*_shoulder_.*_joint": 3.0,
                 ".*_elbow_joint": 2.5,
                 ".*_wrist_.*_joint": 2.5,
             },
            armature=None,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_index_.*_joint",
                ".*_hand_middle_.*_joint",
                ".*_hand_thumb_.*_joint"
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*": 100.0,
            },
            damping={
                ".*": 10.0,
            },
            armature={
                ".*": 0.1
            },
        ),
    },
)




G129_CFG_WITH_DEX1_BASE_FIX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{project_root}/assets/robots/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            # waist joints
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            
            # arms joints
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            
            # fingers joints
            "left_hand_Joint1_1": 0.0,
            "left_hand_Joint2_1": 0.0,
            "right_hand_Joint1_1": 0.0,
            "right_hand_Joint2_1": 0.0,
            
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", 
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint", 
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=0.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            damping={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=None,
            velocity_limit=None,
             stiffness={  # increase the stiffness (kp)
                 ".*_shoulder_.*_joint": 25.0,
                 ".*_elbow_joint": 50.0,
                 ".*_wrist_.*_joint": 40.0,
            },
             damping={    # increase the damping (kd)
                 ".*_shoulder_.*_joint": 2.0,
                 ".*_elbow_joint": 2.0,
                 ".*_wrist_.*_joint": 2.0,
             },
            armature=None,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hand_Joint1_1",
                "left_hand_Joint2_1",
                "right_hand_Joint1_1",
                "right_hand_Joint2_1",
            ],
            effort_limit=None,  # increase the torque limit
            velocity_limit=None,  # set the velocity limit to 0
            stiffness=800.0,    # increase the stiffness (kp)
            damping=3.0,        # increase the damping (kd)
            friction=200.0,
            armature=None,
        ),

    },
)



G129_CFG_WITH_INSPIRE_HAND = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{project_root}/assets/robots/g1-29dof-inspire-base-fix-usd/g1_29dof_with_inspire_rev_1_0.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            # waist joints
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            
            # arms joints
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            
            # fingers joints
            "L_index_proximal_joint": 0.0,
            "L_index_intermediate_joint": 0.0,
            "L_middle_proximal_joint": 0.0,
            "L_middle_intermediate_joint": 0.0,
            "L_pinky_proximal_joint":0.0,
            "L_pinky_intermediate_joint":0.0,
            "L_ring_proximal_joint":0.0,
            "L_ring_intermediate_joint":0.0,
            "L_thumb_proximal_yaw_joint":0.0,
            "L_thumb_proximal_pitch_joint":0.0,
            "L_thumb_intermediate_joint":0.0,
            "L_thumb_distal_joint":0.0,

            "R_index_proximal_joint": 0.0,
            "R_index_intermediate_joint": 0.0,
            "R_middle_proximal_joint": 0.0,
            "R_middle_intermediate_joint": 0.0,
            "R_pinky_proximal_joint":0.0,
            "R_pinky_intermediate_joint":0.0,
            "R_ring_proximal_joint":0.0,
            "R_ring_intermediate_joint":0.0,
            "R_thumb_proximal_yaw_joint":0.0,
            "R_thumb_proximal_pitch_joint":0.0,
            "R_thumb_intermediate_joint":0.0,
            "R_thumb_distal_joint":0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", 
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint", 
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=0.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            damping={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=None,
            velocity_limit=None,
             stiffness={  # increase the stiffness (kp)
                 ".*_shoulder_.*_joint": 25.0,
                 ".*_elbow_joint": 50.0,
                 ".*_wrist_.*_joint": 40.0,
            },
             damping={    # increase the damping (kd)
                 ".*_shoulder_.*_joint": 2.0,
                 ".*_elbow_joint": 2.0,
                 ".*_wrist_.*_joint": 2.0,
             },
            armature=None,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_proximal_joint",
                ".*_index_intermediate_joint",
                ".*_middle_proximal_joint",
                ".*_middle_intermediate_joint",
                ".*_pinky_proximal_joint",
                ".*_pinky_intermediate_joint",
                ".*_ring_proximal_joint",
                ".*_ring_intermediate_joint",
                ".*_thumb_proximal_yaw_joint",
                ".*_thumb_proximal_pitch_joint",
                ".*_thumb_intermediate_joint",
                ".*_thumb_distal_joint",
            ],
            effort_limit=100.0,
            velocity_limit=50,
            stiffness={
                ".*_index_proximal_joint":1000.0,
                ".*_index_intermediate_joint":1000.0,
                ".*_middle_proximal_joint":1000.0,
                ".*_middle_intermediate_joint":1000.0,
                ".*_pinky_proximal_joint":1000.0,
                ".*_pinky_intermediate_joint":1000.0,
                ".*_ring_proximal_joint":1000.0,
                ".*_ring_intermediate_joint":1000.0,
                ".*_thumb_proximal_yaw_joint":1000.0,
                ".*_thumb_proximal_pitch_joint":1000.0,
                ".*_thumb_intermediate_joint":1000.0,
                ".*_thumb_distal_joint":1000.0,
            },
            damping={
                ".*_index_proximal_joint":15,
                ".*_index_intermediate_joint":15,
                ".*_middle_proximal_joint":15,
                ".*_middle_intermediate_joint":15,
                ".*_pinky_proximal_joint":15,
                ".*_pinky_intermediate_joint":15,
                ".*_ring_proximal_joint":15,
                ".*_ring_intermediate_joint":15,
                ".*_thumb_proximal_yaw_joint":15,
                ".*_thumb_proximal_pitch_joint":15,
                ".*_thumb_intermediate_joint":15,
                ".*_thumb_distal_joint":15,
            },
            armature={
                ".*": 0.0
            },
        ),

    },
)

G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #usd_path="/home/ubuntu/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/assets/G1_with_hand_FTP.usd",
        usd_path=f"assets/robots/G1_with_hand_FTP.usd",
        activate_contact_sensors=True, 
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=4,
            # fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        rot=(0.7071, 0, 0, 0.7071),  # Quaternion for no rotation
        #joint_pos={".*": 0.0},
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": -0.7,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            ".*_hip_yaw_joint": 0.06,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", 
                ".*_hip_roll_joint", 
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint", 
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                "waist_yaw_joint": 88.0,
            },
            velocity_limit=100.0,
            stiffness={
                ".*_hip_yaw_joint": 60.0,
                ".*_hip_roll_joint": 60.0,
                ".*_hip_pitch_joint": 60.0,
                ".*_knee_joint": 100.0,
                "waist_yaw_joint": 60.0,
            },
            damping={
                ".*_hip_yaw_joint": 1.0,
                ".*_hip_roll_joint": 1.0,
                ".*_hip_pitch_joint": 1.0,
                ".*_knee_joint": 2.0,
                "waist_yaw_joint": 1.0,
            },
            armature={
                ".*_hip_yaw_joint": 0.01017752004,
                ".*_hip_roll_joint": 0.025101925,
                ".*_hip_pitch_joint": 0.01017752004,
                ".*_knee_joint": 0.025101925,
                "waist_yaw_joint": 0.01017752004,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=50.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=1.0,
            armature=0.00721945,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit=25.0,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=1.0,
            armature=0.003609725,
        ),
        "right-hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "right_index_1_joint",
                "right_little_1_joint",
                "right_middle_1_joint",
                "right_ring_1_joint",
                "right_thumb_1_joint",
                "right_thumb_2_joint",
            ],
            effort_limit=25.0,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=1.0,
        ),
        "left-hand": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_index_1_joint",
                "left_little_1_joint",
                "left_middle_1_joint",
                "left_ring_1_joint",
                "left_thumb_1_joint",
                "left_thumb_2_joint",
            ],
            effort_limit=25.0,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=1.0,
        ),
    },
)

