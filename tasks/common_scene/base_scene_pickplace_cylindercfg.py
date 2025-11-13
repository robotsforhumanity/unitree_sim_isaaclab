# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import   CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")
@configclass
class TableCylinderSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],  # 房间中心点
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            #usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",  # use simple room model
            usd_path="/home/inorbit/inOrbit/RobotSpace/gauss.usda",
            #usd_path="/home/inorbit/inOrbit/splat/robotspace/gauss.usda",
        ),
    )


        # 1. table configuration
    #picture = AssetBaseCfg(
    #    prim_path="/World/envs/env_.*/Room/Picture",    # table in the scene
    #    init_state=AssetBaseCfg.InitialStateCfg(pos=[0.33, 12.92, 1.35],   # initial position [x, y, z]
    #                                            rot=[0.707080, 0.707080,-0.006171, -0.006171]), # initial rotation [x, y, z, w]
    #    spawn=UsdFileCfg(
    #        usd_path="/home/inorbit/inOrbit/Pared.usdc",    # table model file
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), 
    #        scale=[1.5,1.5,1.0],   # set to kinematic object
    #    ),
    #    
    #)

    #cartel = AssetBaseCfg(
    #    prim_path="/World/envs/env_.*/Room/tgn",    # table in the scene
    #    init_state=AssetBaseCfg.InitialStateCfg(pos=[-1.7, 2.0, -0.8],   # initial position [x, y, z]
    #                                            rot=[0.0, 0.0,0.0, 0.0]), # initial rotation [x, y, z, w]
    #    spawn=UsdFileCfg(
    #        usd_path="/home/ubuntu/Robots_For_Humanity/unitree_sim_isaaclab/tasks/common_scene/Cartel_TGN/Cartel_TGN.usdc",    # table model file
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), 
    #        scale=[0.7,0.7,0.7],   # set to kinematic object
    #    ),
    #    
    #)


    #table = AssetBaseCfg(
    #    prim_path="/World/envs/env_.*/table",   
    #    init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.95],  
    #                                            rot=[1.0, 0.0, 0.0, 0.0]), 
    #    spawn=UsdFileCfg(
    #        usd_path="/home/inorbit/inOrbit/mesarotada/mesarotada.usdc",   
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),   
    #    ),
    #)

    '''
    packing_table_3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_3",   
        init_state=AssetBaseCfg.InitialStateCfg(pos=[3.5, 0.55, -0.2],  
                                                rot=[1.0, 0.0, 0.0, 0.0]), 
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",   
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),   
        ),
    )
    packing_table_4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_4",   
        init_state=AssetBaseCfg.InitialStateCfg(pos=[3.5, -5, -0.2],  
                                                rot=[1.0, 0.0, 0.0, 0.0]), 
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",   
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),   
        ),
    )
    packing_table_5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_5",   
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-3.5, -5, -0.2],  
                                                rot=[1.0, 0.0, 0.0, 0.0]), 
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",   
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),   
        ),
    )
    packing_table_6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_6",   
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, -5, -0.2],  
                                                rot=[1.0, 0.0, 0.0, 0.0]), 
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",   
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),   
        ),
    )
    '''
    # Object
    # 2. object configuration (cylinder)     
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",    # object in t0he scene
        init_state=RigidObjectCfg.InitialStateCfg(pos=[3.0, .0, 0.3], # initial poition (pos) 
                                                  rot=[1, 0, 0, 0]), # initial rotation (rot)
        spawn=sim_utils.CylinderCfg(
            radius=0.018,    # cylinder radius (radius)
            height=0.35,     # cylinder height (height)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            ),    # rigid body properties configuration (rigid_props)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.4),    # mass properties configuration (mass)
            collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15), metallic=1.0),    # visual material configuration (visual_material)
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",    # friction combine mode
                restitution_combine_mode="min",    # restitution combine mode
                static_friction=1.5,    # static friction coefficient
                dynamic_friction=1.5,    # dynamic friction coefficient
                restitution=0.0,    # restitution coefficient (no restitution)
            ),
        ),
    )

    #apple = RigidObjectCfg(
    #    prim_path="/World/envs/env_.*/apple",    # object in the scene
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.2, 0.0, 0.1], # initial position (pos) 
    #                                              rot=[0, 0, 0, 1]), # initial rotation (rot)
    #   spawn=UsdFileCfg(
    #        usd_path="/home/ubuntu/Robots_For_Humanity/unitree_sim_isaaclab/tasks/common_scene/Apple/scene.usdc",
    #        scale=[0.0005,0.0005,0.0005],
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(),    # rigid body properties configuration (rigid_props)
    #        mass_props=sim_utils.MassPropertiesCfg(mass=0.4),    # mass properties configuration (mass)
    #        collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
    #    ),
    #)
#
    #plate = RigidObjectCfg(
    #    prim_path="/World/envs/env_.*/plate",    # object in the scene
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, -0.2, 0.05], # initial position (pos) 
    #                                              rot=[0.707, 0.707, 0.0, 0.0]), # initial rotation (rot)
    #   spawn=UsdFileCfg(
    #        usd_path="/home/ubuntu/Robots_For_Humanity/unitree_sim_isaaclab/tasks/common_scene/Plate_white/scene.usdc",
    #        scale=[0.005,0.005,0.005],
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(),    # rigid body properties configuration (rigid_props)
    #        mass_props=sim_utils.MassPropertiesCfg(mass=0.4),    # mass properties configuration (mass)
    #        collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
    #    ),
    #)
    

    #object = RigidObjectCfg(
    #    prim_path="/World/envs/env_.*/object1",    # object in the scene
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.18, -0.15, 0.0], # initial position (pos)
    #                                              rot=[1, 0, 0, 0]), # initial rotation (rot)
    #    spawn=UsdFileCfg(
    #        usd_path="/home/ubuntu/Robots_For_Humanity/unitree_sim_isaaclab/tasks/common_scene/valvula/new_valvula.usd",
    #        scale=[0.5,0.5,0.5],
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(),    # rigid body properties configuration (rigid_props)
    #        mass_props=sim_utils.MassPropertiesCfg(mass=0.4),    # mass properties configuration (mass)
    #        collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
    #    ),
    #)

    #valvula_joint = ArticulationCfg(
    #    prim_path="/World/envs/env_.*/valvulaJoint",  # debe apuntar al root del Articulation
    #    init_state=ArticulationCfg.InitialStateCfg(
    #        pos=[0.0, 0.0, 0.05],  # posición inicial
    #        rot=[1, 0, 0, 0],
    #        joint_pos={"RevoluteJoint": 0.0},
    #        joint_vel ={"RevoluteJoint": 0.0},
    #    ),
    #    spawn=UsdFileCfg(
    #        usd_path="/home/ubuntu/Robots_For_Humanity/unitree_sim_isaaclab/tasks/common_scene/valvula/new_valvula.usd",
    #        scale=[0.02, 0.02, 0.02],
    #        mass_props=sim_utils.MassPropertiesCfg(
    #            mass=0.5,                  # masa de la rueda
    #        ),
    #        collision_props=sim_utils.CollisionPropertiesCfg(),
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #    ),
    #    actuators={
    #    'valvula': ImplicitActuatorCfg(
    #        joint_names_expr=['RevoluteJoint'],  # todos los joints de la articulación
    #        effort_limit=None,
    #        velocity_limit=None,
    #        stiffness={'RevoluteJoint': 0.0},
    #        damping={'RevoluteJoint': 0.0},
    #        #armature={'RevoluteJoint': 0.0},
    #    )
    # }
    #)

    #valvula_fixed = RigidObjectCfg(
    #    prim_path="/World/envs/env_.*/valvulaFixed",    # object in the scene
    #    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, -0.1, 0.35], # initial position (pos)
    #                                              rot=[1, 0, 0, 0]), # initial rotation (rot)
    #    spawn=UsdFileCfg(
    #        usd_path="/home/ubuntu/Robots_For_Humanity/unitree_sim_isaaclab/tasks/common_scene/valvula_fija.usd",
    #        scale=[0.02,0.02,0.02],
    #        rigid_props=sim_utils.RigidBodyPropertiesCfg(),    # rigid body properties configuration (rigid_props)
    #        mass_props=sim_utils.MassPropertiesCfg(mass=0.4),    # mass properties configuration (mass)
    #        collision_props=sim_utils.CollisionPropertiesCfg(),    # collision properties configuration (collision_props)
    #    ),
    #)

    #object = ArticulationCfg(
    #    prim_path="/World/envs/env_.*/Object",    # object in the scene
    #    init_state=ArticulationCfg.InitialStateCfg(pos=[-0.18, 0.40, 0.8], # initial position (pos)
    #                                              rot=[1, 0, 0, 0],
    #                                              joint_pos = {"RevoluteJoint" : 0.0}), # initial rotation (rot)
    #    
    #    spawn=UsdFileCfg(
    #        usd_path="/home/ubuntu/Robots_For_Humanity/unitree_sim_isaaclab/tasks/common_scene/valvula/new_valvula.usd",
    #        scale=[0.5,0.5,0.5],
    #    ),
#
    #    actuators={
    #    "valve": ImplicitActuatorCfg(
    #        joint_names_expr=['RevoluteJoint'],   # todos los joints de la válvula
    #        effort_limit_sim=100.0,    # esfuerzo máximo
    #        velocity_limit_sim=5.0,    # velocidad máxima
    #        stiffness={'RevoluteJoint': 50.0},
    #        damping={'RevoluteJoint': 2.0},
    #    )
    #    },
    #)


    
    # Ground plane
    # 3. ground configuration
    #ground = AssetBaseCfg(
    #    prim_path="/World/GroundPlane",    # ground in the scene
    #    spawn=GroundPlaneCfg( ),    # ground configuration
    #)

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0),    # light intensity
    )

    #world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
    #                                                pos_offset=(1.0, -1.1, 1.1),
    #                                                rot_offset=( 0.84,0.27, -0.35, 0.29),
    #                                                focal_length = 16.5)
    #world_camera.enable_rgb = True
    #world_camera.enable_depth = True
    #world_camera.enable_semantics = True 