# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import   CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")

# NuRec warehouse is COLMAP Y-down; Isaac world is Z-up. Rotate -90 deg about X
# so the reconstructed floor (local y ≈ 0.50 m) becomes world z = 0.
# If the scene looks upside down, swap to +90 deg: rot = [0.70710678, 0.70710678, 0.0, 0.0]
YPF_WAREHOUSE_USDZ = "/home/fran/ypf/YPF_warehouse.usdz"
YPF_MALEMUCO_USDZ = (
    "/home/fran/ypf/Mameluco_V4_YPF_physics_isaacsim/"
    "Mameluco_V4_YPF_physics_isaacsim/MamelucoYPF_physics.usd"
)
YPF_WAREHOUSE_POS = [0.0, 0.0, 0.50]
YPF_WAREHOUSE_ROT = [0.70710678, -0.70710678, 0.0, 0.0]  # (w, x, y, z)

# Spawn inside the reconstructed warehouse. The robot pose is NOT in this file's
# assets — each task must pass YPF_ROBOT_POS to its G1/H12 preset (see env cfgs).
# G1 spawn z is ~0.76 at scale 1. Robot USD is currently scale 0.5, so pelvis ~0.40.
YPF_ROBOT_POS = (0.45, -2.70, 0.42)
YPF_ROBOT_ROT = (0.7071, 0.0, 0.0, 0.7071)  # facing +Y, toward the cylinder
YPF_OBJECT_POS = [0.45, -2.00, 0.15]

@configclass
class TableCylinderSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=YPF_WAREHOUSE_POS,
            rot=YPF_WAREHOUSE_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=YPF_WAREHOUSE_USDZ,
        ),
    )
    # Object
    # 2. object configuration (cylinder)     
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",    # object in the scene
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[YPF_OBJECT_POS[0] + 0.6, YPF_OBJECT_POS[1], 0.35],
            rot=[1, 0, 0, 0],
        ),
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

    # Isaac Sim export with PhysxDeformableBodyAPI (nylon bag, ~1 kg). Keep the
    # USD physics as-is; do not wrap it in a rigid cuboid.
    mameluco = DeformableObjectCfg(
        prim_path="/World/envs/env_.*/mameluco",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=YPF_OBJECT_POS, rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=YPF_MALEMUCO_USDZ,
        ),
    )
    # Ground plane at world z = 0. Visual grid kept (color=None) so it is
    # obvious whether the digital-twin floor is coplanar with the world.
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(color=None, size=(100.0, 100.0)),
    )

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0),    # light intensity
    )

    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(0.45, 0.90, 1.6),
                                                    rot_offset=( -0.00617,0.00617, 0.70708, -0.70708),
                                                    focal_length = 16.5)