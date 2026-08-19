# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from . import mdp

from tasks.common_config import G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# import public scene configuration
# NOTE: the "pitcher" object is currently a cylinder stand-in; swap the object
# spawn in this scene (or in a dedicated common_scene cfg) once a pitcher USD
# is available.
from tasks.common_scene.base_scene_pickplace_cylindercfg import TableCylinderSceneCfg, YPF_ROBOT_POS, YPF_ROBOT_ROT

##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(TableCylinderSceneCfg):
    """object table scene configuration class
    inherits the common table + object scene and adds the BrainCo G1 robot
    """

    # humanoid robot configuration (G1 29dof + BrainCo hands, fixed base)
    # Spawn inside the YPF NuRec warehouse (centroid of the reconstructed interior).
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_brainco_base_fix(
        init_pos=YPF_ROBOT_POS,
        init_rot=YPF_ROBOT_ROT,
    )

    # camera configuration
    front_camera = CameraPresets.g1_brainco_front_camera()
    left_wrist_camera = CameraPresets.left_brainco_wrist_camera()
    right_wrist_camera = CameraPresets.right_brainco_wrist_camera()

##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


@configclass
class ObservationsCfg:
    """
    defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class"""

        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_brainco_state = ObsTerm(func=mdp.get_robot_brainco_joint_states)

        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    # Out-of-workspace reset. Defaults were the old table (y in [0.2, 0.7], z > 0.5),
    # so a coverall on the warehouse floor was "out of range" every step and the
    # reset event kept teleporting it. Bounds match the YPF aisle.
    success = DoneTerm(
        func=mdp.reset_object_estimate,
        params={
            "object_cfg": SceneEntityCfg("mameluco"),
            "min_x": -2.0,
            "max_x": 4.0,
            "min_y": -6.0,
            "max_y": 1.0,
            "min_height": -0.5,
        },
    )


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    reset_object = EventTermCfg(
        func=mdp.reset_nodal_state_uniform,
        mode="reset",
        params={
            "position_range": {
                "x": [-0.05, 0.05],
                "y": [-0.05, 0.05],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("mameluco"),
        },
    )


@configclass
class PickPlacePitcherG129BraincoBaseFixEnvCfg(ManagerBasedRLEnvCfg):
    """
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1,
                                                     env_spacing=2.5,
                                                     replicate_physics=True)
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings — deformable mameluco needs PhysX GPU dynamics
        self.sim.device = "cuda:0"
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 2 ** 22
        self.sim.physx.friction_correlation_distance = 0.00625
        # create event manager
        self.event_manager = SimpleEventManager()

        self.event_manager.register("reset_object_self", SimpleEvent(
            func=lambda env: base_mdp.reset_nodal_state_uniform(
                env,
                torch.arange(env.num_envs, device=env.device),
                position_range={"x": [-0.05, 0.05], "y": [-0.05, 0.05]},
                velocity_range={},
                asset_cfg=SceneEntityCfg("mameluco"),
            )
        ))

        self.event_manager.register("reset_all_self", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device))
        ))
