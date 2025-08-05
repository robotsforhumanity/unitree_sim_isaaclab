# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
ftp state
"""     
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import sys
import os
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


import torch

def get_robot_girl_joint_names() -> list[str]:
    return [
        "left_index_1_joint",
        "left_little_1_joint",
        "left_middle_1_joint",
        "left_ring_1_joint",
        "left_thumb_1_joint",
        "left_thumb_2_joint",
        "right_index_1_joint",
        "right_little_1_joint",
        "right_middle_1_joint",
        "right_ring_1_joint",
        "right_thumb_1_joint",
        "right_thumb_2_joint"
        ]
# global variable to cache the DDS instance
_ftp_dds = None
_dds_initialized = False

def _get_ftp_dds_instance():
    """get the DDS instance, delay initialization"""
    global _ftp_dds, _dds_initialized
    
    if not _dds_initialized:
        try:
            # dynamically import the DDS module
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.ftp_dds import get_ftp_dds
            
            _ftp_dds = get_ftp_dds()
            _ftp_dds.start_communication(enable_publish=True, enable_subscribe=False)
            print("[Observations FTP] DDS communication instance obtained")
            
            # register the cleanup function
            import atexit
            def cleanup_dds():
                try:
                    if _ftp_dds:
                        _ftp_dds.stop_communication()
                        _ftp_dds.cleanup()
                        print("[ftp_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[ftp_state] Error closing DDS: {e}")
            atexit.register(cleanup_dds)
            
        except Exception as e:
            print(f"[Observations FTP] Failed to get DDS instances: {e}")
            _ftp_dds = None
        
        _dds_initialized = True
    
    return _ftp_dds

def get_robot_ftp_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """get the robot gripper joint states and publish them to DDS
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish function
    
    Returns:
        torch.Tensor
    """
    # get the gripper joint positions, velocities, torques
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel  
    joint_torque = env.scene["robot"].data.applied_torque
    
    # get the gripper joint indices (last 14 joints)
    gripper_joint_indices=[30,31,32,33,34,35,36,37,38,39,40,41]

    if len(gripper_joint_indices) == 12:
        # extract the gripper joint states in the specified order
        gripper_positions = joint_pos[:, gripper_joint_indices]
        gripper_velocities = joint_vel[:, gripper_joint_indices]  
        gripper_torques = joint_torque[:, gripper_joint_indices]
        
        # publish to DDS (only publish the data of the first environment)
        if enable_dds and len(gripper_positions) > 0:
            try:

                ftp_dds = _get_ftp_dds_instance()
                if ftp_dds:
                    pos = gripper_positions[0].cpu().numpy()
                    vel = gripper_velocities[0].cpu().numpy()
                    torque = gripper_torques[0].cpu().numpy()
                    left_pos = pos[:6]
                    right_pos = pos[6:]
                    left_vel = vel[:6]
                    right_vel = vel[6:]
                    left_torque = torque[:6]
                    right_torque = torque[6:]
                    ftp_dds.write_hand_states(left_pos, left_vel, left_torque, right_pos, right_vel, right_torque)
            except Exception as e:
                print(f"ftp_state [ftp_state] Failed to write to DDS: {e}")
        
        return gripper_positions
    else:
        # if the gripper joints are not found, return a zero tensor
        return torch.zeros((joint_pos.shape[0], 14))

