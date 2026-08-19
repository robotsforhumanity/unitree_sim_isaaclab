# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
BrainCo hands state observation
Extracts the 12 actuated BrainCo joints and publishes them to DDS.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import sys
import os
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# DDS/shared-memory order (must match dds/brainco_dds.py and
# action_provider_dds.py brainco_hand_joint_mapping):
#   [0..5]  right: pinky, ring, middle, index, thumb_proximal, thumb_metacarpal
#   [6..11] left : pinky, ring, middle, index, thumb_proximal, thumb_metacarpal
BRAINCO_DDS_JOINT_NAMES = [
    "right_pinky_proximal_joint",
    "right_ring_proximal_joint",
    "right_middle_proximal_joint",
    "right_index_proximal_joint",
    "right_thumb_proximal_joint",
    "right_thumb_metacarpal_joint",
    "left_pinky_proximal_joint",
    "left_ring_proximal_joint",
    "left_middle_proximal_joint",
    "left_index_proximal_joint",
    "left_thumb_proximal_joint",
    "left_thumb_metacarpal_joint",
]

_obs_cache = {
    "device": None,
    "batch": None,
    "idx_t": None,
    "idx_batch": None,
    "pos_buf": None,
    "vel_buf": None,
    "torque_buf": None,
    "dds_last_ms": 0,
    "dds_min_interval_ms": 20,
}

# global variable to cache the DDS instance
_brainco_dds = None
_dds_initialized = False


def _get_brainco_dds_instance():
    """get the DDS instance, delay initialization"""
    global _brainco_dds, _dds_initialized

    if not _dds_initialized or _brainco_dds is None:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager
            _brainco_dds = dds_manager.get_object("brainco")
            print("[brainco_state] DDS communication instance obtained")

            import atexit
            def cleanup_dds():
                try:
                    if _brainco_dds:
                        dds_manager.unregister_object("brainco")
                        print("[brainco_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[brainco_state] Error closing DDS: {e}")
            atexit.register(cleanup_dds)

        except Exception as e:
            print(f"[brainco_state] Failed to get DDS instance: {e}")
            _brainco_dds = None

        _dds_initialized = True

    return _brainco_dds


def get_robot_brainco_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """get the BrainCo hands joint states and publish them to DDS

    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
        enable_dds: bool - whether to enable the DDS publish function

    Returns:
        torch.Tensor with the 12 actuated joint positions
    """
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque
    device = joint_pos.device
    batch = joint_pos.shape[0]

    global _obs_cache
    if _obs_cache["device"] != device or _obs_cache["idx_t"] is None:
        # resolve joint indices by name (robust to USD joint ordering)
        all_names = env.scene["robot"].data.joint_names
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        try:
            indices = [name_to_idx[name] for name in BRAINCO_DDS_JOINT_NAMES]
        except KeyError as e:
            raise RuntimeError(
                f"[brainco_state] BrainCo joint not found in robot: {e}. "
                f"Available joints: {all_names}"
            )
        _obs_cache["idx_t"] = torch.tensor(indices, dtype=torch.long, device=device)
        _obs_cache["device"] = device
        _obs_cache["batch"] = None
    idx_t = _obs_cache["idx_t"]
    n = idx_t.numel()

    if _obs_cache["batch"] != batch or _obs_cache["idx_batch"] is None:
        _obs_cache["idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["batch"] = batch

    idx_batch = _obs_cache["idx_batch"]
    pos_buf = _obs_cache["pos_buf"]
    vel_buf = _obs_cache["vel_buf"]
    torque_buf = _obs_cache["torque_buf"]

    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))

    # publish to DDS (only publish the data of the first environment)
    if enable_dds and len(pos_buf) > 0:
        try:
            import time
            now_ms = int(time.time() * 1000)
            if now_ms - _obs_cache["dds_last_ms"] >= _obs_cache["dds_min_interval_ms"]:
                brainco_dds = _get_brainco_dds_instance()
                if brainco_dds:
                    pos = pos_buf[0].contiguous().cpu().numpy()
                    vel = vel_buf[0].contiguous().cpu().numpy()
                    torque = torque_buf[0].contiguous().cpu().numpy()
                    brainco_dds.write_brainco_state(pos, vel, torque)
                    _obs_cache["dds_last_ms"] = now_ms
        except Exception as e:
            print(f"[brainco_state] Failed to write to shared memory: {e}")

    return pos_buf
