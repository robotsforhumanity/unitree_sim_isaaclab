# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
BrainCo hand DDS communication class
Handles the state publishing and command receiving of the BrainCo (Revo) hands.

Protocol (compatible with xr_teleoperate robot_hand_brainco.py):
- Commands  : rt/brainco/left/cmd  and rt/brainco/right/cmd   (MotorCmds_, 6 motors per hand)
- States    : rt/brainco/left/state and rt/brainco/right/state (MotorStates_, 6 motors per hand)
- Motor order per hand (teleop convention): [thumb, thumb_aux, index, middle, ring, pinky]
- q values are normalised to [0, 1] where 0.0 = fully open and 1.0 = fully closed.

Internally (shared memory with the action provider) positions are stored as a
12-element array of joint angles in radians, in the order expected by
action_provider_dds.py brainco_hand_joint_mapping:
    [0..5]  right hand: pinky, ring, middle, index, thumb_proximal, thumb_metacarpal
    [6..11] left  hand: pinky, ring, middle, index, thumb_proximal, thumb_metacarpal
"""

import numpy as np
from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_, unitree_go_msg_dds__MotorState_

# Joint limits from the URDF (lower is always 0.0 = open)
FINGER_UPPER = 1.4661         # pinky/ring/middle/index proximal
THUMB_PROXIMAL_UPPER = 1.0472
THUMB_METACARPAL_UPPER = 1.5184

# Per-hand upper limits in the *internal* order:
# [pinky, ring, middle, index, thumb_proximal, thumb_metacarpal]
HAND_UPPER_LIMITS = np.array([
    FINGER_UPPER, FINGER_UPPER, FINGER_UPPER, FINGER_UPPER,
    THUMB_PROXIMAL_UPPER, THUMB_METACARPAL_UPPER,
], dtype=np.float64)

# teleop (DDS) motor order:  [0:thumb, 1:thumb_aux, 2:index, 3:middle, 4:ring, 5:pinky]
# internal order:            [0:pinky, 1:ring, 2:middle, 3:index, 4:thumb_prox, 5:thumb_meta]
# teleop index -> internal index
TELEOP_TO_INTERNAL = [4, 5, 3, 2, 1, 0]
# internal index -> teleop index
INTERNAL_TO_TELEOP = [5, 4, 3, 2, 0, 1]

NUM_HAND_MOTORS = 6


class BraincoDDS(DDSObject):
    """BrainCo hands DDS communication class - singleton pattern

    Features:
    - Publish the state of both hands to DDS (rt/brainco/left/state, rt/brainco/right/state)
    - Receive the control commands of both hands (rt/brainco/left/cmd, rt/brainco/right/cmd)
    """

    def __init__(self, node_name: str = "brainco"):
        """Initialize the BrainCo hands DDS node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized'):
            return

        super().__init__()
        self.node_name = node_name

        # per-hand state messages (6 motors each)
        self.left_hand_state = MotorStates_()
        self.left_hand_state.states = [unitree_go_msg_dds__MotorState_() for _ in range(NUM_HAND_MOTORS)]
        self.right_hand_state = MotorStates_()
        self.right_hand_state.states = [unitree_go_msg_dds__MotorState_() for _ in range(NUM_HAND_MOTORS)]

        # latest commands per hand in internal order (radians)
        self._right_cmd_rad = np.zeros(NUM_HAND_MOTORS, dtype=np.float64)
        self._left_cmd_rad = np.zeros(NUM_HAND_MOTORS, dtype=np.float64)
        self._cmd_received = False

        self._initialized = True

        # setup the shared memory
        self.setup_shared_memory(
            input_shm_name="isaac_brainco_state",   # read the hands state from Isaac Lab
            input_size=1024,
            output_shm_name="isaac_brainco_cmd",    # output the command to Isaac Lab
            output_size=1024,
        )

        print(f"[{self.node_name}] BrainCo hands DDS node initialized")

    def setup_publisher(self) -> bool:
        """Setup the per-hand state publishers"""
        try:
            self.left_state_publisher = ChannelPublisher("rt/brainco/left/state", MotorStates_)
            self.left_state_publisher.Init()
            self.right_state_publisher = ChannelPublisher("rt/brainco/right/state", MotorStates_)
            self.right_state_publisher.Init()

            print(f"[{self.node_name}] BrainCo state publishers initialized")
            return True
        except Exception as e:
            print(f"brainco_dds [{self.node_name}] State publisher initialization failed: {e}")
            return False

    def setup_subscriber(self) -> bool:
        """Setup the per-hand command subscribers"""
        try:
            self.left_cmd_subscriber = ChannelSubscriber("rt/brainco/left/cmd", MotorCmds_)
            self.left_cmd_subscriber.Init(lambda msg: self.dds_subscriber(msg, "left"), 32)
            self.right_cmd_subscriber = ChannelSubscriber("rt/brainco/right/cmd", MotorCmds_)
            self.right_cmd_subscriber.Init(lambda msg: self.dds_subscriber(msg, "right"), 32)

            print(f"[{self.node_name}] BrainCo command subscribers initialized")
            return True
        except Exception as e:
            print(f"brainco_dds [{self.node_name}] Command subscriber initialization failed: {e}")
            return False

    # ------------------------------------------------------------------
    # normalisation helpers (0.0 = open ... 1.0 = closed  <->  radians)
    # ------------------------------------------------------------------
    @staticmethod
    def _norm_to_rad(q_norm: np.ndarray) -> np.ndarray:
        return np.clip(q_norm, 0.0, 1.0) * HAND_UPPER_LIMITS

    @staticmethod
    def _rad_to_norm(q_rad: np.ndarray) -> np.ndarray:
        return np.clip(q_rad / HAND_UPPER_LIMITS, 0.0, 1.0)

    def dds_publisher(self) -> Any:
        """Convert the Isaac Lab hands state to per-hand DDS messages and publish.

        Expected shared-memory data format:
        {
            "positions":  [12 joint angles in radians, internal order],
            "velocities": [12 joint velocities],
            "torques":    [12 joint torques]
        }
        """
        try:
            data = self.input_shm.read_data()
            if data is None:
                return
            positions = data.get("positions")
            velocities = data.get("velocities")
            torques = data.get("torques")
            if positions is None or len(positions) < 12:
                return

            pos = np.asarray(positions, dtype=np.float64)
            vel = np.asarray(velocities, dtype=np.float64) if velocities else np.zeros(12)
            tau = np.asarray(torques, dtype=np.float64) if torques else np.zeros(12)

            right_norm = self._rad_to_norm(pos[0:6])
            left_norm = self._rad_to_norm(pos[6:12])

            for teleop_idx in range(NUM_HAND_MOTORS):
                internal_idx = TELEOP_TO_INTERNAL[teleop_idx]
                self.right_hand_state.states[teleop_idx].q = float(right_norm[internal_idx])
                self.right_hand_state.states[teleop_idx].dq = float(vel[internal_idx])
                self.right_hand_state.states[teleop_idx].tau_est = float(tau[internal_idx])
                self.left_hand_state.states[teleop_idx].q = float(left_norm[internal_idx])
                self.left_hand_state.states[teleop_idx].dq = float(vel[6 + internal_idx])
                self.left_hand_state.states[teleop_idx].tau_est = float(tau[6 + internal_idx])

            self.left_state_publisher.Write(self.left_hand_state)
            self.right_state_publisher.Write(self.right_hand_state)

        except Exception as e:
            print(f"brainco_dds [{self.node_name}] Error processing publish data: {e}")
            return None

    def dds_subscriber(self, msg: MotorCmds_, datatype: str = None) -> Dict[str, Any]:
        """Process a per-hand command: convert the DDS command to the Isaac Lab format.

        The merged 12-element command (radians, internal order) is written to the
        output shared memory every time either hand receives a new command.
        """
        try:
            if len(msg.cmds) < NUM_HAND_MOTORS:
                return {}

            q_teleop = np.array(
                [float(msg.cmds[i].q) for i in range(NUM_HAND_MOTORS)], dtype=np.float64
            )
            # reorder teleop -> internal, then denormalise to radians
            q_internal_norm = np.empty(NUM_HAND_MOTORS, dtype=np.float64)
            for teleop_idx in range(NUM_HAND_MOTORS):
                q_internal_norm[TELEOP_TO_INTERNAL[teleop_idx]] = q_teleop[teleop_idx]
            q_rad = self._norm_to_rad(q_internal_norm)

            if datatype == "right":
                self._right_cmd_rad = q_rad
            elif datatype == "left":
                self._left_cmd_rad = q_rad
            else:
                return {}
            self._cmd_received = True

            cmd_data = {
                "positions": np.concatenate([self._right_cmd_rad, self._left_cmd_rad]).tolist(),
                "velocities": [0.0] * 12,
                "torques": [0.0] * 12,
                "kp": [0.0] * 12,
                "kd": [0.0] * 12,
            }
            self.output_shm.write_data(cmd_data)

        except Exception as e:
            print(f"brainco_dds [{self.node_name}] Error processing subscribe data: {e}")
            return {}

    def get_brainco_hand_command(self) -> Optional[Dict[str, Any]]:
        """Get the hands control command

        Returns:
            Dict: the hands command, return None if there is no new command
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None

    def write_brainco_state(self, positions, velocities, torques):
        """Write the hands state to the shared memory

        Args:
            positions: 12 joint positions in radians (internal order) list or torch.Tensor
            velocities: 12 joint velocities list or torch.Tensor
            torques: 12 joint torques list or torch.Tensor
        """
        try:
            brainco_data = {
                "positions": positions.tolist() if hasattr(positions, 'tolist') else positions,
                "velocities": velocities.tolist() if hasattr(velocities, 'tolist') else velocities,
                "torques": torques.tolist() if hasattr(torques, 'tolist') else torques,
            }
            if self.input_shm:
                self.input_shm.write_data(brainco_data)
        except Exception as e:
            print(f"brainco_dds [{self.node_name}] Error writing brainco hands state: {e}")
