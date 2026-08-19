# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import pickplace_pitcher_g1_29dof_brainco_env_cfg


gym.register(
    id="Isaac-PickPlace-Pitcher-G129-Brainco-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": pickplace_pitcher_g1_29dof_brainco_env_cfg.PickPlacePitcherG129BraincoBaseFixEnvCfg,
    },
    disable_env_checker=True,
)
