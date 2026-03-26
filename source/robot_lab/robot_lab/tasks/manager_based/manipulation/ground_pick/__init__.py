# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents
from .go2_x5_isaac_env import Go2X5GroundPickIsaacEnv

gym.register(
    id="RobotLab-Isaac-GroundPick-Go2-X5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_x5_ground_pick_env_cfg:Go2X5GroundPickEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5GroundPickPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-GroundPick-Go2-X5-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_x5_ground_pick_env_cfg:Go2X5GroundPickEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2X5GroundPickPPORunnerCfg",
    },
)

__all__ = ["Go2X5GroundPickIsaacEnv"]
