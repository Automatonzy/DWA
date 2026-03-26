# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ground_pick_success(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_eef_object_distance: float,
    close_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["arm_joint7", "arm_joint8"], preserve_order=True),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(robot_cfg.joint_names, preserve_order=robot_cfg.preserve_order)
    gripper_opening = robot.data.joint_pos[:, joint_ids].mean(dim=1)
    distance = torch.norm(obj.data.root_pos_w[:, :3] - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    near_ee = distance < max_eef_object_distance
    closed = gripper_opening < close_threshold
    return lifted & near_ee & closed
