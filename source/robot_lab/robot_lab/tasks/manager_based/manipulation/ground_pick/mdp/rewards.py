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


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w[:, :3] - ee_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def gripper_closed_around_object(
    env: ManagerBasedRLEnv,
    distance_std: float,
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
    near_object = 1.0 - torch.tanh(distance / distance_std)
    gripper_closed = (gripper_opening < close_threshold).float()
    return near_object * gripper_closed


def stable_base_bonus(
    env: ManagerBasedRLEnv,
    roll_pitch_std: float,
    vertical_vel_std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    projected_gravity = robot.data.projected_gravity_b
    tilt = torch.norm(projected_gravity[:, :2], dim=1)
    vertical_vel = torch.abs(robot.data.root_lin_vel_b[:, 2])
    tilt_bonus = torch.exp(-(tilt / roll_pitch_std) ** 2)
    vel_bonus = torch.exp(-(vertical_vel / vertical_vel_std) ** 2)
    return tilt_bonus * vel_bonus


def success_bonus(
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
    return (lifted & near_ee & closed).float()
