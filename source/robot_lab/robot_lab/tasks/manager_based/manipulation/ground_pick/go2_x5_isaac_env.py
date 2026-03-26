# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import robot_lab.tasks  # noqa: F401

from .go2_x5_ground_pick_env_cfg import Go2X5GroundPickEnvCfg


class Go2X5GroundPickIsaacEnv:
    """Adapter-friendly wrapper around the registered Go2-X5 Isaac ground-pick task.

    This wrapper keeps the same interface as the smoke env used by SimpleVLA-RL:
    `reset/get_obs/get_instruction/step/close`, while the underlying simulation uses the
    ManagerBasedRLEnv registered in Go2-X5-lab.
    """

    requires_isaac_app = True
    env_id = "RobotLab-Isaac-GroundPick-Go2-X5-v0"

    def __init__(
        self,
        task_name: str,
        task_id: int,
        trial_id: int,
        trial_seed: int,
        image_size: int = 224,
        max_episode_steps: int = 160,
        instruction: str = "pick up the red block from the ground",
        config=None,
        **kwargs,
    ) -> None:
        del task_id, kwargs
        self.task_name = task_name
        self.trial_id = trial_id
        self.trial_seed = trial_seed
        self.image_size = int(image_size)
        self.max_episode_steps = int(max_episode_steps)
        self.instruction = instruction
        self.config = config or {}
        self.device = self.config.get("go2_device", "cuda:0")
        self.env = None
        self._obs = None
        self._step_count = 0

    def reset(self):
        if self.env is None:
            self._build_env()
        obs = self.env.reset(seed=self.trial_seed)
        if isinstance(obs, tuple):
            obs = obs[0]
        self._step_count = 0
        self._obs = self._collect_obs()
        return self._obs

    def get_obs(self):
        if self._obs is None:
            raise RuntimeError("Environment has not been reset.")
        return self._obs

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def get_instruction(self) -> str:
        return self.instruction

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Environment has not been reset.")
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        action_tensor = torch.as_tensor(action, device=self.env.unwrapped.device)
        obs, reward, terminated, truncated, info = self.env.step(action_tensor)
        del obs, reward, info
        self._step_count += 1
        self._obs = self._collect_obs()
        terminated = bool(terminated[0].item()) if torch.is_tensor(terminated) else bool(terminated)
        truncated = bool(truncated[0].item()) if torch.is_tensor(truncated) else bool(truncated)
        success = self._compute_success()
        return self._obs, 1.0 if success else 0.0, terminated, truncated, {
            "success": success,
            "object_height": self._object_height(),
            "eef_object_distance": self._eef_object_distance(),
            "step_count": self._step_count,
        }

    def _build_env(self):
        env_cfg: Go2X5GroundPickEnvCfg = parse_env_cfg(self.env_id, device=self.device, num_envs=1)
        env_cfg.seed = self.trial_seed
        env_cfg.scene.num_envs = 1
        env_cfg.scene.env_spacing = 4.0
        env_cfg.episode_length_s = self.max_episode_steps * env_cfg.sim.dt * env_cfg.decimation
        env_cfg.scene.dog_camera.height = self.image_size
        env_cfg.scene.dog_camera.width = self.image_size
        env_cfg.scene.arm_camera.height = self.image_size
        env_cfg.scene.arm_camera.width = self.image_size
        env_cfg.observations.policy.enable_corruption = False
        low_level_policy_path = self.config.get("go2_low_level_policy_path", None)
        if low_level_policy_path:
            env_cfg.actions.base_policy.policy_path = str(low_level_policy_path)
        if not getattr(env_cfg.actions.base_policy, "policy_path", ""):
            raise ValueError(
                "go2_low_level_policy_path must point to the frozen Go2 locomotion policy when using the Isaac ground-pick task."
            )
        self.env = gym.make(self.env_id, cfg=env_cfg, render_mode="rgb_array")

    def _collect_obs(self):
        scene = self.env.unwrapped.scene
        dog_rgb = scene["dog_camera"].data.output["rgb"][0].detach().cpu().numpy()
        arm_rgb = scene["arm_camera"].data.output["rgb"][0].detach().cpu().numpy()
        if dog_rgb.shape[-1] == 4:
            dog_rgb = dog_rgb[..., :3]
        if arm_rgb.shape[-1] == 4:
            arm_rgb = arm_rgb[..., :3]
        return {
            "dog_camera_image": dog_rgb.astype(np.uint8, copy=False),
            "arm_camera_image": arm_rgb.astype(np.uint8, copy=False),
        }

    def _object_height(self) -> float:
        obj = self.env.unwrapped.scene["object"]
        return float(obj.data.root_pos_w[0, 2].item())

    def _eef_object_distance(self) -> float:
        obj = self.env.unwrapped.scene["object"]
        ee_frame = self.env.unwrapped.scene["ee_frame"]
        dist = torch.norm(obj.data.root_pos_w[0, :3] - ee_frame.data.target_pos_w[0, 0, :], dim=0)
        return float(dist.item())

    def _compute_success(self) -> bool:
        obj_height = self._object_height()
        eef_distance = self._eef_object_distance()
        robot = self.env.unwrapped.scene["robot"]
        joint_ids, _ = robot.find_joints(["arm_joint7", "arm_joint8"], preserve_order=True)
        gripper_opening = float(robot.data.joint_pos[0, joint_ids].mean().item())
        return obj_height > 0.12 and eef_distance < 0.14 and gripper_opening < 0.018
