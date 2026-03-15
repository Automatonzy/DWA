# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import Go2X5RoughEnvCfg


@configclass
class Go2X5FlatEnvCfg(Go2X5RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # re-enable arm actions for joint tracking (larger scale for wider reach)
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
            "arm_joint1": 1.2,
            "arm_joint2": 1.2,
            "arm_joint3": 1.2,
            "arm_joint4": 0.8,
            "arm_joint5": 0.7,
            "arm_joint6": 0.7,
        }
        # bias arm commands toward forward/back swing (larger on shoulder/elbow)
        if self.commands.arm_joint_pos is not None:
            self.commands.arm_joint_pos.position_range = [
                (-1.2, 1.2),   # arm_joint1 (yaw)
                (-1.2, 1.2),   # arm_joint2 (pitch)
                (-1.2, 1.2),   # arm_joint3 (pitch)
                (-0.8, 0.8),   # arm_joint4
                (-0.7, 0.7),   # arm_joint5
                (-0.7, 0.7),   # arm_joint6
            ]
            self.commands.arm_joint_pos.resampling_time_range = (4.0, 6.0)

        # make base command stay-still more often for arm tracking
        self.commands.base_velocity.rel_standing_envs = 0.3
        self.commands.base_velocity.resampling_time_range = (6.0, 8.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # reward tuning for flat-terrain baseline (simplified + less punitive)
        self.rewards.track_lin_vel_xy_exp.weight = 3.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # base stability emphasis (softer to avoid oscillation)
        self.rewards.lin_vel_z_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.3
        self.rewards.base_height_l2.weight = -0.2
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.stand_still.weight = -1.0
        self.rewards.stand_still.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=self.dog_joint_names)
        self.rewards.joint_pos_penalty.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.dog_joint_names
        )

        # joint / contact penalties (soften)
        self.rewards.joint_torques_l2.weight = -1.5e-5
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_power.weight = -1.0e-5
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces.weight = -1.0e-4
        self.rewards.feet_slide.weight = -0.08
        self.rewards.feet_gait.weight = 0.5
        self.rewards.upward.weight = 1.0

        # arm tracking and smoothness
        self.rewards.arm_joint_pos_tracking_l2.weight = -2.0
        self.rewards.arm_joint_vel_l2.weight = -0.001
        self.rewards.arm_joint_acc_l2.weight = -5.0e-7
        self.rewards.arm_joint_torques_l2.weight = -5.0e-5
        self.rewards.arm_action_rate_l2.weight = -0.005
        self.rewards.arm_joint_pos_limits.weight = -1.0
        self.rewards.arm_joint_deviation_l2.weight = 0.0

        # base-arm coordination
        self.rewards.arm_motion_tilt_penalty.weight = -0.2
        self.rewards.arm_action_in_unstable_base.weight = -0.05
        self.rewards.arm_stable_track_bonus.weight = 3.0
        self.rewards.arm_stable_track_bonus.params["tracking_std"] = 0.3
        self.rewards.arm_stable_track_bonus.params["tilt_std"] = 0.25
        self.rewards.arm_stable_track_bonus.params["vel_z_std"] = 0.35
        self.rewards.arm_stable_track_bonus.params["command_scale"] = 0.2

        # ------------------------------Sim2Sim Randomization (Flat)------------------------------
        # IMU noise (policy observations)
        if self.observations.policy.base_ang_vel is not None:
            self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.02, n_max=0.02)
        if self.observations.policy.projected_gravity is not None:
            self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.03, n_max=0.03)

        # PD gain drift (interval) + broader friction ranges for sim2sim transfer
        self.events.randomize_actuator_gains.mode = "interval"
        self.events.randomize_actuator_gains.interval_range_s = (1.0, 3.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.7, 1.3)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.5, 1.5)
        self.events.randomize_actuator_gains.params["distribution"] = "uniform"

        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.3, 1.3)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.2, 1.1)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.4)

        # Small external disturbances on base
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-8.0, 8.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-2.5, 2.5)
        self.events.randomize_apply_external_force_torque.mode = "interval"
        self.events.randomize_apply_external_force_torque.interval_range_s = (8.0, 15.0)

        # Gentle push perturbations (interval)
        self.events.randomize_push_robot.interval_range_s = (10.0, 18.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}

        # Control timing uncertainty (used by ActionDelayWrapper in train/play)
        self.sim2sim_action_delay_range = (1, 3)
        self.sim2sim_action_hold_prob = 0.1
        self.sim2sim_action_noise_std = 0.01

        # Observation delay (policy only)
        self.sim2sim_obs_delay_steps = 1
        delay_steps = int(self.sim2sim_obs_delay_steps)
        if delay_steps > 0:
            def _apply_delay(term, func):
                if term is None:
                    return
                term.func = func
                if term.params is None:
                    term.params = {}
                term.params["delay_steps"] = delay_steps

            _apply_delay(self.observations.policy.base_ang_vel, mdp.delayed_base_ang_vel)
            _apply_delay(self.observations.policy.projected_gravity, mdp.delayed_projected_gravity)
            _apply_delay(self.observations.policy.joint_pos, mdp.delayed_joint_pos_rel)
            _apply_delay(self.observations.policy.joint_vel, mdp.delayed_joint_vel_rel)
            _apply_delay(self.observations.policy.actions, mdp.delayed_last_action)
            _apply_delay(self.observations.policy.velocity_commands, mdp.delayed_generated_commands)
            _apply_delay(self.observations.policy.arm_joint_command, mdp.delayed_generated_commands)
            if self.observations.policy.velocity_commands is not None:
                self.observations.policy.velocity_commands.params["command_name"] = "base_velocity"
            if self.observations.policy.arm_joint_command is not None:
                self.observations.policy.arm_joint_command.params["command_name"] = "arm_joint_pos"

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2X5FlatEnvCfg":
            self.disable_zero_weight_rewards()
