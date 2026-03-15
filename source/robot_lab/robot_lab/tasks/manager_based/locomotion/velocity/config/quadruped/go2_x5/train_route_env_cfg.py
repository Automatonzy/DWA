# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.assets import GO2_X5_CFG
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


HEIGHT_SCAN_DIM = 187

FLAT_FOUNDATION_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(220.0, 220.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
    },
)


def _zero_height_scan(env, sensor_cfg=None):
    buffer = getattr(env, "_go2_x5_flat_height_scan_zeros", None)
    if buffer is None or buffer.shape[0] != env.num_envs:
        buffer = torch.zeros((env.num_envs, HEIGHT_SCAN_DIM), device=env.device)
        env._go2_x5_flat_height_scan_zeros = buffer
    return buffer


@configclass
class _Go2X5LeggedBaseEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    dog_joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4096
        self.decimation = 8
        self.episode_length_s = 20.0
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation

        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        self.commands.arm_joint_pos = None
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0

        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.dog_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.dog_joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.dog_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.dog_joint_names

        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,
            ".*_thigh_joint": 0.25,
            ".*_calf_joint": 0.25,
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.dog_joint_names

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (0.0, 0.1),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.25, 0.25),
                "y": (-0.25, 0.25),
                "z": (-0.25, 0.25),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            },
        }
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.dog_joint_names

        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.dog_joint_names
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.root_height_below_minimum = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.18, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.root_height_above_maximum = DoneTerm(
            func=mdp.root_height_above_maximum,
            params={"maximum_height": 0.65, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.root_lin_vel_z_above_maximum = DoneTerm(
            func=mdp.root_lin_vel_z_above_maximum,
            params={"maximum_speed": 3.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.root_ang_vel_xy_above_maximum = DoneTerm(
            func=mdp.root_ang_vel_xy_above_maximum,
            params={"maximum_speed": 8.0, "asset_cfg": SceneEntityCfg("robot")},
        )


@configclass
class Go2X5FoundationFlatEnvCfg(_Go2X5LeggedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = FLAT_FOUNDATION_TERRAIN_CFG
        self.scene.terrain.use_terrain_origins = False
        self.scene.terrain.visual_material = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = ObsTerm(func=_zero_height_scan, clip=(-1.0, 1.0), scale=1.0)
        self.observations.critic.height_scan = ObsTerm(func=_zero_height_scan, clip=(-1.0, 1.0), scale=1.0)
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.rel_standing_envs = 0.15
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.5, 1.25)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.45, 1.1)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.2)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.9, 1.1)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "scale"
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.95, 1.05)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
        }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_push_robot.interval_range_s = (8.0, 14.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}

        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.03, n_max=0.03)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.02, n_max=0.02)

        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.weight = -0.08
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = -0.2
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.body_lin_acc_l2.weight = -0.01
        self.rewards.joint_torques_l2.weight = -1.5e-5
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = -1.0e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -0.8
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.contact_forces.weight = -1.0e-4
        self.rewards.track_lin_vel_xy_exp.weight = 4.0
        self.rewards.track_ang_vel_z_exp.weight = 1.8
        self.rewards.feet_air_time.weight = 0.15
        self.rewards.feet_air_time.params["threshold"] = 0.45
        self.rewards.feet_air_time_variance.weight = -0.5
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.15
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_slide.weight = -0.08
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_gait.weight = 0.25
        self.rewards.upward.weight = 1.0

        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.7, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.7, 1.0)

        self.disable_zero_weight_rewards()


@configclass
class Go2X5RobustRoughEnvCfg(_Go2X5LeggedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Rough terrain is materially heavier than the flat foundation task.
        # Keep the default env count aligned with the proven P1 training setup.
        self.scene.num_envs = 2048
        self.scene.terrain.max_init_terrain_level = 3

        self.commands.base_velocity.rel_standing_envs = 0.3
        self.commands.base_velocity.resampling_time_range = (5.0, 7.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.2, 1.5)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.2, 1.3)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.2)
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "scale"
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.85, 1.15)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.02, 0.02),
            "z": (-0.02, 0.02),
        }
        self.events.randomize_actuator_gains.mode = "interval"
        self.events.randomize_actuator_gains.interval_range_s = (1.0, 3.0)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)
        self.events.randomize_apply_external_force_torque.mode = "interval"
        self.events.randomize_apply_external_force_torque.interval_range_s = (8.0, 15.0)
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-10.0, 10.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-3.0, 3.0)
        self.events.randomize_push_robot.interval_range_s = (8.0, 14.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}

        self.observations.policy.base_ang_vel.noise = Unoise(n_min=-0.03, n_max=0.03)
        self.observations.policy.projected_gravity.noise = Unoise(n_min=-0.02, n_max=0.02)

        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -1.8
        self.rewards.ang_vel_xy_l2.weight = -0.12
        self.rewards.flat_orientation_l2.weight = -0.7
        self.rewards.base_height_l2.weight = -0.30
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.body_lin_acc_l2.weight = -0.015
        self.rewards.joint_torques_l2.weight = -1.8e-5
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1.5e-7
        self.rewards.joint_pos_limits.weight = -3.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.joint_power.weight = -1.5e-5
        self.rewards.stand_still.weight = -2.5
        self.rewards.joint_pos_penalty.weight = -0.9
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.015
        self.rewards.undesired_contacts.weight = -1.2
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.track_lin_vel_xy_exp.weight = 4.2
        self.rewards.track_ang_vel_z_exp.weight = 1.9
        self.rewards.feet_air_time.weight = 0.08
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time_variance.weight = -0.8
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.12
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_slide.weight = -0.18
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_gait.weight = 0.35
        self.rewards.upward.weight = 1.0

        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)

        self.sim2sim_action_delay_range = (0, 2)
        self.sim2sim_action_hold_prob = 0.05
        self.sim2sim_action_noise_std = 0.01
        self.sim2sim_obs_delay_steps = 1
        delay_steps = int(self.sim2sim_obs_delay_steps)
        if delay_steps > 0:
            delayed_terms = [
                ("base_lin_vel", mdp.delayed_base_lin_vel),
                ("base_ang_vel", mdp.delayed_base_ang_vel),
                ("projected_gravity", mdp.delayed_projected_gravity),
                ("joint_pos", mdp.delayed_joint_pos_rel),
                ("joint_vel", mdp.delayed_joint_vel_rel),
                ("actions", mdp.delayed_last_action),
                ("velocity_commands", mdp.delayed_generated_commands),
            ]
            for term_name, func in delayed_terms:
                term = getattr(self.observations.policy, term_name, None)
                if term is None:
                    continue
                term.func = func
                if term.params is None:
                    term.params = {}
                term.params["delay_steps"] = delay_steps
            if self.observations.policy.velocity_commands is not None:
                self.observations.policy.velocity_commands.params["command_name"] = "base_velocity"

        self.terminations.illegal_contact = None
        self.terminations.terrain_out_of_bounds = None

        self.disable_zero_weight_rewards()
