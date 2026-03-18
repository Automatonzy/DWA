# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from robot_lab.assets import GO2_X5_CFG  # isort: skip


@configclass
class Go2X5RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    dog_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    arm_joint_names = [
        "arm_joint1", "arm_joint2", "arm_joint3",
        "arm_joint4", "arm_joint5", "arm_joint6",
    ]
    # fmt: on

    # Unified joint names: 12 dog joints + 6 arm joints = 18 total
    joint_names = dog_joint_names + arm_joint_names

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        # enable height scan for rough terrain adaptation
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        # Include all joints (dog + arm) in observations for unified policy
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.arm_joint_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "arm_joint_pos"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.arm_joint_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "arm_joint_pos"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # ------------------------------Commands------------------------------
        self.commands.arm_joint_pos = mdp.ArmJointPositionCommandCfg(
            asset_name="robot",
            joint_names=self.arm_joint_names,
            resampling_time_range=(2.0, 4.0),
            position_range=(-0.25, 0.25),
            use_default_offset=True,
            clip_to_joint_limits=True,
            preserve_order=True,
        )
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        # ------------------------------Arm Pose Sequences (Play)------------------------------
        # Pose set used by play.py when no arm_cmd overrides are provided.
        # Inherited by flat config to keep sequences in sync.
        self.arm_pose_set = [
            [0.98, 3.05, 1.67, 0.29, 0.40, 0.40],
            [-0.98, 3.05, 1.67, 0.29, 0.40, 0.40],
            [1.32, 2.13, 2.13, -0.23, 0.06, 1.55],
            [2.05, 2.75, 2.95, -0.2, 0.1, 0.25],
            [0.0, 2.75, 2.75, -0.2, -0.2, 0.25],
        ]

        # ------------------------------Actions------------------------------
        # Unified action space: 12 dog joints + 6 arm joints = 18 total
        # Use different action scales for different joint types
        self.actions.joint_pos.scale = {
            ".*_hip_joint": 0.125,          # Dog hip joints
            ".*_thigh_joint": 0.25,         # Dog thigh joints
            ".*_calf_joint": 0.25,          # Dog calf joints
            "arm_joint.*": 0.1,             # Arm joints (smaller scale for stability)
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names  # All 18 joints

        # ------------------------------Events------------------------------
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
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!{self.base_link_name}$).*"
        ]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        base_body_cfg = SceneEntityCfg("robot", body_names=[self.base_link_name])
        arm_joint_cfg = SceneEntityCfg("robot", joint_names=self.arm_joint_names, preserve_order=True)

        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -3.2
        self.rewards.ang_vel_xy_l2.weight = -0.15
        self.rewards.flat_orientation_l2.weight = -1.1
        self.rewards.base_height_l2.weight = -1.0
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.params["asset_cfg"] = base_body_cfg
        self.rewards.body_lin_acc_l2.weight = -0.04
        self.rewards.body_lin_acc_l2.params["asset_cfg"] = base_body_cfg

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.015

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.2
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 4.8
        self.rewards.track_ang_vel_z_exp.weight = 2.0

        # Others
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.15
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.6
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 1.0

        # ------------------------------Arm Rewards------------------------------
        # Track arm commands while keeping motions bounded
        self.rewards.arm_joint_pos_tracking_l2 = RewTerm(
            func=mdp.arm_joint_pos_tracking_l2,
            weight=-3.5,
            params={"command_name": "arm_joint_pos", "asset_cfg": arm_joint_cfg},
        )

        # Arm stability rewards - keep movement smooth and within limits
        self.rewards.arm_joint_vel_l2.weight = -0.002
        self.rewards.arm_joint_vel_l2.params["asset_cfg"].joint_names = self.arm_joint_names

        self.rewards.arm_joint_acc_l2.weight = -2.0e-6
        self.rewards.arm_joint_acc_l2.params["asset_cfg"].joint_names = self.arm_joint_names

        self.rewards.arm_joint_torques_l2.weight = -1.0e-4
        self.rewards.arm_joint_torques_l2.params["asset_cfg"].joint_names = self.arm_joint_names

        self.rewards.arm_action_rate_l2.weight = -0.015
        self.rewards.arm_action_rate_l2.params["asset_cfg"].joint_names = self.arm_joint_names

        self.rewards.arm_joint_pos_limits.weight = -5.0
        self.rewards.arm_joint_pos_limits.params["asset_cfg"].joint_names = self.arm_joint_names

        self.rewards.arm_joint_deviation_l2.weight = -0.05
        self.rewards.arm_joint_deviation_l2.params["asset_cfg"].joint_names = self.arm_joint_names

        # Base-arm coupling terms
        self.rewards.arm_motion_tilt_penalty = RewTerm(
            func=mdp.arm_motion_tilt_penalty,
            weight=-0.6,
            params={
                "base_asset_cfg": base_body_cfg,
                "arm_asset_cfg": arm_joint_cfg,
                "tilt_clip": 1.0,
                "vel_clip": 6.0,
            },
        )

        self.rewards.arm_action_in_unstable_base = RewTerm(
            func=mdp.arm_action_in_unstable_base,
            weight=-0.18,
            params={
                "arm_asset_cfg": arm_joint_cfg,
                "base_asset_cfg": base_body_cfg,
                "tilt_threshold": 0.18,
                "lin_vel_z_threshold": 0.4,
                "ang_vel_threshold": 1.5,
            },
        )

        self.rewards.arm_stable_track_bonus = RewTerm(
            func=mdp.arm_stable_track_exp,
            weight=2.5,
            params={
                "command_name": "arm_joint_pos",
                "arm_asset_cfg": arm_joint_cfg,
                "base_asset_cfg": base_body_cfg,
                "tracking_std": 0.1,
                "tilt_std": 0.18,
                "vel_z_std": 0.2,
                "command_scale": 0.25,
            },
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2X5RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None
        self.terminations.terrain_out_of_bounds = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.scene.terrain.max_init_terrain_level = 4

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
