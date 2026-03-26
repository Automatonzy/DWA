# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.navigation.mdp.pre_trained_policy_action import PreTrainedPolicyActionCfg

from robot_lab.assets import GO2_X5_CFG
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as loco_mdp

from . import mdp


DOG_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
ARM_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6",
]
GRIPPER_JOINT_NAMES = ["arm_joint7", "arm_joint8"]


@configclass
class Go2X5GroundPickSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.56, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.16, 0.12), metallic=0.0),
        ),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/arm_link6",
                name="end_effector",
                offset=OffsetCfg(pos=(0.08657, 0.0, 0.0)),
            ),
        ],
    )

    dog_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/dog_vla_camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.30, 0.0, 0.16),
            rot=(-0.3799, 0.5963, 0.5963, -0.3799),
            convention="ros",
        ),
    )

    arm_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/arm_link6/arm_vla_camera",
        update_period=0.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.08657, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class LowLevelPolicyObservationsCfg(ObsGroup):
    base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, scale=0.25)
    projected_gravity = ObsTerm(func=loco_mdp.projected_gravity, scale=1.0)
    velocity_commands = ObsTerm(func=loco_mdp.generated_commands, params={"command_name": "base_velocity"}, scale=1.0)
    joint_pos = ObsTerm(
        func=loco_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES, preserve_order=True)},
        scale=1.0,
    )
    joint_vel = ObsTerm(
        func=loco_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES, preserve_order=True)},
        scale=0.05,
    )
    actions = ObsTerm(func=loco_mdp.last_action, scale=1.0)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ActionsCfg:
    base_policy = PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=os.environ.get("GO2_X5_LOW_LEVEL_POLICY_PATH", ""),
        low_level_decimation=4,
        low_level_actions=mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=DOG_JOINT_NAMES,
            scale={
                ".*_hip_joint": 0.125,
                ".*_thigh_joint": 0.25,
                ".*_calf_joint": 0.25,
            },
            use_default_offset=True,
            clip=None,
            preserve_order=True,
        ),
        low_level_observations=LowLevelPolicyObservationsCfg(),
        debug_vis=False,
    )
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale={
            "arm_joint1": 1.2,
            "arm_joint2": 1.2,
            "arm_joint3": 1.2,
            "arm_joint4": 0.8,
            "arm_joint5": 0.7,
            "arm_joint6": 0.7,
        },
        use_default_offset=True,
        clip=None,
        preserve_order=True,
    )
    gripper_action = mdp.AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={"arm_joint7": 0.044, "arm_joint8": 0.044},
        close_command_expr={"arm_joint7": 0.0, "arm_joint8": 0.0},
        threshold=0.022,
        positive_threshold=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=loco_mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=loco_mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=loco_mdp.projected_gravity, scale=1.0)
        dog_joint_pos = ObsTerm(
            func=loco_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        dog_joint_vel = ObsTerm(
            func=loco_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=DOG_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.0, n_max=1.0),
            scale=0.05,
        )
        arm_joint_pos = ObsTerm(
            func=loco_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES, preserve_order=True)},
            scale=1.0,
        )
        arm_joint_vel = ObsTerm(
            func=loco_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_height = ObsTerm(func=mdp.object_height)
        ee_to_object = ObsTerm(func=mdp.ee_to_object_vector)
        gripper_opening = ObsTerm(func=mdp.gripper_opening)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.08, 0.08),
                "y": (-0.18, 0.18),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14159, 3.14159),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.12}, weight=2.5)
    grasp_closure = RewTerm(
        func=mdp.gripper_closed_around_object,
        params={"distance_std": 0.10, "close_threshold": 0.018},
        weight=2.0,
    )
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.08}, weight=8.0)
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        params={
            "minimal_height": 0.12,
            "max_eef_object_distance": 0.14,
            "close_threshold": 0.018,
        },
        weight=20.0,
    )
    stable_base = RewTerm(
        func=mdp.stable_base_bonus,
        params={"roll_pitch_std": 0.35, "vertical_vel_std": 0.35},
        weight=1.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    arm_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES, preserve_order=True)},
        weight=-1.0e-4,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    robot_fall = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.18, "asset_cfg": SceneEntityCfg("robot")},
    )
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.02, "asset_cfg": SceneEntityCfg("object")},
    )
    success = DoneTerm(
        func=mdp.ground_pick_success,
        params={
            "minimal_height": 0.12,
            "max_eef_object_distance": 0.14,
            "close_threshold": 0.018,
        },
    )


@configclass
class Go2X5GroundPickEnvCfg(ManagerBasedRLEnvCfg):
    scene: Go2X5GroundPickSceneCfg = Go2X5GroundPickSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024

        self.scene.robot = GO2_X5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.38),
            joint_pos={
                ".*L_hip_joint": 0.0,
                ".*R_hip_joint": 0.0,
                "F.*_thigh_joint": 0.8,
                "R.*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
                "arm_joint1": 0.0,
                "arm_joint2": 1.6,
                "arm_joint3": 1.2,
                "arm_joint4": 0.0,
                "arm_joint5": 0.0,
                "arm_joint6": 0.0,
                "arm_joint7": 0.044,
                "arm_joint8": 0.044,
            },
            joint_vel={".*": 0.0},
        )

        self.export_io_descriptors = True
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.4)


@configclass
class Go2X5GroundPickEnvCfg_PLAY(Go2X5GroundPickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
