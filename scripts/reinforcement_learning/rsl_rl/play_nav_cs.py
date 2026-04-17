"""Navigation playback rebuilt on top of the known-good play_cs.py bring-up flow."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "navigation")))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play an RL locomotion policy with a navigation controller.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=20000000, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Flat-Go2-X5-ArmUnlock-v0",
    help="Task name.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard camera follow.")
parser.add_argument("--map", type=str, required=True, help="USD map path.")
parser.add_argument("--nav-map", type=str, required=True, help="Navigation map metadata path.")
parser.add_argument("--path", type=str, default=None, help="Optional precomputed path bundle json.")
parser.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"), help="Goal point in world coordinates.")
parser.add_argument("--goal-yaw", type=float, default=None, help="Optional desired final yaw in radians.")
parser.add_argument(
    "--goal-yaw-tolerance",
    type=float,
    default=0.20,
    help="Allowed absolute final yaw error in radians when --goal-yaw is provided.",
)
parser.add_argument(
    "--spawn",
    type=float,
    nargs=3,
    default=(-2.03488, 5.00164, 1.58),
    metavar=("X", "Y", "YAW"),
    help="Robot spawn pose in world coordinates.",
)
parser.add_argument(
    "--inflate-radius",
    type=float,
    default=0.30,
    help="Obstacle inflation radius in meters used during planning.",
)
parser.add_argument(
    "--local-clearance-radius",
    type=float,
    default=0.12,
    help="Extra obstacle inflation radius in meters used only by the local DWA collision checks.",
)
parser.add_argument("--goal-tolerance", type=float, default=0.35, help="Goal tolerance in meters.")
parser.add_argument("--lookahead-distance", type=float, default=0.6, help="Lookahead distance in meters.")
parser.add_argument("--waypoint-tolerance", type=float, default=0.2, help="Waypoint tolerance in meters.")
parser.add_argument("--max-lin-vel", type=float, default=0.5, help="Maximum linear velocity for the DWA controller.")
parser.add_argument("--max-ang-vel", type=float, default=1.0, help="Maximum yaw rate for the DWA controller.")
parser.add_argument("--max-steps", type=int, default=3000, help="Maximum navigation steps before timeout.")
parser.add_argument("--settle-steps", type=int, default=120, help="Initial steps to hold zero command.")
parser.add_argument("--debug-print-every", type=int, default=60, help="Print debug info every N steps. <=0 disables.")
parser.add_argument(
    "--debug-command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Optional fixed base command applied after settling for debugging.",
)
parser.add_argument(
    "--debug-command-steps",
    type=int,
    default=120,
    help="Number of steps to apply --debug-command after settling.",
)
parser.add_argument(
    "--dataset-dir",
    type=str,
    default=None,
    help="Root directory for saving episode data in example_dataset format. "
         "Defaults to <script_dir>/../../../episodes.",
)
parser.add_argument(
    "--task-id",
    type=int,
    default=1,
    help="Task ID (top-level folder under --dataset-dir). Fixed per task type; defaults to 1.",
)
parser.add_argument(
    "--head-camera",
    action="store_true",
    default=False,
    help="Attach a forward-looking RGB camera on the robot's head and save images.",
)
parser.add_argument(
    "--head-camera-height", type=int, default=480, help="Head camera image height in pixels."
)
parser.add_argument(
    "--head-camera-width", type=int, default=640, help="Head camera image width in pixels."
)
parser.add_argument(
    "--head-camera-save-every",
    type=int,
    default=1,
    help="Save a head-camera image every N steps (0 = never save, just render).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video or args_cli.head_camera:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from rl_utils import camera_follow

import gymnasium as gym
import numpy as np
import torch

from navlib import AStarPlanner, DWAConfig, DWAController, OccupancyGridMap
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

from PIL import Image


def _quat_to_yaw(quat_wxyz: torch.Tensor) -> float:
    w = float(quat_wxyz[0].item())
    x = float(quat_wxyz[1].item())
    y = float(quat_wxyz[2].item())
    z = float(quat_wxyz[3].item())
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _final_yaw_command(yaw_error: float, max_ang_vel: float) -> float:
    align_gain = 1.5
    align_limit = min(max_ang_vel, 0.35)
    return float(np.clip(align_gain * yaw_error, -align_limit, align_limit))


def _robot_pose(robot) -> tuple[float, float, float]:
    pos = robot.data.root_pos_w[0]
    quat = robot.data.root_quat_w[0]
    return float(pos[0].item()), float(pos[1].item()), _quat_to_yaw(quat)


def _robot_height(robot) -> float:
    return float(robot.data.root_pos_w[0][2].item())


def _robot_speed(robot) -> tuple[float, float]:
    lin_vel = robot.data.root_lin_vel_b[0]
    ang_vel = robot.data.root_ang_vel_b[0]
    return float(lin_vel[0].item()), float(ang_vel[2].item())


def _configure_env(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/scene_collision",
        terrain_type="usd",
        usd_path=args_cli.map,
        debug_vis=False,
    )
    env_cfg.scene.sky_light = None

    spawn_x, spawn_y, spawn_yaw = args_cli.spawn
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {
            "x": (spawn_x, spawn_x),
            "y": (spawn_y, spawn_y),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (spawn_yaw, spawn_yaw),
        },
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }

    env_cfg.events.randomize_rigid_body_material.params["static_friction_range"] = (1.0, 1.0)
    env_cfg.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (1.0, 1.0)
    env_cfg.events.randomize_rigid_body_material.params["restitution_range"] = (0.5, 0.5)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-2.0, 2.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.terminations.time_out = None
    env_cfg.terminations.illegal_contact = None
    env_cfg.terminations.terrain_out_of_bounds = None

    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    if args_cli.head_camera:
        # Attach an RGB camera to the robot's head (base link, slightly forward and up).
        # Quaternion (w, x, y, z) = (0.5, -0.5, 0.5, -0.5) rotates the ROS camera frame
        # so its +Z optical axis aligns with the robot's +X (forward), image-up = world +Z.
        env_cfg.scene.head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/head_cam",
            update_period=0.0,  # update every physics step
            height=args_cli.head_camera_height,
            width=args_cli.head_camera_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                # ~28 cm forward and 7 cm up from base origin — top of Go2 head area
                pos=(0.28, 0.0, 0.07),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )


def _load_path_from_bundle(path_file: str | Path) -> list[tuple[float, float]]:
    bundle = json.loads(Path(path_file).read_text())
    path_world = [(float(x), float(y)) for x, y in bundle["path_world"]]
    if len(path_world) < 2:
        raise ValueError("precomputed path must contain at least two waypoints")
    return path_world


def _next_episode_id(dataset_dir: Path) -> int:
    """Return the next available integer episode ID by scanning existing numeric subdirs."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(d.name) for d in dataset_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    return max(existing, default=0) + 1


def _maybe_prepend_start(path_world: list[tuple[float, float]], start_xy: tuple[float, float]) -> list[tuple[float, float]]:
    if not path_world:
        return [start_xy]
    if math.hypot(path_world[0][0] - start_xy[0], path_world[0][1] - start_xy[1]) > 0.05:
        return [start_xy] + path_world
    return path_world


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    _configure_env(env_cfg)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_nav_cs"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0

    arm_term = env.unwrapped.command_manager._terms.get("arm_joint_pos", None)
    base_cmd_term = env.unwrapped.command_manager._terms.get("base_velocity", None)
    if base_cmd_term is None:
        raise RuntimeError("base_velocity command term is not active for this task.")

    robot = env.unwrapped.scene["robot"]
    head_camera = env.unwrapped.scene["head_camera"] if args_cli.head_camera else None

    arm_home_pos = torch.zeros(1, 6, dtype=torch.float32, device=env.unwrapped.device) if arm_term is not None else None
    gripper_joint_ids, _ = robot.find_joints(["arm_joint7", "arm_joint8"], preserve_order=True)
    if len(gripper_joint_ids) != 2:
        gripper_joint_ids = None
    gripper_closed_pos = (
        torch.zeros(1, 2, dtype=torch.float32, device=env.unwrapped.device) if gripper_joint_ids is not None else None
    )
    # Joint ids for the 6-DOF arm (for episode logging)
    arm6_joint_ids, _ = robot.find_joints(
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
        preserve_order=True,
    )

    start_pose = _robot_pose(robot)
    raw_nav_map = OccupancyGridMap.from_meta_file(args_cli.nav_map)
    nav_map = raw_nav_map.inflate(args_cli.inflate_radius)
    local_nav_map = raw_nav_map.inflate(args_cli.local_clearance_radius)

    if args_cli.path is None and args_cli.goal is None:
        raise ValueError("Either --path or --goal is required.")

    planner = AStarPlanner(allow_diagonal=True, heuristic_weight=1.0) if args_cli.path is None else None
    bundled_path_world = _load_path_from_bundle(args_cli.path) if args_cli.path is not None else None
    dwa_cfg = DWAConfig(
        control_dt=dt,
        lookahead_distance=args_cli.lookahead_distance,
        waypoint_tolerance=args_cli.waypoint_tolerance,
        goal_tolerance=args_cli.goal_tolerance,
        max_linear_velocity=args_cli.max_lin_vel,
        max_angular_velocity=args_cli.max_ang_vel,
    )
    controller: DWAController | None = None
    path_world: list[tuple[float, float]] | None = None
    plan = None
    path_initialized = False
    if args_cli.goal is not None:
        goal_xy = (float(args_cli.goal[0]), float(args_cli.goal[1]))
    elif bundled_path_world is not None:
        goal_xy = bundled_path_world[-1]
    else:
        raise ValueError("Unable to determine navigation goal.")

    # Episode directory setup (example_dataset format)
    # Structure: <dataset_dir>/<task_id>/<traj_id>/<traj_id>-1/, <traj_id>-2/
    if args_cli.dataset_dir is not None:
        dataset_root = Path(args_cli.dataset_dir).expanduser().resolve()
    else:
        dataset_root = Path(__file__).resolve().parent.parent.parent.parent / "episodes"
    task_dir = dataset_root / str(args_cli.task_id)
    traj_id = _next_episode_id(task_dir)   # auto-increment within this task
    episode_root = task_dir / str(traj_id)

    # Subtask -1: navigation to goal position
    sub1_img_dir = episode_root / f"{traj_id}-1" / "images" / "front"
    sub1_img_dir.mkdir(parents=True, exist_ok=True)
    # Subtask -2: yaw alignment at goal (only created when --goal-yaw is specified)
    sub2_img_dir: Path | None = None
    if args_cli.goal_yaw is not None:
        sub2_img_dir = episode_root / f"{traj_id}-2" / "images" / "front"
        sub2_img_dir.mkdir(parents=True, exist_ok=True)

    _CSV_HEADER = (
        "时间戳(秒),位置X,位置Y,位置Z,姿态X,姿态Y,姿态Z,姿态W,"
        "线速度X,线速度Y,线速度Z,"
        "关节1,关节2,关节3,关节4,关节5,关节6,夹爪,前摄像头图像"
    )
    sub1_rows: list[str] = []
    sub2_rows: list[str] = []
    sub1_frame_idx = 0
    sub2_frame_idx = 0

    print(f"[INFO] Task {args_cli.task_id}, trajectory {traj_id} → {episode_root}")

    success = False
    timeout = False
    last_debug: dict[str, float | int | bool] = {}
    goal_yaw_error_after = 0.0
    def _hold_success_pose(duration_s: float):
        nonlocal obs
        if duration_s <= 0.0:
            return

        dwell_dt = max(dt, 1.0e-3)
        dwell_steps = max(1, int(math.ceil(duration_s / dwell_dt)))
        zero_cmd = torch.zeros((1, 3), dtype=torch.float32, device=base_cmd_term.device)
        print(f"[INFO] Goal reached. Holding final pose for {duration_s:.1f}s before exit.")

        for _ in range(dwell_steps):
            if not simulation_app.is_running():
                break

            dwell_start = time.time()
            with torch.inference_mode():
                base_cmd_term.vel_command_b[:] = zero_cmd
                if hasattr(base_cmd_term, "is_heading_env"):
                    base_cmd_term.is_heading_env[:] = False
                if hasattr(base_cmd_term, "is_standing_env"):
                    base_cmd_term.is_standing_env[:] = True
                if hasattr(base_cmd_term, "heading_target"):
                    base_cmd_term.heading_target[:] = 0.0
                if arm_home_pos is not None and arm_term is not None:
                    arm_term.command_buffer[:] = arm_home_pos

                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                if desired_gripper_pos is not None and gripper_joint_ids is not None:
                    robot.set_joint_position_target(desired_gripper_pos, joint_ids=gripper_joint_ids)

            if args_cli.keyboard:
                camera_follow(env)

            sleep_time = dwell_dt - (time.time() - dwell_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                desired_arm_pos = arm_home_pos.clone() if arm_home_pos is not None else None
                desired_gripper_pos = gripper_closed_pos.clone() if gripper_closed_pos is not None else None
                pose_before = _robot_pose(robot)
                current_velocity = _robot_speed(robot)
                robot_height = _robot_height(robot)

                if not path_initialized and timestep >= args_cli.settle_steps:
                    settled_start_xy = (pose_before[0], pose_before[1])
                    if bundled_path_world is not None:
                        path_world = _maybe_prepend_start(list(bundled_path_world), settled_start_xy)
                        print(
                            f"[INFO] Loaded precomputed path with {len(path_world)} waypoints from {args_cli.path} "
                            f"after settle pose=({settled_start_xy[0]:.3f}, {settled_start_xy[1]:.3f})."
                        )
                    else:
                        plan = planner.plan(
                            nav_map,
                            start_xy=settled_start_xy,
                            goal_xy=goal_xy,
                            snap_to_free=True,
                            max_snap_distance_m=max(0.5, args_cli.inflate_radius + nav_map.resolution),
                        )
                        path_world = _maybe_prepend_start(plan.path_world, settled_start_xy)
                        print(
                            f"[INFO] Planned online path with {len(path_world)} waypoints and cost {plan.cost:.3f} "
                            f"from settle pose=({settled_start_xy[0]:.3f}, {settled_start_xy[1]:.3f})."
                        )
                    controller = DWAController(path_world=path_world, grid_map=local_nav_map, config=dwa_cfg)
                    path_initialized = True

                if timestep < args_cli.settle_steps or controller is None:
                    command_np = np.zeros(3, dtype=np.float32)
                    distance_to_goal = math.hypot(pose_before[0] - goal_xy[0], pose_before[1] - goal_xy[1])
                    last_debug = {
                        "target_index": 0,
                        "distance_to_target": distance_to_goal,
                        "distance_to_goal": distance_to_goal,
                        "heading_error": 0.0,
                        "reached_goal": False,
                        "clearance": 0.0,
                        "score": 0.0,
                        "settling": True,
                        "debug_command": False,
                    }
                elif args_cli.debug_command is not None and timestep < args_cli.settle_steps + args_cli.debug_command_steps:
                    command_np = np.asarray(args_cli.debug_command, dtype=np.float32)
                    debug = controller.compute_command(pose_before, current_velocity)[1]
                    last_debug = {
                        "target_index": debug.target_index,
                        "target_x": debug.target_point[0],
                        "target_y": debug.target_point[1],
                        "distance_to_target": debug.distance_to_target,
                        "distance_to_goal": debug.distance_to_goal,
                        "heading_error": debug.heading_error,
                        "reached_goal": False,
                        "clearance": debug.clearance,
                        "score": debug.score,
                        "path_distance": debug.path_distance,
                        "near_goal_tracking": debug.near_goal_tracking,
                        "sampled_candidates": debug.sampled_candidates,
                        "feasible_candidates": debug.feasible_candidates,
                        "collision_rejections": debug.collision_rejections,
                        "best_linear_velocity": debug.best_linear_velocity,
                        "best_angular_velocity": debug.best_angular_velocity,
                        "aligning_final_yaw": False,
                        "settling": False,
                        "debug_command": True,
                    }
                else:
                    distance_to_goal_before = math.hypot(pose_before[0] - goal_xy[0], pose_before[1] - goal_xy[1])
                    goal_yaw_error_before = (
                        _wrap_angle(float(args_cli.goal_yaw) - pose_before[2]) if args_cli.goal_yaw is not None else 0.0
                    )
                    yaw_goal_satisfied_before = (
                        args_cli.goal_yaw is None or abs(goal_yaw_error_before) <= args_cli.goal_yaw_tolerance
                    )

                    if args_cli.goal_yaw is not None and distance_to_goal_before <= args_cli.goal_tolerance and not yaw_goal_satisfied_before:
                        angular_command = _final_yaw_command(goal_yaw_error_before, args_cli.max_ang_vel)
                        command_np = np.array([0.0, 0.0, angular_command], dtype=np.float32)
                        last_debug = {
                            "target_index": len(controller.path_world) - 1,
                            "target_x": goal_xy[0],
                            "target_y": goal_xy[1],
                            "distance_to_target": 0.0,
                            "distance_to_goal": distance_to_goal_before,
                            "heading_error": goal_yaw_error_before,
                            "reached_goal": False,
                            "clearance": float(last_debug.get("clearance", 0.0)) if last_debug else 0.0,
                            "score": 0.0,
                            "path_distance": 0.0,
                            "near_goal_tracking": False,
                            "sampled_candidates": 0,
                            "feasible_candidates": 0,
                            "collision_rejections": 0,
                            "best_linear_velocity": 0.0,
                            "best_angular_velocity": angular_command,
                            "aligning_final_yaw": True,
                            "settling": False,
                            "debug_command": False,
                        }
                    else:
                        command_np, debug = controller.compute_command(pose_before, current_velocity)
                        last_debug = {
                            "target_index": debug.target_index,
                            "target_x": debug.target_point[0],
                            "target_y": debug.target_point[1],
                            "distance_to_target": debug.distance_to_target,
                            "distance_to_goal": debug.distance_to_goal,
                            "heading_error": debug.heading_error,
                            "reached_goal": debug.reached_goal,
                            "clearance": debug.clearance,
                            "score": debug.score,
                            "path_distance": debug.path_distance,
                            "near_goal_tracking": debug.near_goal_tracking,
                            "sampled_candidates": debug.sampled_candidates,
                            "feasible_candidates": debug.feasible_candidates,
                            "collision_rejections": debug.collision_rejections,
                            "best_linear_velocity": debug.best_linear_velocity,
                            "best_angular_velocity": debug.best_angular_velocity,
                            "aligning_final_yaw": False,
                            "settling": False,
                            "debug_command": False,
                        }

                desired_base_cmd = torch.tensor(command_np, dtype=torch.float32, device=base_cmd_term.device).unsqueeze(0)
                base_cmd_term.vel_command_b[:] = desired_base_cmd
                if hasattr(base_cmd_term, "is_heading_env"):
                    base_cmd_term.is_heading_env[:] = False
                if hasattr(base_cmd_term, "is_standing_env"):
                    base_cmd_term.is_standing_env[:] = torch.linalg.norm(desired_base_cmd, dim=1) < 1.0e-6
                if hasattr(base_cmd_term, "heading_target"):
                    base_cmd_term.heading_target[:] = 0.0
                if desired_arm_pos is not None and arm_term is not None:
                    arm_term.command_buffer[:] = desired_arm_pos

                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                if desired_gripper_pos is not None and gripper_joint_ids is not None:
                    robot.set_joint_position_target(desired_gripper_pos, joint_ids=gripper_joint_ids)

                pose_after = _robot_pose(robot)
                measured_vx, measured_wz = _robot_speed(robot)
                _lin_vel_b = robot.data.root_lin_vel_b[0]   # (3,) vx vy vz in body frame
                _arm6_pos = robot.data.joint_pos[0, arm6_joint_ids].cpu().tolist()   # [j1..j6]
                _gripper_val = float(robot.data.joint_pos[0, gripper_joint_ids].mean().item()) if gripper_joint_ids is not None else 0.0
                _arm_joints_7d = _arm6_pos + [_gripper_val]
                distance_to_goal_after = math.hypot(pose_after[0] - goal_xy[0], pose_after[1] - goal_xy[1])
                goal_yaw_error_after = (
                    _wrap_angle(float(args_cli.goal_yaw) - pose_after[2]) if args_cli.goal_yaw is not None else 0.0
                )
                yaw_goal_satisfied = args_cli.goal_yaw is None or abs(goal_yaw_error_after) <= args_cli.goal_yaw_tolerance


                if args_cli.debug_print_every > 0 and timestep % args_cli.debug_print_every == 0:
                    print(
                        f"[NAV-CS] step={timestep} pose=({pose_after[0]:.3f}, {pose_after[1]:.3f}, yaw={pose_after[2]:.3f}) "
                        f"z={robot_height:.3f} cmd=({float(command_np[0]):.3f}, {float(command_np[1]):.3f}, {float(command_np[2]):.3f}) "
                        f"measured=({measured_vx:.3f}, {measured_wz:.3f}) goal_dist={distance_to_goal_after:.3f} "
                        f"goal_yaw_err={goal_yaw_error_after:.3f} "
                        f"clearance={last_debug.get('clearance', 0.0):.3f} settling={last_debug['settling']} debug_cmd={last_debug['debug_command']}"
                    )
                    if (
                        not last_debug["settling"]
                        and not last_debug["debug_command"]
                        and (
                            distance_to_goal_after <= 1.0
                            or bool(last_debug.get("aligning_final_yaw", False))
                            or (abs(float(command_np[0])) >= 0.10 and abs(measured_vx) <= 0.03)
                        )
                    ):
                        print(
                            f"[NAV-CS-DIAG] step={timestep} target_idx={int(last_debug.get('target_index', -1))} "
                            f"target=({float(last_debug.get('target_x', 0.0)):.3f}, {float(last_debug.get('target_y', 0.0)):.3f}) "
                            f"dist_target={float(last_debug.get('distance_to_target', 0.0)):.3f} "
                            f"heading_err={float(last_debug.get('heading_error', 0.0)):.3f} "
                            f"path_dist={float(last_debug.get('path_distance', 0.0)):.3f} "
                            f"score={float(last_debug.get('score', 0.0)):.3f} "
                            f"near_goal={bool(last_debug.get('near_goal_tracking', False))} "
                            f"aligning_yaw={bool(last_debug.get('aligning_final_yaw', False))} "
                            f"goal_yaw_err={goal_yaw_error_after:.3f} "
                            f"candidates={int(last_debug.get('sampled_candidates', 0))}/"
                            f"{int(last_debug.get('feasible_candidates', 0))} "
                            f"collision_rej={int(last_debug.get('collision_rejections', 0))} "
                            f"best_cmd=({float(last_debug.get('best_linear_velocity', 0.0)):.3f}, "
                            f"{float(last_debug.get('best_angular_velocity', 0.0)):.3f})"
                        )

                # Determine subtask phase: 1=navigation, 2=yaw-alignment
                _in_yaw_align = bool(last_debug.get("aligning_final_yaw", False))

                # Quaternion (Isaac Lab: w,x,y,z) → CSV order (x,y,z,w)
                _q = robot.data.root_quat_w[0]
                _qx, _qy, _qz, _qw = float(_q[1]), float(_q[2]), float(_q[3]), float(_q[0])

                # Save image and determine filename
                if _in_yaw_align and sub2_img_dir is not None:
                    _img_fname = f"camera0_{sub2_frame_idx:05d}.jpg"
                    _img_dir = sub2_img_dir
                else:
                    _img_fname = f"camera0_{sub1_frame_idx:05d}.jpg"
                    _img_dir = sub1_img_dir

                if head_camera is not None:
                    _rgb = head_camera.data.output["rgb"]
                    Image.fromarray(_rgb[0, :, :, :3].cpu().numpy()).save(
                        str(_img_dir / _img_fname), format="JPEG", quality=90
                    )

                # Build CSV row
                _row = (
                    f"{timestep * dt:.6f},"
                    f"{pose_after[0]:.6f},{pose_after[1]:.6f},{robot_height:.6f},"
                    f"{_qx:.6f},{_qy:.6f},{_qz:.6f},{_qw:.6f},"
                    f"{float(_lin_vel_b[0]):.6f},{float(_lin_vel_b[1]):.6f},{float(_lin_vel_b[2]):.6f},"
                    + ",".join(f"{j:.6f}" for j in _arm_joints_7d)
                    + f",{_img_fname}"
                )

                if _in_yaw_align and args_cli.goal_yaw is not None:
                    sub2_rows.append(_row)
                    sub2_frame_idx += 1
                else:
                    sub1_rows.append(_row)
                    sub1_frame_idx += 1

                timestep += 1
                if distance_to_goal_after <= args_cli.goal_tolerance and yaw_goal_satisfied:
                    success = True
                    break
                if timestep >= args_cli.max_steps:
                    timeout = True
                    break

            if args_cli.video and timestep >= args_cli.video_length:
                timeout = True
                break

            if args_cli.keyboard:
                camera_follow(env)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    # Write episode CSV files
    def _write_csv(csv_path: Path, header: str, rows: list[str]) -> None:
        csv_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    _write_csv(episode_root / f"{traj_id}-1" / "data.csv", _CSV_HEADER, sub1_rows)
    if sub2_rows:
        _write_csv(episode_root / f"{traj_id}-2" / "data.csv", _CSV_HEADER, sub2_rows)

    print(f"[INFO] Navigation run complete. success={success} timeout={timeout} steps={timestep}")
    print(f"[INFO] Task {args_cli.task_id}, trajectory {traj_id} saved to: {episode_root}  "
          f"(subtask-1: {len(sub1_rows)} frames, subtask-2: {len(sub2_rows)} frames)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
