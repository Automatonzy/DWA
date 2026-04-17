"""Navigation playback entrypoint for map-driven waypoint following."""

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

from navlib import AStarPlanner, OccupancyGridMap, PathTrackingConfig, PathTrackingController, save_path_bundle

parser = argparse.ArgumentParser(description="Play an RL locomotion policy with a navigation path follower.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=2000, help="Maximum recorded video length in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate. Navigation uses 1.")
parser.add_argument(
    "--task",
    type=str,
    default="RobotLab-Isaac-Velocity-Flat-Go2-X5-ArmUnlock-v0",
    help="Task name. Defaults to the ArmUnlock flat task to match the existing play_cs.py bring-up flow.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--map", type=str, default=None, help="USD map loaded into the terrain scene.")
parser.add_argument("--nav-map", type=str, required=True, help="Navigation map metadata (.json/.yaml).")
parser.add_argument("--path", type=str, default=None, help="Optional precomputed path bundle (.json).")
parser.add_argument("--goal", type=float, nargs=2, default=None, metavar=("X", "Y"), help="Goal point in world coordinates.")
parser.add_argument(
    "--spawn",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "YAW"),
    help="Override robot spawn pose in world coordinates.",
)
parser.add_argument(
    "--inflate-radius",
    type=float,
    default=0.25,
    help="Obstacle inflation radius in meters used during online planning.",
)
parser.add_argument(
    "--goal-tolerance",
    type=float,
    default=0.35,
    help="Distance threshold in meters for declaring success.",
)
parser.add_argument(
    "--lookahead-distance",
    type=float,
    default=0.6,
    help="Lookahead distance in meters for the path follower.",
)
parser.add_argument(
    "--waypoint-tolerance",
    type=float,
    default=0.2,
    help="Distance threshold in meters for advancing to the next waypoint.",
)
parser.add_argument("--max-lin-vel", type=float, default=0.5, help="Maximum forward speed command.")
parser.add_argument("--max-ang-vel", type=float, default=1.0, help="Maximum yaw rate command.")
parser.add_argument(
    "--debug-command",
    type=float,
    nargs=3,
    default=None,
    metavar=("VX", "VY", "WZ"),
    help="Optional fixed base command used for debugging after the settle phase.",
)
parser.add_argument(
    "--debug-command-steps",
    type=int,
    default=120,
    help="Number of steps to apply --debug-command after the settle phase.",
)
parser.add_argument("--max-steps", type=int, default=3000, help="Maximum navigation steps before timeout.")
parser.add_argument(
    "--settle-steps",
    type=int,
    default=120,
    help="Initial simulation steps to hold zero base command so the robot can settle before navigation.",
)
parser.add_argument(
    "--debug-print-every",
    type=int,
    default=60,
    help="Print robot pose/command every N control steps. Set <= 0 to disable.",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Optional directory for path and trajectory logs. Defaults to checkpoint_dir/nav_runs/<timestamp>/.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import omni.usd
from pxr import Usd, UsdGeom
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.assets import Articulation
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


def _disable_play_randomization(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    env_cfg.observations.policy.enable_corruption = False
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.command_levels_lin_vel = None
        env_cfg.curriculum.command_levels_ang_vel = None

    if getattr(env_cfg, "events", None) is not None:
        for term_name in (
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass_others",
            "randomize_com_positions",
            "randomize_apply_external_force_torque",
            "randomize_push_robot",
            "push_robot",
            "randomize_actuator_gains",
        ):
            if hasattr(env_cfg.events, term_name):
                setattr(env_cfg.events, term_name, None)

        material_event = getattr(env_cfg.events, "randomize_rigid_body_material", None)
        if material_event is not None:
            material_event.params["static_friction_range"] = (1.0, 1.0)
            material_event.params["dynamic_friction_range"] = (1.0, 1.0)
            material_event.params["restitution_range"] = (0.5, 0.5)


def _configure_spawn(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, spawn: tuple[float, float, float] | None):
    if getattr(env_cfg, "events", None) is None or not hasattr(env_cfg.events, "randomize_reset_base"):
        return
    x, y, yaw = spawn if spawn is not None else (0.0, 0.0, 0.0)
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {
            "x": (x, x),
            "y": (y, y),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (yaw, yaw),
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


def _quat_to_yaw(quat_wxyz: torch.Tensor) -> float:
    w = float(quat_wxyz[0].item())
    x = float(quat_wxyz[1].item())
    y = float(quat_wxyz[2].item())
    z = float(quat_wxyz[3].item())
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _robot_pose(robot: Articulation) -> tuple[float, float, float]:
    pos = robot.data.root_pos_w[0]
    quat = robot.data.root_quat_w[0]
    return float(pos[0].item()), float(pos[1].item()), _quat_to_yaw(quat)


def _robot_height(robot: Articulation) -> float:
    return float(robot.data.root_pos_w[0][2].item())


def _add_visual_map_reference(
    map_path: str,
    prim_path: str = "/World/nav_visual",
    source_prim_path: str = "/World/gauss",
) -> None:
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] USD stage is not available; skipping visual map reference.")
        return

    ref_stage = Usd.Stage.Open(map_path)
    if ref_stage is None:
        print(f"[WARN] Failed to open map stage for visual reference: {map_path}")
        return

    source_prim = ref_stage.GetPrimAtPath(source_prim_path)
    if not source_prim.IsValid():
        print(f"[WARN] Visual prim {source_prim_path} not found in {map_path}; skipping visual reference.")
        return

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().ClearReferences()
    prim.GetReferences().AddReference(map_path, source_prim_path)
    UsdGeom.Imageable(prim).MakeVisible()
    print(f"[INFO] Added visual map reference at {prim_path} -> {map_path}{source_prim_path}")


def _robot_speed(robot: Articulation) -> tuple[float, float]:
    lin_vel = robot.data.root_lin_vel_b[0]
    ang_vel = robot.data.root_ang_vel_b[0]
    return float(lin_vel[0].item()), float(ang_vel[2].item())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.map is not None:
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/scene_collision",
            terrain_type="usd",
            usd_path=args_cli.map,
            debug_vis=False,
        )
        env_cfg.scene.sky_light = None

    _disable_play_randomization(env_cfg)
    _configure_spawn(env_cfg, tuple(args_cli.spawn) if args_cli.spawn is not None else None)
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-2.0, 2.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
    if getattr(env_cfg, "curriculum", None) is not None:
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.command_levels_lin_vel = None
        env_cfg.curriculum.command_levels_ang_vel = None
    if getattr(env_cfg, "terminations", None) is not None:
        env_cfg.terminations.time_out = None
        env_cfg.terminations.terrain_out_of_bounds = None
        env_cfg.terminations.illegal_contact = None
    if getattr(env_cfg.scene, "terrain", None) is not None:
        env_cfg.scene.terrain.max_init_terrain_level = None
        terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
        if terrain_generator is not None:
            terrain_generator.num_rows = 5
            terrain_generator.num_cols = 5
            terrain_generator.curriculum = False

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    checkpoint_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = checkpoint_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.map is not None:
        _add_visual_map_reference(args_cli.map)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(checkpoint_dir, "videos", "play_nav"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording navigation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    arm_term = env.unwrapped.command_manager._terms.get("arm_joint_pos", None)
    arm_home_pos = torch.zeros(1, 6, dtype=torch.float32, device=env.unwrapped.device) if arm_term is not None else None
    gripper_joint_ids, _ = robot.find_joints(["arm_joint7", "arm_joint8"], preserve_order=True)
    if len(gripper_joint_ids) != 2:
        gripper_joint_ids = None
    gripper_closed_pos = (
        torch.zeros(1, 2, dtype=torch.float32, device=env.unwrapped.device) if gripper_joint_ids is not None else None
    )
    start_pose = _robot_pose(robot)
    start_xy = (start_pose[0], start_pose[1])

    nav_map = OccupancyGridMap.from_meta_file(args_cli.nav_map).inflate(args_cli.inflate_radius)
    if args_cli.path is not None:
        path_bundle = json.loads(Path(args_cli.path).read_text())
        path_world = [(float(x), float(y)) for x, y in path_bundle["path_world"]]
        print(f"[INFO] Loaded precomputed path with {len(path_world)} waypoints from {args_cli.path}")
    else:
        if args_cli.goal is None:
            raise ValueError("--goal is required when --path is not provided.")
        planner = AStarPlanner(allow_diagonal=True, heuristic_weight=1.0)
        plan = planner.plan(
            nav_map,
            start_xy=start_xy,
            goal_xy=(float(args_cli.goal[0]), float(args_cli.goal[1])),
            snap_to_free=True,
            max_snap_distance_m=max(0.5, args_cli.inflate_radius + nav_map.resolution),
        )
        path_world = plan.path_world
        print(f"[INFO] Planned online path with {len(path_world)} waypoints and cost {plan.cost:.3f}.")

    tracking_cfg = PathTrackingConfig(
        lookahead_distance=args_cli.lookahead_distance,
        waypoint_tolerance=args_cli.waypoint_tolerance,
        goal_tolerance=args_cli.goal_tolerance,
        max_linear_velocity=args_cli.max_lin_vel,
        max_angular_velocity=args_cli.max_ang_vel,
    )
    controller = PathTrackingController(path_world=path_world, config=tracking_cfg)

    if args_cli.output_dir is None:
        run_dir = Path(checkpoint_dir) / "nav_runs" / time.strftime("%Y%m%d_%H%M%S")
    else:
        run_dir = Path(args_cli.output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = run_dir / "trajectory.jsonl"
    summary_path = run_dir / "summary.json"
    if args_cli.path is None and args_cli.goal is not None:
        save_path_bundle(
            run_dir / "planned_path.json",
            grid_map=nav_map,
            plan=plan,
            start_xy=start_xy,
            goal_xy=(float(args_cli.goal[0]), float(args_cli.goal[1])),
            inflation_radius_m=args_cli.inflate_radius,
        )

    base_cmd_term = env.unwrapped.command_manager._terms.get("base_velocity", None)
    if base_cmd_term is None:
        raise RuntimeError("base_velocity command term is not active for this task.")

    dt = env.unwrapped.step_dt
    success = False
    timeout = False
    last_debug: dict[str, float | int | bool] = {}

    with trajectory_path.open("w", encoding="utf-8") as traj_file:
        timestep = 0
        while simulation_app.is_running():
            step_start = time.time()
            with torch.inference_mode():
                pose = _robot_pose(robot)
                robot_height = _robot_height(robot)
                if timestep < args_cli.settle_steps:
                    command_np = (0.0, 0.0, 0.0)
                    debug = controller.compute_command(pose)[1]
                    last_debug = {
                        "target_index": debug.target_index,
                        "distance_to_target": debug.distance_to_target,
                        "distance_to_goal": debug.distance_to_goal,
                        "heading_error": debug.heading_error,
                        "reached_goal": False,
                        "settling": True,
                        "debug_command": False,
                    }
                elif args_cli.debug_command is not None and timestep < args_cli.settle_steps + args_cli.debug_command_steps:
                    command_np = tuple(float(v) for v in args_cli.debug_command)
                    debug = controller.compute_command(pose)[1]
                    last_debug = {
                        "target_index": debug.target_index,
                        "distance_to_target": debug.distance_to_target,
                        "distance_to_goal": debug.distance_to_goal,
                        "heading_error": debug.heading_error,
                        "reached_goal": False,
                        "settling": False,
                        "debug_command": True,
                    }
                else:
                    command_np, debug = controller.compute_command(pose)
                    last_debug = {
                        "target_index": debug.target_index,
                        "distance_to_target": debug.distance_to_target,
                        "distance_to_goal": debug.distance_to_goal,
                        "heading_error": debug.heading_error,
                        "reached_goal": debug.reached_goal,
                        "settling": False,
                        "debug_command": False,
                    }
                command = torch.tensor(command_np, dtype=torch.float32, device=base_cmd_term.device).unsqueeze(0)
                base_cmd_term.vel_command_b[:] = command
                if hasattr(base_cmd_term, "is_heading_env"):
                    base_cmd_term.is_heading_env[:] = False
                if hasattr(base_cmd_term, "is_standing_env"):
                    base_cmd_term.is_standing_env[:] = torch.linalg.norm(command, dim=1) < 1.0e-6
                if hasattr(base_cmd_term, "heading_target"):
                    base_cmd_term.heading_target[:] = 0.0

                if arm_home_pos is not None and arm_term is not None:
                    arm_term.command_buffer[:] = arm_home_pos

                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

                if gripper_closed_pos is not None and gripper_joint_ids is not None:
                    robot.set_joint_position_target(gripper_closed_pos, joint_ids=gripper_joint_ids)

                measured_vx, measured_wz = _robot_speed(robot)
                if args_cli.debug_print_every > 0 and timestep % args_cli.debug_print_every == 0:
                    print(
                        f"[NAV] step={timestep} pose=({pose[0]:.3f}, {pose[1]:.3f}, yaw={pose[2]:.3f}) "
                        f"z={robot_height:.3f} cmd=({float(command_np[0]):.3f}, {float(command_np[1]):.3f}, {float(command_np[2]):.3f}) "
                        f"measured=({measured_vx:.3f}, {measured_wz:.3f}) goal_dist={last_debug['distance_to_goal']:.3f} "
                        f"settling={last_debug['settling']} debug_cmd={last_debug['debug_command']}"
                    )
                traj_file.write(
                    json.dumps(
                        {
                            "step": timestep,
                            "time_s": timestep * dt,
                            "pose": {"x": pose[0], "y": pose[1], "yaw": pose[2]},
                            "target_waypoint": {
                                "index": debug.target_index,
                                "x": debug.target_point[0],
                                "y": debug.target_point[1],
                            },
                            "command": {"vx": float(command_np[0]), "vy": float(command_np[1]), "wz": float(command_np[2])},
                            "measured_velocity": {"vx": measured_vx, "wz": measured_wz},
                            "distance_to_goal": debug.distance_to_goal,
                            "heading_error": debug.heading_error,
                            "reached_goal": debug.reached_goal,
                        }
                    )
                    + "\n"
                )

                timestep += 1
                if debug.reached_goal:
                    success = True
                    break
                if timestep >= args_cli.max_steps:
                    timeout = True
                    break

            if args_cli.video and timestep >= args_cli.video_length:
                timeout = True
                break

            sleep_time = dt - (time.time() - step_start)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    goal_record = (
        {"x": path_world[-1][0], "y": path_world[-1][1]}
        if args_cli.goal is None
        else {"x": args_cli.goal[0], "y": args_cli.goal[1]}
    )
    summary_path.write_text(
        json.dumps(
            {
                "task": task_name,
                "checkpoint": resume_path,
                "map": args_cli.map,
                "nav_map": str(Path(args_cli.nav_map).expanduser().resolve()),
                "path": args_cli.path,
                "success": success,
                "timeout": timeout,
                "steps": timestep,
                "dt": dt,
                "start_pose": {"x": start_pose[0], "y": start_pose[1], "yaw": start_pose[2]},
                "goal": goal_record,
                "last_debug": last_debug,
            },
            indent=2,
        )
    )
    print(f"[INFO] Navigation run complete. success={success} timeout={timeout} steps={timestep}")
    print(f"[INFO] Logs written to: {run_dir}")
    env.close()


if __name__ == "__main__":
    main()
