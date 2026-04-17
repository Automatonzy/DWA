"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=20000000, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--map", type=str, default=None, help="Dir of the map.")
parser.add_argument(
    "--disable-can",
    action="store_true",
    default=False,
    help="Disable importing the can asset from the workspace.",
)
parser.add_argument(
    "--can-usd",
    type=str,
    default=None,
    help="Path to the can USD asset. Defaults to the first USD found under ./can.",
)
parser.add_argument(
    "--can-pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="World position for the can asset. Defaults to a point in front of the robot spawn.",
)
parser.add_argument(
    "--can-rot",
    type=float,
    nargs=4,
    default=(0.70710678, 0.70710678, 0.0, 0.0),
    metavar=("W", "X", "Y", "Z"),
    help="World quaternion for the can asset. Defaults to a +90 degree rotation about the x-axis.",
)
parser.add_argument(
    "--can-scale",
    type=float,
    default=0.02,
    help="Uniform scale applied to the can asset.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# import rl_utils after SimulationApp is instantiated
from rl_utils import camera_follow

import gymnasium as gym
import time
import torch

import isaaclab.sim as sim_utils
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.assets import RigidObjectCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

# optional import for pretrained checkpoints
try:
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    get_published_pretrained_checkpoint = None
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


def _find_default_can_usd() -> str | None:
    """Return the preferred can USD/USDa found under the workspace-local can directory."""
    can_root = Path.cwd() / "can"
    if not can_root.exists():
        return None
    physics_candidates = sorted(can_root.glob("**/*_physics.usda"))
    if physics_candidates:
        return str(physics_candidates[0])
    usd_candidates = sorted(list(can_root.glob("**/*.usda")) + list(can_root.glob("**/*.usd")))
    return str(usd_candidates[0]) if usd_candidates else None


def _parse_can_metadata(can_usd_path: str) -> float | None:
    """Infer can height from the sibling annotation file when available."""
    usd_path = Path(can_usd_path)
    asset_root = usd_path.parent.parent
    annotation_path = asset_root / f"{asset_root.name}_annotation.json"
    if not annotation_path.exists():
        return None

    try:
        annotation = json.loads(annotation_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    height = None
    dimensions = annotation.get("dimensions")
    if isinstance(dimensions, str):
        try:
            dim_values = [float(item.strip()) for item in dimensions.split("*")]
        except ValueError:
            dim_values = []
        if len(dim_values) == 3:
            height = dim_values[2]

    return height


def _build_can_cfg() -> RigidObjectCfg | None:
    """Construct a rigid can asset configuration when a can USD is available."""
    if args_cli.disable_can:
        return None

    can_usd_path = args_cli.can_usd or _find_default_can_usd()
    if can_usd_path is None:
        return None

    can_usd_path = Path(can_usd_path).expanduser().resolve()
    if can_usd_path.suffix == ".usd":
        physics_overlay_path = can_usd_path.with_name(f"{can_usd_path.stem}_physics.usda")
        if physics_overlay_path.exists():
            can_usd_path = physics_overlay_path

    can_usd_path = str(can_usd_path)
    if not os.path.exists(can_usd_path):
        raise FileNotFoundError(f"Can USD asset not found: {can_usd_path}")

    can_height = _parse_can_metadata(can_usd_path)
    if args_cli.can_pos is not None:
        can_pos = tuple(args_cli.can_pos)
    else:
        can_pos = (-1.975, 6.65, 0.71642)

    print(f"[INFO] Loading can asset from: {can_usd_path}")
    print(f"[INFO] Can spawn pose: pos={can_pos}, rot={tuple(args_cli.can_rot)}")

    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Can",
        init_state=RigidObjectCfg.InitialStateCfg(pos=can_pos, rot=tuple(args_cli.can_rot)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=can_usd_path,
            scale=(args_cli.can_scale, args_cli.can_scale, args_cli.can_scale),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=2.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
                sleep_threshold=0.02,
                stabilization_threshold=0.01,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # cs map config
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/scene_collision",
        terrain_type="usd",
        usd_path=args_cli.map,
        debug_vis=False,
    )
    env_cfg.scene.sky_light = None
    can_cfg = _build_can_cfg()
    if can_cfg is not None:
        env_cfg.scene.can = can_cfg
    env_cfg.events.randomize_reset_base.params = {
        "pose_range": {
            "x": (-2.03488, -2.03488),
            "y": (5.00164, 5.00164),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (1.58, 1.58),
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
    env_cfg.events.randomize_rigid_body_material.params["static_friction_range"] = (1.0, 1.0)
    env_cfg.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (1.0, 1.0)
    env_cfg.events.randomize_rigid_body_material.params["restitution_range"] = (0.5, 0.5)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-2.0, 2.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.terminations.illegal_contact = None
    env_cfg.terminations.terrain_out_of_bounds = None

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            raise ImportError(
                "The 'isaaclab.utils.pretrained_checkpoint' module is not available. "
                "Please use --checkpoint instead of --use_pretrained_checkpoint."
            )
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0

    # initialize command terms
    arm_term = env.unwrapped.command_manager._terms.get("arm_joint_pos", None)
    base_cmd_term = env.unwrapped.command_manager._terms.get("base_velocity", None)
    robot = env.unwrapped.scene["robot"]
    arm_home_pos = torch.zeros(1, 6, dtype=torch.float32, device=env.unwrapped.device) if arm_term is not None else None
    gripper_open_pos = torch.tensor([[0.044, 0.044]], dtype=torch.float32, device=env.unwrapped.device)
    gripper_closed_pos = torch.zeros(1, 2, dtype=torch.float32, device=env.unwrapped.device)
    desired_gripper_pos = gripper_closed_pos.clone()
    zero_velocity_start_time = None
    arm_phase = "idle"
    gripper_joint_ids, _ = robot.find_joints(["arm_joint7", "arm_joint8"], preserve_order=True)
    if len(gripper_joint_ids) == 2:
        print(f"[INFO] Direct gripper joint control enabled for joint ids: {gripper_joint_ids}")
    else:
        gripper_joint_ids = None
        print("[WARN] Failed to resolve gripper joint ids for direct control.")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if base_cmd_term is not None:
                if timestep * dt < 2:
                    desired_base_cmd = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32, device=base_cmd_term.device)
                else:
                    desired_base_cmd = torch.zeros(1, 3, dtype=torch.float32, device=base_cmd_term.device)
            else:
                desired_base_cmd = None
            desired_arm_pos = arm_home_pos.clone() if arm_home_pos is not None else None
            desired_gripper_pos = gripper_closed_pos.clone()

            sim_time = timestep * dt
            if desired_base_cmd is not None and torch.linalg.norm(desired_base_cmd, dim=1).max().item() < 1.0e-6:
                if zero_velocity_start_time is None:
                    zero_velocity_start_time = sim_time
                    print(f"[INFO] Base velocity reached zero at t={sim_time:.2f}s; arm command timer started.")
                zero_velocity_elapsed = sim_time - zero_velocity_start_time
                if zero_velocity_elapsed < 8.0:
                    open_progress = min(max(zero_velocity_elapsed / 1.0, 0.0), 1.0)
                    desired_gripper_pos = gripper_closed_pos + (gripper_open_pos - gripper_closed_pos) * open_progress
                elif zero_velocity_elapsed < 9.0:
                    close_progress = min(max((zero_velocity_elapsed - 8.0) / 1.0, 0.0), 1.0)
                    desired_gripper_pos = gripper_open_pos + (gripper_closed_pos - gripper_open_pos) * close_progress
                else:
                    desired_gripper_pos = gripper_closed_pos.clone()
                if zero_velocity_elapsed < 2.0:
                    next_arm_phase = "hold_home"
                elif zero_velocity_elapsed < 5.0:
                    next_arm_phase = "move_to_target"
                    if desired_arm_pos is not None:
                        joint_progress = (zero_velocity_elapsed - 2.0) / 3.0
                        smooth_progress = joint_progress * joint_progress * (3.0 - 2.0 * joint_progress)
                        desired_arm_pos[:, 2] = 1.5 * smooth_progress
                        desired_arm_pos[:, 1] = 1.55 * smooth_progress
                elif zero_velocity_elapsed < 8.0:
                    next_arm_phase = "hold_target"
                    if desired_arm_pos is not None:
                        desired_arm_pos[:, 2] = 1.5
                        desired_arm_pos[:, 1] = 1.55
                elif zero_velocity_elapsed < 10.0:
                    next_arm_phase = "gripper_close_hold"
                    if desired_arm_pos is not None:
                        desired_arm_pos[:, 2] = 1.5
                        desired_arm_pos[:, 1] = 1.55
                elif zero_velocity_elapsed < 13.0:
                    next_arm_phase = "return_stage_1"
                    if desired_arm_pos is not None:
                        joint_progress = (zero_velocity_elapsed - 10.0) / 3.0
                        smooth_progress = joint_progress * joint_progress * (3.0 - 2.0 * joint_progress)
                        desired_arm_pos[:, 2] = 1.5
                        desired_arm_pos[:, 1] = 1.55 - 0.55 * smooth_progress
                elif zero_velocity_elapsed < 16.0:
                    next_arm_phase = "return_stage_2"
                    if desired_arm_pos is not None:
                        joint_progress = (zero_velocity_elapsed - 13.0) / 3.0
                        smooth_progress = joint_progress * joint_progress * (3.0 - 2.0 * joint_progress)
                        desired_arm_pos[:, 2] = 1.5 * (1.0 - smooth_progress)
                        desired_arm_pos[:, 1] = 1.0 * (1.0 - smooth_progress)
                else:
                    next_arm_phase = "hold_reset"
                    if desired_arm_pos is not None:
                        desired_arm_pos.zero_()
            else:
                zero_velocity_start_time = None
                next_arm_phase = "drive_base"

            if next_arm_phase != arm_phase:
                print(f"[INFO] Arm command phase -> {next_arm_phase} at t={sim_time:.2f}s")
                arm_phase = next_arm_phase

            if desired_base_cmd is not None and base_cmd_term is not None:
                base_cmd_term.vel_command_b[:] = desired_base_cmd
                if hasattr(base_cmd_term, "is_heading_env"):
                    base_cmd_term.is_heading_env[:] = False
                if hasattr(base_cmd_term, "is_standing_env"):
                    base_cmd_term.is_standing_env[:] = torch.linalg.norm(desired_base_cmd, dim=1) < 1.0e-6
                if hasattr(base_cmd_term, "heading_target"):
                    base_cmd_term.heading_target[:] = 0.0
            if desired_arm_pos is not None and arm_term is not None:
                arm_term.command_buffer[:] = desired_arm_pos

            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            if desired_gripper_pos is not None and gripper_joint_ids is not None:
                robot.set_joint_position_target(desired_gripper_pos, joint_ids=gripper_joint_ids)
            timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
