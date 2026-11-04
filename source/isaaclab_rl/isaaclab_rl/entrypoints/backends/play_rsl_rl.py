# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os

import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from tensordict import TensorDict

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils import to_dict
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from ...utils.wandb import is_wandb_checkpoint, resolve_wandb_checkpoint
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_published_checkpoint,
    run_playback,
    set_hydra_args,
    show_run_summary,
    startup_screen,
)
from . import cli_args_rsl_rl as cli_args

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
    add_common_play_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument(
        "--gamepad", action="store_true", help="Control base_velocity with a gamepad; Y toggles tracking."
    )
    controls.add_argument(
        "--keyboard", action="store_true", help="Control base_velocity with the keyboard; Y toggles tracking."
    )
    parser.add_argument(
        "--fixed_camera",
        "--fixed-camera",
        action="store_true",
        help="Start manual playback with robot tracking disabled.",
    )
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    if remaining_args_env_registration is not None:
        _, remaining_args_env_registration = setup_preset_cli(parser, remaining_args_env_registration)
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: object, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("rsl_rl", args_cli.task, env_cfg)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def run(argv: list[str]) -> None:
    """Play a checkpoint of an RSL-RL agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        pre_launch_video_config(env_cfg, args_cli)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            manual_control = args_cli.gamepad or args_cli.keyboard
            if manual_control:
                if env_cfg.scene.num_envs != 1:
                    raise ValueError("--gamepad and --keyboard require --num_envs 1.")
                if isinstance(env_cfg, ManagerBasedRLEnvCfg) and hasattr(env_cfg.commands, "base_velocity"):
                    env_cfg.terminations.time_out = None
                    command_cfg = env_cfg.commands.base_velocity
                    controlled_asset_name = command_cfg.asset_name
                    # Keep periodic resampling from overwriting manual commands.
                    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
                    command_cfg.heading_command = False
                    command_cfg.rel_standing_envs = 0.0
                    command_cfg.rel_rotating_envs = 0.0
                    command_cfg.ranges.heading = None
                    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
                    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
                    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
                elif isinstance(env_cfg, DirectRLEnvCfg) and callable(
                    configure_manual_control := getattr(env_cfg, "configure_manual_velocity_control", None)
                ):
                    controlled_asset_name = configure_manual_control()
                else:
                    raise ValueError(
                        "--gamepad and --keyboard require base_velocity commands or a direct task implementing"
                        " configure_manual_velocity_control() and set_external_commands()."
                    )
            # certain randomizations occur in the environment initialization so we set the seed here
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            logger.info(f"Loading experiment from directory: {log_root_path}")
            resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
            if resume_path is None:
                return
            log_dir = os.path.dirname(resume_path)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=resume_path)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            cleanup.callback(lambda: close_env(env))

            screen.stage("Loading policy")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            logger.info(f"Loading model checkpoint from: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            export_model_dir = os.path.join(log_dir, "exported")
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            base_env = env.unwrapped
            command_input = None
            input_active = True
            camera_button_held = False
            camera = None

            if manual_control:
                from isaaclab_teleop import (
                    CLOUDXR_STANDALONE_ENV,
                    create_isaac_teleop_device,
                    poll_control_events,
                )

                from ._playback_camera import PlaybackCamera

                if args_cli.keyboard and not any(viz.key_event_source is not None for viz in base_env.sim.visualizers):
                    raise ValueError("--keyboard requires an interactive window; pass --viz kit or --viz newton_gl.")
                camera = PlaybackCamera(base_env, asset_name=controlled_asset_name, follow=not args_cli.fixed_camera)
                cleanup.callback(camera.close)
                if args_cli.gamepad:
                    from ._playback_gamepad import gamepad_with_camera_cfg

                    input_cfg = gamepad_with_camera_cfg(
                        v_x_sensitivity=2.0,
                        v_y_sensitivity=1.0,
                        omega_z_sensitivity=3.0,
                        dead_zone=0.15,
                        sim_device=str(base_env.device),
                    )
                else:
                    from ._playback_keyboard import wasd_keyboard_cfg

                    input_cfg = wasd_keyboard_cfg(
                        v_x_sensitivity=2.0,
                        v_y_sensitivity=1.0,
                        omega_z_sensitivity=3.0,
                        sim_device=str(base_env.device),
                    )
                    logger.info(
                        "Keyboard: w/a/s/d move, left/right arrows turn; "
                        "b/p/r start/pause/reset; y toggles camera tracking."
                    )
                command_input = cleanup.enter_context(
                    create_isaac_teleop_device(
                        input_cfg,
                        cloudxr_env_file=CLOUDXR_STANDALONE_ENV,
                        use_kit_xr_bridge=False,
                        visualizers=base_env.sim.visualizers,
                    )
                )
                command_input.request_start()
                manual_velocity = torch.zeros(env.num_envs, 3, device=base_env.device)

            obs = env.get_observations()

            def step() -> None:
                nonlocal obs, input_active, camera_button_held

                if command_input is not None:
                    manual_command = command_input.advance()
                    control = poll_control_events(command_input)

                    if args_cli.gamepad:
                        camera_pressed = manual_command is not None and bool(manual_command[3].item() > 0.5)
                        if camera_pressed and not camera_button_held:
                            camera.toggle()
                        camera_button_held = camera_pressed

                    if control.is_active is not None:
                        input_active = control.is_active

                    if control.should_reset:
                        env.reset()
                        policy.reset(
                            torch.ones(
                                env.num_envs,
                                dtype=torch.bool,
                                device=base_env.device,
                            )
                        )
                        input_active = False
                        manual_command = None

                    manual_velocity.zero_()
                    if manual_command is not None and input_active:
                        manual_velocity.copy_(manual_command[:3].reshape(1, 3))

                    # Recompute observations so the policy receives the new command.
                    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
                        base_env.command_manager.get_command("base_velocity").copy_(manual_velocity)
                        obs = TensorDict(
                            base_env.observation_manager.compute(update_history=False),
                            batch_size=[env.num_envs],
                        )
                    else:
                        base_env.set_external_commands(manual_velocity)
                        obs = env.get_observations()

                obs, _, dones, _ = env.step(policy(obs))
                policy.reset(dones)
                if camera is not None:
                    camera.update()

            screen.close()
            run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
