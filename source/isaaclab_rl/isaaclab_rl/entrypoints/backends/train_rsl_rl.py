# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL training backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys
import time
from datetime import datetime
from typing import TextIO

import torch.distributed as dist
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation, report_activity
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils import to_dict
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...diagnostics.locomotion_metrics import RslRlLocomotionMetrics
from ...diagnostics.policy_provenance import write_policy_fingerprint
from ...diagnostics.rsl_rl_finite_check import RslRlFiniteCheck
from ...rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from ...utils.curriculum_checkpoint import RslRlCurriculumCheckpoint
from ...utils.online_curriculum import RslRlOnlineCurriculum
from ...utils.snapshot_curriculum import RslRlSnapshotCurriculum
from ...utils.training_interrupt import training_interrupt_exit
from ...utils.wandb import announce_new_run, is_wandb_checkpoint, resolve_wandb_checkpoint, resolve_wandb_entity
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_train_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    dump_train_configs,
    enable_cameras_for_video,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    scoped_torch_backend_flags,
    set_hydra_args,
    show_run_summary,
    startup_screen,
    wrap_sensor_capture,
    write_run_manifest,
)
from . import cli_args_rsl_rl as cli_args

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


class _TransitionMetricsConsole(io.TextIOBase):
    """터미널의 급전환·LP 상세 지표만 숨기며 기록용 writer는 유지함."""

    def __init__(self, stream: TextIO):
        self._stream = stream

    @property
    def encoding(self) -> str | None:
        return self._stream.encoding

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return self._stream.isatty()

    def fileno(self) -> int:
        return self._stream.fileno()

    def write(self, text: str) -> int:
        visible = "".join(
            line
            for line in text.splitlines(keepends=True)
            if not line.lstrip().startswith(
                (
                    "Metrics/base_velocity/transition_",
                    "Curriculum/learning_progress/",
                    "LP/",
                    "Curriculum/terrain_levels/",
                )
            )
        )
        self._stream.write(visible)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL training arguments."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
    add_common_train_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    parser.add_argument(
        "--check_finite",
        action="store_true",
        help="Diagnose single-device PPO NaN/Inf in states, observations, targets, gradients and optimizer state.",
    )
    parser.add_argument(
        "--record_action_history",
        action="store_true",
        help="Keep policy inputs and model versions; enable finite checks and stop on oversized environment input.",
    )
    parser.add_argument(
        "--action_history_steps", type=int, default=64, help="Number of policy steps to keep in memory."
    )
    parser.add_argument(
        "--action_history_threshold",
        type=float,
        default=100.0,
        help="Stop and save history above this environment-input action magnitude, after wrapper clipping.",
    )
    parser.add_argument(
        "--resume_optimizer_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore the policy optimizer; disable to keep a fresh optimizer and configured learning rate.",
    )
    parser.add_argument(
        "--reset_curriculum",
        action="store_true",
        help="Keep initialized curriculum statistics when resuming; policy and selected optimizer state still load.",
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


def _resolve_checkpoint(args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, log_root_path: str) -> str | None:
    """Resolve the checkpoint to resume from, or None when training starts from scratch."""
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=args_cli.task,
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    if agent_cfg.algorithm.class_name == "Distillation":
        raise ValueError("Distillation training requires --checkpoint.")
    return None


def run(argv: list[str]) -> None:
    """Train an RSL-RL agent while restoring the caller's Torch backend settings."""
    args_cli = _parse_args(argv)
    with scoped_torch_backend_flags(
        cuda_matmul_allow_tf32=True,
        cudnn_allow_tf32=True,
        cudnn_deterministic=False,
        cudnn_benchmark=True,  # Reuse autotuned convolution algorithms for fixed input shapes.
    ):
        _run(args_cli)


def _run(args_cli: argparse.Namespace) -> None:
    """Execute RSL-RL training with parsed arguments."""
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
        pre_launch_video_config(env_cfg, args_cli)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="train")
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            if args_cli.max_iterations is not None:
                agent_cfg.max_iterations = args_cli.max_iterations
            rank = int(os.getenv("RANK", "0")) if args_cli.distributed else None
            if rank is not None:
                agent_cfg.seed += rank
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            logger.info(f"Logging experiment in directory: {log_root_path}")
            run_name = args_cli.run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            logger.info(f"Exact experiment name requested from command line: {run_name}")
            if agent_cfg.run_name:
                run_name += f"_{agent_cfg.run_name}"
            log_dir = os.path.join(log_root_path, run_name)
            # rank 0 writes its settings and videos where a single-GPU run does; other ranks use rank_<rank>/
            rank_dir = log_dir if rank in (None, 0) else os.path.join(log_dir, f"rank_{rank}")
            # All ranks share the run folder, so its manifest is written once.
            if rank in (None, 0):
                write_run_manifest(log_dir, library="rsl_rl", task=args_cli.task, metadata={"agent": args_cli.agent})

            resume_path = _resolve_checkpoint(args_cli, agent_cfg, log_root_path)
            env_cfg.log_dir = rank_dir
            apply_video_recording(env_cfg, rank_dir, args_cli)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            cleanup.callback(lambda: close_env(env))
            env = wrap_sensor_capture(env, rank_dir, args_cli)

            screen.stage("Preparing agent")
            start_time = time.time()
            report_activity("Wrapping environment")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            report_activity(None)
            report_activity("Building policy")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=log_dir, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, to_dict(agent_cfg), log_dir=log_dir, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            report_activity(None)

            cleanup.enter_context(RslRlCurriculumCheckpoint(runner, restore_curriculum=not args_cli.reset_curriculum))

            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)

            runner.add_git_repo_to_log(__file__)
            if resume_path is not None:
                logger.info(f"Loading model checkpoint from: {resume_path}")
                # map to this process's device; the checkpoint's tensors otherwise land on the GPU that saved them
                runner.load(
                    resume_path,
                    load_cfg=cli_args.checkpoint_load_cfg(
                        args_cli.resume_optimizer_state, runner_class=agent_cfg.class_name
                    ),
                    map_location=agent_cfg.device,
                )
            lp_term = getattr(getattr(env_cfg, "curriculum", None), "learning_progress", None)
            if lp_term is not None and getattr(lp_term.params.get("settings"), "diagnostics", False):
                write_policy_fingerprint(runner, rank_dir, resume_path)
            dump_train_configs(rank_dir, env_cfg, agent_cfg)

            if agent_cfg.logger == "wandb":
                announce_new_run(agent_cfg.wandb_project, resolve_wandb_entity())

            screen.close()
            finite_check = (
                RslRlFiniteCheck(
                    runner,
                    rank_dir,
                    record_action_history=args_cli.record_action_history,
                    action_history_steps=args_cli.action_history_steps,
                    action_history_threshold=args_cli.action_history_threshold,
                )
                if args_cli.check_finite or args_cli.record_action_history
                else contextlib.nullcontext()
            )
            locomotion_metrics = RslRlLocomotionMetrics(runner)

            @contextlib.contextmanager
            def probe_context():
                with (
                    locomotion_metrics.suspend(),
                    finite_check.snapshot_probe()
                    if isinstance(finite_check, RslRlFiniteCheck)
                    else contextlib.nullcontext(),
                ):
                    yield

            snapshot_curriculum = RslRlSnapshotCurriculum(
                runner,
                resumed=resume_path is not None,
                probe_context=probe_context,
            )
            with (
                training_interrupt_exit(),
                contextlib.redirect_stdout(_TransitionMetricsConsole(sys.stdout)),
                finite_check,
                locomotion_metrics,
                RslRlOnlineCurriculum(runner, resumed=resume_path is not None),
                snapshot_curriculum,
            ):
                runner.learn(
                    num_learning_iterations=agent_cfg.max_iterations,
                    init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
                )
                print(f"Training time: {round(time.time() - start_time, 2)} seconds")
                # the RL library creates the process group but never destroys it, which torch warns about at exit
                if dist.is_initialized():
                    dist.destroy_process_group()
