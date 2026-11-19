# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a fixed tracking/stair suite: python -m isaaclab_rl.evaluation.locomotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch
import yaml
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import to_dict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

from .checkpoints import evaluation_checkpoint
from .velocity_suite import build_cases, fingerprint, summarize

logger = logging.getLogger(__name__)


class _ConfigLoader(yaml.SafeLoader):
    """Read only the two harmless Python data tags emitted by saved training configs."""


_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node, deep=True))
)
_ConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node, deep=True)),
)


def load_environment_config(path, cfg):
    with Path(path).open() as stream:
        values = yaml.load(stream, Loader=_ConfigLoader)
    # Rendering is selected by the evaluator CLI, independently of the physical benchmark.
    for key in ("visualizer_cfgs", "default_visualizer_cfg"):
        values.get("sim", {}).pop(key, None)
    cfg.from_dict(values)


def restore_training_configs(checkpoint, env_cfg, agent_cfg):
    folder = checkpoint.parent / "params"
    for name, cfg in (("env", env_cfg), ("agent", agent_cfg)):
        path = folder / f"{name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Fixed evaluation requires the saved training configuration: {path}")
        with path.open() as stream:
            cfg.from_dict(yaml.load(stream, Loader=_ConfigLoader))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Velocity-Rough-UnitreeGo2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-iterations", type=int, default=None)
    parser.add_argument("--env-config", default=None, help="Common saved environment config for all compared policies.")
    parser.add_argument("--experiment-name", default="unitree_go2_rough")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--protocol", choices=("tracking", "traversal", "both"), default="both")
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--traversal-speed", type=float, default=0.5)
    parser.add_argument("--case-ids", type=int, nargs="+", default=None)
    parser.add_argument("--trace-case-ids", type=int, nargs="*", default=[])
    add_launcher_args(parser)
    args, overrides = setup_preset_cli(parser, argv)
    if args.num_envs < 1 or (args.episode_seconds is not None and args.episode_seconds <= 0):
        parser.error("num-envs and episode-seconds must be positive.")
    if args.expected_iterations is not None and args.expected_iterations < 1:
        parser.error("expected-iterations must be positive.")
    from isaaclab.paths import ISAACLAB_ROOT

    checkpoint = evaluation_checkpoint(
        args.checkpoint, Path(ISAACLAB_ROOT) / "logs/rsl_rl" / args.experiment_name, args.expected_iterations
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=overrides)
    restore_training_configs(checkpoint, env_cfg, agent_cfg)
    from isaaclab_tasks.core.velocity.mdp.curriculum.catalog import terrain_velocity_catalog
    from isaaclab_tasks.core.velocity.mdp.curriculum.lp.config import LearningProgressCfg
    from isaaclab_tasks.core.velocity.mdp.evaluation.fixed import (
        FixedEvaluationEventsCfg,
        FixedEvaluationVelocityCommand,
        FixedVelocityEvaluation,
    )

    if agent_cfg.actor.class_name != "MLPModel":
        raise ValueError("The fixed locomotion evaluator currently supports MLP actors only.")
    trained_settings = env_cfg.curriculum.learning_progress.params["settings"]
    training = {
        "seed": agent_cfg.seed,
        "strategy": getattr(trained_settings, "sampling_strategy", "lp"),
        "temperature": trained_settings.temperature,
        "return_mode": getattr(trained_settings, "return_mode", "boundary_extrapolated"),
    }
    if args.env_config is not None:
        load_environment_config(args.env_config, env_cfg)
    settings = LearningProgressCfg()
    original = env_cfg.curriculum.learning_progress.params["settings"]
    settings.command_bins = tuple(original.command_bins)
    settings.modes = tuple(original.modes)
    settings.zero_command_scales = tuple(original.zero_command_scales)
    settings.command_threshold = original.command_threshold
    catalog = terrain_velocity_catalog(
        env_cfg.scene.terrain.terrain_generator, env_cfg.commands.base_velocity.ranges, settings
    )
    cases = build_cases(catalog, args.protocol, args.traversal_speed)
    if args.case_ids is not None:
        requested = set(args.case_ids)
        cases = [case for case in cases if case["case_id"] in requested]
        if {case["case_id"] for case in cases} != requested:
            raise ValueError("Unknown case ID for this protocol.")
    plan = {"catalog": catalog.signature, "cases": cases, "trace_case_ids": args.trace_case_ids}
    plan_path = output / "suite.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    initial_policy = checkpoint.parent / "diagnostics" / "initial_policy.json"
    if initial_policy.is_file():
        training["initial_policy"] = json.loads(initial_policy.read_text())
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.log_dir = str(output)
    if args.episode_seconds is not None:
        env_cfg.episode_length_s = args.episode_seconds
    if args.device is not None:
        env_cfg.sim.device = args.device
        agent_cfg.device = args.device
    env_cfg.scene.terrain.terrain_generator.seed = args.seed
    env_cfg.scene.terrain.terrain_generator.curriculum = True
    env_cfg.events = FixedEvaluationEventsCfg()
    for group in vars(env_cfg.observations).values():
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
    env_cfg.commands.base_velocity.class_type = FixedEvaluationVelocityCommand
    env_cfg.commands.base_velocity.debug_vis = False
    env_cfg.commands.base_velocity.resampling_time_range = (10.0, 10.0)
    env_cfg.commands.base_velocity.transition_resampling_time_range = (2.0, 2.0)
    env_cfg.curriculum.terrain_levels = None
    env_cfg.curriculum.learning_progress = CurriculumTermCfg(
        func=FixedVelocityEvaluation,
        params={
            "settings": settings,
            "command_name": "base_velocity",
            "plan_path": str(plan_path),
            "output_dir": str(output),
        },
    )
    contract = to_dict(env_cfg)
    contract.pop("log_dir", None)
    contract["curriculum"]["learning_progress"]["params"].pop("output_dir")
    contract["curriculum"]["learning_progress"]["params"].pop("plan_path")
    source_paths = [Path(__file__), Path(__file__).with_name("velocity_suite.py")]
    import inspect

    source_paths += [
        Path(inspect.getfile(FixedVelocityEvaluation)),
        Path(inspect.getfile(FixedEvaluationVelocityCommand.__bases__[0])),
    ]
    from isaaclab.utils.curriculum.tracking import EpisodeVelocityTracking

    from isaaclab_tasks.core.velocity.mdp import rewards as velocity_rewards
    from isaaclab_tasks.core.velocity.mdp.evaluation.geometry import stair_exit_condition

    source_paths += [
        Path(inspect.getfile(EpisodeVelocityTracking)),
        Path(inspect.getfile(stair_exit_condition)),
        Path(velocity_rewards.__file__),
    ]
    source_paths += [
        Path(inspect.getfile(terrain_velocity_catalog)),
        Path(inspect.getfile(FixedVelocityEvaluation.__bases__[0])),
    ]
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    from .compare import training_source_hash

    manifest = {
        "evaluation_config": str(Path(args.env_config).resolve()) if args.env_config else None,
        "checkpoint": str(checkpoint),
        "clip_actions": agent_cfg.clip_actions,
        "expected_iterations": args.expected_iterations,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "contract_hash": fingerprint(
            {
                "config": contract,
                "cases": cases,
                "sources": sources,
                "runtime_sources": training_source_hash(),
                "clip_actions": agent_cfg.clip_actions,
            }
        ),
        "seed": args.seed,
        "training": training,
        "num_cases": len(cases),
        "protocol": args.protocol,
        "sources": sources,
        "config": contract,
        "status": "incomplete",
        "metric_convention": "clipped_relative_error_eq8; zero_targets_use_separate_normalized_extension",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Evaluating %d fixed cases from %s", len(cases), checkpoint)
    import gymnasium as gym

    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(wrapped, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            runner.load(str(checkpoint), map_location=agent_cfg.device)
            policy = runner.get_inference_policy(device=wrapped.device)
            controller = env.unwrapped.velocity_evaluation
            observations = wrapped.get_observations()
            limit = len(cases) * env.unwrapped.max_episode_length + 1
            with torch.inference_mode():
                for step in range(limit):
                    observations, _, _, _ = wrapped.step(policy(observations))
                    if len(controller.rows) == len(cases):
                        break
                    if step % 1000 == 0:
                        logger.info("Completed %d/%d evaluation cases", len(controller.rows), len(cases))
                else:
                    raise RuntimeError("Evaluation did not finish within the bounded episode budget.")
            report = summarize(controller.rows)
            report["contract_hash"] = manifest["contract_hash"]
            report["checkpoint"] = str(checkpoint)
            (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            manifest["status"] = "complete"
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            overall = report["overall"]
            print(
                f"Evaluation complete: {overall['episodes']} cases, failure={overall['failure_rate']:.3f}, "
                f"stair success={overall.get('traversal_success_rate')}; report: {output / 'summary.json'}"
            )
        finally:
            env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
