# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare controlled learning comparisons and compare completed fixed-suite reports."""

import argparse
import json
import shlex
from pathlib import Path

from .velocity_suite import fingerprint

VARIANTS = {
    "uniform": {"sampling_strategy": "uniform"},
    "lp": {"sampling_strategy": "lp"},
    "lp_beta10": {"sampling_strategy": "lp", "temperature": 10.0},
    "lp_raw_return": {"sampling_strategy": "lp", "reward_scale": 1.0},
    "lp_no_exploration": {"sampling_strategy": "lp", "exploration": 0.0},
    "lp_min32": {"sampling_strategy": "lp", "min_samples": 32},
}


def training_source_hash():
    import hashlib

    from isaaclab.paths import ISAACLAB_ROOT

    root = Path(ISAACLAB_ROOT)
    locations = (
        "source/isaaclab_tasks/isaaclab_tasks/core/velocity",
        "source/isaaclab/isaaclab/utils/curriculum",
        "source/isaaclab/isaaclab/terrains",
        "source/isaaclab_assets/isaaclab_assets/robots/unitree.py",
        "source/isaaclab/isaaclab/envs/mdp",
        "source/isaaclab/isaaclab/envs/manager_based_env.py",
        "source/isaaclab/isaaclab/envs/manager_based_env_cfg.py",
        "source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
        "source/isaaclab/isaaclab/envs/manager_based_rl_env_cfg.py",
        "source/isaaclab/isaaclab/managers",
        "source/isaaclab/isaaclab/actuators",
        "source/isaaclab_newton/isaaclab_newton/physics",
        "source/isaaclab_newton/isaaclab_newton/assets/articulation",
        "source/isaaclab_newton/isaaclab_newton/envs/mdp",
        "source/isaaclab_newton/isaaclab_newton/actuators",
        "source/isaaclab_newton/isaaclab_newton/sensors/contact_sensor",
        "source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py",
        "source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py",
        "source/isaaclab_rl/isaaclab_rl/utils/curriculum_checkpoint.py",
        "source/isaaclab_rl/isaaclab_rl/utils/online_curriculum.py",
        "source/isaaclab_rl/isaaclab_rl/utils/snapshot_curriculum.py",
    )
    files = []
    for name in locations:
        path = root / name
        files.extend(path.rglob("*.py") if path.is_dir() else [path])
    return fingerprint(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)
            if path.is_file()
        }
    )


def check_plan(args):
    plan = json.loads(Path(args.plan).read_text())
    if fingerprint((Path(args.plan).parent / "evaluation_env.yaml").read_text()) != plan["evaluation_config_hash"]:
        raise ValueError("Common evaluation configuration changed after experiment preparation.")
    if plan["training_source_hash"] != training_source_hash():
        raise ValueError("Training source changed after preparing this experiment. Prepare a new comparison plan.")


def prepare(args):
    """Write explicit launch commands; never start long GPU jobs as a side effect."""
    from isaaclab.utils.io import dump_yaml

    from isaaclab_tasks.utils import resolve_task_config

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from isaaclab.paths import ISAACLAB_ROOT

    root = str(ISAACLAB_ROOT)
    base_settings = {
        "sampling_strategy": "lp",
        "temperature": 0.2,
        "return_mode": "episodic",
        "min_samples": 8,
        "measurement_mode": "online_window",
        "update_interval_iterations": 100,
        "update_interval_steps": 2500,
        "reward_scale": 15.0,
        "exploration": 0.1,
        "diagnostics": True,
        "debug_metrics": False,
    }
    jobs = []
    for seed in args.seeds:
        for variant in args.variants:
            settings = dict(base_settings, **VARIANTS[variant])
            overrides = list(args.overrides) + [
                f"env.curriculum.learning_progress.params.settings.{name}="
                f"{str(value).lower() if isinstance(value, bool) else value}"
                for name, value in settings.items()
            ]
            experiment = f"go2_compare_{output.name}_{variant}_s{seed}"
            cfg, agent = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", overrides=overrides)
            cfg.scene.num_envs = args.num_envs
            cfg.seed = seed
            agent.seed = seed
            agent.max_iterations = args.iterations
            agent.logger = args.logger
            agent.experiment_name = experiment
            agent.run_name = variant
            agent.wandb_project = args.log_project_name
            (output / f"{variant}_s{seed}_settings.json").write_text(
                json.dumps({"env": cfg.to_dict(), "agent": agent.to_dict()}, indent=2, default=str)
            )
            common_config = output / "evaluation_env.yaml"
            if not common_config.exists():
                dump_yaml(str(common_config), cfg)
            train = [
                "uv",
                "run",
                "isaaclab",
                "train",
                "--rl_library",
                "rsl_rl",
                "--task",
                args.task,
                "--num_envs",
                str(args.num_envs),
                "--max_iterations",
                str(args.iterations),
                "--seed",
                str(seed),
                "--logger",
                args.logger,
                "--log_project_name",
                args.log_project_name,
                "--run_name",
                variant,
                f"agent.experiment_name={experiment}",
                *overrides,
            ]
            evaluations = []
            for evaluation_seed in args.evaluation_seeds:
                evaluations.append(
                    [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "isaaclab_rl.evaluation.locomotion",
                        "--task",
                        args.task,
                        "--checkpoint",
                        "latest",
                        "--expected-iterations",
                        str(args.iterations),
                        "--env-config",
                        str(common_config),
                        "--experiment-name",
                        experiment,
                        "--seed",
                        str(evaluation_seed),
                        "--num-envs",
                        str(args.eval_num_envs),
                        "--output",
                        str(output / "results" / f"{variant}_s{seed}_e{evaluation_seed}"),
                    ]
                )
            jobs.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "settings": settings,
                    "train": train,
                    "evaluate": evaluations,
                    "overrides": overrides,
                }
            )
    manifest = {
        "task": args.task,
        "root": root,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "training_source_hash": training_source_hash(),
        "evaluation_config_hash": fingerprint((output / "evaluation_env.yaml").read_text()),
        "evaluation_seeds": args.evaluation_seeds,
        "jobs": jobs,
    }
    manifest["plan_hash"] = fingerprint(manifest)
    (output / "experiments.json").write_text(json.dumps(manifest, indent=2))
    for name, commands in (
        ("train", [job["train"] for job in jobs]),
        ("evaluate", [cmd for job in jobs for cmd in job["evaluate"]]),
    ):
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", "cd " + shlex.quote(root)]
        guard = [
            "uv",
            "run",
            "python",
            "-m",
            "isaaclab_rl.evaluation.compare",
            "check",
            "--plan",
            str(output / "experiments.json"),
        ]
        for command in commands:
            lines.append(shlex.join(guard))
            lines.append(shlex.join(command))
        (output / f"run_{name}.sh").write_text("\n".join(lines) + "\n")
    print(
        f"Prepared {len(jobs)} fresh training jobs and {len(jobs) * len(args.evaluation_seeds)} evaluations in {output}"
    )
    print("No training has been started. Run run_train.sh, then run_evaluate.sh.")


def report(args):
    """Keep distinct evaluation contracts separate rather than averaging unlike conditions."""
    contracts = {}
    for folder in args.results:
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
        if manifest["status"] != "complete" or summary["contract_hash"] != manifest["contract_hash"]:
            raise ValueError(f"Incomplete or inconsistent evaluation: {folder}")
        entry = {
            "result": str(folder),
            "checkpoint": manifest["checkpoint"],
            "seed": manifest["seed"],
            "training": manifest.get("training", {}),
            **summary["overall"],
        }
        contracts.setdefault(manifest["contract_hash"], []).append(entry)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"contracts": contracts}, indent=2, allow_nan=False))
    for contract, entries in contracts.items():
        print(f"Evaluation contract {contract[:12]} ({len(entries)} policies):")
        for entry in entries:
            print(
                f"  {entry['result']}: failure={entry['failure_rate']:.3f}, "
                f"stairs={entry.get('traversal_success_rate')}, normalized_vx={entry['normalized_epte_sp/vx']}"
            )
    if len(contracts) > 1:
        print("Different evaluation contracts were kept separate; do not compare across groups.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--output", required=True)
    build.add_argument("--task", default="Isaac-Velocity-Rough-UnitreeGo2")
    build.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=["uniform", "lp"])
    build.add_argument("--seeds", nargs="+", type=int, default=[42])
    build.add_argument("--evaluation-seeds", nargs="+", type=int, default=[101])
    build.add_argument("--num-envs", type=int, default=4096)
    build.add_argument("--eval-num-envs", type=int, default=128)
    build.add_argument("--iterations", type=int, default=10000)
    build.add_argument("--log-project-name", default="go2_lp_compare")
    build.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
    build.add_argument("--overrides", nargs="*", default=[])
    build.set_defaults(func=prepare)
    read = commands.add_parser("report")
    read.add_argument("--results", nargs="+", required=True)
    read.add_argument("--output", required=True)
    read.set_defaults(func=report)
    check = commands.add_parser("check")
    check.add_argument("--plan", required=True)
    check.set_defaults(func=check_plan)
    args = parser.parse_args(argv)
    if args.command == "prepare" and min(args.num_envs, args.eval_num_envs, args.iterations) <= 0:
        parser.error("Environment counts and iterations must be positive.")
    args.func(args)


if __name__ == "__main__":
    main()
