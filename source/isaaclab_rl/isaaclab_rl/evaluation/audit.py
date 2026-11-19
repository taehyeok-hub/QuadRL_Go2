# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect checkpoint task concentration and optional diagnostic windows without a simulator."""

import argparse
import json
from pathlib import Path

import torch

from isaaclab.utils.curriculum import build_velocity_task_catalog
from isaaclab.utils.curriculum.catalog import COMMAND_MODES


def checkpoint_audit(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = (checkpoint.get("infos") or {}).get("isaaclab_curriculum", {}).get("learning_progress")
    if state is None:
        return {"checkpoint": str(path), "iteration": checkpoint.get("iter"), "lp_available": False}
    signature = state["catalog"]
    catalog = build_velocity_task_catalog(
        dict(zip(signature["terrain_names"], signature["column_groups"])),
        signature["num_levels"],
        dict(zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), signature["command_ranges"])),
        tuple(signature["command_bins"]),
        signature["modes"],
    )
    sampler = state["sampler"]
    p = torch.as_tensor(sampler.get("_probabilities", sampler.get("probabilities")), dtype=torch.float64)
    progress = torch.as_tensor(sampler.get("learning_progress", sampler.get("progress")), dtype=torch.float64)
    initialized = torch.as_tensor(sampler.get("initialized", sampler.get("has_baseline")), dtype=torch.bool)
    if state["settings"].get("sampling_strategy") == "uniform":
        p.fill_(1.0 / len(p))
    entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum()
    top = p.topk(min(20, p.numel())).indices.tolist()
    return {
        "checkpoint": str(path),
        "iteration": checkpoint.get("iter"),
        "lp_available": True,
        "settings": state["settings"],
        "tasks": len(p),
        "updates": sampler.get("update_count", sampler.get("num_updates")),
        "entropy": float(entropy),
        "effective_tasks": float(entropy.exp()),
        "max_probability": float(p.max()),
        "top10_mass": float(p.topk(min(10, p.numel())).values.sum()),
        "baseline_tasks": int(initialized.sum()),
        "progress_range": [float(progress.min()), float(progress.max())],
        "top_tasks": [
            {
                "task_id": i,
                "probability": float(p[i]),
                "terrain": catalog.terrain_names[int(catalog.terrain_type_ids[i])],
                "level": int(catalog.terrain_levels[i]),
                "mode": COMMAND_MODES[int(catalog.modes[i])],
                "transition_type": int(catalog.transition_types[i]),
                "LP": float(progress[i]),
            }
            for i in top
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = checkpoint_audit(Path(args.checkpoint).resolve(strict=True))
    if args.windows:
        windows = []
        with Path(args.windows).open() as stream:
            for line in stream:
                record = json.loads(line)
                row = {
                    key: record[key] for key in ("step", "entropy", "effective_tasks", "max_probability", "top10_mass")
                }
                valid = [i for i, count in enumerate(record["counts"]) if count >= 2]
                for key in ("return", "LP_score", "episode_steps", "boundary_truncation", "failure"):
                    if key in record["statistics"] and valid:
                        std = record["statistics"][key]["sample_std"]
                        i = max(valid, key=lambda index: std[index])
                        row[f"{key}_largest_std"] = {"task_id": i, "std": std[i], "count": record["counts"][i]}
                windows.append(row)
        result["windows"] = windows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({key: value for key, value in result.items() if key not in ("top_tasks", "windows")}, indent=2))


if __name__ == "__main__":
    main()
