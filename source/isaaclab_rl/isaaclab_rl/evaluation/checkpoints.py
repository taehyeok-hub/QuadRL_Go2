# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select evaluation checkpoints and reject incomplete comparison runs."""

import re
from pathlib import Path

import torch


def evaluation_checkpoint(selector, experiment_root, expected_iterations=None):
    """Use the newest timestamped run and its highest numeric iteration, never file modification times."""
    if selector == "latest":
        root = Path(experiment_root)
        runs = (
            sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
            if root.exists()
            else []
        )
        if not runs:
            raise FileNotFoundError(f"No training run in {root}.")
        run = runs[-1]
        models = [
            (int(match.group(1)), path)
            for path in run.glob("model_*.pt")
            if (match := re.fullmatch(r"model_(\d+)\.pt", path.name))
        ]
        if not models:
            raise FileNotFoundError(f"Newest run has no checkpoint yet: {run}")
        path = max(models, key=lambda item: item[0])[1]
    else:
        path = Path(selector).expanduser()
    path = path.resolve(strict=True)
    if expected_iterations is not None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        completed = int(checkpoint.get("iter", -1)) + 1
        if completed != expected_iterations:
            raise ValueError(
                f"Incomplete or mismatched comparison run: {path} has {completed} iterations; "
                f"expected {expected_iterations}."
            )
    return path
