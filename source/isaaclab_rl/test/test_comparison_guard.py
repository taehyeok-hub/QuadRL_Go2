# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reject prepared comparisons when shared physical benchmark sources change."""

import json
from types import SimpleNamespace

import pytest

import isaaclab.paths

from isaaclab_rl.evaluation.compare import check_plan, training_source_hash
from isaaclab_rl.evaluation.velocity_suite import fingerprint

pytestmark = pytest.mark.unit


def test_comparison_guard_rejects_changed_terrain_and_robot(tmp_path, monkeypatch):
    monkeypatch.setattr(isaaclab.paths, "ISAACLAB_ROOT", str(tmp_path))
    terrain = tmp_path / "source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py"
    robot = tmp_path / "source/isaaclab_assets/isaaclab_assets/robots/unitree.py"
    source_paths = (
        terrain,
        robot,
        tmp_path / "source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py",
        tmp_path / "source/isaaclab/isaaclab/utils/curriculum/learning_progress.py",
        tmp_path / "source/isaaclab/isaaclab/managers/reward_manager.py",
        tmp_path / "source/isaaclab/isaaclab/envs/manager_based_rl_env.py",
        tmp_path / "source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_rsl_rl.py",
    )
    for path in source_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value = 1\n")
    config = tmp_path / "evaluation_env.yaml"
    config.write_text("seed: 101\n")
    plan = tmp_path / "experiments.json"
    plan.write_text(
        json.dumps(
            {
                "evaluation_config_hash": fingerprint(config.read_text()),
                "training_source_hash": training_source_hash(),
            }
        )
    )
    args = SimpleNamespace(plan=str(plan))
    check_plan(args)
    for path in (terrain, robot):
        original = path.read_text()
        path.write_text("value = 2\n")
        with pytest.raises(ValueError, match="Training source changed"):
            check_plan(args)
        path.write_text(original)
    check_plan(args)
