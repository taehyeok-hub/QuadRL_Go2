# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-suite coverage, censored denominators, and safe config loading."""

import json

import pytest

from isaaclab.utils.curriculum import build_velocity_task_catalog

from isaaclab_rl.evaluation.velocity_suite import build_cases, summarize

pytestmark = pytest.mark.unit


def test_suite_covers_catalog_and_distinguishes_stair_directions():
    catalog = build_velocity_task_catalog(
        {"pyramid_stairs": [0], "pyramid_stairs_inv": [1]},
        2,
        {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_z": (-1.0, 1.0)},
        command_bins=(1, 1, 1),
    )
    cases = build_cases(catalog)
    tracking = [case for case in cases if case["protocol"] == "tracking"]
    traversal = [case for case in cases if case["protocol"] == "traversal"]
    assert {case["task_id"] for case in tracking} == set(range(catalog.num_tasks))
    assert len(traversal) == 12
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert len({case["heading"] for case in traversal}) == 3
    assert cases == build_cases(catalog)


def test_summary_preserves_censored_denominators():
    base = {
        "protocol": "tracking",
        "terrain": "stairs",
        "level": 1,
        "mode": "moving",
        "failed": False,
        "boundary": False,
        "steps": 100,
        "epte_sp": [20.0, None, None],
        "normalized_epte_sp": [30.0, 0.0, 0.0],
        "zero_command_drift": [None, 0.01, 0.02],
    }
    censored = dict(base, boundary=True, steps=50, epte_sp=[None] * 3, normalized_epte_sp=[None] * 3)
    report = summarize([base, censored])["overall"]
    assert report["epte_sp/vx"] == 20.0
    assert report["epte_sp_count/vx"] == 1
    assert report["epte_sp/yaw"] is None
    assert report["boundary_rate"] == 0.5
    json.dumps(report, allow_nan=False)


def test_saved_config_loader_rejects_executable_yaml(tmp_path):
    import yaml

    from isaaclab_rl.evaluation.locomotion import _ConfigLoader

    values = yaml.load(
        "x: !!python/tuple [1, 2]\ns: !!python/object/apply:builtins.slice [null, null, null]", Loader=_ConfigLoader
    )
    assert values == {"x": (1, 2), "s": slice(None)}
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.load('!!python/object/apply:os.system ["false"]', Loader=_ConfigLoader)


def test_comparison_keeps_different_evaluation_contracts_separate(tmp_path):
    from types import SimpleNamespace

    from isaaclab_rl.evaluation.compare import report

    folders = []
    for index, contract in enumerate(("fixed_a", "fixed_b")):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "manifest.json").write_text(
            json.dumps(
                {"status": "complete", "contract_hash": contract, "checkpoint": f"policy_{index}.pt", "seed": 101}
            )
        )
        (folder / "summary.json").write_text(
            json.dumps(
                {"contract_hash": contract, "overall": {"failure_rate": index * 0.1, "normalized_epte_sp/vx": 30.0}}
            )
        )
        folders.append(str(folder))
    output = tmp_path / "comparison.json"
    report(SimpleNamespace(results=folders, output=str(output)))
    result = json.loads(output.read_text())
    assert set(result["contracts"]) == {"fixed_a", "fixed_b"}
    assert all(len(entries) == 1 for entries in result["contracts"].values())


def test_starting_policy_identity_is_stable_and_detects_changed_actor(tmp_path):
    from types import SimpleNamespace

    import torch

    from isaaclab_rl.diagnostics.policy_provenance import write_policy_fingerprint

    runner = SimpleNamespace(alg=SimpleNamespace(actor=torch.nn.Linear(2, 1), critic=torch.nn.Linear(2, 1)))
    write_policy_fingerprint(runner, tmp_path)
    first = json.loads((tmp_path / "diagnostics/initial_policy.json").read_text())
    write_policy_fingerprint(runner, tmp_path)
    assert json.loads((tmp_path / "diagnostics/initial_policy.json").read_text()) == first
    with torch.no_grad():
        runner.alg.actor.weight.add_(1)
    write_policy_fingerprint(runner, tmp_path)
    changed = json.loads((tmp_path / "diagnostics/initial_policy.json").read_text())
    assert changed["actor_sha256"] != first["actor_sha256"]
    assert changed["critic_sha256"] == first["critic_sha256"]


def test_common_environment_config_overrides_training_terrain(tmp_path):
    from isaaclab.utils.io import dump_yaml

    from isaaclab_rl.evaluation.locomotion import load_environment_config

    from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

    training = UnitreeGo2RoughEnvCfg()
    reference = training.copy()
    training.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.23)
    reference.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.20)
    path = tmp_path / "evaluation_env.yaml"
    dump_yaml(str(path), reference)
    training.sim.visualizer_cfgs = []
    load_environment_config(path, training)
    assert training.sim.visualizer_cfgs == []
    assert training.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range == (0.05, 0.20)
