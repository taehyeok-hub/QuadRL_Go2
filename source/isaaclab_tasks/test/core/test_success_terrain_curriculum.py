# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode-end terrain decisions through the real curriculum manager lifecycle."""

from collections import UserDict
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import CurriculumManager, CurriculumTermCfg
from isaaclab.terrains import (
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
    TerrainImporter,
)

from isaaclab_tasks.core.velocity.mdp.curriculum.success import SuccessTerrainCfg, SuccessTerrainCurriculum
from isaaclab_tasks.utils import resolve_task_config

pytestmark = pytest.mark.unit


@pytest.fixture
def environment():
    generator = TerrainGeneratorCfg(
        size=(8.0, 8.0),
        num_rows=3,
        num_cols=2,
        curriculum=True,
        sub_terrains={
            "down": MeshPyramidStairsTerrainCfg(proportion=0.5, border_width=1.0),
            "up": MeshInvertedPyramidStairsTerrainCfg(proportion=0.5, border_width=1.0),
        },
    )
    terrain = TerrainImporter.__new__(TerrainImporter)
    terrain.cfg = SimpleNamespace(terrain_generator=generator)
    terrain.terrain_origins = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]],
            [[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]],
            [[20.0, 0.0, 0.0], [20.0, 10.0, 0.0]],
        ]
    )
    terrain.max_terrain_level = 3
    terrain.terrain_levels = torch.ones(3, dtype=torch.long)
    terrain.terrain_types = torch.tensor([0, 1, 0])
    terrain.env_origins = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types].clone()
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=SimpleNamespace(torch=terrain.env_origins.clone()),
            body_pos_w=SimpleNamespace(torch=torch.zeros(3, 4, 3)),
            projected_gravity_b=SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -1.0]] * 3)),
        ),
        find_bodies=lambda pattern: (list(range(4)), ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
    )
    scene = UserDict(robot=robot)
    scene.terrain = terrain
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        scene=scene,
        sim=SimpleNamespace(is_playing=lambda: True),
        cfg=SimpleNamespace(
            terminations={"fall": "base_contact"},
            commands={"base_velocity": {"range": (-1.0, 1.0)}},
            events={"reset": "center"},
            observations={"policy": "noisy"},
        ),
        step_dt=0.02,
        max_episode_length=1000,
        reset_buf=torch.zeros(3, dtype=torch.bool),
        termination_manager=SimpleNamespace(terminated=torch.zeros(3, dtype=torch.bool)),
    )
    env.command = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    env.command_manager = SimpleNamespace(get_command=lambda name: env.command)
    manager = CurriculumManager(
        SimpleNamespace(
            terrain_levels=CurriculumTermCfg(
                func=SuccessTerrainCurriculum,
                params={
                    "settings": SuccessTerrainCfg(
                        history_length=2, promotion_success_rate=0.5, demotion_success_rate=0.0
                    )
                },
            )
        ),
        env,
    )
    term = manager._class_term_cfgs[0].func
    return env, manager, term


def _pose(env, goal):
    robot = env.scene["robot"]
    robot.data.root_pos_w.torch.copy_(env.scene.terrain.env_origins)
    robot.data.root_pos_w.torch[:, 0] += torch.where(torch.tensor(goal), 3.5, 0.0)
    robot.data.body_pos_w.torch[:] = robot.data.root_pos_w.torch[:, None, :]
    robot.data.body_pos_w.torch[:, :, 0] += torch.tensor([-0.2, -0.2, 0.2, 0.2])
    robot.data.body_pos_w.torch[:, :, 1] += torch.tensor([-0.2, 0.2, -0.2, 0.2])


def test_manager_counts_real_episodes_and_preserves_windows_across_reset(environment):
    env, manager, term = environment
    manager.compute()
    manager.reset()
    assert (term.history.outcomes == -1).all()
    _pose(env, [True, True, True])
    manager.update(0.02)
    # Reaching a goal then falling is a failed episode; stationary/rotating commands never vote.
    env.termination_manager.terminated[1] = True
    env.reset_buf[:] = True
    manager.compute()
    manager.reset()
    assert term.history.outcomes.tolist() == [[1, -1], [0, -1], [-1, -1]]
    assert env.scene.terrain.terrain_levels.tolist() == [1, 1, 1]
    _pose(env, [False, False, False])
    env.termination_manager.terminated[:] = False
    manager.update(0.02)
    manager.compute()
    manager.reset()
    assert env.scene.terrain.terrain_levels.tolist() == [2, 0, 1]
    assert (term.history.outcomes == -1).all()
    # A manual reset during a live episode does not create another outcome.
    env.reset_buf[:] = False
    _pose(env, [False, False, False])
    manager.update(0.02)
    manager.compute()
    manager.reset()
    assert (term.history.outcomes == -1).all()
    assert env.scene.terrain.terrain_levels.tolist() == [2, 0, 1]


def test_checkpoint_keeps_partial_history_with_terrain_assignment(environment):
    env, manager, term = environment
    _pose(env, [True, False, False])
    manager.update(0.02)
    env.reset_buf[:] = True
    manager.compute()
    manager.reset()
    state = manager.state_dict()
    term.history.record([0], [False])
    env.scene.terrain.terrain_levels[:] = 0
    env.cfg.commands["base_velocity"]["range"] = (-2.0, 2.0)
    with pytest.raises(ValueError, match="--reset_curriculum"):
        manager.load_state_dict(state)
    assert env.scene.terrain.terrain_levels.tolist() == [0, 0, 0]
    env.cfg.commands["base_velocity"]["range"] = (-1.0, 1.0)
    _pose(env, [True, True, False])
    manager.update(0.02)
    manager.load_state_dict(state)
    assert term.history.outcomes.tolist() == [[1, -1], [0, -1], [-1, -1]]
    assert env.scene.terrain.terrain_levels.tolist() == [1, 1, 1]
    torch.testing.assert_close(
        env.scene.terrain.env_origins, torch.tensor([[10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [10.0, 0.0, 0.0]])
    )
    _pose(env, [False, False, False])
    manager.update(0.02)
    manager.compute()
    assert env.scene.terrain.terrain_levels.tolist() == [2, 0, 1]


def test_maximum_and_minimum_level_decisions_consume_their_windows(environment, monkeypatch):
    env, manager, term = environment
    env.scene.terrain.update_env_origins(
        torch.arange(3), torch.tensor([True, False, False]), torch.tensor([False, True, False])
    )
    term.history.record([0, 1], [True, False])
    _pose(env, [False, False, False])
    manager.update(0.02)
    env.reset_buf[:] = True
    monkeypatch.setattr(torch, "randint_like", lambda values, high: torch.ones_like(values))
    manager.compute()
    manager.reset()
    assert env.scene.terrain.terrain_levels.tolist() == [1, 0, 1]
    assert (term.history.outcomes == -1).all()


def test_go2_train_play_and_flat_resolve_independently():
    train, _ = resolve_task_config("Isaac-Velocity-Rough-UnitreeGo2", "rsl_rl_cfg_entry_point", overrides=[])
    assert train.curriculum.terrain_levels.func is SuccessTerrainCurriculum
    assert train.curriculum.terrain_levels.params["settings"].history_length == 2
    assert train.curriculum.terrain_levels.params["settings"].promotion_success_rate is None
    assert train.curriculum.terrain_levels.params["settings"].demotion_success_rate == 0.0
    assert train.curriculum.learning_progress is None
    assert train.terminations.assigned_terrain_boundary.time_out
    play, _ = resolve_task_config(
        "Isaac-Velocity-Rough-UnitreeGo2", "rsl_rl_cfg_entry_point", play_mode=True, overrides=[]
    )
    flat, _ = resolve_task_config("Isaac-Velocity-Flat-UnitreeGo2", "rsl_rl_cfg_entry_point", overrides=[])
    assert play.curriculum.terrain_levels is None
    assert flat.curriculum.terrain_levels is None
