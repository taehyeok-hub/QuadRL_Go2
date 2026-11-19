# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LP task assignment, terminal accounting, and checkpoint compatibility without a scene build."""

from collections import UserDict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.envs.mdp.rewards import lin_vel_z_l2
from isaaclab.managers import CommandManager, CurriculumTermCfg, RewardTermCfg, TerminationManager, TerminationTermCfg
from isaaclab.terrains import MeshInvertedPyramidStairsTerrainCfg, MeshPyramidStairsTerrainCfg, TerrainGeneratorCfg

from isaaclab_tasks.core.velocity.mdp.curriculum.lp.commands import LearningProgressVelocityCommand
from isaaclab_tasks.core.velocity.mdp.curriculum.lp.config import LearningProgressCfg
from isaaclab_tasks.core.velocity.mdp.curriculum.lp.learning_progress import LearningProgressTerrainCurriculum
from isaaclab_tasks.core.velocity.mdp.curriculum.terminations import assigned_terrain_boundary
from isaaclab_tasks.utils import resolve_task_config

pytestmark = pytest.mark.unit


def test_resolved_play_config_constructs_manual_velocity_commands(make_environment):
    """Play-mode overrides remain instantiable after lazy callable resolution."""
    from isaaclab.envs.mdp.commands import UniformVelocityCommand

    cfg, _ = resolve_task_config(
        "Isaac-Velocity-Rough-UnitreeGo2", "rsl_rl_cfg_entry_point", play_mode=True, overrides=[]
    )
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env, _ = make_environment(num_envs=1)
    commands = CommandManager(cfg.commands, env)
    commands.reset()
    assert type(commands.get_term("base_velocity")) is UniformVelocityCommand
    assert not commands.get_command("base_velocity").any()
    assert cfg.curriculum.learning_progress is None


@pytest.fixture
def make_environment():
    """Supply physical state only; production command and curriculum code perform all bookkeeping."""

    def make(num_envs=2, settings=None):
        generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            num_rows=2,
            num_cols=2,
            curriculum=True,
            sub_terrains={
                "descending": MeshPyramidStairsTerrainCfg(proportion=0.5),
                "ascending": MeshInvertedPyramidStairsTerrainCfg(proportion=0.5),
            },
        )
        terrain = SimpleNamespace(
            cfg=SimpleNamespace(terrain_generator=generator, physics_material={"friction": 0.8}),
            generation_seed=42,
            terrain_levels=torch.zeros(num_envs, dtype=torch.long),
            terrain_types=torch.zeros(num_envs, dtype=torch.long),
            terrain_origins=torch.tensor(
                [[[10.0, 0.0, 1.0], [10.0, 8.0, -1.0]], [[18.0, 0.0, 2.0], [18.0, 8.0, -2.0]]]
            ),
            env_origins=torch.zeros(num_envs, 3),
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_lin_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
                root_ang_vel_b=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
                root_pos_w=SimpleNamespace(torch=torch.zeros(num_envs, 3)),
            )
        )
        scene = UserDict({"robot": robot})
        scene.terrain = terrain
        scene.env_origins = terrain.env_origins
        terminations = {
            name: torch.zeros(num_envs, dtype=torch.bool) for name in ("assigned_terrain_boundary", "time_out")
        }
        env = SimpleNamespace(
            cfg=SimpleNamespace(
                seed=42,
                decimation=4,
                actions={"joint_pos": {"scale": 0.25}},
                scene=SimpleNamespace(
                    robot=SimpleNamespace(
                        spawn={"usd_path": "go2.usd", "rigid_props": {"disable_gravity": False}},
                        init_state={"joint_pos": {".*": 0.0}},
                        actuators={"legs": {"stiffness": 25.0, "damping": 0.5}},
                        soft_joint_pos_limit_factor=0.9,
                    )
                ),
                sim=SimpleNamespace(
                    dt=0.005,
                    gravity=(0.0, 0.0, -9.81),
                    physics_material={"friction": 0.8},
                    use_newton_actuators=True,
                    physics={"class_type": "NewtonManager", "num_substeps": 1},
                ),
                rewards={"vertical_velocity": RewardTermCfg(func=lin_vel_z_l2, weight=-1.0)},
                observations={"policy": {"enable_corruption": True}},
                events={"reset": {"pose_range": {"x": (-0.5, 0.5)}}},
                terminations={"time_out": {"time_out": True}},
            ),
            num_envs=num_envs,
            device="cpu",
            step_dt=0.02,
            scene=scene,
            sim=SimpleNamespace(
                is_playing=lambda: True,
                vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda callback: None),
            ),
            extras={},
            reward_buf=torch.zeros(num_envs),
            reset_buf=torch.zeros(num_envs, dtype=torch.bool),
            max_episode_length=8,
            termination_manager=SimpleNamespace(
                terminated=torch.zeros(num_envs, dtype=torch.bool),
                active_terms=list(terminations),
                get_term=terminations.__getitem__,
            ),
        )
        command_cfg = UniformVelocityCommandCfg(
            class_type=LearningProgressVelocityCommand,
            asset_name="robot",
            resampling_time_range=(0.02, 0.02),
            transition_resampling_time_range=(0.02, 0.02),
            debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
            ),
        )
        env.command_manager = CommandManager(SimpleNamespace(base_velocity=command_cfg), env)
        settings = settings or LearningProgressCfg(measurement_mode="online", command_bins=(1, 1, 1))
        curriculum = LearningProgressTerrainCurriculum(
            CurriculumTermCfg(func=LearningProgressTerrainCurriculum, params={"settings": settings}), env
        )
        return env, curriculum

    return make


def test_automatic_modes_obey_assigned_ranges_across_command_resampling(make_environment):
    """All modes are reachable without rel_* quotas and retain their task until episode reset."""
    env, curriculum = make_environment(num_envs=256)
    torch.manual_seed(19)
    curriculum(env, slice(None), curriculum.settings)
    command = env.command_manager.get_term("base_velocity")
    assert set(command.task_modes.tolist()) == {0, 1, 2, 3}
    assert command.cfg.rel_standing_envs == command.cfg.rel_rotating_envs == command.cfg.rel_transition_envs == 0.0
    assigned_modes = command.task_modes.clone()
    env.command_manager.reset()
    initial = command.command.clone()
    for _ in range(2):
        env.command_manager.compute(0.02)
        torch.testing.assert_close(command.task_modes, assigned_modes)
        assert torch.all(command.command >= command.task_bounds[:, :, 0])
        assert torch.all(command.command <= command.task_bounds[:, :, 1])
        assert not command.command[command.is_standing_env].any()
        assert not command.command[command.is_rotating_env, :2].any()
    torch.testing.assert_close(command.command[command.is_transition_env], initial[command.is_transition_env])
    torch.testing.assert_close(
        env.scene.env_origins,
        env.scene.terrain.terrain_origins[env.scene.terrain.terrain_levels, env.scene.terrain.terrain_types],
    )


@pytest.mark.parametrize("debug_metrics", [False, True])
def test_terminal_error_and_return_are_captured_before_reset(make_environment, debug_metrics):
    """Record falls and partial returns while distinguishing boundary exits from ordinary early timeouts."""
    env, curriculum = make_environment(
        num_envs=3,
        settings=LearningProgressCfg(
            measurement_mode="online", command_bins=(1, 1, 1), modes=("moving",), debug_metrics=debug_metrics
        ),
    )
    curriculum(env, slice(None), curriculum.settings)
    command = env.command_manager.get_term("base_velocity")
    command.vel_command_b[:] = torch.tensor([1.0, 0.0, 0.0])
    for speed, reward in ((0.8, 2.0), (0.6, 4.0)):
        command.robot.data.root_lin_vel_b.torch[:, 0] = speed
        env.reward_buf[:] = reward
        curriculum.update(0.02)
    env.reset_buf[:] = True
    env.termination_manager.terminated[0] = True
    env.termination_manager.get_term("assigned_terrain_boundary")[1] = True
    env.termination_manager.get_term("time_out")[2] = True
    logs = curriculum(env, slice(None), curriculum.settings)
    command.robot.data.root_lin_vel_b.torch[:] = 0.0
    env.command_manager.reset()
    torch.testing.assert_close(logs["epte_sp/vx"], torch.tensor([82.5]))
    if debug_metrics:
        torch.testing.assert_close(logs["episode_return"], torch.tensor([6.0, 6.0, 6.0]))
        torch.testing.assert_close(logs["LP_return"], torch.tensor([0.4, 0.4, 0.4]))
        assert "terrain/descending/moving/failure_rate" in logs or "terrain/ascending/moving/failure_rate" in logs
    else:
        assert "LP_return" not in logs and "episode_return" not in logs
        assert len(logs) <= 22
        assert not any(key.startswith(("mode/", "level/")) or key.endswith(("/LP", "/episodes_local")) for key in logs)
        assert not any("/moving/" in key for key in logs)
        assert "normalized_epte_sp_coverage" in logs
        assert logs["effective_tasks"] >= curriculum.catalog.num_tasks / 2 - 1e-6
        assert logs["uniform_kl"] <= curriculum.settings.max_uniform_kl + 1e-12
    torch.testing.assert_close(logs["failure_rate"], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(logs["boundary_truncation_rate"], torch.tensor([0.0, 1.0, 0.0]))
    sums, counts = curriculum.sampler.pending_sums, curriculum.sampler.pending_counts
    assert counts.sum() == 3
    torch.testing.assert_close(sums.sum(), torch.tensor(1.2, dtype=torch.float64))
    assert not curriculum.tracking.steps.any()


def test_rectangle_boundary_is_a_timeout_including_diagonal_exits(make_environment):
    """A radial displacement beyond four meters is still inside until a rectangular edge is crossed."""
    env, _ = make_environment(num_envs=4)
    env.scene.env_origins[:] = torch.tensor([10.0, 8.0, -1.0])
    env.scene["robot"].data.root_pos_w.torch[:] = env.scene.env_origins + torch.tensor(
        [[3.0, 3.0, 0.0], [4.0, 0.0, 0.0], [-4.0, -4.0, 0.0], [0.0, 3.99, 0.0]]
    )
    manager = TerminationManager({"boundary": TerminationTermCfg(func=assigned_terrain_boundary, time_out=True)}, env)
    expected = torch.tensor([False, True, True, False])
    torch.testing.assert_close(manager.compute(), expected)
    torch.testing.assert_close(manager.time_outs, expected)
    assert not manager.terminated.any()


def test_resume_restores_lp_history_and_rejects_changed_task_space(make_environment, monkeypatch, caplog):
    """Resume only comparable return histories, preserving task and reward compatibility checks."""
    settings = LearningProgressCfg(
        measurement_mode="online", command_bins=(1, 1, 1), min_samples=4, update_interval_steps=1
    )
    _, original = make_environment(settings=settings)
    for rewards in ([1.0] * 4 + [2.0] * 4, [3.0] * 4 + [1.0] * 4):
        original.sampler.observe(torch.tensor([0] * 4 + [1] * 4), torch.tensor(rewards))
        original.sampler.step()
    original.sampler.observe(torch.tensor([0, 1]), torch.tensor([2.0, 3.0]))
    state = original.state_dict()
    _, restored = make_environment(settings=settings)
    restored.load_state_dict(state)
    sums, counts = restored.sampler.pending_sums, restored.sampler.pending_counts
    torch.testing.assert_close(sums[:2], torch.tensor([2.0, 3.0], dtype=torch.float64) / 15)
    torch.testing.assert_close(counts[:2], torch.tensor([1, 1]))
    _, debug = make_environment(
        settings=LearningProgressCfg(
            measurement_mode="online",
            command_bins=(1, 1, 1),
            min_samples=4,
            update_interval_steps=1,
            debug_metrics=True,
        )
    )
    state["settings"].pop("debug_metrics")
    debug.load_state_dict(state)
    torch.testing.assert_close(debug.sampler.probabilities, original.sampler.probabilities)
    _, incompatible = make_environment(settings=LearningProgressCfg(measurement_mode="online", command_bins=(2, 1, 1)))
    with pytest.raises(ValueError, match="task ranges or settings"):
        incompatible.load_state_dict(state)
    restored.terrain.cfg.terrain_generator.size = (10.0, 8.0)
    with pytest.raises(ValueError, match="terrain configuration"):
        restored.load_state_dict(state)

    changed_env, changed = make_environment(settings=settings)
    changed_env.cfg.rewards["vertical_velocity"].weight = -2.0
    with pytest.raises(ValueError, match="reward configuration or episode horizon"):
        changed.load_state_dict(state)
    changed_env.cfg.rewards["vertical_velocity"].weight = -1.0
    changed_env.max_episode_length = 9
    with pytest.raises(ValueError, match="reward configuration or episode horizon"):
        changed.load_state_dict(state)
    changed_env.max_episode_length = 8
    changed_env.step_dt = 0.04
    with pytest.raises(ValueError, match="reward configuration or episode horizon"):
        changed.load_state_dict(state)

    legacy = dict(state)
    legacy.pop("return_contract")
    debug.load_state_dict(legacy)
    assert debug.sampler.update_count == 0
    assert not debug.sampler.initialized.any()
    assert not debug.sampler.pending_counts.any()
    torch.testing.assert_close(
        debug.sampler.probabilities, torch.full_like(debug.sampler.probabilities, 1 / debug.catalog.num_tasks)
    )
    assert "no return contract" in caplog.text

    legacy = dict(state)
    legacy.pop("sampler_algorithm")
    debug.load_state_dict(state)
    debug.load_state_dict(legacy)
    assert debug.sampler.update_count == 0
    assert not debug.sampler.initialized.any()

    legacy = {**state, "settings": {**state["settings"], "return_mode": "boundary_extrapolated"}}
    debug.load_state_dict(state)
    debug.load_state_dict(legacy)
    assert debug.sampler.update_count == 0
    assert not debug.sampler.initialized.any()
    assert not debug.sampler.pending_counts.any()
    torch.testing.assert_close(
        debug.sampler.probabilities, torch.full_like(debug.sampler.probabilities, 1 / debug.catalog.num_tasks)
    )

    # Multi-rank resume retains synchronized priorities without duplicating one rank's pending samples.
    monkeypatch.setattr(
        "isaaclab_tasks.core.velocity.mdp.curriculum.lp.learning_progress.dist.is_available", lambda: True
    )
    monkeypatch.setattr(
        "isaaclab_tasks.core.velocity.mdp.curriculum.lp.learning_progress.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        "isaaclab_tasks.core.velocity.mdp.curriculum.lp.learning_progress.dist.get_world_size", lambda: 2
    )
    monkeypatch.setattr("isaaclab_tasks.core.velocity.mdp.curriculum.lp.learning_progress.dist.get_rank", lambda: 0)
    _, distributed = make_environment(settings=settings)
    distributed.load_state_dict(state)
    sums, counts = distributed.sampler.pending_sums, distributed.sampler.pending_counts
    assert not sums.any()
    assert not counts.any()
    torch.testing.assert_close(distributed.sampler.mean_returns, original.sampler.mean_returns)
    torch.testing.assert_close(distributed.sampler.probabilities, original.sampler.probabilities)
    assert "discards unsynchronized pending episode windows" in caplog.text
    assert sum(state["sampler"]["pending_counts"]) == 2


@pytest.mark.parametrize("strategy", ["lp", "uniform"])
def test_sampling_ablation_preserves_progress_but_changes_only_assignment(make_environment, strategy):
    env, curriculum = make_environment(
        settings=LearningProgressCfg(
            measurement_mode="online",
            command_bins=(1, 1, 1),
            min_samples=1,
            update_interval_steps=1,
            sampling_strategy=strategy,
        )
    )
    tasks = torch.tensor([0, 1])
    curriculum.sampler.observe(tasks, torch.tensor([0.0, 0.0]))
    curriculum.update(0.02)
    curriculum.sampler.observe(tasks, torch.tensor([3.0, 0.0]))
    curriculum.update(0.02)
    assert curriculum.sampler.learning_progress[0] == 3.0 / 15
    if strategy == "uniform":
        torch.testing.assert_close(
            curriculum.sampler.probabilities,
            torch.full_like(curriculum.sampler.probabilities, 1 / curriculum.catalog.num_tasks),
        )
    else:
        assert curriculum.sampler.probabilities[0] > curriculum.sampler.probabilities[1]


def test_legacy_return_setting_cannot_enable_extrapolation(make_environment):
    env, curriculum = make_environment(
        settings=LearningProgressCfg(
            measurement_mode="online", command_bins=(1, 1, 1), return_mode="boundary_extrapolated"
        )
    )
    curriculum(env, slice(None), curriculum.settings)
    env.reward_buf[:] = 2.0
    curriculum.update(0.02)
    env.reset_buf[:] = True
    curriculum(env, slice(None), curriculum.settings)
    sums, counts = curriculum.sampler.pending_sums, curriculum.sampler.pending_counts
    torch.testing.assert_close(sums.sum(), torch.tensor(4.0 / 15, dtype=torch.float64))
    assert counts.sum() == 2


def test_stair_exit_requires_all_feet_correct_direction_and_stability():
    from isaaclab_tasks.core.velocity.mdp.evaluation.geometry import stair_exit_condition

    root = torch.tensor([[3.4, 0.0]] * 6)
    feet = torch.tensor([[[3.2, 0.1], [3.2, -0.1], [3.6, 0.1], [3.6, -0.1]]] * 6)
    feet[1, 0, 0] = 2.9
    feet[2, :, 0] *= -1
    root[2, 0] *= -1
    gravity = torch.tensor([-1.0, -1.0, -1.0, 0.0, -1.0, -1.0])
    failed = torch.tensor([False, False, False, False, True, False])
    root[5] = torch.tensor([3.4, 3.4])
    feet[5, :, 1] += 3.4
    heading = torch.tensor([0.0] * 5 + [torch.pi / 4])
    result = stair_exit_condition(feet, root, torch.full((6, 2), 3.0), heading, gravity, failed)
    assert result.tolist() == [True, False, False, False, False, True]


def test_distributed_sparse_windows_reduce_only_new_observations(make_environment, monkeypatch):
    """Retained global samples must not be counted again at the next synchronization."""
    _, curriculum = make_environment(
        settings=LearningProgressCfg(
            measurement_mode="online", command_bins=(1, 1, 1), min_samples=3, update_interval_steps=1
        )
    )
    monkeypatch.setattr(
        "isaaclab_tasks.core.velocity.mdp.curriculum.lp.learning_progress.dist.is_available", lambda: True
    )
    monkeypatch.setattr(
        "isaaclab_tasks.core.velocity.mdp.curriculum.lp.learning_progress.dist.is_initialized", lambda: True
    )
    monkeypatch.setattr(
        "isaaclab_tasks.core.velocity.mdp.curriculum.lp.learning_progress.dist.all_reduce",
        lambda values: values.mul_(2),
    )
    curriculum.sampler.observe([0], [15.0])
    curriculum.update(0.02)
    assert curriculum.sampler.pending_counts[0] == 2
    curriculum.update(0.02)
    assert curriculum.sampler.pending_counts[0] == 2
    assert not curriculum.sampler.initialized[0]
    curriculum.sampler.observe([0], [15.0])
    curriculum.update(0.02)
    assert curriculum.sampler.pending_counts[0] == 0
    assert curriculum.sampler.episode_counts[0] == 4
    assert curriculum.sampler.mean_returns[0] == 1.0


def test_snapshot_probe_counts_only_first_terminal_and_ignores_padding(make_environment, monkeypatch):
    """A completed slot must not contribute another episode while slower slots finish."""
    settings = LearningProgressCfg(
        measurement_mode="snapshot", command_bins=(1, 1, 1), modes=("moving",), min_samples=2
    )
    env, curriculum = make_environment(num_envs=3, settings=settings)
    curriculum.tracking.steps[:] = 7
    curriculum._returns[:] = 12.0
    completed_metrics = Mock(return_value={})
    monkeypatch.setattr(curriculum.metrics, "episodes", completed_metrics)
    tasks = torch.tensor([0, 1, -1])
    curriculum.begin_snapshot_batch(tasks)
    curriculum(env, slice(None), settings)
    assert curriculum.task_ids.tolist() == [0, 1, 0]
    assert not curriculum.sampler.sample_count
    for rewards, terminals, resets in (
        ([1.0, 2.0, float("nan")], [True, False, False], [True, False, True]),
        ([float("nan"), 3.0, float("nan")], [False, True, False], [True, True, True]),
    ):
        env.reward_buf[:] = torch.tensor(rewards)
        env.reset_buf[:] = torch.tensor(terminals)
        curriculum.update(0.02)
        assert env.reset_buf.tolist() == resets
        inactive_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
        curriculum(env, inactive_ids, settings)
        if not curriculum.snapshot_batch_complete:
            with pytest.raises(RuntimeError, match="Manual resets"):
                curriculum(env, torch.tensor([1]), settings)
    assert curriculum.snapshot_batch_complete
    torch.testing.assert_close(curriculum.snapshot_batch_returns, torch.tensor([1.0, 5.0, 0.0], dtype=torch.float64))
    assert curriculum._step == curriculum.sampler.control_steps == 0
    assert not curriculum.sampler.pending_counts.any()
    curriculum.end_snapshot_batch()
    curriculum(env, slice(None), settings)
    completed_metrics.assert_not_called()
    assert not curriculum.tracking.steps.any()
    curriculum.snapshot_controller_active = True
    env.reset_buf[:] = False
    env.reward_buf[:] = 4.0
    curriculum.update(0.02)
    env.reset_buf[:] = True
    curriculum(env, slice(None), settings)
    assert curriculum._step == 1
    assert not curriculum.sampler.pending_counts.any()
    assert curriculum.sampler.update_count == 0


def test_snapshot_commit_requires_complete_rounds_and_restores_policy_identity(make_environment, tmp_path):
    """All tasks share a fresh baseline before any signed progress can affect assignment."""
    import json

    settings = LearningProgressCfg(
        measurement_mode="snapshot", command_bins=(1, 1, 1), modes=("moving",), min_samples=2
    )
    _, curriculum = make_environment(settings=settings)
    returns = torch.arange(curriculum.catalog.num_tasks, dtype=torch.float64).unsqueeze(1).repeat(1, 2)
    curriculum.commit_snapshot(returns, 0, "baseline-policy")
    assert curriculum.sampler.update_count == 1
    assert curriculum.sampler.initialized.all()
    assert not curriculum.sampler.learning_progress.any()
    assert curriculum.sampler.window_counts.tolist() == [1] * curriculum.catalog.num_tasks
    baseline = curriculum.state_dict()
    bad_returns = returns.clone()
    bad_returns[0, 0] = float("nan")
    for invalid in (returns[:, :1], bad_returns):
        with pytest.raises(ValueError, match="complete finite"):
            curriculum.commit_snapshot(invalid, 500, "next-policy")
        assert curriculum.state_dict() == baseline
    curriculum.commit_snapshot(returns + 15.0, 500, "next-policy")
    torch.testing.assert_close(
        curriculum.sampler.learning_progress, torch.ones(curriculum.catalog.num_tasks, dtype=torch.float64)
    )
    assert curriculum.last_snapshot_policy_updates == 500
    assert curriculum.last_snapshot_policy_digest == "next-policy"
    curriculum.training_policy_updates = 750
    _, restored = make_environment(settings=settings)
    restored.load_state_dict(curriculum.state_dict())
    assert restored.last_snapshot_policy_updates == 500
    assert restored.last_snapshot_policy_digest == "next-policy"
    assert restored.training_policy_updates == 750
    assert restored.state_dict() == curriculum.state_dict()
    restored._env.cfg.sim.render_interval = 5
    restored._env.cfg.sim.log_dir = "/another/log/directory"
    restored._env.cfg.scene.robot.debug_vis = True
    restored.load_state_dict(curriculum.state_dict())
    invalid_counter = {**curriculum.state_dict(), "training_policy_updates": 499}
    with pytest.raises(ValueError, match="completed training-update count"):
        restored.load_state_dict(invalid_counter)
    legacy_counter = curriculum.state_dict()
    legacy_counter.pop("training_policy_updates")
    restored.load_state_dict(legacy_counter)
    assert restored.training_policy_updates is None
    changed_env, changed = make_environment(settings=settings)
    changed_env.cfg.observations["policy"]["enable_corruption"] = False
    with pytest.raises(ValueError, match="measurement conditions differ"):
        changed.load_state_dict(curriculum.state_dict())
    changed_env.cfg.observations["policy"]["enable_corruption"] = True
    changed.command.cfg.resampling_time_range = (0.04, 0.04)
    with pytest.raises(ValueError, match="measurement conditions differ"):
        changed.load_state_dict(curriculum.state_dict())
    _, changed_cohort = make_environment(num_envs=3, settings=settings)
    with pytest.raises(ValueError, match="measurement conditions differ"):
        changed_cohort.load_state_dict(curriculum.state_dict())
    physical_changes = (
        ("action scale", lambda env, term: env.cfg.actions["joint_pos"].update(scale=0.5)),
        ("actuator stiffness", lambda env, term: env.cfg.scene.robot.actuators["legs"].update(stiffness=30.0)),
        ("physics timestep", lambda env, term: setattr(env.cfg.sim, "dt", 0.0025)),
        ("control decimation", lambda env, term: setattr(env.cfg, "decimation", 8)),
        ("global seed", lambda env, term: setattr(env.cfg, "seed", 99)),
        ("generated terrain seed", lambda env, term: setattr(term.terrain, "generation_seed", 99)),
        ("terrain friction", lambda env, term: term.terrain.cfg.physics_material.update(friction=0.4)),
    )
    for _name, mutate in physical_changes:
        changed_env, changed = make_environment(settings=settings)
        mutate(changed_env, changed)
        with pytest.raises(ValueError, match="measurement conditions differ"):
            changed.load_state_dict(curriculum.state_dict())
    _, changed_terrain_seed = make_environment(settings=settings)
    changed_terrain_seed.terrain.cfg.terrain_generator.seed = 99
    with pytest.raises(ValueError, match="terrain configuration differs"):
        changed_terrain_seed.load_state_dict(curriculum.state_dict())
    old_measurement_contract = curriculum.state_dict()
    old_measurement_contract["measurement_contract"].pop("version")
    with pytest.raises(ValueError, match="measurement conditions differ"):
        restored.load_state_dict(old_measurement_contract)
    legacy_contract = curriculum.state_dict()
    legacy_contract.pop("measurement_contract")
    restored.load_state_dict(legacy_contract)
    assert restored.training_policy_updates == 750
    assert restored.last_snapshot_policy_updates == -1
    assert restored.sampler.update_count == 0
    invalid = curriculum.state_dict()
    invalid["sampler"]["pending_counts"][0] = 1
    with pytest.raises(ValueError, match="incomplete task measurement"):
        restored.load_state_dict(invalid)
    online_settings = LearningProgressCfg(
        measurement_mode="online", command_bins=(1, 1, 1), modes=("moving",), min_samples=2
    )
    _, online = make_environment(settings=online_settings)
    restored.load_state_dict(online.state_dict())
    assert restored.sampler.update_count == 0
    assert restored.last_snapshot_policy_updates == -1

    curriculum.diagnostics = SimpleNamespace(path=tmp_path / "task_windows.jsonl")
    curriculum.training_policy_updates = 1000
    curriculum.commit_snapshot(returns + 30.0, 1000, "third-policy")
    record = json.loads((tmp_path / "snapshot_rounds.jsonl").read_text())
    assert record["counts"] == [2] * curriculum.catalog.num_tasks
    assert record["return_mean"] == (returns[:, 0] + 30).tolist()
    assert record["scaled_return_mean"] == ((returns[:, 0] + 30) / 15).tolist()
    assert record["return_standard_error"] == [0.0] * curriculum.catalog.num_tasks
    assert record["policy_updates"] == 1000
    assert record["previous_policy_updates"] == 500
    assert record["policy_digest"] == "third-policy"
    _, single_sample = make_environment(
        settings=LearningProgressCfg(
            measurement_mode="snapshot", command_bins=(1, 1, 1), modes=("moving",), min_samples=1
        )
    )
    single_sample.diagnostics = SimpleNamespace(path=tmp_path / "task_windows.jsonl")
    single_sample.commit_snapshot(returns[:, :1], 0, "single-sample-policy")
    single_record = json.loads((tmp_path / "snapshot_rounds.jsonl").read_text().splitlines()[-1])
    assert single_record["return_sample_std"] == [None] * single_sample.catalog.num_tasks
    assert single_record["return_standard_error"] == [None] * single_sample.catalog.num_tasks


def test_snapshot_mode_rejects_training_without_its_runner_controller(make_environment):
    """An unsupported backend must not silently leave snapshot task probabilities uniform."""
    env, curriculum = make_environment(settings=LearningProgressCfg(measurement_mode="snapshot"))
    with pytest.raises(RuntimeError, match="snapshot curriculum controller"):
        curriculum.update(0.02)
    assert curriculum._step == 0


def test_online_windows_close_after_ppo_and_preserve_cross_boundary_episode(make_environment, tmp_path):
    """Closing LP statistics neither truncates ongoing PPO episodes nor drops failed returns."""
    import json

    from isaaclab.utils.curriculum.diagnostics import TaskWindowDiagnostics

    settings = LearningProgressCfg(
        measurement_mode="online_window", command_bins=(1, 1, 1), modes=("moving",), min_samples=1
    )
    env, curriculum = make_environment(num_envs=1, settings=settings)
    with pytest.raises(RuntimeError, match="online curriculum controller"):
        curriculum.update(0.02)
    curriculum.online_controller_active = True
    curriculum.begin_online_window(0)
    curriculum.diagnostics = TaskWindowDiagnostics(curriculum.catalog.num_tasks, "cpu", tmp_path / "task_windows.jsonl")
    env.reward_manager = SimpleNamespace(get_episode_sums=lambda ids: {})
    curriculum(env, slice(None), settings)
    first_task = curriculum.task_ids.item()
    env.reward_buf[:] = 15.0
    env.reset_buf[:] = True
    env.termination_manager.terminated[:] = True
    with torch.inference_mode():
        curriculum.update(0.02)
        curriculum(env, slice(None), settings)
    assert curriculum.sampler.pending_counts[first_task] == 1
    env.reset_buf[:] = False
    env.reward_buf[:] = 2.0
    for _ in range(3):
        curriculum.update(0.02)
    active_task = curriculum.task_ids.clone()
    for update in range(1, 100):
        curriculum.complete_online_update(update)
    assert curriculum.sampler.update_count == 0
    curriculum.complete_online_update(100)
    record = json.loads((tmp_path / "online_windows.jsonl").read_text())
    assert record["policy_updates"] == 100 and record["start_policy_updates"] == 0
    assert record["counts"][first_task] == 1
    assert not any(record["progress_valid"])
    episode_record = json.loads((tmp_path / "task_windows.jsonl").read_text())
    assert episode_record["counts"][first_task] == 1
    assert episode_record["statistics"]["return"]["mean"][first_task] == 15.0
    assert curriculum.sampler.update_count == 1
    assert curriculum.sampler.last_updated[first_task]
    assert curriculum._returns.item() == 6.0 and curriculum.tracking.steps.item() == 3
    torch.testing.assert_close(curriculum.task_ids, active_task)
    env.reward_buf[:] = 4.0
    env.reset_buf[:] = True
    curriculum.update(0.02)
    curriculum(env, slice(None), settings)
    assert curriculum.sampler.pending_counts[active_task].item() == 1
    assert curriculum.sampler.pending_sums[active_task].item() == pytest.approx(10.0 / 15.0)
    assert "lp_progress_coverage_fraction" in curriculum._sampling_logs
    with pytest.raises(ValueError, match="consecutive"):
        curriculum.complete_online_update(102)
    for update in range(101, 201):
        curriculum.complete_online_update(update)
    windows = [json.loads(line) for line in (tmp_path / "task_windows.jsonl").read_text().splitlines()]
    assert len(windows) == 2
    assert sum(windows[1]["counts"]) == 1
    assert windows[1]["statistics"]["return"]["mean"][active_task.item()] == 10.0


def test_online_resume_discards_partial_window_and_rebuilds_adjacent_baseline(make_environment):
    """A resumed partial window cannot compare its next full window with a pre-gap baseline."""
    settings = LearningProgressCfg(
        measurement_mode="online_window", command_bins=(1, 1, 1), modes=("moving",), min_samples=2
    )
    _, curriculum = make_environment(settings=settings)
    curriculum.online_controller_active = True
    curriculum.begin_online_window(0)
    curriculum.sampler.observe([0, 0], [15.0, 15.0])
    for update in range(1, 101):
        curriculum.complete_online_update(update)
    boundary_state = curriculum.state_dict()
    _, exact = make_environment(settings=settings)
    exact.load_state_dict(boundary_state)
    exact.online_controller_active = True
    exact.begin_online_window(100)
    assert exact.sampler.last_updated[0]
    exact.sampler.observe([0, 0], [30.0, 30.0])
    for update in range(101, 201):
        exact.complete_online_update(update)
    assert exact.sampler.progress_valid[0] and exact.sampler.learning_progress[0] == 1.0
    for update in range(101, 138):
        curriculum.complete_online_update(update)
    curriculum.sampler.observe([0], [9999.0])
    _, resumed = make_environment(settings=settings)
    resumed.load_state_dict(curriculum.state_dict())
    resumed.online_controller_active = True
    resumed.begin_online_window(137)
    assert not resumed.sampler.pending_counts.any() and not resumed.sampler.last_updated.any()
    resumed.sampler.observe([0, 0], [30.0, 30.0])
    for update in range(138, 237):
        resumed.complete_online_update(update)
    assert resumed.sampler.update_count == 0
    resumed.complete_online_update(237)
    assert resumed.sampler.update_count == 1 and not resumed.sampler.progress_valid.any()
    resumed.sampler.observe([0, 0], [45.0, 45.0])
    for update in range(238, 338):
        resumed.complete_online_update(update)
    assert resumed.sampler.progress_valid[0] and resumed.sampler.learning_progress[0] == 1.0
