# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical and task-space contracts for episodic learning-progress sampling."""

import copy
import math

import pytest
import torch

from isaaclab.utils.curriculum import COMMAND_MODES, LearningProgressCurriculum, build_velocity_task_catalog

pytestmark = pytest.mark.unit


def test_signed_progress_uses_consecutive_means_and_discards_stale_priority():
    """Improvement and regression change priority with their original signs."""
    sampler = LearningProgressCurriculum(
        3, "cpu", temperature=1.0, reward_scale=1.0, exploration=0.0, min_episodes=2, update_interval=1
    )
    sampler.observe([0, 0, 1, 1], [0.0, 2.0, 4.0, 4.0])
    sampler.step()
    torch.testing.assert_close(sampler.probabilities, torch.full((3,), 1 / 3, dtype=torch.float64))
    sampler.observe([0, 0, 1, 1], [2.0, 4.0, 2.0, 2.0])
    sampler.step()
    denominator = math.exp(2) + math.exp(-2) + 1
    expected = torch.tensor([math.exp(2) / denominator, math.exp(-2) / denominator, 1 / denominator])
    torch.testing.assert_close(sampler.probabilities.float(), expected)
    sampler.step()
    torch.testing.assert_close(sampler.probabilities, torch.full((3,), 1 / 3, dtype=torch.float64))


def test_default_reward_scale_and_exploration_have_distinct_effects():
    """A raw-return improvement of 1.5 yields log odds one before the 10% uniform mixture."""
    sampler = LearningProgressCurriculum(2, "cpu", min_episodes=1, update_interval=1)
    sampler.observe([0, 1], [0.0, 0.0])
    sampler.step()
    sampler.observe([0, 1], [1.5, 0.0])
    sampler.step()
    assert sampler.learning_progress.tolist() == pytest.approx([0.1, 0.0])
    assert sampler.probabilities[0].item() == pytest.approx(0.9 * math.e / (math.e + 1) + 0.05)
    assert sampler.mean_returns.tolist() == pytest.approx([0.1, 0.0])


def test_sparse_windows_accumulate_and_only_control_steps_advance_the_update_clock():
    """Sub-threshold observations survive update boundaries until a complete comparison exists."""
    sampler = LearningProgressCurriculum(2, "cpu", min_episodes=2, update_interval=2, reward_scale=1.0)
    sampler.observe([0], [1.0])
    assert not sampler.step()
    assert sampler.step()
    sampler.observe([0], [3.0])
    assert not sampler.step()
    assert sampler.step()
    assert sampler.mean_returns[0].item() == 2.0
    sampler.observe([0, 0], [3.0, 5.0])
    assert not sampler.step()
    assert sampler.step()
    assert sampler.learning_progress[0].item() == 2.0
    assert sampler.metrics()["lp_tasks_with_progress"] == 1
    assert sampler.metrics()["lp_progress_coverage_fraction"] == 0.5


def test_fixed_windows_discard_short_batches_and_require_adjacent_means():
    """A sparse gap cannot turn a historical improvement into current learning progress."""
    sampler = LearningProgressCurriculum(
        2, "cpu", min_episodes=2, update_interval=1, reward_scale=1.0, window_mode="fixed"
    )
    sampler.observe([0, 0, 1], [1.0, 3.0, 100.0])
    sampler.step()
    assert not sampler.pending_counts.any()
    assert sampler.mean_returns.tolist() == [2.0, 0.0]
    assert sampler.last_window_counts.tolist() == [2, 1]
    assert sampler.missing_streak.tolist() == [0, 1]
    assert not sampler.progress_valid.any()

    sampler.observe([0, 1, 1], [1000.0, 4.0, 6.0])
    sampler.step()
    assert not sampler.pending_counts.any()
    assert not sampler.pending_sums.any()
    assert sampler.mean_returns.tolist() == [0.0, 5.0]
    assert not sampler.progress_valid.any()
    assert not sampler.learning_progress.any()
    assert sampler.initialized.tolist() == [True, True]

    sampler.observe([0, 0, 1, 1], [9.0, 11.0, 3.0, 5.0])
    sampler.step()
    assert sampler.learning_progress.tolist() == [0.0, -1.0]
    assert sampler.progress_valid.tolist() == [False, True]
    assert sampler.missing_streak.tolist() == [0, 0]
    assert sampler.window_counts.tolist() == [2, 2]
    assert sampler.metrics()["lp_tasks_with_progress"] == 1

    sampler.observe([0, 0, 1, 1], [9.0, 11.0, 3.0, 5.0])
    sampler.step()
    assert sampler.progress_valid.tolist() == [True, True]
    assert not sampler.learning_progress.any()


@pytest.mark.parametrize("window_mode", ["accumulating", "fixed"])
def test_checkpoint_restores_pending_returns_and_exact_sampler_rng_atomically(window_mode):
    """Resuming preserves pending means and the next draw, while a failed restore leaves state intact."""
    options = dict(min_episodes=2, update_interval=1, reward_scale=1.0, window_mode=window_mode)
    sampler = LearningProgressCurriculum(2, "cpu", **options)
    sampler.observe([0, 0, 1, 1], [1.0, 1.0, 2.0, 2.0])
    sampler.step()
    sampler.sample(5)
    sampler.observe([0], [3.0])
    state = sampler.state_dict()
    restored = LearningProgressCurriculum(2, "cpu", **options)
    restored.load_state_dict(state)
    torch.manual_seed(100)
    expected_draw = sampler.sample(40)
    torch.manual_seed(200)
    torch.testing.assert_close(restored.sample(40), expected_draw)
    restored.observe([0], [5.0])
    restored.step()
    assert restored.learning_progress[0].item() == 3.0
    before = restored.state_dict()
    malformed = copy.deepcopy(before)
    malformed["_probabilities"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="probabilities"):
        restored.load_state_dict(malformed)
    assert restored.state_dict() == before
    if window_mode == "fixed":
        for field, task, value in (
            ("progress_valid", 1, True),
            ("last_window_counts", 0, 0),
            ("missing_streak", 1, 0),
            ("mean_returns", 1, 5.0),
            ("last_window_counts", 1, 1),
        ):
            malformed = copy.deepcopy(before)
            malformed[field][task] = value
            with pytest.raises(ValueError):
                restored.load_state_dict(malformed)
            assert restored.state_dict() == before
    with pytest.raises(ValueError, match="configuration|fields"):
        LearningProgressCurriculum(2, "cpu").load_state_dict(state)


def test_group_constraints_preserve_group_mass_despite_cross_group_progress():
    """Grouping fixes each group's share while LP still redistributes its internal tasks."""
    sampler = LearningProgressCurriculum(
        4, "cpu", min_episodes=1, update_interval=1, group_sizes=(2, 2), reward_scale=1.0, exploration=0.0
    )
    sampler.observe([0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0])
    sampler.step()
    sampler.observe([0, 1, 2, 3], [100.0, 0.0, 1.0, 0.0])
    sampler.step()
    assert sampler.probabilities[:2].sum().item() == pytest.approx(0.5)
    assert sampler.probabilities[2:].sum().item() == pytest.approx(0.5)
    assert sampler.probabilities[0] > sampler.probabilities[1]
    assert sampler.probabilities[2] > sampler.probabilities[3]


@pytest.mark.parametrize("window_mode", ["accumulating", "fixed"])
def test_ema_smooths_valid_progress_but_does_not_sample_stale_estimates(window_mode):
    """A zero new improvement retains smoothed history; no new window removes sampling priority."""
    sampler = LearningProgressCurriculum(
        2,
        "cpu",
        min_episodes=1,
        update_interval=1,
        reward_scale=1.0,
        exploration=0.0,
        temperature=1.0,
        progress_ema_alpha=0.5,
        window_mode=window_mode,
    )
    sampler.observe([0, 1], [0.0, 0.0])
    sampler.step()
    sampler.observe([0, 1], [4.0, 0.0])
    sampler.step()
    assert sampler.probabilities[0].item() == pytest.approx(math.exp(2) / (math.exp(2) + 1))
    sampler.observe([0, 1], [4.0, 0.0])
    sampler.step()
    assert sampler.learning_progress[0].item() == 0.0
    assert sampler.probabilities[0].item() == pytest.approx(math.e / (math.e + 1))
    sampler.step()
    torch.testing.assert_close(sampler.probabilities, torch.tensor([0.5, 0.5], dtype=torch.float64))
    if window_mode == "fixed":
        sampler.observe([0, 1], [104.0, 0.0])
        sampler.step()
        assert not sampler.progress_valid.any()
        assert not sampler.progress_ema.any()
        sampler.observe([0, 1], [106.0, 0.0])
        sampler.step()
        assert sampler.learning_progress.tolist() == [2.0, 0.0]
        assert sampler.progress_ema.tolist() == [1.0, 0.0]


@pytest.mark.parametrize("budget", [0.0, 0.05])
def test_kl_budget_limits_the_final_distribution(budget):
    """A concentrated LP signal cannot exceed the requested divergence from uniform sampling."""
    sampler = LearningProgressCurriculum(
        2, "cpu", min_episodes=1, update_interval=1, reward_scale=1.0, max_uniform_kl=budget
    )
    sampler.observe([0, 1], [0.0, 0.0])
    sampler.step()
    sampler.observe([0, 1], [1000.0, 0.0])
    sampler.step()
    probabilities = sampler.probabilities.tolist()
    divergence = sum(p * math.log(p * 2) for p in probabilities)
    assert divergence <= budget + 1.0e-12
    assert sampler.probabilities[0].item() == pytest.approx(0.5) if budget == 0 else sampler.probabilities[0] > 0.5
    metrics = sampler.metrics()
    assert math.isfinite(metrics["lp_effective_temperature"])
    assert metrics["lp_temperature_is_uniform_limit"] == int(budget == 0)
    if budget:
        temperature = metrics["lp_effective_temperature"]
        expected = 0.9 * torch.softmax(sampler.learning_progress / temperature, dim=0) + 0.05
        torch.testing.assert_close(sampler.probabilities, expected)


def test_temperature_logging_is_restored_and_validation_does_not_change_it():
    """KL temperature follows the live distribution, including after successful or rejected restore."""
    options = dict(min_episodes=1, update_interval=1, reward_scale=1.0, temperature=0.2, max_uniform_kl=0.05)
    sampler = LearningProgressCurriculum(2, "cpu", **options)
    assert sampler.metrics()["lp_effective_temperature"] == pytest.approx(0.2)
    sampler.observe([0, 1], [0.0, 0.0])
    sampler.step()
    sampler.observe([0, 1], [1000.0, 0.0])
    sampler.step()
    before = sampler.state_dict()
    temperature = sampler.metrics()["lp_effective_temperature"]
    assert temperature > 0.2
    restored = LearningProgressCurriculum(2, "cpu", **options)
    restored.load_state_dict(before)
    assert restored.metrics()["lp_effective_temperature"] == temperature
    malformed = copy.deepcopy(before)
    malformed["learning_progress"] = [1.0, 0.0]
    malformed["_probabilities"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="probabilities"):
        restored.validate_state_dict(malformed)
    assert restored.state_dict() == before
    assert restored.metrics()["lp_effective_temperature"] == temperature


def test_invalid_observation_batch_is_rejected_without_partial_accumulation():
    """Bad episode data cannot corrupt already collected statistics."""
    sampler = LearningProgressCurriculum(2, "cpu")
    sampler.observe([0], [2.0])
    before = sampler.state_dict()
    for ids, values in (([0, 2], [1.0, 1.0]), ([0], [float("nan")]), ([0.5], [1.0])):
        with pytest.raises(ValueError):
            sampler.observe(ids, values)
        assert sampler.state_dict() == before
    with pytest.raises(ValueError, match="temperature"):
        LearningProgressCurriculum(2, "cpu", temperature=0.0)


def test_catalog_uses_terrain_types_instead_of_column_proportions():
    """Mode-specific tasks do not multiply standing/rotation tasks by unused velocity axes."""
    catalog = build_velocity_task_catalog(
        {"descending": [0, 1, 2], "ascending": [3]},
        2,
        {"lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
    )
    assert catalog.num_tasks == 76
    assert torch.bincount(catalog.terrain_type_ids).tolist() == [38, 38]
    mode_counts = torch.bincount(catalog.modes).tolist()
    assert mode_counts == [4, 12, 36, 24]
    standing = catalog.modes == COMMAND_MODES.index("standing")
    assert torch.count_nonzero(catalog.bounds[standing]) == 0
    rotating = catalog.modes == COMMAND_MODES.index("rotating")
    assert torch.count_nonzero(catalog.bounds[rotating, :2]) == 0
    assert catalog.column_groups == ((0, 1, 2), (3,))


def test_catalog_filters_impossible_transition_axes_and_preserves_asymmetric_ranges():
    """Disabled axes cannot become reversal tasks; valid asymmetric limits are not symmetrized."""
    catalog = build_velocity_task_catalog(
        {"rough": [0]},
        1,
        {"lin_vel_x": (0.0, 0.0), "lin_vel_y": (-2.0, 0.3), "ang_vel_z": (0.0, 0.0)},
        command_bins=(3, 2, 3),
    )
    assert catalog.num_tasks == 4
    assert catalog.modes.tolist() == [0, 2, 2, 3]
    assert catalog.transition_types.tolist() == [-1, -1, -1, 1]
    assert catalog.bounds[-1, 1].tolist() == pytest.approx([-2.0, 0.3])
    assert catalog.bounds[1, 1].tolist() == pytest.approx([-2.0, -0.85])
    assert catalog.bounds[2, 1].tolist() == pytest.approx([-0.85, 0.3])
