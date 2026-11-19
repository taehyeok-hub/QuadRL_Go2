# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytic contracts for episodic tracking and censored resets."""

import pytest
import torch

from isaaclab.utils.curriculum.tracking import EpisodeVelocityTracking

pytestmark = pytest.mark.unit


def test_failure_penalty_and_non_failure_censoring():
    """A quarter-episode fall adds 75 points; external resets cannot look perfect."""
    tracking = EpisodeVelocityTracking(3, "cpu")
    command = torch.tensor([[1.0, -2.0, 0.5]]).repeat(3, 1)
    for _ in range(2):
        tracking.update(command, command * 0.8)
    result = tracking.finalize(None, horizon=8, failed=torch.tensor([True, False, True]))
    torch.testing.assert_close(result["epte_sp"][[0, 2]], torch.full((2, 3), 80.0))
    assert result["epte_sp_valid"][[0, 2]].all()
    assert not result["epte_sp_valid"][1].any()
    torch.testing.assert_close(result["alive_fraction"], torch.full((3,), 0.25))
    tracking.reset([0, 2])
    for _ in range(6):
        tracking.update(command, command * 0.8)
    complete = tracking.finalize([1], horizon=8, failed=torch.tensor([False]))
    torch.testing.assert_close(complete["epte_sp"], torch.full((1, 3), 20.0))
    assert complete["epte_sp_valid"].all()


def test_zero_and_mixed_commands_are_reported_separately():
    """Zero commands invalidate relative percentages without dropping drift or extension metrics."""
    tracking = EpisodeVelocityTracking(1, "cpu", zero_command_scales=(0.2, 0.1, 0.5))
    tracking.update(torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[0.1, 0.5, 0.1]]))
    tracking.update(torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[0.1, 0.0, 0.5]]))
    result = tracking.finalize(None, horizon=2, failed=torch.tensor([False]))
    assert not result["epte_sp_valid"].any()
    assert result["normalized_epte_sp_valid"].all()
    torch.testing.assert_close(result["normalized_epte_sp"], torch.tensor([[50.0, 25.0, 35.0]]))
    torch.testing.assert_close(result["zero_command_drift"], torch.tensor([[0.1, 0.0, 0.1]]))
    assert result["zero_command_drift_valid"].all()


def test_unobserved_and_nonfinite_episodes_do_not_emit_valid_scores():
    """Initial resets and broken velocity samples cannot silently count as successful tracking."""
    tracking = EpisodeVelocityTracking(1, "cpu")
    empty = tracking.finalize(None, horizon=4, failed=torch.tensor([True]))
    assert not empty["normalized_epte_sp_valid"].any()
    tracking.update(torch.ones(1, 3), torch.tensor([[float("nan"), float("inf"), 1.0]]))
    result = tracking.finalize(None, horizon=4, failed=torch.tensor([True]))
    assert result["epte_sp_valid"].tolist() == [[False, False, True]]
    torch.testing.assert_close(result["epte_sp"], torch.tensor([[0.0, 0.0, 75.0]]))
    assert all(torch.isfinite(value).all() for value in result.values())
