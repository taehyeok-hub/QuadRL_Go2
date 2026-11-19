# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical units, censored swings, and missing terrain samples in locomotion telemetry."""

import math

import pytest
import torch

from isaaclab_tasks.core.velocity.mdp.metrics.gait import GaitMetrics
from isaaclab_tasks.core.velocity.mdp.metrics.observer import foot_ground_heights
from isaaclab_tasks.core.velocity.mdp.metrics.velocity import VelocityMetrics

pytestmark = pytest.mark.unit


def test_tracking_errors_do_not_cancel_and_use_observed_time_per_axis():
    metrics = VelocityMetrics("cpu")
    metrics.sample(torch.tensor([[1.0, -2.0, 0.5], [-1.0, 2.0, -0.5]]), torch.zeros(2, 3), 1.0)
    metrics.sample(torch.tensor([[3.0, 0.0, 1.5], [math.nan, 0.0, -1.5]]), torch.zeros(2, 3), 2.0)
    result = metrics.pop()
    assert result["Velocity/x_mae"] == pytest.approx(2.0)
    assert result["Velocity/x_rmse"] == pytest.approx(math.sqrt(5.0))
    assert result["Velocity/y_mae"] == pytest.approx(2 / 3)
    assert result["Velocity/yaw_mae"] == pytest.approx(7 / 6)
    assert result["Velocity/yaw_rmse"] == pytest.approx(math.sqrt(19 / 12))
    assert metrics.pop() == {}


def _sample(metrics, contact, heights, *, valid=None, duration=None, vz=0.0):
    contact = torch.tensor([contact])
    metrics.sample(
        torch.tensor([vz]),
        torch.tensor([[math.pi / 6, -math.pi / 4]]),
        contact,
        torch.tensor([heights]),
        torch.ones_like(contact) if valid is None else torch.tensor([valid]),
        torch.full(contact.shape, 0.25),
        torch.zeros(contact.shape) if duration is None else torch.tensor([duration]),
        0.1,
    )


def test_complete_swing_survives_logging_boundary_but_initial_airborne_segment_is_censored():
    metrics = GaitMetrics(1, ["left", "right"], "cpu", debug_metrics=True)
    _sample(metrics, [True, False], [0.0, 0.3], vz=1.0)
    _sample(metrics, [False, False], [0.1, 0.4], vz=-1.0)
    first = metrics.pop()
    assert first["Gait/base_vz_mean"] == pytest.approx(0)
    assert first["Gait/base_vz_rms"] == pytest.approx(1)
    assert first["Gait/roll_rms"] == pytest.approx(30)
    assert first["Gait/pitch_rms"] == pytest.approx(45)
    assert first["Gait/stance_ratio"] == pytest.approx(25)
    assert "Gait/swing_time_mean" not in first
    _sample(metrics, [False, False], [0.2, 0.5])
    _sample(metrics, [True, True], [0.0, 0.0], duration=[0.3, 8.0])
    second = metrics.pop()
    assert second["Gait/swing_peak_height_mean"] == pytest.approx(0.2)
    assert second["Gait/swing_time_mean"] == pytest.approx(0.3)
    assert second["Gait/debug/completed_swings"] == 1
    assert second["Gait/stance_foot_speed_mean"] == pytest.approx(0.25)


def test_reset_and_missing_ground_do_not_invent_successful_swing_measurements():
    metrics = GaitMetrics(1, ["foot"], "cpu", debug_metrics=True)
    _sample(metrics, [True], [0.0])
    _sample(metrics, [False], [0.8])
    metrics.reset([0])
    _sample(metrics, [True], [0.0], duration=[0.9])
    _sample(metrics, [False], [math.nan], valid=[False])
    _sample(metrics, [False], [0.1])
    _sample(metrics, [True], [0.0], duration=[0.3])
    result = metrics.pop()
    assert result["Gait/swing_time_mean"] == pytest.approx(0.3)
    assert "Gait/swing_peak_height_mean" not in result
    assert result["Gait/swing_height_mean"] == pytest.approx(0.45)
    assert result["Gait/debug/swing_height_coverage"] == pytest.approx(2 / 3)
    assert metrics.pop() == {"Gait/debug/completed_swings": 0, "Gait/debug/valid_swing_peaks": 0}


def test_ground_height_uses_nearby_valid_hits_without_rough_terrain_plane_fallback():
    feet = torch.tensor([[[0.0, 0.0, 1.2], [1.0, 0.0, 2.3], [4.0, 0.0, 5.0]]])
    hits = torch.tensor([[[0.02, 0.0, 1.0], [1.01, 0.0, 2.0], [4.0, 0.0, math.inf]]])
    height, valid = foot_ground_heights(feet, hits, 0.1)
    assert valid.tolist() == [[True, True, False]]
    torch.testing.assert_close(height[valid], torch.tensor([0.2, 0.3]))
    _, valid = foot_ground_heights(feet, torch.full_like(hits, math.inf), 0.1)
    assert not valid.any()
