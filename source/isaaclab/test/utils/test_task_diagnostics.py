# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytical statistics and bounded-window persistence for curriculum diagnostics."""

import json
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils.curriculum.diagnostics import TaskWindowDiagnostics

pytestmark = pytest.mark.unit


def test_window_variance_probability_concentration_and_reset(tmp_path):
    path = tmp_path / "windows.jsonl"
    # RSL-RL records during inference-mode rollouts and flushes after the PPO update.
    with torch.inference_mode():
        diagnostics = TaskWindowDiagnostics(2, "cpu", path)
        diagnostics.record(
            torch.tensor([0, 0, 1]),
            {"return": torch.tensor([1.0, 3.0, 8.0]), "LP_score": torch.tensor([1.0, 9.0, 8.0])},
        )
    sampler = SimpleNamespace(
        probabilities=torch.tensor([0.5, 0.5]),
        learning_progress=torch.tensor([1.0, -1.0]),
        mean_returns=torch.tensor([2.0, 8.0]),
        update_count=1,
        options={"reward_scale": 15.0},
    )
    diagnostics.write(1000, sampler)
    sampler.probabilities = torch.tensor([1.0, 0.0])
    diagnostics.write(2000, sampler)
    # Ordinary callers can continue recording and flushing the same persistent buffers.
    diagnostics.record(torch.tensor([1]), {"return": torch.tensor([6.0]), "LP_score": torch.tensor([2.0])})
    diagnostics.write(3000, sampler)
    first, second, third = map(json.loads, path.read_text().splitlines())
    assert first["counts"] == [2.0, 1.0]
    assert first["reward_scale"] == 15.0
    assert first["LP"] == [1.0, -1.0]
    assert first["statistics"]["return"]["mean"] == [2.0, 8.0]
    assert first["statistics"]["return"]["sample_std"][0] == pytest.approx(2**0.5)
    assert first["statistics"]["LP_score"]["mean"] == [5.0, 8.0]
    assert first["effective_tasks"] == pytest.approx(2.0)
    assert second["counts"] == [0.0, 0.0]
    assert second["effective_tasks"] == pytest.approx(1.0)
    assert third["counts"] == [0.0, 1.0]
    assert third["statistics"]["return"]["mean"] == [0.0, 6.0]
