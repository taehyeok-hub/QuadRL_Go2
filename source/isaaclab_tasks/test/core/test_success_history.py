# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decision-window behavior independent of terrain and simulator state."""

import copy

import pytest
import torch

from isaaclab_tasks.core.velocity.mdp.curriculum.success_history import EpisodeSuccessHistory

pytestmark = pytest.mark.unit


def test_completed_windows_decide_independently_and_start_fresh():
    history = EpisodeSuccessHistory(3, "cpu")
    up, down = history.record([2, 0], [False, True])
    assert not up.any() and not down.any()
    history.record([1], [False])
    up, down = history.record([0, 2], [False, False])
    assert up.tolist() == [True, False]
    assert down.tolist() == [False, True]
    assert history.outcomes.tolist() == [[-1, -1], [0, -1], [-1, -1]]

    # A decided window cannot keep promoting or demoting on every following episode.
    up, down = history.record([0, 1, 2], [False, False, True])
    assert up.tolist() == [False, False, False]
    assert down.tolist() == [False, True, False]
    up, down = history.record([0, 2], [False, False])
    assert up.tolist() == [False, True]
    assert down.tolist() == [True, False]


def test_checkpoint_continues_partial_windows_and_rejects_invalid_state_atomically():
    history = EpisodeSuccessHistory(2, "cpu", history_length=3)
    history.record([0, 1], [True, False])
    history.record([0], [False])
    saved = history.state_dict()
    restored = EpisodeSuccessHistory(2, "cpu", history_length=3)
    restored.load_state_dict(saved)
    up, down = restored.record([0, 1], [False, False])
    assert up.tolist() == [True, False]
    assert not down.any()
    up, down = restored.record([1], [False])
    assert up.tolist() == [False] and down.tolist() == [True]

    partial = EpisodeSuccessHistory(2, "cpu", history_length=3)
    partial.load_state_dict(saved)
    before = partial.outcomes.clone()
    for invalid in ([[1, 0, 0], [-1, -1, -1]], [[-1, 0, -1], [-1, -1, -1]], [[2, -1, -1], [-1, -1, -1]]):
        corrupted = copy.deepcopy(saved)
        corrupted["outcomes"] = invalid
        with pytest.raises(ValueError):
            partial.load_state_dict(corrupted)
        torch.testing.assert_close(partial.outcomes, before)
    saved["outcomes"][0][0] = 1
    torch.testing.assert_close(partial.outcomes, before)


@pytest.mark.parametrize("ids, outcomes", [([0, 0], [True, False]), ([0, 1], [True])])
def test_invalid_selection_does_not_advance_history(ids, outcomes):
    history = EpisodeSuccessHistory(2, "cpu")
    history.record([0], [False])
    before = history.outcomes.clone()
    with pytest.raises(ValueError):
        history.record(ids, outcomes)
    torch.testing.assert_close(history.outcomes, before)
    up, down = history.record([], [])
    assert up.numel() == down.numel() == 0
    torch.testing.assert_close(history.outcomes, before)


def test_ten_trial_thresholds_preserve_hold_band_and_consume_complete_windows():
    history = EpisodeSuccessHistory(4, "cpu", 10, 0.8, 0.4)
    counts = [8, 7, 5, 4]
    for step in range(10):
        up, down = history.record([0, 1, 2, 3], [step < count for count in counts])
        if step < 9:
            assert not up.any() and not down.any()
    assert up.tolist() == [True, False, False, False]
    assert down.tolist() == [False, False, False, True]
    assert (history.outcomes == -1).all()
    up, down = history.record([0, 1, 2, 3], [True] * 4)
    assert not up.any() and not down.any()


def test_checkpoint_rejects_changed_success_thresholds():
    history = EpisodeSuccessHistory(1, "cpu", 10, 0.8, 0.4)
    history.record([0], [True])
    with pytest.raises(ValueError):
        EpisodeSuccessHistory(1, "cpu", 10, 0.9, 0.4).load_state_dict(history.state_dict())
