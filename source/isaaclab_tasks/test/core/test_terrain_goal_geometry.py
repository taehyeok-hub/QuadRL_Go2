# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Border goal geometry independent of simulator and curriculum history."""

import pytest
import torch

from isaaclab_tasks.core.velocity.mdp.evaluation.geometry import terrain_border_condition

pytestmark = pytest.mark.unit


def test_border_goal_requires_shared_exit_inside_tile_and_upright_survival():
    """Crossing different faces, leaving the tile, falling, or leaving the torso behind cannot succeed."""
    feet = torch.tensor(
        [
            [[3.2, -0.2], [3.6, -0.2], [3.2, 0.2], [3.6, 0.2]],
            [[3.4, 2.8], [3.2, 2.8], [2.8, 3.4], [2.8, 3.2]],
            [[3.2, -0.2], [3.6, -0.2], [3.2, 0.2], [3.6, 0.2]],
            [[3.2, -0.2], [4.1, -0.2], [3.2, 0.2], [3.6, 0.2]],
            [[3.2, -0.2], [3.6, -0.2], [3.2, 0.2], [3.6, 0.2]],
            [[3.2, -0.2], [3.6, -0.2], [3.2, 0.2], [3.6, 0.2]],
            [[-3.2, -0.2], [-3.6, -0.2], [-3.2, 0.2], [-3.6, 0.2]],
            [[-0.2, 3.2], [-0.2, 3.6], [0.2, 3.2], [0.2, 3.6]],
            [[-0.2, -3.2], [-0.2, -3.6], [0.2, -3.2], [0.2, -3.6]],
        ]
    )
    root = torch.tensor(
        [
            [3.4, 0.0],
            [3.4, 3.4],
            [3.0, 0.0],
            [3.4, 0.0],
            [3.4, 0.0],
            [3.4, 0.0],
            [-3.4, 0.0],
            [0.0, 3.4],
            [0.0, -3.4],
        ]
    )
    gravity = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0, -0.5, -1.0, -1.0, -1.0])
    failed = torch.tensor([False, False, False, False, True, False, False, False, False])
    result = terrain_border_condition(feet, root, torch.full((9, 2), 3.0), torch.full((9, 2), 4.0), gravity, failed)
    assert result.tolist() == [True, False, False, False, False, False, True, True, True]
