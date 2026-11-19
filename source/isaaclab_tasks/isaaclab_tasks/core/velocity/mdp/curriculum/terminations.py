# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep task statistics inside the terrain tile assigned by the curriculum."""

import torch


def assigned_terrain_boundary(env, asset_name: str = "robot") -> torch.Tensor:
    """Truncate at the assigned rectangle boundary, including diagonal exits."""
    half_size = torch.tensor(env.scene.terrain.cfg.terrain_generator.size, device=env.device) / 2
    offset = env.scene[asset_name].data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return (offset.abs() >= half_size).any(dim=1)
