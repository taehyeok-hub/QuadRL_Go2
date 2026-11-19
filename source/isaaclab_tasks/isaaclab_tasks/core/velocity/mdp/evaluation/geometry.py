# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Directional, four-foot stair-exit geometry used by fixed evaluation."""

import math

import torch


def stair_exit_condition(feet_xy, root_xy, inner_half_size, heading, gravity_z, failed):
    """Require every foot beyond the stair region, forward progress, and an upright body."""
    outside = (feet_xy.abs() > inner_half_size[:, None, :] + 0.05).any(dim=2).all(dim=1)
    direction = torch.stack((heading.cos(), heading.sin()), dim=1)
    distance = (inner_half_size / direction.abs().clamp_min(1.0e-6)).min(dim=1).values
    progress = (root_xy * direction).sum(dim=1) >= distance
    upright = gravity_z < -math.cos(math.pi / 4)
    return outside & progress & upright & ~failed


def terrain_border_condition(feet_xy, root_xy, inner_half_size, tile_half_size, gravity_z, failed):
    """Require torso and every foot in one outer goal strip, inside the assigned tile."""
    positions = torch.cat((root_xy[:, None, :], feet_xy), dim=1)
    beyond_positive = (positions > inner_half_size[:, None, :] + 0.05).all(dim=1)
    beyond_negative = (positions < -inner_half_size[:, None, :] - 0.05).all(dim=1)
    shared_exit = (beyond_positive | beyond_negative).any(dim=1)
    inside_tile = (positions.abs() <= tile_half_size[:, None, :]).all(dim=(1, 2))
    upright = gravity_z < -math.cos(math.pi / 4)
    return shared_exit & inside_tile & upright & ~failed
