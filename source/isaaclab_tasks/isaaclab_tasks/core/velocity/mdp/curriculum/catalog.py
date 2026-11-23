# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Construct the same velocity task catalog for training and fixed evaluation."""

import numpy as np

from isaaclab.utils.curriculum import build_velocity_task_catalog


def terrain_columns_by_name(generator) -> dict[str, list[int]]:
    """Return the generated columns for each terrain type in a curriculum layout."""
    weights = np.array([terrain.proportion for terrain in generator.sub_terrains.values()])
    cumulative = np.cumsum(weights / weights.sum())
    column_types = np.searchsorted(cumulative, np.arange(generator.num_cols) / generator.num_cols + 0.001, side="right")
    return {
        name: np.flatnonzero(column_types == index).tolist()
        for index, name in enumerate(generator.sub_terrains)
        if np.any(column_types == index)
    }


def terrain_velocity_catalog(generator, ranges, settings, device="cpu"):
    """Group generated columns by type without weighting tasks by duplicate mesh count."""
    columns = terrain_columns_by_name(generator)
    missing = [name for name, cfg in generator.sub_terrains.items() if cfg.proportion > 0 and name not in columns]
    if missing:
        raise ValueError(f"Increase terrain num_cols so every requested type has a mesh column: {missing}")
    return build_velocity_task_catalog(
        columns,
        generator.num_rows,
        {name: getattr(ranges, name) for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
        command_bins=settings.command_bins,
        modes=settings.modes,
        device=device,
    )
