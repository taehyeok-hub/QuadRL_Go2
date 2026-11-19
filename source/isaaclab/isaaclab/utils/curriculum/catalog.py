# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Discrete velocity-task construction from terrain groups and command ranges."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

COMMAND_MODES = ("standing", "rotating", "moving", "transition")


@dataclass
class VelocityTaskCatalog:
    """Tensor task descriptors and a serializable signature for checkpoint compatibility."""

    terrain_columns: torch.Tensor
    terrain_levels: torch.Tensor
    terrain_type_ids: torch.Tensor
    modes: torch.Tensor
    transition_types: torch.Tensor
    bounds: torch.Tensor
    terrain_names: tuple[str, ...]
    column_groups: tuple[tuple[int, ...], ...]
    signature: dict

    @property
    def num_tasks(self) -> int:
        """Number of distinct terrain-level-command tasks."""
        return self.modes.numel()


def build_velocity_task_catalog(
    terrain_columns: Mapping[str, Sequence[int]],
    num_levels: int,
    command_ranges: Mapping[str, tuple[float, float]],
    command_bins: tuple[int, int, int] = (3, 1, 3),
    modes: Sequence[str] = COMMAND_MODES,
    device: str | torch.device = "cpu",
) -> VelocityTaskCatalog:
    """Build valid mode-specific tasks without manually assigning terrain or mode probabilities.

    Terrain names group equivalent generated columns, avoiding duplicate task probability for
    terrain types with more columns. Fixed ranges form one bin regardless of the requested count.
    Transition tasks use six predefined pattern IDs and the configured full command ranges.
    """
    if num_levels < 1 or not terrain_columns:
        raise ValueError("At least one terrain group and one terrain level are required.")
    if len(command_bins) != 3 or any(not isinstance(n, int) or n < 1 for n in command_bins):
        raise ValueError("command_bins must contain three positive integers.")
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in COMMAND_MODES for mode in modes):
        raise ValueError(f"modes must be a nonempty, unique selection from {COMMAND_MODES}.")
    axes = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
    ranges = tuple(tuple(command_ranges[axis]) for axis in axes)
    for limits in ranges:
        if len(limits) != 2 or not all(math.isfinite(value) for value in limits) or limits[0] > limits[1]:
            raise ValueError("Command ranges must be finite ordered pairs.")
    column_groups = tuple(tuple(columns) for columns in terrain_columns.values())
    flattened_columns = [column for group in column_groups for column in group]
    if any(not group for group in column_groups) or any(
        not isinstance(column, int) or column < 0 for column in flattened_columns
    ):
        raise ValueError("Terrain groups must contain nonnegative integer column indices.")
    if len(set(flattened_columns)) != len(flattened_columns):
        raise ValueError("A generated terrain column can belong to only one terrain group.")

    bins = []
    for (lower, upper), count in zip(ranges, command_bins):
        count = 1 if lower == upper else count
        edges = [lower + (upper - lower) * i / count for i in range(count + 1)]
        bins.append(tuple(zip(edges[:-1], edges[1:])))
    zero = (0.0, 0.0)
    mode_tasks = []
    for mode in modes:
        mode_id = COMMAND_MODES.index(mode)
        if mode == "standing":
            mode_tasks.append((mode_id, -1, (zero, zero, zero)))
        elif mode == "rotating":
            mode_tasks.extend((mode_id, -1, (zero, zero, yaw)) for yaw in bins[2] if yaw != zero)
        elif mode == "moving":
            mode_tasks.extend(
                (mode_id, -1, bounds) for bounds in itertools.product(*bins) if bounds[0] != zero or bounds[1] != zero
            )
        else:
            pattern_axes = ((0,), (1,), (0, 1), (0, 1), (0, 2), (1, 2))
            mode_tasks.extend(
                (mode_id, pattern, ranges)
                for pattern, active_axes in enumerate(pattern_axes)
                if all(ranges[axis][0] < 0.0 < ranges[axis][1] for axis in active_axes)
            )
    if not mode_tasks:
        raise ValueError("The selected command modes and ranges produce no valid task.")

    # Each terrain type gets the same initial task set regardless of mesh column proportions.
    descriptors = [
        (columns[0], level, type_id, mode_id, transition, bounds)
        for type_id, columns in enumerate(column_groups)
        for level in range(num_levels)
        for mode_id, transition, bounds in mode_tasks
    ]
    terrain_names = tuple(terrain_columns)
    signature = {
        "terrain_names": terrain_names,
        "column_groups": column_groups,
        "num_levels": num_levels,
        "command_ranges": ranges,
        "command_bins": tuple(command_bins),
        "modes": modes,
        "transition_patterns": 6,
    }
    return VelocityTaskCatalog(
        terrain_columns=torch.tensor([entry[0] for entry in descriptors], dtype=torch.long, device=device),
        terrain_levels=torch.tensor([entry[1] for entry in descriptors], dtype=torch.long, device=device),
        terrain_type_ids=torch.tensor([entry[2] for entry in descriptors], dtype=torch.long, device=device),
        modes=torch.tensor([entry[3] for entry in descriptors], dtype=torch.long, device=device),
        transition_types=torch.tensor([entry[4] for entry in descriptors], dtype=torch.long, device=device),
        bounds=torch.tensor([entry[5] for entry in descriptors], dtype=torch.float32, device=device),
        terrain_names=terrain_names,
        column_groups=column_groups,
        signature=signature,
    )
