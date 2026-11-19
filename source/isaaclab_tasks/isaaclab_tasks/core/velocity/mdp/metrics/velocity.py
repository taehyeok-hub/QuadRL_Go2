# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Time-weighted tracking errors against the command applied during each step."""

import torch


class VelocityMetrics:
    """Accumulate body-frame x/y velocity and yaw-rate errors without reward transforms."""

    def __init__(self, device: str):
        self._totals = torch.zeros(3, 3, dtype=torch.float64, device=device)

    def sample(self, actual: torch.Tensor, command: torch.Tensor, dt: float):
        """Record actual and commanded [vx, vy, yaw rate] in m/s, m/s, rad/s."""
        error = actual - command
        valid = torch.isfinite(error)
        error = torch.where(valid, error, 0).double()
        self._totals[0] += error.abs().sum(dim=0) * dt
        self._totals[1] += error.square().sum(dim=0) * dt
        self._totals[2] += valid.sum(dim=0) * dt

    def pop(self) -> dict[str, float]:
        """Return one window's errors; live episodes continue into the next window."""
        totals = self._totals.cpu().tolist()
        self._totals.zero_()
        metrics = {}
        for axis, absolute, squared, weight in zip(("x", "y", "yaw"), *totals):
            if weight > 0:
                metrics[f"Velocity/{axis}_mae"] = absolute / weight
                metrics[f"Velocity/{axis}_rmse"] = (squared / weight) ** 0.5
        return metrics
