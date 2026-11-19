# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode tracking metrics independent of the simulator, task, and logging backend."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class EpisodeVelocityTracking:
    """Accumulate clipped relative errors and a separate zero-command normalized extension.

    See the learning-progress curriculum guide for Eq. (8), masks, and error conventions.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        zero_command_scales: tuple[float, float, float] = (0.1, 0.1, 0.1),
        command_threshold: float = 1.0e-4,
    ):
        """Create buffers; axis scales have units m/s, m/s, and rad/s, respectively."""
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if len(zero_command_scales) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in zero_command_scales
        ):
            raise ValueError("zero_command_scales must contain three finite positive values.")
        if not math.isfinite(command_threshold) or command_threshold < 0.0:
            raise ValueError("command_threshold must be finite and nonnegative.")
        self.num_envs = num_envs
        self.command_threshold = command_threshold
        self.device = torch.device(device)
        self.zero_command_scales = torch.tensor(zero_command_scales, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._relative_error_sum = torch.zeros(num_envs, 3, device=self.device)
        self._normalized_error_sum = torch.zeros_like(self._relative_error_sum)
        self._zero_drift_sum = torch.zeros_like(self._relative_error_sum)
        self._zero_steps = torch.zeros(num_envs, 3, dtype=torch.long, device=self.device)
        self._all_finite = torch.ones(num_envs, 3, dtype=torch.bool, device=self.device)

    def update(self, command: torch.Tensor, actual: torch.Tensor) -> None:
        """Observe one post-physics sample against the command applied during that step."""
        if command.shape != (self.num_envs, 3) or actual.shape != command.shape:
            raise ValueError("command and actual must have shape (num_envs, 3).")
        # Keep relative tracking and zero-command drift as separate measurements.
        finite = torch.isfinite(command) & torch.isfinite(actual)
        error = torch.nan_to_num(torch.abs(actual - command), nan=math.inf, posinf=math.inf)
        magnitude = torch.nan_to_num(torch.abs(command), nan=0.0, posinf=0.0)
        nonzero = magnitude > self.command_threshold
        relative = (error / magnitude.clamp_min(torch.finfo(error.dtype).tiny)).clamp(max=1.0)
        normalized = (error / torch.maximum(magnitude, self.zero_command_scales)).clamp(max=1.0)
        self._relative_error_sum += torch.where(nonzero, relative, 0.0)
        self._normalized_error_sum += normalized
        self._zero_drift_sum += torch.where(~nonzero & finite, error, 0.0)
        self._zero_steps += (~nonzero).long()
        self._all_finite &= finite
        self.steps += 1

    def finalize(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
        horizon: int,
        failed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return values and validity masks; ``failed`` excludes manual or boundary resets."""
        if horizon <= 0:
            raise ValueError("horizon must be positive.")
        selected = slice(None) if env_ids is None else env_ids
        steps = self.steps[selected]
        if failed.shape != steps.shape or failed.dtype != torch.bool:
            raise ValueError("failed must be a Boolean tensor with one entry per selected environment.")
        # Missing steps after failure contribute maximum error in Eq. (8).
        missing = torch.where(failed, (horizon - steps).clamp_min(0), 0).unsqueeze(-1)
        complete = ((steps >= horizon) | failed) & (steps > 0)
        normalized_valid = complete.unsqueeze(-1) & self._all_finite[selected]
        relative_valid = normalized_valid & (self._zero_steps[selected] == 0)
        relative = ((self._relative_error_sum[selected] + missing) * (100.0 / horizon)).clamp(max=100.0)
        normalized = ((self._normalized_error_sum[selected] + missing) * (100.0 / horizon)).clamp(max=100.0)
        drift_valid = (self._zero_steps[selected] > 0) & self._all_finite[selected]
        drift = self._zero_drift_sum[selected] / self._zero_steps[selected].clamp_min(1)
        return {
            "epte_sp": torch.where(relative_valid, relative, 0.0),
            "epte_sp_valid": relative_valid,
            "normalized_epte_sp": torch.where(normalized_valid, normalized, 0.0),
            "normalized_epte_sp_valid": normalized_valid,
            "zero_command_drift": torch.where(drift_valid, drift, 0.0),
            "zero_command_drift_valid": drift_valid,
            "alive_fraction": (steps.float() / horizon).clamp(max=1.0),
        }

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear completed episodes while retaining all other environments."""
        selected = slice(None) if env_ids is None else env_ids
        self.steps[selected] = 0
        self._relative_error_sum[selected] = 0.0
        self._normalized_error_sum[selected] = 0.0
        self._zero_drift_sum[selected] = 0.0
        self._zero_steps[selected] = 0
        self._all_finite[selected] = True
