# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity commands constrained by externally assigned curriculum tasks."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand


class LearningProgressVelocityCommand(UniformVelocityCommand):
    """Keep a mode and velocity interval fixed until the curriculum assigns another task."""

    def __init__(self, cfg, env):
        if cfg.heading_command or any((cfg.rel_standing_envs, cfg.rel_rotating_envs, cfg.rel_transition_envs)):
            raise ValueError("LP commands own all mode assignments; heading and rel_* quotas must be disabled.")
        super().__init__(cfg, env)
        self.task_bounds = torch.zeros(self.num_envs, 3, 2, device=self.device)
        self.task_modes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._phase = torch.zeros_like(self.task_modes)
        self._yaw_variant = torch.zeros_like(self.task_modes)
        self._patterns = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, -1, 0], [1, 0, 1], [0, 1, 1]],
            device=self.device,
        )
        lower, upper = cfg.transition_resampling_time_range
        if not 0 < lower <= upper:
            raise ValueError("Transition resampling intervals must satisfy 0 < lower <= upper.")

    def assign_tasks(self, env_ids, catalog, task_ids):
        """Assign task constraints without sampling a command before the scene reset."""
        self.task_bounds[env_ids] = catalog.bounds[task_ids]
        self.task_modes[env_ids] = catalog.modes[task_ids]
        self.transition_type[env_ids] = catalog.transition_types[task_ids]
        self.is_standing_env[env_ids] = self.task_modes[env_ids] == 0
        self.is_rotating_env[env_ids] = self.task_modes[env_ids] == 1
        self.is_transition_env[env_ids] = self.task_modes[env_ids] == 3
        self._phase[env_ids] = torch.randint(2, (len(task_ids),), device=self.device)
        self._yaw_variant[env_ids] = 1 - 2 * torch.randint(2, (len(task_ids),), device=self.device)

    def _resample_command(self, env_ids):
        bounds = self.task_bounds[env_ids]
        self.vel_command_b[env_ids] = bounds[:, :, 0] + torch.rand_like(bounds[:, :, 0]) * (
            bounds[:, :, 1] - bounds[:, :, 0]
        )
        # Transition patterns reverse direction inside the task's configured ranges.
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        ids = ids[self.is_transition_env[ids]]
        if len(ids):
            signs = self._patterns[self.transition_type[ids]] * (1 - 2 * self._phase[ids, None])
            signs[:, 2] *= self._yaw_variant[ids]
            limits = self.task_bounds[ids]
            self.vel_command_b[ids] = torch.where(
                signs > 0, limits[:, :, 1], torch.where(signs < 0, limits[:, :, 0], 0.0)
            )
            self._phase[ids] = 1 - self._phase[ids]
            self.time_left[ids] = torch.empty(len(ids), device=self.device).uniform_(
                *self.cfg.transition_resampling_time_range
            )
        self._update_command()

    def _update_command(self):
        self.vel_command_b[self.is_standing_env] = 0.0
        self.vel_command_b[self.is_rotating_env, :2] = 0.0
