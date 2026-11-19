# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""First-episode accounting for a fixed cohort of curriculum measurement cases."""

import torch


class SnapshotProbeBatch:
    """Measure one actual episode per valid slot and retire completed cases."""

    def __init__(self, task_ids: torch.Tensor, num_tasks: int):
        if task_ids.ndim != 1 or task_ids.dtype != torch.long:
            raise ValueError("Snapshot task IDs must be a one-dimensional long tensor.")
        if ((task_ids < -1) | (task_ids >= num_tasks)).any() or not (task_ids >= 0).any():
            raise ValueError("Snapshot task IDs must contain valid cases and optional -1 padding.")
        self.task_ids = task_ids.clone()
        self.valid = task_ids >= 0
        self.finished = ~self.valid
        self.returns = torch.zeros_like(task_ids, dtype=torch.float64)
        self.steps = 0

    @property
    def complete(self) -> bool:
        """Whether every requested case reached its first real terminal event."""
        return bool(self.finished.all())

    def update(self, rewards: torch.Tensor, reset_buf: torch.Tensor):
        """Capture first-terminal returns and request safe resets of all inactive slots."""
        if rewards.shape != self.returns.shape or reset_buf.shape != self.returns.shape:
            raise ValueError("Snapshot reward and terminal vectors must match the cohort.")
        active = ~self.finished
        if not torch.isfinite(rewards[active]).all():
            raise FloatingPointError("Non-finite reward in snapshot measurement.")
        self.returns[active] += rewards[active].double()
        if not torch.isfinite(self.returns).all():
            raise FloatingPointError("Snapshot episodic return overflowed.")
        self.finished |= active & reset_buf.bool()
        self.steps += 1
        # Use the simulator's normal reset lifecycle every step, including padding.
        # Completed bodies must not keep integrating after their terminal state.
        reset_buf.copy_(self.finished)
