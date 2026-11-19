# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment episode histories for success-based curriculum decisions."""

import math
from collections.abc import Sequence

import torch


class EpisodeSuccessHistory:
    """Consume independent success windows using configurable promotion/demotion rates.

    Windows are independent per environment. Ordinary episode resets do not clear
    history; each completed decision clears its own window, even at a level bound.
    """

    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        history_length: int = 2,
        promotion_success_rate: float | None = None,
        demotion_success_rate: float = 0.0,
    ):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer.")
        if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 1:
            raise ValueError("history_length must be a positive integer.")
        if promotion_success_rate is not None and not 0 <= demotion_success_rate < promotion_success_rate <= 1:
            raise ValueError("Success thresholds must satisfy 0 <= demotion < promotion <= 1.")
        self.promotion_success_rate = promotion_success_rate
        self.demotion_success_rate = demotion_success_rate
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.history_length = history_length
        self.outcomes = torch.full((num_envs, history_length), -1, dtype=torch.int8, device=self.device)

    def record(
        self, env_ids: torch.Tensor | Sequence[int], outcomes: torch.Tensor | Sequence[bool]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record one completed episode per selected environment and consume full windows."""
        ids = torch.as_tensor(env_ids, device=self.device)
        if ids.ndim != 1 or (ids.numel() and ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("env_ids must be a one-dimensional integer selection.")
        ids = ids.long()
        values = torch.as_tensor(outcomes, device=self.device)
        if values.shape != ids.shape or (values.numel() and values.dtype != torch.bool):
            raise ValueError("outcomes must be a boolean vector matching env_ids.")
        if ((ids < 0) | (ids >= self.num_envs)).any() or ids.unique().numel() != ids.numel():
            raise ValueError("env_ids must contain unique valid environment indices.")
        if not ids.numel():
            empty = torch.zeros(0, dtype=torch.bool, device=self.device)
            return empty, empty.clone()

        # Only selected episodes advance; every full window produces exactly one decision.
        window = self.outcomes[ids].roll(1, dims=1)
        window[:, 0] = values.to(torch.int8)
        complete = (window >= 0).all(dim=1)
        successes = (window == 1).sum(dim=1)
        required = (
            1 if self.promotion_success_rate is None else math.ceil(self.promotion_success_rate * self.history_length)
        )
        move_up = complete & (successes >= required)
        move_down = complete & (successes <= math.floor(self.demotion_success_rate * self.history_length)) & ~move_up
        window[complete] = -1
        self.outcomes[ids] = window
        return move_up, move_down

    def state_dict(self) -> dict:
        """Save unfinished decision windows without any simulator state."""
        return {
            "version": 2,
            "promotion_success_rate": self.promotion_success_rate,
            "demotion_success_rate": self.demotion_success_rate,
            "num_envs": self.num_envs,
            "history_length": self.history_length,
            "outcomes": self.outcomes.cpu().tolist(),
        }

    def load_state_dict(self, state: dict):
        """Restore only compatible unfinished windows; reject invalid states atomically."""
        if (
            state.get("version") != 2
            or state.get("promotion_success_rate") != self.promotion_success_rate
            or state.get("demotion_success_rate") != self.demotion_success_rate
            or state.get("num_envs") != self.num_envs
            or state.get("history_length") != self.history_length
        ):
            raise ValueError("Success-history checkpoint dimensions or version differ.")
        values = torch.as_tensor(state["outcomes"], device=self.device)
        if (
            values.shape != self.outcomes.shape
            or values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or ((values < -1) | (values > 1)).any()
        ):
            raise ValueError("Success-history checkpoint must contain only -1, 0, and 1 in the expected shape.")
        valid = values >= 0
        if valid.all(dim=1).any() or ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("Success-history checkpoint contains consumed or chronologically invalid windows.")
        self.outcomes.copy_(values)
