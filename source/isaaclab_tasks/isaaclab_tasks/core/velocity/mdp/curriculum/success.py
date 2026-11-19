# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Connect per-environment success histories to terrain difficulty."""

from __future__ import annotations

import math

import numpy as np
import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.terrains import MeshInvertedPyramidStairsTerrainCfg, MeshPyramidStairsTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.dict import to_dict

from ..evaluation.geometry import terrain_border_condition
from .success_history import EpisodeSuccessHistory


@configclass
class SuccessTerrainCfg:
    """Training-episode success windows with separate promotion and demotion thresholds."""

    history_length: int = 2                         # default : 2
    promotion_success_rate: float | None = None     # None : 한 번이라도 성공하면 승급
    demotion_success_rate: float = 0.0              # 두 번 모두 실패하면 강등
    goal_region_width: float = 1.0
    min_command_speed: float = 0.2


class SuccessTerrainCurriculum(ManagerTermBase):
    """ Update terrain levels from completed training-episode success windows.

    Success requires torso and all four feet in the same outer goal strip, followed
    by an episode ending without failure. Stair strips use the mesh's flat border;
    other terrains use an explicit goal strip, which need not be flat. Ordinary
    standing/rotation episodes with no translational command do not vote.
    """

    reset_on_checkpoint_load = True

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.settings = cfg.params["settings"]
        self.robot = env.scene[cfg.params.get("asset_name", "robot")]
        self.command_name = cfg.params.get("command_name", "base_velocity")
        self.terrain = env.scene.terrain
        generator = self.terrain.cfg.terrain_generator
        if generator is None or not generator.curriculum:
            raise ValueError("Success curriculum requires a generated terrain with curriculum=True.")
        if not math.isfinite(self.settings.goal_region_width) or self.settings.goal_region_width <= 0:
            raise ValueError("goal_region_width must be positive and finite.")
        if not math.isfinite(self.settings.min_command_speed) or self.settings.min_command_speed <= 0:
            raise ValueError("min_command_speed must be positive and finite.")
        self._feet, _ = self.robot.find_bodies(cfg.params.get("foot_body_pattern", ".*_foot"))
        if len(self._feet) != 4:
            raise ValueError("Success terrain curriculum requires four resolved foot bodies.")
        self.history = EpisodeSuccessHistory(
            self.num_envs,
            self.device,
            self.settings.history_length,
            self.settings.promotion_success_rate,
            self.settings.demotion_success_rate,
        )
        self._all_ids = torch.arange(self.num_envs, device=self.device)
        self._goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._eligible = torch.zeros_like(self._goal)
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._half_size = torch.tensor(generator.size, device=self.device) / 2

        # Match generator column allocation; distinguish center-to-edge ascent from descent.
        configs = list(generator.sub_terrains.values())
        weights = np.array([sub.proportion for sub in configs])
        columns = np.searchsorted(
            np.cumsum(weights / weights.sum()), np.arange(generator.num_cols) / generator.num_cols + 0.001, side="right"
        )
        widths, ascent, descent = [], [], []
        for index in columns:
            sub = configs[index]
            is_stair = isinstance(sub, MeshPyramidStairsTerrainCfg)
            width = sub.border_width if is_stair else self.settings.goal_region_width
            if not math.isfinite(width) or width <= 0 or width >= min(generator.size) / 2:
                raise ValueError("Each goal strip must have positive width smaller than the tile half-size.")
            widths.append(width)
            ascent.append(isinstance(sub, MeshInvertedPyramidStairsTerrainCfg))
            descent.append(is_stair and not ascent[-1])
        self._inner_half_sizes = self._half_size[None, :] - torch.tensor(widths, device=self.device)[:, None]
        self._ascent_columns = torch.tensor(ascent, device=self.device)
        self._descent_columns = torch.tensor(descent, device=self.device)

    def update(self, dt: float):
        """Latch actual goal visits before physical reset, never infer unobserved success."""
        self._steps += 1
        command = self._env.command_manager.get_command(self.command_name)
        self._eligible |= command[:, :2].square().sum(dim=1) >= self.settings.min_command_speed**2
        origins = self.terrain.env_origins[:, :2]
        feet = self.robot.data.body_pos_w.torch[:, self._feet, :2] - origins[:, None, :]
        root = self.robot.data.root_pos_w.torch[:, :2] - origins
        columns = self.terrain.terrain_types
        self._goal |= (
            terrain_border_condition(
                feet,
                root,
                self._inner_half_sizes[columns],
                self._half_size.expand(self.num_envs, -1),
                self.robot.data.projected_gravity_b.torch[:, 2],
                self._env.termination_manager.terminated,
            )
            & self._eligible
        )

    def __call__(
        self,
        env,
        env_ids,
        settings: SuccessTerrainCfg,
        command_name="base_velocity",
        asset_name="robot",
        foot_body_pattern=".*_foot",
    ):
        ids = self._all_ids[env_ids]
        ended = ids[(self._steps[ids] > 0) & env.reset_buf[ids]]
        eligible = ended[self._eligible[ended]]
        logs = {"level": self.terrain.terrain_levels.float().mean()}
        if len(ended):
            logs["eligible_fraction"] = self._eligible[ended].float().mean()
        if len(eligible):
            success = self._goal[eligible] & ~env.termination_manager.terminated[eligible]
            columns = self.terrain.terrain_types[eligible].clone()
            up, down = self.history.record(eligible, success)
            self.terrain.update_env_origins(eligible, up, down)
            logs.update(
                success_rate=success.float().mean(), promotion_rate=up.float().mean(), demotion_rate=down.float().mean()
            )
            for label, mask in (("ascent", self._ascent_columns[columns]), ("descent", self._descent_columns[columns])):
                if mask.any():
                    logs[f"{label}_success_rate"] = success[mask].float().mean()
            logs["level"] = self.terrain.terrain_levels.float().mean()
        self.reset(ids)
        return logs

    def reset(self, env_ids=None):
        """Clear partial episode tracking while preserving the success history across resets."""
        ids = slice(None) if env_ids is None else env_ids
        self._goal[ids] = False
        self._eligible[ids] = False
        self._steps[ids] = 0

    def _contract(self):
        generator = self.terrain.cfg.terrain_generator
        return {
            "params": to_dict(self.cfg.params),
            "terrain": generator.to_dict(),
            "terminations": to_dict(self._env.cfg.terminations),
            "commands": to_dict(self._env.cfg.commands),
            "events": to_dict(self._env.cfg.events),
            "observations": to_dict(self._env.cfg.observations),
            "horizon_steps": self._env.max_episode_length,
            "step_dt": self._env.step_dt,
        }

    def state_dict(self):
        """Preserve histories with their exact terrain assignments; episodes restart on load."""
        return {
            "version": 1,
            "contract": self._contract(),
            "history": self.history.state_dict(),
            "levels": self.terrain.terrain_levels.cpu().tolist(),
            "columns": self.terrain.terrain_types.cpu().tolist(),
        }

    def load_state_dict(self, state):
        if state.get("version") != 1 or state.get("contract") != self._contract():
            raise ValueError("Success curriculum checkpoint configuration differs; use --reset_curriculum.")
        candidate = EpisodeSuccessHistory(
            self.num_envs,
            self.device,
            self.settings.history_length,
            self.settings.promotion_success_rate,
            self.settings.demotion_success_rate,
        )
        candidate.load_state_dict(state["history"])
        levels = torch.as_tensor(state["levels"], device=self.device)
        columns = torch.as_tensor(state["columns"], device=self.device)
        if (
            levels.shape != (self.num_envs,)
            or columns.shape != (self.num_envs,)
            or levels.dtype != torch.long
            or columns.dtype != torch.long
        ):
            raise ValueError("Checkpoint terrain assignments must be integer vectors matching num_envs.")
        if ((levels < 0) | (levels >= self.terrain.max_terrain_level)).any() or (
            (columns < 0) | (columns >= len(self._inner_half_sizes))
        ).any():
            raise ValueError("Checkpoint terrain assignments are outside the generated grid.")
        self.history = candidate
        self.terrain.terrain_levels.copy_(levels)
        self.terrain.terrain_types.copy_(columns)
        self.terrain.env_origins[:] = self.terrain.terrain_origins[levels, columns]
        self.reset()
