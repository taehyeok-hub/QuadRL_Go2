# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic task execution independent of training curriculum priorities."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.curriculum.catalog import COMMAND_MODES
from isaaclab.utils.math import quat_from_euler_xyz

from ..curriculum.lp.commands import LearningProgressVelocityCommand
from ..curriculum.lp.config import LearningProgressCfg
from ..curriculum.lp.learning_progress import LearningProgressTerrainCurriculum
from .geometry import stair_exit_condition


class FixedEvaluationVelocityCommand(LearningProgressVelocityCommand):
    """Use fixed within-bin representatives and transition timing without random resampling."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sample_fraction = torch.full((self.num_envs, 3), 0.5, device=self.device)

    def _resample_command(self, env_ids):
        bounds = self.task_bounds[env_ids]
        self.vel_command_b[env_ids] = bounds[:, :, 0] + self.sample_fraction[env_ids] * (
            bounds[:, :, 1] - bounds[:, :, 0]
        )
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
            self.time_left[ids] = self.cfg.transition_resampling_time_range[0]
        self._update_command()


class FixedVelocityEvaluation(LearningProgressTerrainCurriculum):
    """Record terminal states and execute each fixed case exactly once; never update LP."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.plan = json.loads(Path(cfg.params["plan_path"]).read_text())
        if json.dumps(self.plan["catalog"], sort_keys=True) != json.dumps(self.catalog.signature, sort_keys=True):
            raise ValueError("Evaluation plan does not match the configured terrain and command catalog.")
        self.cases = self.plan["cases"]
        self.output_dir = Path(cfg.params["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.current = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.headings = torch.zeros(self.num_envs, device=self.device)
        self._is_traversal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._inner = torch.ones(self.num_envs, 2, device=self.device)
        self._cursor = 0
        self.rows = []
        self.trace_ids = set(self.plan.get("trace_case_ids", []))
        self._traces = {}
        self._feet, _ = self.command.robot.find_bodies(".*_foot")
        if len(self._feet) != 4:
            raise ValueError("Traversal evaluation requires exactly four foot bodies.")
        env.velocity_evaluation = self

    def update(self, dt):
        actual = torch.cat(
            (self.command.robot.data.root_lin_vel_b.torch[:, :2], self.command.robot.data.root_ang_vel_b.torch[:, 2:3]),
            dim=1,
        )
        self.tracking.update(self.command.command, actual)
        self._returns += self._env.reward_buf
        feet = self.command.robot.data.body_pos_w.torch[:, self._feet, :2] - self.terrain.env_origins[:, None, :2]
        offset = self.command.robot.data.root_pos_w.torch[:, :2] - self.terrain.env_origins[:, :2]
        valid = (
            stair_exit_condition(
                feet,
                offset,
                self._inner,
                self.headings,
                self.command.robot.data.projected_gravity_b.torch[:, 2],
                self._env.reset_terminated,
            )
            & self._is_traversal
        )
        self._hold_steps = torch.where(valid, self._hold_steps + 1, 0)
        self._goal |= self._hold_steps >= math.ceil(0.5 / dt)
        for env_id, case_index in enumerate(self.current.tolist() if self.trace_ids else []):
            if case_index >= 0 and self.cases[case_index]["case_id"] in self.trace_ids:
                self._traces.setdefault(case_index, []).append(
                    {
                        "step": int(self.tracking.steps[env_id]),
                        "position": self.command.robot.data.root_pos_w.torch[env_id].tolist(),
                        "velocity": actual[env_id].tolist(),
                        "command": self.command.command[env_id].tolist(),
                        "gravity": self.command.robot.data.projected_gravity_b.torch[env_id].tolist(),
                        "joints": self.command.robot.data.joint_pos.torch[env_id].tolist(),
                        "feet": feet[env_id].tolist(),
                    }
                )

    def __call__(
        self, env, env_ids, settings: LearningProgressCfg, command_name="base_velocity", plan_path="", output_dir=""
    ):
        ids = self._all_ids[env_ids]
        ended = ids[(self.current[ids] >= 0) & env.reset_buf[ids] & (self.tracking.steps[ids] > 0)]
        if len(ended):
            self._complete(ended)
        self.tracking.reset(ids)
        self._returns[ids] = 0
        self._goal[ids] = False
        self._hold_steps[ids] = 0
        for env_id in ids.tolist():
            if self._cursor >= len(self.cases):
                self.current[env_id] = -1
                self.command.task_bounds[env_id] = 0
                self.command.is_transition_env[env_id] = False
                self.command.is_standing_env[env_id] = True
                continue
            index = self._cursor
            self._cursor += 1
            case = self.cases[index]
            task = case["task_id"]
            slot = torch.tensor([env_id], device=self.device)
            task_id = torch.tensor([task], device=self.device)
            self.current[env_id] = index
            self.task_ids[env_id] = task
            level = int(self.catalog.terrain_levels[task])
            type_id = int(self.catalog.terrain_type_ids[task])
            group = self.catalog.column_groups[type_id]
            column = case["column"]
            if column not in group:
                raise ValueError("Evaluation case column does not belong to its terrain type.")
            self.terrain.terrain_levels[env_id] = level
            self.terrain.terrain_types[env_id] = column
            self.terrain.env_origins[env_id] = self.terrain.terrain_origins[level, column]
            self.command.assign_tasks(slot, self.catalog, task_id)
            self.command.sample_fraction[env_id] = torch.tensor(case["fractions"], device=self.device)
            self.command._phase[env_id] = case["phase"]
            self.command._yaw_variant[env_id] = case["yaw_variant"]
            self.headings[env_id] = case["heading"]
            self._is_traversal[env_id] = case["protocol"] == "traversal"
            if self._is_traversal[env_id]:
                velocity = torch.tensor(case["command"], device=self.device)
                self.command.task_bounds[env_id] = velocity[:, None].expand(3, 2)
                self.command.is_transition_env[env_id] = False
                self.command.is_standing_env[env_id] = False
                self.command.is_rotating_env[env_id] = False
            name = self.catalog.terrain_names[type_id]
            border = getattr(self.terrain.cfg.terrain_generator.sub_terrains[name], "border_width", 0.0)
            self._inner[env_id] = torch.tensor(self.terrain.cfg.terrain_generator.size, device=self.device) / 2 - border
        return {}

    def _complete(self, ids):
        env = self._env
        failed = env.reset_terminated[ids]
        metrics = self.tracking.finalize(ids, env.max_episode_length, failed)
        rewards = env.reward_manager.get_episode_sums(ids)
        reasons = {name: env.termination_manager.get_term(name)[ids] for name in env.termination_manager.active_terms}
        for local, env_id in enumerate(ids.tolist()):
            index = int(self.current[env_id])
            case = self.cases[index]
            task = case["task_id"]
            row = dict(
                case,
                terrain=self.catalog.terrain_names[int(self.catalog.terrain_type_ids[task])],
                level=int(self.catalog.terrain_levels[task]),
                mode=COMMAND_MODES[int(self.catalog.modes[task])],
                failed=bool(failed[local]),
                steps=int(self.tracking.steps[env_id]),
                boundary=bool(reasons["assigned_terrain_boundary"][local]),
                goal_reached=bool(self._goal[env_id]),
                success=bool(self._goal[env_id] & ~failed[local]),
                episode_return=float(self._returns[env_id]),
                rewards={key: float(value[local]) for key, value in rewards.items()},
                termination=[key for key, value in reasons.items() if value[local]],
            )
            for key in ("epte_sp", "normalized_epte_sp", "zero_command_drift"):
                row[key] = [
                    float(metrics[key][local, axis]) if metrics[key + "_valid"][local, axis] else None
                    for axis in range(3)
                ]
            self.rows.append(row)
            with (self.output_dir / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if index in self._traces:
                with (self.output_dir / f"trace_{case['case_id']:05d}.json").open("w") as stream:
                    json.dump(self._traces.pop(index), stream, allow_nan=False)

    def state_dict(self):
        return {}


def fixed_evaluation_reset(env, env_ids):
    """Use default joints, zero velocity, and the case heading without reset randomization."""
    controller = env.velocity_evaluation
    robot = controller.command.robot
    pose = robot.data.default_root_pose.torch[env_ids].clone()
    pose[:, :3] += env.scene.env_origins[env_ids]
    heading = controller.headings[env_ids]
    pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(heading), torch.zeros_like(heading), heading)
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(len(heading), 6, device=env.device), env_ids=env_ids
    )
    robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch[env_ids], env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(robot.data.default_joint_vel.torch[env_ids]), env_ids=env_ids
    )


@configclass
class FixedEvaluationEventsCfg:
    reset_robot = EventTermCfg(func=fixed_evaluation_reset, mode="reset")
