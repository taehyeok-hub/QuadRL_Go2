# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measured joint-limit metrics for Go2 training."""

import contextlib

import torch
import torch.distributed as dist

TOLERANCE_RAD = 1e-6


class JointLimitMetrics(contextlib.AbstractContextManager):
    def __init__(self, task, runner):
        self.runner = runner
        self.suspended = False
        env = runner.env.unwrapped
        from isaaclab_tasks.core.velocity.mdp.curriculum.catalog import terrain_columns_by_name

        self.prefix = f"JointLimits/{task}"
        self.robot = env.scene["robot"]
        self.terrain = env.scene.terrain
        self.groups = ["all"]
        generator = self.terrain.cfg.terrain_generator if self.terrain is not None else None
        self.column_groups = None
        if generator is not None and generator.curriculum:
            self.levels = generator.num_rows
            self.column_groups = torch.empty(generator.num_cols, dtype=torch.long, device=env.device)
            for name, columns in terrain_columns_by_name(generator).items():
                self.column_groups[columns] = len(self.groups)
                self.groups.extend(f"{name}/level_{level}" for level in range(self.levels))

        self.samples = torch.zeros(len(self.groups), dtype=torch.long, device=env.device)
        shape = (2, len(self.groups), len(self.robot.joint_names))
        self.counts = torch.zeros(shape, dtype=torch.long, device=env.device)
        self.peaks = torch.zeros(shape, device=env.device)

        self.manager = env.curriculum_manager

    def __enter__(self):
        self.update = self.manager.update
        self.log_iteration = self.runner.logger.log

        def capture(dt):
            if not self.suspended:
                self.capture()
            return self.update(dt)

        def log(it, *args, **kwargs):
            self.log_iteration(it, *args, **kwargs)
            self.log(self.runner.logger.writer, it)

        self.manager.update = capture
        self.runner.logger.log = log
        return self

    @contextlib.contextmanager
    def suspend(self):
        previous = self.suspended
        self.suspended = True
        try:
            yield
        finally:
            self.suspended = previous

    def __exit__(self, *exc):
        self.manager.update = self.update
        self.runner.logger.log = self.log_iteration

    def capture(self):
        positions = self.robot.data.joint_pos.torch
        hard = self.robot.data.joint_pos_limits.torch
        soft = self.robot.data.soft_joint_pos_limits.torch
        limits = torch.stack((hard, soft))
        lower, upper = limits.unbind(dim=-1)
        excess = torch.maximum(lower - positions, positions - upper).clamp_min_(0)
        valid = torch.isfinite(excess)
        exceeded = valid & (excess > TOLERANCE_RAD)
        excess = torch.where(valid, excess, 0)

        self.samples[0] += positions.shape[0]
        self.counts[:, 0] += exceeded.sum(dim=1)
        self.peaks[:, 0].copy_(torch.maximum(self.peaks[:, 0], excess.amax(dim=1)))
        if self.column_groups is not None:
            groups = self.column_groups[self.terrain.terrain_types] + self.terrain.terrain_levels
            self.samples.scatter_add_(0, groups, torch.ones_like(groups))
            indices = groups.view(1, -1, 1).expand_as(excess)
            self.counts.scatter_add_(1, indices, exceeded.long())
            self.peaks.scatter_reduce_(1, indices, excess, reduce="amax")

    def log(self, writer, iteration):
        if dist.is_initialized():
            dist.all_reduce(self.samples)
            dist.all_reduce(self.counts)
            dist.all_reduce(self.peaks, op=dist.ReduceOp.MAX)

        if writer is not None:
            samples = self.samples.tolist()
            counts = self.counts.tolist()
            peaks = self.peaks.tolist()
            for group, sample_count in enumerate(samples):
                if not sample_count:
                    continue
                prefix = f"{self.prefix}/{self.groups[group]}"
                writer.add_scalar(f"{prefix}/env_steps", sample_count, iteration)
                for kind, label in enumerate(("hard", "soft")):
                    joint_counts = counts[kind][group]
                    joint_peaks = peaks[kind][group]
                    writer.add_scalar(f"{prefix}/{label}_exceedances", sum(joint_counts), iteration)
                    writer.add_scalar(f"{prefix}/{label}_peak_excess_rad", max(joint_peaks), iteration)
                    for joint, count, peak in zip(self.robot.joint_names, joint_counts, joint_peaks, strict=True):
                        if group == 0 or count:
                            writer.add_scalar(f"{prefix}/{label}/{joint}/exceedances", count, iteration)
                            writer.add_scalar(f"{prefix}/{label}/{joint}/peak_excess_rad", peak, iteration)
        self.samples.zero_()
        self.counts.zero_()
        self.peaks.zero_()
