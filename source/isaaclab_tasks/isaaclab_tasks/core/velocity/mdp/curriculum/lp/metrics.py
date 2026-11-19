# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Logger-neutral aggregation for velocity curriculum diagnostics."""

import torch

from isaaclab.utils.curriculum.catalog import COMMAND_MODES


class VelocityCurriculumMetrics:
    """Aggregate episode vectors without hiding which episodes are eligible for EPTE-SP."""

    def __init__(self, catalog, debug_metrics: bool = False):
        self.catalog = catalog
        self.debug_metrics = debug_metrics
        if debug_metrics:
            self._sums = torch.zeros(catalog.num_tasks, 3, device=catalog.bounds.device)
            self._counts = torch.zeros_like(self._sums)

    def episodes(self, values, tasks, failed, truncated):
        logs = {
            "failure_rate": failed.float(),
            "boundary_truncation_rate": truncated.float(),
        }
        if self.debug_metrics:
            logs["alive_fraction"] = values["alive_fraction"]
        for axis, name in enumerate(("vx", "vy", "yaw")):
            for key in ("epte_sp", "normalized_epte_sp", "zero_command_drift"):
                if key == "zero_command_drift" and not self.debug_metrics:
                    continue
                mask = values[f"{key}_valid"][:, axis]
                if self.debug_metrics or key == "epte_sp":
                    logs[f"{key}_coverage/{name}"] = mask.float()
                if mask.any():
                    logs[f"{key}/{name}"] = values[key][mask, axis].clone()
            if not self.debug_metrics:
                continue
            valid = values["normalized_epte_sp_valid"][:, axis]
            self._sums[:, axis].scatter_add_(0, tasks[valid], values["normalized_epte_sp"][valid, axis])
            self._counts[:, axis].scatter_add_(0, tasks[valid], torch.ones_like(tasks[valid], dtype=torch.float))
            seen = self._counts[:, axis] > 0
            if seen.any():
                logs[f"historical_observed_task_macro_normalized_epte_sp/{name}"] = (
                    self._sums[seen, axis] / self._counts[seen, axis]
                ).mean()
        if not self.debug_metrics:
            logs["normalized_epte_sp_coverage"] = values["normalized_epte_sp_valid"].all(dim=1).float()
            return logs
        for type_id, terrain in enumerate(self.catalog.terrain_names):
            for mode_id, mode in enumerate(COMMAND_MODES):
                mask = (self.catalog.terrain_type_ids[tasks] == type_id) & (self.catalog.modes[tasks] == mode_id)
                if mask.any():
                    valid = values["normalized_epte_sp_valid"].all(dim=1) & mask
                    prefix = f"terrain/{terrain}/{mode}"
                    logs[f"{prefix}/epte_coverage"] = valid[mask].float()
                    logs[f"{prefix}/failure_rate"] = failed[mask].float()
                    if valid.any():
                        logs[f"{prefix}/normalized_epte_sp"] = values["normalized_epte_sp"][valid].mean(dim=1)
        return logs


def sampling_metrics(sampler, catalog, debug_metrics: bool = False):
    """Summarize marginal probabilities; episode counts follow the sampler synchronization scope."""
    logs = {"updates": float(sampler.update_count)}
    probabilities = sampler.probabilities
    entropy = -torch.special.xlogy(probabilities, probabilities).sum()
    logs.update(
        max_probability=probabilities.max().clone(),
        effective_tasks=entropy.exp(),
        uniform_kl=torch.special.xlogy(probabilities, probabilities * probabilities.numel()).sum().clamp_min(0),
    )
    if debug_metrics:
        logs["num_tasks"] = float(catalog.num_tasks)
        logs.update(sampler.metrics())
    for dimension, labels, indices in (
        ("terrain", catalog.terrain_names, catalog.terrain_type_ids),
        ("mode", COMMAND_MODES, catalog.modes),
        ("level", tuple(map(str, range(int(catalog.terrain_levels.max()) + 1))), catalog.terrain_levels),
    ):
        if not debug_metrics and dimension != "terrain":
            continue
        for index, name in enumerate(labels):
            mask = indices == index
            logs[f"{dimension}/{name}/probability"] = probabilities[mask].sum().clone()
            if debug_metrics and mask.any():
                logs[f"{dimension}/{name}/LP"] = sampler.learning_progress[mask].mean().clone()
                logs[f"{dimension}/{name}/episodes"] = sampler.episode_counts[mask].sum().float().clone()
    return logs
