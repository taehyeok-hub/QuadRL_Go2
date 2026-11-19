# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded per-task statistics written outside scalar dashboards."""

import json
from pathlib import Path

import torch


class TaskWindowDiagnostics:
    """Keep sufficient statistics for one window, not every episode or physics sample."""

    def __init__(self, num_tasks, device, path, metadata=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if metadata is not None:
            (self.path.parent / "task_catalog.json").write_text(json.dumps(metadata, indent=2, default=str))
        self.num_tasks = num_tasks
        self.device = device
        self._sums = {}
        self._squares = {}
        # These buffers persist across inference-mode rollouts and ordinary PPO updates.
        with torch.inference_mode(False):
            self._counts = torch.zeros(num_tasks, device=device, dtype=torch.float64)

    def record(self, task_ids, values):
        self._counts.index_add_(0, task_ids, torch.ones_like(task_ids, dtype=torch.float64))
        for name, value in values.items():
            value = value.detach().to(torch.float64)
            if name not in self._sums:
                with torch.inference_mode(False):
                    self._sums[name] = torch.zeros_like(self._counts)
                    self._squares[name] = torch.zeros_like(self._counts)
            self._sums[name].index_add_(0, task_ids, value)
            self._squares[name].index_add_(0, task_ids, value.square())

    def write(self, step, sampler):
        p = sampler.probabilities.detach().double()
        entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum()
        record = {
            "step": int(step),
            "updates": sampler.update_count,
            "scope": "rank_local_window",
            "entropy": float(entropy),
            "effective_tasks": float(entropy.exp()),
            "max_probability": float(p.max()),
            "top10_mass": float(p.topk(min(10, p.numel())).values.sum()),
            "probabilities": p.cpu().tolist(),
            "LP": sampler.learning_progress.cpu().tolist(),
            "baseline_returns": sampler.mean_returns.cpu().tolist(),
            "reward_scale": sampler.options["reward_scale"],
            "counts": self._counts.cpu().tolist(),
            "statistics": {},
        }
        n = self._counts.clamp_min(1)
        for name, sums in self._sums.items():
            variance = (self._squares[name] - sums.square() / n).clamp_min(0) / (n - 1).clamp_min(1)
            record["statistics"][name] = {
                "mean": (sums / n).cpu().tolist(),
                "sample_std": variance.sqrt().cpu().tolist(),
            }
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self._counts.zero_()
        for values in (self._sums, self._squares):
            for tensor in values.values():
                tensor.zero_()
