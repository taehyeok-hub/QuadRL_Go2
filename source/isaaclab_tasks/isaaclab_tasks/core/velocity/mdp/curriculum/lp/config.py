# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Settings shared by learning-progress velocity curricula."""

from isaaclab.utils import configclass


@configclass
class LearningProgressCfg:
    """Discretize configured ranges and adapt task probabilities without promotion rules."""

    command_bins: tuple[int, int, int] = (3, 1, 3)
    modes: tuple[str, ...] = ("standing", "rotating", "moving", "transition")
    temperature: float = 0.2
    exploration: float = 0.1
    reward_scale: float = 15.0
    min_samples: int = 8
    update_interval_steps: int = 2500
    measurement_mode: str = "online_window"
    update_interval_iterations: int = 100
    evaluation_interval_iterations: int = 100
    evaluation_seed: int = 101
    group_sizes: tuple[int, ...] | None = None
    progress_ema_alpha: float = 1.0
    # KL <= log(2) keeps the effective task count at least half of the catalog.
    max_uniform_kl: float | None = 0.6931471805599453
    zero_command_scales: tuple[float, float, float] = (0.1, 0.1, 0.1)
    command_threshold: float = 1.0e-4
    debug_metrics: bool = False
    sampling_strategy: str = "lp"
    return_mode: str = "episodic"
    diagnostics: bool = False
