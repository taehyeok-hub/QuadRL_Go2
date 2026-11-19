# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Signed episodic learning progress and softmax task sampling (LP-ACRL).

The paper's equations 5--7 use mean episodic returns, their signed change,
and softmax(LP / temperature). Window size, update cadence, constant reward
scaling, missing-window handling and uniform exploration are engineering
choices here. In particular, returns are never divided by episode duration.
"""

from __future__ import annotations

import math
from numbers import Real

import torch


class LearningProgressCurriculum:
    """Sample tasks from recent, signed improvement in completed episodes.

    Call ``observe`` once per batch of completed episodes and ``step`` once per
    global control step, independent of the number of parallel environments.
    By default, each task keeps an accumulating window until at least
    ``min_episodes`` are available at an update boundary. Fixed windows instead
    discard every batch at the boundary and compare only adjacent reliable
    windows. An unmeasured fixed-window mean is zero with ``last_updated=False``;
    ``progress_valid`` distinguishes a measured zero change from missing progress.

    Means and progress are in units of ``return / reward_scale``. Probabilities
    are ``(1-exploration) * softmax(progress/temperature) + exploration/N``.
    With the other extensions disabled, zero exploration recovers the paper's softmax equation.
    Optional contiguous groups preserve their size/N probability mass; EMA
    smooths valid progress estimates, and a KL budget softens the final mixed
    distribution by increasing its temperature. Defaults retain the supplied sampler
    version-1 checkpoint format and sampling behavior.
    Exact RNG replay requires the same torch device backend (CPU or CUDA), but
    neither task statistics nor the checkpoint depend on environment count.
    """

    _FLOAT_FIELDS = ("pending_sums", "mean_returns", "learning_progress", "_probabilities")
    _COUNT_FIELDS = ("pending_counts", "episode_counts", "window_counts")
    _BOOL_FIELDS = ("initialized", "last_updated")

    def __init__(
        self,
        num_tasks,
        device,
        *,
        temperature=0.1,
        exploration=0.1,
        min_episodes=8,
        update_interval=2500,
        reward_scale=15.0,
        seed=42,
        group_sizes=None,
        progress_ema_alpha=1.0,
        max_uniform_kl=None,
        window_mode="accumulating",
    ):
        self._integer(num_tasks, "num_tasks", minimum=1)
        self._integer(min_episodes, "min_episodes", minimum=1)
        self._integer(update_interval, "update_interval", minimum=1)
        self._integer(seed, "seed", maximum=2**63 - 1)
        if window_mode not in ("accumulating", "fixed"):
            raise ValueError("window_mode must be accumulating or fixed.")
        for name, value in (("temperature", temperature), ("reward_scale", reward_scale)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if (
            isinstance(exploration, bool)
            or not isinstance(exploration, Real)
            or not math.isfinite(exploration)
            or not 0 <= exploration <= 1
        ):
            raise ValueError("exploration must be finite and in [0, 1].")
        if group_sizes is not None:
            if (
                not isinstance(group_sizes, (tuple, list))
                or not group_sizes
                or any(type(size) is not int or size < 1 for size in group_sizes)
                or sum(group_sizes) != num_tasks
            ):
                raise ValueError("group_sizes must contain positive integers summing to num_tasks.")
            group_sizes = tuple(group_sizes)
        if (
            isinstance(progress_ema_alpha, bool)
            or not isinstance(progress_ema_alpha, Real)
            or not math.isfinite(progress_ema_alpha)
            or not 0 < progress_ema_alpha <= 1
        ):
            raise ValueError("progress_ema_alpha must be finite and in (0, 1].")
        if max_uniform_kl is not None and (
            isinstance(max_uniform_kl, bool)
            or not isinstance(max_uniform_kl, Real)
            or not math.isfinite(max_uniform_kl)
            or max_uniform_kl < 0
        ):
            raise ValueError("max_uniform_kl must be finite and nonnegative, or None.")
        self.num_tasks, self.device, self.seed = num_tasks, torch.device(device), seed
        self.options = dict(
            temperature=float(temperature),
            exploration=float(exploration),
            min_episodes=min_episodes,
            update_interval=update_interval,
            reward_scale=float(reward_scale),
        )
        # Omit default extension keys so existing version-1 checkpoints remain
        # byte-for-byte equivalent as Python state dictionaries.
        if group_sizes is not None:
            self.options["group_sizes"] = group_sizes
        if progress_ema_alpha != 1:
            self.options["progress_ema_alpha"] = float(progress_ema_alpha)
        if max_uniform_kl is not None:
            self.options["max_uniform_kl"] = float(max_uniform_kl)
        if window_mode != "accumulating":
            self.options["window_mode"] = window_mode
        self._fixed_windows = window_mode == "fixed"
        self._float_fields = self._FLOAT_FIELDS + (("progress_ema",) if progress_ema_alpha != 1 else ())
        self._count_fields = self._COUNT_FIELDS + (
            ("last_window_counts", "missing_streak") if self._fixed_windows else ()
        )
        self._bool_fields = self._BOOL_FIELDS + (("progress_valid",) if self._fixed_windows else ())
        for name in self._float_fields:
            setattr(self, name, torch.zeros(num_tasks, dtype=torch.float64, device=self.device))
        for name in self._count_fields:
            setattr(self, name, torch.zeros(num_tasks, dtype=torch.long, device=self.device))
        for name in self._bool_fields:
            setattr(self, name, torch.zeros(num_tasks, dtype=torch.bool, device=self.device))
        probabilities, self._effective_log_temperature = self._distribution_details(self.learning_progress)
        self._probabilities.copy_(probabilities)
        self.control_steps = self.update_count = self.sample_count = 0
        self._generator = torch.Generator(device=self.device).manual_seed(seed)

    @staticmethod
    def _integer(value, name, *, minimum=0, maximum=None):
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"{name} must be an integer in the supported range.")

    def _distribution(self, progress):
        return self._distribution_details(progress)[0]

    def _distribution_details(self, progress):
        """Compute probabilities and their softmax temperature without mutating sampler state."""
        exploration = self.options["exploration"]

        def mixed(log_temperature=None):
            inverse_temperature = (
                math.exp(-log_temperature) if log_temperature is not None and log_temperature >= 0 else None
            )
            if inverse_temperature == 0:
                return torch.full_like(progress, 1 / self.num_tasks)
            blocks = []
            for scores in progress.split(self.options.get("group_sizes", (self.num_tasks,))):
                if log_temperature is None:
                    # Keep legacy arithmetic. Negative infinity after centering
                    # and division safely means exp(-inf)=0.
                    logits = (scores - scores.max()) / self.options["temperature"]
                elif log_temperature < 0:
                    logits = (scores - scores.max()) / math.exp(log_temperature)
                else:
                    # Scaling first avoids overflowing the centered difference
                    # when a KL-limited temperature exceeds float64's range.
                    scaled = scores * inverse_temperature
                    logits = scaled - scaled.max()
                blocks.append(torch.softmax(logits, dim=0) * (len(scores) / self.num_tasks))
            weights = torch.cat(blocks)
            return (1 - exploration) * weights + exploration / self.num_tasks

        budget = self.options.get("max_uniform_kl")
        if budget == 0:
            return torch.full_like(progress, 1 / self.num_tasks), math.inf
        probabilities = mixed()
        if budget is None or self._uniform_kl(probabilities) <= budget:
            return probabilities, math.log(self.options["temperature"])
        low = math.log(self.options["temperature"])
        span = 1.0
        high = low + span
        while self._uniform_kl(mixed(high)) > budget:
            span *= 2
            high = low + span
        # Keep the feasible endpoint, so rounding cannot return the violating
        # side of the hard budget. Search consumes no sampler RNG state.
        for _ in range(64):
            middle = (low + high) / 2
            candidate = mixed(middle)
            if self._uniform_kl(candidate) > budget:
                low = middle
            else:
                high = middle
        return mixed(high), high

    def _uniform_kl(self, probabilities):
        return float(torch.special.xlogy(probabilities, probabilities * self.num_tasks).sum().clamp_min(0))

    @property
    def probabilities(self):
        """A copy of current task probabilities on the configured device."""
        return self._probabilities.clone()

    @torch.no_grad()
    def observe(self, task_ids, returns):
        """Accumulate completed episodic returns; reject invalid batches atomically."""
        try:
            ids = torch.as_tensor(task_ids, device=self.device)
            values = torch.as_tensor(returns, device=self.device)
            if ids.ndim != 1 or values.shape != ids.shape:
                raise ValueError("task_ids and returns must be matching one-dimensional vectors.")
            if ids.numel() == 0:
                return
            if ids.dtype == torch.bool or ids.is_floating_point() or ids.is_complex():
                raise ValueError("task_ids must contain integer indices.")
            ids = ids.to(torch.long)
            if ((ids < 0) | (ids >= self.num_tasks)).any():
                raise ValueError("task_ids are outside the task catalog.")
            if values.dtype == torch.bool or values.is_complex():
                raise ValueError("returns must contain finite real numbers.")
            # Preserve double precision for Python sequences, rather than first
            # rounding them through torch's default float32 tensor conversion.
            values = torch.as_tensor(returns, dtype=torch.float64, device=self.device)
            values = values / self.options["reward_scale"]
            if not torch.isfinite(values).all():
                raise ValueError("returns and scaled returns must be finite.")
            counts = torch.bincount(ids, minlength=self.num_tasks)
            sums = torch.zeros_like(self.pending_sums).index_add_(0, ids, values)
            pending_sums = self.pending_sums + sums
            limit = torch.iinfo(torch.long).max
            if ((counts > limit - self.pending_counts) | (counts > limit - self.episode_counts)).any():
                raise ValueError("Episode counters would overflow.")
            if not torch.isfinite(pending_sums).all():
                raise ValueError("Pending episodic returns would overflow.")
        except (TypeError, RuntimeError, OverflowError) as exc:
            raise ValueError("Invalid completed-episode batch.") from exc
        self.pending_sums.copy_(pending_sums)
        self.pending_counts.add_(counts)
        self.episode_counts.add_(counts)

    @torch.no_grad()
    def step(self):
        """Advance one control step; return whether an update boundary was reached."""
        next_step = self.control_steps + 1
        if next_step % self.options["update_interval"]:
            self.control_steps = next_step
            return False
        reliable = self.pending_counts >= self.options["min_episodes"]
        active = reliable & (self.last_updated if self._fixed_windows else self.initialized)
        means = torch.zeros_like(self.mean_returns)
        means[reliable] = self.pending_sums[reliable] / self.pending_counts[reliable]
        progress = torch.zeros_like(self.learning_progress)
        progress[active] = means[active] - self.mean_returns[active]
        if not torch.isfinite(progress).all():
            raise ValueError("Learning-progress difference would overflow.")
        scores = progress
        if "progress_ema_alpha" in self.options:
            alpha = self.options["progress_ema_alpha"]
            progress_ema = self.progress_ema.clone()
            if self._fixed_windows:
                progress_ema[~active] = 0
            progress_ema[active] = alpha * progress[active] + (1 - alpha) * progress_ema[active]
            if not torch.isfinite(progress_ema).all():
                raise ValueError("Smoothed learning progress would overflow.")
            # Preserve memory across sparse windows, but never sample from a
            # stale estimate or a task with only its first baseline window.
            scores = torch.where(active, progress_ema, 0.0)
        probabilities, log_temperature = self._distribution_details(scores)
        self.learning_progress.copy_(progress)
        if "progress_ema_alpha" in self.options:
            self.progress_ema.copy_(progress_ema)
        self._probabilities.copy_(probabilities)
        self._effective_log_temperature = log_temperature
        if self._fixed_windows:
            self.mean_returns.copy_(means)
            self.progress_valid.copy_(active)
            self.last_window_counts.copy_(self.pending_counts)
            self.missing_streak.add_(1)
            self.missing_streak[reliable] = 0
        else:
            self.mean_returns[reliable] = means[reliable]
        self.initialized[reliable] = True
        self.last_updated.copy_(reliable)
        self.window_counts.add_(reliable)
        if self._fixed_windows:
            self.pending_sums.zero_()
            self.pending_counts.zero_()
        else:
            self.pending_sums[reliable] = 0
            self.pending_counts[reliable] = 0
        self.control_steps = next_step
        self.update_count += 1
        return True

    @torch.no_grad()
    def sample(self, count):
        """Draw with replacement using this sampler's private torch generator."""
        self._integer(count, "sample count")
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        result = torch.multinomial(self._probabilities, count, replacement=True, generator=self._generator)
        self.sample_count += count
        return result

    def metrics(self):
        """Scalar summaries suitable for logging; means retain the common reward scale."""
        p = self._probabilities
        entropy = float(-torch.special.xlogy(p, p).sum())
        initialized = int(self.initialized.sum())
        with_progress = int((self.progress_valid if self._fixed_windows else self.window_counts >= 2).sum())
        measured = self.last_updated if self._fixed_windows else self.initialized
        maximum_temperature = torch.finfo(torch.float64).max
        temperature_capped = self._effective_log_temperature > math.log(maximum_temperature)
        effective_temperature = maximum_temperature if temperature_capped else math.exp(self._effective_log_temperature)
        metrics = {
            "lp_control_steps": self.control_steps,
            "lp_updates": self.update_count,
            "lp_samples_drawn": self.sample_count,
            "lp_episodes_observed": int(self.episode_counts.sum()),
            "lp_pending_episodes": int(self.pending_counts.sum()),
            "lp_tasks_initialized": initialized,
            "lp_tasks_updated": int(self.last_updated.sum()),
            "lp_baseline_coverage_fraction": initialized / self.num_tasks,
            # Legacy mode counts historical two-window coverage; fixed mode
            # counts only currently valid adjacent-window comparisons.
            "lp_tasks_with_progress": with_progress,
            "lp_progress_coverage_fraction": with_progress / self.num_tasks,
            "lp_probability_min": float(p.min()),
            "lp_probability_max": float(p.max()),
            "lp_entropy": entropy,
            "lp_effective_tasks": math.exp(entropy),
            "lp_uniform_kl": self._uniform_kl(p),
            "lp_temperature": self.options["temperature"],
            # Keep scalar/JSON logging finite even when an exact uniform budget
            # needs infinite temperature; the flags preserve that distinction.
            "lp_effective_temperature": effective_temperature,
            "lp_temperature_capped": int(temperature_capped),
            "lp_temperature_is_uniform_limit": int(self._effective_log_temperature == math.inf),
            "lp_progress_mean": float(self.learning_progress.mean()),
            "lp_progress_min": float(self.learning_progress.min()),
            "lp_progress_max": float(self.learning_progress.max()),
            "lp_scaled_mean_return": float(self.mean_returns[measured].mean()) if measured.any() else 0.0,
        }
        if self._fixed_windows:
            metrics.update(
                lp_current_mean_coverage_fraction=int(self.last_updated.sum()) / self.num_tasks,
                lp_last_window_episodes=int(self.last_window_counts.sum()),
                lp_last_window_count_min=int(self.last_window_counts.min()),
                lp_last_window_count_mean=float(self.last_window_counts.double().mean()),
                lp_last_window_count_max=int(self.last_window_counts.max()),
                lp_missing_streak_mean=float(self.missing_streak.double().mean()),
                lp_missing_streak_max=int(self.missing_streak.max()),
            )
        return metrics

    def state_dict(self):
        """Return independent, serializable state with no per-environment fields."""
        return {
            "version": 1,
            "num_tasks": self.num_tasks,
            "options": dict(self.options),
            "seed": self.seed,
            "control_steps": self.control_steps,
            "update_count": self.update_count,
            "sample_count": self.sample_count,
            "rng_device_type": self.device.type,
            "rng_state": self._generator.get_state().cpu().tolist(),
            **{
                name: getattr(self, name).cpu().tolist()
                for name in (*self._float_fields, *self._count_fields, *self._bool_fields)
            },
        }

    def _validate_fixed_window_state(self, prepared, rounds):
        """Check current-window reliability and freshness after common vector validation."""
        reliable = prepared["last_updated"]
        valid = prepared["progress_valid"]
        counts = prepared["last_window_counts"]
        streak = prepared["missing_streak"]
        if not torch.equal(reliable, counts >= self.options["min_episodes"]):
            raise ValueError("Saved fixed-window reliability does not match its sample counts.")
        if (prepared["mean_returns"][~reliable] != 0).any():
            raise ValueError("Unmeasured fixed windows must have zero saved means.")
        if (
            (valid & (~reliable | (prepared["window_counts"] < 2))).any()
            or (prepared["learning_progress"][~valid] != 0).any()
            or (rounds >= 2 and ((prepared["window_counts"] == rounds) & ~valid).any())
        ):
            raise ValueError("Saved fixed-window progress lacks adjacent reliable windows.")
        if (
            (streak > rounds).any()
            or (streak[reliable] != 0).any()
            or (rounds > 0 and (streak[~reliable] == 0).any())
            or (streak[~prepared["initialized"]] != rounds).any()
            or (streak > rounds - prepared["window_counts"]).any()
            or (rounds == 0 and counts.any())
        ):
            raise ValueError("Saved fixed-window missing streak is inconsistent.")
        for total, pending, latest, windows, current in zip(
            prepared["episode_counts"].tolist(),
            prepared["pending_counts"].tolist(),
            counts.tolist(),
            prepared["window_counts"].tolist(),
            reliable.tolist(),
        ):
            if total < pending + latest + (windows - int(current)) * self.options["min_episodes"]:
                raise ValueError("Saved episode counters cannot support the fixed windows.")

    def _validated_state(self, state):
        try:
            if not isinstance(state, dict) or set(state) != set(self.state_dict()):
                raise ValueError("Saved learning-progress state has missing or unknown fields.")
            if type(state["version"]) is not int or state["version"] != 1:
                raise ValueError("Unsupported learning-progress checkpoint version.")
            if type(state["num_tasks"]) is not int or state["num_tasks"] != self.num_tasks:
                raise ValueError("Saved learning-progress task catalog size differs.")
            options = state["options"]
            if (
                not isinstance(options, dict)
                or set(options) != set(self.options)
                or any(type(options[k]) is not type(v) or options[k] != v for k, v in self.options.items())
                or ("group_sizes" in options and any(type(size) is not int for size in options["group_sizes"]))
            ):
                raise ValueError("Saved learning-progress configuration differs.")
            for name in ("control_steps", "update_count", "sample_count", "seed"):
                self._integer(state[name], name, maximum=2**63 - 1 if name == "seed" else None)
            if state["update_count"] != state["control_steps"] // self.options["update_interval"]:
                raise ValueError("Saved curriculum update clock is inconsistent.")

            def vector(name, dtype):
                raw = torch.as_tensor(state[name], device=self.device)
                if raw.shape != (self.num_tasks,) or raw.is_complex():
                    raise ValueError(f"Invalid saved {name} shape or type.")
                if dtype == torch.bool and raw.dtype != torch.bool:
                    raise ValueError(f"Saved {name} must contain booleans.")
                if dtype == torch.long and (raw.dtype == torch.bool or raw.is_floating_point()):
                    raise ValueError(f"Saved {name} must contain integer counters.")
                if dtype == torch.float64 and raw.dtype == torch.bool:
                    raise ValueError(f"Saved {name} must contain real numbers.")
                result = torch.as_tensor(state[name], dtype=dtype, device=self.device).clone()
                if not torch.isfinite(result).all() or (dtype == torch.long and (result < 0).any()):
                    raise ValueError(f"Saved {name} has nonfinite or negative counter values.")
                return result

            prepared = {
                name: vector(name, dtype)
                for names, dtype in (
                    (self._float_fields, torch.float64),
                    (self._count_fields, torch.long),
                    (self._bool_fields, torch.bool),
                )
                for name in names
            }
            if not torch.equal(prepared["initialized"], prepared["window_counts"] > 0):
                raise ValueError("Saved baseline initialization does not match completed windows.")
            if (prepared["window_counts"] > state["update_count"]).any():
                raise ValueError("Saved completed-window count exceeds the update clock.")
            if (prepared["last_updated"] & ~prepared["initialized"]).any():
                raise ValueError("Saved updated task has no baseline.")
            if (prepared["mean_returns"][~prepared["initialized"]] != 0).any():
                raise ValueError("Uninitialized tasks must have zero saved baseline.")
            if (prepared["learning_progress"][~prepared["last_updated"] | (prepared["window_counts"] < 2)] != 0).any():
                raise ValueError("Saved progress is stale or lacks two reliable windows.")
            if (prepared["pending_sums"][prepared["pending_counts"] == 0] != 0).any():
                raise ValueError("Empty pending windows must have zero sum.")
            if self._fixed_windows:
                self._validate_fixed_window_state(prepared, state["update_count"])
            for total, pending, windows in zip(
                prepared["episode_counts"].tolist(),
                prepared["pending_counts"].tolist(),
                prepared["window_counts"].tolist(),
            ):
                if total < pending + windows * self.options["min_episodes"]:
                    raise ValueError("Saved episode counters cannot support the recorded windows.")
            p = prepared["_probabilities"]
            scores = prepared["learning_progress"]
            if "progress_ema_alpha" in self.options:
                if (prepared["progress_ema"][prepared["window_counts"] < 2] != 0).any():
                    raise ValueError("Saved smoothed progress lacks two reliable windows.")
                active = (
                    prepared["progress_valid"]
                    if self._fixed_windows
                    else prepared["last_updated"] & (prepared["window_counts"] >= 2)
                )
                if self._fixed_windows and (prepared["progress_ema"][~active] != 0).any():
                    raise ValueError("Saved fixed-window EMA crosses a missing measurement.")
                scores = torch.where(active, prepared["progress_ema"], 0.0)
            expected_probabilities, log_temperature = self._distribution_details(scores)
            if (p < 0).any() or (p > 1).any() or not torch.allclose(p, expected_probabilities, rtol=1e-12, atol=1e-15):
                raise ValueError("Saved probabilities do not match learning progress.")
            if state["rng_device_type"] != self.device.type:
                raise ValueError("Exact curriculum RNG restoration requires the same device backend.")
            raw_rng = torch.as_tensor(state["rng_state"], device="cpu")
            if (
                raw_rng.ndim != 1
                or raw_rng.dtype == torch.bool
                or raw_rng.is_floating_point()
                or raw_rng.is_complex()
                or (raw_rng < 0).any()
                or (raw_rng > 255).any()
            ):
                raise ValueError("Saved RNG state must be a byte vector.")
            generator = torch.Generator(device=self.device)
            generator.set_state(raw_rng.to(torch.uint8))
        except (KeyError, TypeError, RuntimeError, OverflowError) as exc:
            raise ValueError("Invalid saved learning-progress curriculum state.") from exc
        return prepared, generator, log_temperature

    def validate_state_dict(self, state):
        """Validate configuration, statistics and RNG without mutating live state."""
        self._validated_state(state)

    def load_state_dict(self, state):
        """Restore atomically after full validation; environment count is irrelevant."""
        prepared, generator, log_temperature = self._validated_state(state)
        for name, value in prepared.items():
            getattr(self, name).copy_(value)
        self._generator = generator
        self._effective_log_temperature = log_temperature
        for name in ("control_steps", "update_count", "sample_count", "seed"):
            setattr(self, name, state[name])
