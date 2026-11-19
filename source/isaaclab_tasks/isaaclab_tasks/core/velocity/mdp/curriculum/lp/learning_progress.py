# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Connect learning-progress task sampling to manager-based velocity environments."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.curriculum import LearningProgressCurriculum
from isaaclab.utils.curriculum.diagnostics import TaskWindowDiagnostics
from isaaclab.utils.curriculum.tracking import EpisodeVelocityTracking
from isaaclab.utils.dict import to_dict

from ..catalog import terrain_velocity_catalog
from .commands import LearningProgressVelocityCommand
from .config import LearningProgressCfg
from .metrics import VelocityCurriculumMetrics, sampling_metrics
from .snapshot_probe import SnapshotProbeBatch

logger = logging.getLogger(__name__)


class LearningProgressTerrainCurriculum(ManagerTermBase):
    """Sample terrain, difficulty, and command mode from signed learning progress."""

    # Group LP diagnostics separately from ordinary terrain curriculum metrics.
    metric_namespace = "LP"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.settings = cfg.params["settings"]
        self.command = env.command_manager.get_term(cfg.params["command_name"])
        if not isinstance(self.command, LearningProgressVelocityCommand):
            raise TypeError("LP curriculum requires LearningProgressVelocityCommand.")
        self.terrain = env.scene.terrain
        generator = self.terrain.cfg.terrain_generator
        if generator is None or not generator.curriculum:
            raise ValueError("LP terrain sampling requires a generator with curriculum=True.")
        if self.settings.update_interval_steps < 1:
            raise ValueError("update_interval_steps must be positive.")
        if self.settings.measurement_mode not in ("snapshot", "online", "online_window"):
            raise ValueError("measurement_mode must be snapshot, online, or online_window.")
        self.snapshot_enabled = self.settings.measurement_mode == "snapshot"
        self.online_window_enabled = self.settings.measurement_mode == "online_window"
        if (
            (self.snapshot_enabled or self.online_window_enabled)
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            raise ValueError("PPO-boundary LP measurement currently supports single-device training only.")
        if type(self.settings.update_interval_iterations) is not int or self.settings.update_interval_iterations < 1:
            raise ValueError("update_interval_iterations must be a positive integer.")
        if self.settings.evaluation_interval_iterations < 1:
            raise ValueError("evaluation_interval_iterations must be positive.")
        self.snapshot_controller_active = False
        self.online_controller_active = False
        self._online_window_start_updates = 0
        self._snapshot_batch = None
        self.last_snapshot_policy_updates = -1
        self.last_snapshot_policy_digest = None
        self.training_policy_updates = None

        if self.settings.sampling_strategy not in ("lp", "uniform"):
            raise ValueError("sampling_strategy must be lp or uniform.")
        if self.settings.return_mode not in ("boundary_extrapolated", "episodic"):
            raise ValueError("return_mode must be episodic.")
        if self.settings.return_mode == "boundary_extrapolated":
            logger.warning("boundary_extrapolated is retired; using actual episodic returns without extrapolation.")
            self.settings.return_mode = "episodic"
        self.catalog = terrain_velocity_catalog(generator, self.command.cfg.ranges, self.settings, self.device)
        self._seed = getattr(env.cfg, "seed", None)
        if self._seed is None:
            self._seed = 42
        self.sampler = self._make_sampler()
        self._remember_shared_statistics()
        self.tracking = EpisodeVelocityTracking(
            self.num_envs, self.device, self.settings.zero_command_scales, self.settings.command_threshold
        )
        self.task_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._returns = torch.zeros(self.num_envs, device=self.device)
        self._step = 0
        self._all_ids = torch.arange(self.num_envs, device=self.device)
        self.metrics = VelocityCurriculumMetrics(self.catalog, self.settings.debug_metrics)
        self.diagnostics = None
        if self.settings.diagnostics:
            self.diagnostics = TaskWindowDiagnostics(
                self.catalog.num_tasks,
                self.device,
                Path(env.cfg.log_dir) / "diagnostics" / "task_windows.jsonl",
                metadata={"catalog": self.catalog.signature, "settings": self.settings.to_dict()},
            )
        self._sampling_logs = {}
        self._refresh_sampling_logs()

    def update(self, dt: float):
        """Measure applied commands before reset; synchronize LP only at a shared step cadence."""
        if self._snapshot_batch is not None:
            self._snapshot_batch.update(self._env.reward_buf, self._env.reset_buf)
            return
        if self.snapshot_enabled and not self.snapshot_controller_active:
            raise RuntimeError("Snapshot LP requires the RSL-RL snapshot curriculum controller.")
        if self.online_window_enabled and not self.online_controller_active:
            raise RuntimeError("Online-window LP requires the RSL-RL online curriculum controller.")
        self._returns += self._env.reward_buf
        actual = torch.cat(
            (self.command.robot.data.root_lin_vel_b.torch[:, :2], self.command.robot.data.root_ang_vel_b.torch[:, 2:3]),
            dim=1,
        )
        self.tracking.update(self.command.command, actual)
        if self.snapshot_enabled or self.online_window_enabled:
            self._step += 1
            if (
                self.snapshot_enabled
                and self.diagnostics is not None
                and self._step % self.settings.update_interval_steps == 0
            ):
                self.diagnostics.write(self._step, self.sampler)
            return
        boundary = (self.sampler.control_steps + 1) % self.settings.update_interval_steps == 0
        if boundary and dist.is_available() and dist.is_initialized():
            # Reduce only new rank-local observations; incomplete global windows are already shared.
            for name, previous in self._shared_statistics.items():
                values = getattr(self.sampler, name)
                delta = values - previous
                dist.all_reduce(delta)
                values.copy_(previous + delta)
        updated = self.sampler.step()
        self._step = self.sampler.control_steps
        if updated:
            self._remember_shared_statistics()
            if self.diagnostics is not None:
                self.diagnostics.write(self._step, self.sampler)
            self._refresh_sampling_logs()

    def __call__(self, env, env_ids, settings: LearningProgressCfg, command_name: str = "base_velocity"):
        ids = self._all_ids[env_ids]
        if self._snapshot_batch is not None:
            if self._snapshot_batch.steps:
                if not self._snapshot_batch.finished[ids].all():
                    raise RuntimeError("Manual resets of live cases are not supported during snapshot measurement.")
            else:
                self._assign_tasks(ids, self._snapshot_batch.task_ids[ids].clamp_min(0))
            return {}
        logs = dict(self._sampling_logs)
        completed = ids[(self.task_ids[ids] >= 0) & env.reset_buf[ids] & (self.tracking.steps[ids] > 0)]
        if len(completed):
            failed = env.termination_manager.terminated[completed]
            values = self.tracking.finalize(completed, env.max_episode_length, failed)
            tasks = self.task_ids[completed]
            scores = self._returns[completed].clone()
            boundary = torch.zeros_like(failed)
            if "assigned_terrain_boundary" in env.termination_manager.active_terms:
                boundary = ~failed & env.termination_manager.get_term("assigned_terrain_boundary")[completed]
            if not self.snapshot_enabled:
                self.sampler.observe(tasks, scores)
            if self.diagnostics is not None:
                details = {
                    "return": self._returns[completed],
                    "LP_score": scores / self.settings.reward_scale,
                    "episode_steps": self.tracking.steps[completed],
                    "failure": failed.float(),
                    "boundary_truncation": boundary.float(),
                }
                if self.online_window_enabled:
                    end_steps = torch.full_like(scores, self._step, dtype=torch.float64)
                    details["episode_end_control_step"] = end_steps
                    details["episode_start_control_step"] = end_steps - self.tracking.steps[completed].double()
                details.update(
                    {f"reward/{key}": value for key, value in env.reward_manager.get_episode_sums(completed).items()}
                )
                details.update(
                    {
                        f"termination/{key}": env.termination_manager.get_term(key)[completed].float()
                        for key in env.termination_manager.active_terms
                    }
                )
                self.diagnostics.record(tasks, details)
            logs.update(self.metrics.episodes(values, tasks, failed, boundary))
            if self.settings.debug_metrics:
                logs["episode_return"] = self._returns[completed].clone()
                logs["LP_return"] = scores / self.settings.reward_scale
        self.tracking.reset(ids)
        self._returns[ids] = 0.0

        # Select the next task before the reset event reads env_origins.
        tasks = self.sampler.sample(len(ids))
        self._assign_tasks(ids, tasks)
        return logs

    def _assign_tasks(self, ids, tasks):
        self.task_ids[ids] = tasks
        levels = self.catalog.terrain_levels[tasks]
        columns = self.catalog.terrain_columns[tasks].clone()
        for type_id, group in enumerate(self.catalog.column_groups):
            selected = self.catalog.terrain_type_ids[tasks] == type_id
            choices = torch.tensor(group, device=self.device)
            columns[selected] = choices[torch.randint(len(group), (int(selected.sum()),), device=self.device)]
        self.terrain.terrain_levels[ids] = levels
        self.terrain.terrain_types[ids] = columns
        self.terrain.env_origins[ids] = self.terrain.terrain_origins[levels, columns]
        self.command.assign_tasks(ids, self.catalog, tasks)

    def begin_online_window(self, policy_updates: int):
        """Attach the PPO clock without stepping, resetting, or evaluating the environment."""
        if not self.online_window_enabled or not self.online_controller_active:
            raise RuntimeError("Online-window LP requires its online curriculum controller.")
        if type(policy_updates) is not int or policy_updates < 0:
            raise ValueError("Completed policy updates must be a nonnegative integer.")
        if self.training_policy_updates is None:
            self._online_window_start_updates = policy_updates
        elif self.training_policy_updates != policy_updates:
            raise ValueError("Online LP and the runner disagree on completed policy updates.")
        self.training_policy_updates = policy_updates

    def complete_online_update(self, policy_updates: int):
        """Close LP statistics after a successful PPO update, retaining full running episodes."""
        if not self.online_window_enabled or not self.online_controller_active:
            raise RuntimeError("Online-window LP requires its online curriculum controller.")
        if (
            type(policy_updates) is not int
            or self.training_policy_updates is None
            or (policy_updates != self.training_policy_updates + 1)
        ):
            raise ValueError("Online LP requires consecutive completed PPO updates.")
        if policy_updates - self._online_window_start_updates < self.settings.update_interval_iterations:
            self.training_policy_updates = policy_updates
            return
        started = time.perf_counter()
        # Commit all task statistics together. No unfinished episode state is touched.
        candidate = self._make_sampler()
        candidate.load_state_dict(self.sampler.state_dict())
        candidate.step()
        record = {
            "window": candidate.update_count,
            "scope": "training_completed_episode_window",
            "start_policy_updates": self._online_window_start_updates,
            "policy_updates": policy_updates,
            "training_control_steps": self._step,
            "counts": candidate.last_window_counts.cpu().tolist(),
            "mean_valid": candidate.last_updated.cpu().tolist(),
            "progress_valid": candidate.progress_valid.cpu().tolist(),
            "missing_streak": candidate.missing_streak.cpu().tolist(),
            "scaled_return_mean": candidate.mean_returns.cpu().tolist(),
            "reward_scale": self.settings.reward_scale,
            "LP": candidate.learning_progress.cpu().tolist(),
            "probabilities": candidate.probabilities.cpu().tolist(),
            "metrics": candidate.metrics(),
        }
        if self.diagnostics is not None:
            with self.diagnostics.path.with_name("online_windows.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            self.diagnostics.write(self._step, candidate)
        self.sampler = candidate
        self.training_policy_updates = policy_updates
        self._online_window_start_updates = policy_updates
        self._remember_shared_statistics()
        self._refresh_sampling_logs()
        self._sampling_logs["lp_update_seconds"] = time.perf_counter() - started
        logger.info(
            "Online LP: window %s closed at PPO update %s; measured %s/%s tasks, adjacent comparisons %s/%s; "
            "boundary handling %.3fs.",
            candidate.update_count,
            policy_updates,
            int(candidate.last_updated.sum()),
            self.catalog.num_tasks,
            int(candidate.progress_valid.sum()),
            self.catalog.num_tasks,
            self._sampling_logs["lp_update_seconds"],
        )
        if candidate.update_count >= 2 and not candidate.progress_valid.any():
            logger.warning(
                "Online LP has no adjacent reliable task measurements; inspect window counts and missing streaks."
            )

    def begin_snapshot_batch(self, task_ids: torch.Tensor):
        """Assign a fixed task vector on the next explicit environment reset."""
        if not self.snapshot_enabled or self._snapshot_batch is not None:
            raise RuntimeError("Snapshot cohorts require snapshot mode and no active cohort.")
        task_ids = torch.as_tensor(task_ids, device=self.device)
        if task_ids.shape != (self.num_envs,):
            raise ValueError("Snapshot cohort must provide one task ID for every environment slot.")
        self._snapshot_batch = SnapshotProbeBatch(task_ids, self.catalog.num_tasks)

    @property
    def snapshot_batch_active(self) -> torch.Tensor:
        """Slots still measuring their first episode; retired slots receive zero controls."""
        if self._snapshot_batch is None:
            raise RuntimeError("There is no active snapshot measurement cohort.")
        return ~self._snapshot_batch.finished

    @property
    def snapshot_batch_complete(self) -> bool:
        """Whether all valid slots reached their first terminal event."""
        return self._snapshot_batch is not None and self._snapshot_batch.complete

    @property
    def snapshot_batch_returns(self) -> torch.Tensor:
        """Raw completed returns in slot order, including zero-valued padding."""
        if not self.snapshot_batch_complete:
            raise RuntimeError("Snapshot cohort has incomplete episodes.")
        return self._snapshot_batch.returns.clone()

    def end_snapshot_batch(self):
        """Release a cohort and discard interrupted training episodes before the fresh reset."""
        if self._snapshot_batch is not None:
            self.tracking.reset(self._all_ids)
            self._returns.zero_()
            self._env.reset_buf.zero_()
        self._snapshot_batch = None

    def commit_snapshot(self, returns: torch.Tensor, policy_updates: int, policy_digest: str):
        """Update signed LP only after every task has the same complete measurement count."""
        if not self.snapshot_enabled or self._snapshot_batch is not None:
            raise RuntimeError("Commit requires snapshot mode with no active cohort.")
        if isinstance(policy_updates, bool) or not isinstance(policy_updates, int):
            raise ValueError("Snapshot policy update count must be an integer.")
        if self.training_policy_updates is not None and policy_updates > self.training_policy_updates:
            raise ValueError("Snapshot policy updates cannot exceed completed training updates.")
        if (
            policy_updates <= self.last_snapshot_policy_updates
            or not isinstance(policy_digest, str)
            or not policy_digest
        ):
            raise ValueError("Snapshot policy identity must be present and newer than the previous round.")
        values = torch.as_tensor(returns, dtype=torch.float64, device=self.device)
        expected = (self.catalog.num_tasks, self.settings.min_samples)
        if values.shape != expected or not torch.isfinite(values).all():
            raise ValueError("Snapshot returns must be a complete finite task-by-repetition matrix.")
        if self.sampler.pending_counts.any():
            raise RuntimeError("Snapshot sampler unexpectedly contains incomplete pending measurements.")

        # Validate the entire new round in a candidate before replacing live sampling statistics.
        candidate = self._make_sampler()
        candidate.load_state_dict(self.sampler.state_dict())
        tasks = torch.arange(self.catalog.num_tasks, device=self.device).repeat_interleave(self.settings.min_samples)
        candidate.observe(tasks, values.flatten())
        candidate.step()
        sample_std = values.std(dim=1, correction=1) if self.settings.min_samples > 1 else None
        record = {
            "round": candidate.update_count,
            "scope": "frozen_policy_all_tasks",
            "training_control_steps": self._step,
            "policy_updates": policy_updates,
            "previous_policy_updates": self.last_snapshot_policy_updates,
            "policy_digest": policy_digest,
            "previous_policy_digest": self.last_snapshot_policy_digest,
            "evaluation_seed": self.settings.evaluation_seed,
            "counts": [self.settings.min_samples] * self.catalog.num_tasks,
            "return_mean": values.mean(dim=1).cpu().tolist(),
            "scaled_return_mean": candidate.mean_returns.cpu().tolist(),
            "return_sample_std": (
                sample_std.cpu().tolist() if sample_std is not None else [None] * self.catalog.num_tasks
            ),
            "return_standard_error": (
                (sample_std / math.sqrt(self.settings.min_samples)).cpu().tolist()
                if sample_std is not None
                else [None] * self.catalog.num_tasks
            ),
            "reward_scale": self.settings.reward_scale,
            "LP": candidate.learning_progress.cpu().tolist(),
            "probabilities": candidate.probabilities.cpu().tolist(),
        }
        if self.diagnostics is not None:
            with self.diagnostics.path.with_name("snapshot_rounds.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        self.sampler = candidate
        self.last_snapshot_policy_updates = policy_updates
        self.last_snapshot_policy_digest = policy_digest
        self._remember_shared_statistics()
        self._refresh_sampling_logs()

    def _refresh_sampling_logs(self):
        self._sampling_logs = sampling_metrics(self.sampler, self.catalog, self.settings.debug_metrics)
        if self.online_window_enabled:
            self._sampling_logs.update(
                {
                    name: value
                    for name, value in self.sampler.metrics().items()
                    if name.startswith(("lp_last_window_", "lp_missing_", "lp_effective_temperature", "lp_temperature"))
                    or name in ("lp_current_mean_coverage_fraction", "lp_progress_coverage_fraction")
                }
            )
            self._sampling_logs["lp_window_start_policy_updates"] = self._online_window_start_updates
            probabilities = self.sampler.probabilities
            self._sampling_logs["lp_unmeasured_probability_mass"] = float(
                probabilities[~self.sampler.last_updated].sum()
            )
            self._sampling_logs["lp_no_progress_probability_mass"] = float(
                probabilities[~self.sampler.progress_valid].sum()
            )
        if (self.snapshot_enabled or self.online_window_enabled) and "lp_control_steps" in self._sampling_logs:
            self._sampling_logs["lp_measurement_ticks"] = self._sampling_logs.pop("lp_control_steps")

    def state_dict(self):
        """Persist task identity and LP history; physical episodes restart on resume."""
        state = {
            "catalog": self.catalog.signature,
            "terrain": self._terrain_signature(),
            "return_contract": self._return_contract(),
            "settings": self.settings.to_dict(),
            "sampler_algorithm": (
                "online_window_lp_acrl_v1"
                if self.online_window_enabled
                else "snapshot_lp_acrl_v1"
                if self.snapshot_enabled
                else "uploaded_lp_acrl_v1"
            ),
            "sampler": self.sampler.state_dict(),
            "step": self._step,
            "last_snapshot_policy_updates": self.last_snapshot_policy_updates,
            "last_snapshot_policy_digest": self.last_snapshot_policy_digest,
            "training_policy_updates": self.training_policy_updates,
        }
        if self.snapshot_enabled or self.online_window_enabled:
            state["measurement_contract"] = self._measurement_contract()
        if self.online_window_enabled:
            state["online_window_start_updates"] = self._online_window_start_updates
        return state

    def load_state_dict(self, state):
        if self.online_window_enabled:
            self._load_online_state(state)
            return
        if (
            state.get("sampler_algorithm")
            != ("snapshot_lp_acrl_v1" if self.snapshot_enabled else "uploaded_lp_acrl_v1")
            or "return_contract" not in state
            or state.get("settings", {}).get("return_mode") != "episodic"
        ):
            logger.warning(
                "LP checkpoint uses a different measurement algorithm, has no return contract, "
                "or uses extrapolated returns; "
                "resetting LP statistics and retaining policy weights."
            )
            self.sampler = self._make_sampler()
            self._remember_shared_statistics()
            self._step = 0
            self.last_snapshot_policy_updates = -1
            self.last_snapshot_policy_digest = None
            self.training_policy_updates = None
            self._refresh_sampling_logs()
            return
        if self.snapshot_enabled:
            if "measurement_contract" not in state:
                training_updates = state.get("training_policy_updates")
                if training_updates is not None and (
                    isinstance(training_updates, bool) or not isinstance(training_updates, int) or training_updates < 0
                ):
                    raise ValueError("Snapshot checkpoint has an invalid completed training-update count.")
                logger.warning(
                    "Snapshot LP checkpoint has no measurement contract; resetting LP baselines while retaining "
                    "policy weights and the completed training-update count."
                )
                self.sampler = self._make_sampler()
                self._remember_shared_statistics()
                self._step = state["step"]
                self.last_snapshot_policy_updates = -1
                self.last_snapshot_policy_digest = None
                self.training_policy_updates = training_updates
                self._refresh_sampling_logs()
                return
            if state["measurement_contract"] != self._measurement_contract():
                raise ValueError(
                    "Snapshot LP checkpoint measurement conditions differ; use --reset_curriculum to rebuild baselines."
                )
        if state["return_contract"] != self._return_contract():
            raise ValueError("LP checkpoint reward configuration or episode horizon differs from this environment.")
        saved_settings = dict(state["settings"])
        saved_settings.setdefault("sampling_strategy", "lp")
        saved_settings.setdefault("return_mode", "boundary_extrapolated")
        saved_settings.setdefault("measurement_mode", "online")
        for key in ("evaluation_interval_iterations", "evaluation_seed", "update_interval_iterations"):
            saved_settings.setdefault(key, getattr(self.settings, key))
        settings = self.settings.to_dict()
        for key in ("debug_metrics", "diagnostics"):
            saved_settings.pop(key, None)
            settings.pop(key, None)
        if state["catalog"] != self.catalog.signature or saved_settings != settings:
            raise ValueError("LP checkpoint task ranges or settings differ from this environment.")
        if state["terrain"] != self._terrain_signature():
            raise ValueError("LP checkpoint terrain configuration differs from this environment.")
        sampler_state = state["sampler"]
        if self.snapshot_enabled:
            rounds = sampler_state["update_count"]
            if (
                any(sampler_state["pending_counts"])
                or any(sampler_state["pending_sums"])
                or any(count != rounds for count in sampler_state["window_counts"])
                or any(bool(value) != (rounds > 0) for value in sampler_state["initialized"])
            ):
                raise ValueError("Snapshot checkpoint contains incomplete task measurement rounds.")
            previous = state.get("last_snapshot_policy_updates", -1)
            digest = state.get("last_snapshot_policy_digest")
            if (
                isinstance(previous, bool)
                or not isinstance(previous, int)
                or (rounds > 0 and (previous < 0 or not isinstance(digest, str) or not digest))
                or (rounds == 0 and (previous != -1 or digest is not None))
            ):
                raise ValueError("Snapshot checkpoint has invalid policy measurement identity.")
            training_updates = state.get("training_policy_updates")
            if training_updates is not None and (
                isinstance(training_updates, bool)
                or not isinstance(training_updates, int)
                or training_updates < max(previous, 0)
            ):
                raise ValueError("Snapshot checkpoint has an invalid completed training-update count.")
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            sampler_state = dict(sampler_state)
            for name in ("pending_sums", "pending_counts"):
                sampler_state[name] = [0] * self.catalog.num_tasks
            logger.warning("Distributed LP resume discards unsynchronized pending episode windows.")
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            sampler_state = dict(sampler_state)
            fresh_state = self._make_sampler().state_dict()
            sampler_state["rng_state"] = fresh_state["rng_state"]
            sampler_state["seed"] = self._seed
            logger.warning("Distributed LP resume restarts the non-root rank's task RNG from its configured seed.")
        self.sampler.load_state_dict(sampler_state)
        self._step = state["step"] if self.snapshot_enabled else self.sampler.control_steps
        self.last_snapshot_policy_updates = state.get("last_snapshot_policy_updates", -1)
        self.last_snapshot_policy_digest = state.get("last_snapshot_policy_digest")
        self.training_policy_updates = state.get("training_policy_updates")
        self._remember_shared_statistics()
        self._refresh_sampling_logs()

    def _load_online_state(self, state):
        training_updates = state.get("training_policy_updates")
        if training_updates is not None and (type(training_updates) is not int or training_updates < 0):
            raise ValueError("Online LP checkpoint has an invalid completed training-update count.")
        if state.get("sampler_algorithm") != "online_window_lp_acrl_v1":
            logger.warning(
                "LP checkpoint uses a different measurement algorithm; rebuilding online-window LP baselines "
                "while retaining policy weights and the completed training-update count."
            )
            self.sampler = self._make_sampler()
            self.training_policy_updates = training_updates
            self._online_window_start_updates = training_updates or 0
            self._step = 0
            self._remember_shared_statistics()
            self._refresh_sampling_logs()
            return
        if state.get("measurement_contract") != self._measurement_contract():
            raise ValueError("Online LP checkpoint measurement conditions differ; use --reset_curriculum.")
        if state.get("return_contract") != self._return_contract():
            raise ValueError("Online LP checkpoint reward configuration or episode horizon differs.")
        settings, saved_settings = self.settings.to_dict(), dict(state.get("settings", {}))
        # These do not change the online measurement or sampling distribution.
        for key in (
            "diagnostics",
            "debug_metrics",
            "evaluation_interval_iterations",
            "evaluation_seed",
            "update_interval_steps",
        ):
            settings.pop(key, None)
            saved_settings.pop(key, None)
        if state.get("catalog") != self.catalog.signature or saved_settings != settings:
            raise ValueError("Online LP checkpoint task ranges or settings differ; use --reset_curriculum.")
        if state.get("terrain") != self._terrain_signature():
            raise ValueError("Online LP checkpoint terrain configuration differs; use --reset_curriculum.")
        start = state.get("online_window_start_updates")
        step = state.get("step")
        if (
            training_updates is None
            or type(start) is not int
            or start < 0
            or not 0 <= training_updates - start < self.settings.update_interval_iterations
            or type(step) is not int
            or step < 0
        ):
            raise ValueError("Online LP checkpoint has an invalid PPO window clock.")
        candidate = self._make_sampler()
        candidate.load_state_dict(state["sampler"])
        if training_updates == start and candidate.pending_counts.any():
            raise ValueError("Online LP checkpoint has pending samples outside a training window.")
        if training_updates != start:
            # Simulator episodes restart. Keeping either the partial samples or their old
            # predecessor would compare nonadjacent windows on the next full close.
            logger.warning(
                "Online LP resume discards the partial window and rebuilds its baseline; "
                "the next full window starts at completed PPO update %s.",
                training_updates,
            )
            candidate = self._make_sampler()
            start = training_updates
        self.sampler = candidate
        self.training_policy_updates = training_updates
        self._online_window_start_updates = start
        self._step = step
        self._remember_shared_statistics()
        self._refresh_sampling_logs()

    def _make_sampler(self):
        return LearningProgressCurriculum(
            self.catalog.num_tasks,
            self.device,
            temperature=self.settings.temperature,
            exploration=1.0 if self.settings.sampling_strategy == "uniform" else self.settings.exploration,
            min_episodes=self.settings.min_samples,
            update_interval=1
            if (self.snapshot_enabled or self.online_window_enabled)
            else self.settings.update_interval_steps,
            reward_scale=self.settings.reward_scale,
            seed=self._seed,
            group_sizes=self.settings.group_sizes,
            progress_ema_alpha=self.settings.progress_ema_alpha,
            max_uniform_kl=self.settings.max_uniform_kl,
            window_mode="fixed" if self.online_window_enabled else "accumulating",
        )

    def _remember_shared_statistics(self):
        self._shared_statistics = {
            name: getattr(self.sampler, name).clone() for name in ("pending_sums", "pending_counts", "episode_counts")
        }

    def _return_contract(self):
        return {
            "horizon_steps": self._env.max_episode_length,
            "step_dt": self._env.step_dt,
            "rewards": to_dict(self._env.cfg.rewards),
        }

    def _measurement_contract(self):
        cfg = self._env.cfg
        robot = cfg.scene.robot
        return {
            "version": 2,
            "observations": to_dict(cfg.observations),
            "events": to_dict(cfg.events),
            "terminations": to_dict(cfg.terminations),
            "actions": to_dict(cfg.actions),
            "robot": {
                name: to_dict(getattr(robot, name, None))
                for name in (
                    "class_type",
                    "spawn",
                    "init_state",
                    "collision_group",
                    "articulation_root_prim_path",
                    "soft_joint_pos_limit_factor",
                    "joint_ordering",
                    "body_ordering",
                    "actuators",
                )
            },
            "simulation": {
                name: to_dict(getattr(cfg.sim, name))
                for name in ("dt", "gravity", "physics_material", "use_newton_actuators", "physics")
            },
            "decimation": cfg.decimation,
            "global_seed": getattr(cfg, "seed", None),
            "terrain_generation_seed": getattr(self.terrain, "generation_seed", None),
            "terrain_physics_material": to_dict(self.terrain.cfg.physics_material),
            "num_envs": self.num_envs,
            "resampling_time_range": self.command.cfg.resampling_time_range,
            "transition_resampling_time_range": self.command.cfg.transition_resampling_time_range,
        }

    def _terrain_signature(self):
        cfg = self.terrain.cfg.terrain_generator
        return {
            name: cfg.to_dict()[name]
            for name in (
                "seed",
                "size",
                "num_rows",
                "num_cols",
                "border_width",
                "border_height",
                "difficulty_range",
                "horizontal_scale",
                "vertical_scale",
                "slope_threshold",
                "sub_terrains",
            )
        }
