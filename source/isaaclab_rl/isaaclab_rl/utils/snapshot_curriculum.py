# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Measure curriculum progress between complete frozen-policy evaluation rounds."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import logging
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner

logger = logging.getLogger(__name__)


class RslRlSnapshotCurriculum(contextlib.AbstractContextManager):
    """Pause PPO after completed updates and score every task with one frozen actor.

    Construct before installing model instrumentation. An actor template is copied once,
    then refreshed using weights and normalization buffers for each round. Probe transitions
    never enter PPO storage or its episode logger. The simulator advances physically;
    training counters and event timers exclude these probe steps.
    """

    def __init__(
        self,
        runner: OnPolicyRunner,
        *,
        resumed: bool = False,
        probe_context: Callable[[], contextlib.AbstractContextManager] = contextlib.nullcontext,
    ):
        self._runner = runner
        self._probe_context = probe_context
        self._stack = contextlib.ExitStack()
        self._last_obs = None
        manager = getattr(runner.env.unwrapped, "curriculum_manager", None)
        terms = [] if manager is None else [cfg.func for cfg in manager._class_term_cfgs]
        self._terms = [term for term in terms if getattr(term, "snapshot_enabled", False)]
        self._actor = None
        self._completed_updates = 0
        if not self._terms:
            return
        if len(self._terms) != 1:
            raise ValueError("Frozen-policy LP supports one snapshot curriculum term per environment.")
        if not isinstance(runner, OnPolicyRunner) or not isinstance(runner.alg, PPO) or runner.is_distributed:
            raise ValueError("Frozen-policy LP requires single-device RSL-RL OnPolicyRunner with PPO.")
        physics_cfg = getattr(getattr(getattr(runner.env.unwrapped, "cfg", None), "sim", None), "physics", None)
        if getattr(getattr(physics_cfg, "solver_cfg", None), "use_mujoco_cpu", False):
            raise ValueError("Frozen-policy LP requires a solver with isolated per-environment resets.")
        actor = runner.alg.get_policy()
        if not isinstance(actor, MLPModel) or actor.is_recurrent or runner.alg.critic.is_recurrent:
            raise ValueError("Frozen-policy LP requires an MLP actor and non-recurrent actor/critic models.")
        self._actor = copy.deepcopy(actor).eval()
        self._actor.requires_grad_(False)
        if resumed:
            restored_updates = self._terms[0].training_policy_updates
            self._completed_updates = (
                restored_updates if restored_updates is not None else runner.current_learning_iteration + 1
            )

    def __enter__(self):
        if not self._terms:
            return self
        try:
            term = self._terms[0]
            self._stack.callback(setattr, term, "snapshot_controller_active", term.snapshot_controller_active)
            term.snapshot_controller_active = True
            previous = term.last_snapshot_policy_updates
            if previous is not None and previous > self._completed_updates:
                raise ValueError("Snapshot curriculum baseline belongs to a newer policy than the loaded checkpoint.")
            term.training_policy_updates = self._completed_updates
            if (
                previous is None
                or previous < 0
                or self._completed_updates - previous >= term.settings.evaluation_interval_iterations
            ):
                self._evaluate()
            self._replace(self._runner.alg, "compute_returns", self._compute_returns)
            self._replace(self._runner.alg, "update", self._update)
            return self
        except BaseException:
            self._stack.close()
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def _replace(self, owner: Any, name: str, replacement: Callable, *, stack=None):
        stack = self._stack if stack is None else stack
        original = getattr(owner, name)
        if name in vars(owner):
            stack.callback(setattr, owner, name, vars(owner)[name])
        else:
            stack.callback(delattr, owner, name)
        setattr(owner, name, lambda *args, **kwargs: replacement(original, *args, **kwargs))

    def _compute_returns(self, original: Callable, obs):
        # Retain the runner's actual observation object to refresh it after a training reset.
        self._last_obs = obs
        return original(obs)

    def _update(self, original: Callable):
        result = original()
        self._completed_updates += 1
        self._terms[0].training_policy_updates = self._completed_updates
        previous = self._terms[0].last_snapshot_policy_updates
        interval = self._terms[0].settings.evaluation_interval_iterations
        if previous is None or self._completed_updates - previous >= interval:
            if self._runner.alg.storage.step != 0:
                raise RuntimeError("Frozen-policy LP must run after PPO clears rollout storage.")
            self._evaluate()
        return result

    @torch.inference_mode()
    def _evaluate(self):
        with self._probe_context():
            self._measure_and_reset()

    def _measure_and_reset(self):
        term = self._terms[0]
        env = self._runner.env
        base_env = env.unwrapped
        self._actor.load_state_dict(self._runner.alg.get_policy().state_dict())
        self._actor.eval()
        policy_digest = _policy_digest(self._actor)
        case_tasks = torch.arange(term.catalog.num_tasks, device=env.device).repeat_interleave(
            term.settings.min_samples
        )
        scores = torch.empty(len(case_tasks), dtype=torch.float64, device=env.device)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        common_steps, sim_steps = base_env.common_step_counter, base_env._sim_step_counter
        event_state = _event_timers(base_env.event_manager)
        logger.info(
            "Frozen-policy LP: evaluating %s episodes at policy update %s.", len(case_tasks), self._completed_updates
        )
        try:
            for batch_index, start in enumerate(range(0, len(case_tasks), env.num_envs)):
                count = min(env.num_envs, len(case_tasks) - start)
                tasks = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
                tasks[:count] = case_tasks[start : start + count]
                _seed_probe(term.settings.evaluation_seed + batch_index)
                base_env.common_step_counter, base_env._sim_step_counter = 0, 0
                _reset_probe_timers(base_env.event_manager)
                term.begin_snapshot_batch(tasks)
                try:
                    obs, _ = env.reset()
                    _check_probe_observations(obs)
                    with contextlib.ExitStack() as reset_stack:
                        self._replace(base_env, "_reset_idx", _reset_retired_slots, stack=reset_stack)
                        for _ in range(int(env.max_episode_length)):
                            # Keep the full sampling shape so retirements cannot shift live action noise.
                            actions = self._actor(obs.to(self._runner.device), stochastic_output=True)
                            actions[~term.snapshot_batch_active.to(actions.device)] = 0
                            if not torch.isfinite(actions).all():
                                raise FloatingPointError("Frozen-policy LP produced non-finite actions.")
                            obs, _, _, _ = env.step(actions.to(env.device))
                            _check_probe_observations(obs)
                            if term.snapshot_batch_complete:
                                break
                    if not term.snapshot_batch_complete:
                        raise RuntimeError("Frozen-policy LP cohort did not finish within the episode horizon.")
                    scores[start : start + count] = term.snapshot_batch_returns[:count]
                finally:
                    term.end_snapshot_batch()
            if not torch.isfinite(scores).all():
                raise FloatingPointError("Frozen-policy LP produced non-finite returns; LP was not updated.")
            term.commit_snapshot(
                scores.view(term.catalog.num_tasks, term.settings.min_samples),
                self._completed_updates,
                policy_digest,
            )
        finally:
            # Restore the training random stream before its necessary fresh episode reset.
            term.end_snapshot_batch()
            base_env.common_step_counter, base_env._sim_step_counter = common_steps, sim_steps
            _restore_event_timers(base_env.event_manager, event_state)
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            obs, _ = env.reset()
            base_env.extras["log"] = {}
            self._runner.logger.cur_reward_sum.zero_()
            self._runner.logger.cur_episode_length.zero_()
            if hasattr(self._runner.logger, "cur_ereward_sum"):
                self._runner.logger.cur_ereward_sum.zero_()
                self._runner.logger.cur_ireward_sum.zero_()
            if self._last_obs is not None:
                self._last_obs.update_(obs.to(self._runner.device))


def _check_probe_observations(obs):
    if not torch.stack([torch.isfinite(value).all() for value in obs.values(True, True)]).all():
        raise FloatingPointError("Frozen-policy LP received non-finite observations.")


def _reset_retired_slots(original: Callable, *args, **kwargs):
    """Run the entire normal slot reset without consuming the live cases' random stream."""
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        return original(*args, **kwargs)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _policy_digest(actor: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _seed_probe(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def _event_timers(manager) -> dict[str, list[torch.Tensor]]:
    return {
        name: [value.clone() for value in getattr(manager, name)]
        for name in (
            "_interval_term_time_left",
            "_reset_term_last_triggered_step_id",
            "_reset_term_last_triggered_once",
        )
    }


def _reset_probe_timers(manager):
    # Fixed seeds also fix interval phase and reset-event cooldowns for each cohort.
    for cfg, timer in zip(manager._mode_term_cfgs.get("interval", []), manager._interval_term_time_left):
        lower, upper = cfg.interval_range_s
        timer.copy_(torch.rand_like(timer) * (upper - lower) + lower)
    for values in manager._reset_term_last_triggered_step_id:
        values.zero_()
    for values in manager._reset_term_last_triggered_once:
        values.zero_()


def _restore_event_timers(manager, state: dict[str, list[torch.Tensor]]):
    for name, previous in state.items():
        for current, saved in zip(getattr(manager, name), previous):
            current.copy_(saved)
