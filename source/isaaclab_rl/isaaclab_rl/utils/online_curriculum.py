# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Close fixed curriculum measurement windows after completed PPO updates."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner


class RslRlOnlineCurriculum(contextlib.AbstractContextManager):
    """Notify one fixed-window task curriculum after successful single-device PPO updates.

    The task term owns window statistics and resume invalidation. This controller only
    counts completed updates; it never evaluates a policy or resets an environment.
    """

    def __init__(self, runner: OnPolicyRunner, *, resumed: bool = False):
        self._runner = runner
        self._stack = contextlib.ExitStack()
        manager = getattr(runner.env.unwrapped, "curriculum_manager", None)
        terms = [] if manager is None else [cfg.func for cfg in manager._class_term_cfgs]
        self._terms = [term for term in terms if getattr(term, "online_window_enabled", False)]
        self._completed_updates = 0
        if not self._terms:
            return
        if len(self._terms) != 1:
            raise ValueError("Fixed-window LP supports one online curriculum term per environment.")
        if not isinstance(runner, OnPolicyRunner) or not isinstance(runner.alg, PPO) or runner.is_distributed:
            raise ValueError("Fixed-window LP requires single-device RSL-RL OnPolicyRunner with PPO.")
        if resumed:
            restored = self._terms[0].training_policy_updates
            self._completed_updates = restored if restored is not None else runner.current_learning_iteration + 1
        if type(self._completed_updates) is not int or self._completed_updates < 0:
            raise ValueError("Completed policy-update count must be a nonnegative integer.")

    def __enter__(self):
        if not self._terms:
            return self
        try:
            term = self._terms[0]
            self._stack.callback(setattr, term, "online_controller_active", term.online_controller_active)
            term.online_controller_active = True
            term.begin_online_window(self._completed_updates)
            original = self._runner.alg.update
            if "update" in vars(self._runner.alg):
                self._stack.callback(setattr, self._runner.alg, "update", vars(self._runner.alg)["update"])
            else:
                self._stack.callback(delattr, self._runner.alg, "update")
            self._runner.alg.update = lambda: self._update(original)
            return self
        except BaseException:
            self._stack.close()
            raise

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)

    def _update(self, original: Callable):
        result = original()
        if self._runner.alg.storage.step != 0:
            raise RuntimeError("Fixed-window LP must run after PPO clears rollout storage.")
        completed_updates = self._completed_updates + 1
        self._terms[0].complete_online_update(completed_updates)
        self._completed_updates = completed_updates
        return result
