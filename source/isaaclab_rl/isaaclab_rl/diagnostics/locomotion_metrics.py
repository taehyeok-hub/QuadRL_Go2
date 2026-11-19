# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Route reward-independent locomotion telemetry to the native RSL-RL writer."""

from __future__ import annotations

import contextlib

from isaaclab.utils import instantiate


class RslRlLocomotionMetrics(contextlib.AbstractContextManager):
    """Observe pre-reset steps and report per rollout or per playback interval.

    The environment's curriculum update is the existing post-physics, pre-reset hook.
    No recorder, observation computation, random sampling, or environment step is added.
    """

    def __init__(self, runner, *, playback: bool = False):
        self._runner = runner
        self._env = runner.env.unwrapped
        self._stack = contextlib.ExitStack()
        self._playback = playback
        self._suspended = 0
        self._steps = 0
        self._window_steps = 0
        self._iteration = runner.current_learning_iteration
        cfg = getattr(self._env.cfg, "locomotion_metrics", None)
        self._observer = (
            instantiate(cfg, env=self._env)
            if cfg is not None and cfg.enabled and not runner.logger.disable_logs
            else None
        )

    def _replace(self, owner, name, replacement):
        if name in vars(owner):
            self._stack.callback(setattr, owner, name, vars(owner)[name])
        else:
            self._stack.callback(delattr, owner, name)
        setattr(owner, name, replacement)

    def __enter__(self):
        if self._observer is None:
            return self
        try:
            update = self._env.curriculum_manager.update
            reset = self._env._reset_idx

            def observe(dt):
                if not self._suspended:
                    self._observer.sample(dt)
                    self._steps += 1
                    self._window_steps += 1
                    if self._playback and self._window_steps >= self._observer.cfg.playback_interval_steps:
                        self.flush()
                return update(dt)

            def reset_swings(env_ids):
                self._observer.reset(env_ids)
                return reset(env_ids)

            self._replace(self._env.curriculum_manager, "update", observe)
            self._replace(self._env, "_reset_idx", reset_swings)
            if not self._playback:
                original_update = self._runner.alg.update

                def report_update(*args, **kwargs):
                    result = original_update(*args, **kwargs)
                    self.flush()
                    self._iteration += 1
                    return result

                self._replace(self._runner.alg, "update", report_update)
            return self
        except BaseException:
            self._stack.close()
            raise

    def flush(self):
        """Write only valid measurements, without adding them to episode rewards or console output."""
        if self._observer is None or not self._window_steps:
            return
        values = self._observer.pop()
        writer = self._runner.logger.writer
        if writer is not None:
            step = self._steps if self._playback else self._iteration
            for name, value in values.items():
                writer.add_scalar(name, value, step)
        self._window_steps = 0

    @contextlib.contextmanager
    def suspend(self):
        """Exclude frozen-policy curriculum probes from training telemetry."""
        self._suspended += 1
        try:
            yield
        finally:
            self._suspended -= 1

    def __exit__(self, *exc):
        try:
            self.flush()
        finally:
            self._stack.__exit__(*exc)
