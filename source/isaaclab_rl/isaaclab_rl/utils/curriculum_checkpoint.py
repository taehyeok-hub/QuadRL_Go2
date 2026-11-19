# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Persist stateful environment curricula inside RSL-RL model checkpoints."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class RslRlCurriculumCheckpoint(contextlib.AbstractContextManager):
    """Attach curriculum persistence to a runner for the lifetime of this context.

    The environment manager owns curriculum serialization and compatibility validation;
    this adapter also checks wrapper action clipping for snapshot and fixed online-window measurements. Only completed
    curriculum statistics are restored; simulator and unfinished episode state are not replayed.
    Disable ``restore_curriculum`` to keep fresh curriculum statistics when loading a policy.
    """

    def __init__(self, runner: Any, *, restore_curriculum: bool = True):
        self._runner = runner
        self._restore_curriculum = restore_curriculum
        self._manager = getattr(runner.env.unwrapped, "curriculum_manager", None)
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        if self._manager is None or not hasattr(self._manager, "state_dict"):
            return self
        if not self._manager.state_dict():
            return self

        # Save through the original runner so uploaded checkpoints include curriculum state.
        original_save = self._runner.save
        original_load = self._runner.load

        def save(path: str, infos: dict | None = None):
            combined = dict(infos or {})
            combined["isaaclab_curriculum"] = self._manager.state_dict()
            runtime_contract = self._runtime_contract()
            if runtime_contract is not None:
                combined["isaaclab_curriculum_runtime_contract"] = runtime_contract
            return original_save(path, infos=combined)

        def load(*args, **kwargs):
            infos = original_load(*args, **kwargs)
            if not self._restore_curriculum:
                logger.warning(
                    "Curriculum reset requested: checkpoint policy and selected training state were loaded; "
                    "curriculum statistics remain at initialization (--reset_curriculum)."
                )
            elif isinstance(infos, dict) and "isaaclab_curriculum" in infos:
                runtime_contract = self._runtime_contract()
                if (
                    runtime_contract is not None
                    and infos.get("isaaclab_curriculum_runtime_contract") != runtime_contract
                ):
                    raise ValueError(
                        "LP checkpoint action-clipping conditions differ or are missing; "
                        "use --reset_curriculum to rebuild baselines while retaining policy weights."
                    )
                self._manager.load_state_dict(infos["isaaclab_curriculum"])
                terms = getattr(self._manager, "_class_term_cfgs", ())
                if any(getattr(cfg.func, "reset_on_checkpoint_load", False) for cfg in terms):
                    self._runner.env.reset()
            else:
                logger.warning(
                    "Checkpoint has no curriculum state; starting curriculum statistics from initialization."
                )
            return infos

        for name, replacement in (("save", save), ("load", load)):
            if name in self._runner.__dict__:
                self._stack.callback(setattr, self._runner, name, self._runner.__dict__[name])
            else:
                self._stack.callback(delattr, self._runner, name)
            setattr(self._runner, name, replacement)
        return self

    def _runtime_contract(self) -> dict | None:
        terms = getattr(self._manager, "_class_term_cfgs", ())
        if any(
            getattr(cfg.func, "snapshot_enabled", False) or getattr(cfg.func, "online_window_enabled", False)
            for cfg in terms
        ):
            return {"clip_actions": getattr(self._runner.env, "clip_actions", None)}
        return None

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)
