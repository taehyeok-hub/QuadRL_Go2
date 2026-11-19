# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity curricula: LP-specific integration lives in :mod:`.lp`."""

import importlib.util as _import_util
import sys as _sys

from . import lp as _lp

# Keep saved pre-move class paths readable without eagerly importing LP for success curricula.
for _name in ("config", "commands", "learning_progress", "metrics", "snapshot_probe"):
    _target = f"{__name__}.lp.{_name}"
    _spec = _import_util.find_spec(_target)
    _spec.loader = _import_util.LazyLoader(_spec.loader)
    _module = _import_util.module_from_spec(_spec)
    _sys.modules[_target] = _module
    _spec.loader.exec_module(_module)
    _sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
    setattr(_lp, _name, _module)
