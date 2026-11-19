# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preserve an interrupted training job's failure status for shell experiment runners."""

import contextlib


@contextlib.contextmanager
def training_interrupt_exit():
    """Allow cleanup, then return shell status 130 instead of treating Ctrl+C as completion."""
    try:
        yield
    except KeyboardInterrupt:
        raise SystemExit(130) from None
