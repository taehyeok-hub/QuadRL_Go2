# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback's WASD translation and arrow-key yaw from keyboard snapshots."""

import numpy as np
import pytest

pytest.importorskip("isaacteleop")

from isaacteleop.deviceio_trackers import EvdevKeyCode
from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource, find_sources
from isaacteleop.retargeting_engine.interface import TensorGroup
from isaacteleop.schema import KeyboardOutput

from isaaclab_rl.entrypoints.backends._playback_keyboard import wasd_keyboard_cfg


def test_wasd_translation_and_arrow_yaw():
    """WASD and yaw arrows compose independently, with no Ctrl/Shift remapping or held commands after release."""
    pipeline = wasd_keyboard_cfg(v_x_sensitivity=2.0, v_y_sensitivity=1.0, omega_z_sensitivity=3.0).pipeline_builder()
    source = find_sources(pipeline, KeyboardSource)[0]
    snapshot = TensorGroup(source.input_spec()["deviceio_keyboard"])

    def advance(keys):
        snapshot[0] = None if keys is None else KeyboardOutput(held_keys=keys)
        return pipeline({source.name: {"deviceio_keyboard": snapshot}})["action"][0]

    np.testing.assert_allclose(advance([EvdevKeyCode.KeyW, EvdevKeyCode.KeyA]), [2.0, 1.0, 0.0])
    np.testing.assert_allclose(advance([EvdevKeyCode.KeyS, EvdevKeyCode.KeyD]), [-2.0, -1.0, 0.0])

    np.testing.assert_allclose(advance([EvdevKeyCode.ArrowLeft]), [0.0, 0.0, 3.0])
    np.testing.assert_allclose(advance([EvdevKeyCode.ArrowRight]), [0.0, 0.0, -3.0])
    np.testing.assert_allclose(
        advance([EvdevKeyCode.KeyW, EvdevKeyCode.KeyA, EvdevKeyCode.ArrowRight]), [2.0, 1.0, -3.0]
    )
    np.testing.assert_allclose(advance([EvdevKeyCode.KeyW, EvdevKeyCode.KeyA]), [2.0, 1.0, 0.0])

    for modifier in (
        EvdevKeyCode.ControlLeft,
        EvdevKeyCode.ControlRight,
        EvdevKeyCode.ShiftLeft,
        EvdevKeyCode.ShiftRight,
    ):
        np.testing.assert_allclose(advance([modifier, EvdevKeyCode.KeyA]), [0.0, 1.0, 0.0])

    np.testing.assert_array_equal(advance([]), [0.0, 0.0, 0.0])
    advance([EvdevKeyCode.KeyA, EvdevKeyCode.ArrowLeft])
    np.testing.assert_array_equal(advance(None), [0.0, 0.0, 0.0])
