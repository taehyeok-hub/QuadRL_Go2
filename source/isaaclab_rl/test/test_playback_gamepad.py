# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback's velocity and camera commands from encoded gamepad snapshots."""

import numpy as np
import pytest

pytest.importorskip("isaacteleop")

from isaacteleop.retargeting_engine.deviceio_source_nodes import GamepadSource, find_sources
from isaacteleop.retargeting_engine.interface import TensorGroup
from isaacteleop.schema import GamepadInputState, GamepadSnapshot

from isaaclab_rl.entrypoints.backends._playback_gamepad import gamepad_with_camera_cfg


def test_gamepad_camera_button_preserves_velocity_and_clears_on_disconnect():
    """Y is independent of the sticks and other buttons; unavailable input clears all commands."""
    pipeline = gamepad_with_camera_cfg(
        v_x_sensitivity=2.0, v_y_sensitivity=1.0, omega_z_sensitivity=3.0, dead_zone=0.15
    ).pipeline_builder()
    source = find_sources(pipeline, GamepadSource)[0]
    snapshot = TensorGroup(source.input_spec()["deviceio_gamepad"])

    def advance(inputs):
        snapshot[0] = inputs
        return pipeline({source.name: {"deviceio_gamepad": snapshot}})["action"][0]

    command = advance(
        GamepadSnapshot(
            GamepadInputState(north_click=True, thumbstick_left_x=-0.5, thumbstick_left_y=0.75, thumbstick_right_x=0.25)
        )
    )
    np.testing.assert_allclose(command, [1.5, -0.5, 0.75, 1.0])

    command = advance(
        GamepadSnapshot(
            GamepadInputState(south_click=True, west_click=True, thumbstick_left_y=0.05, thumbstick_right_x=-0.5)
        )
    )
    np.testing.assert_allclose(command, [0.0, 0.0, -1.5, 0.0])

    advance(GamepadSnapshot(GamepadInputState(north_click=True, thumbstick_left_y=1.0)))
    np.testing.assert_array_equal(advance(None), [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(advance(GamepadSnapshot()), [0.0, 0.0, 0.0, 0.0])
