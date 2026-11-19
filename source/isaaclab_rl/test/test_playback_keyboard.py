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
from isaacteleop.retargeting_engine.interface import (
    ComputeContext,
    ExecutionEvents,
    ExecutionState,
    GraphTime,
    TensorGroup,
)
from isaacteleop.schema import KeyAction, KeyboardOutput, KeyEvent

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


def test_wasd_tap_between_frames():
    """A press and release between snapshots produce one command, without latching motion."""
    pipeline = wasd_keyboard_cfg(v_x_sensitivity=2.0, v_y_sensitivity=1.0, omega_z_sensitivity=3.0).pipeline_builder()
    source = find_sources(pipeline, KeyboardSource)[0]
    snapshot = TensorGroup(source.input_spec()["deviceio_keyboard"])
    events = [
        KeyEvent(timestamp_ns=index, code=key, action=action)
        for index, (key, action) in enumerate(
            [
                (EvdevKeyCode.KeyW, KeyAction.PRESS),
                (EvdevKeyCode.KeyW, KeyAction.RELEASE),
                (EvdevKeyCode.ArrowRight, KeyAction.PRESS),
                (EvdevKeyCode.ArrowRight, KeyAction.RELEASE),
            ]
        )
    ]
    snapshot[0] = KeyboardOutput(held_keys=[], events=events)
    command = pipeline({source.name: {"deviceio_keyboard": snapshot}})["action"][0]
    np.testing.assert_allclose(command, [2.0, 0.0, -3.0])

    snapshot[0] = KeyboardOutput(held_keys=[])
    command = pipeline({source.name: {"deviceio_keyboard": snapshot}})["action"][0]
    np.testing.assert_array_equal(command, [0.0, 0.0, 0.0])


def test_remote_keyboard_repeat_grace():
    """Remote translation and yaw pulses combine and expire independently."""
    pipeline = wasd_keyboard_cfg(v_x_sensitivity=2.0, repeat_grace_s=0.1).pipeline_builder()
    source = find_sources(pipeline, KeyboardSource)[0]
    snapshot = TensorGroup(source.input_spec()["deviceio_keyboard"])

    def advance(time_ms, tap=False, running=True, yaw=False, held=()):
        keys = ([EvdevKeyCode.KeyW] if tap else []) + ([EvdevKeyCode.ArrowLeft] if yaw else [])
        events = [
            KeyEvent(timestamp_ns=time_ms * 1000000 + index, code=key, action=action)
            for key in keys
            for index, action in enumerate((KeyAction.PRESS, KeyAction.RELEASE))
        ]
        snapshot[0] = KeyboardOutput(held_keys=list(held), events=events)
        context = ComputeContext(
            graph_time=GraphTime(sim_time_ns=0, real_time_ns=time_ms * 1000000),
            execution_events=ExecutionEvents(
                execution_state=ExecutionState.RUNNING if running else ExecutionState.PAUSED
            ),
        )
        return pipeline.execute_pipeline({source.name: {"deviceio_keyboard": snapshot}}, context)["action"][0]

    for time_ms in range(0, 100, 20):
        np.testing.assert_allclose(advance(time_ms, tap=time_ms % 40 == 0), [2.0, 0.0, 0.0])
    np.testing.assert_allclose(advance(160), [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(advance(181), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(advance(200, tap=True), [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(advance(220, running=False), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(advance(240), [0.0, 0.0, 0.0])

    # Translation and yaw can arrive in different snapshots without replacing one another.
    np.testing.assert_allclose(advance(300, tap=True), [2.0, 0.0, 0.0])
    np.testing.assert_allclose(advance(320, yaw=True), [2.0, 0.0, 1.0])
    np.testing.assert_allclose(advance(380, tap=True), [2.0, 0.0, 1.0])
    np.testing.assert_allclose(advance(421), [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(advance(481), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(advance(500, yaw=True, held=[EvdevKeyCode.KeyW]), [2.0, 0.0, 1.0])
