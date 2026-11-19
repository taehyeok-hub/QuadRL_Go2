# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback input using w/a/s/d for translation and left/right arrows for yaw."""

import math

import numpy as np

from isaacteleop.deviceio_trackers import EvdevKeyCode
from isaacteleop.retargeters import KeyboardToSe2Retargeter, KeyboardToSe2RetargeterConfig
from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardPressedType, KeyboardSource
from isaacteleop.retargeting_engine.interface import (
    ComputeContext,
    ExecutionState,
    OptionalType,
    OutputCombiner,
    RetargeterIO,
    RetargeterIOType,
)

from isaaclab_teleop import IsaacTeleopCfg, se2_keyboard_teleop_cfg


class _WasdKeyboard(KeyboardToSe2Retargeter):
    """Read physical w/a/s/d and arrow keys independently of letter case."""

    def __init__(self, config: KeyboardToSe2RetargeterConfig, name: str, repeat_grace_s: float = 0.0):
        super().__init__(config, name)
        self._repeat_grace_ns = int(repeat_grace_s * 1.0e9)
        self._pressed_until: np.ndarray | None = None

    def input_spec(self) -> RetargeterIOType:
        return {**super().input_spec(), "keyboard_pressed": OptionalType(KeyboardPressedType())}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context: ComputeContext) -> None:
        command = np.zeros(3, dtype=np.float32)
        held = inputs["keyboard_held"]
        running = context.execution_events.execution_state == ExecutionState.RUNNING
        if self._pressed_until is not None and (held.is_none or context.execution_events.reset or not running):
            self._pressed_until.fill(0)
        if not held.is_none:
            bitmap = held[0]
            # Preserve taps that begin and end between input snapshots for one frame.
            pressed = inputs["keyboard_pressed"]
            if not pressed.is_none:
                bitmap = np.logical_or(bitmap, pressed[0])
            if self._repeat_grace_ns > 0 and running:
                now = context.graph_time.real_time_ns
                if self._pressed_until is None:
                    self._pressed_until = np.zeros(bitmap.shape, dtype=np.int64)
                self._pressed_until[np.asarray(bitmap, dtype=bool)] = now + self._repeat_grace_ns
                bitmap = np.logical_or(bitmap, self._pressed_until > now)
            forward = float(bitmap[EvdevKeyCode.KeyW]) - float(bitmap[EvdevKeyCode.KeyS])
            left = float(bitmap[EvdevKeyCode.KeyA]) - float(bitmap[EvdevKeyCode.KeyD])
            yaw = float(bitmap[EvdevKeyCode.ArrowLeft]) - float(bitmap[EvdevKeyCode.ArrowRight])
            command[0] = forward * self._config.v_x_sensitivity
            command[1] = left * self._config.v_y_sensitivity
            command[2] = yaw * self._config.omega_z_sensitivity
        outputs["base_command"][0] = command


def wasd_keyboard_cfg(
    v_x_sensitivity: float = 0.8,
    v_y_sensitivity: float = 0.4,
    omega_z_sensitivity: float = 1.0,
    sim_device: str = "cpu",
    repeat_grace_s: float = 0.0,
) -> IsaacTeleopCfg:
    """Build WASD playback, optionally bridging remote key pulses for ``repeat_grace_s`` seconds.

    A nonzero grace also delays stopping by up to that duration after the last key pulse.
    """
    if not math.isfinite(repeat_grace_s) or repeat_grace_s < 0.0:
        raise ValueError("repeat_grace_s must be finite and nonnegative.")
    cfg = se2_keyboard_teleop_cfg(v_x_sensitivity, v_y_sensitivity, omega_z_sensitivity, sim_device)

    def build_pipeline() -> OutputCombiner:
        source = KeyboardSource("keyboard")
        motion = _WasdKeyboard(
            KeyboardToSe2RetargeterConfig(v_x_sensitivity, v_y_sensitivity, omega_z_sensitivity),
            name="wasd",
            repeat_grace_s=repeat_grace_s,
        ).connect(
            {
                "keyboard_held": source.output("keyboard_held"),
                "keyboard_pressed": source.output("keyboard_pressed"),
            }
        )
        return OutputCombiner({"action": motion.output("base_command")})

    cfg.pipeline_builder = build_pipeline
    return cfg
