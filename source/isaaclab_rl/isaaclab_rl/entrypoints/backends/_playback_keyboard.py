# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playback input using w/a/s/d for translation and left/right arrows for yaw."""

import numpy as np

from isaacteleop.deviceio_trackers import EvdevKeyCode
from isaacteleop.retargeters import KeyboardToSe2Retargeter, KeyboardToSe2RetargeterConfig
from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource
from isaacteleop.retargeting_engine.interface import ComputeContext, OutputCombiner, RetargeterIO

from isaaclab_teleop import IsaacTeleopCfg, se2_keyboard_teleop_cfg


class _WasdKeyboard(KeyboardToSe2Retargeter):
    """Read physical w/a/s/d and arrow keys independently of letter case."""

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context: ComputeContext) -> None:
        command = np.zeros(3, dtype=np.float32)
        held = inputs["keyboard_held"]
        if not held.is_none:
            bitmap = held[0]
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
) -> IsaacTeleopCfg:
    """Build playback's w/a/s/d movement and left/right-arrow rotation pipeline."""
    cfg = se2_keyboard_teleop_cfg(v_x_sensitivity, v_y_sensitivity, omega_z_sensitivity, sim_device)

    def build_pipeline() -> OutputCombiner:
        source = KeyboardSource("keyboard")
        motion = _WasdKeyboard(
            KeyboardToSe2RetargeterConfig(v_x_sensitivity, v_y_sensitivity, omega_z_sensitivity), name="wasd"
        ).connect({"keyboard_held": source.output("keyboard_held")})
        return OutputCombiner({"action": motion.output("base_command")})

    cfg.pipeline_builder = build_pipeline
    return cfg
