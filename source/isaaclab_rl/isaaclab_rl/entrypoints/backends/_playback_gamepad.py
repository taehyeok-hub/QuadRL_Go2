# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SE(2) playback input with the gamepad's north (Xbox Y) button appended."""

from isaacteleop.retargeters import TensorReorderer
from isaacteleop.retargeting_engine.deviceio_source_nodes import GamepadSource, find_sources
from isaacteleop.retargeting_engine.interface import (
    BaseRetargeter,
    ComputeContext,
    OutputCombiner,
    RetargeterIO,
    RetargeterIOType,
    TensorGroupType,
)
from isaacteleop.retargeting_engine.interface.tensor_group_type import OptionalType
from isaacteleop.retargeting_engine.tensor_types import FloatType, GamepadInput, GamepadInputIndex

from isaaclab_teleop import IsaacTeleopCfg, se2_gamepad_teleop_cfg


class _CameraButton(BaseRetargeter):
    """Read Y independently of the velocity retargeter's stick bindings."""

    def input_spec(self) -> RetargeterIOType:
        return {"gamepad": OptionalType(GamepadInput())}

    def output_spec(self) -> RetargeterIOType:
        return {"camera": TensorGroupType("camera", [FloatType("pressed")])}

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context: ComputeContext) -> None:
        gamepad = inputs["gamepad"]
        outputs["camera"][0] = 0.0 if gamepad.is_none else float(gamepad[GamepadInputIndex.NORTH_CLICK])


def gamepad_with_camera_cfg(**kwargs) -> IsaacTeleopCfg:
    """Extend the standard gamepad pipeline with a fourth, camera-button value."""
    cfg = se2_gamepad_teleop_cfg(**kwargs)
    build_motion = cfg.pipeline_builder

    def build_pipeline() -> OutputCombiner:
        motion = build_motion()
        source = find_sources(motion, GamepadSource)[0]
        camera = _CameraButton(name="playback_camera").connect({"gamepad": source.output("gamepad")})
        action = TensorReorderer(
            input_config={"velocity": ["vx", "vy", "wz"], "camera": ["pressed"]},
            output_order=["vx", "vy", "wz", "pressed"],
            input_types={"velocity": "array", "camera": "scalar"},
            name="playback_command",
        ).connect({"velocity": motion.output("action"), "camera": camera.output("camera")})
        return OutputCombiner({"action": action.output("output")})

    cfg.pipeline_builder = build_pipeline
    return cfg
