# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior tests for timed Go2 homing targets without a simulation fixture."""

import torch

from scripts.tools.go2_homing import Go2HomingController, go2_joint_pose


def test_go2_homing_poses_follow_joint_names_instead_of_storage_order():
    """Each independently configured joint must receive its angle in articulation order."""
    joint_names = [
        "RR_calf_joint",
        "FL_hip_joint",
        "RL_thigh_joint",
        "FR_calf_joint",
        "RR_hip_joint",
        "FL_thigh_joint",
        "RL_calf_joint",
        "FR_hip_joint",
        "RR_thigh_joint",
        "FL_calf_joint",
        "RL_hip_joint",
        "FR_thigh_joint",
    ]
    reference = torch.zeros(2, 12, dtype=torch.float64)

    positions = {
        "FL": (0.1, 1.1, -2.1),
        "FR": (0.2, 1.2, -2.2),
        "RL": (0.3, 1.3, -2.3),
        "RR": (0.4, 1.4, -2.4),
    }
    prepare_position = go2_joint_pose(joint_names, positions, reference)

    torch.testing.assert_close(
        prepare_position,
        torch.tensor([-2.4, 0.1, 1.3, -2.2, 0.4, 1.1, -2.3, 0.2, 1.4, -2.1, 0.3, 1.2], dtype=torch.float64),
    )


def test_homing_sequence_interpolates_in_seconds_and_holds_each_endpoint():
    """Each environment moves from its own measured pose before sharing the two targets."""
    initial = torch.tensor([[0.0, 0.0], [0.2, 0.4]])
    prepare_position = torch.tensor([1.0, 2.0])
    homing_position = torch.tensor([3.0, 4.0])
    controller = Go2HomingController(initial, prepare_position, homing_position, move_time=2.0, hold_time=1.0)

    torch.testing.assert_close(controller.compute(1.0), torch.tensor([[0.5, 1.0], [0.6, 1.2]]))
    torch.testing.assert_close(controller.compute(1.0), prepare_position.expand(2, -1))
    assert controller.stage.tolist() == [1, 1]
    torch.testing.assert_close(controller.compute(0.5), prepare_position.expand(2, -1))
    assert controller.stage.tolist() == [1, 1]
    torch.testing.assert_close(controller.compute(0.5), prepare_position.expand(2, -1))
    assert controller.stage.tolist() == [2, 2]
    torch.testing.assert_close(controller.compute(1.0), torch.tensor([[2.0, 3.0], [2.0, 3.0]]))
    torch.testing.assert_close(controller.compute(1.0), homing_position.expand(2, -1))
    assert controller.stage.tolist() == [3, 3]
    torch.testing.assert_close(controller.compute(20.0), homing_position.expand(2, -1))


def test_single_pose_holds_until_restarted_from_new_measured_positions():
    """Selecting one pose stops the sequence; a later request starts from the measured state."""
    prepare_position = torch.tensor([1.0, 2.0])
    homing_position = torch.tensor([3.0, 4.0])
    controller = Go2HomingController(torch.zeros(2, 2), prepare_position, homing_position, move_time=2.0, hold_time=0.5)
    measured = torch.tensor([[0.0, 0.0], [0.2, 0.4]])
    controller.start(measured, pose="prepare_position")
    measured.add_(10.0)

    torch.testing.assert_close(controller.compute(1.0), torch.tensor([[0.5, 1.0], [0.6, 1.2]]))
    torch.testing.assert_close(controller.compute(1.0), prepare_position.expand(2, -1))
    torch.testing.assert_close(controller.compute(20.0), prepare_position.expand(2, -1))

    controller.start(torch.tensor([[2.0, 2.0], [4.0, 6.0]]), pose="homing_position")
    torch.testing.assert_close(controller.compute(1.0), torch.tensor([[2.5, 3.0], [3.5, 5.0]]))
    torch.testing.assert_close(controller.compute(1.0), homing_position.expand(2, -1))
    torch.testing.assert_close(controller.compute(20.0), homing_position.expand(2, -1))
