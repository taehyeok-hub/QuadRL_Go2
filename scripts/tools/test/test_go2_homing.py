# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior tests for Go2 homing targets without a simulation fixture."""

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


def test_homing_sequence_uses_each_measured_start_and_holds_the_final_pose():
    """Each cosine move uses its own duration and copied measured start before holding the final target."""
    initial = torch.tensor([[0.0, 0.0], [0.2, 0.4]])
    prepare_position = torch.tensor([1.0, 2.0])
    homing_position = torch.tensor([3.0, 4.0])
    controller = Go2HomingController(initial, prepare_position, homing_position, prepare_period=4, homing_period=2)
    measured = initial.clone()

    torch.testing.assert_close(controller.homing(measured), initial)
    measured.add_(10.0)
    torch.testing.assert_close(
        controller.homing(measured), torch.tensor([[0.14644661, 0.29289322], [0.31715729, 0.63431458]])
    )
    torch.testing.assert_close(controller.homing(measured), torch.tensor([[0.5, 1.0], [0.6, 1.2]]))
    controller.homing(measured)
    torch.testing.assert_close(controller.homing(measured), prepare_position.expand(2, -1))

    # The second move starts from the measured pose even when it lags behind the prepare target.
    measured.copy_(torch.tensor([[0.8, 1.6], [0.6, 1.2]]))
    torch.testing.assert_close(controller.homing(measured), torch.tensor([[0.8, 1.6], [0.6, 1.2]]))
    measured.add_(10.0)
    torch.testing.assert_close(controller.homing(measured), torch.tensor([[1.9, 2.8], [1.8, 2.6]]))
    torch.testing.assert_close(controller.homing(measured), homing_position.expand(2, -1))
    torch.testing.assert_close(controller.homing(measured), homing_position.expand(2, -1))
