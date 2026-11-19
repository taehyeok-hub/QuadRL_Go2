# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2의 자세 붕괴가 접촉 없이도 에피소드 종료로 연결되는지 검증함."""

import math
from collections import UserDict
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import TerminationManager

from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg


@pytest.mark.parametrize("config_type", [UnitreeGo2FlatEnvCfg, UnitreeGo2RoughEnvCfg])
def test_go2_terminates_only_beyond_horizontal_without_base_contact(config_type):
    """큰 기울기는 허용하며 90도 초과를 timeout과 구분되는 실제 종료로 처리함."""
    cfg = config_type()
    # base 접촉 없이 새 자세 조건만으로 종료되는 상황을 재현함.
    cfg.terminations.base_contact = None
    if hasattr(cfg.terminations, "assigned_terrain_boundary"):
        cfg.terminations.assigned_terrain_boundary = None
    angles = torch.tensor([0.0, 45.0, 89.9, 90.0, 90.1, 180.0], dtype=torch.float64) * math.pi / 180.0
    gravity = torch.stack((angles.sin(), torch.zeros_like(angles), -angles.cos()), dim=-1)
    gravity[3] = torch.tensor([1.0, 0.0, 0.0], dtype=gravity.dtype)
    robot = SimpleNamespace(data=SimpleNamespace(projected_gravity_b=SimpleNamespace(torch=gravity)))
    env = SimpleNamespace(
        scene=UserDict({"robot": robot}),
        num_envs=len(angles),
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        episode_length_buf=torch.zeros(len(angles), dtype=torch.long),
        max_episode_length=1000,
    )
    manager = TerminationManager(cfg.terminations, env)
    expected = torch.tensor([False, False, False, False, True, True])

    assert torch.equal(manager.compute(), expected)
    assert torch.equal(manager.terminated, expected)
    assert not manager.time_outs.any()
