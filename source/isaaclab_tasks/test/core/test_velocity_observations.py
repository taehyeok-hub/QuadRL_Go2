# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for the shared asymmetric velocity observations."""

from collections import UserDict
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ObservationManager

from isaaclab_tasks.core.velocity.velocity_env_cfg import ObservationsCfg


@pytest.mark.parametrize("height_scan", [False, True])
def test_velocity_observations_separate_noisy_actor_and_privileged_state(height_scan):
    """Keep simulator velocity and terrain scans out of the actor, and noise out of the critic."""
    cfg = ObservationsCfg()
    if not height_scan:
        cfg.privileged_obs.height_scan = None

    def proxy(value):
        return SimpleNamespace(torch=value)

    data = SimpleNamespace(
        root_ang_vel_b=proxy(torch.tensor([[0.1, 0.2, 0.3]]).repeat(2, 1)),
        projected_gravity_b=proxy(torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)),
        root_lin_vel_b=proxy(torch.tensor([[5.0, 6.0, 7.0]]).repeat(2, 1)),
        joint_pos=proxy(torch.zeros(2, 12)),
        default_joint_pos=proxy(torch.zeros(2, 12)),
        joint_vel=proxy(torch.zeros(2, 12)),
        default_joint_vel=proxy(torch.zeros(2, 12)),
    )
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            pos_w=proxy(torch.tensor([[0.0, 0.0, 0.7]]).repeat(2, 1)), ray_hits_w=proxy(torch.zeros(2, 187, 3))
        )
    )
    scene = UserDict({"robot": SimpleNamespace(data=data), "height_scanner": scanner})
    scene.sensors = {"height_scanner": scanner}
    commands = torch.tensor([[0.2, -0.1, 0.4]]).repeat(2, 1)
    actions = torch.arange(12, dtype=torch.float).repeat(2, 1)
    env = SimpleNamespace(
        scene=scene,
        num_envs=2,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        command_manager=SimpleNamespace(get_command=lambda name: commands),
        action_manager=SimpleNamespace(action=actions),
    )
    manager = ObservationManager(cfg, env)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10)
        original = manager.compute()
        torch.manual_seed(20)
        noisy = manager.compute()
        data.root_lin_vel_b.torch.add_(10.0)
        torch.manual_seed(10)
        changed_velocity = manager.compute()
        if height_scan:
            scanner.data.ray_hits_w.torch[..., 2].add_(0.1)
            torch.manual_seed(10)
            changed_terrain = manager.compute()

    assert original["actor_obs"].shape == original["critic_base_obs"].shape == (2, 45)
    assert original["privileged_obs"].shape == (2, 190 if height_scan else 3)
    assert not torch.equal(original["actor_obs"], noisy["actor_obs"])
    torch.testing.assert_close(original["critic_base_obs"], noisy["critic_base_obs"], rtol=0, atol=0)
    torch.testing.assert_close(original["privileged_obs"], noisy["privileged_obs"], rtol=0, atol=0)
    torch.testing.assert_close(original["actor_obs"][:, 6:9], commands, rtol=0, atol=0)
    torch.testing.assert_close(original["actor_obs"][:, -12:], actions, rtol=0, atol=0)
    torch.testing.assert_close(original["critic_base_obs"][:, :3], data.root_ang_vel_b.torch, rtol=0, atol=0)
    torch.testing.assert_close(changed_velocity["actor_obs"], original["actor_obs"], rtol=0, atol=0)
    torch.testing.assert_close(changed_velocity["critic_base_obs"], original["critic_base_obs"], rtol=0, atol=0)
    torch.testing.assert_close(changed_velocity["privileged_obs"][:, :3], data.root_lin_vel_b.torch, rtol=0, atol=0)
    if height_scan:
        torch.testing.assert_close(changed_velocity["privileged_obs"][:, 3:], torch.full((2, 187), 0.2))
        torch.testing.assert_close(changed_terrain["actor_obs"], original["actor_obs"], rtol=0, atol=0)
        torch.testing.assert_close(changed_terrain["critic_base_obs"], original["critic_base_obs"], rtol=0, atol=0)
        torch.testing.assert_close(changed_terrain["privileged_obs"][:, 3:], torch.full((2, 187), 0.1))
