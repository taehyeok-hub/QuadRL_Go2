# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardTermCfg, SceneEntityCfg

from isaaclab_tasks.core.velocity import mdp


@pytest.fixture
def commands():
    """Commands on either side of a custom 0.1 activation threshold."""
    return torch.tensor(
        [
            [0.2, 0.0, 0.0],  # forward
            [0.0, -0.2, 0.0],  # lateral
            [0.08, 0.08, 0.0],  # planar speed exceeds the threshold
            [0.0, 0.0, 0.2],  # positive yaw only
            [0.0, 0.0, -0.2],  # negative yaw only
            [0.0, 0.0, 0.0],  # standing
            [0.1, 0.0, 0.0],  # linear threshold
            [0.0, 0.0, 0.1],  # positive yaw threshold
            [0.0, 0.0, -0.1],  # negative yaw threshold
            [0.0, 0.0, 0.09],  # subthreshold yaw
            [0.08, 0.0, 0.08],  # linear and angular thresholds apply separately
        ]
    )


@pytest.mark.parametrize(
    ("reward_func", "params", "active_value"),
    [
        pytest.param(mdp.feet_air_time, {"threshold": 0.5}, 0.2, id="air-time"),
        pytest.param(mdp.feet_air_time_positive_biped, {"threshold": 0.5}, 0.3, id="biped-air-time"),
        pytest.param(mdp.feet_air_time_variance, {}, 0.04, id="air-time-variance"),
    ],
)
def test_feet_air_time_terms_activate_for_translation_or_turning(commands, reward_func, params, active_value):
    """Turning in either direction enables each term; zero and threshold commands keep it disabled."""

    def timing(values):
        return SimpleNamespace(torch=torch.tensor(values).repeat(len(commands), 1))

    first_contact = timing([True, False])
    sensor = SimpleNamespace(
        compute_first_contact=lambda dt: first_contact,
        data=SimpleNamespace(
            last_air_time=timing([0.7, 0.3]),
            last_contact_time=timing([0.1, 0.3]),
            current_air_time=timing([0.3, 0.0]),
            current_contact_time=timing([0.0, 0.4]),
        ),
    )
    env = SimpleNamespace(
        step_dt=0.02,
        scene=SimpleNamespace(sensors={"contact_forces": sensor}),
        command_manager=SimpleNamespace(get_command={"base_velocity": commands}.__getitem__),
    )

    value = reward_func(
        env,
        command_name="base_velocity",
        sensor_cfg=SceneEntityCfg("contact_forces"),
        command_threshold=0.1,
        **params,
    )

    expected = torch.tensor([active_value] * 5 + [0.0] * 6)
    torch.testing.assert_close(value, expected)


_DEFAULT_FOOT_POSITIONS_XY = [(0.2, 0.1), (0.2, -0.1), (-0.2, 0.1), (-0.2, -0.1)]


def test_feet_clearance_uses_plane_relative_link_geometry_and_motion_gates():
    """Only moving foot links contribute their squared height error times actual XY speed."""
    commands = torch.tensor(
        [[0.2, 0.0, 0.0]] * 5
        + [
            [0.0, 0.0, 0.0],  # standing
            [0.0, 0.0, 0.2],  # positive yaw only
            [0.0, 0.0, -0.2],  # negative yaw only
            [0.05, 0.0, 0.0],  # planar command threshold
            [0.0, 0.0, 0.05],  # positive yaw threshold
            [0.0, 0.0, -0.05],  # negative yaw threshold
            [0.04, 0.04, 0.0],  # planar norm exceeds the threshold
            [0.04, 0.0, 0.04],  # planar and angular thresholds apply separately
        ]
    )
    body_ids = [4, 1, 5, 2]
    positions = torch.zeros(len(commands), 6, 3)
    positions[:, :, 2] = 1.7
    positions[:, body_ids, 2] = 0.78
    positions[:, 4, 2] = 0.88
    velocities = torch.ones_like(positions)
    velocities[:, body_ids, :] = 0.0
    velocities[:, 4, 0] = 0.05

    velocities[0, 4, 0] = 0.049
    velocities[2, 4, :] = torch.tensor([0.03, 0.04, 3.0])
    positions[3, 4, 2] = 0.78
    # Mixed feet: 0.1 m error at 0.05 m/s, target height, 0.2 m error at 0.1 m/s,
    # and a foot moving vertically without horizontal motion.
    velocities[4, 4, :] = torch.tensor([0.03, 0.04, 0.0])
    velocities[4, 1, :] = torch.tensor([0.3, 0.4, 0.0])
    positions[4, 5, 2] = 0.98
    velocities[4, 5, 0] = 0.1
    positions[4, 2, 2] = 0.88
    velocities[4, 2, 2] = 0.5

    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=SimpleNamespace(torch=positions),
            body_link_lin_vel_w=SimpleNamespace(torch=velocities),
            body_pos_w=SimpleNamespace(torch=torch.zeros_like(positions)),
            body_lin_vel_w=SimpleNamespace(torch=torch.zeros_like(velocities)),
        )
    )
    env = SimpleNamespace(
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_command={"base_velocity": commands}.__getitem__),
    )

    cost = mdp.feet_clearance(
        env,
        command_name="base_velocity",
        target_height=0.08,
        min_foot_speed=0.05,
        plane_z=0.7,
        asset_cfg=SceneEntityCfg("robot", body_ids=body_ids),
    )

    torch.testing.assert_close(
        cost,
        torch.tensor([0.0, 0.0005, 0.0005, 0.0, 0.0045, 0.0, 0.0005, 0.0005, 0.0, 0.0, 0.0, 0.0005, 0.0]),
    )


def _command_gate_env(commands):
    forces = torch.zeros(len(commands), 4, 3)
    forces[:, [0, 3], 2] = 2.0

    def timing(values):
        return SimpleNamespace(torch=torch.tensor(values).repeat(len(commands), 1))

    sensor = SimpleNamespace(
        compute_first_contact=lambda dt: timing([True, False, False, False]),
        data=SimpleNamespace(
            net_forces_w=SimpleNamespace(torch=forces),
            net_normal_forces_w=SimpleNamespace(torch=forces.clone()),
            last_air_time=timing([0.7, 0.3, 0.7, 0.3]),
            last_contact_time=timing([0.1, 0.3, 0.1, 0.3]),
            current_air_time=timing([0.3, 0.0, 0.3, 0.0]),
            current_contact_time=timing([0.0, 0.4, 0.0, 0.4]),
        ),
    )
    asset = SimpleNamespace(
        joint_names=["hip_joint"],
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.full((len(commands), 1), 0.1)),
            default_joint_pos=SimpleNamespace(torch=torch.zeros(len(commands), 1)),
            joint_vel=SimpleNamespace(torch=torch.full((len(commands), 1), 2.0)),
            root_pos_w=SimpleNamespace(torch=torch.zeros(len(commands), 3)),
            root_quat_w=timing([0.0, 0.0, 0.0, 1.0]),
            body_link_pos_w=SimpleNamespace(
                torch=torch.tensor([[x + 1.0, y, -0.3] for x, y in _DEFAULT_FOOT_POSITIONS_XY])
                .unsqueeze(0)
                .repeat(len(commands), 1, 1)
            ),
        ),
    )

    class Scene(dict):
        sensors = {"contact_forces": sensor}

    return SimpleNamespace(
        num_envs=len(commands),
        device="cpu",
        step_dt=0.02,
        scene=Scene(robot=asset),
        command_manager=SimpleNamespace(get_command={"base_velocity": commands}.__getitem__),
    )


def test_standing_and_trot_terms_use_custom_command_threshold(commands):
    """One threshold controls standing contacts, posture tolerance, and moving gait rewards."""
    env = _command_gate_env(commands)
    sensor_cfg = SceneEntityCfg("contact_forces")

    contact_penalty = mdp.stand_still_contact(env, "base_velocity", sensor_cfg, command_threshold=0.1)
    torch.testing.assert_close(contact_penalty, torch.tensor([0.0] * 5 + [4.0] + [0.0] * 3 + [4.0] * 2))

    standing = torch.tensor([0.0] * 5 + [1.0] + [0.0] * 3 + [1.0] * 2)
    for reward_func, params, active_value in (
        (mdp.stand_still_joint_deviation_l1, {}, 0.1),
        (mdp.stand_still_foot_position_l2, {"default_foot_positions": _DEFAULT_FOOT_POSITIONS_XY}, 1.0),
        (mdp.stand_still_joint_velocity, {"velocity_sigma": 1.0}, 4.0),
    ):
        penalty = reward_func(
            env,
            command_name="base_velocity",
            asset_cfg=SceneEntityCfg("robot"),
            command_threshold=0.1,
            **params,
        )
        torch.testing.assert_close(penalty, active_value * standing)

    trot = mdp.trot_gait_pattern(RewardTermCfg(func=mdp.trot_gait_pattern, weight=1.0, params={"ema_alpha": 0.99}), env)
    trot_reward = trot(env, "base_velocity", sensor_cfg, balance_weight=0.0, command_threshold=0.1)
    torch.testing.assert_close(trot_reward, torch.tensor([1.0] * 5 + [0.0] * 6))

    posture_cfg = RewardTermCfg(
        func=mdp.variable_position,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "std_standing": {"hip_joint": 0.1},
            "std_moving": {"hip_joint": 0.2},
            "command_threshold": 0.1,
        },
    )
    posture = mdp.variable_position(posture_cfg, env)
    posture_reward = posture(env, **posture_cfg.params)
    expected_posture = torch.tensor([0.77880078] * 5 + [0.36787944] + [0.77880078] * 3 + [0.36787944] * 2)
    torch.testing.assert_close(posture_reward, expected_posture)


def test_locomotion_terms_use_default_command_threshold():
    """The default 0.05 threshold treats both yaw signs and planar speed consistently."""
    commands = torch.tensor(
        [
            [0.0, 0.0, 0.04],
            [0.0, 0.0, -0.04],
            [0.0, 0.0, 0.05],
            [0.0, 0.0, -0.05],
            [0.0, 0.0, 0.06],
            [0.0, 0.0, -0.06],
            [0.06, 0.0, 0.0],
            [0.04, 0.04, 0.0],
            [0.04, 0.0, 0.04],
            [0.0, 0.0, 0.0],
        ]
    )
    env = _command_gate_env(commands)
    moving = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    standing = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    for reward_func, params, active_value in (
        (mdp.feet_air_time, {"threshold": 0.5}, 0.2),
        (mdp.feet_air_time_positive_biped, {"threshold": 0.5}, 0.3),
        (mdp.feet_air_time_variance, {}, 0.04),
    ):
        reward = reward_func(
            env,
            command_name="base_velocity",
            sensor_cfg=SceneEntityCfg("contact_forces", body_ids=[0, 1]),
            **params,
        )
        torch.testing.assert_close(reward, active_value * moving)

    trot = mdp.trot_gait_pattern(RewardTermCfg(func=mdp.trot_gait_pattern, weight=1.0, params={"ema_alpha": 0.99}), env)
    torch.testing.assert_close(trot(env, "base_velocity", SceneEntityCfg("contact_forces"), balance_weight=0.0), moving)

    contact = mdp.stand_still_contact(env, "base_velocity", SceneEntityCfg("contact_forces"))
    torch.testing.assert_close(contact, 4.0 * standing)
    for reward_func, params, active_value in (
        (mdp.stand_still_joint_deviation_l1, {}, 0.1),
        (mdp.stand_still_foot_position_l2, {"default_foot_positions": _DEFAULT_FOOT_POSITIONS_XY}, 1.0),
        (mdp.stand_still_joint_velocity, {"velocity_sigma": 1.0}, 4.0),
    ):
        penalty = reward_func(env, "base_velocity", asset_cfg=SceneEntityCfg("robot"), **params)
        torch.testing.assert_close(penalty, active_value * standing)

    posture_cfg = RewardTermCfg(
        func=mdp.variable_position,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "std_standing": {"hip_joint": 0.1},
            "std_moving": {"hip_joint": 0.2},
        },
    )
    posture = mdp.variable_position(posture_cfg, env)
    torch.testing.assert_close(
        posture(env, **posture_cfg.params),
        torch.tensor([0.36787944] * 2 + [0.77880078] * 6 + [0.36787944] * 2),
    )


def _timing_env(air_time, contact_time, commands):
    sensor = SimpleNamespace(
        data=SimpleNamespace(
            current_air_time=SimpleNamespace(torch=torch.tensor(air_time)),
            current_contact_time=SimpleNamespace(torch=torch.tensor(contact_time)),
        )
    )
    return SimpleNamespace(
        num_envs=len(commands),
        device="cpu",
        step_dt=0.02,
        scene=SimpleNamespace(sensors={"contact_forces": sensor}),
        command_manager=SimpleNamespace(get_command={"base_velocity": commands}.__getitem__),
    )


def test_gait_timing_rewards_turning_steps_without_locking_a_support_pattern(commands):
    """Short diagonal steps beat persistent 3/1 support, with time for contact transitions."""
    air_time = [[0.0, 0.2, 0.2, 0.0]] * len(commands) + [
        [0.2, 0.0, 0.0, 0.2],  # opposite diagonal in stance
        [0.0, 0.0, 0.0, 0.2],  # three feet in stance
        [0.0, 0.2, 0.2, 0.2],  # one foot in stance
        [0.0, 0.8, 0.8, 0.0],  # one fixed diagonal held too long
        [0.0, 0.02, 0.0, 0.0],  # brief three-foot overlap
        [0.0, 0.0, 0.0, 0.0],  # brief four-foot overlap
        [0.2, 0.2, 0.2, 0.2],  # fully airborne
    ]
    contact_time = [[0.2, 0.0, 0.0, 0.2]] * len(commands) + [
        [0.0, 0.2, 0.2, 0.0],
        [0.2, 0.2, 0.2, 0.0],
        [0.2, 0.0, 0.0, 0.0],
        [0.8, 0.0, 0.0, 0.8],
        [0.02, 0.0, 0.02, 0.02],
        [0.02, 0.02, 0.02, 0.02],
        [0.0, 0.0, 0.0, 0.0],
    ]
    commands = torch.cat((commands, torch.tensor([[0.0, 0.0, 0.2]] * 7)))
    env = _timing_env(air_time, contact_time, commands)

    value = mdp.feet_gait_timing(env, "base_velocity", SceneEntityCfg("contact_forces"), command_threshold=0.1)

    torch.testing.assert_close(value[:11], torch.tensor([1.0] * 5 + [0.0] * 6))
    diagonal, three_stance, one_stance, fixed_diagonal, short_three, short_four, airborne = value[11:]
    assert diagonal == 1.0
    assert three_stance < 0.1
    assert one_stance < 0.1
    assert fixed_diagonal < 0.01
    assert short_three > 0.9
    assert short_four > 0.9
    assert airborne == -1.0


def test_flat_gait_rewards_prefer_short_turning_steps_to_a_three_foot_pivot():
    """Configured gait terms must not make a short diagonal touchdown worse than not stepping."""
    from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

    cfg = UnitreeGo2FlatEnvCfg()
    env = _timing_env(
        [[0.0, 0.02, 0.02, 0.0], [0.0, 0.0, 0.0, 1.0]],
        [[0.02, 0.0, 0.0, 0.02], [1.0, 1.0, 1.0, 0.0]],
        torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]]),
    )
    sensor = env.scene.sensors["contact_forces"]
    sensor.data.last_air_time = SimpleNamespace(torch=torch.full((2, 4), 0.2))
    sensor.data.last_contact_time = SimpleNamespace(torch=torch.full((2, 4), 0.2))
    sensor.compute_first_contact = lambda dt: SimpleNamespace(
        torch=torch.tensor([[True, False, False, True], [False, False, False, False]])
    )
    forces = torch.zeros(2, 4, 3)
    forces[0, [0, 3], 2] = 2.0
    forces[1, [0, 1, 2], 2] = 2.0
    sensor.data.net_normal_forces_w = SimpleNamespace(torch=forces)
    reward = torch.zeros(2)
    for name in ("feet_air_time", "feet_air_time_variance", "trot_gait_pattern", "gait_timing"):
        term = getattr(cfg.rewards, name, None)
        if term is None or term.weight == 0.0:
            continue
        func = term.func(term, env) if isinstance(term.func, type) else term.func
        reward += term.weight * func(env, **term.params)

    assert reward[0] > reward[1], "A short turning step must earn more gait reward than a stationary 3/1 pivot."
