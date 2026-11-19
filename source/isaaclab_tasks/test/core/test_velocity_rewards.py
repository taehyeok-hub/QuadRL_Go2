# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import RewardManager, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.string import resolve_matching_names

from isaaclab_tasks.core.velocity import mdp


def test_yaw_footplacement_matches_quadflat_frames_gates_and_weighting():
    """회전 전용 발 속도 비용의 좌표계, 발 선택, 명령 경계와 가중치·시간 배율을 검증함."""
    num_envs = 12
    commands = torch.tensor([[0.0, 0.0, 0.2]] * num_envs)
    commands[4] = torch.tensor([0.099, 0.0, 0.2])
    commands[5] = torch.tensor([0.08, 0.08, 0.2])
    commands[6] = torch.tensor([0.1, 0.0, 0.2])
    commands[7, 2] = 0.1
    commands[8, 2] = -0.2
    commands[9] = 0.0
    commands[10] = torch.tensor([0.2, 0.0, 0.2])
    commands[11, 2] = -0.1

    body_names = ["base", "RR_foot", "FL_foot", "RL_foot", "FR_foot"]
    velocity = torch.zeros(num_envs, len(body_names), 3)
    velocity[:, 0] = 100.0  # 발이 아닌 링크의 속도는 합산하지 않아야 함.
    velocity[:, 2] = torch.tensor([1.0, 2.0, -3.0])
    velocity[0, 1] = torch.tensor([-2.0, 0.0, 1.0])  # 여러 발의 비용을 합산해야 함.
    velocity[1, 2] = torch.tensor([1.0, 0.0, 2.0])
    velocity[2, 2] = torch.tensor([1.0, 0.0, 1.0])
    velocity[3, 2] = torch.tensor([0.0, 5.0, 0.0])
    roll = torch.zeros(num_envs)
    pitch = torch.zeros(num_envs)
    yaw = torch.zeros(num_envs)
    pitch[2] = torch.pi / 4  # pitch를 제거하면 원본과 다른 비용을 계산하게 됨.
    yaw[1] = torch.pi / 2
    asset = SimpleNamespace(
        num_bodies=len(body_names),
        body_names=body_names,
        find_bodies=lambda names, preserve_order: resolve_matching_names(names, body_names, preserve_order),
        data=SimpleNamespace(
            body_link_lin_vel_w=SimpleNamespace(torch=velocity),
            root_quat_w=SimpleNamespace(torch=quat_from_euler_xyz(roll, pitch, yaw)),
            root_lin_vel_w=SimpleNamespace(torch=velocity[:, 2].clone()),
            root_lin_vel_b=SimpleNamespace(torch=velocity[:, 2].clone()),
        ),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_command={"base_velocity": commands}.__getitem__),
    )
    manager = RewardManager(
        {
            "yaw_footplacement": RewardTermCfg(
                func=mdp.yaw_footplacement,
                weight=-0.5,
                params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
            )
        },
        env,
    )

    torch.testing.assert_close(
        manager.compute(0.02),
        torch.tensor([-0.07, -0.02, -0.02, 0.0, -0.04, 0.0, 0.0, 0.0, -0.04, 0.0, 0.0, 0.0]),
        atol=1.0e-7,
        rtol=0.0,
    )


def test_go2_flat_foot_separation_matches_quadflat_geometry_and_weighting():
    """실제 평지 설정의 네 발 선택, 수평 네 쌍의 선형 비용과 가중치·시간 배율을 검증함."""
    from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

    cfg = UnitreeGo2FlatEnvCfg()
    # 각 행은 FL, FR, RL, RR 순서이며 기대값은 좌표에서 독립적으로 계산함.
    feet_position = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.15, 0.0], [0.15, 0.15, 0.0]],  # 임계값에서 비용 0임.
            [[0.0, 0.0, 0.0]] * 4,  # 네 쌍이 모두 겹치므로 비용은 4 * 0.15임.
            [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],  # 앞 쌍만 0.10임.
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.10, 1.0, 0.0]],  # 뒤 쌍만 0.05임.
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.03, 0.0], [1.0, 0.08, 0.0]],  # 좌우 합은 0.12 + 0.07임.
            [[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]],  # 대각짝 겹침은 평가하지 않음.
            [[0.0, 0.0, -1.0], [1.0, 0.0, 1.0], [0.0, 0.03, 3.0], [1.0, 0.08, -2.0]],  # 높이는 비용과 무관함.
            [[2.0, -1.0, 0.0], [2.0, 0.0, 0.0], [1.97, -1.0, 0.0], [1.92, 0.0, 0.0]],  # 90도 회전·이동한 4번 행임.
            [[0.0, 0.0, 0.0], [0.06, 0.08, 0.0], [0.0, 0.1, 0.0], [0.06, 0.18, 0.0]],  # 네 변이 0.1인 평행사변형임.
        ]
    )
    num_envs = len(feet_position)
    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    body_names = ["FR_thigh", "RR_foot", "base", "FL_foot", "RL_foot", "FR_foot"]
    positions = torch.full((num_envs, len(body_names), 3), 100.0)
    positions[:, [body_names.index(name) for name in feet]] = feet_position
    asset = SimpleNamespace(
        num_bodies=len(body_names),
        body_names=body_names,
        find_bodies=lambda names, preserve_order: resolve_matching_names(names, body_names, preserve_order),
        data=SimpleNamespace(body_link_pos_w=SimpleNamespace(torch=positions)),
    )

    # 같은 수평 배치의 4·6·7번 행에서 이동·정지·회전 명령과 접촉 상태를 다르게 제공함.
    commands = torch.zeros(num_envs, 3)
    commands[4, 0] = 1.0
    commands[7, 2] = 0.5
    forces = torch.zeros(num_envs, 4, 3)
    forces[6, :, 2] = 100.0
    forces[7, [0, 3], 2] = 100.0
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=SimpleNamespace(torch=forces)))

    class Scene(dict):
        sensors = {"contact_forces": sensor}

    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene=Scene(robot=asset, contact_forces=sensor),
        command_manager=SimpleNamespace(get_command={"base_velocity": commands}.__getitem__),
    )
    manager = RewardManager({"foot_sep_simple": cfg.rewards.foot_sep_simple}, env)

    torch.testing.assert_close(
        manager.compute(cfg.decimation * cfg.sim.dt),
        torch.tensor([0.0, -0.12, -0.02, -0.01, -0.038, 0.0, -0.038, -0.038, -0.04]),
        atol=1.0e-7,
        rtol=0.0,
    )


def test_go2_flat_foot_direction_rewards_match_quadflat_geometry_and_gates():
    """실제 보상 설정의 방향 기준, 속력 가중 평균, 접촉·명령 경계와 좌표 변환을 검증함."""
    from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

    cfg = UnitreeGo2FlatEnvCfg()
    num_envs = 23
    commands = torch.tensor([[1.0, 0.0, 0.0]] * num_envs)
    base_velocity = torch.tensor([[1.0, 0.0, 0.0]] * num_envs)
    foot_velocity = torch.zeros(num_envs, 4, 3)
    foot_velocity[:, 0, 0] = 1.0
    forces = torch.zeros_like(foot_velocity)
    roll = torch.zeros(num_envs)
    pitch = torch.zeros(num_envs)
    yaw = torch.zeros(num_envs)

    foot_velocity[1, 0] = torch.tensor([0.0, 1.0, 0.0])  # 직각
    foot_velocity[2, 0, 0] = -1.0  # 반대 방향
    foot_velocity[3, 1, 0] = -3.0  # 정렬 속력 1, 역정렬 속력 3
    base_velocity[4, 0] = -1.0  # 명령과 실제 진행 방향이 반대임
    yaw[5] = torch.pi / 2
    foot_velocity[5, 0] = torch.tensor([0.0, 1.0, 0.0])
    forces[6, 0, 2] = 1.001  # 접촉한 발은 제외함
    forces[7, 0, 2] = 1.0  # 정확히 1 N이면 스윙으로 판정함
    foot_velocity[8, 0, 0] = 0.03  # 최소 발 속력은 초과해야 함
    commands[9] = torch.tensor([0.099, 0.099, 0.0])  # 명령 벡터 크기가 아닌 각 성분으로 판정함
    commands[10, 0] = 0.1  # 정확히 명령 임계값이면 활성화함
    base_velocity[11, 0] = 0.05  # 최소 몸통 속력은 초과해야 함
    foot_velocity[11, 0, 0] = -1.0
    commands[12] = torch.tensor([0.0, 0.0, -0.2])  # 순수 회전 중 선속도 드리프트
    foot_velocity[12, 0, 0] = -1.0
    commands[13] = 0.0  # 정지 명령에서는 드리프트해도 비활성화함
    foot_velocity[13, 0, 0] = -1.0
    foot_velocity[14] = 0.0
    foot_velocity[14, 0, 2] = 2.0  # 수직으로만 움직이는 발은 평가하지 않음
    forces[15, 0, 0] = 2.0  # 힘의 norm이 아닌 월드 Z 성분으로 접촉을 판정함
    foot_velocity[16, 0, 0] = 0.0301
    base_velocity[17, 0] = 0.0501
    foot_velocity[17, 0, 0] = -1.0
    commands[18] = torch.tensor([0.0, 0.0, 0.2])
    base_velocity[18] = 0.0  # 제자리 회전에서는 두 항 모두 0임
    roll[19], pitch[19], yaw[19] = torch.pi / 4, torch.pi / 6, torch.pi / 2
    foot_velocity[19, 0] = torch.tensor([0.0, 1.0, 0.0])  # 발 속도에는 yaw만 제거함
    foot_velocity[20, 1, 0] = -3.0
    forces[20, 1, 2] = 2.0  # 빠른 역방향 발도 접촉 상태이면 평균에서 제외함
    commands[21] = torch.tensor([0.0, 1.0, 0.0])
    foot_velocity[21, 0] = torch.tensor([0.0, 1.0, 0.0])  # 횡이동도 같은 방식으로 평가함
    base_velocity[22] = torch.tensor([0.0, 1.0, 0.0])  # 실제 진행 방향이 명령에 직각임

    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    asset_names = ["base", "RR_foot", "FL_foot", "FR_thigh", "RL_foot", "FR_foot"]
    sensor_names = ["FR_foot", "RL_foot", "base", "RR_foot", "FL_foot"]
    link_velocity = torch.zeros(num_envs, len(asset_names), 3)
    contact_forces = torch.zeros(num_envs, len(sensor_names), 3)
    link_velocity[:, [asset_names.index(name) for name in feet]] = foot_velocity
    contact_forces[:, [sensor_names.index(name) for name in feet]] = forces
    asset = SimpleNamespace(
        num_bodies=len(asset_names),
        body_names=asset_names,
        find_bodies=lambda names, preserve_order: resolve_matching_names(names, asset_names, preserve_order),
        data=SimpleNamespace(
            body_link_lin_vel_w=SimpleNamespace(torch=link_velocity),
            root_lin_vel_b=SimpleNamespace(torch=base_velocity),
            root_quat_w=SimpleNamespace(torch=quat_from_euler_xyz(roll, pitch, yaw)),
        ),
    )
    sensor = SimpleNamespace(
        num_sensors=len(sensor_names),
        body_names=sensor_names,
        find_sensors=lambda names, preserve_order: resolve_matching_names(names, sensor_names, preserve_order),
        data=SimpleNamespace(
            net_forces_w=SimpleNamespace(torch=contact_forces),
            net_forces_w_history=SimpleNamespace(torch=torch.full((num_envs, 3, len(sensor_names), 3), 100.0)),
        ),
    )

    class Scene(dict):
        sensors = {"contact_forces": sensor}

    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene=Scene(robot=asset, contact_forces=sensor),
        command_manager=SimpleNamespace(get_command={"base_velocity": commands}.__getitem__),
    )
    manager = RewardManager(
        {"foot_alignment": cfg.rewards.foot_alignment, "foot_misalignment": cfg.rewards.foot_misalignment}, env
    )
    reward = manager.compute(cfg.decimation * cfg.sim.dt)
    expected = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.25, 0.75],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 0.5],
        ]
    )
    weighted_terms = torch.tensor(
        [[value[0] for _, value in manager.get_active_iterable_terms(i)] for i in range(num_envs)]
    )
    torch.testing.assert_close(weighted_terms, expected * torch.tensor([0.05, -0.05]), atol=5.0e-6, rtol=0.0)
    torch.testing.assert_close(reward, 0.001 * (expected[:, 0] - expected[:, 1]), atol=2.0e-7, rtol=0.0)


@pytest.mark.parametrize("rough_terrain", [False, True], ids=["flat", "rough"])
def test_go2_height_reward_uses_ground_relative_exponential(rough_terrain):
    """The configured height kernel retains its scale when rough terrain changes elevation."""
    from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
    from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

    cfg = UnitreeGo2RoughEnvCfg() if rough_terrain else UnitreeGo2FlatEnvCfg()
    positions = torch.zeros(5, 3)
    positions[:, 2] = torch.tensor([0.28, 0.23, 0.33, 0.18, 0.38])
    scene = {"robot": SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=positions)))}
    if rough_terrain:
        positions[:, 2] += torch.tensor([0.0, 0.5, -0.4, 1.2, 0.0])
        # Average only finite terrain hits; the last environment has no hits and uses world Z = 0.
        ray_hits = torch.zeros(5, 3, 3)
        ray_hits[..., 2] = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, float("inf")],
                [-0.4, -0.4, -0.4],
                [1.1, 1.3, float("nan")],
                [float("inf"), float("nan"), -float("inf")],
            ]
        )
        scene["height_scanner"] = SimpleNamespace(data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=ray_hits)))
    env = SimpleNamespace(
        num_envs=5,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene=scene,
    )
    manager = RewardManager({"base_height": cfg.rewards.base_height}, env)

    value = manager.compute(cfg.decimation * cfg.sim.dt)

    torch.testing.assert_close(value, torch.tensor([0.04, 0.0242612264, 0.0242612264, 0.0054134113, 0.0054134113]))


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


def test_go2_rough_feet_clearance_uses_local_finite_terrain_hits():
    """Each selected foot uses nearby terrain, independent of world elevation and ray order."""
    from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

    cfg = UnitreeGo2RoughEnvCfg()
    feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    body_names = ["base", "RR_foot", "FL_foot", "other", "FR_foot", "RL_foot"]
    body_ids = [body_names.index(name) for name in feet]
    positions = torch.full((3, len(body_names), 3), 100.0)
    positions[:, body_ids] = torch.tensor([[1.0, 1.0, 0.48], [1.0, -1.0, 0.18], [-1.0, 1.0, 0.98], [-1.0, -1.0, -0.12]])
    velocities = torch.full_like(positions, 100.0)
    velocities[:, body_ids] = torch.tensor([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
    ray_hits = torch.tensor(
        [
            [-1.0, -1.0, -0.2],
            [-1.0, 1.0, 0.7],
            [1.0, -1.0, 0.1],
            [1.0, 1.0, 0.3],
            [float("nan"), 1.0, 100.0],
            [1.0, float("nan"), 100.0],
            [1.0, 1.0, float("nan")],
            [float("inf"), float("inf"), float("inf")],
        ]
    ).repeat(3, 1, 1)
    # Move robot and terrain together; reversing sample order must not change the local heights.
    translation = torch.tensor([4.0, -3.0, 2.0])
    positions[1] += translation
    ray_hits[1] = ray_hits[1].flip(0) + translation
    # With no valid terrain sample, use the configured plane_z fallback (zero here).
    ray_hits[2] = float("inf")
    positions[2, body_ids, 2] = torch.tensor([0.18, 0.08, 0.08, 0.08])
    asset = SimpleNamespace(
        num_bodies=len(body_names),
        body_names=body_names,
        find_bodies=lambda names, preserve_order: resolve_matching_names(names, body_names, preserve_order),
        data=SimpleNamespace(
            body_link_pos_w=SimpleNamespace(torch=positions),
            body_link_lin_vel_w=SimpleNamespace(torch=velocities),
        ),
    )
    env = SimpleNamespace(
        num_envs=3,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        scene={
            "robot": asset,
            "height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=SimpleNamespace(torch=ray_hits))),
        },
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[0.2, 0.0, 0.0]] * 3)),
    )
    manager = RewardManager({"feet_clearance": cfg.rewards.feet_clearance}, env)

    # First two rows: (0.1² * 0.1 + 0.2² * 0.2) * -10 * 0.02; fallback row: 0.1² * 0.1 * -10 * 0.02.
    torch.testing.assert_close(manager.compute(cfg.decimation * cfg.sim.dt), torch.tensor([-0.0018, -0.0018, -0.0002]))


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
