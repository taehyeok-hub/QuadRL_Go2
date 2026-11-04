# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""속도 추종을 목적으로 한 보행 환경을 위한 Reward 함수들"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import index_fill_
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _is_moving_command(command: torch.Tensor, command_threshold: float) -> torch.Tensor:
    """Return whether a planar or yaw command exceeds the motion threshold."""
    return (torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold) | (
        command[:, 2].abs() > command_threshold
    )


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """몸통 높이가 목표 높이에 가까울수록 지수 보상을 반환함.

    base_height_l2로 몸통 높이 오차의 제곱을 계산하고, exp(-(h - target_height) ** 2 / height_sigma)로 변환함.
    목표 높이에서 반환값은 1이고, 높이 오차가 커질수록 0으로 수렴하도록 구성됨.
    가중치와 제어 시간 간격은 RewardManager에서 별도로 곱함.

    Args:
        env: 보상을 계산할 강화학습 환경임.
        target_height: 평지 지면이 월드 Z=0일 때의 목표 몸통 높이.
        height_sigma: 높이 오차 제곱을 나누는 양의 민감도 계수이며 단위는 m²임. not std, 수식의 분모에 직접 사용하는 값.
        asset_cfg: 몸통 높이를 읽을 로봇의 선택 설정임.

    Returns:
        환경별 지수 보상이며, 가중치와 시간 배율을 적용하기 전의 값임.
    """
    height_error = mdp.base_height_l2(env, target_height=target_height, asset_cfg=asset_cfg)
    return torch.exp(-height_error / height_sigma)


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """착지 시점에 완료된 스윙 시간을 평가하는 방식의 feet air time reward 함수임.

    이 함수는 발이 착지했을 때, 직전 공중 체류 시간이 기준값(threshold)보다 긴 스텝을 수행할 때 보상을 부여함.
    이를 통해 로봇이 발을 지면에서 들어 올려 스텝을 수행하도록 유도함.
    보상은 각 발이 공중에 머문 시간을 합산하여 계산함.

    이동 명령이 작거나 로봇이 스텝을 수행할 필요가 없을 때, 보상을 0으로 함.
    """
    # scene에 등록된 접촉 센서를 가져옴. (: ContactSensor 타입 힌트)
    # : 센서가 접촉 여부와 시간을 관리해 함수 내에서 직접 시간을 누적할 필요가 없음.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 이번 제어 스텝에서 착지한 발을 찾고, 착지 직전에 얼마나 공중에 있었는지 읽어와 착지한 발의 시간 점수를 합산함.
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)

    # Zero command일 때에는 보상 X.
    commands = env.command_manager.get_command(command_name)
    reward *= _is_moving_command(commands, command_threshold)
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """2족 로봇이 발로 긴 스텝을 수행하도록 보상을 부여하는 함수.

    이전과 동일하게 기준값(threshold)까지 스텝에 대한 보상을 부여하며, 한 번에 한 발만 공중에 있도록 유도함.

    이동 명령이 작은 경우, 즉 로봇이 스텝을 수행할 필요가 없는 경우에는 보상을 0으로 설정함.
    """

    # scene에 등록된 접촉 센서를 가져옴.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)

    # no reward for zero command
    commands = env.command_manager.get_command(command_name)
    moving = (torch.linalg.vector_norm(commands[:, :2], dim=1) > command_threshold) | (
        commands[:, 2].abs() > command_threshold
    )
    reward *= moving
    return reward


def feet_air_time_variance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    max_time: float = 0.5,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """발들 사이의 스윙 시간 및 지지 시간 차이에 페널티를 부여하는 함수.

    각 발의 마지막으로 완료된 스윙 시간과 지지 시간[s]을 각각 max_time[s] 이하로 제한한 뒤,
    발들 사이의 분산을 계산하여 합산함.

    이동 명령이 0에 가까우면 페널티는 0임.

    Raises:
        RuntimeError: ``sensor_cfg``으로 선택된 몸체가 2개 미만인 경우 발생한다.
        이 경우 ``torch.var``가 ``nan``을 반환할 수 있기 때문이다.
    """
    # scene에 등록된 접촉 센서를 가져옴.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 각 발의 마지막으로 완료된 (공중 체류 시간, 지지 시간)을 읽어옴.
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time.torch[:, sensor_cfg.body_ids]

    # 발들 사이의 차이를 계산하는 것이기에, 선택된 발이 최소 2개인지 검사. 2개 미만이면 RuntimeError 발생.
    num_feet = last_air_time.shape[1]
    if num_feet < 2:
        raise RuntimeError(
            f"'feet_air_time_variance' needs at least two bodies to take a variance across feet, but"
            f" '{sensor_cfg.name}' resolved to {num_feet}. Widen 'body_names' on its 'sensor_cfg' to"
            " cover every foot."
        )

    # 시간을 상한으로 제한하고, 발들 사이의 분산을 계산함. 모든 발의 시간이 같으면 0이고, 차이가 커질수록 커짐.
    # 스윙 시간과 지지 시간을 서로 같게 만들라는 뜻은 아님. (not 스윙시간 = 지지시간)
    penalty = torch.var(torch.clip(last_air_time, max=max_time), dim=1) + torch.var(
        torch.clip(last_contact_time, max=max_time), dim=1
    )

    # 이동 명령이 작으면 페널티를 0으로 설정.
    commands = env.command_manager.get_command(command_name)
    moving = (torch.linalg.vector_norm(commands[:, :2], dim=1) > command_threshold) | (
        commands[:, 2].abs() > command_threshold
    )
    penalty *= moving
    return penalty


class trot_gait_pattern(ManagerTermBase):
    """대각선 지지를 보상하고, 대각선 사용 편향과 공중 상태를 감점하는 함수 클래스"""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if not 0.0 <= cfg.params["ema_alpha"] < 1.0:
            raise ValueError("ema_alpha는 0 이상, 1 미만이어야 합니다.")

        # 양수: FL/RR 편향, 음수: FR/RL 편향, 0: 균형
        self.diagonal_bias = torch.zeros(env.num_envs, device=env.device)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self.diagonal_bias[selected] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        airborne_penalty: float = 1.0,
        balance_weight: float = 0.5,
        ema_alpha: float = 0.99,
        command_threshold: float = 0.05,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact = contact_sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, 2] > 1.0

        if contact.shape[1] != 4:
            raise ValueError("발을 FL, FR, RL, RR 순서로 정확히 4개 선택해야 합니다.")

        fl, fr, rl, rr = contact.unbind(dim=1)
        pair_a = fl & rr & ~fr & ~rl
        pair_b = fr & rl & ~fl & ~rr
        valid_pair = pair_a | pair_b

        commands = env.command_manager.get_command(command_name)
        moving = (torch.linalg.vector_norm(commands[:, :2], dim=1) > command_threshold) | (
            commands[:, 2].abs() > command_threshold
        )

        # 이동 중 유효한 대각선 지지가 관측될 때만 EMA 갱신
        target_bias = pair_a.float() - pair_b.float()
        updated_bias = ema_alpha * self.diagonal_bias + (1.0 - ema_alpha) * target_bias
        self.diagonal_bias.copy_(torch.where(moving & valid_pair, updated_bias, self.diagonal_bias))

        all_airborne = ~contact.any(dim=1)

        return (
            (valid_pair & moving).float()
            - balance_weight * self.diagonal_bias.abs() * moving.float()
            - airborne_penalty * all_airborne.float()
        )


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """발의 미끄러짐에 페널티를 부여한다.

    이 함수는 발이 지면에서 미끄러지는 동작에 대해 페널티를 부여하는데,
    지면을 지지하는 발이 수평 방향으로 움직이지 않도록 유도함.

    각 발의 선속도 크기에 이진 접촉 신호(binary contact sensor)를 곱하여 값을 계산함.
    이를 통해 지면에 접촉한 발에만 페널티를 적용함.
    """

    # 접촉 센서와 로봇 데이터를 가져옴.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    # 최근 접촉력으로 접촉 여부를 판정함. (1N 이상) (Env, Contact, Foot, xyz)
    contacts = (
        contact_sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )

    # 발의 수평 선속도(world 기준)를 계산하고, 접촉 여부와 곱하여 페널티를 계산함. (Env, Contact, Foot)
    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def feet_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    min_foot_speed: float,
    plane_z: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """평면 위에서 움직이는 발 링크 원점의 높이 오차에 수평 속력을 가중한 비용을 반환한다.

    발 높이는 발 링크 원점의 월드 Z 위치에서 ``plane_z``를 뺀 값이다.
    수평 선속력도 월드 좌표계의 발 링크 원점을 기준으로 계산한다.
    각 발의 수평 속력이 ``min_foot_speed`` 이상일 때만 높이 오차의 제곱에
    해당 발의 실제 수평 속력 전체를 곱한다. 발별 비용을 합산하며, 정지 명령에서는 0을 반환한다.

    Args:
        env: 환경 인스턴스.
        command_name: 평면 선속도와 yaw 각속도 명령을 제공하는 명령 항의 이름.
        target_height: 평면 지면으로부터 발 링크 원점까지의 목표 높이. 단위는 m.
        min_foot_speed: 발별 비용을 계산하기 위한 최소 수평 선속력. 단위는 m/s.
        plane_z: 평면 지면의 월드 Z 좌표. 단위는 m.
        asset_cfg: 평가할 로봇과 발 링크를 선택하는 설정.
        command_threshold: 평면 명령 속력과 yaw 각속도 명령의 크기에 각각 적용하는 이동 판정 임계값.

    Returns:
        환경별 0 이상의 비용을 담은 ``(num_envs,)`` 형태의 텐서. 보상 설정에서 음수 가중치를 적용한다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2] - plane_z
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    foot_speed_xy = torch.linalg.vector_norm(foot_vel_xy, dim=-1)

    per_foot_cost = (foot_height - target_height).square() * foot_speed_xy
    cost = torch.where(foot_speed_xy >= min_foot_speed, per_foot_cost, 0.0).sum(dim=1)
    command = env.command_manager.get_command(command_name)
    return cost * _is_moving_command(command, command_threshold)


def _swing_foot_direction_score(
    env: ManagerBasedRLEnv,
    reference_velocity: torch.Tensor,
    active: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    min_foot_speed: float,
    contact_threshold: float,
    opposed: bool,
) -> torch.Tensor:
    """기준 방향에 대한 스윙 발의 정렬 또는 역정렬 점수를 발 속력으로 가중 평균함."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_velocity_w = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids]
    heading = yaw_quat(asset.data.root_quat_w.torch)[:, None, :]
    foot_velocity_xy = quat_apply_inverse(heading, foot_velocity_w)[..., :2]
    foot_speed = torch.linalg.vector_norm(foot_velocity_xy, dim=-1)

    eps = 1.0e-6
    foot_direction = foot_velocity_xy / (foot_speed.unsqueeze(-1) + eps)
    reference_direction = reference_velocity / (
        torch.linalg.vector_norm(reference_velocity, dim=-1, keepdim=True) + eps
    )
    cosine = (foot_direction * reference_direction.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)
    score = 0.5 * (1.0 - cosine) if opposed else 0.5 * (1.0 + cosine)

    contact = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2] > contact_threshold
    valid = ~contact & (foot_speed > min_foot_speed) & active.unsqueeze(1)
    weights = torch.where(valid, foot_speed, 0.0)
    average = (score * weights).sum(dim=1) / (weights.sum(dim=1) + eps)
    return torch.where(valid.any(dim=1), average, 0.0)


def foot_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
    min_foot_speed: float = 0.03,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """스윙 중인 발의 수평 이동 방향이 선속도 명령 방향과 일치할수록 보상함.

    quadflat_lab과 동일하게 발 링크 원점의 월드 속도를 역 yaw 회전하며, 몸통 속도는 빼지 않음.
    현재 총 접촉력의 월드 Z 성분이 contact_threshold 이하이고 수평 속력이 min_foot_speed를 넘는 발을 평가함.
    선속도 명령의 X 또는 Y 성분 절댓값이 command_threshold 이상일 때만 활성화함.
    (1 + cos(theta)) / 2를 발 속력으로 가중 평균하므로 같은 방향은 1, 직각은 0.5, 반대 방향은 0에 가까움.
    접촉 이력은 사용하지 않으며, 순수 회전 명령이나 유효한 스윙 발이 없는 경우 0을 반환함.
    로봇과 센서의 발 선택은 같은 순서여야 하며, 가중치와 시간 배율은 RewardManager에서 별도로 적용함.

    Args:
        command_name: 몸통 기준 vx, vy, yaw 각속도 명령을 제공하는 명령 항의 이름임.
        sensor_cfg: 현재 접촉력을 읽을 발 센서의 선택 설정임.
        asset_cfg: 발 링크 속도와 몸통 자세를 읽을 로봇의 선택 설정임.
        command_threshold: 각 선속도 명령 성분의 활성화 임계값이며 단위는 m/s임.
        min_foot_speed: 평가할 발의 최소 수평 속력이며 단위는 m/s임.
        contact_threshold: 접촉으로 판정할 월드 Z 힘의 임계값이며 단위는 N임.
    """
    command = env.command_manager.get_command(command_name)
    active = (command[:, :2].abs() >= command_threshold).any(dim=1)
    return _swing_foot_direction_score(
        env, command[:, :2], active, asset_cfg, sensor_cfg, min_foot_speed, contact_threshold, opposed=False
    )


def foot_misalignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.1,
    min_foot_speed: float = 0.03,
    min_base_speed: float = 0.05,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """스윙 발이 실제 몸통 진행 방향과 어긋날수록 커지는 비용을 반환함.

    기준은 명령 방향이 아니라 root_lin_vel_b의 XY 방향이며, 음의 가중치로 적용하여 역정렬을 감점함.
    quadflat_lab의 계산을 유지하므로 발 속도는 yaw 좌표계, 몸통 속도는 roll과 pitch도 반영한 몸통 좌표계임.
    발 속도에서 몸통 속도는 빼지 않으며, 기울어진 자세에서는 두 좌표계가 일치하지 않을 수 있음.
    현재 총 접촉력의 월드 Z 성분이 contact_threshold 이하이고 수평 속력이 min_foot_speed를 넘는 발을 평가함.
    몸통 XY 속력이 min_base_speed를 넘고, vx, vy, yaw 명령 중 하나의 절댓값이 command_threshold 이상일 때 활성화함.
    (1 - cos(theta)) / 2를 발 속력으로 가중 평균하므로 같은 방향은 0, 직각은 0.5, 반대 방향은 1에 가까움.
    접촉 이력은 사용하지 않으며, 정지 명령이나 유효한 스윙 발이 없는 경우 0을 반환함.
    로봇과 센서의 발 선택은 같은 순서여야 하며, 가중치와 시간 배율은 RewardManager에서 별도로 적용함.

    Args:
        command_name: 몸통 기준 vx, vy, yaw 각속도 명령을 제공하는 명령 항의 이름임.
        sensor_cfg: 현재 접촉력을 읽을 발 센서의 선택 설정임.
        asset_cfg: 발 링크 속도와 실제 몸통 속도를 읽을 로봇의 선택 설정임.
        command_threshold: 각 명령 성분의 활성화 임계값이며 선속도는 m/s, yaw 각속도는 rad/s 기준임.
        min_foot_speed: 평가할 발의 최소 수평 속력이며 단위는 m/s임.
        min_base_speed: 실제 진행 방향을 평가할 최소 몸통 XY 속력이며 단위는 m/s임.
        contact_threshold: 접촉으로 판정할 월드 Z 힘의 임계값이며 단위는 N임.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    base_velocity = asset.data.root_lin_vel_b.torch[:, :2]
    active = (command[:, :3].abs() >= command_threshold).any(dim=1)
    active &= torch.linalg.vector_norm(base_velocity, dim=-1) > min_base_speed
    return _swing_foot_direction_score(
        env, base_velocity, active, asset_cfg, sensor_cfg, min_foot_speed, contact_threshold, opposed=True
    )


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """중력 기준으로 수평을 유지하는 로봇 좌표계에서 XY축 선속도 명령의 추종 성능을 보상한다.

    지수함수 커널로 추종 성능을 평가한다.

    월드 좌표계	: 시뮬레이션 공간에 고정되어, 축 방향이 그대로 유지됨.
    중력 정렬 로봇 좌표계(yaw frame): 로봇의 yaw 방향을 따라 회전하며, roll·pitch는 반영하지 않고 수평을 유지함.

    """

    # 로봇 데이터를 가져옴.
    asset: Articulation = env.scene[asset_cfg.name]

    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """월드 좌표계에서 yaw 각속도 명령의 추종 성능을 지수함수 커널로 평가하여 보상을 부여함."""

    # 로봇 데이터를 가져옴.
    asset: Articulation = env.scene[asset_cfg.name]

    # 명령 각속도와 실제 각속도의 차이를 제곱하고 지수함수 커널로 변환하여 보상으로 사용함.
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w.torch[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """명령의 크기가 매우 작을 때, 기본 관절 위치에서 벗어난 정도에 페널티를 부여하는 함수."""

    # 현재의 로봇의 명령을 가져옴. (Env, Command)
    command = env.command_manager.get_command(command_name)

    # 평면 이동과 yaw 회전 명령이 모두 작을 때만 기본 관절각 편차를 감점함.
    standing = (torch.linalg.vector_norm(command[:, :2], dim=1) < command_threshold) & (
        command[:, 2].abs() < command_threshold
    )
    return mdp.joint_deviation_l1(env, asset_cfg) * standing


def stand_still_foot_position_l2(
    env,
    command_name: str,
    default_foot_positions: list[tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """정지 명령에서 기본 수평 발 배치로부터의 오차를 감점한다."""
    robot = env.scene[asset_cfg.name]

    foot_pos_w = robot.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    root_pos_w = robot.data.root_pos_w.torch
    root_quat_w = robot.data.root_quat_w.torch

    # 베이스를 원점으로 이동
    relative_foot_pos_w = foot_pos_w - root_pos_w[:, None, :]

    # yaw만 역변환: roll·pitch에 따라 좌표축을 기울이지 않음
    root_yaw_quat = yaw_quat(root_quat_w)
    foot_pos_heading = quat_apply_inverse(
        root_yaw_quat[:, None, :].expand(-1, foot_pos_w.shape[1], -1),
        relative_foot_pos_w,
    )

    # 네 발의 수평 위치 오차: 발별 제곱 거리의 평균
    foot_xy = foot_pos_heading[:, :, :2]
    default_xy = foot_xy.new_tensor(default_foot_positions)[None, :, :]
    penalty = (foot_xy - default_xy).square().sum(dim=(1, 2)) / 4.0

    # 평면 이동·회전 명령이 모두 작을 때만 적용
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.vector_norm(command[:, :2], dim=1) < command_threshold) & (
        command[:, 2].abs() < command_threshold
    )

    return penalty * standing.to(penalty.dtype)


def stand_still_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize missing foot contacts during standing commands.

    The sensor must select exactly four feet.
    """
    # Count feet whose current world-Z contact force exceeds the threshold.
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2] > 1.0
    num_contact = contact.float().sum(dim=1)
    penalty = torch.square(4.0 - num_contact)

    # Require both translation and yaw commands to be small.
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.vector_norm(command[:, :2], dim=1) < command_threshold) & (
        command[:, 2].abs() < command_threshold
    )

    return penalty * standing.float()


class variable_position(ManagerTermBase):
    """Reward the default joint pose with standing and moving tolerances."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]

        # Match sigma values to the selected joint order.
        joint_ids = asset_cfg.joint_ids
        if isinstance(joint_ids, slice):
            joint_names = asset.joint_names[joint_ids]
        else:
            if isinstance(joint_ids, torch.Tensor):
                joint_ids = joint_ids.tolist()
            joint_names = [asset.joint_names[index] for index in joint_ids]

        if not joint_names:
            raise ValueError("variable_position requires at least one joint.")

        self._std = {}
        for key in ("std_standing", "std_moving"):
            indices, _, values = resolve_matching_names_values(cfg.params[key], joint_names, preserve_order=False)
            if indices != list(range(len(joint_names))):
                raise ValueError(f"{key} must cover every selected joint.")
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"{key} values must be finite and positive.")

            self._std[key] = torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std_standing: dict[str, float],
        std_moving: dict[str, float],
        asset_cfg: SceneEntityCfg,
        command_threshold: float = 0.05,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        # Use the same standing condition as stand_still_contact.
        standing = (torch.linalg.vector_norm(command[:, :2], dim=1) < command_threshold) & (
            command[:, 2].abs() < command_threshold
        )
        std = torch.where(
            standing.unsqueeze(-1),
            self._std["std_standing"],
            self._std["std_moving"],
        )

        # Always compare against the configured default joint positions.
        joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        reference = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        error = (joint_pos - reference) / std

        return torch.exp(-torch.mean(torch.square(error), dim=1))


class smoothness(ManagerTermBase):
    """Penalize second differences of scaled raw joint-position actions."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._action_term = env.action_manager.get_term(cfg.params["action_name"])
        self._action_scale = float(self._action_term.cfg.scale)

        self._prev_action = torch.zeros_like(self._action_term.raw_actions)
        self._prev_prev_action = torch.zeros_like(self._prev_action)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        index_fill_(self._prev_action, env_ids, 0.0)
        index_fill_(self._prev_prev_action, env_ids, 0.0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        action_name: str = "joint_pos",
    ) -> torch.Tensor:
        action = self._action_term.raw_actions

        second_diff = action - 2.0 * self._prev_action + self._prev_prev_action
        penalty = torch.sum(torch.square(second_diff), dim=1)

        # Evaluate from the third policy step of each episode.
        valid_history = env.episode_length_buf >= 3
        penalty = penalty * valid_history.float() * self._action_scale**2

        # Update history even during the first two policy steps.
        self._prev_prev_action.copy_(self._prev_action)
        self._prev_action.copy_(action)

        return penalty


def stand_still_joint_velocity(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_sigma: float,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize normalized joint velocities during standing commands."""
    if velocity_sigma <= 0.0:
        raise ValueError("velocity_sigma must be positive.")

    # 평면 이동과 yaw 회전 명령이 모두 작을 때 정지로 판정.
    command = env.command_manager.get_command(command_name)
    standing = (torch.linalg.vector_norm(command[:, :2], dim=1) < command_threshold) & (
        command[:, 2].abs() < command_threshold
    )

    # 선택된 관절의 현재 속도. 단위: rad/s.
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]

    # 관절 수에 영향을 덜 받도록 합 대신 평균 사용.
    normalized_velocity = joint_vel / velocity_sigma
    penalty = normalized_velocity.square().mean(dim=1)

    return penalty * standing.to(penalty.dtype)
