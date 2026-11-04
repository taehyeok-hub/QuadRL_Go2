# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

""" 보상 항목을 구성하는 데 사용할 수 있는 Public Functions.

각 함수를 isaaclab.managers.RewardTermCfg 객체에 전달하면,
해당 함수가 계산하는 보상을 환경의 보상 항목에 포함할 수 있음.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...managers import SceneEntityCfg
from ...managers.manager_base import ManagerTermBase
from ...managers.manager_term_cfg import RewardTermCfg
from ...utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, wrap_to_pi

if TYPE_CHECKING:
    from ...assets import Articulation, RigidObject
    from ...sensors import ContactSensor, RayCaster
    from .. import ManagerBasedRLEnv

"""
General-purpose rewards functions.
"""


def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ 살아있을 때 주는 보상 함수. """
    return (~env.termination_manager.terminated).float()


def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ 에피소드의 제한 시간 초과에 해당하지 않는 종료에 페널티를 부여하는 함수. --> 넘어짐 등 
    
    [ 실제 보상 기여분 = 함수 반환값 X weight X step_dt ] 으로 반영
    """
    return env.termination_manager.terminated.float()


def terminated_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ 환경의 시간 간격과 무관하게, 조기 종료 시 한 번의 고정 페널티를 부여함.

    RewardManager는 모든 보상 항목에 환경의 스텝 시간 간격을 곱해, 일반적인 종료 페널티에서는 sim.dt와 decimation에 따라 크기가 달라짐.

    이 함수는 변환값을 스텝 시간간격으로 나누어서 RewardManager의 시간 배율을 상쇄함.
    따라서 에피소드가 종료되는 스텝에서, 설정한 가중치만큼의 값이 그대로 반영됨.

    [ 실제 보상 기여분 = 함수 반환값 X weight ]
    """
    return env.termination_manager.terminated.float() / env.step_dt


class is_terminated_term(ManagerTermBase):
    """ 에피소드 시간 초과를 제외하고, 어떤 종료 조건에 벌점을 줄지 지정하고 그 지정한 종료 조건에 페널티를 부여하는 함수.

    매개변수:

    term_keys: 페널티를 적용할 종료 조건의 이름.

    문자열, 문자열 목록 또는 정규표현식으로 지정할 수 있고, 시간 초과가 발생한 경우에는 0을 반환함. 
    그 외에는 선택한 종료 조건 중 활성화된 조건의 개수를 합산하여 반환함. 두 개의 종료조건이 동시에 활성화되면 2를 반환함.
    """
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # 생성 시, 벌점을 줄지 대상 조건의 이름을 찾아 저장.
        term_keys = cfg.params["term_keys"]
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # 매 보상 계산 때는 선택한 조건들의 발생 여부를 더해서 반환.
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            reset_buf += env.termination_manager.get_term(term)
        # 시간 초과가 발생한 경우에는 0을 반환함. 시간 초과와 다른 실패 조건이 동시에 발생해도 0을 반환함. 
        return (reset_buf * (~env.termination_manager.time_outs)).float()


class survival_success_rate(ManagerTermBase):
    """ 에피소드 생존 여부를 성공 지표로 기록하는 함수 클래스.

    보상값은 항상 0을 반환하며, 지표만 기록하도록 함.
    에피소드가 리셋될 때 extras["log"]에 Metrics/success_rate를 기록함.
    
    조기 종료 없이 제한 시간에 도달한 에피소드를 성공으로 간주함.
    """
    def reset(self, env_ids: torch.Tensor) -> None:
        # 에피소드가 끝나 리셋될 때 결과를 확인하고 extras["log"]에 기록함. (env_ids : 리셋되는 환경 idx)
        survived = self._env.termination_manager.time_outs[env_ids] 
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = survived.float().mean()

    # 매 보상 계산에서는 0을 반환함.
    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.zeros(env.num_envs, device=env.device)











"""
Root penalties.
"""


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 몸통좌표계의 Z축 선속도에 제곱 L2 페널티를 부여하는 함수. """
    
    # 로봇의 RigidObject를 가져와서, root_lin_vel_b.torch[:, 2](body 기반)를 제곱하여 반환함.
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b.torch[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 몸통좌표계의 XY축 각속도에 제곱 L2 페널티를 부여하는 함수. """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 몸통의 평면이 아닌 방향에 대해 L2 제곱 커널을 사용하여 페널티를 부여하는 함수.

    이는 투영된 중력 벡터의 xy 성분을 페널티로 적용하여 계산됨.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """ 몸통의 높이를 목표 높이로부터 페널티를 부여하는 함수. L2 제곱 커널을 사용함.

    Note:
        평면 지형의 경우, 목표 높이는 월드 프레임에 위치함. 
        거친 지형의 경우, 센서 읽기는 지형을 고려하여 목표 높이를 조정할 수 있음.

    Note:
        ``max_distance`` 내에서 아무것도 충돌하지 않는 레이는 ``inf``를 보고하여 클램핑된 값을 보고하지 않으므로, 
        유한한 충돌만이 지형 조정된 목표 높이에 평균화됨. 모든 레이가 실패하는 환경은 조정되지 않은 ``target_height``로 돌아감.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data, ignoring rays that missed the terrain
        ray_hits_z = sensor.data.ray_hits_w.torch[..., 2]
        finite_hits = torch.isfinite(ray_hits_z)
        safe_hits = torch.where(finite_hits, ray_hits_z, torch.zeros_like(ray_hits_z))
        hit_count = finite_hits.sum(dim=1).clamp(min=1)
        terrain_height = safe_hits.sum(dim=1) / hit_count
        adjusted_target_height = target_height + terrain_height
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w.torch[:, 2] - adjusted_target_height)


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.linalg.norm(asset.data.body_lin_acc_w.torch[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques
        contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """ 선택한 관절의 속도를 L1-kernal을 활용해 페널티를 계산하는 함수. """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 선택한 관절의 속도를 제곱하여 (L2-kernal을 활용해) 페널티를 계산하는 함수.

    .. note::
        `asset_cfg.joint_ids` 에 지정된 관절만 계산에 포함됨.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ 선택한 관절의 가속도를 제곱하여 (L2-kernal을 활용해) 페널티를 계산하는 함수.

    .. note::
       `asset_cfg.joint_ids` 에 지정된 관절만 계산에 포함됨.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc.torch[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint positions that deviate from a target value using an L2 squared kernel.

    The joint positions are wrapped to ``[-pi, pi]`` before the deviation is computed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(joint_pos - target), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """ 관절 속도가 soft limit을 초과하면 패널티를 가하는 함수.

    관절 속도가 soft limits에 soft_ratio를 곱한 임계값을 초과한 양을 합산함.
    이 함수는 관절 속도와 soft limits 사이의 차이의 절대값을 계산함.

    실물의 속도·토크 한계는 제어기와 모터 설정에 반영하지만, 이 리워드만으로는 제한을 보장하지 못하기에, 
    바로 적용하기보다는 관절이 허용속도에 자주 접근하거나 초과할 때 선택적으로 활용함.

    Args:
        soft_ratio: 감점 시작 임계값을 정하는 속도 한계의 비율임. 1보다 작으면 한계 도달 전부터 감점함.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = (
        torch.abs(asset.data.joint_vel.torch[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits.torch[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


"""
Action penalties.
"""


def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
        - asset.actuators.computed_effort.torch[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """최근 법선 접촉력 이력의 최대값이 임계값을 넘은 부위 수를 반환함.

    짧은 접촉을 놓치지 않도록 이력을 사용하므로, 접촉이 끝나도 해당 기록이 남아 있으면 판정에 포함함.
    임계값을 1 N에서 0.1 N으로 낮추면 이력의 최대값이 0.1 N 초과 1 N 이하인 부위도 추가 판정함.
    부위당 감점 가중치는 유지되며, 추가 판정되는 약한 접촉이 드물면 보상 변화도 작은 편임.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch
    is_contact = (
        torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    )
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """ 선택한 부위 중 접촉한 부위가 하나도 없으면 1, 하나라도 있으면 0을 반환하도록 하는 함수. """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact


def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch
    # compute the violation
    violation = (
        torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    )
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)


"""
Velocity-tracking rewards.
"""


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b.torch[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b.torch[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


"""
Pose-tracking rewards.
"""


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using the L2 norm.

    The error [m] is the L2 norm between the commanded position (resolved into the world frame from the
    asset's root pose) and the current position of the asset's body in the world frame. The command is
    expected to be a pose command whose first three entries are the desired position in the root frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w.torch, asset.data.root_quat_w.torch, des_pos_b)
    curr_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids][:, 0]  # type: ignore
    return torch.linalg.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position error using the tanh kernel.

    The position error [m] is computed as in :func:`position_command_error` and mapped through a tanh
    kernel with standard deviation ``std`` [m], yielding a bounded reward in ``[0, 1)``.
    """
    distance = position_command_error(env, command_name, asset_cfg)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the orientation error using the shortest-path quaternion distance.

    The error [rad] is the shortest-path angle between the commanded orientation (resolved into the
    world frame from the asset's root pose) and the current orientation of the asset's body in the world
    frame. The command is expected to be a pose command whose entries ``[3:7]`` are the desired
    orientation quaternion in the root frame.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations in the world frame
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w.torch, des_quat_b)
    curr_quat_w = asset.data.body_quat_w.torch[:, asset_cfg.body_ids][:, 0]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)
