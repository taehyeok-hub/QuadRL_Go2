# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import index_fill_

if TYPE_CHECKING:
    from ... import ManagerBasedEnv
    from .commands_cfg import NormalVelocityCommandCfg, UniformVelocityCommandCfg

logger = logging.getLogger(__name__)


class UniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: UniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if not 0.0 <= self.cfg.rel_standing_envs <= 1.0:
            raise ValueError("rel_standing_envs must be in [0, 1].")
        if not 0.0 <= self.cfg.rel_rotating_envs <= 1.0:
            raise ValueError("rel_rotating_envs must be in [0, 1].")
        if not 0.0 <= self.cfg.rel_transition_envs <= 1.0:
            raise ValueError("rel_transition_envs must be in [0, 1].")
        if self.cfg.rel_standing_envs + self.cfg.rel_rotating_envs + self.cfg.rel_transition_envs > 1.0:
            raise ValueError("rel_standing_envs + rel_rotating_envs + rel_transition_envs must be <= 1.")
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            logger.warning(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.is_rotating_env = torch.zeros_like(self.is_heading_env)
        # 전환 환경과 역할은 한 번만 배정하며, 에피소드 리셋에서도 유지함.
        self.is_transition_env = torch.zeros_like(self.is_heading_env)
        self.transition_type = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._num_transition_envs = round(self.num_envs * self.cfg.rel_transition_envs)
        if (
            self._num_transition_envs / self.num_envs + self.cfg.rel_standing_envs + self.cfg.rel_rotating_envs
            > 1.0 + 1.0e-12
        ):
            raise ValueError("The rounded transition quota leaves insufficient environments for standing/rotating.")
        if self.cfg.rel_transition_envs > 0.0:
            self._initialize_transition_envs()
        # -- metrics: finalized per-episode means/rates, written at reset() and read by the base class
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)
        # -- per-episode running sums (cleared at episode reset)
        self._error_xy_sum = torch.zeros(self.num_envs, device=self.device)
        self._error_yaw_sum = torch.zeros(self.num_envs, device=self.device)
        self._step_count = torch.zeros(self.num_envs, device=self.device)

        # adds (optional) cmd kind and element names for leapp export
        # during export, semantic data about this command will be used to annotate the command input
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/velocity"
        self.cfg.element_names = self.cfg.element_names or ["lin_vel_x", "lin_vel_y", "ang_vel_z"]

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        if self._num_transition_envs:
            msg += f"\n\tFixed transition environments: {self._num_transition_envs}/{self.num_envs}"
            msg += f"\n\tTransition resampling time range: {self.cfg.transition_resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # accumulate per-step tracking error sums; the per-episode mean is finalized in
        # :meth:`reset` so the value is independent of episode length and
        # ``resampling_time_range`` (no magic ``max_command_step`` divisor).
        error_xy = torch.linalg.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
        error_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b.torch[:, 2])
        self._error_xy_sum += error_xy
        self._error_yaw_sum += error_yaw
        self._step_count += 1.0
        if self._num_transition_envs:
            self._update_transition_metrics(error_xy, error_yaw)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # Finalize the just-ended episode's metrics into ``self.metrics`` BEFORE the base
        # class reads them. ``success_rate`` is per-env binary: the *episode-mean* error
        # is below both thresholds. Then super().reset() logs and zeros ``self.metrics``;
        # we zero the running sums for ``env_ids`` afterwards so the next episode starts clean.
        if env_ids is None:
            env_ids = slice(None)
        denom = self._step_count[env_ids].clamp_min(1.0)
        mean_error_xy = self._error_xy_sum[env_ids] / denom
        mean_error_yaw = self._error_yaw_sum[env_ids] / denom
        self.metrics["error_vel_xy"][env_ids] = mean_error_xy
        self.metrics["error_vel_yaw"][env_ids] = mean_error_yaw
        self.metrics["success_rate"][env_ids] = (
            (mean_error_xy < self.cfg.vel_xy_success_threshold) & (mean_error_yaw < self.cfg.vel_yaw_success_threshold)
        ).float()
        transition_metrics = self._reset_transition_metrics(env_ids) if self._num_transition_envs else {}
        extras = super().reset(env_ids)
        # Route success_rate to the unified ``Metrics/success_rate`` path (shared TensorBoard
        # / wandb card across tasks); pop it from the returned dict so CommandManager does
        # not additionally log it under ``Metrics/<term_name>/success_rate``.
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = extras.pop("success_rate")
        index_fill_(self._error_xy_sum, env_ids, 0.0)
        index_fill_(self._error_yaw_sum, env_ids, 0.0)
        index_fill_(self._step_count, env_ids, 0.0)
        extras.update(transition_metrics)
        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        r = torch.empty(num_envs, device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # Select mutually exclusive standing and rotating environments.
        mode = r.uniform_(0.0, 1.0)
        if self._num_transition_envs:
            # 전환 환경을 제외한 집단에서 추첨하되, 기존 비율은 전체 환경 기준으로 유지함.
            mode *= (self.num_envs - self._num_transition_envs) / self.num_envs
        self.is_standing_env[env_ids] = mode < self.cfg.rel_standing_envs
        self.is_rotating_env[env_ids] = (mode >= self.cfg.rel_standing_envs) & (
            mode < self.cfg.rel_standing_envs + self.cfg.rel_rotating_envs
        )

        # Apply rotation constraints immediately, including after a reset.
        if self.cfg.rel_rotating_envs > 0.0:
            self.vel_command_b[env_ids, :2] = self.vel_command_b[env_ids, :2].masked_fill(
                self.is_rotating_env[env_ids].unsqueeze(-1), 0.0
            )
        if self._num_transition_envs:
            self._resample_transition_commands(env_ids)

    def _update_command(self):
        """Apply heading control and standing or rotating constraints."""
        # Compute angular velocity from heading direction.
        if self.cfg.heading_command:
            heading_error = math_utils.wrap_to_pi(self.heading_target - self.robot.data.heading_w.torch)
            heading_ang_vel = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
            self.vel_command_b[:, 2] = torch.where(
                self.is_heading_env & ~self.is_rotating_env & ~self.is_transition_env,
                heading_ang_vel,
                self.vel_command_b[:, 2],
            )

        # Rotating environments keep their sampled yaw velocity.
        if self.cfg.rel_rotating_envs > 0.0:
            index_fill_(self.vel_command_b[:, :2], self.is_rotating_env, 0.0)

        # Standing environments have zero velocity in all components.
        index_fill_(self.vel_command_b, self.is_standing_env, 0.0)

    def _initialize_transition_envs(self) -> None:
        """고정 전환 환경과 여섯 역할, 명령 패턴 및 전환 지표 버퍼를 준비함."""
        lower, upper = self.cfg.transition_resampling_time_range
        if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower <= upper):
            raise ValueError("transition_resampling_time_range must be finite and satisfy 0 < lower <= upper.")
        for name, (lower, upper) in (
            ("lin_vel_x", self.cfg.ranges.lin_vel_x),
            ("lin_vel_y", self.cfg.ranges.lin_vel_y),
            ("ang_vel_z", self.cfg.ranges.ang_vel_z),
        ):
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < 0.0 < upper):
                raise ValueError(f"Transition commands require finite {name} limits satisfying lower < 0 < upper.")

        self._transition_env_ids = torch.randperm(self.num_envs, device=self.device)[: self._num_transition_envs]
        self.is_transition_env[self._transition_env_ids] = True
        roles = torch.arange(self._num_transition_envs, device=self.device)
        self.transition_type[self._transition_env_ids] = roles % 6
        self._transition_variants = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._transition_variants[self._transition_env_ids] = (roles // 6) % 2 == 1
        self._transition_phase_offset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._transition_phase_offset[self._transition_env_ids] = torch.randint(
            4, (self._num_transition_envs,), device=self.device
        )
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)
        # 0:X, 1:Y, 2:같은 부호 대각선, 3:반대 부호 대각선, 4:X+yaw, 5:Y+yaw 순서임.
        # 부호만 저장하고 실제 속도는 재샘플링 시 현재 명령 범위의 끝값에서 읽음.
        self._transition_patterns = torch.tensor(
            [
                [[1, 0, 0], [-1, 0, 0], [1, 0, 0], [-1, 0, 0]],
                [[0, 1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0]],
                [[1, 1, 0], [-1, -1, 0], [1, 1, 0], [-1, -1, 0]],
                [[1, -1, 0], [-1, 1, 0], [1, -1, 0], [-1, 1, 0]],
                [[1, 0, 1], [-1, 0, -1], [1, 0, 1], [-1, 0, -1]],
                [[0, 1, 1], [0, -1, -1], [0, 1, 1], [0, -1, -1]],
            ],
            dtype=torch.int8,
            device=self.device,
        )
        self._transition_age = torch.full((self.num_envs,), math.inf, device=self.device)
        self._transition_pending_recovery = torch.zeros_like(self.is_transition_env)
        self._transition_count = torch.zeros(self.num_envs, device=self.device)
        self._transition_window_steps = torch.zeros_like(self._transition_count)
        self._transition_error_xy_sum = torch.zeros_like(self._transition_count)
        self._transition_error_yaw_sum = torch.zeros_like(self._transition_count)
        self._transition_peak_tilt = torch.zeros_like(self._transition_count)
        self._transition_airborne_steps = torch.zeros_like(self._transition_count)
        self._transition_recovery_sum = torch.zeros_like(self._transition_count)
        self._transition_recovered_count = torch.zeros_like(self._transition_count)
        self._transition_contact_sensor = None
        if self.cfg.transition_contact_sensor_cfg is not None:
            self.cfg.transition_contact_sensor_cfg.resolve(self._env.scene)
            self._transition_contact_sensor = self._env.scene.sensors[self.cfg.transition_contact_sensor_cfg.name]
            body_ids = self.cfg.transition_contact_sensor_cfg.body_ids
            num_feet = (
                len(range(self._transition_contact_sensor.num_sensors)[body_ids])
                if isinstance(body_ids, slice)
                else len(body_ids)
            )
            if num_feet != 4:
                raise ValueError("transition_contact_sensor_cfg must select exactly four feet.")

    def _selected_transition_ids(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        """선택한 환경 중 전환 전용 환경의 인덱스만 반환함."""
        selected = self._all_env_ids[env_ids]
        return selected[self.is_transition_env[selected]]

    def _resample_transition_commands(self, env_ids: Sequence[int] | slice) -> None:
        """고정된 역할의 다음 명령을 즉시 적용하고 전환 전용 타이머를 갱신함."""
        selected = self._selected_transition_ids(env_ids)
        if len(selected) == 0:
            return
        roles = self.transition_type[selected]
        phase = (self.command_counter[selected] + self._transition_phase_offset[selected]) % 4
        signs = self._transition_patterns[roles, phase]
        variants = self._transition_variants[selected]
        signs[:, 2] *= torch.where(variants & ((roles == 4) | (roles == 5)), -1, 1)
        for axis, bounds in enumerate(
            (self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)
        ):
            self.vel_command_b[selected, axis] = torch.where(
                signs[:, axis] > 0, bounds[1], torch.where(signs[:, axis] < 0, bounds[0], 0.0)
            )
        self.is_standing_env[selected] = False
        self.is_rotating_env[selected] = False
        self.is_heading_env[selected] = False
        self.time_left[selected] = self.time_left[selected].uniform_(*self.cfg.transition_resampling_time_range)
        # 최초 명령은 전환 횟수에 포함하지 않으며, 리셋 시 command_counter가 0으로 돌아감.
        changed = self.command_counter[selected] > 0
        self._transition_count[selected] += changed.float()
        self._transition_age[selected] = torch.where(changed, 0.0, math.inf)
        self._transition_pending_recovery[selected] = changed

    def _update_transition_metrics(self, error_xy: torch.Tensor, error_yaw: torch.Tensor) -> None:
        """전환 후 0.5초의 자세·추종·접촉 상태와 목표 속도 최초 도달 시간을 기록함.

        명령 매니저가 관측하는 비종료 스텝을 집계하며, 종료 직전 물리 스텝의 상태는 포함하지 않음.
        전환 직후의 종료 여부는 리셋 시 transition_failure_rate로 별도 기록함.
        """
        selected = self._transition_env_ids
        age = self._transition_age[selected]
        in_window = age < 0.5
        weight = in_window.float()
        self._transition_window_steps[selected] += weight
        self._transition_error_xy_sum[selected] += error_xy[selected] * weight
        self._transition_error_yaw_sum[selected] += error_yaw[selected] * weight
        gravity_z = self.robot.data.projected_gravity_b.torch[selected, 2]
        tilt_deg = torch.acos((-gravity_z).clamp(-1.0, 1.0)) * (180.0 / math.pi)
        self._transition_peak_tilt[selected] = torch.maximum(
            self._transition_peak_tilt[selected], torch.where(in_window, tilt_deg, 0.0)
        )
        if self._transition_contact_sensor is not None:
            contact = (
                self._transition_contact_sensor.data.net_normal_forces_w.torch[
                    :, self.cfg.transition_contact_sensor_cfg.body_ids, 2
                ][selected]
                > 1.0
            )
            self._transition_airborne_steps[selected] += (~contact.any(dim=1)).float() * weight
        reached = (
            self._transition_pending_recovery[selected]
            & (error_xy[selected] < self.cfg.vel_xy_success_threshold)
            & (error_yaw[selected] < self.cfg.vel_yaw_success_threshold)
        )
        self._transition_recovered_count[selected] += reached.float()
        self._transition_recovery_sum[selected] += torch.where(reached, age + self._env.step_dt, 0.0)
        self._transition_pending_recovery[selected] &= ~reached
        self._transition_age[selected] += self._env.step_dt

    def _reset_transition_metrics(self, env_ids: Sequence[int] | slice) -> dict[str, torch.Tensor]:
        """리셋하는 전환 환경만 집계하여 일반 환경의 0 값으로 지표가 희석되지 않도록 함."""
        selected = self._selected_transition_ids(env_ids)
        if len(selected) == 0:
            return {}
        count = self._transition_count[selected].sum()
        window_steps = self._transition_window_steps[selected].sum().clamp_min(1.0)
        recovered = self._transition_recovered_count[selected].sum()
        metrics = {
            "transition_count": count,
            "transition_error_vel_xy": self._transition_error_xy_sum[selected].sum() / window_steps,
            "transition_error_vel_yaw": self._transition_error_yaw_sum[selected].sum() / window_steps,
            "transition_peak_tilt_deg": self._transition_peak_tilt[selected].max(),
            "transition_recovery_time_s": self._transition_recovery_sum[selected].sum() / recovered.clamp_min(1.0),
            "transition_recovery_rate": recovered / count.clamp_min(1.0),
        }
        if self._transition_contact_sensor is not None:
            metrics["transition_all_feet_airborne_ratio"] = (
                self._transition_airborne_steps[selected].sum() / window_steps
            )
        if hasattr(self._env, "reset_terminated"):
            recent = self._transition_age[selected] < 0.5
            failures = self._env.reset_terminated[selected] & recent
            metrics["transition_failure_rate"] = failures.float().sum() / count.clamp_min(1.0)
        for buffer in (
            self._transition_count,
            self._transition_window_steps,
            self._transition_error_xy_sum,
            self._transition_error_yaw_sum,
            self._transition_peak_tilt,
            self._transition_airborne_steps,
            self._transition_recovery_sum,
            self._transition_recovered_count,
        ):
            buffer[selected] = 0.0
        self._transition_age[selected] = math.inf
        self._transition_pending_recovery[selected] = False
        return metrics

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.torch.clone()
        offset = self.cfg.marker_pos_offset
        base_pos_w[:, 0] += offset[0]
        base_pos_w[:, 1] += offset[1]
        base_pos_w[:, 2] += offset[2]
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b.torch[:, :2]
        )
        # display markers
        environment_ids = self._env.scene._ALL_INDICES
        self.goal_vel_visualizer.visualize(
            base_pos_w,
            vel_des_arrow_quat,
            vel_des_arrow_scale,
            environment_ids=environment_ids,
        )
        self.current_vel_visualizer.visualize(
            base_pos_w,
            vel_arrow_quat,
            vel_arrow_scale,
            environment_ids=environment_ids,
        )

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


class NormalVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: NormalVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: NormalVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        if cfg.rel_transition_envs != 0.0:
            raise ValueError("Fixed transition environments are supported only by UniformVelocityCommand.")
        super().__init__(cfg, env)
        # create buffers for zero commands envs
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _resample_command(self, env_ids):
        # sample velocity commands
        num_envs = len(range(self.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        r = torch.empty(num_envs, device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.normal_(mean=self.cfg.ranges.mean_vel[0], std=self.cfg.ranges.std_vel[0])
        self.vel_command_b[env_ids, 0] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.normal_(mean=self.cfg.ranges.mean_vel[1], std=self.cfg.ranges.std_vel[1])
        self.vel_command_b[env_ids, 1] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)
        # -- angular velocity - yaw direction
        self.vel_command_b[env_ids, 2] = r.normal_(mean=self.cfg.ranges.mean_vel[2], std=self.cfg.ranges.std_vel[2])
        self.vel_command_b[env_ids, 2] *= torch.where(r.uniform_(0.0, 1.0) <= 0.5, 1.0, -1.0)

        # update element wise zero velocity command
        # TODO what is zero prob ?
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Sets velocity command to zero for standing envs."""
        # Enforce standing (i.e., zero velocity command) for standing envs
        index_fill_(self.vel_command_b, self.is_standing_env, 0.0)

        # Enforce zero velocity for individual elements
        index_fill_(self.vel_command_b[:, 0], self.is_zero_vel_x_env, 0.0)
        index_fill_(self.vel_command_b[:, 1], self.is_zero_vel_y_env, 0.0)
        index_fill_(self.vel_command_b[:, 2], self.is_zero_vel_yaw_env, 0.0)
