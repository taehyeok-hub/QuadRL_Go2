# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read existing locomotion sensors without adding rewards or policy observations."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from .gait import GaitMetrics
from .velocity import VelocityMetrics


@configclass
class LocomotionMetricsCfg:
    """Optional telemetry shared by training and checkpoint playback."""

    class_type: str = "isaaclab_tasks.core.velocity.mdp.metrics:LocomotionMetricObserver"
    enabled: bool = True
    asset_name: str = "robot"
    command_name: str = "base_velocity"
    contact_sensor_name: str = "contact_forces"
    height_sensor_name: str = "height_scanner"
    foot_body_pattern: str = ".*_foot"
    max_ground_sample_distance: float = 0.1
    """Maximum XY distance [m] to an existing terrain ray hit; farther hits are invalid."""
    plane_z: float = 0.0
    """Ground Z [m] used only for an explicitly configured plane terrain."""
    playback_interval_steps: int = 24
    debug_metrics: bool = False


def foot_ground_heights(
    foot_positions: torch.Tensor, ray_hits: torch.Tensor, max_distance: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate foot-link height [m] using the nearest valid terrain sample in XY."""
    valid_hits = torch.isfinite(ray_hits).all(dim=-1)
    distance_squared = (foot_positions[:, :, None, :2] - ray_hits[:, None, :, :2]).square().sum(dim=-1)
    distance_squared = distance_squared.masked_fill(~valid_hits[:, None, :], torch.inf)
    distance_squared, indices = distance_squared.min(dim=-1)
    ground_z = ray_hits[:, :, 2].gather(1, indices)
    height = foot_positions[:, :, 2] - ground_z
    valid = (distance_squared <= max_distance**2) & torch.isfinite(height)
    return height, valid


class LocomotionMetricObserver:
    """Bind robot feet by name and sample the physical state before automatic resets."""

    def __init__(self, cfg: LocomotionMetricsCfg, env):
        if cfg.max_ground_sample_distance <= 0 or cfg.playback_interval_steps < 1:
            raise ValueError("Metric ground distance and playback interval must be positive.")
        self.cfg = cfg
        self._env = env
        self._robot = env.scene[cfg.asset_name]
        self._contacts = env.scene[cfg.contact_sensor_name]
        self._foot_ids, names = self._robot.find_bodies(cfg.foot_body_pattern)
        if not names or not self._contacts.cfg.track_air_time:
            raise ValueError("Gait metrics require foot bodies and contact track_air_time=True.")
        self._contact_ids, contact_names = self._contacts.find_bodies(names, preserve_order=True)
        if contact_names != names:
            raise ValueError("Gait contact sensor must contain every robot foot in matching order.")
        self._height_sensor = env.scene.sensors.get(cfg.height_sensor_name)
        self._plane = env.scene.terrain.cfg.terrain_type == "plane"
        if self._height_sensor is None and not self._plane:
            raise ValueError("Rough-terrain gait heights require an existing height scanner.")
        self._gait = GaitMetrics(env.num_envs, names, str(env.device), debug_metrics=cfg.debug_metrics)
        self._velocity = VelocityMetrics(str(env.device))

    def reset(self, env_ids):
        """Forget reset-truncated swings, keeping the current reporting window."""
        self._gait.reset(env_ids)

    @torch.no_grad()
    def sample(self, dt: float):
        """Use the still-applied command and pre-reset physical state for this step."""
        data = self._robot.data
        actual = torch.cat((data.root_lin_vel_b.torch[:, :2], data.root_ang_vel_b.torch[:, 2:3]), dim=1)
        self._velocity.sample(actual, self._env.command_manager.get_command(self.cfg.command_name), dt)
        gravity = data.projected_gravity_b.torch
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        pitch = torch.atan2(gravity[:, 0], gravity[:, 1:].square().sum(dim=1).sqrt())
        positions = data.body_link_pos_w.torch[:, self._foot_ids]
        if self._plane:
            height = positions[:, :, 2] - self.cfg.plane_z
            valid = torch.isfinite(height)
        else:
            height, valid = foot_ground_heights(
                positions, self._height_sensor.data.ray_hits_w.torch, self.cfg.max_ground_sample_distance
            )
        contacts = self._contacts.data
        forces = contacts.net_normal_forces_w.torch[:, self._contact_ids]
        contact = forces.square().sum(dim=-1) > self._contacts.cfg.force_threshold**2
        speed = data.body_link_lin_vel_w.torch[:, self._foot_ids, :2].norm(dim=-1)
        self._gait.sample(
            data.root_link_lin_vel_w.torch[:, 2],
            torch.stack((roll, pitch), dim=1),
            contact,
            height,
            valid,
            speed,
            contacts.last_air_time.torch[:, self._contact_ids],
            dt,
        )

    def pop(self) -> dict[str, float]:
        """Return Gait/* and Velocity/* scalar values for the current reporting window."""
        return self._gait.pop() | self._velocity.pop()
