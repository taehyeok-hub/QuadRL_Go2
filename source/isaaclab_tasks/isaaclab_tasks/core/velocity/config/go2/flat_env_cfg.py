# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree Go2 velocity-tracking environment on flat terrain."""

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2Rewards, UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatRewards(UnitreeGo2Rewards):
    """Use the shared Go2 rewards without terrain-height sensor corrections."""

    def __post_init__(self):
        self.base_height.params["sensor_cfg"] = None
        self.feet_clearance.params["height_sensor_cfg"] = None


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    """Configuration for the Unitree Go2 velocity-tracking environment on flat terrain."""

    rewards: UnitreeGo2FlatRewards = UnitreeGo2FlatRewards()

    def __post_init__(self):
        regular_command = self.commands.base_velocity.copy()
        super().__post_init__()
        self.commands.base_velocity = regular_command
        self.curriculum.learning_progress = None
        self.terminations.assigned_terrain_boundary = None

        # --- physics ---
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 256
        newton_mjwarp.solver_cfg.nconmax = 35
        self.sim.physics.default = newton_mjwarp
        
        # --- scene ---
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        # --- observations ---
        self.observations.privileged_obs.height_scan = None
        # observation noise
        # self.observations.actor_obs.enable_corruption = False

        # --- commands ---
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 2.0)  # x, m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)  # y, m/s
        self.commands.base_velocity.ranges.ang_vel_z = (-3.0, 3.0)  # yaw, rad/s

        # --- events ---
        # self.events.add_base_mass = None
        # self.events.base_com = None
        # self.events.push_robot = None
        # self.events.reset_base.params["pose_range"] = {}
        # self.events.reset_base.params["velocity_range"] = {}
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # --- rewards ---
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.ang_vel_xy_l2.weight = -0.2
        self.rewards.base_height.weight = 0.0

        # curriculum
        self.curriculum.terrain_levels = None
