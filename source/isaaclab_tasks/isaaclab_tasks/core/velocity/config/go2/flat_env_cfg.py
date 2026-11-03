# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree Go2 velocity-tracking environment on flat terrain."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .rough_env_cfg import UnitreeGo2Rewards, UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatRewards(UnitreeGo2Rewards):
    """Flat-terrain rewards for the Unitree Go2."""

    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=-10.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "target_height": 0.08,
            "min_foot_speed": 0.05,
            # The flat terrain importer spawns the ground plane at world Z = 0.
            "plane_z": 0.0,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
        },
    )

    stand_still_joint_velocity = RewTerm(
        func=mdp.stand_still_joint_velocity,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "velocity_sigma": 1.0,
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    stand_still_foot_position = RewTerm(
        func=mdp.stand_still_foot_position_l2,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "default_foot_positions": [
                (0.220, 0.145),  # FL
                (0.220, -0.145),  # FR
                (-0.170, 0.145),  # RL
                (-0.170, -0.145),  # RR
            ],
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
        },
    )


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    """Configuration for the Unitree Go2 velocity-tracking environment on flat terrain."""

    rewards: UnitreeGo2FlatRewards = UnitreeGo2FlatRewards()

    def __post_init__(self):
        super().__post_init__()

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

        # --- rewards ---

        # tasks
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # # essential penalty
        self.rewards.termination.weight = -1.0
        self.rewards.collisions.weight = -1.0
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -4.5e-7
        self.rewards.dof_pos_limits.weight = -10.0
        self.rewards.stand_still_contact.weight = -0.5

        self.rewards.variable_position.weight = 1.0
        self.rewards.trot_gait_pattern.weight = 0.5

        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.ang_vel_xy_l2.weight = -0.2
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_slide.weight = -0.1
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.smoothness.weight = -0.01

        self.rewards.base_height_l2.weight = -20.0
        self.rewards.base_height_l2.params["sensor_cfg"] = None

        # curriculum
        self.curriculum.terrain_levels = None
