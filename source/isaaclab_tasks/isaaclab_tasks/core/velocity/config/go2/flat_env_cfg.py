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
    """Go2 평지 보상 항목을 정의함.

    몸통 높이 보상은 ``2 * exp(-(h - 0.30) ** 2 / 0.005)`` 형태임.
    목표 높이는 0.30 m이며, 시간 배율 적용 전 최대 보상은 +2임.
    제어 시간 간격 0.02초에서는 최대 +0.04를 한 스텝의 보상에 반영함.
    상속받은 높이 제곱 페널티는 평지 환경 설정에서 해제해 중복 적용을 방지함.
    """

    base_height = RewTerm(
        func=mdp.base_height_exp,
        weight=0.0,
        params={"target_height": 0.28, "height_sigma": 0.005},
    )

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

    foot_alignment = RewTerm(
        func=mdp.foot_alignment,
        weight=0.05,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "min_foot_speed": 0.03,
            "contact_threshold": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
        },
    )

    foot_misalignment = RewTerm(
        func=mdp.foot_misalignment,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "min_foot_speed": 0.03,
            "min_base_speed": 0.05,
            "contact_threshold": 1.0,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True
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
        # observation noise
        self.observations.actor_obs.enable_corruption = False

        # --- events ---
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {}
        self.events.reset_base.params["velocity_range"] = {}
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # --- rewards ---
        # tasks
        self.rewards.track_lin_vel_xy_exp.weight = 5.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # essential penalty
        self.rewards.termination.weight = -1.0
        self.rewards.collisions.weight = -1.0
        self.rewards.dof_torques_l2.weight = -0.0001
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
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.smoothness.weight = -0.01

        self.rewards.base_height_l2 = None
        self.rewards.base_height = 2.0

        # curriculum
        self.curriculum.terrain_levels = None
