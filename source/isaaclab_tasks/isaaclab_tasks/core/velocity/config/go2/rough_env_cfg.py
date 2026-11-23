# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree Go2 velocity-tracking environment on rough terrain."""

import math

from isaaclab.envs.mdp import joint_pos_out_of_limit
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from ... import mdp
from ...mdp.curriculum.success import SuccessTerrainCfg, SuccessTerrainCurriculum
from ...mdp.curriculum.terminations import assigned_terrain_boundary
from ...mdp.metrics import LocomotionMetricsCfg
from ...velocity_env_cfg import LocomotionVelocityRoughEnvCfg, ObservationsCfg, RewardsCfg, TerminationsCfg

# UNITREE_GO2_CFG.init_state.pos is (0, 0, 0.28)
_GO2_STANDING_HEIGHT = 0.28


@configclass
class UnitreeGo2Rewards(RewardsCfg):
    """Unitree Go2 환경에서 사용할 보상 항목과 기본 설정을 정의하는 클래스.

    공통 보상 설정인 RewardsCfg를 상속받아 기존 보상 항목을 유지하고, Go2 환경에 필요한 보상 항목을 추가한다.

    각 항목에는 사용할 보상 함수(func), 가중치(weight), 함수에 전달할
    설정값(params)을 지정한다. 실제 보상값 계산은 func에 지정된 함수가 수행한다.

    이 클래스의 인스턴스는 UnitreeGo2RoughEnvCfg의 rewards 속성에 연결된다.
    환경 클래스에서는 self.rewards를 통해 여기에 정의된 항목에 접근하고,
    환경에 맞게 가중치나 설정값을 변경할 수 있다.

    보상 항목의 정의를 한곳에 모아 관리하고, 로봇·지형·액션 등의 환경 설정과
    구분하여 재사용하거나 수정하기 쉽게 만드는 것이 목적이다.
    """

    termination = RewTerm(
        func=mdp.is_terminated,  # func=mdp.terminated_penalty,
        weight=0.0,
    )

    smoothness = RewTerm(
        func=mdp.smoothness,
        weight=0.0,
        params={"action_name": "joint_pos"},
    )

    base_height = RewTerm(
        func=mdp.base_height_exp,
        weight=0.0,
        params={
            "target_height": _GO2_STANDING_HEIGHT,
            "height_sigma": 0.005,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    trot_gait_pattern = RewTerm(
        func=mdp.trot_gait_pattern,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "airborne_penalty": 1.0,
            "balance_weight": 0.5,
            "ema_alpha": 0.99,
        },
    )

    feet_clearance = RewTerm(
        func=mdp.feet_clearance,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "target_height": 0.08,
            "min_foot_speed": 0.05,
            # Use world Z = 0 only when no terrain rays hit; Flat disables the sensor.
            "plane_z": 0.0,
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            ),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            ),
            "threshold": 0.5,
        },
    )

    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            ),
            "max_time": 0.5,
        },
    )

    foot_alignment = RewTerm(
        func=mdp.foot_alignment,
        weight=0.0,
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
        weight=0.0,
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

    variable_position = RewTerm(
        func=mdp.variable_position,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_joint",
                    ".*_thigh_joint",
                    ".*_calf_joint",
                ],
            ),
            "std_standing": {
                ".*_hip_joint": 0.05,
                ".*_thigh_joint": 0.1,
                ".*_calf_joint": 0.15,
            },
            "std_moving": {
                ".*_hip_joint": 0.25,  # 0.15
                ".*_thigh_joint": 0.60,  # 0.35
                ".*_calf_joint": 0.50,
            },
            "command_threshold": 0.05,
        },
    )

    yaw_footplacement = RewTerm(
        func=mdp.yaw_footplacement,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )


@configclass
class UnitreeGo2TerminationsCfg(TerminationsCfg):
    """Go2 평지·험지 환경에 공통으로 적용하는 종료 조건.

    몸통 +Z축과 월드 +Z축 사이의 기울기가 90도를 초과하면 접촉 여부와 관계없이 종료함.
    관절 하나라도 로봇 모델의 물리적 위치 하한이나 상한을 벗어나면 종료함.
    """

    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(90.0)})
    joint_pos_out_of_limit = DoneTerm(func=joint_pos_out_of_limit, params={"use_soft_limits": False})


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Unitree Go2의 험지 속도 추종 학습 환경 전체를 구성하는 클래스.

    공통 보행 환경 설정인 LocomotionVelocityRoughEnvCfg를 상속받아,
    물리 엔진, 로봇 모델, 지형, 액션, 보상, 종료 조건 등을 Go2에 맞게 설정한다.

    rewards 속성에는 UnitreeGo2Rewards()로 생성한 보상 설정을 넣는다.
    따라서 self.rewards는 UnitreeGo2Rewards에 정의된 보상 항목과
    RewardsCfg에서 상속받은 공통 보상 항목을 포함한다.

    __post_init__에서는 부모 환경의 초기 설정을 적용한 뒤,
    Go2 험지 환경에 맞게 세부 설정을 조정한다.
    이 과정에서 self.rewards를 통해 기존 보상 항목의 가중치나 params를
    변경하며, 여기서 변경한 값이 해당 환경에서 사용할 최종 설정이 된다.

    보상 항목의 정의는 UnitreeGo2Rewards에서 관리하고,
    이 클래스에서는 그 보상을 로봇·지형·액션과 함께 조합하여
    실제 학습에 사용할 환경 구성을 완성하는 것이 목적이다.
    """

    observations: ObservationsCfg = ObservationsCfg(
        actor_obs=ObservationsCfg.ActorObsCfg(last_actions=None, last_last_actions=None),
        critic_base_obs=ObservationsCfg.CriticBaseObsCfg(last_actions=None, last_last_actions=None),
    )

    # --- rewards ---
    rewards: UnitreeGo2Rewards = UnitreeGo2Rewards()

    # --- termination ---
    terminations: UnitreeGo2TerminationsCfg = UnitreeGo2TerminationsCfg()

    # Physical gait and tracking telemetry, independent of rewards and curriculum.
    locomotion_metrics: LocomotionMetricsCfg = LocomotionMetricsCfg()

    def __post_init__(self):
        super().__post_init__()

        # simulation
        self.sim.use_newton_actuators = True

        # scene
        self.scene.robot = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # scale down the terrains because the robot is small
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["boxes"].grid_height_range = (0.025, 0.1)
        terrains["pyramid_stairs"].step_height_range = (0.05, 0.20)
        terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.20)
        terrains["random_rough"].noise_range = (0.01, 0.06)
        terrains["random_rough"].noise_step = 0.01

        # Use two-trial success histories to update terrain difficulty.
        self.curriculum.learning_progress = None
        self.curriculum.terrain_levels = CurrTerm(
            func=SuccessTerrainCurriculum, params={"settings": SuccessTerrainCfg()}
        )
        self.commands.base_velocity = replace(
            self.commands.base_velocity,
            class_type=mdp.UniformVelocityCommand,
            rel_standing_envs=0.025,
            rel_rotating_envs=0.025,
            rel_transition_envs=0.0,
            heading_command=False,
        )
        self.scene.terrain.terrain_generator.curriculum = True
        self.terminations.assigned_terrain_boundary = DoneTerm(func=assigned_terrain_boundary, time_out=True)

        # --- actions ---
        self.actions.joint_pos.scale = 0.25

        # -- commands ---
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)  # x, m/s
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)  # y, m/s
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)  # yaw, rad/s

        # --- rewards ---
        # tasks
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        self.rewards.termination.weight = -1.0
        self.rewards.collisions.weight = -1.0  # -1.0 에서 변경
        # --> collision = 0.0 변경 시, iteration time이 커지는 경향 발생.
        
        self.rewards.dof_pos_limits.weight = -30.0
        self.rewards.dof_vel_limits.weight = 0.0
        self.rewards.dof_acc_l2.weight = -4.5e-7  # -4.5e-7 에서 변경
        self.rewards.dof_torques_l2.weight = -0.0001

        self.rewards.stand_still_contact.weight = -0.5
        self.rewards.stand_still_foot_position.weight = -10.0

        self.rewards.flat_orientation_l2.weight = -0.0  # -5.0
        self.rewards.ang_vel_xy_l2.weight = -0.1    # 0.5
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.smoothness.weight = -0.01
        # self.rewards.base_height.weight = 2.0

        self.rewards.variable_position.weight = 1.0
        self.rewards.trot_gait_pattern.weight = 0.5
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time_variance.weight = -1.0
        # self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_clearance.weight = -5.0
        self.rewards.foot_alignment.weight = 0.05
        self.rewards.foot_misalignment.weight = -0.05

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"

    def play_mode(self):
        """Use ordinary velocity commands for interactive policy playback."""
        super().play_mode()
        self.curriculum.terrain_levels = None
        self.curriculum.learning_progress = None
        self.commands.base_velocity.class_type = mdp.UniformVelocityCommand
        self.terminations.assigned_terrain_boundary = None
