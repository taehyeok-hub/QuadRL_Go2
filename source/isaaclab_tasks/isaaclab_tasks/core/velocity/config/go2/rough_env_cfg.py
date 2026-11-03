# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree Go2 velocity-tracking environment on rough terrain."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from ... import mdp
from ...velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

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

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={"target_height": _GO2_STANDING_HEIGHT},
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

    stand_still_joint_position = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    stand_still_contact = RewTerm(
        func=mdp.stand_still_contact,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.05,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
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
                ".*_hip_joint": 0.15,
                ".*_thigh_joint": 0.35,
                ".*_calf_joint": 0.5,
            },
            "command_threshold": 0.05,
        },
    )


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

    # --- rewards ---
    rewards: UnitreeGo2Rewards = UnitreeGo2Rewards()

    def __post_init__(self):
        super().__post_init__()

        # simulation
        # execute the DC motor actuators through the backend-native path
        self.sim.use_newton_actuators = True

        # scene
        self.scene.robot = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # scale down the terrains because the robot is small
        terrains = self.scene.terrain.terrain_generator.sub_terrains
        terrains["boxes"].grid_height_range = (0.025, 0.1)
        terrains["random_rough"].noise_range = (0.01, 0.06)
        terrains["random_rough"].noise_step = 0.01
        terrains["pyramid_stairs"].step_height_range = (0.025, 0.12)
        terrains["pyramid_stairs_inv"].step_height_range = (0.025, 0.12)

        # actions
        self.actions.joint_pos.scale = 0.25

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # base_height_l2: rough terrain, adjust target height using the height-scanner rays.
        self.rewards.base_height_l2.weight = -30.0
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner")

        # action_rate_l2: halved from the shared default of -0.01. The stronger penalty locked
        # one hind foot into a low-amplitude, dragging gait on flat terrain with Newton.
        self.rewards.action_rate_l2.weight = -0.005

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
