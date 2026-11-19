# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """스케일 적용 전 환경 입력을 ±10으로 제한하며 이전 action 관측에도 제한된 값을 사용함.

    평지 설정도 이 값을 상속하며 정책이 샘플링한 원래 출력은 PPO와 진단 기록에 유지함.
    """

    num_steps_per_env = 24  # 24 steps로 데이터를 수집하고, 1번 PPO 학습을 수행 (24 steps + PPO학습 = 1 iteration)
    init_at_random_ep_len = False
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2_rough"
    obs_groups = {
        "actor": ["actor_obs"],
        "critic": ["critic_base_obs", "privileged_obs"],
    }
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    clip_actions = 10.0     # raw_action에 대한 clamping 진행


@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    init_at_random_ep_len = True

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 20000  # 5000
        self.experiment_name = "unitree_go2_flat"
        # self.actor.hidden_dims = [128, 128, 128]
        # self.critic.hidden_dims = [128, 128, 128]
