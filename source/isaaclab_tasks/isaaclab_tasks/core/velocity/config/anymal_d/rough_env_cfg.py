# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ANYmal-D velocity-tracking environment on rough terrain."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

from ... import mdp
from ...velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg


@configclass
class AnymalDRewards(RewardsCfg):
    """Reward configuration for the ANYmal-D velocity-tracking environments."""

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )


@configclass
class AnymalDRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the ANYmal-D velocity-tracking environment on rough terrain."""

    rewards: AnymalDRewards = AnymalDRewards()

    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = replace(
            ANYMAL_D_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=replace(ANYMAL_D_CFG.init_state, pos=(0.0, 0.0, 0.65)),
        )

        # rewards
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*THIGH"
