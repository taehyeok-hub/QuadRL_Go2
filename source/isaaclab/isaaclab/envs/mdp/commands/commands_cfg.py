# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass, replace

if TYPE_CHECKING:
    from .null_command import NullCommand
    from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand
    from .pose_command import UniformPoseCommand
    from .velocity_command import NormalVelocityCommand, UniformVelocityCommand


@configclass
class NullCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""

    class_type: type["NullCommand"] | str = "{DIR}.null_command:NullCommand"

    def __post_init__(self):
        """Post initialization."""
        # set the resampling time range to infinity to avoid resampling
        self.resampling_time_range = (math.inf, math.inf)


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """균등 분포를 활용한 uniform velocity command generator 설정 클래스."""

    class_type: type["UniformVelocityCommand"] | str = "{DIR}.velocity_command:UniformVelocityCommand"

    asset_name: str = MISSING
    # 명령 생성의 대상이 되는 환경 내 로봇 또는 자산의 이름. 

    heading_command: bool = False
    # Heading 명령을 사용할지 여부를 결정하는 플래그. 기본값은 False임.
    #
    # True이면 지정된 범위에서 목표 방향을 균등하게 샘플링하고, 목표 방향과 현재 방향의 오차를 이용하여 각속도 명령을 계산함.
    # False이면 지정된 범위에서 각속도 명령을 직접 균등하게 샘플링함.


    heading_control_stiffness: float = 1.0
    # heading 오차를 각속도 명령으로 변환할 때 사용하는 비례계수, 기본값 = 1.0.

    rel_standing_envs: float = 0.0
    # 각 환경이 정지 상태로 선택될 확률.

    rel_rotating_envs: float = 0.0
    # 각 환경이 회전 상태로 선택될 확률.

    rel_heading_envs: float = 1.0
    # 각 환경이 목표 방향의 오차로 계산한 각속도 명령을 따르도록 선택될 확률. 
    # 나머지 환경은 직접 샘플링한 각속도 명령을 따름. 기본값은 1.0.

    # 이 설정은 heading_command가 True일 때만 사용됨.
    



    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    vel_xy_success_threshold: float = 0.5
    """Threshold on the per-episode mean XY velocity error norm [m/s]."""

    vel_yaw_success_threshold: float = 0.4
    """Threshold on the per-episode mean yaw velocity error [rad/s]."""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.5)
    """Offset [m] applied to the robot root position when placing velocity visualization markers.

    The default of ``(0.0, 0.0, 0.5)`` works well for quadrupeds. For taller robots such as
    humanoids, increase the Z component (e.g. ``(0.0, 0.0, 1.8)``) so the arrows appear above
    the robot's head rather than clipping through the torso.
    """

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = replace(
        GREEN_ARROW_X_MARKER_CFG, prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = replace(
        BLUE_ARROW_X_MARKER_CFG, prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type["NormalVelocityCommand"] | str = "{DIR}.velocity_command:NormalVelocityCommand"
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution (in m/s).

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """

        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution (in m/s).

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """

        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type["UniformPoseCommand"] | str = "{DIR}.pose_command:UniformPoseCommand"

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    position_success_threshold: float | None = None
    """If set, position-error norm [m] below this value (per step) is required for success.

    When both position and orientation thresholds are set, both conditions must be satisfied.
    The episode-level binary "ever successful" is mean-reduced across environments and logged
    under ``Metrics/success_rate``. Defaults to ``None``."""

    orientation_success_threshold: float | None = None
    """If set, orientation-error norm [rad] below this value (per step) is required for success.

    When both position and orientation thresholds are set, both conditions must be satisfied.
    The episode-level binary "ever successful" is mean-reduced across environments and logged
    under ``Metrics/success_rate``. Defaults to ``None``."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = replace(
        FRAME_MARKER_CFG, prim_path="/Visuals/Command/goal_pose"
    )
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = replace(
        FRAME_MARKER_CFG, prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class UniformPose2dCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type["UniformPose2dCommand"] | str = "{DIR}.pose_2d_command:UniformPose2dCommand"

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""

    position_success_threshold: float | None = None
    """If set, XY position-error norm below this value (per step) flags the episode as successful.

    The episode-level binary "ever within threshold" is mean-reduced across environments and
    logged under ``Metrics/success_rate``. Defaults to ``None`` (success tracking disabled)."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = replace(
        GREEN_ARROW_X_MARKER_CFG, prim_path="/Visuals/Command/pose_goal"
    )
    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.2, 0.2, 0.8)
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)


@configclass
class TerrainBasedPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the terrain-based position command generator."""

    class_type: type["TerrainBasedPose2dCommand"] | str = "{DIR}.pose_2d_command:TerrainBasedPose2dCommand"

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the sampled commands."""
