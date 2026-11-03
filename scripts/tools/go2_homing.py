# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect two Go2 homing poses in the existing Gymnasium flat-terrain environment.

Run ``uv run python scripts/tools/go2_homing.py`` to open the Newton GL viewer.
Press N to replay both poses, 1 or 2 to move to one pose, and Escape to exit.
The final pose stays under position control until the window is closed.

For a bounded run without a window::

    uv run python scripts/tools/go2_homing.py --visualizer none --steps 650 --fast

This tool does not load a policy or communicate with a physical robot.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import time
from collections.abc import Mapping, Sequence

import torch

logger = logging.getLogger("isaaclab.tools.go2_homing")

# User settings: each leg has independent (hip, thigh, calf).
prepare_position = {
    "FL": (0.0, 1.20, -2.60),   # [rad]
    "FR": (0.0, 1.20, -2.60),
    "RL": (0.0, 1.20, -2.60),
    "RR": (0.0, 1.20, -2.60),
}
homing_position = {
    "FL": (0.0, 0.920, -1.850),
    "FR": (0.0, 0.920, -1.850),
    "RL": (0.0, 0.920, -1.850),
    "RR": (0.0, 0.920, -1.850),
}
kp = 100.0  
kd = 2.0 

# Joint-space PD, evaluated by the actuator at every 0.001 s physics step:
#   tau = kp * (q_des - q) - kd * dq



def go2_joint_pose(
    joint_names: Sequence[str], positions: Mapping[str, Sequence[float]], reference: torch.Tensor
) -> torch.Tensor:
    """각 다리의 관절각을 선택된 Go2 관절 순서에 맞춰 배열한다."""
    values = {
        f"{leg}_{joint}_joint": angle
        for leg, angles in positions.items()
        for joint, angle in zip(("hip", "thigh", "calf"), angles, strict=True)
    }
    return reference.new_tensor([values[name] for name in joint_names])


class Go2HomingController:
    """지정한 시간 동안 두 자세로 이동하고 각 자세를 유지하기 위한 관절 목표값을 생성한다.

    모든 환경은 동시에 시작한다. ``stage``는 준비 자세로 이동할 때 0,
    준비 자세를 유지할 때 1, 호밍 자세로 이동할 때 2, 호밍 자세를 유지할 때 3이다.
    단계 3은 목표 궤적이 끝났음을 나타내며, 실제 로봇이 안정화되었음을 의미하지는 않는다.
    이 컨트롤러는 액추에이터 토크를 계산하지 않는다.
    """

    stage_names = ("MOVE_TO_PREPARE", "HOLD_PREPARE", "MOVE_TO_HOMING", "HOLD_HOMING")

    def __init__(
        self,
        q_initial: torch.Tensor,
        prepare_position: torch.Tensor,
        homing_position: torch.Tensor,
        move_time: float = 4.0,
        hold_time: float = 2.0,
    ):
        if not math.isfinite(move_time) or move_time <= 0.0:
            raise ValueError("move_time must be finite and positive.")
        if not math.isfinite(hold_time) or hold_time < 0.0:
            raise ValueError("hold_time must be finite and nonnegative.")
        self.prepare_position = prepare_position.to(q_initial).expand_as(q_initial).clone()
        self.homing_position = homing_position.to(q_initial).expand_as(q_initial).clone()
        self.q_start = q_initial.clone()
        self.stage = torch.zeros(q_initial.shape[0], dtype=torch.long, device=q_initial.device)
        self.elapsed = q_initial.new_zeros(q_initial.shape[0])
        self.move_time = move_time
        self.hold_time = hold_time
        self._sequence = True

    def start(self, q_current: torch.Tensor, pose: str = "sequence") -> None:
        """측정된 현재 관절 위치에서 한 자세 또는 두 자세로의 이동을 다시 시작한다."""
        if pose not in ("sequence", "prepare_position", "homing_position"):
            raise ValueError(f"Unknown homing pose: {pose}")
        self.q_start.copy_(q_current)
        self.elapsed.zero_()
        self.stage.fill_(2 if pose == "homing_position" else 0)
        self._sequence = pose == "sequence"

    def compute(self, dt: float) -> torch.Tensor:
        """시뮬레이션 시간(초)을 진행하고 절대 관절 목표값을 반환한다."""
        self.elapsed += dt
        moving = (self.stage == 0) | (self.stage == 2)
        target = torch.where((self.stage < 2)[:, None], self.prepare_position, self.homing_position)
        progress = (self.elapsed / self.move_time).clamp(0.0, 1.0)
        alpha = 0.5 * (1.0 - torch.cos(torch.pi * progress))
        interpolated = self.q_start + alpha[:, None] * (target - self.q_start)
        q_des = torch.where(moving[:, None], interpolated, target)

        move_done = moving & (self.elapsed >= self.move_time)
        hold_done = (self.stage == 1) & (self.elapsed >= self.hold_time) & self._sequence
        self.stage[move_done] += 1
        self.elapsed[move_done] = 0.0
        # 이전 목표 자세에서 다음 이동을 이어가 기준 궤적을 유지한다.
        self.q_start[hold_done] = self.prepare_position[hold_done]
        self.stage[hold_done] = 2
        self.elapsed[hold_done] = 0.0
        return q_des


def _configure_environment(cfg, args: argparse.Namespace) -> None:
    """Go2의 장면과 동작 명령 경로를 유지하면서 강화학습 에피소드의 부수 효과를 제거한다."""
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    # 네이티브 명시적 PD에서 호밍 게인(kp=100, kd=2)에 맞춰 물리 갱신 간격을 줄인다.
    # 관절 목표값은 기존과 동일하게 50 Hz로 갱신한다.
    cfg.sim.dt = 0.001
    cfg.decimation = 20
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.curriculum = {}
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.reset_base.params["pose_range"] = {}
    cfg.events.reset_base.params["velocity_range"] = {}
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    command = cfg.commands.base_velocity
    command.rel_standing_envs = 1.0
    command.rel_rotating_envs = 0.0
    command.heading_command = False
    command.debug_vis = False
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    cfg.actions.joint_pos.clip = None
    # 위 관절 공간 PD 식에서 stiffness는 kp, damping은 kd에 해당한다.
    cfg.scene.robot.actuators["base_legs"].stiffness = args.kp
    cfg.scene.robot.actuators["base_legs"].damping = args.kd


def _run_homing(env, args: argparse.Namespace) -> None:
    """호밍 목표 자세를 선택하고 Gym의 위치 동작 명령으로 전달한다."""
    from isaaclab.visualizers.key_event_source import add_key_callbacks

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    action_cfg = base_env.cfg.actions.joint_pos
    joint_ids, joint_names = robot.find_joints(action_cfg.joint_names, preserve_order=action_cfg.preserve_order)
    q_initial = robot.data.joint_pos.torch[:, joint_ids].clone()
    q_default = robot.data.default_joint_pos.torch[:, joint_ids].clone()
    prepare_target = go2_joint_pose(joint_names, prepare_position, q_initial)
    homing_target = go2_joint_pose(joint_names, homing_position, q_initial)
    limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
    for target in (prepare_target, homing_target):
        if ((target < limits[..., 0]) | (target > limits[..., 1])).any().item():
            raise ValueError("A homing pose exceeds the configured soft joint limits.")
    controller = Go2HomingController(q_initial, prepare_target, homing_target, args.move_time, args.hold_time)
    requests = [args.pose]
    remove_keys = add_key_callbacks(
        base_env.sim.visualizers,
        {
            "N": lambda: requests.append("sequence"),
            "1": lambda: requests.append("prepare_position"),
            "2": lambda: requests.append("homing_position"),
            "Escape": lambda: requests.append("quit"),
        },
    )
    logger.info("Go2 homing: N=sequence, 1=prepare_position, 2=homing_position, Escape=exit.")
    logger.info(
        "Joint-space PD: tau = %.3f * (q_des - q) - %.3f * dq; then DC motor torque clipping.", args.kp, args.kd
    )
    scale = float(action_cfg.scale)
    previous_stage = -1
    step_count = 0
    q_des = q_initial
    try:
        with torch.inference_mode():
            while base_env.sim.is_running() and (args.steps == 0 or step_count < args.steps):
                start_time = time.monotonic()
                if requests:
                    pose = requests.pop()
                    requests.clear()
                    if pose == "quit":
                        break
                    controller.start(robot.data.joint_pos.torch[:, joint_ids], pose)
                    previous_stage = -1
                if not base_env.sim.is_playing():
                    base_env.sim.render()
                    time.sleep(0.01)
                    continue
                q_des = controller.compute(base_env.step_dt)
                # 원본 Gym 환경을 사용하므로 정책 래퍼가 이 동작 명령을 제한하지 않는다.
                actions = (q_des - q_default) / scale
                _, _, terminated, truncated, _ = env.step(actions)
                if (terminated | truncated).any().item():
                    raise RuntimeError("The pose-viewing environment unexpectedly reset during homing.")
                stage = int(controller.stage[0].item())
                if stage != previous_stage:
                    logger.info("Go2 homing phase: %s", controller.stage_names[stage])
                    previous_stage = stage
                step_count += 1
                if not args.fast:
                    time.sleep(max(0.0, base_env.step_dt - (time.monotonic() - start_time)))
    finally:
        remove_keys()
    error = (robot.data.joint_pos.torch[:, joint_ids] - q_des).abs().max().item()
    logger.info(
        "Homing finished: steps=%d phase=%s max_joint_error=%.6f rad",
        step_count,
        controller.stage_names[int(controller.stage[0].item())],
        error,
    )


def main(argv: list[str] | None = None) -> int:
    """등록된 학습 태스크를 변경하지 않고 Go2 자세 뷰어를 실행한다."""
    import warp as wp

    wp.config.enable_backward = False
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--move-time", type=float, default=5.0, help="Seconds per homing move.")
    parser.add_argument("--hold-time", type=float, default=2.0, help="Seconds to hold prepare_position before homing.")
    parser.add_argument("--pose", choices=("sequence", "prepare_position", "homing_position"), default="sequence")
    parser.add_argument("--kp", type=float, default=kp, help="Homing position gain [N m/rad].")
    parser.add_argument("--kd", type=float, default=kd, help="Homing velocity gain [N m s/rad].")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps; 0 holds indefinitely.")
    parser.add_argument("--fast", action="store_true", help="Do not pace the simulation to real time.")
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 0:
        parser.error("num-envs must be positive and steps must be nonnegative.")
    if not all(math.isfinite(value) and value > 0.0 for value in (args.move_time, args.kp)):
        parser.error("move-time and kp must be finite and positive.")
    if not all(math.isfinite(value) and value >= 0.0 for value in (args.hold_time, args.kd)):
        parser.error("hold-time and kd must be finite and nonnegative.")
    if args.visualizer is None:
        args.visualizer = ["newton_gl"]

    import gymnasium as gym

    from isaaclab.visualizers import VisualizerCfg

    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("Isaac-Velocity-Flat-UnitreeGo2", None, play_mode=True, overrides=[])
    _configure_environment(cfg, args)
    cfg.sim.visualizer_cfgs = [VisualizerCfg(eye=(1.2, -1.2, 0.8), lookat=(0.0, 0.0, 0.2))]
    with launch_simulation(cfg, args):
        with gym.make("Isaac-Velocity-Flat-UnitreeGo2", cfg=cfg) as env:
            env.reset()
            with contextlib.suppress(KeyboardInterrupt):
                _run_homing(env, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
