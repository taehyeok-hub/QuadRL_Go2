# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move a Go2 through two joint poses in a standalone scene and print measured foot positions.

Run ``uv run python scripts/tools/go2_homing.py`` to open the Newton GL viewer.
Edit the initial pose, target poses, individual periods, and PD gains below.
The final pose remains under position control until the window is closed.
``get_foot_positions(robot)`` returns measured foot link positions relative to the
body, in meters, with shape (num_envs, 4, 3) and leg order FL, FR, RL, RR.

For a bounded run without a window::

    uv run python scripts/tools/go2_homing.py --visualizer none --steps 6000 --fast
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

# User settings: each leg has independent (hip, thigh, calf), in radians.
# Initial joint angles must remain within the Go2 joint limits; calf=0 is outside them.
initial_position = {".*_hip_joint": 0.0, ".*_thigh_joint": 0.920, ".*_calf_joint": -1.850}
prepare_position = {
    "FL": (0.0, 1.20, -2.60),
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
# Each period is independently editable; 1000 physics steps at 0.001 s is one second.
prepare_period = 1000
homing_period = 1000
settle_period = 2000  # Hold the final target this long before reporting foot positions.
kp = 100.0
kd = 2.0


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
    """측정 관절각에서 두 목표 자세로 차례로 이동하는 관절 목표각 생성기."""

    def __init__(
        self,
        q_initial: torch.Tensor,
        prepare_position: torch.Tensor,
        homing_position: torch.Tensor,
        prepare_period: int = 2000,
        homing_period: int = 4000,
    ):
        if prepare_period < 1 or homing_period < 1:
            raise ValueError("prepare_period and homing_period must be positive.")
        self.homing_state1 = prepare_position.to(q_initial).expand_as(q_initial).clone()
        self.homing_state2 = homing_position.to(q_initial).expand_as(q_initial).clone()
        self.q_start = torch.zeros_like(q_initial)
        self.q_target = torch.zeros_like(q_initial)
        self.prepare_period = prepare_period
        self.homing_period = homing_period
        self.period = 0
        self.homing_step = 0
        self.homing_flag = 0

    @staticmethod
    def interpolation_factor(elapsed: float, duration: float) -> float:
        """0에서 1까지 반코사인 곡선을 따르는 보간 계수를 반환한다."""
        progress = min(max(elapsed / duration, 0.0), 1.0)
        return 0.5 * (1.0 - math.cos(math.pi * progress))

    def homing(self, act_q: torch.Tensor) -> torch.Tensor:
        """물리 한 step에 사용할 절대 관절 목표각 q_des를 반환한다."""
        if self.homing_flag == 0:
            self.q_start = act_q.clone()
            self.q_target = self.homing_state1
            self.period = self.prepare_period
            self.homing_step = 0
            self.homing_flag = 1

        elif self.homing_flag == 1:
            self.homing_step += 1
            if self.homing_step >= self.period:
                self.homing_flag = 2

        elif self.homing_flag == 2:
            self.q_start = act_q.clone()
            self.q_target = self.homing_state2
            self.period = self.homing_period
            self.homing_step = 0
            self.homing_flag = 3

        elif self.homing_flag == 3:
            self.homing_step += 1
            if self.homing_step >= self.period:
                self.homing_flag = 4

        else:
            return self.homing_state2

        alpha = self.interpolation_factor(self.homing_step, self.period)
        return self.q_start + (self.q_target - self.q_start) * alpha


def get_foot_positions(robot) -> torch.Tensor:
    """발 링크 원점의 몸통 기준 좌표를 (num_envs, 4, 3) 텐서로 반환한다.

    발 순서는 FL, FR, RL, RR이며, 각 행은 미터 단위의 (x, y, z)이다.
    """
    from isaaclab.utils.math import quat_apply_inverse

    foot_ids, _ = robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
    relative = robot.data.body_link_pos_w.torch[:, foot_ids] - robot.data.root_pos_w.torch[:, None, :]
    return quat_apply_inverse(robot.data.root_quat_w.torch[:, None, :], relative)


def _run_homing(sim, robot, args: argparse.Namespace) -> torch.Tensor:
    """보간한 q_des를 액추에이터에 직접 전달하고 물리 시뮬레이션을 진행한다."""
    joint_ids, joint_names = robot.find_joints(".*")
    q_initial = robot.data.joint_pos.torch[:, joint_ids].clone()
    prepare_target = go2_joint_pose(joint_names, prepare_position, q_initial)
    homing_target = go2_joint_pose(joint_names, homing_position, q_initial)
    limits = robot.data.soft_joint_pos_limits.torch[:, joint_ids]
    for target in (prepare_target, homing_target):
        if ((target < limits[..., 0]) | (target > limits[..., 1])).any().item():
            raise ValueError("A homing pose exceeds the configured soft joint limits.")
    controller = Go2HomingController(q_initial, prepare_target, homing_target, prepare_period, homing_period)
    sim_dt = sim.get_physics_dt()
    step_count = 0
    hold_steps = 0
    frame_start = time.monotonic()
    with torch.inference_mode():
        while sim.is_running() and (args.steps == 0 or step_count < args.steps):
            if not sim.is_playing():
                sim.render()
                time.sleep(0.01)
                frame_start = time.monotonic()
                continue
            q_des = controller.homing(robot.data.joint_pos.torch[:, joint_ids])
            robot.actuators.target_command.set_position_index(value=q_des, joint_ids=joint_ids)
            # Go2's actuator computes tau = kp * (q_des - q) - kd * dq,
            # then clips it to the DC motor's torque-speed limits.
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim_dt)
            step_count += 1
            if controller.homing_flag == 2:
                foot_position = get_foot_positions(robot)
                logger.info("foot_position [prepare_position], BODY, FL/FR/RL/RR, meters:\n%s", foot_position[0].cpu())
            if controller.homing_flag == 4:
                hold_steps += 1
                if hold_steps == settle_period:
                    foot_position = get_foot_positions(robot)
                    logger.info(
                        "foot_position [homing_position], BODY, FL/FR/RL/RR, meters:\n%s", foot_position[0].cpu()
                    )
            if step_count % sim.cfg.render_interval == 0:
                sim.render()
                if not args.fast:
                    time.sleep(max(0.0, sim_dt * sim.cfg.render_interval - (time.monotonic() - frame_start)))
                frame_start = time.monotonic()
    foot_position = get_foot_positions(robot)
    if hold_steps < settle_period:
        logger.info("foot_position [current], BODY, FL/FR/RL/RR, meters:\n%s", foot_position[0].cpu())
    return foot_position


def main(argv: list[str] | None = None) -> int:
    """빈 장면에 지면과 Go2를 생성하고 호밍 동작을 실행한다."""
    import warp as wp

    wp.config.enable_backward = False
    wp.config.quiet = True
    from isaaclab.app import add_launcher_args, launch_simulation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=0, help="Stop after this many physics steps; 0 holds indefinitely."
    )
    parser.add_argument("--fast", action="store_true", help="Do not pace the simulation to real time.")
    add_launcher_args(parser)
    args = parser.parse_args(argv)
    if args.steps < 0:
        parser.error("steps must be nonnegative.")
    if settle_period < 1:
        parser.error("settle_period must be positive.")
    if args.visualizer is None:
        args.visualizer = ["newton_gl"]

    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass, instantiate, replace
    from isaaclab.utils.timer import Timer
    from isaaclab.visualizers import VisualizerCfg

    from isaaclab_assets import UNITREE_GO2_CFG

    logging.getLogger("isaaclab").setLevel(logging.WARNING)
    logging.getLogger("isaaclab_newton").setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)
    Timer.enable_display_output = False
    torch.set_printoptions(precision=6, sci_mode=False)

    @configclass
    class Go2SceneCfg(InteractiveSceneCfg):
        """지면, 조명, Go2 한 대로 구성한 장면."""

        ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=750.0))
        robot = replace(UNITREE_GO2_CFG, prim_path="{ENV_REGEX_NS}/Robot")

    scene_cfg = Go2SceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot.init_state.joint_pos = initial_position
    scene_cfg.robot.init_state.joint_vel = {".*": 0.0}
    scene_cfg.robot.actuators["base_legs"].stiffness = kp
    scene_cfg.robot.actuators["base_legs"].damping = kd
    sim_cfg = sim_utils.SimulationCfg(
        device=args.device,
        dt=0.001,
        render_interval=20,
        logging_level="WARNING",
        use_newton_actuators=True,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=256, nconmax=35, integrator="implicitfast", use_mujoco_contacts=False),
            num_substeps=2,
            default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
        ),
        visualizer_cfgs=[VisualizerCfg(eye=(1.2, -1.2, 0.8), lookat=(0.0, 0.0, 0.2))],
    )
    with launch_simulation(sim_cfg, args):
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(scene_cfg)
        robot = scene["robot"]
        sim.reset()
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch)
        robot.reset()
        with contextlib.suppress(KeyboardInterrupt):
            _run_homing(sim, robot, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
