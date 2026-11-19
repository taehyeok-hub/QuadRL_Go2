# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-reset measurement, probe exclusion, and noninterference with native PPO."""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from isaaclab_rl.diagnostics.locomotion_metrics import RslRlLocomotionMetrics

from isaaclab_tasks.core.velocity.mdp.metrics import LocomotionMetricsCfg

pytestmark = pytest.mark.unit


def _array(value):
    return SimpleNamespace(torch=torch.tensor(value, dtype=torch.float))


class MeasurementEnv:
    """Two feet with reversed contact ordering and a terminal-state overwrite on reset."""

    num_envs = 2
    num_actions = 1
    device = "cpu"

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(locomotion_metrics=LocomotionMetricsCfg(playback_interval_steps=2))
        self.command = torch.zeros(2, 3)
        self.command_manager = SimpleNamespace(get_command=lambda name: self.command)
        self.curriculum_manager = SimpleNamespace(update=lambda dt: None)
        self.data = SimpleNamespace(
            root_lin_vel_b=_array([[0, 0, 0], [0, 0, 0]]),
            root_ang_vel_b=_array([[0, 0, 0], [0, 0, 0]]),
            root_link_lin_vel_w=_array([[0, 0, 2], [0, 0, -2]]),
            projected_gravity_b=_array([[0, 0, -1], [0, 0, -1]]),
            body_link_pos_w=_array([[[0, 0, 0.05], [0, 0, 0.05]]] * 2),
            body_link_lin_vel_w=_array([[[0, 0, 0], [0, 0, 0]]] * 2),
        )
        robot = SimpleNamespace(data=self.data, find_bodies=lambda pattern: ([0, 1], ["left_foot", "right_foot"]))
        contact = SimpleNamespace(
            cfg=SimpleNamespace(track_air_time=True, force_threshold=0.0),
            find_bodies=lambda names, preserve_order: ([1, 0], ["left_foot", "right_foot"]),
            data=SimpleNamespace(
                net_normal_forces_w=_array([[[0, 0, 0], [0, 0, 10]]] * 2),
                last_air_time=_array([[0, 0], [0, 0]]),
            ),
        )

        class Scene(dict):
            sensors = {}
            terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_type="plane"))

        self.scene = Scene(robot=robot, contact_forces=contact)
        self.obs = torch.zeros(2, 2)
        self.steps = 0

    def get_observations(self):
        return TensorDict({"policy": self.obs.clone()}, batch_size=[2])

    def _reset_idx(self, env_ids):
        self.data.root_lin_vel_b.torch[env_ids] = 1000

    def step(self, actions):
        self.command.zero_()
        self.data.root_lin_vel_b.torch[:, 0] = actions[:, 0]
        self.data.root_lin_vel_b.torch[:, 1:] = 0
        self.curriculum_manager.update(0.02)
        self._reset_idx(torch.arange(2))
        self.command.fill_(999)
        self.steps += 1
        self.obs.copy_(torch.rand_like(self.obs))
        return self.get_observations(), -actions[:, 0].square(), torch.zeros(2), {"log": {}}


def _runner():
    cfg = {
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [4],
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [4]},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": 2,
        "save_interval": 100,
    }
    return OnPolicyRunner(MeasurementEnv(), cfg, log_dir=None, device="cpu")


def test_playback_logs_pre_reset_errors_excludes_probes_and_closes_partial_window(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    runner = _runner()
    runner.logger.writer = SummaryWriter(str(tmp_path))
    original_update = runner.env.curriculum_manager.update
    rng = torch.random.get_rng_state()
    with RslRlLocomotionMetrics(runner, playback=True) as metrics:
        runner.env.step(torch.ones(2, 1))
        with metrics.suspend():
            runner.env.step(torch.full((2, 1), 500.0))
        runner.env.step(torch.ones(2, 1))
        runner.env.step(torch.ones(2, 1))
    runner.logger.writer.close()
    events = EventAccumulator(str(tmp_path)).Reload()
    points = events.Scalars("Velocity/x_mae")
    assert [(point.step, point.value) for point in points] == [(2, 1.0), (3, 1.0)]
    assert events.Scalars("Gait/base_vz_rms")[0].value == 2.0
    assert events.Scalars("Gait/stance_ratio")[0].value == 50.0
    assert "Gait/swing_time_mean" not in events.Tags()["scalars"]
    assert runner.env.curriculum_manager.update == original_update
    # The fixture itself consumes RNG; telemetry must consume none beyond those four steps.
    final_rng = torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    for _ in range(4):
        torch.rand(2, 2)
    assert torch.equal(torch.random.get_rng_state(), final_rng)


def test_metrics_preserve_actual_ppo_parameters_observations_and_rng():
    outcomes = []
    for enabled in (False, True):
        torch.manual_seed(42)
        runner = _runner()
        rng_before = torch.random.get_rng_state().clone()
        controller = RslRlLocomotionMetrics(runner) if enabled else contextlib.nullcontext()
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        with controller:
            runner.learn(3)
        outcomes.append(
            (
                runner.alg.actor.state_dict()
                | {f"critic/{key}": value for key, value in runner.alg.critic.state_dict().items()},
                runner.env.obs.clone(),
                torch.random.get_rng_state().clone(),
            )
        )
    for key, tensor in outcomes[0][0].items():
        torch.testing.assert_close(tensor, outcomes[1][0][key], rtol=0, atol=0)
    torch.testing.assert_close(outcomes[0][1], outcomes[1][1], rtol=0, atol=0)
    assert torch.equal(outcomes[0][2], outcomes[1][2])
