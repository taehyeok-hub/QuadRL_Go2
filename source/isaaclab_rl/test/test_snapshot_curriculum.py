# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen-policy curriculum isolation at the installed RSL-RL PPO boundary."""

import contextlib
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from isaaclab_rl.utils.snapshot_curriculum import RslRlSnapshotCurriculum

pytestmark = pytest.mark.unit


class ProbeTerm:
    def __init__(self):
        self.snapshot_enabled = True
        self.snapshot_controller_active = False
        self.catalog = SimpleNamespace(num_tasks=2)
        self.settings = SimpleNamespace(min_samples=2, evaluation_interval_iterations=2, evaluation_seed=101)
        self.last_snapshot_policy_updates = None
        self.training_policy_updates = None
        self.active_tasks = None
        self.assignments = []
        self.commits = []

    def begin_snapshot_batch(self, tasks):
        self.active_tasks = tasks.clone()
        self.assignments.append(tasks.clone())
        self.returns = torch.zeros(len(tasks))
        self.finished = tasks < 0
        self.steps = 0

    @property
    def snapshot_batch_active(self):
        return ~self.finished

    @property
    def snapshot_batch_complete(self):
        return self.steps == 2

    @property
    def snapshot_batch_returns(self):
        assert self.snapshot_batch_complete
        return self.returns.clone()

    def end_snapshot_batch(self):
        self.active_tasks = None

    def commit_snapshot(self, values, policy_updates, policy_digest):
        self.commits.append((values.clone(), policy_updates, policy_digest))
        self.last_snapshot_policy_updates = policy_updates


class ProbeEnv:
    num_envs = 3
    num_actions = 1
    device = "cpu"
    max_episode_length = 2

    def __init__(self, term):
        self.unwrapped = self
        self.term = term
        self.curriculum_manager = SimpleNamespace(_class_term_cfgs=[SimpleNamespace(func=term)])
        self.common_step_counter = 37
        self._sim_step_counter = 74
        self.event_manager = SimpleNamespace(
            _interval_term_time_left=[torch.tensor([3.0])],
            _reset_term_last_triggered_step_id=[torch.tensor([12, 13, 14])],
            _reset_term_last_triggered_once=[torch.tensor([True, True, True])],
            _mode_term_cfgs={"interval": [SimpleNamespace(interval_range_s=(1.0, 2.0))]},
        )
        self.extras = {"log": {"training": 1}}
        self.training_resets = self.probe_steps = self.training_steps = 0
        self.reset_draws = 1
        self.parked_actions = []
        self.fail_probe = self.nonfinite_probe_obs = self.nonfinite_terminal_obs = self.history_suspended = False
        self.reset()

    @contextlib.contextmanager
    def probe_context(self):
        self.history_suspended = True
        try:
            yield
        finally:
            self.history_suspended = False

    def reset(self):
        random.random()
        np.random.random()
        torch.rand(())
        if self.term.active_tasks is None:
            self.training_resets += 1
            self.obs = torch.full((self.num_envs, 2), float(100 + self.training_resets))
        else:
            assert self.history_suspended
            self.obs = self.term.active_tasks.clamp_min(0).float().unsqueeze(-1).repeat(1, 2)
            if self.nonfinite_probe_obs:
                self.obs[0, 0] = float("nan")
        return self.get_observations(), self.extras

    def get_observations(self):
        return TensorDict({"policy": self.obs.clone()}, batch_size=[self.num_envs])

    def _reset_idx(self, env_ids):
        for _ in range(self.reset_draws):
            random.random()
            np.random.random()
            torch.rand(len(env_ids))
        self.obs[env_ids] = 0

    def step(self, actions):
        self.common_step_counter += 1
        self._sim_step_counter += 2
        self.event_manager._interval_term_time_left[0].sub_(0.1)
        if self.term.active_tasks is not None:
            assert self.history_suspended
            self.probe_steps += 1
            if self.fail_probe:
                raise RuntimeError("probe physics failed")
            active = self.term.snapshot_batch_active
            self.parked_actions.append(actions[~active].clone())
            self.term.steps += 1
            self.term.returns[active] += actions[active, 0] + self.term.active_tasks[active].float()
            self.term.finished[0] = True
            if self.term.steps == 2:
                self.term.finished[:] = True
            self._reset_idx(self.term.finished.nonzero(as_tuple=False).flatten())
        else:
            assert self.term.snapshot_controller_active
            self.training_steps += 1
        self.obs.add_(1)
        if self.term.active_tasks is not None and self.term.steps == 2 and self.nonfinite_terminal_obs:
            self.obs[0, 0] = float("nan")
        self.extras["log"] = {"probe": self.term.active_tasks is not None}
        return self.get_observations(), torch.ones(self.num_envs), torch.zeros(self.num_envs), self.extras


def _make_runner():
    env = ProbeEnv(ProbeTerm())
    obs = env.get_observations()
    actor = MLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        1,
        hidden_dims=(4,),
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
    )
    critic = MLPModel(obs, {"critic": ["policy"]}, "critic", 1, hidden_dims=(4,))
    runner = OnPolicyRunner.__new__(OnPolicyRunner)
    runner.env = env
    runner.alg = PPO(
        actor, critic, RolloutStorage("rl", env.num_envs, 2, obs, [1]), num_learning_epochs=1, num_mini_batches=1
    )
    runner.device = "cpu"
    runner.is_distributed = False
    runner.current_learning_iteration = 0
    runner.cfg = {"num_steps_per_env": 2, "algorithm": {}, "save_interval": 10}
    runner.logger = SimpleNamespace(
        cur_reward_sum=torch.ones(env.num_envs),
        cur_episode_length=torch.ones(env.num_envs),
        ep_extras=[{"completed": 7}],
        init_logging_writer=Mock(),
        process_env_step=Mock(),
        log=Mock(),
        writer=None,
        save_model=Mock(),
    )
    return runner


@pytest.fixture
def runner():
    return _make_runner()


def test_initial_probe_preserves_policy_storage_and_training_random_stream(runner):
    """Only the final training reset consumes the original random stream."""
    state = {key: value.clone() for key, value in runner.alg.actor.state_dict().items()}
    runner.alg.storage.rewards.fill_(123.0)
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    random.random()
    np.random.random()
    torch.rand(())
    expected = random.random(), np.random.random(), torch.rand(())
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    timers = runner.env.event_manager._interval_term_time_left[0].clone()
    with RslRlSnapshotCurriculum(runner, probe_context=runner.env.probe_context):
        assert [value[1] for value in runner.env.term.commits] == [0]
        assert len(runner.env.term.commits[0][2]) == 64
        assert torch.cat(runner.env.term.assignments).tolist() == [0, 0, 1, 1, -1, -1]
        for key, value in state.items():
            torch.testing.assert_close(runner.alg.actor.state_dict()[key], value, rtol=0, atol=0)
        assert runner.alg.actor.training
        assert runner.alg.storage.step == 0 and runner.alg.transition.actions is None
        assert (runner.alg.storage.rewards == 123.0).all()
        assert runner.logger.cur_reward_sum.count_nonzero() == 0
        assert runner.logger.ep_extras == [{"completed": 7}]
        runner.logger.process_env_step.assert_not_called()
        assert runner.env.common_step_counter == 37 and runner.env._sim_step_counter == 74
        torch.testing.assert_close(runner.env.event_manager._interval_term_time_left[0], timers)
        assert runner.env.extras["log"] == {}
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(()), expected[2], rtol=0, atol=0)
    assert not runner.env.term.snapshot_controller_active
    assert "update" not in runner.alg.__dict__ and "compute_returns" not in runner.alg.__dict__


def test_actual_ppo_excludes_probes_and_refreshes_the_next_rollout_observation(runner):
    original_act = runner.alg.act
    seen = []

    def act(obs):
        torch.testing.assert_close(obs["policy"], runner.env.get_observations()["policy"])
        seen.append(obs["policy"].clone())
        return original_act(obs)

    runner.alg.act = act
    with RslRlSnapshotCurriculum(runner, probe_context=runner.env.probe_context):
        runner.learn(3)
    assert runner.env.training_steps == len(seen) == runner.logger.process_env_step.call_count == 6
    assert runner.env.probe_steps == 8
    assert [value[1] for value in runner.env.term.commits] == [0, 2]
    assert runner.env.term.commits[0][2] != runner.env.term.commits[1][2]
    assert runner.alg.storage.step == 0 and runner.env.common_step_counter == 43
    assert runner.logger.ep_extras == [{"completed": 7}]


def test_retired_slot_resets_preserve_live_case_randomness_and_zero_controls(runner):
    """Extra draws inside per-slot physical resets cannot alter live stochastic probes."""
    actor_state = {key: value.clone() for key, value in runner.alg.actor.state_dict().items()}
    with RslRlSnapshotCurriculum(runner, probe_context=runner.env.probe_context):
        expected = runner.env.term.commits[0][0]
    other = _make_runner()
    other.alg.actor.load_state_dict(actor_state)
    other.env.reset_draws = 13
    with RslRlSnapshotCurriculum(other, probe_context=other.env.probe_context):
        torch.testing.assert_close(other.env.term.commits[0][0], expected, rtol=0, atol=0)
    assert all(not actions.count_nonzero() for actions in other.env.parked_actions)
    assert "_reset_idx" not in vars(other.env)


@pytest.mark.parametrize(
    "failure, exception, message",
    [
        ("fail_probe", RuntimeError, "probe physics failed"),
        ("nonfinite_probe_obs", FloatingPointError, "non-finite observations"),
        ("nonfinite_terminal_obs", FloatingPointError, "non-finite observations"),
    ],
)
def test_failed_probe_restores_lifetimes_and_never_commits(runner, failure, exception, message):
    setattr(runner.env, failure, True)
    with pytest.raises(exception, match=message):
        with RslRlSnapshotCurriculum(runner, probe_context=runner.env.probe_context):
            pass
    assert runner.env.term.commits == [] and runner.env.term.active_tasks is None
    assert runner.env.common_step_counter == 37 and runner.env._sim_step_counter == 74
    assert not runner.env.history_suspended and not runner.env.term.snapshot_controller_active
    assert "update" not in runner.alg.__dict__ and "compute_returns" not in runner.alg.__dict__
    assert runner.env.training_resets == 2


@pytest.mark.parametrize("baseline", [4, 5])
def test_resume_within_interval_keeps_baseline_and_distributed_training_fails_clearly(runner, baseline):
    runner.current_learning_iteration = 4
    runner.env.term.last_snapshot_policy_updates = baseline
    with RslRlSnapshotCurriculum(runner, resumed=True, probe_context=runner.env.probe_context):
        assert runner.env.term.commits == []
    runner.is_distributed = True
    with pytest.raises(ValueError, match="single-device"):
        RslRlSnapshotCurriculum(runner)
    runner.is_distributed = False
    runner.env.cfg = SimpleNamespace(
        sim=SimpleNamespace(physics=SimpleNamespace(solver_cfg=SimpleNamespace(use_mujoco_cpu=True)))
    )
    with pytest.raises(ValueError, match="isolated per-environment resets"):
        RslRlSnapshotCurriculum(runner)


def test_resume_rejects_a_baseline_from_a_future_policy(runner):
    runner.env.term.last_snapshot_policy_updates = 8
    with pytest.raises(ValueError, match="newer policy"):
        with RslRlSnapshotCurriculum(runner, resumed=True):
            pass
    assert not runner.env.term.snapshot_controller_active


def test_completed_update_counter_survives_two_actual_runner_resumes(runner, tmp_path):
    """Installed runner labels repeat on resume; LP cadence must use its persisted update count."""
    runner.env.term.settings.evaluation_interval_iterations = 1
    with RslRlSnapshotCurriculum(runner, probe_context=runner.env.probe_context):
        runner.learn(2)
    assert runner.current_learning_iteration == 1
    assert runner.env.term.training_policy_updates == 2
    for completed in (3, 4):
        path = str(tmp_path / f"resume_{completed}.pt")
        term = runner.env.term
        runner.save(
            path, infos={"updates": term.training_policy_updates, "baseline": term.last_snapshot_policy_updates}
        )
        runner = _make_runner()
        infos = runner.load(path)
        runner.env.term.training_policy_updates = infos["updates"]
        runner.env.term.last_snapshot_policy_updates = infos["baseline"]
        runner.env.term.settings.evaluation_interval_iterations = 1
        with RslRlSnapshotCurriculum(runner, resumed=True, probe_context=runner.env.probe_context):
            runner.learn(1)
        assert runner.current_learning_iteration == 1
        assert runner.env.term.training_policy_updates == completed
        assert runner.env.term.last_snapshot_policy_updates == completed
